/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include <cub/cub.cuh>

#include <cassert>
#include <cstdint>

#include "common.h"
#include "gdeflate_constants.h"
#include "optimal_parse/sa.cuh"

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

namespace gdeflate
{

template <typename offset_type>
constexpr offset_type INVALID_OFFSET = ~0;

template <typename offset_type>
union match_t
{
  // Note, that we are using a union to share the underlying memory
  // such that we can access `len` and `off` both independently and
  // also through the 32-bit field `data`. By packing the struct
  // within the union, we make sure that no implicit compiler behavior
  // is going to introduce padding bytes for aligning the struct
  // members.
#ifdef _MSC_VER
#pragma pack(1)
#endif // _MSC_VER
  struct
  {
    offset_type len;
    offset_type off;
  }
#ifdef __GNUC__
  __attribute__((packed))
#endif // __GNUC__
  ;
  using int_type = std::conditional_t<std::is_same_v<offset_type, uint16_t>, uint32_t, uint64_t>;
  int_type data;
};

template <unsigned SIZE, typename offset_type>
__device__ void selectBestMatch(
  const cg::thread_block_tile<SIZE> &g,
  match_t<offset_type> &best_match,
  const match_t<offset_type> &candidate_match,
  unsigned int max_offset,
  unsigned int max_match_len
)
{

  bool can_encode = candidate_match.off < max_offset;
  bool better_match = candidate_match.len > best_match.len ||
                      (candidate_match.len == best_match.len && candidate_match.off < best_match.off);

  if (can_encode && better_match)
  {
    best_match = candidate_match;
  }

  for (int offset = 1; offset <= g.size() / 2; offset <<= 1)
  {

    match_t<offset_type> new_match;
    new_match.data = g.shfl_down(best_match.data, offset);

    better_match = new_match.len > best_match.len ||
                   (new_match.len == best_match.len && new_match.off < best_match.off);

    best_match = better_match ? new_match : best_match;
  }

  best_match.data = g.shfl(best_match.data, 0);
}

template <typename LCP_t, unsigned SIZE, typename offset_type>
__device__ void groupFindLongestMatch(
  const cg::thread_block_tile<SIZE> &g,
  offset_type &best_match_len,
  offset_type &best_match_off,
  const uint32_t pos,
  const offset_type *sa,
  const offset_type *inv_sa,
  const LCP_t *lcp,
  uint32_t buffer_len,
  uint32_t max_offset,
  uint32_t max_match_len
)
{
  int lane = g.thread_rank();

  uint32_t suffix_start = inv_sa[pos];

  match_t<offset_type> best_match;
  best_match.len = best_match_len;
  best_match.off = best_match_off;
  offset_type prev_match_len = std::numeric_limits<offset_type>::max();

  for (uint32_t wave = 0; wave <= min(buffer_len / g.size(), 128); wave++)
  {

    uint32_t idx = suffix_start - (g.size() / 2) + lane;
    idx += (lane < g.size() / 2) ? -(wave * g.size() / 2) : wave * g.size() / 2;

    bool oob = idx >= buffer_len;
    // TODO: fuse lcp and sa into 32bits
    offset_type candidate_match_len = sa::getLaneMatchLength(g, oob ? 0 : lcp[idx]);
    offset_type candidate_sa_pos = oob ? INVALID_OFFSET<offset_type> : sa[idx + (lane >= g.size() / 2)];
    offset_type candidate_match_off = (candidate_sa_pos < pos) ? pos - candidate_sa_pos : INVALID_OFFSET<offset_type>;

    candidate_match_len = min(candidate_match_len, prev_match_len);

    bool potential_matches = g.ballot(candidate_match_len >= max(best_match.len, gdeflateMinMatchLength));

    if (!potential_matches || best_match.len >= max_match_len)
    {
      break;
    }

    match_t<offset_type> candidate_match;
    candidate_match.len = candidate_match_len;
    candidate_match.off = candidate_match_off;
    selectBestMatch(g, best_match, candidate_match, max_offset, max_match_len);

    prev_match_len = g.shfl(candidate_match_len, lane < g.size() / 2 ? 0 : g.size() - 1);
  }

  best_match_len = best_match.len;
  best_match_off = best_match.off;
  assert(best_match_off > 0);
}

template <typename LCP_t, unsigned size, bool deflate64 = false, typename offset_type>
__device__ void lz_compress_tile_longest_matches(
  cg::thread_block_tile<size> g,
  const unsigned char *input,
  const unsigned int input_bytes,
  const unsigned int maxLength,
  offset_type *length,
  offset_type *distance,
  offset_type *sa,
  offset_type *inv_sa,
  const LCP_t *lcp,
  unsigned int pos
)
{
  // Note: in practice this never happens as it is guarded against in the parent function
  if (input_bytes == 0)
  {
    return;
  }

  // Thread 0 writes out first byte as a literal
  if (pos == 0)
  {
    length[0] = 1;
    distance[0] = 0;
    return;
  }
  constexpr int maxDictionaryLength = deflate64 ? gdeflateMaxDictionaryLength : deflateMaxDictionaryLength;

  unsigned int maxOffset = min(pos, maxDictionaryLength);
  unsigned int maxMatchLength = min(maxLength, input_bytes - pos);
  offset_type matchOffset = INVALID_OFFSET<offset_type>;
  offset_type matchLength = 1;

  groupFindLongestMatch<LCP_t>(g, matchLength, matchOffset, pos, sa, inv_sa, lcp, input_bytes, maxOffset, maxMatchLength);
  assert(matchLength <= maxMatchLength);

  if (g.thread_rank() == 0)
  {
    length[pos] = matchLength < gdeflateMinMatchLength ? 1 : min(matchLength, maxMatchLength);
    distance[pos] = matchLength < gdeflateMinMatchLength ? 0 : matchOffset;
  }
}

template <typename LCP_t = uint16_t, typename offset_type>
__global__ void lz_compress_longest_matches(
  const unsigned char *const *input_ptrs,
  const size_t *input_bytes,
  const unsigned int maxLength,
  offset_type **length_ptrs,
  offset_type **distance_ptrs,
  const unsigned int num_tiles_host,
  offset_type **sa_ptrs,
  offset_type **inv_sa_ptrs,
  const LCP_t *const *lcp_ptrs,
  unsigned int ctas_per_chunk,
  const int *device_num_tiles
)
{
  const unsigned int num_tiles = device_num_tiles ? static_cast<unsigned int>(*device_num_tiles) : num_tiles_host;

  auto chunk = blockIdx.x / ctas_per_chunk;
  if (chunk >= num_tiles)
  {
    return;
  }
  auto sub_chunk = blockIdx.x % ctas_per_chunk;

  auto chunk_size = input_bytes[chunk];

  auto thread_block = cg::this_thread_block();
  cg::thread_block_tile<4> g = cg::tiled_partition<4>(thread_block);
  uint32_t meta_group_size = thread_block.size() / g.size();
  uint32_t meta_group_rank = thread_block.thread_rank() / g.size();

  auto sub_chunk_size = (chunk_size + ctas_per_chunk - 1) / ctas_per_chunk;
  __shared__ uint32_t cta_pos;
  cta_pos = sub_chunk * sub_chunk_size + meta_group_size;
  __syncthreads();

  uint32_t cg_pos = sub_chunk * sub_chunk_size + meta_group_rank;

  // work stealing because finding a match can take a varied amount of time.
  while (cg_pos < min((sub_chunk + 1) * sub_chunk_size, chunk_size))
  {
    lz_compress_tile_longest_matches<LCP_t>(
      g,
      input_ptrs[chunk],
      (unsigned int)input_bytes[chunk],
      maxLength,
      length_ptrs[chunk],
      distance_ptrs[chunk],
      sa_ptrs[chunk],
      inv_sa_ptrs[chunk],
      lcp_ptrs[chunk],
      cg_pos
    );
    uint32_t pos = 0;
    if (g.thread_rank() == 0)
    {
      pos = atomicAdd(&cta_pos, 1);
    }
    cg_pos = g.shfl(pos, 0);
  }
}

template <typename CostMetric, typename offset_type>
__device__ void lz_compress_tile_optimal_parse(
  const unsigned char *input,
  const unsigned int input_bytes,
  const unsigned int maxLength,
  CostMetric &cost_evaluator,
  unsigned int *cost,
  offset_type *length,
  offset_type *distance,
  uint8_t *literals
)
{
  // length stores the match length. Match length of 1 indicates a literal
  // distance indicates the match offset or offset in the literal stream. TODO: 16 bits allows for only 64k literals.
  // literals contains just the literal stream without any copies

  assert(blockDim.x == WARP_SIZE_U);
  assert(blockDim.y == 1);

  if (input_bytes == 0)
  {
    return;
  }

  if (threadIdx.x == 0)
  {
    cost[input_bytes - 1] = cost_evaluator.literal_cost(input[input_bytes - 1]);
    length[input_bytes - 1] = 1;
  }
  __syncwarp();

  typedef nvcomp::cub::WarpReduce<nvcomp::cub::KeyValuePair<offset_type, unsigned int>> WarpReduce;
  __shared__ typename WarpReduce::TempStorage temp_storage;

  // Compute minimum bit cost to go from each byte position to the end of stream
  for (int pos = input_bytes - 2; pos >= 0; --pos)
  {
    // Get literal cost
    nvcomp::cub::KeyValuePair<offset_type, unsigned int> c(1, cost[pos + 1] + cost_evaluator.literal_cost(input[pos]));

    if (length[pos] >= gdeflateMinMatchLength)
    {
      // Find the best match length
      assert(distance[pos] > 0);
      uint8_t distance_cost = cost_evaluator.distance_cost(distance[pos]);
      for (offset_type l = gdeflateMinMatchLength + threadIdx.x;
           l < gdeflateMinMatchLength +
                 roundUpTo(offset_type(length[pos] - gdeflateMinMatchLength + 1), offset_type(WARP_SIZE));
           l += offset_type(WARP_SIZE))
      {
        unsigned int match_cost = (l <= length[pos]) ? distance_cost + cost_evaluator.length_cost(l) + cost[pos + l]
                                                     : std::numeric_limits<unsigned int>::max();
        if (match_cost < c.value)
        {
          c.key = l;
          c.value = match_cost;
        }
        // Get minimum cost length
        c = WarpReduce(temp_storage).Reduce(c, nvcomp::cub::ArgMin());
        c.key = __shfl_sync(WARP_ALL, c.key, 0);
        c.value = __shfl_sync(WARP_ALL, c.value, 0);
      }
    }

    if (threadIdx.x == 0)
    {
      length[pos] = c.key;
      cost[pos] = c.value;
    }
    __syncwarp();
  }
}

template <typename offset_type>
inline __device__ void write_chosen_match(
  const unsigned char *input,
  const unsigned int input_bytes,
  offset_type *length,
  offset_type *distance,
  uint8_t *literals,
  unsigned int *num_symbols,
  unsigned int *num_literals
)
{
  // Now write chosen matches to the output
  __shared__ unsigned int literal_pos[256];
  unsigned int pos = 0;
  unsigned int nsymbols = 0;
  unsigned int nliterals = 0;
  uint16_t literal_i = 0;
  int8_t t = threadIdx.x;
  while (pos < input_bytes)
  {
    unsigned int matchOffset = distance[pos];
    unsigned int matchLength = length[pos];

    if (matchLength < gdeflateMinMatchLength)
    {
      // Literal
      if (t == 0)
      {
        literal_pos[literal_i] = pos;
        length[nsymbols] = 1;
        distance[nsymbols] = nliterals;
      }
      literal_i++;
      pos += 1;
      nliterals++;
    }
    else
    {
      // Copy
      if (t == 0)
      {
        length[nsymbols] = matchLength;
        distance[nsymbols] = matchOffset;
      }
      pos += matchLength;
    }
    nsymbols++;
    if (literal_i == 256)
    {
      __syncwarp();
      for (int i = t; i < literal_i; i += WARP_SIZE)
      {
        literals[nliterals - literal_i + i] = input[literal_pos[i]];
      }
      literal_i = 0;
    }
  }

  // Copy the remaining ones
  __syncwarp();
  for (int i = t; i < literal_i; i += WARP_SIZE)
  {
    literals[nliterals - literal_i + i] = input[literal_pos[i]];
  }

  if (t == 0)
  {
    *num_symbols = nsymbols;
    *num_literals = nliterals;
  }
}
} // namespace gdeflate
