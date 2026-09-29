/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cub/cub.cuh>

#include <cuda/atomic>

#include "snappy/unsnap.cuh"
#include "SnappyBlockUtils.cuh"

namespace nvcomp
{

#define HASH_BITS 12

// TBD: Tentatively limits to 2-byte codes to prevent long copy search followed by long literal
// encoding
#define MAX_LITERAL_LENGTH 256

#define MAX_COPY_LENGTH 64 // Syntax limit
#define MAX_COPY_DISTANCE 32768 // Matches encoder limit as described in snappy format description

const int COMP_THREADS_PER_BLOCK = 64; // 2 warps per stream, 1 stream per block
const uint32_t DECOMP_THREADS_PER_BLOCK = 96;

/**
 * @brief snappy compressor state
 **/
struct snap_state_s
{
  const uint8_t *src; ///< Ptr to uncompressed data
  uint32_t src_len; ///< Uncompressed data length
  uint8_t *dst_base; ///< Base ptr to output compressed data
  uint8_t *dst; ///< Current ptr to uncompressed data
  uint8_t *end; ///< End of uncompressed data buffer
  uint32_t literal_length; ///< Number of literal bytes
  uint32_t copy_length; ///< Number of copy bytes
  uint32_t copy_distance; ///< Distance for copy bytes
  uint16_t __align__(4) hash_map[1 << HASH_BITS]; ///< Low 16-bit offset from hash
};

static inline __device__ uint32_t get_max_compressed_length(uint32_t source_bytes)
{
  // This is an estimate from the original snappy library
  return 32 + source_bytes + source_bytes / 6;
}

/**
 * @brief 12-bit hash from four consecutive bytes
 **/
static inline __device__ uint32_t snap_hash(uint32_t v)
{
  return (v * ((1 << 20) + (0x2a00) + (0x6a) + 1)) >> (32 - HASH_BITS);
}

/**
 * @brief Outputs a snappy literal symbol
 *
 * @param dst Destination compressed byte stream
 * @param end End of compressed data buffer
 * @param src Pointer to literal bytes
 * @param len_minus1 Number of literal bytes minus 1
 * @param t Thread in warp
 *
 * @return Updated pointer to compressed byte stream
 **/
static inline __device__ uint8_t *
StoreLiterals(uint8_t *dst, uint8_t *end, const uint8_t *src, uint32_t len_minus1, uint32_t t)
{
  if (len_minus1 < 60)
  {
    if (!t && dst < end)
    {
      dst[0] = (len_minus1 << 2);
    }
    dst += 1;
  }
  else if (len_minus1 <= 0xff)
  {
    if (!t && dst + 1 < end)
    {
      dst[0] = 60 << 2;
      dst[1] = len_minus1;
    }
    dst += 2;
  }
#if MAX_LITERAL_LENGTH > 0xff
  else if (len_minus1 <= 0xffff)
  {
    if (!t && dst + 2 < end)
    {
      dst[0] = 61 << 2;
      dst[1] = len_minus1;
      dst[2] = len_minus1 >> 8;
    }
    dst += 3;
  }
  else if (len_minus1 <= 0xffffff)
  {
    if (!t && dst + 3 < end)
    {
      dst[0] = 62 << 2;
      dst[1] = len_minus1;
      dst[2] = len_minus1 >> 8;
      dst[3] = len_minus1 >> 16;
    }
    dst += 4;
  }
  else
  {
    if (!t && dst + 4 < end)
    {
      dst[0] = 63 << 2;
      dst[1] = len_minus1;
      dst[2] = len_minus1 >> 8;
      dst[3] = len_minus1 >> 16;
      dst[4] = len_minus1 >> 24;
    }
    dst += 5;
  }
#endif
  for (uint32_t i = t; i <= len_minus1; i += 32)
  {
    // Note: during testing this condition is always true, since we do not pass buffers that
    // are too small to hold the compressed data
    if (dst + i < end)
    {
      dst[i] = src[i];
    }
  }
  return dst + len_minus1 + 1;
}

/**
 * @brief Outputs a snappy copy symbol (assumed to be called by a single thread)
 *
 * @param dst Destination compressed byte stream
 * @param end End of compressed data buffer
 * @param copy_len Copy length
 * @param distance Copy distance
 *
 * @return Updated pointer to compressed byte stream
 **/
static inline __device__ uint8_t *StoreCopy(uint8_t *dst, uint8_t *end, uint32_t copy_len, uint32_t distance)
{
  if (copy_len < 12 && distance < 2048)
  {
    // xxxxxx01.oooooooo: copy with 3-bit length, 11-bit offset
    // Note: during testing this condition is always true, since we do not pass buffers that
    // are too small to hold the compressed data
    if (dst + 2 <= end)
    {
      dst[0] = ((distance & 0x700) >> 3) | ((copy_len - 4) << 2) | 0x01;
      dst[1] = distance;
    }
    return dst + 2;
  }
  else
  {
    // xxxxxx1x: copy with 6-bit length, 16-bit offset
    // Note: during testing this condition is always true, since we do not pass buffers that
    // are too small to hold the compressed data
    if (dst + 3 <= end)
    {
      dst[0] = ((copy_len - 1) << 2) | 0x2;
      dst[1] = distance;
      dst[2] = distance >> 8;
    }
    return dst + 3;
  }
}

/**
 * @brief Finds the first occurence of a consecutive 4-byte match in the input sequence,
 * or at most MAX_LITERAL_LENGTH bytes
 *
 * @param s Compressor state (copy_length set to 4 if a match is found, zero otherwise)
 * @param src Uncompressed buffer
 * @param pos0 Position in uncompressed buffer
 * @param t thread in warp
 *
 * @return Number of bytes before first match (literal length)
 **/
static inline __device__ uint32_t FindFourByteMatch(
  const uint8_t *src,
  uint16_t *hash_map,
  const uint32_t len,
  const uint32_t pos0,
  const uint32_t t,
  uint32_t &copy_length,
  uint32_t &copy_distance
)
{
  uint32_t pos = pos0;
  uint32_t maxpos = pos0 + MAX_LITERAL_LENGTH - 31;
  uint32_t match_mask, literal_cnt;
  copy_length = 0;
  do
  {
    // Each thread loads a 4byte data with a bit of offset, then we check if there are any 2 hashes that match in the warp
    bool valid4 = (pos + t + 4 <= len);
    uint32_t data32 = (valid4) ? unaligned_load32_slow(src + pos + t) : 0;
    uint32_t hash = (valid4) ? snap_hash(data32) : 0;
    uint32_t local_match = __match_any_sync(WARP_ALL, hash);
    // Each thread only receives the matching lane indices plus itself, therefore local_match is never zero
    uint32_t local_match_lane = 31 - __clz(local_match & ((1 << t) - 1));

    uint32_t local_match_data = SHFL(data32, min(local_match_lane, t));
    uint32_t offset, match;
    if (valid4)
    {
      // We are within the buffer, did we match with another lane?
      if (local_match_lane < t && local_match_data == data32)
      {
        match = 1;
        offset = pos + local_match_lane;
      }
      else
      {
        offset = (pos & ~0xffff) | hash_map[hash];
        if (offset >= pos)
        {
          offset = (offset >= 0x10000) ? offset - 0x10000 : pos;
        }
        match =
          (offset < pos && offset + MAX_COPY_DISTANCE >= pos + t &&
#ifdef IMPLICIT_BUFFER_ALIGNMENT
           unaligned_load32<false>(src + offset, offset)
#else
           unaligned_load32<true>(src + offset, offset)
#endif // IMPLICIT_BUFFER_ALIGNMENT
             == data32);
      }
    }
    else
    {
      // we don't have 4 bytes available starting at pos+t -> Cannot be any match
      match = 0;
      local_match = 0;
      offset = pos + t;
    }

    // Note:
    // `compute-sanitizer --tool racecheck` reports a race between
    // the write of hash_map and the read of hash_map if we don't place
    // a __syncwarp() here. Even though it seems that __ballot_sync() alone
    // acts as a barrier here, an MRE (minimal reproducible example) was
    // NOT producing the same racecheck hazard. Therefore, adding
    // an explicit __syncwarp() here to be on the safe side.
    __syncwarp();
    match_mask = BALLOT(match);
    if (match_mask != 0)
    {
      // There were some matches, pick the one closest to the beginning
      literal_cnt = __ffs(match_mask) - 1;

      // Given offset is different from lane to lane, we need to copy it from the right lane
      offset = SHFL(offset, literal_cnt);
      copy_distance = pos + literal_cnt - offset;
      copy_length = 4;

      local_match &= (0x2 << literal_cnt) - 1;
    }
    else
    {
      // There weren't any matches, continue
      literal_cnt = 32;
    }

    // Update hash up to the first 4 bytes of the copy length
    if (t <= literal_cnt && t == 31 - __clz(local_match))
    {
      hash_map[hash] = pos + t;
    }
    pos += literal_cnt;

    // Sync the potentially updated hash map before proceeding
    __syncwarp();
  } while (literal_cnt == 32 && pos < maxpos);
  return min(pos, len) - pos0;
}

/// @brief Returns the number of matching bytes for two byte sequences up to 63 bytes
static inline __device__ uint32_t Match60(const uint8_t *src1, const uint8_t *src2, uint32_t len, uint32_t t)
{
  uint32_t mismatch = BALLOT(t >= len || src1[t] != src2[t]);
  if (mismatch == 0)
  {
    mismatch = BALLOT(32 + t >= len || src1[32 + t] != src2[32 + t]);
    return 31 + __ffs(mismatch); // mismatch cannot be zero here if len <= 63
  }
  else
  {
    return __ffs(mismatch) - 1;
  }
}

/**
 * @brief Snappy compression device function
 * See http://github.com/google/snappy/blob/master/format_description.txt
 *
 * @param[in] inputs Source/Destination buffer information per block
 * @param[out] status Compression status per chunk
 * @param[in] count Number of blocks to compress
 **/
inline __device__ void do_snap(
  const uint8_t *__restrict__ device_in_ptr,
  const uint64_t device_in_bytes,
  uint8_t *const __restrict__ device_out_ptr,
  const uint64_t device_out_available_bytes,
  nvcompStatus_t *__restrict__ status,
  uint64_t *device_out_bytes
)
{
  assert(blockDim.x == COMP_THREADS_PER_BLOCK); // code below assumes there are 2 warps on dim x
  __shared__ __align__(16) snap_state_s state_g;

  snap_state_s *const s = &state_g;
  uint32_t t = threadIdx.x;
  uint32_t pos;
  const uint8_t *src;

  if (!t)
  {
    const uint8_t *src = device_in_ptr;
    uint32_t src_len = static_cast<uint32_t>(device_in_bytes);
    uint8_t *dst = device_out_ptr;
    uint32_t dst_len = device_out_available_bytes;
    if (dst_len == 0)
    {
      dst_len = get_max_compressed_length(src_len);
    }

    uint8_t *end = dst + dst_len;
    s->src = src;
    s->src_len = src_len;
    s->dst_base = dst;
    s->end = end;
    while (src_len > 0x7f)
    {
      if (dst < end)
      {
        dst[0] = src_len | 0x80;
      }
      dst++;
      src_len >>= 7;
    }
    if (dst < end)
    {
      dst[0] = src_len;
    }
    s->dst = dst + 1;
    s->literal_length = 0;
    s->copy_length = 0;
    s->copy_distance = 0;
  }

  // Each thread zeros out uint32_t elements
  for (uint32_t i = t; i < sizeof(s->hash_map) / sizeof(uint32_t); i += COMP_THREADS_PER_BLOCK)
  {
    reinterpret_cast<uint32_t *>(s->hash_map)[i] = 0;
  }
  __syncthreads();

  src = s->src;
  pos = 0;
  while (pos < s->src_len)
  {
    uint32_t literal_len = s->literal_length;
    uint32_t copy_len = s->copy_length;
    uint32_t copy_distance = s->copy_distance;
    __syncthreads();
    if (t < WARP_SIZE_U)
    {
      // WARP0: Encode literals and copies
      uint8_t *dst = s->dst;
      uint8_t *end = s->end;
      if (literal_len > 0)
      {
        dst = StoreLiterals(dst, end, src + pos, literal_len - 1, t);
        pos += literal_len;
      }
      if (copy_len > 0)
      {
        if (t == 0)
        {
          dst = StoreCopy(dst, end, copy_len, copy_distance);
        }
        pos += copy_len;
      }
      SYNCWARP();
      if (t == 0)
      {
        s->dst = dst;
      }
    }
    else
    {
      pos += literal_len + copy_len;
      // WARP1: Find a match using 12-bit hashes of 4-byte blocks
      uint32_t t5 = t & 0x1f;
      // Update literal_len, copy_len, and copy_distance
      literal_len = FindFourByteMatch(src, s->hash_map, s->src_len, pos, t5, copy_len, copy_distance);
      if (copy_len != 0)
      {
        uint32_t match_pos = pos + literal_len + copy_len; // NOTE: copy_len is always 4 here
        copy_len +=
          Match60(src + match_pos, src + match_pos - copy_distance, min(s->src_len - match_pos, 64 - copy_len), t5);
      }
      if (t5 == 0)
      {
        s->literal_length = literal_len;
        s->copy_length = copy_len;
        s->copy_distance = copy_distance;
      }
    }
    __syncthreads();
  }
  if (!t)
  {
    *device_out_bytes = s->dst - s->dst_base;
    if (status)
    {
      *status = (s->dst > s->end) ? nvcompErrorCannotCompress : nvcompSuccess;
    }
  }
}

} // namespace nvcomp
