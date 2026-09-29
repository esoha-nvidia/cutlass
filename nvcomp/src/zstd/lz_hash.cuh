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
#include <optional>

#include "constants.cuh"
#include "types.cuh"
#include "utils.cuh"

// #define ZSTD_LZ_LOGGING 1
// #define LZ_DETAIL_LOGGING 1

namespace zstd
{

using word_type = uint32_t;
using offset_type = uint32_t;

// This restricts us to 4GB chunk sizes (total buffer can be up to
// max(size_t)). We actually artificially restrict it to much less, to
// limit what we have to test, as well as to encourage users to exploit some
// parallelism.
using position_type = uint32_t;
using item_type = uint32_t;

/**
 * @brief The number of threads to use per chunk in compression.
 */
constexpr const int COMP_THREADS_PER_CHUNK = WARP_SIZE;

/**
 * @brief The number of elements in the hash table to use while performing
 * compression.
 */
constexpr const position_type HASH_BITS = 15U; // 17-18 is a point of diminishing returns.
constexpr const position_type HASH_TABLE_SIZE = 1U << HASH_BITS;

/**
 * @brief The value used to explicitly represent and invalid offset. This
 * denotes an empty slot in the hashtable.
 */
constexpr const int NULL_OFFSET = -1;

// TODO: this should be tunable by the user (fast / faster mode). It drives much of the runtime cost.
constexpr const int MAX_LITERAL_SEARCH_DEPTH = 2;

constexpr int END_BOUNDARY_FALLBACK_THRESH = WARP_SIZE + 12;

// + 4: will read up to 4 bytes before the thread's loc
// + 11: will read up to 11 bytes in per-thread length checking after the current byte
constexpr __device__ int NOMINAL_READ_BYTE_COUNT = WARP_SIZE + 11 + 4;

namespace
{

// Holds sequence values in a register / shmem cache until
// the end or 32 sequences are ready to write
struct LZSequenceOutputCache
{
  uint16_t lit_len[32];
  uint16_t mat_len[32];
  uint32_t offset[32];

  LZSequenceOutputCache() = default;

  inline __device__ void set_match_length(const uint32_t ix, const uint16_t match_length)
  {
    if (threadIdx.x == ix % 32)
    {
      mat_len[threadIdx.x] = match_length;
    }
  }

  inline __device__ void set_literal_length(const uint32_t ix, const uint16_t literal_length)
  {
    if (threadIdx.x == ix % 32)
    {
      lit_len[threadIdx.x] = literal_length;
    }
  }

  inline __device__ void set_offset(const uint32_t ix, const uint32_t new_offset)
  {
    if (threadIdx.x == ix % 32)
    {
      offset[threadIdx.x] = new_offset;
    }
  }
};

struct MatchHypothesis
{
  int match_offset;
  int offset_cost;
  int match_length;
  int prior_bytes;
  bool longer_match;
  bool is_match_valid;
  bool is_best;
  MatchHypothesis() = default;
  __device__ MatchHypothesis(int match_loc, bool is_hash)
      : match_offset(NULL_OFFSET)
      , offset_cost(0)
      , match_length(0)
      , prior_bytes(0)
      , longer_match(false)
      , is_match_valid(false)
      , is_best(false)
  {}
};

// This is "skinny" because it'll be shared amongst threads
struct SkinnyHypothesis
{
  int match_loc;
  int match_offset;
  int offset_cost;
  int match_length;
  bool is_match_valid;
  __device__ SkinnyHypothesis(int match_loc)
      : match_loc(match_loc)
      , match_offset(NULL_OFFSET)
      , offset_cost(0)
      , match_length(0)
      , is_match_valid(false)
  {}
};

} // namespace

inline __device__ int numValidThreadsToMask(const int numValidThreads)
{
  return WARP_ALL >> (WARP_SIZE - numValidThreads);
}

constexpr int INVALID_POSITION = -1;

inline __device__ uint32_t hash(uint32_t val)
{
  // needs to be 12 bits
  return (__brev(val) + (val ^ 0xc375)) & (HASH_TABLE_SIZE - 1);
}

inline __device__ uint32_t readWord(const uint8_t *const address)
{
  uint32_t word = 0;
#pragma unroll
  for (unsigned ix = 0; ix < 4; ++ix)
  {
    word |= uint32_t{address[ix]} << uint32_t{8 * ix};
  }

  return word;
}

struct MatchRes
{
  int byte_match;
  int prior_bytes;
  bool longer_match;

  __device__ MatchRes(const int align_req)
      : byte_match(0)
      , prior_bytes(0)
      , longer_match(false)
  {}
};

struct ReadWrapper
{
  uint32_t word_previous;
  uint32_t word_current;
  uint32_t word_next;
  uint32_t word_next2;
};

template <bool do_prev_read>
inline __device__ void do_read(const uint32_t *word_ptr, ReadWrapper &reads)
{
  if constexpr (do_prev_read)
  {
    reads.word_previous = *(word_ptr - 1);
  }
  reads.word_current = *(word_ptr++);
  reads.word_next = *(word_ptr++);
  reads.word_next2 = *(word_ptr);
}

inline __device__ MatchRes
checkMatch(const ReadWrapper &match_reads, const ReadWrapper &compare_reads, const bool do_previous, const int align_req)
{
  MatchRes res{align_req};

  // Get the first word
  uint32_t first_read_word = __funnelshift_r(match_reads.word_current, match_reads.word_next, align_req * 8);
  if (first_read_word != compare_reads.word_current)
  {
    return res;
  }

  if (do_previous)
  {
    uint32_t prior_word = __funnelshift_r(match_reads.word_previous, match_reads.word_current, align_req * 8);
    uint32_t bit_match_res = __brev(prior_word xor compare_reads.word_previous);
    int first_mismatch_bit = __ffs(bit_match_res) - 1;
    res.prior_bytes = first_mismatch_bit == -1 ? 4 : first_mismatch_bit / 8;
#ifdef ZSTD_LZ_LOGGING
    printf(
      "tid %d prior bytes %u %u mismatch res %u prior bytes %u align req %d\n",
      threadIdx.x,
      prior_word,
      compare_reads.word_previous,
      bit_match_res,
      res.prior_bytes,
      align_req
    );
#endif
  }

  uint32_t second_read_word = __funnelshift_r(match_reads.word_next, match_reads.word_next2, align_req * 8);
  uint32_t bit_match_res = second_read_word xor compare_reads.word_next;
  int first_mismatch_bit = __ffs(bit_match_res) - 1;

  if (first_mismatch_bit != -1)
  {
    res.byte_match = 4 + first_mismatch_bit / 8;
    return res;
  }

  uint32_t third = match_reads.word_next2 >> (align_req * 8);
  bit_match_res = third xor compare_reads.word_next2;
  first_mismatch_bit = __ffs(bit_match_res) - 1;
  int max_match = 4 - align_req;
  if (first_mismatch_bit == -1)
  {
    res.longer_match = true;
    res.byte_match = 8 + max_match;
  }
  else
  {
    const int match_bytes = first_mismatch_bit / 8;
    res.longer_match = match_bytes >= max_match;
    res.byte_match = 8 + std::min(max_match, match_bytes);
  }

  return res;
}

inline __device__ void
insertHashTableWarp(int *hashTable, const offset_type pos, const uint32_t next, bool active, const int numValidThreads)
{
  if (threadIdx.x < numValidThreads)
  {
    position_type hashPos = hash(next);

    atomicMax(&hashTable[hashPos], pos);
  }
}

inline __device__ void
fill_lz_hash_table(const uint8_t *input_buffer, const size_t lz_window_warmstart_size, int *hashTable)
{
  uint32_t hashIndex = 0;

  // Load 128 + 4 bytes into SHMEM
  constexpr int LOAD_SIZE = 128 + sizeof(uint32_t);
  __shared__ uint8_t shared_array[LOAD_SIZE];

  // Loads into shmem, then does 4 hash insertions per thread
  while (hashIndex + LOAD_SIZE < lz_window_warmstart_size)
  {
    uint8_t vals[5];
#pragma unroll
    for (int ix = 0; ix < 4; ++ix)
    {
      vals[ix] = input_buffer[hashIndex + threadIdx.x + 32 * ix];
    }
    bool extra = threadIdx.x + 128 < LOAD_SIZE;
    if (extra)
    {
      shared_array[threadIdx.x + 128] = input_buffer[hashIndex + threadIdx.x + 128];
    }

#pragma unroll
    for (int ix = 0; ix < 4; ++ix)
    {
      shared_array[threadIdx.x + 32 * ix] = vals[ix];
    }

    uint32_t this_val = 0;
    __syncwarp(WARP_ALL);

    // Initialize the value
    const int ix_base_thread = 4 * (threadIdx.x);
    int ix_shared;
    for (ix_shared = ix_base_thread; ix_shared < ix_base_thread + 4; ++ix_shared)
    {
      this_val += uint32_t{shared_array[ix_shared]} << uint32_t{8U * (ix_shared - ix_base_thread)};
    }

    // Iteration
    int ix_hash;
    for (ix_hash = 4 * threadIdx.x; ix_hash < 4 * (threadIdx.x + 1); ++ix_hash)
    {
      insertHashTableWarp(hashTable, hashIndex + ix_hash, this_val, true /*active*/, WARP_SIZE /*num active threads*/);

      // Setup final iteration
      this_val >>= 8;
      this_val <<= 8;
      this_val += uint32_t{shared_array[ix_shared++]} << uint32_t{(sizeof(uint32_t) - 1) * 8};
    }
    insertHashTableWarp(hashTable, hashIndex + ix_hash, this_val, true /*active*/, WARP_SIZE /*num active threads*/);
    hashIndex += 128;
    __syncwarp(WARP_ALL);
  }
}

inline __device__ position_type lengthOfMatch(
  const uint8_t *const data,
  const position_type prev_location,
  const position_type next_location,
  const position_type length,
  const position_type input_length
)
{
  assert(prev_location < next_location);
  position_type max_match_length = min(length - next_location, ZSTD_COMP_MAX_MATCH_LENGTH - input_length);
  position_type match_length = max_match_length;
  for (position_type j = 0; j < max_match_length; j += WARP_SIZE_U)
  {
    const position_type i = threadIdx.x + j;
    int match = i < max_match_length ? (data[prev_location + i] != data[next_location + i]) : 1;
    match = __ballot_sync(WARP_ALL, match);
    if (match)
    {
      match_length = j + __ffs(match) - 1;
      break;
    }
  }
  return match_length;
}

inline __device__ bool score_new_match(
  SkinnyHypothesis &best_hyp,
  const int new_match_len,
  const int new_offset_cost,
  const int new_offset,
  const int gain,
  const int current_loc
)
{
  bool is_best = true;
  if (best_hyp.is_match_valid)
  {
    // This scoring matches CPU L1 compression
    // The motivation is:
    // 1) It's expensive to keep scoring
    // 2) Every time you advance to a new match location,
    //    that's more literals that have to be compressed.
    // + gain on score_old is an amount of hysteresis that prioritizes a match that came earlier.
    int score_new = (new_match_len - new_offset_cost) * gain;
    int score_old = (best_hyp.match_length - best_hyp.offset_cost) * gain + gain;
#ifdef ZSTD_LZ_LOGGING
    print0(
      "score considered_gain %d old match score: %d new match score: %d compute %d best match len %d best offset %d "
      "high set bit %d this match len %d this offset %d\n",
      gain,
      score_old,
      score_new,
      (new_match_len - new_offset_cost),
      best_hyp.match_length,
      best_hyp.match_offset,
      highest_set_bit(best_hyp.match_offset),
      new_match_len,
      new_offset
    );
#endif
    is_best = score_new > score_old;
  }
  if (is_best)
  {
    best_hyp.match_length = new_match_len;
    best_hyp.offset_cost = new_offset_cost;
    best_hyp.match_offset = new_offset;
    best_hyp.is_match_valid = true;
    best_hyp.match_loc = current_loc;
  }
  return is_best;
}

inline __device__ void compute_match_length(
  const int decomp_idx,
  const int current_loc,
  const int offset,
  const uint8_t *input_buff,
  const size_t decomp_size,
  int &match_len
)
{
  // Compute the full match length
  int this_start_output = decomp_idx + current_loc + match_len;
  int match_offset_loc = this_start_output - offset;
  match_len += lengthOfMatch(input_buff, match_offset_loc, this_start_output, decomp_size, match_len);
}

inline __device__ void get_next_match(
  SkinnyHypothesis &best_hyp,
  unsigned vote,
  int match_val,
  const int &decomp_idx,
  const uint8_t *&input_buff,
  const int &decomp_size,
  MatchHypothesis &repeat_match_hyp,
  MatchHypothesis &hash_match_hyp,
  int *repeat_offsets
)
{
  // This updates by powers of 2 each time we find a possible match.
  // This causes us to prefer stopping working on this iteration,
  // and prioritizes smaller literal lengths
  int gain = 0;
  int iter = 0;

  int next_loc = __ffs(vote) - 1;
  int next_match_val = __shfl_sync(WARP_ALL, match_val, next_loc);
  best_hyp.match_loc = next_loc;

  int prior_offset = __shfl_up_sync(WARP_ALL, hash_match_hyp.match_offset, 1);

  // This clears the thread's hash match bits if the hash offset is exactly the same (or +1) as the previous thread's hash offset
  if (threadIdx.x > next_loc and hash_match_hyp.is_match_valid and
      (hash_match_hyp.match_offset == prior_offset or hash_match_hyp.match_offset - prior_offset == 1))
  {
    match_val &= LZ_REPEAT_OFFSET_MASK;
  }

  prior_offset = __shfl_up_sync(WARP_ALL, repeat_match_hyp.match_offset, 1);
  // This clears the thread's repeat match bits if the repeat offset is exactly the same as the previous thread's repeat offset
  if (threadIdx.x > next_loc and repeat_match_hyp.is_match_valid and repeat_match_hyp.match_offset == prior_offset)
  {
    match_val &= LZ_HASH_OFFSET_MASK;
  }

  // Make sure correct match_val is passed
  if (match_val & LZ_HAS_HASH_OFFSET_BIT)
  {
    assert(hash_match_hyp.is_match_valid);
  }
  if (match_val & LZ_HAS_REPEAT_OFFSET_BIT)
  {
    assert(repeat_match_hyp.is_match_valid);
  }
  // Set Default value for Coverity.
  // repeat_offset_cost will be used only if match_value & LZ_HAS_REPEAT_OFFSET_BIT is true and thus only if repeat_match_hyp.is_match_valid is true.
  int repeat_offset_cost = 0;
  if (repeat_match_hyp.is_match_valid)
  {
    if (repeat_match_hyp.match_offset == repeat_offsets[0] or repeat_match_hyp.match_offset == repeat_offsets[1])
    {
      repeat_offset_cost = 0;
    }
    else
    {
      repeat_offset_cost = repeat_match_hyp.offset_cost;
    }
  }

  // TODO: consider adding repeat offset 3 -- this should help low compression cases
  // We could also assess all the repeat offsets to give more options -- now that match length finding isn't the bottleneck
  // Note, we started with 2 offsets because this matched ZSTD CPU L1

  // TODO: the scoring should take into account the prior bytes.
  // The prior byte scoring takes advantage of the fact that the hash table isn't complete -- so a previous thread
  // that should've found this hash entry and started earlier, didn't.
  while (vote)
  {
    int current_loc = next_loc;
    int this_match_val = next_match_val;
    vote ^= 1U << current_loc;

    // This shuffle can provide all the info needed to determine whether match length compute is required (even for multiple matches...)    // Separately, we'll try to make this happen as infrequently as possible.
    if (not this_match_val or (current_loc - best_hyp.match_loc) > MAX_LITERAL_SEARCH_DEPTH)
    {
      return;
    }
    if (vote)
    {
      next_loc = __ffs(vote) - 1;
    }
    int repeat_match_len = __shfl_sync(WARP_ALL, repeat_match_hyp.match_length, current_loc);
    int repeat_offset = __shfl_sync(WARP_ALL, repeat_match_hyp.match_offset, current_loc);
    int this_repeat_offset_cost = __shfl_sync(WARP_ALL, repeat_offset_cost, current_loc);
    int hash_match_len = __shfl_sync(WARP_ALL, hash_match_hyp.match_length, current_loc);
    int hash_offset = __shfl_sync(WARP_ALL, hash_match_hyp.match_offset, current_loc);
    int this_hash_offset_cost = __shfl_sync(WARP_ALL, hash_match_hyp.offset_cost, current_loc);
    next_match_val = __shfl_sync(WARP_ALL, match_val, next_loc);

#ifdef ZSTD_LZ_LOGGING
    print0(
      "current_loc %d best match thread %d vote %u this match val %d\n",
      current_loc,
      best_hyp.match_loc,
      vote,
      this_match_val
    );
#endif

    gain += (1 << iter++);

    if (this_match_val & LZ_REPEAT_OFFSET_LENGTH_COMPUTE_BIT)
    {
      compute_match_length(decomp_idx, current_loc, repeat_offset, input_buff, decomp_size, repeat_match_len);
    }

    if (this_match_val & LZ_HASH_OFFSET_LENGTH_COMPUTE_BIT)
    {
      compute_match_length(decomp_idx, current_loc, hash_offset, input_buff, decomp_size, hash_match_len);
    }

    if (this_match_val & LZ_HAS_REPEAT_OFFSET_BIT)
    {
#ifdef ZSTD_LZ_LOGGING
      print0(
        "repeat add match loc %d match offset %d longer match %d match len %d\n",
        current_loc,
        repeat_match_hyp.match_offset,
        repeat_match_hyp.longer_match,
        repeat_match_len
      );
#endif

      repeat_match_hyp.is_best =
        score_new_match(best_hyp, repeat_match_len, this_repeat_offset_cost, repeat_offset, gain, current_loc);
    }

    if (this_match_val & LZ_HAS_HASH_OFFSET_BIT)
    {
#ifdef ZSTD_LZ_LOGGING
      print0("hash add match loc %d match offset %d match len %d\n", current_loc, hash_offset, hash_match_len);
#endif
      const bool is_best =
        score_new_match(best_hyp, hash_match_len, this_hash_offset_cost, hash_offset, gain, current_loc);

      hash_match_hyp.is_best = is_best;
      if (not is_best)
      {
// Failed hash match. Return.
#ifdef ZSTD_LZ_LOGGING
        print0("abort because hash match failed\n");
#endif
        return;
      }
    }
  }
}

struct DecompDataSharedVals
{
  bool waiting;
  uint8_t __align__(128) stage_buff[256];
  int start_offset;
  int base_align_req;
};

// This loads directly into the buffer until we get to 128 byte aligned location.
// Frequently, this will save
struct DecompDataCache
{
  cg::thread_block group;
  DecompDataSharedVals &shared_vals;
  const uint8_t *&input_buff;
  const int &input_buff_size;

  __device__ DecompDataCache(DecompDataSharedVals &shared_vals, const uint8_t *&input_buff, const int &input_buff_size)
      : group(cg::this_thread_block())
      , shared_vals(shared_vals)
      , input_buff(input_buff)
      , input_buff_size(input_buff_size)
  {
    shared_vals.waiting = false;
    shared_vals.start_offset = 0;
    shared_vals.base_align_req = (uintptr_t)input_buff % 128;
    refill(0);
  }

  inline __device__ void refill(const int load_ix)
  {
#ifdef ZSTD_LZ_LOGGING
    print0("refill offset %d\n", shared_vals.start_offset);
#endif

    __syncwarp();

    int align_offset = (load_ix + shared_vals.base_align_req) % 128;
    if (load_ix - align_offset + 256 <= input_buff_size)
    {
      shared_vals.start_offset = load_ix - align_offset;
      cg::memcpy_async(
        group,
        shared_vals.stage_buff,
        ::cuda::aligned_size_t<128>(256),
        input_buff + shared_vals.start_offset,
        ::cuda::aligned_size_t<128>(256)
      );
      shared_vals.waiting = true;
    }
    else
    {
      shared_vals.start_offset = std::numeric_limits<int>::lowest();
    }

    __syncwarp();
  }

  inline __device__ void finish_wait()
  {
    if (shared_vals.waiting)
    {
      cg::wait(group);
      shared_vals.waiting = false;
    }
  }

  inline __device__ uint8_t get_byte(const int position)
  {
    const uint8_t *this_input;
    if (position < shared_vals.start_offset + 256 and position >= shared_vals.start_offset)
    {
      this_input = shared_vals.stage_buff + position - shared_vals.start_offset;
    }
    else
    {
      this_input = input_buff + position;
    }
    return *this_input;
  }

  inline __device__ void read_buff(uint8_t *result, const int buff_size, const int offset)
  {
    finish_wait();
    __syncwarp();

    for (int ix = threadIdx.x; ix < buff_size; ix += WARP_SIZE)
    {
      int this_offset = offset + ix;
      const uint8_t *this_input{};
      if (this_offset < shared_vals.start_offset + 256)
      {
        this_input = shared_vals.stage_buff + this_offset - shared_vals.start_offset;
      }
      else
      {
        this_input = input_buff + this_offset;
      }
      result[ix] = *this_input;
    }
    __syncwarp();
    const int final_offset = offset + WARP_SIZE;
    // 192 was chosen as "if less than 64 bytes are remaining in the cache, asynchronously advance the cache"
    if (final_offset >= shared_vals.start_offset + 192)
    {
      refill(final_offset);
    }
  }
};

inline __device__ void do_initialize_hyp(
  MatchHypothesis &match_hyp,
  const position_type match_position,
  const ReadWrapper &match_reads,
  const ReadWrapper &compare_reads,
  const int ix_offset,
  const bool do_previous,
  const int align_req
)
{
  auto match_res = checkMatch(match_reads, compare_reads, do_previous, align_req);
  match_hyp.is_match_valid = match_res.byte_match >= ZSTD_MIN_MATCH_LENGTH;
#ifdef LZ_DETAIL_LOGGING
  printf(
    "loc %d offset %d match len %d location %u longer match %d\n",
    threadIdx.x,
    ix_offset,
    match_res.byte_match,
    match_position,
    match_res.longer_match
  );
#endif

  if (match_hyp.is_match_valid)
  {
    // We use the values of these variables in uninitialized state to indicate an invalid match with fewer shuffles
    match_hyp.match_offset = ix_offset;
    match_hyp.longer_match = match_res.longer_match;
    match_hyp.match_length = match_res.byte_match;
    match_hyp.prior_bytes = match_res.prior_bytes;
  }
}

inline __device__ void select_lz_matches_and_update_decomp(
  unsigned &vote,
  int &decomp_idx,
  const uint8_t *&input_buff,
  const int match_val,
  MatchHypothesis &repeat_match_hyp,
  MatchHypothesis &hash_match_hyp,
  const int &decomp_size,
  int *const hashTable,
  const uint32_t read_val,
  int &tokenStart,
  CompressSequenceBuffer &sequences,
  uint8_t *literals,
  int *repeat_offsets,
  int &nsequences,
  int &nliterals,
  const bool active,
  const int numValidThreads,
  LZSequenceOutputCache &output_cache
)
{
  bool do_update_hash_table = active;
  bool is_lit = active;
  const position_type base_decomp_idx = decomp_idx;

  while (true)
  {
    SkinnyHypothesis best_hyp{32};
    get_next_match(
      best_hyp,
      vote,
      match_val,
      base_decomp_idx,
      input_buff,
      decomp_size,
      repeat_match_hyp,
      hash_match_hyp,
      repeat_offsets
    );
    assert(best_hyp.match_length >= 4);

    if (threadIdx.x >= best_hyp.match_loc and threadIdx.x - best_hyp.match_loc < best_hyp.match_length)
    {
      if (threadIdx.x != best_hyp.match_loc)
      {
        do_update_hash_table = false;
      }
      is_lit = false;
    }

    int ix_end_of_match = best_hyp.match_loc + best_hyp.match_length;

    // Insert up to the beginning of the match in the hash table, but don't extend to words
    // in the repeated match (where offset < match length)
    int num_lits = base_decomp_idx + best_hyp.match_loc - tokenStart;

    if (num_lits > 0)
    {
      // Check whether a backwards search could provide advantages, up to warp size values
      int incr = 0; // default value for Coverity to stop the issue. Real value will be set by first thread in warp.
      if (threadIdx.x == best_hyp.match_loc)
      {
        incr = repeat_match_hyp.is_best ? repeat_match_hyp.prior_bytes : hash_match_hyp.prior_bytes;
      }

      incr = std::min(
        std::min(num_lits, __shfl_sync(WARP_ALL, incr, best_hyp.match_loc)),
        ZSTD_COMP_MAX_MATCH_LENGTH - best_hyp.match_length
      );

      best_hyp.match_length += incr;
      num_lits -= incr;
#ifdef ZSTD_LZ_LOGGING
      if (incr > 0)
      {
        print0("Added %d bytes before the match. This is really useful for an imperfect hash table.\n", incr);
      }
#endif
      if (incr > best_hyp.match_loc)
      {
        // Need to reduce the number of literals that already occured
        nliterals -= (incr - best_hyp.match_loc);
      }

      if (best_hyp.match_loc - threadIdx.x <= incr and best_hyp.match_loc - threadIdx.x > 0)
      {
        is_lit = false;
      }
    }

    // -> write our token and literal length
    uint32_t output_offset = best_hyp.match_offset;
    if (num_lits > 0 and best_hyp.match_offset == repeat_offsets[0])
    {
      output_offset = 1;
    }
    else
    {
      if (best_hyp.match_offset == repeat_offsets[1])
      {
        output_offset = num_lits > 0 ? 2 : 1;
      }
      else
      {
        output_offset = best_hyp.match_offset + 3;
      }

      __syncwarp();
      if (threadIdx.x == 0)
      {
        repeat_offsets[1] = repeat_offsets[0];
        repeat_offsets[0] = best_hyp.match_offset;
      }
      __syncwarp();
    }

    // Write the sequence
    output_cache.set_offset(nsequences, output_offset);
    output_cache.set_match_length(nsequences, best_hyp.match_length);
    output_cache.set_literal_length(nsequences, num_lits);
    ++nsequences;
    if (nsequences % WARP_SIZE == 0)
    {
      sequences.set_literal_length(nsequences - WARP_SIZE + threadIdx.x, output_cache.lit_len[threadIdx.x]);
      sequences.set_match_length(nsequences - WARP_SIZE + threadIdx.x, output_cache.mat_len[threadIdx.x]);
      sequences.set_offset(nsequences - WARP_SIZE + threadIdx.x, output_cache.offset[threadIdx.x]);
    }

// update our position
#ifdef ZSTD_LZ_LOGGING
    print0(
      "seq %d match len %d lit len %d off %d best off %d total literals %d decomp bytes %d idx %d rep off %u %u match "
      "thread %d\n",
      nsequences - 1,
      best_hyp.match_length,
      num_lits,
      output_offset,
      best_hyp.match_offset,
      nliterals,
      decomp_size,
      decomp_idx,
      repeat_offsets[0],
      repeat_offsets[1],
      best_hyp.match_loc
    );
#endif

    decomp_idx = tokenStart + best_hyp.match_length + num_lits;
    tokenStart = decomp_idx;

    // See if we should continue
    if (ix_end_of_match >= WARP_SIZE)
    {
      break;
    }
    vote = vote & ~((1 << ix_end_of_match) - 1);

    if (not vote)
    {
      break;
    }
  }

  insertHashTableWarp(hashTable, base_decomp_idx + threadIdx.x, read_val, do_update_hash_table, numValidThreads);

  __syncwarp();

  decomp_idx = max(decomp_idx, base_decomp_idx + numValidThreads);

  unsigned lit_mask = __ballot_sync(WARP_ALL, is_lit);
  const int ix_lit = nliterals + __popc(lit_mask & ((1 << threadIdx.x) - 1));
  nliterals += __popc(lit_mask);
  if (is_lit)
  {
    literals[ix_lit] = read_val & 0xff;
  }
}

inline __device__ void initialize_hyp_fallback(
  MatchHypothesis &match_hyp,
  const uint32_t read_word,
  const ReadWrapper &compare_reads,
  const int ix_offset
)
{
#ifdef LZ_DETAIL_LOGGING
  printf("loc %d offset %d read val %u\n", threadIdx.x, ix_offset, compare_reads.word_current);
#endif

  if (read_word == compare_reads.word_current)
  {
    match_hyp.match_offset = ix_offset;
    match_hyp.longer_match = true;
    match_hyp.match_length = 4;
    match_hyp.is_match_valid = true;
  }
}

inline __device__ void initialize_hyp(
  MatchHypothesis &match_hyp,
  const int match_position,
  const uint8_t *&input_buff,
  const int this_idx,
  const int decomp_size,
  const int decomp_idx,
  const ReadWrapper &compare_reads,
  const ReadWrapper &match_reads,
  const int align_req,
  const bool do_prior
)
{
  const int ix_offset = this_idx - match_position;
  if (decomp_idx < (decomp_size - END_BOUNDARY_FALLBACK_THRESH))
  {
    do_initialize_hyp(
      match_hyp,
      match_position,
      match_reads,
      compare_reads,
      ix_offset,
      do_prior and decomp_idx >= 4 and match_position >= 8,
      align_req
    );
  }
  else
  {
    initialize_hyp_fallback(match_hyp, readWord(&input_buff[match_position]), compare_reads, ix_offset);
  }
}

inline __device__ void lz_compress_tile_greedy_hash(
  const unsigned char *decompData,
  const int decompBytes,
  int *const hashTable,
  CompressSequenceBuffer &sequences,
  uint8_t *literals,
  int &nsequences,
  int &nliterals,
  int *repeat_offsets,
  uint32_t start_offset
)
{
  __shared__ uint8_t iter_load_buff[WARP_SIZE + NOMINAL_READ_BYTE_COUNT];
  uint8_t *offset_iter_buff = iter_load_buff + 4;

  __shared__ DecompDataSharedVals cache_shared_vals;
  int decomp_idx = start_offset;
  int tokenStart = start_offset;
  __shared__ int decomp_size;

  decomp_size = decompBytes;

  const __shared__ uint8_t *input_buff;
  input_buff = decompData;

  DecompDataCache data_cache{cache_shared_vals, input_buff, decomp_size};

  __shared__ ReadWrapper compare_read_array[WARP_SIZE];
  ReadWrapper &compare_reads = compare_read_array[threadIdx.x];
  // ReadWrapper compare_reads;

  assert(blockDim.x == COMP_THREADS_PER_CHUNK);
  static_assert(
    COMP_THREADS_PER_CHUNK <= 32,
    "Compression can be done with at "
    "most one warp"
  );
  __shared__ LZSequenceOutputCache output_cache;

  nsequences = 0;
  nliterals = 0;

  __syncwarp();

  while (true)
  {

    if (decomp_idx + ZSTD_MIN_MATCH_LENGTH >= decomp_size)
    {
      // no match -- literals to the end. Does this after the last sequence
      // int rem_literals = decomp_size - tokenStart;
      int rem_literals = decomp_size - decomp_idx;

      if (threadIdx.x < rem_literals)
      {
        literals[nliterals + threadIdx.x] = input_buff[decomp_idx + threadIdx.x];
      }

      nliterals += rem_literals;
#ifdef ZSTD_LZ_LOGGING
      asm("exit;");
#endif
      break;
    }

    // begin adding tokens to the hash table until we find a match
    // Small bug fix -- the +1 allows us to find matches for the last 4 bytes of the buffer
    const int numValidThreads = min(static_cast<int>(decomp_size - decomp_idx - ZSTD_MIN_MATCH_LENGTH + 1), 32);
#ifdef ZSTD_LZ_LOGGING
    print0(
      "decomp idx %d num valid %d prev offsets %u %u\n",
      decomp_idx,
      numValidThreads,
      repeat_offsets[0],
      repeat_offsets[1]
    );
#endif

    bool active = threadIdx.x < numValidThreads;

    position_type this_idx = decomp_idx + threadIdx.x;

    MatchHypothesis repeat_match_hyp(threadIdx.x, false /*is hash*/);
    MatchHypothesis hash_match_hyp(threadIdx.x, true /*is hash*/);

    // Note, in the below, we use scopes to communicate to the compiler explicitly
    // when we're done with values so that the registers can be re-used.
    // In theory the compiler could figure this out on its own, but it's been shown
    // that this approach can help with register usage.
    {
      const int prior_offset = std::min(decomp_idx, 4);
      const int read_bytes = std::min(NOMINAL_READ_BYTE_COUNT, decomp_size - decomp_idx + prior_offset);
      // We load data such that a maximum of 4 previous bytes are also loaded,
      // and the byte at decomp_idx is cached at iter_load_buff[4] (using offset_iter_buff)
      data_cache.read_buff(&iter_load_buff[4 - prior_offset], read_bytes, decomp_idx - prior_offset);
    }

    int repeat_position = INVALID_POSITION;
    int hash_position = INVALID_POSITION;
    if (active)
    {
      ReadWrapper hyp_reads;
      {
        int ix_recent_offset = this_idx - tokenStart == 0;
        if (repeat_offsets[ix_recent_offset] > 0)
        {
          const position_type repeat_ix_offset = repeat_offsets[ix_recent_offset];
          if (this_idx > repeat_ix_offset)
          {
            repeat_position = this_idx - repeat_ix_offset;
            if (decomp_idx < (decomp_size - END_BOUNDARY_FALLBACK_THRESH))
            {
              int repeat_align_req = repeat_position % 4;
              const uint32_t *word_ptr =
                reinterpret_cast<const uint32_t *>(input_buff + repeat_position - repeat_align_req);
              // Note, we don't read the prior word because previous byte match finding only applies to the hash matches
              do_read<false /*do read prior word*/>(word_ptr, hyp_reads);
            }
          }
        }
      }

      // Get a match from the hash table, or repeat offsets
      {
        int compare_align_req = threadIdx.x % 4;
        uint32_t *aligned_shmem = reinterpret_cast<uint32_t *>(&offset_iter_buff[threadIdx.x - compare_align_req]);
        do_read<true>(aligned_shmem, compare_reads);
        compare_reads.word_previous =
          __funnelshift_r(compare_reads.word_previous, compare_reads.word_current, compare_align_req * 8);
        compare_reads.word_current =
          __funnelshift_r(compare_reads.word_current, compare_reads.word_next, compare_align_req * 8);

        // Start this read here to hide as much latency as possible
        hash_position = hashTable[hash(compare_reads.word_current)];

        //
        if (decomp_idx < (decomp_size - END_BOUNDARY_FALLBACK_THRESH))
        {
          compare_reads.word_next =
            __funnelshift_r(compare_reads.word_next, compare_reads.word_next2, compare_align_req * 8);
          compare_reads.word_next2 >>= compare_align_req * 8;
          if (compare_align_req != 0)
          {
            for (int ix = 4 - compare_align_req; ix < 4; ++ix)
            {
              compare_reads.word_next2 += (offset_iter_buff[threadIdx.x + 8 + ix] << (ix * 8));
            }
          }
        }
      }

#ifdef LZ_DETAIL_LOGGING
      printf(
        "loc %d read val %u this idx %u should be %u\n",
        threadIdx.x,
        compare_reads.word_current,
        this_idx,
        readWord(&iter_load_buff[4 + threadIdx.x])
      );
#endif

      // try to find a local match -- a local match is a match of the same word within the warp
      int match_mask_self =
        __match_any_sync(numValidThreadsToMask(numValidThreads), static_cast<unsigned>(compare_reads.word_current));

      if (repeat_position != INVALID_POSITION)
      {
        int repeat_align_req = repeat_position % 4;
        initialize_hyp(
          repeat_match_hyp,
          repeat_position,
          input_buff,
          this_idx,
          decomp_size,
          decomp_idx,
          compare_reads,
          hyp_reads,
          repeat_align_req,
          false /*prior valid*/
        );
      }
      if (match_mask_self)
      {
        int first_loc = __ffs(match_mask_self) - 1;
        // determine the global position for the finding thread
        if (first_loc < threadIdx.x)
        {
          hash_position = first_loc + decomp_idx;
        }
      }

      if (hash_position != INVALID_POSITION and this_idx - hash_position < MAX_OFFSET)
      {
        // do the reads
        int hash_align_req = hash_position % 4;
        if (decomp_idx < (decomp_size - END_BOUNDARY_FALLBACK_THRESH))
        {
          const uint32_t *word_ptr = reinterpret_cast<const uint32_t *>(input_buff + hash_position - hash_align_req);
          if (hash_position >= 8)
          {
            do_read<true>(word_ptr, hyp_reads);
          }
          else
          {
            do_read<false>(word_ptr, hyp_reads);
          }
        }
        initialize_hyp(
          hash_match_hyp,
          hash_position,
          input_buff,
          this_idx,
          decomp_size,
          decomp_idx,
          compare_reads,
          hyp_reads,
          hash_align_req,
          true /* prior valid*/
        );
      }
    }

    // Now, inspect the matches to find the best.
    int match_val = repeat_match_hyp.is_match_valid * LZ_HAS_REPEAT_OFFSET_BIT +
                    hash_match_hyp.is_match_valid * LZ_HAS_HASH_OFFSET_BIT;
    unsigned vote = __ballot_sync(WARP_ALL, match_val);
    if (not vote)
    {
      // insert everything into hash table
      insertHashTableWarp(hashTable, decomp_idx + threadIdx.x, compare_reads.word_current, active, numValidThreads);

      const int ix_lit = nliterals + threadIdx.x;
      nliterals += numValidThreads;
      if (active)
      {
        literals[ix_lit] = compare_reads.word_current & 0xff;
      }

      decomp_idx += numValidThreads;
      continue;
    }

    if (repeat_match_hyp.is_match_valid)
    {
      repeat_match_hyp.offset_cost = highest_set_bit(repeat_match_hyp.match_offset);
      if (repeat_match_hyp.longer_match)
      {
        match_val |= LZ_REPEAT_OFFSET_LENGTH_COMPUTE_BIT;
      }
    }

    if (hash_match_hyp.is_match_valid)
    {
      hash_match_hyp.offset_cost = highest_set_bit(hash_match_hyp.match_offset);
      if (hash_match_hyp.longer_match)
      {
        match_val |= LZ_HASH_OFFSET_LENGTH_COMPUTE_BIT;
      }
    }

    select_lz_matches_and_update_decomp(
      vote,
      decomp_idx,
      input_buff,
      match_val,
      repeat_match_hyp,
      hash_match_hyp,
      decomp_size,
      hashTable,
      compare_reads.word_current,
      tokenStart,
      sequences,
      literals,
      repeat_offsets,
      nsequences,
      nliterals,
      active,
      numValidThreads,
      output_cache
    );
  }

  const int final_seq_count = nsequences % WARP_SIZE;

  if (threadIdx.x < final_seq_count)
  {
    sequences.set_literal_length(nsequences - final_seq_count + threadIdx.x, output_cache.lit_len[threadIdx.x]);
    sequences.set_match_length(nsequences - final_seq_count + threadIdx.x, output_cache.mat_len[threadIdx.x]);
    sequences.set_offset(nsequences - final_seq_count + threadIdx.x, output_cache.offset[threadIdx.x]);
  }
  __syncwarp();
}

} // namespace zstd
