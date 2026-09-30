/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <cub/cub.cuh>

#include "ans/constants.hpp"

namespace ans_gpu_lib::detail
{

// ----------------------------------------------------------------------
// Parallel variant of the single-threaded `normalize_counts_chunk_dx`.
//
// Callers handle empty chunks before invoking this helper.
struct normalize_max_op
{
  __device__ __forceinline__ uint32_t operator()(const uint32_t &a, const uint32_t &b) const { return a > b ? a : b; }
};

static inline __device__ void normalize_counts_chunk_parallel(
  const unsigned int idx_in_block,
  const uint32_t *chunk_counts,
  int *chunk_norm_counts,
  uint32_t *max_symbol_value_out,
  const IndexT uncomp_chunk_size,
  uint8_t tablelog
)
{

  constexpr int N_SYMBOLS = NV_SYMBOL_COUNT;
  static_assert(
    N_SYMBOLS % NUM_COMP_THREADS_PER_CTA == 0,
    "NUM_COMP_THREADS_PER_CTA must evenly divide NV_SYMBOL_COUNT"
  );
  constexpr int ITEMS_PER_THREAD = N_SYMBOLS / NUM_COMP_THREADS_PER_CTA;

  constexpr int16_t NOT_ASSIGNED = -2;

  assert(uncomp_chunk_size > 0);

  // Phase 1 — parallel classification and track the largest index with a non-zero count
  const IndexT total_initial = uncomp_chunk_size;
  const uint32_t table_size = 1u << tablelog;
  const uint32_t low_prob_threshold = total_initial / table_size;
  // 3 * total_initial cannot overflow: a chunk holds at most 1 << 24 symbols.
  const uint32_t one_threshold = (3 * total_initial) / (2 * table_size);

  const uint32_t base = idx_in_block * static_cast<uint32_t>(ITEMS_PER_THREAD);

  uint32_t my_last_nonzero = 0;
  uint32_t my_remaining_dec = 0;
  uint64_t my_total_dec = 0;

#pragma unroll
  for (int e = 0; e < ITEMS_PER_THREAD; ++e)
  {
    const uint32_t i = base + static_cast<uint32_t>(e);
    const uint32_t c = chunk_counts[i];
    int16_t nc = NOT_ASSIGNED;
    if (c == 0)
    {
      nc = 0;
    }
    else
    {
      my_last_nonzero = i;
      if (c <= low_prob_threshold)
      {
        nc = -1;
        my_remaining_dec += 1u;
        my_total_dec += static_cast<uint64_t>(c);
      }
      else if (c <= one_threshold)
      {
        nc = 1;
        my_remaining_dec += 1u;
        my_total_dec += static_cast<uint64_t>(c);
      }
    }
    chunk_norm_counts[i] = nc;
  }

  using BlockReduceU32 = nvcomp::cub::BlockReduce<uint32_t, NUM_COMP_THREADS_PER_CTA>;
  using BlockReduceU64 = nvcomp::cub::BlockReduce<uint64_t, NUM_COMP_THREADS_PER_CTA>;
  __shared__ typename BlockReduceU32::TempStorage last_nonzero_temp;
  __shared__ typename BlockReduceU32::TempStorage remaining_dec_temp;
  __shared__ typename BlockReduceU64::TempStorage total_dec_temp;

  uint32_t last_nonzero_red = BlockReduceU32(last_nonzero_temp).Reduce(my_last_nonzero, normalize_max_op());
  uint32_t remaining_dec_red = BlockReduceU32(remaining_dec_temp).Sum(my_remaining_dec);
  uint64_t total_dec_red = BlockReduceU64(total_dec_temp).Sum(my_total_dec);
  __syncthreads();

  __shared__ uint32_t s_max_symbol_value;
  __shared__ uint32_t s_remaining;
  __shared__ uint64_t s_total;

  if (idx_in_block == 0)
  {
    s_max_symbol_value = last_nonzero_red;
    s_remaining = table_size - remaining_dec_red;
    s_total = total_initial - total_dec_red;
    *max_symbol_value_out = last_nonzero_red;
  }
  __syncthreads();

  uint32_t max_symbol_value = s_max_symbol_value;
  uint32_t remaining = s_remaining;
  IndexT total = static_cast<IndexT>(s_total);
  __syncthreads();

  // Note: With the current tablelog of 10, the following four conditions are never true in testing.
  if (remaining == 0)
  {
    return;
  }

  /* every symbol was less than the one_threshold, symbol probabilities are close
  * to uniform, and thus the data is probably incompressible */
  if ((total / remaining) > one_threshold)
  {
    if (idx_in_block == 0)
    {
      uint32_t local_one_threshold = static_cast<uint32_t>((3 * total) / (2 * remaining));
      uint32_t local_remaining = remaining;
      IndexT local_total = total;
      for (uint32_t i = 0; i <= max_symbol_value; ++i)
      {
        if ((chunk_norm_counts[i] == NOT_ASSIGNED) && (chunk_counts[i] <= local_one_threshold))
        {
          chunk_norm_counts[i] = 1;
          local_remaining -= 1;
          local_total -= chunk_counts[i];
        }
      }
      s_remaining = local_remaining;
      s_total = local_total;
    }
    __syncthreads();
    remaining = s_remaining;
    total = static_cast<IndexT>(s_total);
  }

  /* every symbol was less than the one_threshold, symbol probabilities are close
  * to uniform, and thus the data is probably incompressible */
  if (table_size - remaining == max_symbol_value + 1)
  {
    if (idx_in_block == 0)
    {
      uint32_t max_symbol = 0;
      uint32_t max_count = 0;
      for (uint32_t i = 0; i <= max_symbol_value; ++i)
      {
        if (chunk_counts[i] > max_count)
        {
          max_symbol = i;
          max_count = chunk_counts[i];
        }
      }
      chunk_norm_counts[max_symbol] += static_cast<int>(remaining);
    }
    return;
  }

  // all symbols were low probability, distribute the rest of table range
  if (total == 0)
  {
    if (idx_in_block == 0)
    {
      uint32_t local_remaining = remaining;
      for (uint32_t i = 0; local_remaining > 0; i = (i + 1) % (max_symbol_value + 1))
      {
        if (chunk_norm_counts[i] > 0)
        {
          chunk_norm_counts[i]++;
          local_remaining--;
        }
      }
    }
    return;
  }

  /* distribute higher probability symbols: The idea is to partition the remaining space
     * left in the table evenly among the remaining counts. By adding the value mid to 
     * calculate the step and then right shifting to get actual values, we ensure that
     * the normalized frequencies sum to the exactly remaining space left in the table,
     * as long as total is less than mid. */
  const uint64_t freq_offset_log = 62u - tablelog;
  const uint64_t mid = (1ULL << (freq_offset_log - 1)) - 1ULL;
  const uint64_t step = ((1ULL << freq_offset_log) * remaining + mid) / total;

  uint64_t my_contrib[ITEMS_PER_THREAD];
  int my_existing[ITEMS_PER_THREAD];
  bool my_active[ITEMS_PER_THREAD];

#pragma unroll
  for (int e = 0; e < ITEMS_PER_THREAD; ++e)
  {
    const uint32_t i = base + static_cast<uint32_t>(e);
    my_existing[e] = chunk_norm_counts[i];
    const bool active = (i <= max_symbol_value) && (my_existing[e] == NOT_ASSIGNED);
    my_active[e] = active;
    my_contrib[e] = active ? (step * static_cast<uint64_t>(chunk_counts[i])) : 0ULL;
  }

  using BlockScanU64 = nvcomp::cub::BlockScan<uint64_t, NUM_COMP_THREADS_PER_CTA>;
  __shared__ typename BlockScanU64::TempStorage bs_temp;

  uint64_t my_prefix[ITEMS_PER_THREAD];
  BlockScanU64(bs_temp).ExclusiveSum(my_contrib, my_prefix);
  __syncthreads();

#pragma unroll
  for (int e = 0; e < ITEMS_PER_THREAD; ++e)
  {
    if (my_active[e])
    {
      const uint32_t i = base + static_cast<uint32_t>(e);
      const uint64_t cumul_before = mid + my_prefix[e];
      const uint64_t cumul_after = cumul_before + my_contrib[e];
      const uint64_t start = cumul_before >> freq_offset_log;
      const uint64_t end = cumul_after >> freq_offset_log;
      chunk_norm_counts[i] = static_cast<int>(end - start);
    }
  }
}

// nvcompDX duplicate. Sequential on the Dx path; the host kernel uses
// normalize_counts_chunk_parallel.
template <typename T>
static inline __device__ void normalize_counts_chunk_dx(
  const unsigned int idx_in_block,
  const uint32_t *chunk_counts,
  T *chunk_norm_counts,
  uint32_t max_symbol_value,
  uint32_t uncomp_chunk_size,
  uint8_t tablelog
)
{

  uint32_t total = uncomp_chunk_size;

  if (total == 0)
  {
    return;
  }

  if (idx_in_block == 0)
  {

    const int16_t NOT_ASSIGNED = -2;
    uint32_t table_size = 1 << tablelog;
    uint32_t low_prob_threshold = total / table_size;
    uint32_t one_threshold = (3 * total) / (2 * table_size);
    uint32_t remaining = table_size;

    for (uint32_t i = 0; i < NV_SYMBOL_COUNT; ++i)
    {
      int16_t nc = NOT_ASSIGNED;
      if (chunk_counts[i] == 0)
      {
        nc = 0;
      }
      else if (chunk_counts[i] <= low_prob_threshold)
      {
        // Use -1 to designate a character that occurs with freq less than 1 / table_size
        nc = -1;
        remaining -= 1;
        total -= chunk_counts[i];
      }
      else if (chunk_counts[i] <= one_threshold)
      {
        /* a normalized frequency of 1 is assigned to characters whose frequencies are 
             * between 1 / table_size and (3 / 2) * table_size
             */
        nc = 1;
        remaining -= 1;
        total -= chunk_counts[i];
      }
      chunk_norm_counts[i] = nc;
    }

    // Note: With the current tablelog of 10, the following four conditions are never true in testing.
    if (remaining == 0)
    {
      return;
    }
    // if remaining < 2/3rds table_size
    if ((total / remaining) > one_threshold)
    {
      one_threshold = (3 * total) / (2 * remaining);
      for (uint32_t i = 0; i <= max_symbol_value; ++i)
      {
        if ((chunk_norm_counts[i] == NOT_ASSIGNED) && (chunk_counts[i] <= one_threshold))
        {
          chunk_norm_counts[i] = 1;
          remaining -= 1;
          total -= chunk_counts[i];
        }
      }
    }

    /* every symbol was less than the one_threshold, symbol probabilities are close
     * to uniform, and thus the data is probably incompressible */
    if (table_size - remaining == max_symbol_value + 1)
    {
      uint32_t max_symbol = 0;
      uint32_t max_count = 0;
      for (uint32_t i = 0; i <= max_symbol_value; ++i)
      {
        if (chunk_counts[i] > max_count)
        {
          max_symbol = i;
          max_count = chunk_counts[i];
        }
      }
      chunk_norm_counts[max_symbol] += remaining;
      return;
    }

    // all symbols were low probability, distribute the rest of table range
    if (total == 0)
    {
      for (uint32_t i = 0; remaining > 0; i = (i + 1) % (max_symbol_value + 1))
      {
        if (chunk_norm_counts[i] > 0)
        {
          chunk_norm_counts[i]++;
          remaining--;
        }
      }
      return;
    }

    /* distribute higher probability symbols: The idea is to partition the remaining space
     * left in the table evenly among the remaining counts. By adding the value mid to 
     * calculate the step and then right shifting to get actual values, we ensure that
     * the normalized frequencies sum to the exactly remaining space left in the table,
     * as long as total is less than mid. */
    uint64_t freq_offset_log = 62 - tablelog;

    uint64_t mid = (1ULL << (freq_offset_log - 1)) - 1;
    uint64_t step = ((1ULL << freq_offset_log) * remaining + mid) / total;

    uint64_t cumul_total = mid;
    for (uint32_t i = 0; i <= max_symbol_value; ++i)
    {
      if (chunk_norm_counts[i] == NOT_ASSIGNED)
      {
        uint64_t start = cumul_total >> freq_offset_log;
        cumul_total += step * chunk_counts[i];
        uint64_t end = cumul_total >> freq_offset_log;
        uint64_t norm = end - start;
        chunk_norm_counts[i] = norm;

        if (norm == 0)
        {
          // error assigning frequencies, tablelog likely too small
          return;
        }
      }
    }
  }
}

} // namespace ans_gpu_lib::detail
