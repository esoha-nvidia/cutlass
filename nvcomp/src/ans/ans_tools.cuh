/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include <cassert>
#include <cstdint>

#include "CudaConstants.h"
#include "zstd/utils.cuh"

namespace nvcomp
{

namespace ans_shared
{

// Compute a regular histogram
inline __device__ void frequency_histogram(uint32_t *counts, const uint8_t *src, int src_size)
{
  constexpr int PER_THREAD_HIST_COUNT = 4;
  constexpr int PER_ITER_WARP_COUNT = PER_THREAD_HIST_COUNT * WARP_SIZE;
  while (src_size >= PER_ITER_WARP_COUNT)
  {
    const uint8_t *src_ptr = src + threadIdx.x * PER_THREAD_HIST_COUNT;
    uint8_t symbol1 = src_ptr[0];
    uint8_t symbol2 = src_ptr[1];
    uint8_t symbol3 = src_ptr[2];
    uint8_t symbol4 = src_ptr[3];
    atomicAdd(&counts[symbol1], 1);
    atomicAdd(&counts[symbol2], 1);
    atomicAdd(&counts[symbol3], 1);
    atomicAdd(&counts[symbol4], 1);
    src_size -= PER_ITER_WARP_COUNT;
    src += PER_ITER_WARP_COUNT;
  }

  for (size_t ix = threadIdx.x; ix < src_size; ix += WARP_SIZE_U)
  {
    uint8_t symbol = src[ix];
    atomicAdd(&counts[symbol], 1);
  }
}

// Returns true on success, false on failure
inline __device__ bool normalize_frequencies(
  int16_t *norm_counts,
  uint32_t tablelog,
  const uint32_t *counts,
  size_t total,
  uint32_t max_symbol_value
)
{

  if (threadIdx.x == 0)
  {
    const int16_t NOT_ASSIGNED = -2;
    uint32_t table_size = 1 << tablelog;
    uint32_t low_prob_threshold = total / table_size;
    uint32_t one_threshold = (3 * total) / (2 * table_size);
    uint32_t remaining = table_size;
    for (uint32_t ix_symbol = 0; ix_symbol <= max_symbol_value; ++ix_symbol)
    {
      int16_t nc = NOT_ASSIGNED;
      if (counts[ix_symbol] == 0)
      {
        nc = 0;
      }
      else if (counts[ix_symbol] <= low_prob_threshold)
      {
        // Use -1 to designate a character that occurs with freq less than 1 / table_size
        nc = -1;
        remaining -= 1;

        total -= counts[ix_symbol];
      }
      else if (counts[ix_symbol] <= one_threshold)
      {
        /* a normalized frequency of 1 is assigned to characters whose frequencies are 
                * between 1 / table_size and (3 / 2) * table_size
                */
        nc = 1;
        remaining -= 1;
        total -= counts[ix_symbol];
      }
      norm_counts[ix_symbol] = nc;
    }

    // Note: With the current tablelog used in ZSTD, the following four conditions are never true in testing.
    if (remaining == 0)
    {
      return true;
    }

    // if remaining < 2/3rds table_size
    if ((total / remaining) > one_threshold)
    {
      one_threshold = (3 * total) / (2 * remaining);
      for (uint32_t ix_symbol = 0; ix_symbol < max_symbol_value; ++ix_symbol)
      {
        if ((norm_counts[ix_symbol] == NOT_ASSIGNED) && (counts[ix_symbol] <= one_threshold))
        {
          norm_counts[ix_symbol] = 1;
          remaining -= 1;
          total -= counts[ix_symbol];
        }
      }
    }

    /* every symbol was less than the one_threshold, symbol probabilities are close
        * to uniform, and thus the data is probably incompressible */
    if (table_size - remaining == max_symbol_value + 1)
    {
      uint32_t max_symbol = 0;
      uint32_t max_count = 0;
      for (uint32_t ix_symbol = 0; ix_symbol <= max_symbol_value; ++ix_symbol)
      {
        if (counts[ix_symbol] > max_count)
        {
          max_symbol = ix_symbol;
          max_count = counts[ix_symbol];
        }
      }
      norm_counts[max_symbol] += remaining;
      return true;
    }

    // all symbols were low probability, distribute the rest of table range
    if (total == 0)
    {
      for (uint32_t ix_symbol = 0; remaining > 0; ix_symbol = (ix_symbol + 1) % (max_symbol_value + 1))
      {
        if (norm_counts[ix_symbol] > 0)
        {
          norm_counts[ix_symbol]++;
          remaining--;
        }
      }
      return true;
    }

    /* distribute higher probability symbols: The idea is to partition the remaining space
        * left in the table evenly among the remaining counts. By adding the value mid to 
        * calculate the step and then right shifting to get actual values, we ensure that
        * the normalized frequencies sum to the exactly remaining space left in the table,
        * as long as total is less than mid. */

    uint64_t freq_offset_log = 62 - tablelog;

    uint64_t mid = (1ULL << (freq_offset_log - 1)) - 1;
    assert(total < mid);
    uint64_t step = ((1ULL << freq_offset_log) * remaining + mid) / total;

    uint64_t cumul_total = mid;
    for (uint32_t ix_symbol = 0; ix_symbol <= max_symbol_value; ++ix_symbol)
    {
      if (norm_counts[ix_symbol] == NOT_ASSIGNED)
      {
        uint64_t start = cumul_total >> freq_offset_log;
        cumul_total += step * counts[ix_symbol];
        uint64_t end = cumul_total >> freq_offset_log;
        uint64_t norm = end - start;
        norm_counts[ix_symbol] = norm;
        remaining -= norm;

        if (norm == 0)
        {
          // error assigning frequencies, tablelog likely too small
          return false;
        }
      }
    }
    assert(remaining == 0);
  } // end threadIdx.x == 0
  return true;
}

} // namespace ans_shared
} // namespace nvcomp
