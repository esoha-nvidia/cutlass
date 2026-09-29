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

#include <cstdint>

#include "gdeflate/gdeflate_constants.h"

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

namespace lzma
{

struct DataAggregator;

}

namespace sa
{

typedef uint8_t U8;
typedef uint16_t U16;
typedef uint32_t U32;
typedef uint64_t U64;

template <typename offset_type>
size_t saGetTempStorageSize(size_t max_chunk_size, size_t batch_size);

template <int saMaxMatchLength, typename LCP_t, typename offset_type>
void saBuild(
  U8 *device_temp,
  const U8 *const *device_data_ptrs,
  const size_t *device_data_sizes,
  size_t max_chunk_size,
  size_t batch_size,
  offset_type **device_sa_out,
  offset_type **device_inv_sa_out,
  LCP_t **device_lcp_out,
  cudaStream_t stream
);

template <unsigned SIZE, typename ParentT>
__device__ unsigned int getLaneMatchLength(const cg::thread_block_tile<SIZE, ParentT> &g, int LCP)
{
  auto lane = g.thread_rank();
  constexpr auto g_hf_size = SIZE / 2;
  for (int offset = 1; offset <= g.size() / 4; offset <<= 1)
  {
    // first half of lanes do shuffle up
    // second half of lanes do shuffle down
    int srcLane = lane < g_hf_size ? lane + offset : lane - offset;
    unsigned int n = g.shfl(LCP, srcLane);
    // using width in __shfl_sync does modulo, but we want current lane value when out of bounds instead
    n = (lane < g_hf_size && srcLane >= g_hf_size) || (lane >= g_hf_size && lane - offset < g_hf_size) ? LCP : n;
    LCP = min(LCP, n);
  }

  return LCP;
}
} // namespace sa