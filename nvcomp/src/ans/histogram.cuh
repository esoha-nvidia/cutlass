/*
* Copyright (c) 2023-2026, NVIDIA CORPORATION. All rights reserved.
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

#include "ans/histogram_common.cuh"
#include "ans/types.cuh"

namespace ans_gpu_lib
{
namespace detail
{

// One warp's contiguous slice of the chunk's symbols (or FP8 pairs), in symbol
// (pair) units. Produced by partition_symbols_across_warps.
struct WarpHistRange
{
  int in_start; // index of this warp's first symbol/pair
  int size; // number of symbols/pairs this warp owns (>= 0)
};

// Partition `num` symbols (or FP8 pairs) evenly across `num_warps`, rounding each
// warp's share up to `align` so the warp-level loads stay aligned. Replaces the
// three copies of this rounding/min/max block (char/fp16/fp8).
inline __device__ WarpHistRange partition_symbols_across_warps(IndexT num, uint32_t wid, uint32_t num_warps, int align)
{
  const int max_per_warp = static_cast<int>(nvcomp::roundUpTo(nvcomp::roundUpDiv(num, num_warps), align));
  const int in_start = max_per_warp * static_cast<int>(wid);
  int size = min(max_per_warp, static_cast<int>(num) - in_start);
  size = max(size, 0);
  return {in_start, size};
}

// Generic CTA-wide Phase-1 histogram. Policy-specific behavior (symbol count, chunk
// prefix, side-band layout, count rows, warp kernel) comes from EncodePolicy.
// Writes the chunk size prefix and returns the ANS bitstream start via
// ans_comp_chunk. All warps accumulate into one CTA-shared count row in
// shared_cta_counts[0..NV_SYMBOL_COUNT). Ends CTA-synchronized.
template <typename EncodePolicy, int BLOCK_DIM_X, typename CG>
__device__ void compute_histogram(
  const void *uncomp_chunk,
  uint8_t *comp_chunk,
  IndexT uncomp_chunk_size_bytes,
  uint8_t *&ans_comp_chunk,
  uint32_t *shared_cta_counts,
  CG &group
)
{
  const int tid = group.thread_rank();
  const int idx_in_warp = tid % WARP_SIZE;
  const int wid = tid / WARP_SIZE;
  constexpr int NUM_WARPS = BLOCK_DIM_X / WARP_SIZE;

  const IndexT num = EncodePolicy::num_symbols(uncomp_chunk_size_bytes);

  // First bytes of comp_chunk carry the size prefix so the decoder knows where the
  // ANS bitstream (and, for fp16/fp8, the side-band) start.
  if (tid == 0)
  {
    ans_comp_chunk = EncodePolicy::histogram_write_chunk_prefix(comp_chunk, uncomp_chunk_size_bytes);
  }

  // Zero the single CTA-shared count row.
  for (int i = tid; i < NV_SYMBOL_COUNT; i += BLOCK_DIM_X)
  {
    shared_cta_counts[i] = 0;
  }
  __syncthreads();

  const WarpHistRange range = partition_symbols_across_warps(num, wid, NUM_WARPS, EncodePolicy::PARTITION_ALIGN);

  const uint8_t *ip = reinterpret_cast<const uint8_t *>(uncomp_chunk) + EncodePolicy::input_byte_offset(range.in_start);
  uint8_t *sideband = EncodePolicy::histogram_sideband_warp_base(comp_chunk, range.in_start);

  // Run this warp's slice through the shared engine; the peel is enabled iff the
  // policy declares it (char only). All warps accumulate into the one shared row.
  histogram_warp_impl<EncodePolicy, EncodePolicy::HAS_PEEL>(idx_in_warp, shared_cta_counts, ip, range.size, sideband);

  if (tid == 0)
  {
    EncodePolicy::write_chunk_tail(comp_chunk, uncomp_chunk, uncomp_chunk_size_bytes);
  }

  group.sync();
}

} // namespace detail
} // namespace ans_gpu_lib
