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

#include "ans/EncodePolicy.hpp"
#include "ans/histogram_common.cuh"

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
// per-type copies of this rounding/min/max block (char/fp16/fp8/fp32).
inline __device__ WarpHistRange partition_symbols_across_warps(IndexT num, uint32_t wid, uint32_t num_warps, int align)
{
  const int max_per_warp = static_cast<int>(nvcomp::roundUpTo(nvcomp::roundUpDiv(num, num_warps), align));
  const int in_start = max_per_warp * static_cast<int>(wid);
  int size = min(max_per_warp, static_cast<int>(num) - in_start);
  size = max(size, 0);
  return {in_start, size};
}

// Symbols this warp counts: the whole slice when shift == 0, else its leading 1/(2^shift).
// A nonempty slice always keeps at least one symbol, so a sample never contributes nothing.
inline __device__ int sampled_hist_size(int slice_size, uint32_t shift)
{
  return min(slice_size, max(slice_size >> shift, 1));
}

// Generic CTA-wide Phase-1 histogram. Policy-specific behavior (symbol count,
// mantissa layout, count rows, warp kernel) comes from EncodePolicy.
// All warps accumulate into one CTA-shared count row in shared_cta_counts[0..NV_SYMBOL_COUNT).
// Ends CTA-synchronized.
template <typename EncodePolicy, int BLOCK_DIM_X>
__device__ void compute_histogram(
  const void *uncomp_chunk,
  uint8_t *comp_chunk,
  IndexT uncomp_chunk_size_bytes,
  uint8_t num_sub_chunks,
  uint32_t *shared_cta_counts,
  uint32_t sample_shift, // 0 = exact; else count only this warp's leading range >> shift
  uint8_t *hist_stage, // [NUM_WARPS][stage_bytes_per_warp] cp.async window
  int stage_bytes_per_warp
)
{
  const int tid = static_cast<int>(threadIdx.x);
  const int idx_in_warp = tid % WARP_SIZE;
  const int wid = tid / WARP_SIZE;
  constexpr int NUM_WARPS = BLOCK_DIM_X / WARP_SIZE;

  const IndexT num = EncodePolicy::num_symbols(uncomp_chunk_size_bytes);

  // Zero the single CTA-shared count row.
  for (int i = tid; i < NV_SYMBOL_COUNT; i += BLOCK_DIM_X)
  {
    shared_cta_counts[i] = 0;
  }
  __syncthreads();

  const WarpHistRange range = partition_symbols_across_warps(num, wid, NUM_WARPS, EncodePolicy::PARTITION_ALIGN);

  const uint8_t *ip = reinterpret_cast<const uint8_t *>(uncomp_chunk) + EncodePolicy::input_byte_offset(range.in_start);
  uint8_t *mantissas = nullptr;
  if constexpr (!EncodePolicy::ENCODE_WRITES_MANTISSAS)
  {
    mantissas = EncodePolicy::histogram_mantissas_warp_base(comp_chunk, range.in_start, num_sub_chunks);
  }

  // Run this warp's slice through the shared engine. All warps accumulate into the
  // one shared row.
  //
  // SAMPLING: when sample_shift > 0 the histogram counts only the LEADING 1/(2^shift) of
  // this warp's symbols
  const int hist_size = sampled_hist_size(range.size, sample_shift);
  static_assert(
    EncodePolicy::HIST_FLOOR == HistFloor::None || EncodePolicy::ENCODE_WRITES_MANTISSAS ||
      EncodePolicy::MANTISSA_BYTES_PER_SYMBOL == 0,
    "a sampled histogram walks only the leading part of each warp's slice, so it cannot also be "
    "responsible for writing the mantissas: a sampling policy must either emit them from the encode "
    "pass (ENCODE_WRITES_MANTISSAS) or have none (MANTISSA_BYTES_PER_SYMBOL == 0)"
  );
  histogram_warp_impl<EncodePolicy>(
    idx_in_warp,
    shared_cta_counts,
    ip,
    hist_size,
    mantissas,
    hist_stage + wid * stage_bytes_per_warp
  );

  if (tid == 0)
  {
    EncodePolicy::write_chunk_tail(comp_chunk, uncomp_chunk, uncomp_chunk_size_bytes, num_sub_chunks);
  }

  __syncthreads();
}

} // namespace detail
} // namespace ans_gpu_lib
