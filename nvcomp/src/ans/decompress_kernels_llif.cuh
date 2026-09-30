/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
*/

#include <ans/decompress_kernels.cuh>

namespace ans_gpu_lib
{
namespace detail
{

// __launch_bounds__: the block width is NUM_DECOMP_THREADS_PER_CTA
// (= 256, one thread per symbol for the fused table build), and the
// occupancy target is derived from the per-arch target and this mode's decoding
// table footprint via ans_decomp_min_blocks_per_sm.
// Templated on the compile-time decode MODE and STATES_PER_LANE (0 = from the
// bitstream). The host launches the Fp16/Fp8/Fp32 specialization when the data type
// is known from the decompress opts (so that kernel is register-allocated for
// only that decode path), and the Generic specialization otherwise.
template <ans_gpu_lib::detail::DecodeMode MODE, uint8_t STATES_PER_LANE, bool BOUNDS_CHECK, bool ONE_SUBCHUNK_PER_WARP>
__global__
__launch_bounds__(NUM_DECOMP_THREADS_PER_CTA, ans_decomp_min_blocks_per_sm(decode_table_capacity_log<MODE>())) void decompress_kernel(
  const __grid_constant__ void *const *const comp_chunks,
  const __grid_constant__ size_t *const comp_chunk_sizes,
  __grid_constant__ void *const *const uncomp_chunks,
  const __grid_constant__ size_t *const uncomp_chunk_sizes,
  __grid_constant__ size_t *const actual_uncomp_chunk_sizes,
  __grid_constant__ nvcompStatus_t *const block_statuses,
  const __grid_constant__ uint8_t max_sub_chunk_count,
  const __grid_constant__ nvcompType_t expected_data_type,
  const __grid_constant__ uint8_t expected_states_per_lane,
  const __grid_constant__ uint8_t skip_validate
)
{
  // One CTA handles one chunk (grid.x), so every warp shares the same index.
  volatile __shared__ uint32_t chunk_idx_handoff;
  if (threadIdx.x == 0)
  {
    chunk_idx_handoff = blockIdx.x;
  }
  __syncthreads();

  decompress_chunk<MODE, STATES_PER_LANE, BOUNDS_CHECK, ONE_SUBCHUNK_PER_WARP>(
    comp_chunks,
    comp_chunk_sizes,
    uncomp_chunks,
    uncomp_chunk_sizes,
    actual_uncomp_chunk_sizes,
    block_statuses,
    max_sub_chunk_count,
    expected_data_type,
    expected_states_per_lane,
    skip_validate,
    chunk_idx_handoff
  );
}

} // namespace detail
} //namespace ans_gpu_lib
