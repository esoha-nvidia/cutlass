/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <ans/decompress_kernels.cuh>

namespace ans_gpu_lib
{
namespace detail
{

// __launch_bounds__: the block width is the fixed NUM_DECOMP_WARPS_PER_CTA *
// WARP_SIZE (= 128; the fused table build requires it), and the occupancy target
// ANS_DECOMP_MIN_BLOCKS_PER_SM is per-arch tunable via the CONFIG_ANS_* /
// ANS_ARCH_ID table in ans_arch_profile.cuh, selected from the compiled
// __CUDA_ARCH__.
// Templated on the compile-time decode MODE. The host launches the Fp16/Fp8
// specialization when the data type is known from the decompress opts (so that
// kernel is register-allocated for only that decode path), and the Generic
// specialization otherwise (runtime branch on the bitstream, e.g. char/default).
template <ans_gpu_lib::detail::DecodeMode MODE, bool BOUNDS_CHECK>
__global__ __launch_bounds__(NUM_DECOMP_WARPS_PER_CTA *WARP_SIZE, ANS_DECOMP_MIN_BLOCKS_PER_SM) void decompress_kernel(
  const void *const *comp_chunks,
  const size_t *comp_chunk_sizes,
  void *const *uncomp_chunks,
  const size_t *uncomp_chunk_sizes,
  size_t *actual_uncomp_chunk_sizes,
  nvcompStatus_t *block_statuses,
  const size_t batch_size,
  uint8_t max_sub_chunk_count
)
{

  constexpr int BLOCK_DIM_X = NUM_DECOMP_WARPS_PER_CTA * WARP_SIZE;

  auto bid = blockIdx.x;

  decompress_chunk<MODE, BOUNDS_CHECK, BLOCK_DIM_X>(
    comp_chunks[bid],
    comp_chunk_sizes[bid],
    uncomp_chunks,
    uncomp_chunk_sizes[bid],
    actual_uncomp_chunk_sizes ? &actual_uncomp_chunk_sizes[bid] : nullptr,
    block_statuses ? &block_statuses[bid] : nullptr,
    max_sub_chunk_count
  );
}

} // namespace detail
} //namespace ans_gpu_lib