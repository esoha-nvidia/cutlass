/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <cassert>
#include <stdexcept>
#include <type_traits>

#include "ans.h"
#include "ans/ans_utils.cuh"
#include "ans/decompress_kernels_llif.cuh"
#include "common.h"
#include "device_guard.h"
#include "exception.hpp"
#include "Logging.h"
#include "nvcomp/shared_types.h"
#include "nvcomp_common_deps/hlif_shared_types.hpp"

using namespace ans_gpu_lib;
using namespace ans_gpu_lib::detail;
using nvcomp::cuda_dim_cast;
using nvcomp::DeviceGuard;
using nvcomp::narrow_cast;
using nvcomp::roundUpDiv;

namespace ans
{

void decompressGetTempSize(size_t num_chunks, size_t /* max_uncompressed_chunk_size */, size_t *temp_bytes)
{
  if (temp_bytes == nullptr)
  {
    throw nvcomp::NVCompException(nvcompErrorInvalidValue, "temp_bytes must not be null");
  }

  // Fused decompress: each CTA builds its chunk's decoding table directly into
  // shared memory, so no global decoding-table workspace is required.
  *temp_bytes = 0;
}

void decompressAsync(
  const void *const *comp_chunks,
  const size_t *comp_chunk_sizes,
  const size_t *uncomp_chunk_sizes,
  size_t *device_actual_uncomp_chunk_sizes,
  size_t batch_size,
  void *const /*temp_ptr*/,
  const size_t /*temp_bytes*/,
  void *const *uncomp_chunks,
  nvcompStatus_t *device_statuses,
  uint8_t max_sub_chunk_count,
  nvcompType_t data_type,
  cudaStream_t stream
)
{
  // Fused: the decoding table is built per-CTA inside decompress_kernel.

  // The launch grid is driven directly by the (max) sub chunk count. When 0
  // (auto), use the worst-case grid that covers the maximum possible sub chunks.
  // Warps per CTA is fixed (NUM_DECOMP_WARPS_PER_CTA), not arch dependent.
  int num_ctas_per_chunk;
  if (max_sub_chunk_count > 0)
  {
    num_ctas_per_chunk = nvcomp::roundUpDiv(max_sub_chunk_count, NUM_DECOMP_WARPS_PER_CTA);
    num_ctas_per_chunk = min(num_ctas_per_chunk, MAX_SUB_CHUNKS_PER_CHUNK / NUM_DECOMP_WARPS_PER_CTA);
    num_ctas_per_chunk = max(num_ctas_per_chunk, 1);
  }
  else
  {
    num_ctas_per_chunk = nvcomp::roundUpDiv(MAX_SUB_CHUNKS_PER_CHUNK, NUM_DECOMP_WARPS_PER_CTA);
  }

  dim3 grid(cuda_dim_cast(batch_size), num_ctas_per_chunk);
  dim3 block(NUM_DECOMP_WARPS_PER_CTA * WARP_SIZE);

  // Bounds checking active when debugging.
#ifndef NDEBUG
  constexpr bool BOUNDS_CHECK = true;
#else
  constexpr bool BOUNDS_CHECK = false;
#endif

  using ans_gpu_lib::detail::DecodeMode;

  // Pick the kernel by data type: fp16/fp8 launch a type-specialized kernel
  // (register-allocated for only that decode path); everything else (char/default,
  // or unknown type) launches the Generic kernel that branches on the bitstream.
  // The bitstream is still self-describing, so Generic decodes any type correctly.
  decltype(&ans_gpu_lib::detail::decompress_kernel<DecodeMode::Generic, BOUNDS_CHECK>) kernel = nullptr;

  switch (data_type)
  {
    case NVCOMP_TYPE_FLOAT16:
      kernel = ans_gpu_lib::detail::decompress_kernel<DecodeMode::Fp16, BOUNDS_CHECK>;
      break;
    case NVCOMP_TYPE_FLOAT8_E4M3:
      kernel = ans_gpu_lib::detail::decompress_kernel<DecodeMode::Fp8, BOUNDS_CHECK>;
      break;
    default:
      kernel = ans_gpu_lib::detail::decompress_kernel<DecodeMode::Generic, BOUNDS_CHECK>;
      break;
  }

  kernel<<<grid, block, 0, stream>>>(
    (const void *const *)comp_chunks,
    comp_chunk_sizes,
    (void *const *)uncomp_chunks,
    uncomp_chunk_sizes,
    device_actual_uncomp_chunk_sizes,
    device_statuses,
    batch_size,
    max_sub_chunk_count
  );
  CUDA_CHECK(cudaGetLastError());
}

void getDecompressSizeAsync(
  const void *const *device_compressed_ptrs,
  const size_t *device_compressed_bytes,
  size_t *device_uncompressed_bytes,
  size_t batch_size,
  cudaStream_t stream
)
{
  // 1 thread handles 1 chunk
  const dim3 block(256);
  const dim3 grid(cuda_dim_cast(roundUpDiv(batch_size, block.x)));

  ans_gpu_lib::detail::decompress_get_sizes_kernel<<<grid, block, 0, stream>>>(
    (const void *const *)device_compressed_ptrs,
    device_compressed_bytes,
    device_uncompressed_bytes,
    batch_size
  );
  CUDA_CHECK(cudaGetLastError());
}

} // namespace ans
