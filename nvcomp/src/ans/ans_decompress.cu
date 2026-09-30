/*
 * Copyright (c) 2021-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <cassert>
#include <string>
#include <type_traits>

#include "ans.h"
#include "ans/ans_utils.cuh"
#include "ans/decompress_kernels_llif.cuh"
#include "common.h"
#include "device_guard.h"
#include "exception.hpp"
#include "Logging.h"
#include "nvcomp.hpp"
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
  uint8_t states_per_lane,
  uint8_t skip_validate,
  cudaStream_t stream
)
{
  // Fused: the decoding table is built per-CTA inside decompress_kernel.

  const uint8_t launch_sub_chunk_count = resolve_decomp_launch_sub_chunk_count(max_sub_chunk_count);
  int num_ctas_per_chunk = nvcomp::roundUpDiv(launch_sub_chunk_count, NUM_DECOMP_WARPS_PER_CTA);

  dim3 grid(cuda_dim_cast(batch_size), num_ctas_per_chunk);
  dim3 block(NUM_DECOMP_THREADS_PER_CTA);

  // Bounds checking active when debugging.
#ifndef NDEBUG
  constexpr bool BOUNDS_CHECK = true;
#else
  constexpr bool BOUNDS_CHECK = false;
#endif

  if constexpr (BOUNDS_CHECK)
  {
    nvcomp::try_clear_device_statuses(batch_size, device_statuses, stream);
  }

  using ans_gpu_lib::detail::DecodeMode;

  decltype(&ans_gpu_lib::detail::decompress_kernel<DecodeMode::Generic, 0, BOUNDS_CHECK, false>) kernel = nullptr;

  auto select_kernel = [&](auto one_subchunk_per_warp) {
    constexpr bool ONE_SUBCHUNK_PER_WARP = decltype(one_subchunk_per_warp)::value;

    // states_per_lane 0: Generic, state count read from the bitstream. 1 / 2: the type's own
    // decode mode with that count pinned.
    // MSVC fails to treat ONE_SUBCHUNK_PER_WARP as a constant expression when it is captured
    // through this doubly-nested lambda rather than passed in as its own template parameter.
    const auto set_typed_kernel = [&](auto mode, auto one_subchunk_per_warp_tag) {
      constexpr DecodeMode MODE = decltype(mode)::value;
      constexpr bool ONE_SUBCHUNK_PER_WARP = decltype(one_subchunk_per_warp_tag)::value;
      if (states_per_lane == 1)
      {
        kernel = &ans_gpu_lib::detail::decompress_kernel<MODE, 1, BOUNDS_CHECK, ONE_SUBCHUNK_PER_WARP>;
      }
      else if (states_per_lane == 2)
      {
        kernel = &ans_gpu_lib::detail::decompress_kernel<MODE, 2, BOUNDS_CHECK, ONE_SUBCHUNK_PER_WARP>;
      }
      else
      {
        kernel = &ans_gpu_lib::detail::decompress_kernel<DecodeMode::Generic, 0, BOUNDS_CHECK, ONE_SUBCHUNK_PER_WARP>;
      }
    };

    switch (data_type)
    {
      case NVCOMP_TYPE_BITS:
        // Unknown type: only the bitstream knows, so Generic regardless of states_per_lane.
        kernel = &ans_gpu_lib::detail::decompress_kernel<DecodeMode::Generic, 0, BOUNDS_CHECK, ONE_SUBCHUNK_PER_WARP>;
        break;
      case NVCOMP_TYPE_FLOAT16:
        set_typed_kernel(std::integral_constant<DecodeMode, DecodeMode::Fp16>{}, one_subchunk_per_warp);
        break;
      case NVCOMP_TYPE_FLOAT8_E4M3:
        set_typed_kernel(std::integral_constant<DecodeMode, DecodeMode::Fp8>{}, one_subchunk_per_warp);
        break;
      case NVCOMP_TYPE_FLOAT32:
        set_typed_kernel(std::integral_constant<DecodeMode, DecodeMode::Fp32>{}, one_subchunk_per_warp);
        break;
      case NVCOMP_TYPE_CHAR:
      case NVCOMP_TYPE_UCHAR:
        set_typed_kernel(std::integral_constant<DecodeMode, DecodeMode::Char>{}, one_subchunk_per_warp);
        break;
      default:
        throw nvcomp::NVCompException(
          nvcompErrorNotSupported,
          "Unsupported ANS decompression data_type: " + std::to_string(static_cast<int>(data_type))
        );
    }
  };

  if (decomp_one_subchunk_per_warp(max_sub_chunk_count))
  {
    select_kernel(std::true_type{});
  }
  else
  {
    select_kernel(std::false_type{});
  }

#ifndef NDEBUG
  ans_assert_smem_within_estimate(kernel, [](int arch_id) { return ans_decompress_smem_bytes(arch_id, MAX_TABLELOG); });
#endif

  kernel<<<grid, block, 0, stream>>>(
    (const void *const *)comp_chunks,
    comp_chunk_sizes,
    (void *const *)uncomp_chunks,
    uncomp_chunk_sizes,
    device_actual_uncomp_chunk_sizes,
    device_statuses,
    max_sub_chunk_count,
    data_type,
    states_per_lane,
    skip_validate
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
