/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or its
 * affiliates is strictly prohibited.
 */

#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "cascaded/next/primitives/kibi_cascader/kibi_cascader_constants.cuh"
#include "cascaded/next/terminal_codecs/terminal_cascaded/terminal_cascaded_kernels.cuh"
#include "cascaded/next/terminal_codecs/terminal_cascaded/terminal_cascaded_utils.cuh"
#include "cascaded/universal/universal_header.cuh"
#include "nvcomp.h"

namespace nvcomp::cascaded::next::terminal_cascaded
{

namespace detail
{

template <typename Launcher>
inline nvcompStatus_t dispatch_search_space(const uint32_t search_space, const Launcher &launch)
{
  using kibi_cascader::SEARCH_DELTA;
  using kibi_cascader::SEARCH_FOR;
  using kibi_cascader::SEARCH_NONE;
  using kibi_cascader::SEARCH_RLE;

  switch (search_space)
  {
    case SEARCH_NONE:
      return launch(std::integral_constant<uint32_t, SEARCH_NONE>{});
    case SEARCH_DELTA:
      return launch(std::integral_constant<uint32_t, SEARCH_DELTA>{});
    case SEARCH_FOR:
      return launch(std::integral_constant<uint32_t, SEARCH_FOR>{});
    case SEARCH_DELTA | SEARCH_FOR:
      return launch(std::integral_constant<uint32_t, SEARCH_DELTA | SEARCH_FOR>{});
    case SEARCH_RLE:
      return launch(std::integral_constant<uint32_t, SEARCH_RLE>{});
    case SEARCH_DELTA | SEARCH_RLE:
      return launch(std::integral_constant<uint32_t, SEARCH_DELTA | SEARCH_RLE>{});
    case SEARCH_FOR | SEARCH_RLE:
      return launch(std::integral_constant<uint32_t, SEARCH_FOR | SEARCH_RLE>{});
    case TERMINAL_CASCADED_SEARCH_ALL:
      return launch(std::integral_constant<uint32_t, TERMINAL_CASCADED_SEARCH_ALL>{});
    default:
      return nvcompErrorNotSupported;
  }
}

template <typename T, kibi_cascader::kibi_cascader_optimization_search_space SearchSpace>
inline nvcompStatus_t launch_compress(
  const void *const *device_uncompressed_chunk_ptrs,
  const size_t *device_uncompressed_chunk_bytes,
  const size_t num_chunks,
  void *const *device_compressed_chunk_ptrs,
  size_t *const device_compressed_chunk_bytes,
  const nvcompType_t data_type,
  nvcompStatus_t *const device_statuses,
  const cudaStream_t stream
)
{
  compress_kernel<T, SearchSpace><<<static_cast<unsigned int>(num_chunks), THREADS_PER_CTA, 0, stream>>>(
    device_uncompressed_chunk_ptrs,
    device_uncompressed_chunk_bytes,
    device_compressed_chunk_ptrs,
    device_compressed_chunk_bytes,
    data_type,
    device_statuses
  );
  return cudaPeekAtLastError() == cudaSuccess ? nvcompSuccess : nvcompErrorCudaError;
}

template <typename T>
inline nvcompStatus_t compress_dispatch(
  const void *const *device_uncompressed_chunk_ptrs,
  const size_t *device_uncompressed_chunk_bytes,
  const size_t num_chunks,
  void *const *device_compressed_chunk_ptrs,
  size_t *const device_compressed_chunk_bytes,
  const nvcompType_t data_type,
  const uint32_t search_space,
  nvcompStatus_t *const device_statuses,
  const cudaStream_t stream
)
{
  return dispatch_search_space(search_space, [&](const auto search_space_constant) {
    return launch_compress<T, decltype(search_space_constant)::value>(
      device_uncompressed_chunk_ptrs,
      device_uncompressed_chunk_bytes,
      num_chunks,
      device_compressed_chunk_ptrs,
      device_compressed_chunk_bytes,
      data_type,
      device_statuses,
      stream
    );
  });
}

template <typename T, kibi_cascader::kibi_cascader_optimization_search_space SearchSpace>
inline nvcompStatus_t launch_decompress(
  const void *const *device_compressed_chunk_ptrs,
  const size_t *device_compressed_chunk_bytes,
  const size_t *device_uncompressed_buffer_bytes,
  size_t *const device_uncompressed_chunk_bytes,
  const size_t num_chunks,
  void *const *device_uncompressed_chunk_ptrs,
  const nvcompType_t data_type,
  nvcompStatus_t *const device_statuses,
  const cudaStream_t stream
)
{
  decompress_kernel<T, SearchSpace><<<static_cast<unsigned int>(num_chunks), THREADS_PER_CTA, 0, stream>>>(
    device_compressed_chunk_ptrs,
    device_compressed_chunk_bytes,
    device_uncompressed_buffer_bytes,
    device_uncompressed_chunk_bytes,
    device_uncompressed_chunk_ptrs,
    data_type,
    device_statuses
  );
  return cudaPeekAtLastError() == cudaSuccess ? nvcompSuccess : nvcompErrorCudaError;
}

template <typename T>
inline nvcompStatus_t decompress_dispatch(
  const void *const *device_compressed_chunk_ptrs,
  const size_t *device_compressed_chunk_bytes,
  const size_t *device_uncompressed_buffer_bytes,
  size_t *const device_uncompressed_chunk_bytes,
  const size_t num_chunks,
  void *const *device_uncompressed_chunk_ptrs,
  const nvcompType_t data_type,
  const uint32_t search_space,
  nvcompStatus_t *const device_statuses,
  const cudaStream_t stream
)
{
  return dispatch_search_space(search_space, [&](const auto search_space_constant) {
    return launch_decompress<T, decltype(search_space_constant)::value>(
      device_compressed_chunk_ptrs,
      device_compressed_chunk_bytes,
      device_uncompressed_buffer_bytes,
      device_uncompressed_chunk_bytes,
      num_chunks,
      device_uncompressed_chunk_ptrs,
      data_type,
      device_statuses,
      stream
    );
  });
}

} // namespace detail

inline nvcompStatus_t get_compression_temp_size(
  [[maybe_unused]] const size_t num_chunks,
  [[maybe_unused]] const size_t max_uncompressed_chunk_bytes,
  [[maybe_unused]] const nvcompType_t data_type,
  size_t *const temp_bytes
)
{
  if (temp_bytes == nullptr)
  {
    return nvcompErrorInvalidValue;
  }
  *temp_bytes = 0;
  return nvcompSuccess;
}

inline nvcompStatus_t get_max_output_chunk_size(
  const size_t max_uncompressed_chunk_bytes,
  const nvcompType_t data_type,
  size_t *const max_compressed_chunk_bytes
)
{
  if (max_compressed_chunk_bytes == nullptr || max_uncompressed_chunk_bytes == 0u ||
      max_uncompressed_chunk_bytes > nvcompCascadedCompressionMaxAllowedChunkSize)
  {
    return nvcompErrorInvalidValue;
  }
  if (data_type != NVCOMP_TYPE_INT && data_type != NVCOMP_TYPE_UINT && data_type != NVCOMP_TYPE_LONGLONG &&
      data_type != NVCOMP_TYPE_ULONGLONG)
  {
    return nvcompErrorNotSupported;
  }

  *max_compressed_chunk_bytes =
    terminal_cascaded::max_compressed_chunk_bytes(static_cast<uint32_t>(max_uncompressed_chunk_bytes));
  return nvcompSuccess;
}

inline nvcompStatus_t compress_async(
  const void *const *device_uncompressed_chunk_ptrs,
  const size_t *device_uncompressed_chunk_bytes,
  [[maybe_unused]] const size_t max_uncompressed_chunk_bytes,
  const size_t num_chunks,
  [[maybe_unused]] void *const device_temp_ptr,
  [[maybe_unused]] const size_t temp_bytes,
  void *const *device_compressed_chunk_ptrs,
  size_t *const device_compressed_chunk_bytes,
  const nvcompType_t data_type,
  const kibi_cascader::kibi_cascader_optimization_search_space search_space,
  nvcompStatus_t *const device_statuses,
  const cudaStream_t stream
)
{
  switch (data_type)
  {
    case NVCOMP_TYPE_INT:
    case NVCOMP_TYPE_UINT:
      return detail::compress_dispatch<uint32_t>(
        device_uncompressed_chunk_ptrs,
        device_uncompressed_chunk_bytes,
        num_chunks,
        device_compressed_chunk_ptrs,
        device_compressed_chunk_bytes,
        data_type,
        search_space,
        device_statuses,
        stream
      );
    case NVCOMP_TYPE_LONGLONG:
    case NVCOMP_TYPE_ULONGLONG:
      return detail::compress_dispatch<uint64_t>(
        device_uncompressed_chunk_ptrs,
        device_uncompressed_chunk_bytes,
        num_chunks,
        device_compressed_chunk_ptrs,
        device_compressed_chunk_bytes,
        data_type,
        search_space,
        device_statuses,
        stream
      );
    default:
      return nvcompErrorNotSupported;
  }
}

inline nvcompStatus_t get_decompression_temp_size(
  [[maybe_unused]] const size_t num_chunks,
  [[maybe_unused]] const size_t max_compressed_chunk_bytes,
  [[maybe_unused]] const nvcompType_t data_type,
  size_t *const temp_bytes
)
{
  if (temp_bytes == nullptr)
  {
    return nvcompErrorInvalidValue;
  }
  *temp_bytes = 0;
  return nvcompSuccess;
}

inline nvcompStatus_t decompress_async(
  const void *const *device_compressed_chunk_ptrs,
  const size_t *device_compressed_chunk_bytes,
  const size_t *device_uncompressed_buffer_bytes,
  size_t *const device_uncompressed_chunk_bytes,
  const size_t num_chunks,
  [[maybe_unused]] void *const device_temp_ptr,
  [[maybe_unused]] const size_t temp_bytes,
  void *const *device_uncompressed_chunk_ptrs,
  const nvcompType_t data_type,
  const kibi_cascader::kibi_cascader_optimization_search_space search_space,
  nvcompStatus_t *const device_statuses,
  const cudaStream_t stream
)
{
  switch (data_type)
  {
    case NVCOMP_TYPE_INT:
    case NVCOMP_TYPE_UINT:
      return detail::decompress_dispatch<uint32_t>(
        device_compressed_chunk_ptrs,
        device_compressed_chunk_bytes,
        device_uncompressed_buffer_bytes,
        device_uncompressed_chunk_bytes,
        num_chunks,
        device_uncompressed_chunk_ptrs,
        data_type,
        search_space,
        device_statuses,
        stream
      );
    case NVCOMP_TYPE_LONGLONG:
    case NVCOMP_TYPE_ULONGLONG:
      return detail::decompress_dispatch<uint64_t>(
        device_compressed_chunk_ptrs,
        device_compressed_chunk_bytes,
        device_uncompressed_buffer_bytes,
        device_uncompressed_chunk_bytes,
        num_chunks,
        device_uncompressed_chunk_ptrs,
        data_type,
        search_space,
        device_statuses,
        stream
      );
    default:
      return nvcompErrorNotSupported;
  }
}

template <bool ERR_ON_MODE_MISMATCH = true>
inline nvcompStatus_t decompress_self_dispatch_async(
  const void *const *device_compressed_chunk_ptrs,
  const size_t *device_compressed_chunk_bytes,
  const size_t *device_uncompressed_buffer_bytes,
  size_t *const device_uncompressed_chunk_bytes,
  const size_t num_chunks,
  void *const *device_uncompressed_chunk_ptrs,
  const nvcompType_t expected_data_type,
  nvcompStatus_t *const device_statuses,
  const cudaStream_t stream
)
{
  const auto launch_uint32 = [&]() {
    self_dispatch_decompress_kernel<uint32_t, ERR_ON_MODE_MISMATCH>
      <<<static_cast<unsigned int>(num_chunks), THREADS_PER_CTA, 0, stream>>>(
        device_compressed_chunk_ptrs,
        device_compressed_chunk_bytes,
        device_uncompressed_buffer_bytes,
        device_uncompressed_chunk_bytes,
        device_uncompressed_chunk_ptrs,
        expected_data_type,
        device_statuses
      );
    return cudaPeekAtLastError() == cudaSuccess ? nvcompSuccess : nvcompErrorCudaError;
  };
  const auto launch_uint64 = [&]() {
    self_dispatch_decompress_kernel<uint64_t, ERR_ON_MODE_MISMATCH>
      <<<static_cast<unsigned int>(num_chunks), THREADS_PER_CTA, 0, stream>>>(
        device_compressed_chunk_ptrs,
        device_compressed_chunk_bytes,
        device_uncompressed_buffer_bytes,
        device_uncompressed_chunk_bytes,
        device_uncompressed_chunk_ptrs,
        expected_data_type,
        device_statuses
      );
    return cudaPeekAtLastError() == cudaSuccess ? nvcompSuccess : nvcompErrorCudaError;
  };

  switch (expected_data_type)
  {
    case NVCOMP_TYPE_BITS:
      if (const nvcompStatus_t status = launch_uint32(); status != nvcompSuccess)
      {
        return status;
      }
      return launch_uint64();
    case NVCOMP_TYPE_INT:
    case NVCOMP_TYPE_UINT:
      return launch_uint32();
    case NVCOMP_TYPE_LONGLONG:
    case NVCOMP_TYPE_ULONGLONG:
      return launch_uint64();
    default:
      return nvcompErrorNotSupported;
  }
}

} // namespace nvcomp::cascaded::next::terminal_cascaded
