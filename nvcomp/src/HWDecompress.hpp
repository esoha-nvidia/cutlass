/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
*/

#pragma once

#include <algorithm>
#include <chrono>
#include <ciso646>
#include <cstring>
#include <numeric>
#include <utility>
#include <vector>

#ifdef USE_NVTX
#include <nvtx3/nvtx3.hpp>
#endif

#include "allocators/host_pinned.hpp"
#include "common.h"
#include "CudaDriver.h"
#include "CudaUtils.h"
#include "device_guard.h"
#include "exception.hpp"
#include "gdeflate/gzip_util.cuh"
#include "gzip/GzipConstants.cuh"
#include "HWDecompressMacros.h"
#include "Logging.h"
#include "nvcomp.hpp"
#include "snappy/util.cuh"
// Debug macro, debugging issues related to DE
// #define CHECK_DE_COMPAT

namespace nvcomp
{

bool hw_supports_decomp_engine(nvcompFormatType_t algorithm, int device_id);

/**
 * We sort inputs to the decompress engine to help with load balancing of
 * different aspects of the DE. This function is a shared API to provide
 * the amount of scratch memory required for this sort
*/
size_t get_sort_scratch_req(size_t num_chunks, size_t max_uncompressed_chunk_bytes);

size_t get_decompress_engine_max_size(int device_id);

/**
 * @brief Check if sorting should be used for hardware decompression
 * @param decompress_opts Decompression options
 * @return true if sorting should be used
 */
template <typename DecompressOptsType>
bool should_use_sorting(const DecompressOptsType &decompress_opts)
{
  return decompress_opts.sort_before_hw_decompress && (decompress_opts.backend == NVCOMP_DECOMPRESS_BACKEND_HARDWARE ||
                                                       decompress_opts.backend == NVCOMP_DECOMPRESS_BACKEND_DEFAULT);
}

inline void FillCuDEParamsOnHost(
  const void *const *device_in_ptr,
  const size_t *device_in_bytes,
  void *const *device_out_ptr,
  CUmemDecompressParams *de_params,
  size_t *device_out_bytes,
  const size_t *device_buffer_bytes,
  size_t num_chunks,
  nvcompStatus_t *pinned_statuses,
  bool do_parse_header,
  bool use_sorting,
  CUmemDecompressAlgorithm algo,
  const void *const *host_comp_chunk_buffers,
  cudaStream_t stream,
  bool force_sync
)
{
  CUDA_CHECK(cudaLaunchHostLambda(
    stream,
    [=]() {
      std::vector<std::pair<size_t, size_t>> key_index;
      if (use_sorting)
      {
        key_index.resize(num_chunks);
        for (size_t i = 0; i < num_chunks; ++i)
        {
          key_index[i] = {device_buffer_bytes[i], i};
        }
        std::sort(
          key_index.begin(),
          key_index.end(),
          [](const std::pair<size_t, size_t> &a, const std::pair<size_t, size_t> &b) { return a.first > b.first; }
        );
      }
      // Key index is sorted in descending order of buffer size. key_index[j].second is the original index of the chunk.
      for (size_t j = 0; j < num_chunks; ++j)
      {
        const size_t ix_chunk = use_sorting ? key_index[j].second : j;
        int header_size = 0;
        // Bytes trailing the payload that the DE must not read. A gzip member ends with an 8-byte
        // CRC32 + ISIZE trailer; other framings handled here (Snappy) have none.
        size_t trailer_size = 0;
        if (do_parse_header)
        {
          if (host_comp_chunk_buffers == nullptr)
          {
            throw NVCompException(nvcompErrorInvalidValue, std::string("Host compressed chunk buffers are nullptr"));
          }
          if (algo == CU_MEM_DECOMPRESS_ALGORITHM_DEFLATE)
          {
            header_size = deflate::parse_gzip_header(
              static_cast<const uint8_t *>(host_comp_chunk_buffers[ix_chunk]),
              device_in_bytes[ix_chunk]
            );
            trailer_size = GZIP_FOOTER_BYTES;
          }
          else if (algo == CU_MEM_DECOMPRESS_ALGORITHM_SNAPPY)
          {
            uint32_t snappy_header_size;
            int32_t error;
            snappy::get_uncompressed_size(
              static_cast<const uint8_t *>(host_comp_chunk_buffers[ix_chunk]),
              device_in_bytes[ix_chunk],
              snappy_header_size,
              error
            );
            header_size = error ? -1 : static_cast<int>(snappy_header_size);
          }
          if (header_size == -1)
          {
            throw NVCompException(
              nvcompErrorCannotDecompress,
              "Failed to parse header for chunk " + std::to_string(ix_chunk)
            );
          }
        }
        de_params[ix_chunk].src =
          reinterpret_cast<const void *>(static_cast<const char *>(device_in_ptr[ix_chunk]) + header_size);
        de_params[ix_chunk].dst = device_out_ptr[ix_chunk];
        de_params[ix_chunk].srcNumBytes = device_in_bytes[ix_chunk] - header_size - trailer_size;
        de_params[ix_chunk].dstNumBytes = 0;
        de_params[ix_chunk].algo = algo;
        // We won't fill device_out_bytes here, because it is not host accessible when called through the HLIF,
        // and we can only get here through the HLIF.
        // TODO: Restore this when we bump up RMM version
        // device_out_bytes[ix_chunk] = 0;  // WAR for HW only setting 32 lsb for this value
        de_params[ix_chunk].dstActBytes = reinterpret_cast<cuuint32_t *>(&device_out_bytes[ix_chunk]);
        if (pinned_statuses != nullptr)
        {
          pinned_statuses[ix_chunk] = nvcompSuccess;
        }
      }
    },
    force_sync
  ));
}

template <bool do_fill_params = true, typename DecompParamsFn_t>
bool do_hw_decomp(
  void *const *device_uncompressed_chunk_ptrs,
  size_t *device_uncompressed_chunk_bytes,
  size_t num_chunks,
  const void *const *device_compressed_chunk_ptrs,
  const size_t *device_compressed_chunk_bytes,
  const volatile size_t *host_buffer_bytes,
  nvcompStatus_t *device_statuses,
  int device_id,
  cudaStream_t stream,
  DecompParamsFn_t &&params_fn,
  CUmemDecompressParams *params,
  const bool force_hw_decomp
)
{
  {
#ifdef USE_NVTX
    nvtx3::scoped_range nvtx_fill_params{"fill DE params"};
#endif
    if constexpr (do_fill_params)
    {
      params_fn(
        device_compressed_chunk_ptrs,
        device_compressed_chunk_bytes,
        device_uncompressed_chunk_ptrs,
        params,
        device_uncompressed_chunk_bytes,
        num_chunks,
        device_statuses,
        stream
      );
    }
  }

  // Have to synchronize before calling the driver API
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Zero the padding on host. Merging this into the params kernel
  // degrades performance, as it unnecessarily occupies the interconnect's bandwidth.
  for (size_t ix_chunk = 0; ix_chunk < num_chunks; ++ix_chunk)
  {
    memset(params[ix_chunk].padding, 0, sizeof(params[ix_chunk].padding));
  }
  if (not force_hw_decomp)
  {
    // After the stream sync, host_buffer_bytes is filled. Check whether it could be too large.
    size_t max_chunk_buffer_size = *std::max_element(host_buffer_bytes, host_buffer_bytes + num_chunks);

    size_t max_supported_size = get_decompress_engine_max_size(device_id);
    if (max_supported_size < max_chunk_buffer_size)
    {
      // If it's too large, the HW API would fail. Short circuit and use the SM.
      LOG_INFO(
        "Cannot use HW decompression because the max chunk buffer size ({}) bytes is greater than max supported size "
        "({}) bytes",
        max_chunk_buffer_size,
        max_supported_size
      );
      return false;
    }
  }

#ifdef CHECK_DE_COMPAT
  for (size_t ix_chunk = 0; ix_chunk < num_chunks; ++ix_chunk)
  {
    bool srcPtrIsHwDecompressCapable, dstPtrIsHwDecompressCapable, dstBytesPtrIsHwDecompressCapable;
    CudaDriver::cuPointerGetAttribute(
      &srcPtrIsHwDecompressCapable,
      CU_POINTER_ATTRIBUTE_IS_HW_DECOMPRESS_CAPABLE,
      (CUdeviceptr)params[ix_chunk].src
    );
    CudaDriver::cuPointerGetAttribute(
      &dstPtrIsHwDecompressCapable,
      CU_POINTER_ATTRIBUTE_IS_HW_DECOMPRESS_CAPABLE,
      (CUdeviceptr)params[ix_chunk].dst
    );
    CudaDriver::cuPointerGetAttribute(
      &dstBytesPtrIsHwDecompressCapable,
      CU_POINTER_ATTRIBUTE_IS_HW_DECOMPRESS_CAPABLE,
      (CUdeviceptr)params[ix_chunk].dstActBytes
    );
    uint8_t padding[sizeof(params[ix_chunk].padding)] = {0};
    bool zerod = memcmp(params[ix_chunk].padding, padding, sizeof(padding)) != 0;
    printf(
      "ix chunk %lu, srcNumBytes %lu zerod %d ptrs capable %d %d %d algo %d dstNumBytes %lu\n",
      ix_chunk,
      params[ix_chunk].srcNumBytes,
      zerod,
      srcPtrIsHwDecompressCapable,
      dstPtrIsHwDecompressCapable,
      dstBytesPtrIsHwDecompressCapable,
      params[ix_chunk].algo,
      params[ix_chunk].dstNumBytes
    );
  }
#endif // CHECK_DE_COMPAT

  size_t error_index = 0;
  auto err = CUDA_SUCCESS;
  {
#ifdef USE_NVTX
    nvtx3::scoped_range nvtx_de_api{"DE CUDA Driver API"};
#endif
    err = CudaDriver::cuMemBatchDecompressAsync(
      params,
      num_chunks,
      0 /* flags, according to cuda.h, must be 0 */,
      &error_index,
      stream
    );
  }

  if (CUDA_SUCCESS != err)
  {
    if (error_index == SIZE_MAX)
    {
      LOG_ERROR("HW DE error index = {}", SIZE_MAX);
      throw NVCompException(
        nvcompErrorCannotDecompress,
        std::string("DE failed to decompress. err code ") + std::to_string(err)
      );
    }

    // Otherwise DE failed, return false so we fallback to SM
    LOG_ERROR("HW DE failed with error code {}", std::to_string(err));
    return false;
  }

  LOG_INFO("HW decompression API launch successful");
  return true;
}

template <bool do_fill_params = true, typename DecompParamsFn_t>
bool do_hw_decomp(
  void *const *device_uncompressed_chunk_ptrs,
  size_t *device_uncompressed_chunk_bytes,
  size_t num_chunks,
  const void *const *device_compressed_chunk_ptrs,
  const size_t *device_compressed_chunk_bytes,
  const size_t *device_uncompressed_buffer_bytes,
  nvcompStatus_t *device_statuses,
  cudaStream_t stream,
  DecompParamsFn_t &&params_fn,
  nvcompFormatType_t comp_format,
  const bool force_hw_decomp
)
{
  DeviceGuard device_guard(stream);
  int device_id;
  CUDA_CHECK(cudaGetDevice(&device_id));

  bool use_hw_decomp = hw_supports_decomp_engine(comp_format, device_id);
  if (not use_hw_decomp)
  {
    LOG_INFO("HW Decompression engine is not supported");
    return false;
  }

#ifdef USE_NVTX
  nvtx3::scoped_range hw_decomp{"HWDecomp"};
#endif

  size_t pinned_params_size = sizeof(CUmemDecompressParams) * num_chunks;
  size_t pinned_buffer_bytes_size = force_hw_decomp ? 0 : sizeof(size_t) * num_chunks;
  HostPinnedGuard
    guard(pinned_params_size + pinned_buffer_bytes_size + alignof(size_t) - 1, alignof(CUmemDecompressParams), stream);

  // Pinned memory layout
  // [CUmemDecompressParams, num_chunks]
  // [size_t, num_chunks]
  CUmemDecompressParams *params = static_cast<CUmemDecompressParams *>(guard.get_ptr());
  size_t *host_buffer_bytes = nullptr;
  if (not force_hw_decomp)
  {
    assert(device_uncompressed_buffer_bytes != nullptr);
    host_buffer_bytes = roundUpToAlignment<size_t>(params + num_chunks);
    CUDA_CHECK(cudaMemcpyAsync(
      host_buffer_bytes,
      device_uncompressed_buffer_bytes,
      pinned_buffer_bytes_size,
      cudaMemcpyDefault, // device_uncompressed_buffer_bytes can be host pinned or device memory.
      stream
    ));
  }

  bool res = do_hw_decomp<do_fill_params, DecompParamsFn_t>(
    device_uncompressed_chunk_ptrs,
    device_uncompressed_chunk_bytes,
    num_chunks,
    device_compressed_chunk_ptrs,
    device_compressed_chunk_bytes,
    host_buffer_bytes,
    device_statuses,
    device_id,
    stream,
    params_fn,
    params,
    force_hw_decomp
  );

  return res;
}

} // namespace nvcomp
