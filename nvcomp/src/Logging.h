/*
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

#include <cuda_runtime_api.h> // For cudaStream_t

#include "nvcomp/shared_types.h"

#ifndef _MSC_VER
// Some headers included by spdlog don't follow the strict requirements of -Weffc++,
// and since we have warnings as errors, it needs to be suppressed temporarily.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"

// In debug Linux builds, at least one header has a "#pragma GCC optimize("Og")"
// that GCC supports, but nvcc doesn't recognize, so we need to suppress the
// "unrecognized GCC pragma" warning.
#ifdef __CUDACC__
#pragma nv_diagnostic push
#pragma nv_diag_suppress 1675
#endif
#endif

#define LOG_ENABLED 1

#define LOGGER_NAME "nvcomp"
#define LOGGER_LOG_LEVEL_ENV "NVCOMP_LOG_LEVEL"
#define LOGGER_LOG_MASK_ENV "NVCOMP_LOG_MASK"
#define LOGGER_LOG_FILE_ENV "NVCOMP_LOG_FILE"

#define LOGGER_BEGIN_NAMESPACE                                                                                         \
  namespace nvcompLogger                                                                                               \
  {                                                                                                                    \
  namespace cuLibLogger                                                                                                \
  {
#define LOGGER_END_NAMESPACE                                                                                           \
  }                                                                                                                    \
  }                                                                                                                    \
  using namespace nvcompLogger;

#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif
#include <cuLibLogger/cuLibLogger.h>
#include <fmt/chrono.h>
#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/printf.h>

#ifndef _MSC_VER
#ifdef __CUDACC__
#pragma nv_diagnostic pop
#endif
#pragma GCC diagnostic pop
#endif

#include <optional>
#include <type_traits>
#include <utility>

#include <stdint.h>

namespace nvcomp
{

template <typename... Args>
void logApi(const char *const function_name, const fmt::string_view &format, Args &&...args)
{
  auto &logger = cuLibLogger::Logger::Instance();
  logger.SetMostRecentApi(function_name);
  if (logger.ShouldLogMessage(cuLibLogger::Logger::Level::Api, cuLibLogger::Logger::Mask::ApiMask))
  {
    logger.Log(cuLibLogger::Logger::Level::Api, cuLibLogger::Logger::Mask::ApiMask, format, std::forward<Args>(args)...);
  }
}

// LLIF call logging

template <typename... Args>
void logBatchedCompressGetRequiredAlignments(
  const char *const function_name,
  nvcompAlignmentRequirements_t *alignment_requirements,
  const char *const options_format_string = "",
  Args &&...options_args
)
{
  std::string format_string = std::string("options=(") + options_format_string + "), alignment_requirements={:#x}";
  fmt::string_view str = format_string;
  logApi(function_name, str, std::forward<Args>(options_args)..., reinterpret_cast<uintptr_t>(alignment_requirements));
}

template <typename... Args>
void logBatchedCompressGetTempSizeAsync(
  const char *const function_name,
  size_t num_chunks,
  size_t max_uncompressed_chunk_bytes,
  size_t *temp_bytes,
  size_t max_total_uncompressed_bytes,
  const char *const options_format_string = "",
  Args &&...options_args
)
{
  std::string format_string = std::string("num_chunks={}, max_uncomp_chunk_bytes={}, options=(") +
                              options_format_string + "), temp_bytes={:#x}, max_total_uncomp_bytes={}";
  fmt::string_view str = format_string;
  logApi(
    function_name,
    str,
    num_chunks,
    max_uncompressed_chunk_bytes,
    std::forward<Args>(options_args)...,
    reinterpret_cast<uintptr_t>(temp_bytes),
    max_total_uncompressed_bytes
  );
}

template <typename... Args>
void logBatchedCompressGetMaxOutputChunkSize(
  const char *const function_name,
  size_t max_uncompressed_chunk_bytes,
  size_t *max_compressed_bytes,
  const char *const options_format_string = "",
  Args &&...options_args
)
{
  std::string format_string = std::string("max_uncomp_chunk_bytes={}, options=(") + options_format_string +
                              "), max_comp_bytes={:#x}";
  fmt::string_view str = format_string;
  logApi(
    function_name,
    str,
    max_uncompressed_chunk_bytes,
    std::forward<Args>(options_args)...,
    reinterpret_cast<uintptr_t>(max_compressed_bytes)
  );
}

template <typename... Args>
void logBatchedCompressAsync(
  const char *const function_name,
  const void *const *device_uncompressed_ptrs,
  const size_t *device_uncompressed_bytes,
  size_t max_uncompressed_chunk_bytes,
  size_t num_chunks,
  void *device_temp_ptr,
  size_t temp_bytes,
  void *const *device_compressed_ptrs,
  size_t *device_compressed_bytes,
  nvcompStatus_t *device_statuses,
  cudaStream_t stream,
  const char *const options_format_string = "",
  Args &&...options_args
)
{
  std::string format_string = std::string(
                                "d_uncomp_ptrs={:#x}, d_uncomp_bytes={:#x}, max_uncomp_chunk_bytes={}, num_chunks={}, "
                                "d_temp_ptr={:#x}, temp_bytes={}, d_comp_ptrs={:#x}, d_comp_bytes={:#x}, options=("
                              ) +
                              options_format_string + "), d_statuses={:#x}, stream={:#x}";

  fmt::string_view str = format_string;
  logApi(
    function_name,
    str,
    reinterpret_cast<uintptr_t>(device_uncompressed_ptrs),
    reinterpret_cast<uintptr_t>(device_uncompressed_bytes),
    max_uncompressed_chunk_bytes,
    num_chunks,
    reinterpret_cast<uintptr_t>(device_temp_ptr),
    temp_bytes,
    reinterpret_cast<uintptr_t>(device_compressed_ptrs),
    reinterpret_cast<uintptr_t>(device_compressed_bytes),
    std::forward<Args>(options_args)...,
    reinterpret_cast<uintptr_t>(device_statuses),
    reinterpret_cast<uintptr_t>(stream)
  );
}

template <typename... Args>
void logBatchedDecompressGetRequiredAlignments(
  const char *const function_name,
  nvcompAlignmentRequirements_t *alignment_requirements,
  const char *const options_format_string = "",
  Args &&...options_args
)
{
  std::string format_string = std::string("options=(") + options_format_string + "), alignment_requirements={:#x}";
  fmt::string_view str = format_string;
  logApi(function_name, str, std::forward<Args>(options_args)..., reinterpret_cast<uintptr_t>(alignment_requirements));
}

template <typename... Args>
void logBatchedDecompressGetTempSizeAsync(
  const char *const function_name,
  size_t num_chunks,
  size_t max_uncompressed_chunk_bytes,
  size_t *temp_bytes,
  size_t max_uncompressed_total_bytes,
  const char *const options_format_string = "",
  Args &&...options_args
)
{
  std::string format_string = std::string("num_chunks={}, max_uncompressed_chunk_bytes={}, decomp_opts=(") +
                              options_format_string + "), temp_bytes={:#x}, max_uncomp_total_bytes={}";
  fmt::string_view str = format_string;
  logApi(
    function_name,
    str,
    num_chunks,
    max_uncompressed_chunk_bytes,
    std::forward<Args>(options_args)...,
    reinterpret_cast<uintptr_t>(temp_bytes),
    max_uncompressed_total_bytes
  );
}

inline void logBatchedDecompressGetTempSizeSync(
  const char *const function_name,
  const void *const *device_compressed_ptrs,
  const size_t *device_compressed_bytes,
  size_t num_chunks,
  size_t max_uncompressed_chunk_bytes,
  size_t *temp_bytes,
  size_t max_total_uncompressed_bytes,
  nvcompDecompressBackend_t backend,
  nvcompStatus_t *device_statuses,
  cudaStream_t stream
)
{
  std::string format_string = std::string(
    "d_comp_ptrs={:#x}, d_comp_bytes={:#x}, num_chunks={}, max_uncompressed_chunk_bytes={}, temp_bytes={:#x}, "
    "max_total_uncompressed_bytes={}, decompress_backend={}, d_statuses={:#x}, stream={:#x}"
  );
  fmt::string_view str = format_string;
  logApi(
    function_name,
    str,
    reinterpret_cast<uintptr_t>(device_compressed_ptrs),
    reinterpret_cast<uintptr_t>(device_compressed_bytes),
    num_chunks,
    max_uncompressed_chunk_bytes,
    reinterpret_cast<uintptr_t>(temp_bytes),
    max_total_uncompressed_bytes,
    static_cast<int>(backend),
    reinterpret_cast<uintptr_t>(device_statuses),
    reinterpret_cast<uintptr_t>(stream)
  );
}

inline void logBatchedDecompressAsync(
  const char *const function_name,
  const void *const *device_compressed_ptrs,
  const size_t *device_compressed_bytes,
  const size_t *device_uncompressed_bytes,
  size_t *device_actual_uncompressed_bytes,
  size_t num_chunks,
  void *const device_temp_ptr,
  size_t temp_bytes,
  void *const *device_uncompressed_ptrs,
  nvcompDecompressBackend_t backend,
  nvcompStatus_t *device_statuses,
  cudaStream_t stream
)
{
  std::string format_string = std::string(
    "d_comp_ptrs={:#x}, d_comp_bytes={:#x}, d_uncomp_bytes={:#x}, d_actual_uncomp_bytes={:#x}, num_chunks={}, "
    "d_temp_ptr={:#x}, temp_bytes={}, d_uncomp_ptrs={:#x}, decompress_backend={}, d_statuses={:#x}, stream={:#x}"
  );
  fmt::string_view str = format_string;
  logApi(
    function_name,
    str,
    reinterpret_cast<uintptr_t>(device_compressed_ptrs),
    reinterpret_cast<uintptr_t>(device_compressed_bytes),
    reinterpret_cast<uintptr_t>(device_uncompressed_bytes),
    reinterpret_cast<uintptr_t>(device_actual_uncompressed_bytes),
    num_chunks,
    reinterpret_cast<uintptr_t>(device_temp_ptr),
    temp_bytes,
    reinterpret_cast<uintptr_t>(device_uncompressed_ptrs),
    static_cast<int>(backend),
    reinterpret_cast<uintptr_t>(device_statuses),
    reinterpret_cast<uintptr_t>(stream)
  );
}

inline void logBatchedGetDecompressSizeAsync(
  const char *const function_name,
  const void *const *device_compressed_ptrs,
  const size_t *device_compressed_bytes,
  size_t *device_uncompressed_bytes,
  size_t num_chunks,
  cudaStream_t stream
)
{
  std::string format_string =
    std::string("d_comp_ptrs={:#x}, d_comp_bytes={:#x}, d_uncomp_bytes={:#x}, num_chunks={}, stream={:#x}");
  fmt::string_view str = format_string;
  logApi(
    function_name,
    str,
    reinterpret_cast<uintptr_t>(device_compressed_ptrs),
    reinterpret_cast<uintptr_t>(device_compressed_bytes),
    reinterpret_cast<uintptr_t>(device_uncompressed_bytes),
    num_chunks,
    reinterpret_cast<uintptr_t>(stream)
  );
}

} // namespace nvcomp
