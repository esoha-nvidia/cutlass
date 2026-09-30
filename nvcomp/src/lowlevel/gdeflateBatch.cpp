/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
*/

#include <cassert>
#include <iostream>
#include <list>
#include <map>
#include <mutex>
#include <sstream>
#include <vector>

#include "Check.h"
#include "common.h"
#include "gdeflate/common.h"
#include "gdeflate/gdeflate.h"
#include "gdeflate/gdeflateKernels.h"
#include "Logging.h"
#include "lowlevel/nvcomp_private.h"
#include "nvcomp.h"
#include "nvcomp/gdeflate.h"
#include "type_macros.h"

using nvcomp::Check;

// Note: no "catch" statements are triggered during testing.
namespace
{

constexpr const char *gdeflate_compress_opts_format_str = "{{algo={:d}}}";
constexpr const char *gdeflate_decompress_opts_format_str = "{{backend={:d}}}";

#define GDEFLATE_WITH_TRAILING_COMPRESS_OPTS_ARGS(callable, opts, ...) callable(__VA_ARGS__, opts.algorithm)

#define GDEFLATE_WITH_TRAILING_DECOMPRESS_OPTS_ARGS(callable, opts, ...) callable(__VA_ARGS__, int(opts.backend))

#define GDEFLATE_LOG_WITH_COMPRESS_OPTS(log_func, opts, ...)                                                           \
  GDEFLATE_WITH_TRAILING_COMPRESS_OPTS_ARGS(log_func, opts, __func__, __VA_ARGS__, gdeflate_compress_opts_format_str)

#define GDEFLATE_LOG_WITH_DECOMPRESS_OPTS(log_func, opts, ...)                                                         \
  GDEFLATE_WITH_TRAILING_DECOMPRESS_OPTS_ARGS(                                                                         \
    log_func,                                                                                                          \
    opts,                                                                                                              \
    __func__,                                                                                                          \
    __VA_ARGS__,                                                                                                       \
    gdeflate_decompress_opts_format_str                                                                                \
  )

#define GDEFLATE_CHECK_COMPRESS_OPTS(opts) NVCOMP_WRAP_CHECK_FUNC(check_compress_opts, opts)

#define GDEFLATE_CHECK_DECOMPRESS_OPTS(opts) NVCOMP_WRAP_CHECK_FUNC(check_decompress_opts, opts)

nvcompStatus_t check_compress_opts(const nvcompBatchedGdeflateCompressOpts_t &opts)
{
  const bool supported = opts.algorithm >= 0 && opts.algorithm <= 5;
  if (supported)
  {
    return nvcompSuccess;
  }

  GDEFLATE_WITH_TRAILING_COMPRESS_OPTS_ARGS(
    LOG_ERROR,
    opts,
    std::string("Unsupported options: ") + gdeflate_compress_opts_format_str
  );
  return nvcompErrorNotSupported;
}

nvcompStatus_t check_decompress_opts(const nvcompBatchedGdeflateDecompressOpts_t &opts)
{
  const bool supported = opts.backend == NVCOMP_DECOMPRESS_BACKEND_DEFAULT ||
                         opts.backend == NVCOMP_DECOMPRESS_BACKEND_CUDA;
  if (supported)
  {
    return nvcompSuccess;
  }

  // Backend error
  switch (opts.backend)
  {
    case NVCOMP_DECOMPRESS_BACKEND_DEFAULT:
    case NVCOMP_DECOMPRESS_BACKEND_CUDA:
      break;
    case NVCOMP_DECOMPRESS_BACKEND_HARDWARE:
      LOG_ERROR("Hardware backend is not available for Gdeflate.");
      return nvcompErrorInvalidValue;
    default:
      LOG_ERROR(
        "Decompress backend invalid. Choose either NVCOMP_DECOMPRESS_BACKEND_DEFAULT or NVCOMP_DECOMPRESS_BACKEND_CUDA."
      );
      return nvcompErrorInvalidValue;
  }

  // Other errors
  GDEFLATE_WITH_TRAILING_DECOMPRESS_OPTS_ARGS(
    LOG_ERROR,
    opts,
    std::string("Unsupported options: ") + gdeflate_decompress_opts_format_str
  );
  return nvcompErrorNotSupported;
}

} // namespace

static_assert(sizeof(nvcompBatchedGdeflateCompressOpts_t) == 64);
static_assert(sizeof(nvcompBatchedGdeflateDecompressOpts_t) == 64);

namespace gdeflate
{
gdeflate_compression_algo getCompressionAlgo(int algorithm)
{
  switch (algorithm)
  {
    case (0):
      return ENTROPY_ONLY;
    case (1):
      return HIGH_THROUGHPUT;
    case (2):
      return MEDIUM_COMPRESSION;
    //placeholder for further compression level support.
    //Will fall into MEDIUM_COMPRESSION at this point.
    case (3):
      return MEDIUM_COMPRESSION;
    case (4):
      return L6_OP_COMPRESSION;
    case (5):
      return HIGH_COMPRESSION;
    default:
      throw nvcomp::NVCompException(
        nvcompErrorInvalidValue,
        "Invalid compress_opts.algorithm value (not 0, 1, 2, 3, 4, 5)"
      );
  }
}
} // namespace gdeflate

nvcompStatus_t nvcompBatchedGdeflateDecompressGetRequiredAlignments(
  nvcompBatchedGdeflateDecompressOpts_t decompress_opts,
  nvcompAlignmentRequirements_t *alignment_requirements
)
{
  GDEFLATE_LOG_WITH_DECOMPRESS_OPTS(
    nvcomp::logBatchedDecompressGetRequiredAlignments,
    decompress_opts,
    alignment_requirements
  );
  GDEFLATE_CHECK_DECOMPRESS_OPTS(decompress_opts);
  NVCOMP_CHECK_NOT_NULL(alignment_requirements);

  // Input is cast to uint32_t* in decompressAsync, so 4.
  // Output is only ever cast to uint8_t*, so 1.
  // The temp buffer is not used, implying the smallest valid alignment, 1.
  alignment_requirements->input = 4;
  alignment_requirements->output = 1;
  alignment_requirements->temp = 1;

  static_assert(
    ((nvcompGdeflateRequiredDecompressionAlignment) & (nvcompGdeflateRequiredDecompressionAlignment - 1)) == 0,
    "Minimum decompression alignment is invalid."
  );
  assert(
    std::max(std::max(alignment_requirements->input, alignment_requirements->output), alignment_requirements->temp) <=
    nvcompGdeflateRequiredDecompressionAlignment
  );

  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedGdeflateDecompressGetTempSize(
  const size_t num_chunks,
  const size_t max_uncompressed_chunk_bytes,
  nvcompBatchedGdeflateDecompressOpts_t decompress_opts,
  size_t *const temp_bytes,
  const size_t max_total_uncompressed_bytes, // unused, except for logging
  cudaStream_t stream // unused, except for logging
)
{
  GDEFLATE_LOG_WITH_DECOMPRESS_OPTS(
    nvcomp::logBatchedDecompressGetTempSize,
    decompress_opts,
    num_chunks,
    max_uncompressed_chunk_bytes,
    temp_bytes,
    max_total_uncompressed_bytes,
    stream
  );

  GDEFLATE_CHECK_DECOMPRESS_OPTS(decompress_opts);
  NVCOMP_CHECK_NOT_NULL(temp_bytes);
  NVCOMP_CHECK_ALIGNMENT(temp_bytes);
  NVCOMP_CHECK_CHUNK_SIZE(max_uncompressed_chunk_bytes, nvcompGdeflateDecompressionMaxAllowedChunkSize);

  try
  {
    gdeflate::decompressGetTempSize(num_chunks, max_uncompressed_chunk_bytes, temp_bytes);
  }
  catch (const std::exception &e)
  {
    LOG_ERROR("{}", e.what());
    return Check::exception_to_error(e, "nvcompBatchedGdeflateDecompressGetTempSize()");
  }

  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedGdeflateDecompressGetTempSizeSync(
  [[maybe_unused]] const void *const *const device_compressed_chunk_ptrs,
  [[maybe_unused]] const size_t *const device_compressed_chunk_bytes,
  size_t num_chunks,
  size_t max_uncompressed_chunk_bytes,
  size_t *temp_bytes,
  size_t max_total_uncompressed_bytes,
  nvcompBatchedGdeflateDecompressOpts_t decompress_opts,
  nvcompStatus_t *device_statuses,
  cudaStream_t stream
)
{
  NVCOMP_CHECK_ALIGNMENT(device_statuses);

  nvcompStatus_t result = nvcomp::try_clear_device_statuses<true>(num_chunks, device_statuses, stream);
  if (result != nvcompSuccess)
  {
    return result;
  }

  return nvcompBatchedGdeflateDecompressGetTempSize(
    num_chunks,
    max_uncompressed_chunk_bytes,
    decompress_opts,
    temp_bytes,
    max_total_uncompressed_bytes,
    stream
  );
}

nvcompStatus_t nvcompBatchedGdeflateDecompressAsync(
  const void *const *device_compressed_chunk_ptrs,
  const size_t *device_compressed_chunk_bytes,
  const size_t *device_uncompressed_buffer_bytes,
  size_t *device_uncompressed_chunk_bytes,
  size_t num_chunks,
  void *const device_temp_ptr, // unused, except for logging
  size_t temp_bytes, // unused, except for logging
  void *const *device_uncompressed_chunk_ptrs,
  nvcompBatchedGdeflateDecompressOpts_t decompress_opts,
  nvcompStatus_t *device_statuses,
  cudaStream_t stream
)
{
  // NOTE: if we start using `max_uncompressed_chunk_bytes`, we need to check
  // to make sure it is not zero, as we have notified users to supply zero if
  // they are not finding the maximum size.

  nvcomp::logBatchedDecompressAsync(
    __func__,
    device_compressed_chunk_ptrs,
    device_compressed_chunk_bytes,
    device_uncompressed_buffer_bytes,
    device_uncompressed_chunk_bytes,
    num_chunks,
    device_temp_ptr,
    temp_bytes,
    device_uncompressed_chunk_ptrs,
    decompress_opts.backend,
    device_statuses,
    stream
  );

  NVCOMP_CHECK_NOT_NULL(device_compressed_chunk_ptrs);
  NVCOMP_CHECK_NOT_NULL(device_compressed_chunk_bytes);
  NVCOMP_CHECK_NOT_NULL(device_uncompressed_buffer_bytes);
  NVCOMP_CHECK_NOT_NULL(device_uncompressed_chunk_ptrs);
  NVCOMP_CHECK_NOT_NULL(device_uncompressed_chunk_bytes);
  NVCOMP_CHECK_NOT_NULL(device_statuses);
  if (temp_bytes > 0)
  {
    NVCOMP_CHECK_NOT_NULL(device_temp_ptr);
  }

  GDEFLATE_CHECK_DECOMPRESS_OPTS(decompress_opts);

  // Check device pointer alignment
  NVCOMP_CHECK_ALIGNMENT(device_compressed_chunk_ptrs);
  NVCOMP_CHECK_ALIGNMENT(device_compressed_chunk_bytes);
  NVCOMP_CHECK_ALIGNMENT(device_uncompressed_buffer_bytes);
  NVCOMP_CHECK_ALIGNMENT(device_uncompressed_chunk_bytes);
  NVCOMP_CHECK_ALIGNMENT(device_uncompressed_chunk_ptrs);
  NVCOMP_CHECK_ALIGNMENT(device_statuses);

  // Note:
  // - Batch size is used as grid size, which CUDA limits to 2^31 - 1
  // - It is not an exact limit, but a good enough threshold
  NVCOMP_CHECK_BATCH_SIZE(num_chunks, std::numeric_limits<int>::max());

  try
  {
    LOG_INFO("Launching SM decompression");
    // Use device_status_ptrs as temp space to store gdeflate statuses
    static_assert(
      sizeof(nvcompStatus_t) == sizeof(gdeflate::gdeflateStatus_t),
      "Mismatched sizes of nvcompStatus_t and gdeflateStatus_t"
    );
    auto gdeflate_device_statuses = reinterpret_cast<gdeflate::gdeflateStatus_t *>(device_statuses);

    // Run the decompression kernel
    gdeflate::decompressAsync(
      device_compressed_chunk_ptrs,
      device_compressed_chunk_bytes,
      device_uncompressed_buffer_bytes,
      device_uncompressed_chunk_bytes,
      num_chunks,
      device_uncompressed_chunk_ptrs,
      gdeflate_device_statuses,
      stream
    );

    // Launch a kernel to convert the output statuses
    nvcomp::convertGdeflateOutputStatuses(device_statuses, num_chunks, stream);
  }
  catch (const std::exception &e)
  {
    LOG_ERROR("{}", e.what());
    return Check::exception_to_error(e, "nvcompBatchedGdeflateDecompressAsync()");
  }

  return nvcompSuccess;
}
nvcompStatus_t nvcompBatchedGdeflateDecompressAsyncEx(
  const void *const *device_compressed_chunk_ptrs,
  const size_t *device_compressed_chunk_bytes,
  const size_t *device_uncompressed_buffer_bytes,
  size_t *device_uncompressed_chunk_bytes,
  size_t num_chunks,
  void *const device_temp_ptr,
  size_t temp_bytes,
  void *const *device_uncompressed_chunk_ptrs,
  nvcompStatus_t *device_statuses,
  [[maybe_unused]] nvcompBatchedGdeflateCompressOpts_t compress_opts,
  nvcompBatchedGdeflateDecompressOpts_t decompress_opts,
  cudaStream_t stream,
  [[maybe_unused]] const void *const *host_comp_chunk_buffers
)
{
  return nvcompBatchedGdeflateDecompressAsync(
    device_compressed_chunk_ptrs,
    device_compressed_chunk_bytes,
    device_uncompressed_buffer_bytes,
    device_uncompressed_chunk_bytes,
    num_chunks,
    device_temp_ptr,
    temp_bytes,
    device_uncompressed_chunk_ptrs,
    decompress_opts,
    device_statuses,
    stream
  );
}
nvcompStatus_t nvcompBatchedGdeflateGetDecompressSizeAsync(
  const void *const *device_compressed_chunk_ptrs,
  const size_t *device_compressed_chunk_bytes,
  size_t *device_uncompressed_chunk_bytes,
  size_t num_chunks,
  cudaStream_t stream
)
{
  nvcomp::logBatchedGetDecompressSizeAsync(
    __func__,
    device_compressed_chunk_ptrs,
    device_compressed_chunk_bytes,
    device_uncompressed_chunk_bytes,
    num_chunks,
    stream
  );

  // error check inputs
  NVCOMP_CHECK_NOT_NULL(device_compressed_chunk_ptrs);
  NVCOMP_CHECK_NOT_NULL(device_compressed_chunk_bytes);
  NVCOMP_CHECK_NOT_NULL(device_uncompressed_chunk_bytes);

  // Check device pointer alignment
  NVCOMP_CHECK_ALIGNMENT(device_compressed_chunk_ptrs);
  NVCOMP_CHECK_ALIGNMENT(device_compressed_chunk_bytes);
  NVCOMP_CHECK_ALIGNMENT(device_uncompressed_chunk_bytes);

  try
  {
    gdeflate::getDecompressSizeAsync(
      device_compressed_chunk_ptrs,
      device_compressed_chunk_bytes,
      device_uncompressed_chunk_bytes,
      num_chunks,
      stream
    );
  }
  catch (const std::exception &e)
  {
    LOG_ERROR("{}", e.what());
    return Check::exception_to_error(e, "nvcompBatchedGdeflateDecompressAsync()");
  }

  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedGdeflateCompressGetRequiredAlignments(
  nvcompBatchedGdeflateCompressOpts_t format_opts,
  nvcompAlignmentRequirements_t *alignment_requirements
)
{
  GDEFLATE_LOG_WITH_COMPRESS_OPTS(nvcomp::logBatchedCompressGetRequiredAlignments, format_opts, alignment_requirements);
  GDEFLATE_CHECK_COMPRESS_OPTS(format_opts);
  NVCOMP_CHECK_NOT_NULL(alignment_requirements);

  // Only ever cast to pointer to ((un)signed) char or uint8_t.
  alignment_requirements->input = 1;
  // Cast to uint32_t* in compressAsync.
  alignment_requirements->output = 4;
  // Cast to uint16_t** in compressAsync.
  alignment_requirements->temp = sizeof(std::uint16_t *);

  static_assert(
    ((nvcompGdeflateRequiredCompressionAlignment) & (nvcompGdeflateRequiredCompressionAlignment - 1)) == 0,
    "Minimum decompression alignment is invalid."
  );
  assert(
    std::max(std::max(alignment_requirements->input, alignment_requirements->output), alignment_requirements->temp) <=
    nvcompGdeflateRequiredCompressionAlignment
  );

  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedGdeflateCompressGetTempSize(
  const size_t num_chunks,
  const size_t max_uncompressed_chunk_bytes,
  nvcompBatchedGdeflateCompressOpts_t format_opts,
  size_t *const temp_bytes,
  const size_t max_total_uncompressed_bytes, // unused, except for logging
  cudaStream_t stream // unused, except for logging
)
{
  GDEFLATE_LOG_WITH_COMPRESS_OPTS(
    nvcomp::logBatchedCompressGetTempSize,
    format_opts,
    num_chunks,
    max_uncompressed_chunk_bytes,
    temp_bytes,
    max_total_uncompressed_bytes,
    stream
  );
  GDEFLATE_CHECK_COMPRESS_OPTS(format_opts);

  NVCOMP_CHECK_NOT_NULL(temp_bytes);
  NVCOMP_CHECK_ALIGNMENT(temp_bytes);

  NVCOMP_CHECK_CHUNK_SIZE(max_uncompressed_chunk_bytes, nvcompGdeflateCompressionMaxAllowedChunkSize);

  try
  {
    gdeflate::gdeflate_compression_algo algo = gdeflate::getCompressionAlgo(format_opts.algorithm);
    gdeflate::compressGetTempSize(num_chunks, max_uncompressed_chunk_bytes, temp_bytes, algo);
  }
  catch (const std::exception &e)
  {
    LOG_ERROR("{}", e.what());
    return Check::exception_to_error(e, "nvcompBatchedGdeflateCompressGetTempSize()");
  }
  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedGdeflateCompressGetTempSizeSync(
  [[maybe_unused]] const void *const *const device_uncompressed_chunk_ptrs,
  [[maybe_unused]] const size_t *const device_uncompressed_chunk_bytes,
  size_t num_chunks,
  size_t max_uncompressed_chunk_bytes,
  nvcompBatchedGdeflateCompressOpts_t compress_opts,
  size_t *temp_bytes,
  size_t max_total_uncompressed_bytes,
  cudaStream_t stream
)
{
  return nvcompBatchedGdeflateCompressGetTempSize(
    num_chunks,
    max_uncompressed_chunk_bytes,
    compress_opts,
    temp_bytes,
    max_total_uncompressed_bytes,
    stream
  );
}

nvcompStatus_t nvcompBatchedGdeflateCompressGetMaxOutputChunkSize(
  size_t max_uncompressed_chunk_bytes,
  nvcompBatchedGdeflateCompressOpts_t format_opts,
  size_t *max_compressed_chunk_bytes
)
{
  GDEFLATE_LOG_WITH_COMPRESS_OPTS(
    nvcomp::logBatchedCompressGetMaxOutputChunkSize,
    format_opts,
    max_uncompressed_chunk_bytes,
    max_compressed_chunk_bytes
  );
  GDEFLATE_CHECK_COMPRESS_OPTS(format_opts);

  NVCOMP_CHECK_NOT_NULL(max_compressed_chunk_bytes);
  NVCOMP_CHECK_ALIGNMENT(max_compressed_chunk_bytes);

  NVCOMP_CHECK_CHUNK_SIZE(max_uncompressed_chunk_bytes, nvcompGdeflateCompressionMaxAllowedChunkSize);

  try
  {
    gdeflate::compressGetMaxOutputChunkSize(max_uncompressed_chunk_bytes, max_compressed_chunk_bytes);
  }
  catch (const std::exception &e)
  {
    LOG_ERROR("{}", e.what());
    return Check::exception_to_error(e, "nvcompBatchedGdeflateCompressGetMaxOutputChunkSize()");
  }

  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedGdeflateCompressAsync(
  const void *const *const device_uncompressed_chunk_ptrs,
  const size_t *const device_uncompressed_chunk_bytes,
  const size_t max_uncompressed_chunk_bytes,
  const size_t num_chunks,
  void *const device_temp_ptr,
  const size_t temp_bytes,
  void *const *const device_compressed_chunk_ptrs,
  size_t *const device_compressed_chunk_bytes,
  nvcompBatchedGdeflateCompressOpts_t format_opts,
  nvcompStatus_t *device_statuses, // optional
  cudaStream_t stream
)
{
  GDEFLATE_LOG_WITH_COMPRESS_OPTS(
    nvcomp::logBatchedCompressAsync,
    format_opts,
    device_uncompressed_chunk_ptrs,
    device_uncompressed_chunk_bytes,
    max_uncompressed_chunk_bytes,
    num_chunks,
    device_temp_ptr,
    temp_bytes,
    device_compressed_chunk_ptrs,
    device_compressed_chunk_bytes,
    device_statuses,
    stream
  );
  GDEFLATE_CHECK_COMPRESS_OPTS(format_opts);

  // Check device pointer is not NULL
  NVCOMP_CHECK_NOT_NULL(device_uncompressed_chunk_ptrs);
  NVCOMP_CHECK_NOT_NULL(device_uncompressed_chunk_bytes);
  if (temp_bytes > 0)
  {
    NVCOMP_CHECK_NOT_NULL(device_temp_ptr);
  }
  NVCOMP_CHECK_NOT_NULL(device_compressed_chunk_ptrs);
  NVCOMP_CHECK_NOT_NULL(device_compressed_chunk_bytes);

  nvcompAlignmentRequirements_t align_reqs{};
  NVCOMP_WRAP_CHECK_FUNC(nvcompBatchedGdeflateCompressGetRequiredAlignments, format_opts, &align_reqs);
  cuLibLogger::Logger::Instance().SetMostRecentApi(__func__);

  // Check device pointer alignment
  NVCOMP_CHECK_ALIGNMENT(device_uncompressed_chunk_ptrs);
  NVCOMP_CHECK_ALIGNMENT(device_uncompressed_chunk_bytes);
  NVCOMP_CHECK_ALIGNMENT(device_temp_ptr, align_reqs.temp);
  NVCOMP_CHECK_ALIGNMENT(device_compressed_chunk_ptrs);
  NVCOMP_CHECK_ALIGNMENT(device_compressed_chunk_bytes);
  NVCOMP_CHECK_ALIGNMENT(device_statuses);

  NVCOMP_CHECK_CHUNK_SIZE(max_uncompressed_chunk_bytes, nvcompGdeflateCompressionMaxAllowedChunkSize);

  // Note:
  // - Batch size is used as grid size, which CUDA limits to 2^31 - 1
  // - It is not an exact limit, but a good enough threshold
  NVCOMP_CHECK_BATCH_SIZE(num_chunks, std::numeric_limits<int>::max());

  try
  {
    gdeflate::gdeflate_compression_algo algo = gdeflate::getCompressionAlgo(format_opts.algorithm);
    gdeflate::compressAsync(
      device_uncompressed_chunk_ptrs,
      device_uncompressed_chunk_bytes,
      max_uncompressed_chunk_bytes,
      num_chunks,
      device_temp_ptr,
      temp_bytes,
      device_compressed_chunk_ptrs,
      device_compressed_chunk_bytes,
      algo,
      device_statuses,
      stream
    );
  }
  catch (const std::exception &e)
  {
    LOG_ERROR("{}", e.what());
    return Check::exception_to_error(e, "nvcompBatchedGdeflateCompressAsync()");
  }
  return nvcompSuccess;
}
