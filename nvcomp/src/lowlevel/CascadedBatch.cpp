/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
*/

#include "cascaded/composite/composite_constants.cuh"
#include "common.h"
#include "Logging.h"
#include "lowlevel/Check.h"
#include "lowlevel/nvcomp_private.h"
#include "nvcomp/cascaded.h"
#include "nvcomp/utils.hpp"

using nvcomp::Check;
using nvcomp::isAligned;
using nvcomp::roundUpTo;

namespace
{

constexpr const char *CASCADED_COMPRESS_OPTS_FORMAT_STR =
  "{{common_opts={{data_type={:d}, mode={:d}, terminal_codec_flags={:#x}, "
  "coarse_grained_encoding_flags={:#x}}}, compression_level={:d}, fine_grained_encoding_flags={:#x}}}";
constexpr const char *CASCADED_DECOMPRESS_OPTS_FORMAT_STR =
  "{{backend={:d}, common_opts={{data_type={:d}, mode={:d}, terminal_codec_flags={:#x}, "
  "coarse_grained_encoding_flags={:#x}}}}}";

#define CASCADED_WITH_TRAILING_COMPRESS_OPTS_ARGS(callable, opts, ...)                                                 \
  callable(                                                                                                            \
    __VA_ARGS__,                                                                                                       \
    int(opts.common_opts.data_type),                                                                                   \
    int(opts.common_opts.mode),                                                                                        \
    opts.common_opts.terminal_codec_flags,                                                                             \
    opts.common_opts.coarse_grained_encoding_flags,                                                                    \
    unsigned(opts.compression_level),                                                                                  \
    opts.fine_grained_encoding_flags                                                                                   \
  )

#define CASCADED_WITH_TRAILING_DECOMPRESS_OPTS_ARGS(callable, opts, ...)                                               \
  callable(                                                                                                            \
    __VA_ARGS__,                                                                                                       \
    int(opts.backend),                                                                                                 \
    int(opts.common_opts.data_type),                                                                                   \
    int(opts.common_opts.mode),                                                                                        \
    opts.common_opts.terminal_codec_flags,                                                                             \
    opts.common_opts.coarse_grained_encoding_flags                                                                     \
  )

#define CASCADED_LOG_WITH_COMPRESS_OPTS(log_func, opts, ...)                                                           \
  CASCADED_WITH_TRAILING_COMPRESS_OPTS_ARGS(log_func, opts, __func__, __VA_ARGS__, CASCADED_COMPRESS_OPTS_FORMAT_STR)

#define CASCADED_LOG_WITH_DECOMPRESS_OPTS(log_func, opts, ...)                                                         \
  CASCADED_WITH_TRAILING_DECOMPRESS_OPTS_ARGS(                                                                         \
    log_func,                                                                                                          \
    opts,                                                                                                              \
    __func__,                                                                                                          \
    __VA_ARGS__,                                                                                                       \
    CASCADED_DECOMPRESS_OPTS_FORMAT_STR                                                                                \
  )

#define CASCADED_CHECK_COMPRESS_OPTS(opts) NVCOMP_WRAP_CHECK_FUNC(check_compress_opts, opts)

#define CASCADED_CHECK_DECOMPRESS_OPTS(opts) NVCOMP_WRAP_CHECK_FUNC(check_decompress_opts, opts)

bool is_supported_data_type(const nvcompType_t data_type)
{
  return data_type == NVCOMP_TYPE_CHAR || data_type == NVCOMP_TYPE_UCHAR || data_type == NVCOMP_TYPE_SHORT ||
         data_type == NVCOMP_TYPE_USHORT || data_type == NVCOMP_TYPE_INT || data_type == NVCOMP_TYPE_UINT ||
         data_type == NVCOMP_TYPE_LONGLONG || data_type == NVCOMP_TYPE_ULONGLONG;
}

bool is_supported_symmetric_data_type(const nvcompType_t data_type)
{
  return data_type == NVCOMP_TYPE_INT || data_type == NVCOMP_TYPE_UINT || data_type == NVCOMP_TYPE_LONGLONG ||
         data_type == NVCOMP_TYPE_ULONGLONG;
}

bool is_supported_common_opts(const nvcompCascadedCommonOpts_t &common_opts, const bool allow_unspecified)
{
  if (common_opts.mode == NVCOMP_CASCADED_MODE_UNSPECIFIED)
  {
    return allow_unspecified;
  }

  constexpr uint64_t SUPPORTED_TERMINAL_CODECS = NVCOMP_CASCADED_TERMINAL_CODEC_CASCADED_BITPACK;
  constexpr uint64_t SUPPORTED_COARSE_GRAINED_ENCODINGS = NVCOMP_CASCADED_COARSE_GRAINED_ENCODING_NONE;
  const bool supported_mode_and_type =
    (common_opts.mode == NVCOMP_CASCADED_MODE_ASYMMETRIC && is_supported_data_type(common_opts.data_type)) ||
    (common_opts.mode == NVCOMP_CASCADED_MODE_SYMMETRIC && is_supported_symmetric_data_type(common_opts.data_type));
  return supported_mode_and_type && common_opts.terminal_codec_flags != NVCOMP_CASCADED_TERMINAL_CODEC_UNSPECIFIED &&
         (common_opts.terminal_codec_flags & ~SUPPORTED_TERMINAL_CODECS) == 0u &&
         (common_opts.coarse_grained_encoding_flags & ~SUPPORTED_COARSE_GRAINED_ENCODINGS) == 0u;
}

nvcompStatus_t check_compress_opts(const nvcompBatchedCascadedCompressOpts_t &opts)
{
  constexpr uint64_t ASYMMETRIC_REQUIRED_ENCODINGS = NVCOMP_CASCADED_FINE_GRAINED_ENCODING_FOR;
  const bool has_required_encodings = opts.common_opts.mode != NVCOMP_CASCADED_MODE_ASYMMETRIC ||
                                      (opts.fine_grained_encoding_flags & ASYMMETRIC_REQUIRED_ENCODINGS) ==
                                        ASYMMETRIC_REQUIRED_ENCODINGS;
  const bool supported = is_supported_common_opts(opts.common_opts, false) &&
                         (opts.fine_grained_encoding_flags & ~NVCOMP_CASCADED_FINE_GRAINED_ENCODING_ALL) == 0u &&
                         has_required_encodings;
  if (supported)
  {
    return nvcompSuccess;
  }

  CASCADED_WITH_TRAILING_COMPRESS_OPTS_ARGS(
    LOG_ERROR,
    opts,
    std::string("Unsupported options: ") + CASCADED_COMPRESS_OPTS_FORMAT_STR
  );
  return nvcompErrorNotSupported;
}

nvcompStatus_t check_decompress_opts(const nvcompBatchedCascadedDecompressOpts_t &opts)
{
  const bool supported_backend = opts.backend == NVCOMP_DECOMPRESS_BACKEND_DEFAULT ||
                                 opts.backend == NVCOMP_DECOMPRESS_BACKEND_CUDA;
  const bool supported = supported_backend && is_supported_common_opts(opts.common_opts, true);
  if (supported)
  {
    return nvcompSuccess;
  }

  switch (opts.backend)
  {
    case NVCOMP_DECOMPRESS_BACKEND_DEFAULT:
    case NVCOMP_DECOMPRESS_BACKEND_CUDA:
      break;
    case NVCOMP_DECOMPRESS_BACKEND_HARDWARE:
      LOG_ERROR("Hardware backend is not available for Cascaded.");
      return nvcompErrorInvalidValue;
    default:
      LOG_ERROR(
        "Decompress backend invalid. Choose either NVCOMP_DECOMPRESS_BACKEND_DEFAULT or NVCOMP_DECOMPRESS_BACKEND_CUDA."
      );
      return nvcompErrorInvalidValue;
  }

  CASCADED_WITH_TRAILING_DECOMPRESS_OPTS_ARGS(
    LOG_ERROR,
    opts,
    std::string("Unsupported options: ") + CASCADED_DECOMPRESS_OPTS_FORMAT_STR
  );
  return nvcompErrorNotSupported;
}

} // namespace

namespace nvcomp
{
// Forward declarations for functions in CascadedBatchCuda.cu
// These are in a separate file, because with both CUDA and spdlog
// in the same file, nvcc sometimes uses too much memory for the
// build machines.
// TODO(mpayrits): Check if still true with culiblogger.
nvcompStatus_t cascadedCompressAsyncDispatch(
  const void *const *device_uncompressed_chunk_ptrs,
  const size_t *device_uncompressed_chunk_bytes,
  size_t num_chunks,
  void *const *device_compressed_chunk_ptrs,
  size_t *device_compressed_chunk_bytes,
  const nvcompBatchedCascadedCompressOpts_t format_opts,
  nvcompStatus_t *device_statuses,
  cudaStream_t stream
);
nvcompStatus_t cascadedCompressGetMaxOutputChunkSizeDispatch(
  size_t max_uncompressed_chunk_bytes,
  nvcompBatchedCascadedCompressOpts_t format_opts,
  size_t *max_compressed_chunk_bytes
);
nvcompStatus_t cascadedDecompressAsyncDispatch(
  const void *const *device_compressed_chunk_ptrs,
  const size_t *device_compressed_chunk_bytes,
  const size_t *device_uncompressed_buffer_bytes,
  size_t *device_uncompressed_chunk_bytes,
  size_t num_chunks,
  void *const *device_uncompressed_chunk_ptrs,
  nvcompStatus_t *device_statuses,
  cudaStream_t stream
);
nvcompStatus_t cascadedDecompressAsyncDispatch(
  const void *const *device_compressed_chunk_ptrs,
  const size_t *device_compressed_chunk_bytes,
  const size_t *device_uncompressed_buffer_bytes,
  size_t *device_uncompressed_chunk_bytes,
  size_t num_chunks,
  void *const *device_uncompressed_chunk_ptrs,
  nvcompStatus_t *device_statuses,
  nvcompCascadedCommonOpts_t common_opts,
  cudaStream_t stream
);
nvcompStatus_t cascadedGetDecompressSizeAsyncDispatch(
  const void *const *device_compressed_chunk_ptrs,
  const size_t *device_compressed_chunk_bytes,
  size_t *device_uncompressed_chunk_bytes,
  size_t num_chunks,
  cudaStream_t stream
);
} // namespace nvcomp

static_assert(sizeof(nvcompBatchedCascadedCompressOpts_t) == 64);
static_assert(sizeof(nvcompBatchedCascadedDecompressOpts_t) == 64);

nvcompStatus_t nvcompBatchedCascadedCompressGetRequiredAlignments(
  nvcompBatchedCascadedCompressOpts_t format_opts,
  nvcompAlignmentRequirements_t *alignment_requirements
)
{
  CASCADED_LOG_WITH_COMPRESS_OPTS(nvcomp::logBatchedCompressGetRequiredAlignments, format_opts, alignment_requirements);
  CASCADED_CHECK_COMPRESS_OPTS(format_opts);
  NVCOMP_CHECK_NOT_NULL(alignment_requirements);

  // Cast to corresponding type in cascaded_batched_compression_typed.
  alignment_requirements->input = nvcomp::sizeOfnvcompType(format_opts.common_opts.data_type);
  // Cast to uint32_t* in do_cascaded_compression_kernel.
  alignment_requirements->output = 4;
  alignment_requirements->temp = 1; // Not used.

  static_assert(
    ((nvcompCascadedRequiredCompressionAlignment) & (nvcompCascadedRequiredCompressionAlignment - 1)) == 0,
    "Minimum compression alignment is invalid."
  );
  assert(
    std::max(std::max(alignment_requirements->input, alignment_requirements->output), alignment_requirements->temp) <=
    nvcompCascadedRequiredCompressionAlignment
  );

  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedCascadedCompressGetTempSize(
  size_t num_chunks, // unused, except for logging
  size_t max_uncompressed_chunk_bytes, // unused, except for logging
  nvcompBatchedCascadedCompressOpts_t compress_opts,
  size_t *temp_bytes,
  size_t max_total_uncompressed_bytes, // unused, except for logging
  cudaStream_t stream // unused, except for logging
)
{
  CASCADED_LOG_WITH_COMPRESS_OPTS(
    nvcomp::logBatchedCompressGetTempSize,
    compress_opts,
    num_chunks,
    max_uncompressed_chunk_bytes,
    temp_bytes,
    max_total_uncompressed_bytes,
    stream
  );

  CASCADED_CHECK_COMPRESS_OPTS(compress_opts);
  NVCOMP_CHECK_NOT_NULL(temp_bytes);
  NVCOMP_CHECK_ALIGNMENT(temp_bytes);

  *temp_bytes = 0;

  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedCascadedCompressGetTempSizeSync(
  [[maybe_unused]] const void *const *const device_uncompressed_chunk_ptrs,
  [[maybe_unused]] const size_t *const device_uncompressed_chunk_bytes,
  size_t num_chunks,
  size_t max_uncompressed_chunk_bytes,
  nvcompBatchedCascadedCompressOpts_t compress_opts,
  size_t *temp_bytes,
  size_t max_total_uncompressed_bytes,
  cudaStream_t stream
)
{
  return nvcompBatchedCascadedCompressGetTempSize(
    num_chunks,
    max_uncompressed_chunk_bytes,
    compress_opts,
    temp_bytes,
    max_total_uncompressed_bytes,
    stream
  );
}

nvcompStatus_t nvcompBatchedCascadedCompressGetMaxOutputChunkSize(
  size_t max_uncompressed_chunk_bytes,
  nvcompBatchedCascadedCompressOpts_t format_opts,
  size_t *max_compressed_chunk_bytes
)
{
  CASCADED_LOG_WITH_COMPRESS_OPTS(
    nvcomp::logBatchedCompressGetMaxOutputChunkSize,
    format_opts,
    max_uncompressed_chunk_bytes,
    max_compressed_chunk_bytes
  );
  CASCADED_CHECK_COMPRESS_OPTS(format_opts);

  NVCOMP_CHECK_NOT_NULL(max_compressed_chunk_bytes);
  NVCOMP_CHECK_ALIGNMENT(max_compressed_chunk_bytes);

  NVCOMP_CHECK_CHUNK_SIZE(max_uncompressed_chunk_bytes, nvcompCascadedCompressionMaxAllowedChunkSize);

  return nvcomp::cascadedCompressGetMaxOutputChunkSizeDispatch(
    max_uncompressed_chunk_bytes,
    format_opts,
    max_compressed_chunk_bytes
  );
}

nvcompStatus_t nvcompBatchedCascadedCompressAsync(
  const void *const *device_uncompressed_chunk_ptrs,
  const size_t *device_uncompressed_chunk_bytes,
  size_t max_uncompressed_chunk_bytes, // unused, except for logging
  size_t num_chunks,
  void *device_temp_ptr, // unused, except for logging
  size_t temp_bytes, // unused, except for logging
  void *const *device_compressed_chunk_ptrs,
  size_t *device_compressed_chunk_bytes,
  const nvcompBatchedCascadedCompressOpts_t format_opts,
  nvcompStatus_t *device_statuses, // optional
  cudaStream_t stream
)
{
  CASCADED_LOG_WITH_COMPRESS_OPTS(
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
  CASCADED_CHECK_COMPRESS_OPTS(format_opts);

  NVCOMP_CHECK_NOT_NULL(device_uncompressed_chunk_ptrs);
  NVCOMP_CHECK_NOT_NULL(device_uncompressed_chunk_bytes);
  NVCOMP_CHECK_NOT_NULL(device_compressed_chunk_ptrs);
  NVCOMP_CHECK_NOT_NULL(device_compressed_chunk_bytes);
  if (temp_bytes > 0)
  {
    NVCOMP_CHECK_NOT_NULL(device_temp_ptr);
  }

  // Check device pointer alignment
  NVCOMP_CHECK_ALIGNMENT(device_uncompressed_chunk_ptrs);
  NVCOMP_CHECK_ALIGNMENT(device_uncompressed_chunk_bytes);
  NVCOMP_CHECK_ALIGNMENT(device_compressed_chunk_ptrs);
  NVCOMP_CHECK_ALIGNMENT(device_compressed_chunk_bytes);
  NVCOMP_CHECK_ALIGNMENT(device_statuses);

  // Batch size is used as grid size, which CUDA limits to 2^31 - 1
  NVCOMP_CHECK_BATCH_SIZE(num_chunks, std::numeric_limits<int>::max());

  return nvcomp::cascadedCompressAsyncDispatch(
    device_uncompressed_chunk_ptrs,
    device_uncompressed_chunk_bytes,
    num_chunks,
    device_compressed_chunk_ptrs,
    device_compressed_chunk_bytes,
    format_opts,
    device_statuses,
    stream
  );
}

nvcompStatus_t nvcompBatchedCascadedDecompressGetRequiredAlignments(
  nvcompBatchedCascadedDecompressOpts_t decompress_opts,
  nvcompAlignmentRequirements_t *alignment_requirements
)
{
  CASCADED_LOG_WITH_DECOMPRESS_OPTS(
    nvcomp::logBatchedDecompressGetRequiredAlignments,
    decompress_opts,
    alignment_requirements
  );
  CASCADED_CHECK_DECOMPRESS_OPTS(decompress_opts);
  NVCOMP_CHECK_NOT_NULL(alignment_requirements);

  // Input is cast to uint32_t* in cascaded_decompression_fcn, so 4 bytes.
  // Output depends on the opts.common_opts.data_type used during compression and its
  // alignment is the same as the alignment of the compression input. Since we
  // do not have access to compression options, the worst-case scenario is used,
  // which is 8 bytes.
  // The temp buffer is not used, implying the smallest valid alignment, 1.
  alignment_requirements->input = 4;
  alignment_requirements->output = 8;
  alignment_requirements->temp = 1;

  static_assert(
    ((nvcompCascadedRequiredDecompressionAlignment) & (nvcompCascadedRequiredDecompressionAlignment - 1)) == 0,
    "Minimum decompression alignment is invalid."
  );
  assert(
    std::max(std::max(alignment_requirements->input, alignment_requirements->output), alignment_requirements->temp) <=
    nvcompCascadedRequiredDecompressionAlignment
  );

  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedCascadedDecompressGetTempSize(
  size_t num_chunks, // unused, except for logging
  size_t max_uncompressed_chunk_bytes,
  nvcompBatchedCascadedDecompressOpts_t decompress_opts,
  size_t *temp_bytes,
  size_t max_total_uncompressed_bytes, // unused, except for logging
  cudaStream_t stream // unused, except for logging
)
{
  CASCADED_LOG_WITH_DECOMPRESS_OPTS(
    nvcomp::logBatchedDecompressGetTempSize,
    decompress_opts,
    num_chunks,
    max_uncompressed_chunk_bytes,
    temp_bytes,
    max_total_uncompressed_bytes,
    stream
  );

  CASCADED_CHECK_DECOMPRESS_OPTS(decompress_opts);
  NVCOMP_CHECK_NOT_NULL(temp_bytes);
  NVCOMP_CHECK_ALIGNMENT(temp_bytes);
  NVCOMP_CHECK_CHUNK_SIZE(max_uncompressed_chunk_bytes, nvcompCascadedDecompressionMaxAllowedChunkSize);

  *temp_bytes = 0;

  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedCascadedDecompressGetTempSizeSync(
  [[maybe_unused]] const void *const *const device_compressed_chunk_ptrs,
  [[maybe_unused]] const size_t *const device_compressed_chunk_bytes,
  size_t num_chunks,
  size_t max_uncompressed_chunk_bytes,
  size_t *temp_bytes,
  size_t max_total_uncompressed_bytes,
  nvcompBatchedCascadedDecompressOpts_t decompress_opts,
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

  return nvcompBatchedCascadedDecompressGetTempSize(
    num_chunks,
    max_uncompressed_chunk_bytes,
    decompress_opts,
    temp_bytes,
    max_total_uncompressed_bytes,
    stream
  );
}

nvcompStatus_t nvcompBatchedCascadedDecompressAsync(
  const void *const *device_compressed_chunk_ptrs,
  const size_t *device_compressed_chunk_bytes,
  const size_t *device_uncompressed_buffer_bytes,
  size_t *device_uncompressed_chunk_bytes,
  size_t num_chunks,
  void *const device_temp_ptr, // unused, except for logging
  size_t temp_bytes, // unused, except for logging
  void *const *device_uncompressed_chunk_ptrs,
  nvcompBatchedCascadedDecompressOpts_t decompress_opts,
  nvcompStatus_t *device_statuses,
  cudaStream_t stream
)
{
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

  CASCADED_CHECK_DECOMPRESS_OPTS(decompress_opts);

  // Check device pointer alignment
  NVCOMP_CHECK_ALIGNMENT(device_compressed_chunk_ptrs);
  NVCOMP_CHECK_ALIGNMENT(device_compressed_chunk_bytes);
  NVCOMP_CHECK_ALIGNMENT(device_uncompressed_buffer_bytes);
  NVCOMP_CHECK_ALIGNMENT(device_uncompressed_chunk_bytes);
  NVCOMP_CHECK_ALIGNMENT(device_uncompressed_chunk_ptrs);
  NVCOMP_CHECK_ALIGNMENT(device_statuses);

  // Batch size is used as grid size, which CUDA limits to 2^31 - 1
  NVCOMP_CHECK_BATCH_SIZE(num_chunks, std::numeric_limits<int>::max());

  LOG_INFO("Launching SM decompression");
  if (decompress_opts.common_opts.mode != NVCOMP_CASCADED_MODE_UNSPECIFIED)
  {
    return nvcomp::cascadedDecompressAsyncDispatch(
      device_compressed_chunk_ptrs,
      device_compressed_chunk_bytes,
      device_uncompressed_buffer_bytes,
      device_uncompressed_chunk_bytes,
      num_chunks,
      device_uncompressed_chunk_ptrs,
      device_statuses,
      decompress_opts.common_opts,
      stream
    );
  }

  return nvcomp::cascadedDecompressAsyncDispatch(
    device_compressed_chunk_ptrs,
    device_compressed_chunk_bytes,
    device_uncompressed_buffer_bytes,
    device_uncompressed_chunk_bytes,
    num_chunks,
    device_uncompressed_chunk_ptrs,
    device_statuses,
    stream
  );
}

nvcompStatus_t nvcompBatchedCascadedDecompressAsyncEx(
  const void *const *device_compressed_chunk_ptrs,
  const size_t *device_compressed_chunk_bytes,
  const size_t *device_uncompressed_buffer_bytes,
  size_t *device_uncompressed_chunk_bytes,
  size_t num_chunks,
  void *const device_temp_ptr,
  size_t temp_bytes,
  void *const *device_uncompressed_chunk_ptrs,
  nvcompStatus_t *device_statuses,
  nvcompBatchedCascadedCompressOpts_t compress_opts,
  nvcompBatchedCascadedDecompressOpts_t decompress_opts,
  cudaStream_t stream,
  [[maybe_unused]] const void *const * /*host_comp_chunk_buffers*/
)
{
  decompress_opts.common_opts = compress_opts.common_opts;
  return nvcompBatchedCascadedDecompressAsync(
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

nvcompStatus_t nvcompBatchedCascadedGetDecompressSizeAsync(
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

  NVCOMP_CHECK_NOT_NULL(device_compressed_chunk_ptrs);
  NVCOMP_CHECK_NOT_NULL(device_compressed_chunk_bytes);
  NVCOMP_CHECK_NOT_NULL(device_uncompressed_chunk_bytes);

  // Check device pointer alignment
  NVCOMP_CHECK_ALIGNMENT(device_compressed_chunk_ptrs);
  NVCOMP_CHECK_ALIGNMENT(device_compressed_chunk_bytes);
  NVCOMP_CHECK_ALIGNMENT(device_uncompressed_chunk_bytes);

  return nvcomp::cascadedGetDecompressSizeAsyncDispatch(
    device_compressed_chunk_ptrs,
    device_compressed_chunk_bytes,
    device_uncompressed_chunk_bytes,
    num_chunks,
    stream
  );
}
