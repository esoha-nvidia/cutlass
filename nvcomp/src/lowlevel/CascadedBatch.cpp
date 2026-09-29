/*
 * Copyright (c) 2021-2026, NVIDIA CORPORATION. All rights reserved.
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

#include "cascaded/composite/composite_constants.cuh"
#include "common.h"
#include "Logging.h"
#include "lowlevel/Check.h"
#include "lowlevel/nvcomp_private.h"
#include "nvcomp.h"
#include "nvcomp/cascaded.h"
#include "nvcomp/utils.hpp"
#include "type_macros.h"

using nvcomp::Check;
using nvcomp::isAligned;
using nvcomp::roundUpTo;

namespace
{

constexpr const char *cascaded_compress_opts_format_str = "{{num_RLEs={:d}, num_deltas={:d}}}";
constexpr const char *cascaded_decompress_opts_format_str = "{{backend={:d}}}";

#define CASCADED_WITH_TRAILING_COMPRESS_OPTS_ARGS(callable, opts, ...)                                                 \
  callable(__VA_ARGS__, opts.internal_chunk_bytes, int(opts.data_type), opts.num_RLEs, opts.num_deltas, opts.use_bp)

#define CASCADED_WITH_TRAILING_DECOMPRESS_OPTS_ARGS(callable, opts, ...) callable(__VA_ARGS__, int(opts.backend))

#define CASCADED_LOG_WITH_COMPRESS_OPTS(log_func, opts, ...)                                                           \
  CASCADED_WITH_TRAILING_COMPRESS_OPTS_ARGS(log_func, opts, __func__, __VA_ARGS__, cascaded_compress_opts_format_str)

#define CASCADED_LOG_WITH_DECOMPRESS_OPTS(log_func, opts, ...)                                                         \
  CASCADED_WITH_TRAILING_DECOMPRESS_OPTS_ARGS(                                                                         \
    log_func,                                                                                                          \
    opts,                                                                                                              \
    __func__,                                                                                                          \
    __VA_ARGS__,                                                                                                       \
    cascaded_decompress_opts_format_str                                                                                \
  )

#define CASCADED_CHECK_COMPRESS_OPTS(opts) NVCOMP_WRAP_CHECK_FUNC(check_compress_opts, opts)

#define CASCADED_CHECK_DECOMPRESS_OPTS(opts) NVCOMP_WRAP_CHECK_FUNC(check_decompress_opts, opts)

nvcompStatus_t check_compress_opts(const nvcompBatchedCascadedCompressOpts_t &opts)
{
  // TODO: we should also check the validity of internal_chunk_bytes.
  const bool is_valid_type = opts.data_type == NVCOMP_TYPE_CHAR || opts.data_type == NVCOMP_TYPE_UCHAR ||
                             opts.data_type == NVCOMP_TYPE_SHORT || opts.data_type == NVCOMP_TYPE_USHORT ||
                             opts.data_type == NVCOMP_TYPE_INT || opts.data_type == NVCOMP_TYPE_UINT ||
                             opts.data_type == NVCOMP_TYPE_LONGLONG || opts.data_type == NVCOMP_TYPE_ULONGLONG;
  const bool supported = opts.num_RLEs >= 0 && opts.num_RLEs <= composite::max_num_rle_layers && opts.num_deltas >= 0 &&
                         opts.num_deltas <= composite::max_num_delta_layers && (opts.use_bp == 0 || opts.use_bp == 1) &&
                         is_valid_type;
  if (supported)
  {
    return nvcompSuccess;
  }

  CASCADED_WITH_TRAILING_COMPRESS_OPTS_ARGS(
    LOG_ERROR,
    opts,
    std::string("Unsupported options: ") + cascaded_compress_opts_format_str
  );
  return nvcompErrorNotSupported;
}

nvcompStatus_t check_decompress_opts(const nvcompBatchedCascadedDecompressOpts_t &opts)
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
      LOG_ERROR("Hardware backend is not available for Cascaded.");
      return nvcompErrorInvalidValue;
    default:
      LOG_ERROR(
        "Decompress backend invalid. Choose either NVCOMP_DECOMPRESS_BACKEND_DEFAULT or NVCOMP_DECOMPRESS_BACKEND_CUDA."
      );
      return nvcompErrorInvalidValue;
  }

  // Other errors
  CASCADED_WITH_TRAILING_DECOMPRESS_OPTS_ARGS(
    LOG_ERROR,
    opts,
    std::string("Unsupported options: ") + cascaded_decompress_opts_format_str
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
nvcompStatus_t cascadedCompressAsyncPart2(
  const void *const *device_uncompressed_chunk_ptrs,
  const size_t *device_uncompressed_chunk_bytes,
  size_t num_chunks,
  void *const *device_compressed_chunk_ptrs,
  size_t *device_compressed_chunk_bytes,
  const nvcompBatchedCascadedCompressOpts_t format_opts,
  nvcompStatus_t *device_statuses,
  cudaStream_t stream
);
nvcompStatus_t cascadedDecompressAsyncPart2(
  const void *const *device_compressed_chunk_ptrs,
  const size_t *device_compressed_chunk_bytes,
  const size_t *device_uncompressed_buffer_bytes,
  size_t *device_uncompressed_chunk_bytes,
  size_t num_chunks,
  void *const *device_uncompressed_chunk_ptrs,
  nvcompStatus_t *device_statuses,
  cudaStream_t stream
);
nvcompStatus_t cascadedDecompressAsyncPart2(
  const void *const *device_compressed_chunk_ptrs,
  const size_t *device_compressed_chunk_bytes,
  const size_t *device_uncompressed_buffer_bytes,
  size_t *device_uncompressed_chunk_bytes,
  size_t num_chunks,
  void *const *device_uncompressed_chunk_ptrs,
  nvcompStatus_t *device_statuses,
  nvcompType_t type,
  cudaStream_t stream
);
nvcompStatus_t cascadedGetDecompressSizeAsyncPart2(
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
  alignment_requirements->input = nvcomp::sizeOfnvcompType(format_opts.data_type);
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

nvcompStatus_t nvcompBatchedCascadedCompressGetTempSizeAsync(
  size_t num_chunks, // unused, except for logging
  size_t max_uncompressed_chunk_bytes, // unused, except for logging
  nvcompBatchedCascadedCompressOpts_t compress_opts,
  size_t *temp_bytes,
  size_t max_total_uncompressed_bytes
) // unused, except for logging
{
  CASCADED_LOG_WITH_COMPRESS_OPTS(
    nvcomp::logBatchedCompressGetTempSizeAsync,
    compress_opts,
    num_chunks,
    max_uncompressed_chunk_bytes,
    temp_bytes,
    max_total_uncompressed_bytes
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
  [[maybe_unused]] cudaStream_t stream
)
{
  return nvcompBatchedCascadedCompressGetTempSizeAsync(
    num_chunks,
    max_uncompressed_chunk_bytes,
    compress_opts,
    temp_bytes,
    max_total_uncompressed_bytes
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

  *max_compressed_chunk_bytes = roundUpTo(max_uncompressed_chunk_bytes, 4) + 8;

  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedCascadedCompressAsync(
  const void *const *device_uncompressed_chunk_ptrs,
  const size_t *device_uncompressed_chunk_bytes,
  size_t max_uncompressed_chunk_bytes, // not used
  size_t num_chunks,
  void *device_temp_ptr, // not used
  size_t temp_bytes, // not used
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

  // Check device pointer alignment
  NVCOMP_CHECK_ALIGNMENT(device_uncompressed_chunk_ptrs);
  NVCOMP_CHECK_ALIGNMENT(device_uncompressed_chunk_bytes);
  NVCOMP_CHECK_ALIGNMENT(device_compressed_chunk_ptrs);
  NVCOMP_CHECK_ALIGNMENT(device_compressed_chunk_bytes);
  NVCOMP_CHECK_ALIGNMENT(device_statuses);

  // Batch size is used as grid size, which CUDA limits to 2^31 - 1
  NVCOMP_CHECK_BATCH_SIZE(num_chunks, std::numeric_limits<int>::max());

  return nvcomp::cascadedCompressAsyncPart2(
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
  // Output depends on the opts.data_type used during compression and its
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

nvcompStatus_t nvcompBatchedCascadedDecompressGetTempSizeAsync(
  size_t num_chunks, // unused, except for logging
  size_t max_uncompressed_chunk_bytes, // unused, except for logging
  nvcompBatchedCascadedDecompressOpts_t decompress_opts,
  size_t *temp_bytes,
  size_t max_total_uncompressed_bytes
) // unused, except for loggign
{
  CASCADED_LOG_WITH_DECOMPRESS_OPTS(
    nvcomp::logBatchedDecompressGetTempSizeAsync,
    decompress_opts,
    num_chunks,
    max_uncompressed_chunk_bytes,
    temp_bytes,
    max_total_uncompressed_bytes
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

  return nvcompBatchedCascadedDecompressGetTempSizeAsync(
    num_chunks,
    max_uncompressed_chunk_bytes,
    decompress_opts,
    temp_bytes,
    max_total_uncompressed_bytes
  );
}

nvcompStatus_t nvcompBatchedCascadedDecompressAsync(
  const void *const *device_compressed_chunk_ptrs,
  const size_t *device_compressed_chunk_bytes,
  const size_t *device_uncompressed_buffer_bytes,
  size_t *device_uncompressed_chunk_bytes,
  size_t num_chunks,
  void *const device_temp_ptr, // not used
  size_t temp_bytes, // not used
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
  // Note:
  // In contrast to other decompressors, Cascaded does not have checks for the absence of
  // `device_statuses` and `device_uncompressed_chunk_bytes`.
  NVCOMP_CHECK_NOT_NULL(device_uncompressed_chunk_bytes);
  NVCOMP_CHECK_NOT_NULL(device_statuses);

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
  return nvcomp::cascadedDecompressAsyncPart2(
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
  void *const /*device_temp_ptr*/,
  size_t /*temp_bytes*/,
  void *const *device_uncompressed_chunk_ptrs,
  nvcompStatus_t *device_statuses,
  nvcompBatchedCascadedCompressOpts_t compress_opts,
  nvcompBatchedCascadedDecompressOpts_t /*decompress_opts*/,
  cudaStream_t stream,
  const void *const * /*host_comp_chunk_buffers*/
)
{
  return nvcomp::cascadedDecompressAsyncPart2(
    device_compressed_chunk_ptrs,
    device_compressed_chunk_bytes,
    device_uncompressed_buffer_bytes,
    device_uncompressed_chunk_bytes,
    num_chunks,
    device_uncompressed_chunk_ptrs,
    device_statuses,
    compress_opts.data_type,
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

  return nvcomp::cascadedGetDecompressSizeAsyncPart2(
    device_compressed_chunk_ptrs,
    device_compressed_chunk_bytes,
    device_uncompressed_chunk_bytes,
    num_chunks,
    stream
  );
}
