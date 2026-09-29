/*
 * Copyright (c) 2017-2026, NVIDIA CORPORATION. All rights reserved.
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

#include <cassert>
#include <iostream>
#include <list>
#include <map>
#include <mutex>
#include <sstream>
#include <vector>

#include "ans/ans.h"
#include "ans/constants.hpp"
#include "Check.h"
#include "common.h"
#include "CudaUtils.h"
#include "Logging.h"
#include "lowlevel/nvcomp_private.h"
#include "nvcomp.h"
#include "nvcomp/ans.h"
#include "type_macros.h"

using nvcomp::Check;
using nvcomp::CudaUtils;

static_assert(sizeof(nvcompBatchedANSCompressOpts_t) == 64);
static_assert(sizeof(nvcompBatchedANSDecompressOpts_t) == 64);

namespace
{

// TODO(mpayrits): The analog of this anonymous namespace appears in all
// *Batch.cpp files. We could unify this and use C++ templates with
// customization points.
// In fact, most of these Batch files could be reduced in such a way. It might
// be worthwhile to turn things upside down and maintain a highly templated
// non-redundant C++ high-level implementation that is thinly wrapped by both
// the HLIF _and_ the LLIF.

constexpr const char *ans_compress_opts_format_str = "{{type={:d}, data_type={:d}}}";
constexpr const char *ans_decompress_opts_format_str = "{{backend={:d}, data_type={:d}}}";

#define ANS_WITH_TRAILING_COMPRESS_OPTS_ARGS(callable, opts, ...)                                                      \
  callable(__VA_ARGS__, int(opts.type), int(opts.data_type))

#define ANS_WITH_TRAILING_DECOMPRESS_OPTS_ARGS(callable, opts, ...)                                                    \
  callable(__VA_ARGS__, int(opts.backend), int(opts.data_type))

#define ANS_LOG_WITH_COMPRESS_OPTS(log_func, opts, ...)                                                                \
  ANS_WITH_TRAILING_COMPRESS_OPTS_ARGS(log_func, opts, __func__, __VA_ARGS__, ans_compress_opts_format_str)

#define ANS_LOG_WITH_DECOMPRESS_OPTS(log_func, opts, ...)                                                              \
  ANS_WITH_TRAILING_DECOMPRESS_OPTS_ARGS(log_func, opts, __func__, __VA_ARGS__, ans_decompress_opts_format_str)

#define ANS_CHECK_COMPRESS_OPTS(opts) NVCOMP_WRAP_CHECK_FUNC(check_compress_opts, opts)

#define ANS_CHECK_DECOMPRESS_OPTS(opts) NVCOMP_WRAP_CHECK_FUNC(check_decompress_opts, opts)

// Valid when 0 (auto) or a power of 2 in [4, 64].
static bool is_valid_max_sub_chunk_count(unsigned max_sub_chunk_count)
{
  if (max_sub_chunk_count == 0)
  {
    return true;
  }
  if (max_sub_chunk_count < ans_gpu_lib::MIN_SUB_CHUNKS_PER_CHUNK ||
      max_sub_chunk_count > ans_gpu_lib::MAX_SUB_CHUNKS_PER_CHUNK)
  {
    return false;
  }
  // Must be a power of 2
  return (max_sub_chunk_count & (max_sub_chunk_count - 1)) == 0;
}

nvcompStatus_t check_compress_opts(const nvcompBatchedANSCompressOpts_t &opts)
{
  if (!is_valid_max_sub_chunk_count(opts.max_sub_chunk_count))
  {
    LOG_ERROR(
      "max_sub_chunk_count must be 0 (auto) or a power-of-2 in [{}, {}], got {}",
      ans_gpu_lib::MIN_SUB_CHUNKS_PER_CHUNK,
      ans_gpu_lib::MAX_SUB_CHUNKS_PER_CHUNK,
      opts.max_sub_chunk_count
    );
    return nvcompErrorInvalidValue;
  }

  const bool supported = opts.type == nvcomp_rANS &&
                         (opts.data_type == NVCOMP_TYPE_CHAR || opts.data_type == NVCOMP_TYPE_UCHAR ||
                          opts.data_type == NVCOMP_TYPE_FLOAT16 || opts.data_type == NVCOMP_TYPE_FLOAT8_E4M3);
  if (supported)
  {
    return nvcompSuccess;
  }

  ANS_WITH_TRAILING_COMPRESS_OPTS_ARGS(
    LOG_ERROR,
    opts,
    std::string("Unsupported options: ") + ans_compress_opts_format_str
  );
  return nvcompErrorNotSupported;
}

nvcompStatus_t check_decompress_opts(const nvcompBatchedANSDecompressOpts_t &opts)
{
  if (!is_valid_max_sub_chunk_count(opts.max_sub_chunk_count))
  {
    LOG_ERROR(
      "max_sub_chunk_count must be 0 (auto) or a power-of-2 in [{}, {}], got {}",
      ans_gpu_lib::MIN_SUB_CHUNKS_PER_CHUNK,
      ans_gpu_lib::MAX_SUB_CHUNKS_PER_CHUNK,
      opts.max_sub_chunk_count
    );
    return nvcompErrorInvalidValue;
  }

  const bool data_type_supported = opts.data_type == NVCOMP_TYPE_CHAR || opts.data_type == NVCOMP_TYPE_UCHAR ||
                                   opts.data_type == NVCOMP_TYPE_FLOAT16 || opts.data_type == NVCOMP_TYPE_FLOAT8_E4M3;
  if (!data_type_supported)
  {
    LOG_ERROR("Unsupported decompression data_type: {}", int(opts.data_type));
    return nvcompErrorNotSupported;
  }

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
      LOG_ERROR("Hardware backend is not available for ANS.");
      return nvcompErrorInvalidValue;
    default:
      LOG_ERROR(
        "Decompress backend invalid. Choose either NVCOMP_DECOMPRESS_BACKEND_DEFAULT or "
        "NVCOMP_DECOMPRESS_BACKEND_CUDA."
      );
      return nvcompErrorInvalidValue;
  }

  // Other errors
  ANS_WITH_TRAILING_DECOMPRESS_OPTS_ARGS(
    LOG_ERROR,
    opts,
    std::string("Unsupported options: ") + ans_decompress_opts_format_str
  );
  return nvcompErrorNotSupported;
}

} // namespace

nvcompStatus_t nvcompBatchedANSDecompressGetRequiredAlignments(
  nvcompBatchedANSDecompressOpts_t decompress_opts,
  nvcompAlignmentRequirements_t *alignment_requirements
)
{
  ANS_LOG_WITH_DECOMPRESS_OPTS(
    nvcomp::logBatchedDecompressGetRequiredAlignments,
    decompress_opts,
    alignment_requirements
  );
  ANS_CHECK_DECOMPRESS_OPTS(decompress_opts);
  NVCOMP_CHECK_NOT_NULL(alignment_requirements);

  // TODO: Values were copied from our public header. Provide reasoning.
  alignment_requirements->input = 8;
  // The decompressed output is a typed array, so enforce its natural alignment.
  // NVCOMP_TYPE_CHAR requests that the element type be auto-detected from the
  // bitstream, so the true type is unknown here; in that case fall back to the
  // alignment of the largest type: FP16.
  if (decompress_opts.data_type == NVCOMP_TYPE_CHAR)
  {
    alignment_requirements->output = alignof(uint16_t);
  }
  else
  {
    alignment_requirements->output = nvcomp::sizeOfnvcompType(decompress_opts.data_type);
  }
  // Decompress is fully fused and uses no temp scratch, so it imposes no temp alignment.
  alignment_requirements->temp = 1;

  static_assert(
    ((nvcompANSRequiredDecompressionAlignment) & (nvcompANSRequiredDecompressionAlignment - 1)) == 0,
    "Minimum decompression alignment is invalid."
  );
  assert(
    std::max(std::max(alignment_requirements->input, alignment_requirements->output), alignment_requirements->temp) <=
    nvcompANSRequiredDecompressionAlignment
  );

  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedANSDecompressGetTempSizeAsync(
  const size_t num_chunks,
  const size_t max_uncompressed_chunk_bytes,
  nvcompBatchedANSDecompressOpts_t decompress_opts,
  size_t *const temp_bytes,
  const size_t max_total_uncompressed_bytes
) // unused, except for logging
{
  ANS_LOG_WITH_DECOMPRESS_OPTS(
    nvcomp::logBatchedDecompressGetTempSizeAsync,
    decompress_opts,
    num_chunks,
    max_uncompressed_chunk_bytes,
    temp_bytes,
    max_total_uncompressed_bytes
  );

  ANS_CHECK_DECOMPRESS_OPTS(decompress_opts);
  NVCOMP_CHECK_NOT_NULL(temp_bytes);
  NVCOMP_CHECK_ALIGNMENT(temp_bytes);
  NVCOMP_CHECK_CHUNK_SIZE(max_uncompressed_chunk_bytes, nvcompANSDecompressionMaxAllowedChunkSize);

  try
  {
    ans::decompressGetTempSize(num_chunks, max_uncompressed_chunk_bytes, temp_bytes);
  }
  catch (const std::exception &e)
  {
    LOG_ERROR("{}", e.what());
    return Check::exception_to_error(e, "nvcompBatchedANSDecompressGetTempSizeAsync()");
  }
  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedANSDecompressGetTempSizeSync(
  [[maybe_unused]] const void *const *const device_compressed_chunk_ptrs,
  [[maybe_unused]] const size_t *const device_compressed_chunk_bytes,
  size_t num_chunks,
  size_t max_uncompressed_chunk_bytes,
  size_t *temp_bytes,
  size_t max_total_uncompressed_bytes,
  nvcompBatchedANSDecompressOpts_t decompress_opts,
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

  return nvcompBatchedANSDecompressGetTempSizeAsync(
    num_chunks,
    max_uncompressed_chunk_bytes,
    decompress_opts,
    temp_bytes,
    max_total_uncompressed_bytes
  );
}

nvcompStatus_t nvcompBatchedANSDecompressAsync(
  const void *const *device_compressed_chunk_ptrs,
  const size_t *device_compressed_chunk_bytes,
  const size_t *device_uncompressed_buffer_bytes,
  size_t *device_uncompressed_chunk_bytes,
  size_t num_chunks,
  void *const device_temp_ptr,
  const size_t temp_bytes,
  void *const *device_uncompressed_chunk_ptrs,
  nvcompBatchedANSDecompressOpts_t decompress_opts,
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
  if (temp_bytes > 0)
  {
    NVCOMP_CHECK_NOT_NULL(device_temp_ptr);
  }
  NVCOMP_CHECK_NOT_NULL(device_uncompressed_chunk_ptrs);
  if ((device_uncompressed_chunk_bytes == nullptr) != (device_statuses == nullptr))
  {
    LOG_ERROR("Both device_actual_uncompressed_bytes and device_statuses should be valid or nullptr");
    return nvcompErrorInvalidValue;
  }
  ANS_CHECK_DECOMPRESS_OPTS(decompress_opts);

  nvcompAlignmentRequirements_t align_reqs{};
  NVCOMP_WRAP_CHECK_FUNC(nvcompBatchedANSDecompressGetRequiredAlignments, decompress_opts, &align_reqs);
  cuLibLogger::Logger::Instance().SetMostRecentApi(__func__);

  // Check device pointer alignment
  NVCOMP_CHECK_ALIGNMENT(device_compressed_chunk_ptrs);
  NVCOMP_CHECK_ALIGNMENT(device_compressed_chunk_bytes);
  NVCOMP_CHECK_ALIGNMENT(device_uncompressed_buffer_bytes);
  NVCOMP_CHECK_ALIGNMENT(device_uncompressed_chunk_bytes);
  NVCOMP_CHECK_ALIGNMENT(device_temp_ptr, align_reqs.temp);
  NVCOMP_CHECK_ALIGNMENT(device_uncompressed_chunk_ptrs);
  NVCOMP_CHECK_ALIGNMENT(device_statuses);

  // Batch size is used as grid size, which CUDA limits to 2^31 - 1
  NVCOMP_CHECK_BATCH_SIZE(num_chunks, std::numeric_limits<int>::max());

  try
  {
    LOG_INFO("Launching SM decompression");
    ans::decompressAsync(
      device_compressed_chunk_ptrs,
      device_compressed_chunk_bytes,
      device_uncompressed_buffer_bytes,
      device_uncompressed_chunk_bytes,
      num_chunks,
      device_temp_ptr,
      temp_bytes,
      device_uncompressed_chunk_ptrs,
      device_statuses,
      decompress_opts.max_sub_chunk_count,
      decompress_opts.data_type,
      stream
    );
  }
  catch (const std::exception &e)
  {
    return Check::exception_to_error(e, "nvcompBatchedANSDecompressAsync()");
  }
  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedANSDecompressAsyncEx(
  const void *const *device_compressed_chunk_ptrs,
  const size_t *device_compressed_chunk_bytes,
  const size_t *device_uncompressed_buffer_bytes,
  size_t *device_uncompressed_chunk_bytes,
  size_t num_chunks,
  void *const device_temp_ptr,
  size_t temp_bytes,
  void *const *device_uncompressed_chunk_ptrs,
  nvcompStatus_t *device_statuses,
  nvcompBatchedANSCompressOpts_t compress_opts,
  nvcompBatchedANSDecompressOpts_t decompress_opts,
  cudaStream_t stream,
  [[maybe_unused]] const void *const *host_comp_chunk_buffers
)
{
  if (decompress_opts.max_sub_chunk_count == 0 && compress_opts.max_sub_chunk_count != 0)
  {
    decompress_opts.max_sub_chunk_count = compress_opts.max_sub_chunk_count;
  }
  // Default decompress data_type to the compress data_type.
  if (decompress_opts.data_type == NVCOMP_TYPE_CHAR && compress_opts.data_type != NVCOMP_TYPE_CHAR)
  {
    decompress_opts.data_type = compress_opts.data_type;
  }
  return nvcompBatchedANSDecompressAsync(
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

nvcompStatus_t nvcompBatchedANSCompressGetRequiredAlignments(
  nvcompBatchedANSCompressOpts_t compress_opts,
  nvcompAlignmentRequirements_t *alignment_requirements
)
{
  ANS_LOG_WITH_COMPRESS_OPTS(nvcomp::logBatchedCompressGetRequiredAlignments, compress_opts, alignment_requirements);
  ANS_CHECK_COMPRESS_OPTS(compress_opts);
  NVCOMP_CHECK_NOT_NULL(alignment_requirements);

  // CHAR/UCHAR require 4-byte input alignment; the encoder selects uchar4 or uint4
  // staging according to whether the pointer also satisfies 16-byte alignment.
  // FP16/FP8 require uint2 (8-byte) alignment and handle the 8-mod-16 case explicitly.
  // Providing 16-byte-aligned input can improve performance for every mode.
  // Output alignment is stipulated by the first `size_t` integer present in the compressed bitstream.
  switch (compress_opts.data_type)
  {
    case NVCOMP_TYPE_CHAR:
    case NVCOMP_TYPE_UCHAR:
      alignment_requirements->input = 4;
      break;
    case NVCOMP_TYPE_FLOAT16:
    case NVCOMP_TYPE_FLOAT8_E4M3:
      alignment_requirements->input = sizeof(uint2);
      break;
    default:
      // ANS_CHECK_COMPRESS_OPTS should handle this, and we should not end up here.
      assert(0);
      break;
  }
  // Output is cast to size_t* in prepare_encoding_table.
  alignment_requirements->output = sizeof(size_t);
  // Compression uses no temporary storage.
  alignment_requirements->temp = 1;

  static_assert(
    ((nvcompANSRequiredCompressionAlignment) & (nvcompANSRequiredCompressionAlignment - 1)) == 0,
    "Minimum compression alignment is invalid."
  );
  assert(
    std::max(std::max(alignment_requirements->input, alignment_requirements->output), alignment_requirements->temp) <=
    nvcompANSRequiredCompressionAlignment
  );

  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedANSCompressGetTempSizeAsync(
  size_t num_chunks,
  size_t max_uncompressed_chunk_bytes,
  nvcompBatchedANSCompressOpts_t compress_opts,
  size_t *temp_bytes,
  size_t max_total_uncompressed_bytes
) // unused, except for logging
{
  ANS_LOG_WITH_COMPRESS_OPTS(
    nvcomp::logBatchedCompressGetTempSizeAsync,
    compress_opts,
    num_chunks,
    max_uncompressed_chunk_bytes,
    temp_bytes,
    max_total_uncompressed_bytes
  );
  ANS_CHECK_COMPRESS_OPTS(compress_opts);

  NVCOMP_CHECK_NOT_NULL(temp_bytes);
  NVCOMP_CHECK_ALIGNMENT(temp_bytes);
  NVCOMP_CHECK_CHUNK_SIZE(max_uncompressed_chunk_bytes, nvcompANSCompressionMaxAllowedChunkSize);

  try
  {
    ans::compressGetTempSize(num_chunks, max_uncompressed_chunk_bytes, compress_opts, temp_bytes);
  }
  catch (const std::exception &e)
  {
    LOG_ERROR("{}", e.what());
    return Check::exception_to_error(e, "nvcompBatchedANSCompressGetTempSizeAsync()");
  }

  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedANSCompressGetTempSizeSync(
  [[maybe_unused]] const void *const *const device_uncompressed_chunk_ptrs,
  [[maybe_unused]] const size_t *const device_uncompressed_chunk_bytes,
  size_t num_chunks,
  size_t max_uncompressed_chunk_bytes,
  nvcompBatchedANSCompressOpts_t compress_opts,
  size_t *temp_bytes,
  size_t max_total_uncompressed_bytes,
  [[maybe_unused]] cudaStream_t stream
)
{
  return nvcompBatchedANSCompressGetTempSizeAsync(
    num_chunks,
    max_uncompressed_chunk_bytes,
    compress_opts,
    temp_bytes,
    max_total_uncompressed_bytes
  );
}

nvcompStatus_t nvcompBatchedANSCompressGetMaxOutputChunkSize(
  size_t max_uncompressed_chunk_bytes,
  nvcompBatchedANSCompressOpts_t compress_opts,
  size_t *max_compressed_chunk_bytes
)
{
  nvcomp::logBatchedCompressGetMaxOutputChunkSize(__func__, max_uncompressed_chunk_bytes, max_compressed_chunk_bytes);

  ANS_CHECK_COMPRESS_OPTS(compress_opts);

  NVCOMP_CHECK_NOT_NULL(max_compressed_chunk_bytes);
  NVCOMP_CHECK_ALIGNMENT(max_compressed_chunk_bytes);

  NVCOMP_CHECK_CHUNK_SIZE(max_uncompressed_chunk_bytes, nvcompANSCompressionMaxAllowedChunkSize);

  try
  {
    ans::compressGetMaxOutputChunkSize(max_uncompressed_chunk_bytes, max_compressed_chunk_bytes);
  }
  catch (const std::exception &e)
  {
    LOG_ERROR("{}", e.what());
    return Check::exception_to_error(e, "nvcompBatchedANSCompressGetMaxOutputChunkSize()");
  }

  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedANSCompressAsync(
  const void *const *device_uncompressed_chunk_ptrs,
  const size_t *device_uncompressed_chunk_bytes,
  size_t max_uncompressed_chunk_bytes,
  size_t num_chunks,
  void *device_temp_ptr,
  size_t temp_bytes,
  void *const *device_compressed_chunk_ptrs,
  size_t *device_compressed_chunk_bytes,
  nvcompBatchedANSCompressOpts_t format_opts,
  nvcompStatus_t *device_statuses, // optional
  cudaStream_t stream
)
{
  ANS_LOG_WITH_COMPRESS_OPTS(
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
  ANS_CHECK_COMPRESS_OPTS(format_opts);

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
  NVCOMP_WRAP_CHECK_FUNC(nvcompBatchedANSCompressGetRequiredAlignments, format_opts, &align_reqs);
  cuLibLogger::Logger::Instance().SetMostRecentApi(__func__);

  // Check device pointer alignment
  NVCOMP_CHECK_ALIGNMENT(device_uncompressed_chunk_ptrs);
  NVCOMP_CHECK_ALIGNMENT(device_uncompressed_chunk_bytes);
  NVCOMP_CHECK_ALIGNMENT(device_temp_ptr, align_reqs.temp);
  NVCOMP_CHECK_ALIGNMENT(device_compressed_chunk_ptrs);
  NVCOMP_CHECK_ALIGNMENT(device_compressed_chunk_bytes);
  NVCOMP_CHECK_ALIGNMENT(device_statuses);

  NVCOMP_CHECK_CHUNK_SIZE(max_uncompressed_chunk_bytes, nvcompANSCompressionMaxAllowedChunkSize);

  // Batch size is used as grid size, which CUDA limits to 2^31 - 1
  NVCOMP_CHECK_BATCH_SIZE(num_chunks, std::numeric_limits<int>::max());

  try
  {
    ans::compressAsync(
      device_uncompressed_chunk_ptrs,
      device_uncompressed_chunk_bytes,
      max_uncompressed_chunk_bytes,
      num_chunks,
      device_temp_ptr,
      temp_bytes,
      device_compressed_chunk_ptrs,
      device_compressed_chunk_bytes,
      format_opts,
      device_statuses,
      stream
    );
  }
  catch (const std::exception &e)
  {
    LOG_ERROR("{}", e.what());
    return Check::exception_to_error(e, "nvcompBatchedANSCompressAsync()");
  }
  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedANSGetDecompressSizeAsync(
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

  // Check device pointer is not NULL
  NVCOMP_CHECK_NOT_NULL(device_compressed_chunk_ptrs);
  NVCOMP_CHECK_NOT_NULL(device_compressed_chunk_bytes);
  NVCOMP_CHECK_NOT_NULL(device_uncompressed_chunk_bytes);

  // Check device pointer alignment
  NVCOMP_CHECK_ALIGNMENT(device_compressed_chunk_ptrs);
  NVCOMP_CHECK_ALIGNMENT(device_compressed_chunk_bytes);
  NVCOMP_CHECK_ALIGNMENT(device_uncompressed_chunk_bytes);

  try
  {
    ans::getDecompressSizeAsync(
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
    return Check::exception_to_error(e, "nvcompBatchedANSGetDecompressSizeAsync()");
  }
  return nvcompSuccess;
}
