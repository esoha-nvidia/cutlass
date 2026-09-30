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

constexpr const char *ans_compress_opts_format_str =
  "{{type={:d}, data_type={:d}, max_sub_chunk_count={:d}, states_per_lane={:d}, histogram_reduction_log2={:d}}}";
constexpr const char *ans_decompress_opts_format_str =
  "{{backend={:d}, data_type={:d}, max_sub_chunk_count={:d}, states_per_lane={:d}, skip_validate={:d}}}";

#define ANS_WITH_TRAILING_COMPRESS_OPTS_ARGS(callable, opts, ...)                                                      \
  callable(                                                                                                            \
    __VA_ARGS__,                                                                                                       \
    int(opts.type),                                                                                                    \
    int(opts.data_type),                                                                                               \
    int(opts.max_sub_chunk_count),                                                                                     \
    int(opts.states_per_lane),                                                                                         \
    int(opts.histogram_reduction_log2)                                                                                 \
  )

#define ANS_WITH_TRAILING_DECOMPRESS_OPTS_ARGS(callable, opts, ...)                                                    \
  callable(                                                                                                            \
    __VA_ARGS__,                                                                                                       \
    int(opts.backend),                                                                                                 \
    int(opts.data_type),                                                                                               \
    int(opts.max_sub_chunk_count),                                                                                     \
    int(opts.states_per_lane),                                                                                         \
    int(opts.skip_validate)                                                                                            \
  )

#define ANS_LOG_WITH_COMPRESS_OPTS(log_func, opts, ...)                                                                \
  ANS_WITH_TRAILING_COMPRESS_OPTS_ARGS(log_func, opts, __func__, __VA_ARGS__, ans_compress_opts_format_str)

#define ANS_LOG_WITH_DECOMPRESS_OPTS(log_func, opts, ...)                                                              \
  ANS_WITH_TRAILING_DECOMPRESS_OPTS_ARGS(log_func, opts, __func__, __VA_ARGS__, ans_decompress_opts_format_str)

#define ANS_CHECK_COMPRESS_OPTS(opts) NVCOMP_WRAP_CHECK_FUNC(check_compress_opts, opts)

#define ANS_CHECK_DECOMPRESS_OPTS(opts) NVCOMP_WRAP_CHECK_FUNC(check_decompress_opts, opts)

// Valid when 0 or a power of 2 in [4, 64]. Compress 0 means default 8; decompress 0
// uses whatever compression used.
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

// 0 = auto, 1 = single stream, 2 = two streams.
static bool is_valid_states_per_lane(unsigned states_per_lane)
{
  switch (states_per_lane)
  {
    case 0:
    case 1:
    case 2:
      return true;
    default:
      return false;
  }
}

nvcompStatus_t check_compress_opts(const nvcompBatchedANSCompressOpts_t &opts)
{
  if (!is_valid_max_sub_chunk_count(opts.max_sub_chunk_count))
  {
    LOG_ERROR(
      "max_sub_chunk_count must be 0 (default 8) or a power-of-2 in [{}, {}], got {}",
      ans_gpu_lib::MIN_SUB_CHUNKS_PER_CHUNK,
      ans_gpu_lib::MAX_SUB_CHUNKS_PER_CHUNK,
      opts.max_sub_chunk_count
    );
    return nvcompErrorInvalidValue;
  }

  if (!is_valid_states_per_lane(opts.states_per_lane))
  {
    LOG_ERROR("states_per_lane must be 0 (auto), 1 (single stream), or 2 (two streams), got {}", opts.states_per_lane);
    return nvcompErrorInvalidValue;
  }

  if (opts.histogram_reduction_log2 > nvcompANSMaxHistogramReductionLog2)
  {
    LOG_ERROR(
      "histogram_reduction_log2 must be 0 (exact) through {} (keep 1/{} of each warp slice), got {}",
      nvcompANSMaxHistogramReductionLog2,
      1u << nvcompANSMaxHistogramReductionLog2,
      opts.histogram_reduction_log2
    );
    return nvcompErrorInvalidValue;
  }

  const bool supported = opts.type == nvcomp_rANS &&
                         (opts.data_type == NVCOMP_TYPE_CHAR || opts.data_type == NVCOMP_TYPE_UCHAR ||
                          opts.data_type == NVCOMP_TYPE_FLOAT16 || opts.data_type == NVCOMP_TYPE_FLOAT8_E4M3 ||
                          opts.data_type == NVCOMP_TYPE_FLOAT32);
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
      "max_sub_chunk_count must be 0 (use compression setting) or a power-of-2 in [{}, {}], got {}",
      ans_gpu_lib::MIN_SUB_CHUNKS_PER_CHUNK,
      ans_gpu_lib::MAX_SUB_CHUNKS_PER_CHUNK,
      opts.max_sub_chunk_count
    );
    return nvcompErrorInvalidValue;
  }

  if (!is_valid_states_per_lane(opts.states_per_lane))
  {
    LOG_ERROR("states_per_lane must be 0 (auto), 1 (single stream), or 2 (two streams), got {}", opts.states_per_lane);
    return nvcompErrorInvalidValue;
  }

  // BITS is the decompress-only "unknown" sentinel: decode whatever the bitstream says.
  // Compression has no such mode, so check_compress_opts still rejects it.
  const bool data_type_supported = opts.data_type == NVCOMP_TYPE_BITS || opts.data_type == NVCOMP_TYPE_CHAR ||
                                   opts.data_type == NVCOMP_TYPE_UCHAR || opts.data_type == NVCOMP_TYPE_FLOAT16 ||
                                   opts.data_type == NVCOMP_TYPE_FLOAT8_E4M3 || opts.data_type == NVCOMP_TYPE_FLOAT32;
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

  alignment_requirements->input = 16;
  // Uncompressed output is 16 B-aligned: fp16 block decoders store each lane's tile as
  // STG.128; fp8 uses STG.64.
  alignment_requirements->output = 16;
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

nvcompStatus_t nvcompBatchedANSDecompressGetTempSize(
  const size_t num_chunks,
  const size_t max_uncompressed_chunk_bytes,
  nvcompBatchedANSDecompressOpts_t decompress_opts,
  size_t *const temp_bytes,
  const size_t max_total_uncompressed_bytes, // unused, except for logging
  cudaStream_t stream // unused, except for logging
)
{
  ANS_LOG_WITH_DECOMPRESS_OPTS(
    nvcomp::logBatchedDecompressGetTempSize,
    decompress_opts,
    num_chunks,
    max_uncompressed_chunk_bytes,
    temp_bytes,
    max_total_uncompressed_bytes,
    stream
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
    return Check::exception_to_error(e, "nvcompBatchedANSDecompressGetTempSize()");
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

  return nvcompBatchedANSDecompressGetTempSize(
    num_chunks,
    max_uncompressed_chunk_bytes,
    decompress_opts,
    temp_bytes,
    max_total_uncompressed_bytes,
    stream
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
  NVCOMP_CHECK_NOT_NULL(device_uncompressed_chunk_bytes);
  NVCOMP_CHECK_NOT_NULL(device_statuses);
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
      decompress_opts.states_per_lane,
      decompress_opts.skip_validate,
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
  if (decompress_opts.max_sub_chunk_count == 0)
  {
    decompress_opts.max_sub_chunk_count = ans_gpu_lib::resolve_max_sub_chunk_count(compress_opts.max_sub_chunk_count);
  }
  // The caller handed us the compress opts, so an unknown decompress type can adopt the
  // concrete one instead of costing a Generic launch. Then resolve the state count from
  // the explicit decompress setting, the compress setting, or the type's default.
  if (decompress_opts.data_type == NVCOMP_TYPE_BITS)
  {
    decompress_opts.data_type = compress_opts.data_type;
  }
  decompress_opts.states_per_lane = ans::resolveDecompressStatesPerLane(
    decompress_opts.states_per_lane,
    compress_opts.states_per_lane,
    decompress_opts.data_type
  );
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

  // Uncompressed input is 16 B-aligned: the fp16 encoder reads each lane's tile as LDG.128.
  alignment_requirements->input = 16;
  // Compressed output is 16 B-aligned so sub-chunk slots stay uint4-aligned.
  alignment_requirements->output = 16;
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

nvcompStatus_t nvcompBatchedANSCompressGetTempSize(
  size_t num_chunks,
  size_t max_uncompressed_chunk_bytes,
  nvcompBatchedANSCompressOpts_t compress_opts,
  size_t *temp_bytes,
  size_t max_total_uncompressed_bytes, // unused, except for logging
  cudaStream_t stream // unused, except for logging
)
{
  ANS_LOG_WITH_COMPRESS_OPTS(
    nvcomp::logBatchedCompressGetTempSize,
    compress_opts,
    num_chunks,
    max_uncompressed_chunk_bytes,
    temp_bytes,
    max_total_uncompressed_bytes,
    stream
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
    return Check::exception_to_error(e, "nvcompBatchedANSCompressGetTempSize()");
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
  cudaStream_t stream
)
{
  return nvcompBatchedANSCompressGetTempSize(
    num_chunks,
    max_uncompressed_chunk_bytes,
    compress_opts,
    temp_bytes,
    max_total_uncompressed_bytes,
    stream
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
    ans::compressGetMaxOutputChunkSize(max_uncompressed_chunk_bytes, compress_opts, max_compressed_chunk_bytes);
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
