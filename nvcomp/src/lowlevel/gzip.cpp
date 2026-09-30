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

#include "nvcomp/gzip.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <mutex>
#include <sstream>
#include <vector>

#include "Check.h"
#include "common.h"
#include "CorrectnessChecks.cuh"
#include "Environment.hpp"
#include "exception.hpp"
#include "gdeflate/common.h"
#include "gdeflate/deflate.h"
#include "gdeflate/gdeflate.h"
#include "gzip/GzipConstants.cuh"
#include "gzip/GzipKernels.cuh"
#include "HWDecompress.hpp"
#include "Logging.h"
#include "lookahead_gzip/include/lookahead_gzip.h"
#include "lookahead_gzip/include/lookahead_gzip_client.h"
#include "lowlevel/nvcomp_private.h"
#include "nvcomp.h"
#include "nvcomp/utils.hpp"
#include "type_macros.h"

using namespace nvcomp;

static_assert(sizeof(nvcompBatchedGzipCompressOpts_t) == 64);
static_assert(sizeof(nvcompBatchedGzipDecompressOpts_t) == 64);

namespace
{

constexpr const char *gzip_compress_opts_format_str = "{{algorithm={:d}}}";
constexpr const char *gzip_decompress_opts_format_str = "{{backend={:d}, algorithm={:d}}}";

#define GZIP_WITH_TRAILING_COMPRESS_OPTS_ARGS(callable, opts, ...) callable(__VA_ARGS__, opts.algorithm)

#define GZIP_LOG_WITH_COMPRESS_OPTS(log_func, opts, ...)                                                               \
  GZIP_WITH_TRAILING_COMPRESS_OPTS_ARGS(log_func, opts, __func__, __VA_ARGS__, gzip_compress_opts_format_str)

#define GZIP_CHECK_COMPRESS_OPTS(opts) NVCOMP_WRAP_CHECK_FUNC(check_compress_opts, opts)

#define GZIP_WITH_TRAILING_DECOMPRESS_OPTS_ARGS(callable, opts, ...)                                                   \
  callable(__VA_ARGS__, int(opts.backend), int(opts.algorithm))

#define GZIP_LOG_WITH_DECOMPRESS_OPTS(log_func, opts, ...)                                                             \
  GZIP_WITH_TRAILING_DECOMPRESS_OPTS_ARGS(log_func, opts, __func__, __VA_ARGS__, gzip_decompress_opts_format_str)

#define GZIP_CHECK_DECOMPRESS_OPTS(opts) NVCOMP_WRAP_CHECK_FUNC(check_decompress_opts, opts)

nvcompStatus_t check_compress_opts(const nvcompBatchedGzipCompressOpts_t &opts)
{
  const bool supported = opts.algorithm >= 0 && opts.algorithm <= 5;
  if (supported)
  {
    return nvcompSuccess;
  }

  GZIP_WITH_TRAILING_COMPRESS_OPTS_ARGS(
    LOG_ERROR,
    opts,
    std::string("Unsupported options: ") + gzip_compress_opts_format_str
  );
  return nvcompErrorNotSupported;
}

nvcompStatus_t check_decompress_opts(const nvcompBatchedGzipDecompressOpts_t &opts)
{
  const bool backend_supported = opts.backend == NVCOMP_DECOMPRESS_BACKEND_DEFAULT ||
                                 opts.backend == NVCOMP_DECOMPRESS_BACKEND_CUDA ||
                                 opts.backend == NVCOMP_DECOMPRESS_BACKEND_HARDWARE;
  const bool algorithm_supported = opts.algorithm == NVCOMP_GZIP_DECOMPRESS_ALGORITHM_LOOKAHEAD ||
                                   opts.algorithm == NVCOMP_GZIP_DECOMPRESS_ALGORITHM_NAIVE;
  if (backend_supported && algorithm_supported)
  {
    if (opts.backend == NVCOMP_DECOMPRESS_BACKEND_HARDWARE && opts.sort_before_hw_decompress != 0 &&
        opts.sort_before_hw_decompress != 1)
    {
      LOG_ERROR("sort_before_hw_decompress must be 0 or 1, got {}.", opts.sort_before_hw_decompress);
      return nvcompErrorInvalidValue;
    }
    return nvcompSuccess;
  }

  // Other errors
  GZIP_WITH_TRAILING_DECOMPRESS_OPTS_ARGS(
    LOG_ERROR,
    opts,
    std::string("Unsupported options: ") + gzip_decompress_opts_format_str
  );
  return nvcompErrorNotSupported;
}

} // namespace

nvcompStatus_t nvcompBatchedGzipDecompressGetRequiredAlignments(
  nvcompBatchedGzipDecompressOpts_t decompress_opts,
  nvcompAlignmentRequirements_t *alignment_requirements
)
{
  GZIP_LOG_WITH_DECOMPRESS_OPTS(
    nvcomp::logBatchedDecompressGetRequiredAlignments,
    decompress_opts,
    alignment_requirements
  );
  GZIP_CHECK_DECOMPRESS_OPTS(decompress_opts);
  NVCOMP_CHECK_NOT_NULL(alignment_requirements);

  // Note: The decompress engine (DE) requires 1-byte alignment.

  if (decompress_opts.algorithm == NVCOMP_GZIP_DECOMPRESS_ALGORITHM_LOOKAHEAD)
  {
    NVCOMP_WRAP_CHECK_FUNC(nvcompLookaheadGzipDecompressGetRequiredAlignments, alignment_requirements);
  }
  else
  {
    // TODO: Values were copied from our public header. Provide reasoning.
    alignment_requirements->input = 1;
    alignment_requirements->output = 1;
    alignment_requirements->temp = 1;
  }

  static_assert(
    ((nvcompGzipRequiredDecompressionAlignment) & (nvcompGzipRequiredDecompressionAlignment - 1)) == 0,
    "Minimum decompression alignment is invalid."
  );
  assert(
    std::max(std::max(alignment_requirements->input, alignment_requirements->output), alignment_requirements->temp) <=
    nvcompGzipRequiredDecompressionAlignment
  );

  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedGzipDecompressGetTempSize(
  const size_t num_chunks,
  const size_t max_uncompressed_chunk_bytes,
  nvcompBatchedGzipDecompressOpts_t decompress_opts,
  size_t *const temp_bytes,
  const size_t max_total_uncompressed_bytes, // unused, except for logging
  cudaStream_t stream
)
{
  GZIP_LOG_WITH_DECOMPRESS_OPTS(
    nvcomp::logBatchedDecompressGetTempSize,
    decompress_opts,
    num_chunks,
    max_uncompressed_chunk_bytes,
    temp_bytes,
    max_total_uncompressed_bytes,
    stream
  );

  GZIP_CHECK_DECOMPRESS_OPTS(decompress_opts);
  NVCOMP_CHECK_NOT_NULL(temp_bytes);
  NVCOMP_CHECK_ALIGNMENT(temp_bytes);

  switch (decompress_opts.algorithm)
  {
    case NVCOMP_GZIP_DECOMPRESS_ALGORITHM_NAIVE:
      NVCOMP_CHECK_CHUNK_SIZE(max_uncompressed_chunk_bytes, nvcompGzipNaiveDecompressionMaxAllowedChunkSize);
      break;
    case NVCOMP_GZIP_DECOMPRESS_ALGORITHM_LOOKAHEAD:
      NVCOMP_CHECK_CHUNK_SIZE(max_uncompressed_chunk_bytes, nvcompGzipLookaheadDecompressionMaxAllowedChunkSize);
      break;
    default:
      assert(0);
  }

  try
  {
    if (decompress_opts.algorithm == NVCOMP_GZIP_DECOMPRESS_ALGORITHM_LOOKAHEAD)
    {
      lookaheadGzipConfig_t config;
      lookahead_gzip::LookaheadGzipOneshotClient::createConfig(num_chunks, stream, &config);
      lookahead_gzip::LookaheadGzipOneshotClient::decompressGetTempSize(&config, temp_bytes);
    }
    else
    {
      gdeflate::decompressGetTempSize(num_chunks, max_uncompressed_chunk_bytes, temp_bytes);

      if (decompress_opts.sort_before_hw_decompress)
      {
        *temp_bytes += get_sort_scratch_req(num_chunks, max_uncompressed_chunk_bytes);
      }
    }

    // Allocating temporary space for correctness checks
    const auto deflate_correctness_env = nvcomp::getenv(CHECK_DEFLATE_CORRECTNESS_ENV);
    const bool check_deflate_correctness = not(deflate_correctness_env.empty() or deflate_correctness_env == "0");
    if (check_deflate_correctness)
    {
      // If correctness checks are enabled, we need to allocate space for
      // the device status array.
      *temp_bytes += sizeof(nvcomp_deflate::DeflateCorrectnessChecker<true>) * num_chunks +
                     alignof(nvcomp_deflate::DeflateCorrectnessChecker<true>) - 1;
    }
  }
  catch (const std::exception &e)
  {
    LOG_ERROR("{}", e.what());
    return Check::exception_to_error(e, "nvcompBatchedGzipDecompressGetTempSize()");
  }
  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedGzipDecompressGetTempSizeSync(
  [[maybe_unused]] const void *const *const device_compressed_chunk_ptrs,
  [[maybe_unused]] const size_t *const device_compressed_chunk_bytes,
  size_t num_chunks,
  size_t max_uncompressed_chunk_bytes,
  size_t *temp_bytes,
  size_t max_total_uncompressed_bytes,
  nvcompBatchedGzipDecompressOpts_t decompress_opts,
  nvcompStatus_t *device_statuses,
  cudaStream_t stream
)
{
  NVCOMP_CHECK_ALIGNMENT(device_statuses);

  nvcompStatus_t result = try_clear_device_statuses<true>(num_chunks, device_statuses, stream);
  if (result != nvcompSuccess)
  {
    return result;
  }

  return nvcompBatchedGzipDecompressGetTempSize(
    num_chunks,
    max_uncompressed_chunk_bytes,
    decompress_opts,
    temp_bytes,
    max_total_uncompressed_bytes,
    stream
  );
}

nvcompStatus_t nvcompBatchedGzipDecompressAsync(
  const void *const *device_compressed_chunk_ptrs,
  const size_t *device_compressed_chunk_bytes,
  const size_t *device_uncompressed_buffer_bytes,
  size_t *device_uncompressed_chunk_bytes,
  size_t num_chunks,
  void *const device_temp_ptr,
  size_t temp_bytes,
  void *const *device_uncompressed_chunk_ptrs,
  nvcompBatchedGzipDecompressOpts_t decompress_opts,
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

  GZIP_CHECK_DECOMPRESS_OPTS(decompress_opts);

  // error check inputs
  NVCOMP_CHECK_NOT_NULL(device_compressed_chunk_ptrs);
  NVCOMP_CHECK_NOT_NULL(device_compressed_chunk_bytes);
  NVCOMP_CHECK_NOT_NULL(device_uncompressed_buffer_bytes);
  NVCOMP_CHECK_NOT_NULL(device_uncompressed_chunk_bytes);
  NVCOMP_CHECK_NOT_NULL(device_uncompressed_chunk_ptrs);
  NVCOMP_CHECK_NOT_NULL(device_statuses);

  try
  {
    if (decompress_opts.algorithm == NVCOMP_GZIP_DECOMPRESS_ALGORITHM_LOOKAHEAD)
    {
      return nvcompLookaheadGzipDecompressAsync(
        device_compressed_chunk_ptrs,
        device_compressed_chunk_bytes,
        device_uncompressed_chunk_ptrs,
        device_uncompressed_buffer_bytes,
        device_uncompressed_chunk_bytes,
        device_temp_ptr,
        temp_bytes,
        num_chunks,
        device_statuses,
        stream
      );
    }
    else
    {
      return nvcomp_deflate::DetailDeflateDecompress(
        device_compressed_chunk_ptrs,
        device_compressed_chunk_bytes,
        device_uncompressed_buffer_bytes,
        device_uncompressed_chunk_bytes,
        num_chunks,
        device_temp_ptr,
        temp_bytes,
        device_uncompressed_chunk_ptrs,
        device_statuses,
        nullptr,
        &decompress_opts,
        true, /*gzip*/
        stream
      );
    }
  }
  catch (const std::exception &e)
  {
    LOG_ERROR("{}", e.what());
    return Check::exception_to_error(e, "nvcompBatchedGzipDecompressAsync()");
  }
}

nvcompStatus_t nvcompBatchedGzipDecompressAsyncEx(
  const void *const *device_compressed_chunk_ptrs,
  const size_t *device_compressed_chunk_bytes,
  const size_t *device_uncompressed_buffer_bytes,
  size_t *device_uncompressed_chunk_bytes,
  size_t num_chunks,
  void *const device_temp_ptr,
  size_t temp_bytes,
  void *const *device_uncompressed_chunk_ptrs,
  nvcompStatus_t *device_statuses,
  [[maybe_unused]] nvcompBatchedGzipCompressOpts_t compress_opts,
  nvcompBatchedGzipDecompressOpts_t decompress_opts,
  cudaStream_t stream,
  const void *const *host_comp_chunk_buffers
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

  GZIP_CHECK_DECOMPRESS_OPTS(decompress_opts);

  // error check inputs
  NVCOMP_CHECK_NOT_NULL(device_compressed_chunk_ptrs);
  NVCOMP_CHECK_NOT_NULL(device_compressed_chunk_bytes);
  NVCOMP_CHECK_NOT_NULL(device_uncompressed_buffer_bytes);
  NVCOMP_CHECK_NOT_NULL(device_uncompressed_chunk_bytes);
  NVCOMP_CHECK_NOT_NULL(device_uncompressed_chunk_ptrs);
  NVCOMP_CHECK_NOT_NULL(device_statuses);
  if (temp_bytes > 0)
  {
    NVCOMP_CHECK_NOT_NULL(device_temp_ptr);
  }

  try
  {
    if (decompress_opts.algorithm == NVCOMP_GZIP_DECOMPRESS_ALGORITHM_LOOKAHEAD)
    {
      lookaheadGzipConfig_t config;
      lookahead_gzip::LookaheadGzipOneshotClient::createConfig(num_chunks, stream, &config);
      lookahead_gzip::LookaheadGzipOneshotClient
        client(&config, temp_bytes, reinterpret_cast<uint8_t *>(device_temp_ptr));

      client.decompress(
        reinterpret_cast<const uint32_t *const *>(device_compressed_chunk_ptrs),
        device_compressed_chunk_bytes,
        reinterpret_cast<char *const *>(device_uncompressed_chunk_ptrs),
        device_uncompressed_buffer_bytes,
        device_uncompressed_chunk_bytes,
        reinterpret_cast<uint8_t *>(device_temp_ptr),
        device_statuses,
        stream
      );
      return nvcompSuccess;
    }
    else
    {
      return nvcomp_deflate::DetailDeflateDecompress(
        device_compressed_chunk_ptrs,
        device_compressed_chunk_bytes,
        device_uncompressed_buffer_bytes,
        device_uncompressed_chunk_bytes,
        num_chunks,
        device_temp_ptr,
        temp_bytes,
        device_uncompressed_chunk_ptrs,
        device_statuses,
        nullptr,
        &decompress_opts,
        true, /*gzip*/
        stream,
        host_comp_chunk_buffers
      );
    }
  }
  catch (const std::exception &e)
  {
    LOG_ERROR("{}", e.what());
    return Check::exception_to_error(e, "nvcompBatchedGzipDecompressAsyncEx()");
  }
}

// We use the chunked GetDecompressSizeAsync function for both gzip algorithms here
// because the gzip header only returns the uncompressed size modulo INT_MAX, which is insufficient
// for accurately determining the size of large files.

// TODO: Investigate if a template parameter for lookahead gzip processing without
// writing output could perform size determination faster for small chunk.
nvcompStatus_t nvcompBatchedGzipGetDecompressSizeAsync(
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

  NVCOMP_CHECK_ALIGNMENT(device_compressed_chunk_ptrs);
  NVCOMP_CHECK_ALIGNMENT(device_compressed_chunk_bytes);
  NVCOMP_CHECK_ALIGNMENT(device_uncompressed_chunk_bytes);

  try
  {
    void *const *device_uncompressed_chunk_ptrs = nullptr;
    const size_t *device_uncompressed_buffer_bytes = nullptr;
    nvcomp_deflate::deflateStatus_t *device_statuses = nullptr;

    nvcomp_deflate::DeflateDecompressSizeAsync(
      device_compressed_chunk_ptrs,
      device_compressed_chunk_bytes,
      device_uncompressed_chunk_ptrs,
      device_uncompressed_buffer_bytes,
      device_uncompressed_chunk_bytes,
      device_statuses,
      nvcomp::narrow_cast<int>(num_chunks),
      stream,
      true /* gzip header parser*/
    );
  }
  catch (const std::exception &e)
  {
    LOG_ERROR("{}", e.what());
    return Check::exception_to_error(e, "nvcompBatchedGzipGetDecompressSizeAsync()");
  }

  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedGzipCompressGetRequiredAlignments(
  nvcompBatchedGzipCompressOpts_t format_opts,
  nvcompAlignmentRequirements_t *alignment_requirements
)
{
  GZIP_CHECK_COMPRESS_OPTS(format_opts);
  NVCOMP_CHECK_NOT_NULL(alignment_requirements);

  // Mirrors the deflate compressor's requirements (deflateBatch.cpp:665-670):
  //   input  : 1                              (only cast to uint8_t*)
  //   output : sizeof(unsigned long long) = 8 (bitwriter casts to ull*; also matches
  //                                             nvcompDeflateRequiredCompressionAlignment)
  //   temp   : sizeof(uint16_t*)          = 8 (compressAsync casts to uint16_t**)
  alignment_requirements->input = 1;
  alignment_requirements->output = sizeof(unsigned long long);
  alignment_requirements->temp = sizeof(uint16_t *);

  static_assert(
    ((nvcompGzipRequiredCompressionAlignment) & (nvcompGzipRequiredCompressionAlignment - 1)) == 0,
    "Minimum compression alignment is invalid."
  );
  assert(
    std::max(std::max(alignment_requirements->input, alignment_requirements->output), alignment_requirements->temp) <=
    nvcompGzipRequiredCompressionAlignment
  );

  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedGzipCompressAsync(
  const void *const *device_uncompressed_chunk_ptrs,
  const size_t *device_uncompressed_chunk_bytes,
  size_t max_uncompressed_chunk_bytes,
  size_t num_chunks,
  void *device_temp_ptr,
  size_t temp_bytes,
  void *const *device_compressed_chunk_ptrs,
  size_t *device_compressed_chunk_bytes,
  nvcompBatchedGzipCompressOpts_t format_opts,
  nvcompStatus_t *device_statuses,
  cudaStream_t stream
)
{
  GZIP_CHECK_COMPRESS_OPTS(format_opts);
  NVCOMP_CHECK_NOT_NULL(device_uncompressed_chunk_ptrs);
  NVCOMP_CHECK_NOT_NULL(device_uncompressed_chunk_bytes);
  NVCOMP_CHECK_NOT_NULL(device_temp_ptr);
  NVCOMP_CHECK_NOT_NULL(device_compressed_chunk_ptrs);
  NVCOMP_CHECK_NOT_NULL(device_compressed_chunk_bytes);

  nvcompAlignmentRequirements_t align_reqs{};
  NVCOMP_WRAP_CHECK_FUNC(nvcompBatchedGzipCompressGetRequiredAlignments, format_opts, &align_reqs);
  cuLibLogger::Logger::Instance().SetMostRecentApi(__func__);

  NVCOMP_CHECK_ALIGNMENT(device_uncompressed_chunk_ptrs);
  NVCOMP_CHECK_ALIGNMENT(device_uncompressed_chunk_bytes);
  NVCOMP_CHECK_ALIGNMENT(device_temp_ptr, align_reqs.temp);
  NVCOMP_CHECK_ALIGNMENT(device_compressed_chunk_ptrs);
  NVCOMP_CHECK_ALIGNMENT(device_compressed_chunk_bytes);
  NVCOMP_CHECK_ALIGNMENT(device_statuses);

  NVCOMP_CHECK_CHUNK_SIZE(max_uncompressed_chunk_bytes, nvcompGzipCompressionMaxAllowedChunkSize);

  // Note:
  // - Batch size is used as grid size, which CUDA limits to 2^31 - 1
  // - It is not an exact limit, but a good enough threshold
  NVCOMP_CHECK_BATCH_SIZE(num_chunks, std::numeric_limits<int>::max());

  try
  {
    static_assert(
      sizeof(nvcompStatus_t) == sizeof(nvcomp_deflate::deflateStatus_t),
      "Mismatched sizes of nvcompStatus_t and deflateStatus_t"
    );

    gdeflate::gdeflate_compression_algo algorithm = gdeflate::getCompressionAlgo(format_opts.algorithm);

    gzipBatchCompress(
      reinterpret_cast<const uint8_t *const *>(device_uncompressed_chunk_ptrs),
      device_uncompressed_chunk_bytes,
      max_uncompressed_chunk_bytes,
      num_chunks,
      static_cast<uint8_t *>(device_temp_ptr),
      temp_bytes,
      reinterpret_cast<uint8_t *const *>(device_compressed_chunk_ptrs),
      device_compressed_chunk_bytes,
      device_statuses,
      algorithm,
      stream
    );
  }
  catch (const std::exception &e)
  {
    LOG_ERROR("{}", e.what());
    return Check::exception_to_error(e, "nvcompBatchedGzipCompressAsync()");
  }

  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedGzipCompressGetTempSize(
  size_t num_chunks,
  size_t max_uncompressed_chunk_bytes,
  nvcompBatchedGzipCompressOpts_t format_opts,
  size_t *temp_bytes,
  size_t max_total_uncompressed_bytes, // unused, except for logging
  cudaStream_t stream // unused, except for logging
)
{
  GZIP_LOG_WITH_COMPRESS_OPTS(
    nvcomp::logBatchedCompressGetTempSize,
    format_opts,
    num_chunks,
    max_uncompressed_chunk_bytes,
    temp_bytes,
    max_total_uncompressed_bytes,
    stream
  );

  GZIP_CHECK_COMPRESS_OPTS(format_opts);
  NVCOMP_CHECK_NOT_NULL(temp_bytes);
  NVCOMP_CHECK_ALIGNMENT(temp_bytes);
  NVCOMP_CHECK_CHUNK_SIZE(max_uncompressed_chunk_bytes, nvcompGzipCompressionMaxAllowedChunkSize);
  NVCOMP_CHECK_BATCH_SIZE(num_chunks, std::numeric_limits<int>::max());

  try
  {
    gdeflate::gdeflate_compression_algo algorithm = gdeflate::getCompressionAlgo(format_opts.algorithm);

    const size_t max_total_deflate_blocks = gzip_max_total_deflate_blocks(max_uncompressed_chunk_bytes, num_chunks);

    gdeflate::compressGetTempSize(max_total_deflate_blocks, DEFLATE_BLOCK_SIZE, temp_bytes, algorithm);

    // Slot stride feeds the defrag frontier-flag (read_done) sizing inside the scratch handle.
    size_t per_sub_block_max = 0;
    nvcomp_deflate::DeflateCompressGetMaxOutputChunkSize(DEFLATE_BLOCK_SIZE, &per_sub_block_max);
    const size_t slot_stride = gzip_deflate_slot_stride(per_sub_block_max);

    *temp_bytes += CompressionScratchHandle::required_tmp_bytes(num_chunks, max_total_deflate_blocks, slot_stride);
  }
  catch (const std::exception &e)
  {
    LOG_ERROR("{}", e.what());
    return Check::exception_to_error(e, "nvcompBatchedGzipCompressGetTempSize()");
  }

  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedGzipCompressGetTempSizeSync(
  [[maybe_unused]] const void *const *const device_uncompressed_chunk_ptrs,
  [[maybe_unused]] const size_t *const device_uncompressed_chunk_bytes,
  size_t num_chunks,
  size_t max_uncompressed_chunk_bytes,
  nvcompBatchedGzipCompressOpts_t compress_opts,
  size_t *temp_bytes,
  size_t max_total_uncompressed_bytes,
  cudaStream_t stream
)
{
  return nvcompBatchedGzipCompressGetTempSize(
    num_chunks,
    max_uncompressed_chunk_bytes,
    compress_opts,
    temp_bytes,
    max_total_uncompressed_bytes,
    stream
  );
}

nvcompStatus_t nvcompBatchedGzipCompressGetMaxOutputChunkSize(
  size_t max_uncompressed_chunk_bytes,
  nvcompBatchedGzipCompressOpts_t format_opts,
  size_t *max_compressed_chunk_bytes
)
{
  GZIP_CHECK_COMPRESS_OPTS(format_opts);
  NVCOMP_CHECK_NOT_NULL(max_compressed_chunk_bytes);
  NVCOMP_CHECK_ALIGNMENT(max_compressed_chunk_bytes);
  NVCOMP_CHECK_CHUNK_SIZE(max_uncompressed_chunk_bytes, nvcompGzipCompressionMaxAllowedChunkSize);

  try
  {
    // Worst-case compressed size of one DEFLATE_BLOCK_SIZE sub-block (the gdeflate output slot).
    size_t per_sub_block_max = 0;
    nvcomp_deflate::DeflateCompressGetMaxOutputChunkSize(DEFLATE_BLOCK_SIZE, &per_sub_block_max);

    // An empty chunk instead emits a single 2-byte empty BT1 deflate block (0x03 0x00).
    constexpr size_t EMPTY_DEFLATE_BLOCK_BYTES = 2;

    const size_t num_deflates = roundUpDiv(max_uncompressed_chunk_bytes, DEFLATE_BLOCK_SIZE);
    const size_t content = num_deflates == 0 ? EMPTY_DEFLATE_BLOCK_BYTES
                                             : num_deflates * gzip_deflate_slot_stride(per_sub_block_max);

    *max_compressed_chunk_bytes = gzip_deflate_slot_base<gzipOperatingMode::ONESHOT>() + content + GZIP_FOOTER_BYTES;
  }
  catch (const std::exception &e)
  {
    LOG_ERROR("{}", e.what());
    return Check::exception_to_error(e, "nvcompBatchedGzipCompressGetMaxOutputChunkSize()");
  }

  return nvcompSuccess;
}
