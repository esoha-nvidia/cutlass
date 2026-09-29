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

#include "Check.h"
#include "common.h"
#include "CorrectnessChecks.cuh"
#include "CudaUtils.h"
#include "Environment.hpp"
#include "HWDecompress.hpp"
#include "Logging.h"
#include "lowlevel/nvcomp_private.h"
#include "LZ4CompressionKernels.h"
#include "LZ4Constants.cuh"
#include "nvcomp.h"
#include "nvcomp/lz4.h"
#include "type_macros.h"

using nvcomp::Check;
using nvcomp::CudaUtils;
using namespace nvcomp;
using namespace nvcomp::lowlevel;

namespace
{

constexpr const char *lz4_compress_opts_format_str = "{{data_type={:d}}}";
constexpr const char *lz4_decompress_opts_format_str = "{{backend={:d}}}";

#define LZ4_WITH_TRAILING_COMPRESS_OPTS_ARGS(callable, opts, ...) callable(__VA_ARGS__, int(opts.data_type))

#define LZ4_WITH_TRAILING_DECOMPRESS_OPTS_ARGS(callable, opts, ...) callable(__VA_ARGS__, int(opts.backend))

#define LZ4_LOG_WITH_COMPRESS_OPTS(log_func, opts, ...)                                                                \
  LZ4_WITH_TRAILING_COMPRESS_OPTS_ARGS(log_func, opts, __func__, __VA_ARGS__, lz4_compress_opts_format_str)

#define LZ4_LOG_WITH_DECOMPRESS_OPTS(log_func, opts, ...)                                                              \
  LZ4_WITH_TRAILING_DECOMPRESS_OPTS_ARGS(log_func, opts, __func__, __VA_ARGS__, lz4_decompress_opts_format_str)

#define LZ4_CHECK_COMPRESS_OPTS(opts) NVCOMP_WRAP_CHECK_FUNC(check_compress_opts, opts)

#define LZ4_CHECK_DECOMPRESS_OPTS(opts) NVCOMP_WRAP_CHECK_FUNC(check_decompress_opts, opts)

bool is_supported_lz4_data_type(nvcompType_t data_type)
{
  switch (data_type)
  {
    case NVCOMP_TYPE_BITS:
    case NVCOMP_TYPE_CHAR:
    case NVCOMP_TYPE_UCHAR:
    case NVCOMP_TYPE_SHORT:
    case NVCOMP_TYPE_USHORT:
    case NVCOMP_TYPE_INT:
    case NVCOMP_TYPE_UINT:
      return true;
    default:
      return false;
  }
}

bool is_supported_bitshuffle_mode(nvcompBitshuffleMode_t mode)
{
  return mode == NVCOMP_BITSHUFFLE_NONE || mode == NVCOMP_BITSHUFFLE_MSB_FIRST || mode == NVCOMP_BITSHUFFLE_LSB_FIRST;
}

nvcompStatus_t check_compress_opts(const nvcompBatchedLZ4CompressOpts_t &opts)
{
  if (is_supported_lz4_data_type(opts.data_type) && is_supported_bitshuffle_mode(opts.bitshuffle_mode))
  {
    return nvcompSuccess;
  }

  LZ4_WITH_TRAILING_COMPRESS_OPTS_ARGS(
    LOG_ERROR,
    opts,
    std::string("Unsupported options: ") + lz4_compress_opts_format_str
  );
  return nvcompErrorNotSupported;
}

nvcompStatus_t check_decompress_opts(const nvcompBatchedLZ4DecompressOpts_t &opts)
{
  const bool supported = opts.backend == NVCOMP_DECOMPRESS_BACKEND_DEFAULT ||
                         opts.backend == NVCOMP_DECOMPRESS_BACKEND_CUDA ||
                         opts.backend == NVCOMP_DECOMPRESS_BACKEND_HARDWARE;
  if (supported)
  {
    if (opts.backend == NVCOMP_DECOMPRESS_BACKEND_HARDWARE && opts.sort_before_hw_decompress != 0 &&
        opts.sort_before_hw_decompress != 1)
    {
      LOG_ERROR("sort_before_hw_decompress must be 0 or 1, got {}.", opts.sort_before_hw_decompress);
      return nvcompErrorInvalidValue;
    }
    // data_type is only consulted when bitshuffle is enabled.
    const bool opts_supported = is_supported_bitshuffle_mode(opts.bitshuffle_mode) &&
                                (opts.bitshuffle_mode == NVCOMP_BITSHUFFLE_NONE ||
                                 is_supported_lz4_data_type(opts.data_type));
    if (opts_supported)
    {
      return nvcompSuccess;
    }
  }

  // Other errors
  LZ4_WITH_TRAILING_DECOMPRESS_OPTS_ARGS(
    LOG_ERROR,
    opts,
    std::string("Unsupported options: ") + lz4_decompress_opts_format_str
  );
  return nvcompErrorNotSupported;
}

} // namespace

static_assert(sizeof(nvcompBatchedLZ4CompressOpts_t) == 64);
static_assert(sizeof(nvcompBatchedLZ4DecompressOpts_t) == 64);

nvcompStatus_t nvcompBatchedLZ4DecompressGetRequiredAlignments(
  nvcompBatchedLZ4DecompressOpts_t decompress_opts,
  nvcompAlignmentRequirements_t *alignment_requirements
)
{
  LZ4_LOG_WITH_DECOMPRESS_OPTS(
    nvcomp::logBatchedDecompressGetRequiredAlignments,
    decompress_opts,
    alignment_requirements
  );
  LZ4_CHECK_DECOMPRESS_OPTS(decompress_opts);
  NVCOMP_CHECK_NOT_NULL(alignment_requirements);

  // Note: The decompress engine (DE) requires 1-byte alignment.

  // Input and output are only ever cast to uint8_t*, so 1.
  // The temp buffer is not used, implying the smallest valid alignment, 1.
  alignment_requirements->input = 1;
  alignment_requirements->output = 1;
  if (decompress_opts.bitshuffle_mode != NVCOMP_BITSHUFFLE_NONE)
  {
    alignment_requirements->output = nvcomp::sizeOfnvcompType(decompress_opts.data_type);
  }
  alignment_requirements->temp = 1;

  static_assert(
    ((nvcompLZ4RequiredDecompressionAlignment) & (nvcompLZ4RequiredDecompressionAlignment - 1)) == 0,
    "Minimum decompression alignment is invalid."
  );
  assert(
    std::max(std::max(alignment_requirements->input, alignment_requirements->output), alignment_requirements->temp) <=
    nvcompLZ4RequiredDecompressionAlignment
  );

  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedLZ4DecompressGetTempSizeAsync(
  const size_t num_chunks,
  const size_t max_uncompressed_chunk_bytes,
  nvcompBatchedLZ4DecompressOpts_t decompress_opts,
  size_t *const temp_bytes,
  const size_t max_total_uncompressed_bytes
)
{
  LZ4_LOG_WITH_DECOMPRESS_OPTS(
    nvcomp::logBatchedDecompressGetTempSizeAsync,
    decompress_opts,
    num_chunks,
    max_uncompressed_chunk_bytes,
    temp_bytes,
    max_total_uncompressed_bytes
  );

  // Error check inputs
  LZ4_CHECK_DECOMPRESS_OPTS(decompress_opts);
  NVCOMP_CHECK_NOT_NULL(temp_bytes);
  NVCOMP_CHECK_ALIGNMENT(temp_bytes);
  NVCOMP_CHECK_CHUNK_SIZE(max_uncompressed_chunk_bytes, nvcompLZ4DecompressionMaxAllowedChunkSize);

  try
  {
    // LZ4 doesn't need any workspace in GPU memory for decompression itself
    *temp_bytes = 0;

    // Allocating temporary space for sorting
    if (decompress_opts.sort_before_hw_decompress)
    {
      *temp_bytes += get_sort_scratch_req(num_chunks, max_uncompressed_chunk_bytes);
    }
    // Allocating temporary space for correctness checks
    const auto lz4_correctness_env = nvcomp::getenv(CHECK_LZ4_CORRECTNESS_ENV);
    const bool check_lz4_correctness = not(lz4_correctness_env.empty() or lz4_correctness_env == "0");
    if (check_lz4_correctness)
    {
      // If correctness checks are enabled, we need to allocate space for
      // the device status array.
      *temp_bytes += sizeof(LZ4CorrectnessChecker<true>) * num_chunks + alignof(LZ4CorrectnessChecker<true>) - 1;
    }
  }
  catch (const std::exception &e)
  {
    LOG_ERROR("{}", e.what());
    return Check::exception_to_error(e, "nvcompBatchedLZ4DecompressGetTempSizeAsync()");
  }

  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedLZ4DecompressGetTempSizeSync(
  [[maybe_unused]] const void *const *const device_compressed_chunk_ptrs,
  [[maybe_unused]] const size_t *const device_compressed_chunk_bytes,
  size_t num_chunks,
  size_t max_uncompressed_chunk_bytes,
  size_t *temp_bytes,
  size_t max_total_uncompressed_bytes,
  nvcompBatchedLZ4DecompressOpts_t decompress_opts,
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

  return nvcompBatchedLZ4DecompressGetTempSizeAsync(
    num_chunks,
    max_uncompressed_chunk_bytes,
    decompress_opts,
    temp_bytes,
    max_total_uncompressed_bytes
  );
}

nvcompStatus_t nvcompBatchedLZ4DecompressAsyncEx(
  const void *const *device_compressed_chunk_ptrs,
  const size_t *device_compressed_chunk_bytes,
  const size_t *device_uncompressed_buffer_bytes,
  size_t *device_uncompressed_chunk_bytes,
  size_t num_chunks,
  void *const device_temp_ptr,
  size_t temp_bytes,
  void *const *device_uncompressed_chunk_ptrs,
  nvcompStatus_t *device_statuses,
  [[maybe_unused]] nvcompBatchedLZ4CompressOpts_t compress_opts,
  nvcompBatchedLZ4DecompressOpts_t decompress_opts,
  cudaStream_t stream,
  const void *const *host_comp_chunk_buffers
)
{
  bool host_setup_mode = host_comp_chunk_buffers != nullptr;
  bool force_sync = host_setup_mode;
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

  const auto lz4_correctness_env = nvcomp::getenv(CHECK_LZ4_CORRECTNESS_ENV);
  const bool check_lz4_correctness = not(lz4_correctness_env.empty() or lz4_correctness_env == "0");

  // error check inputs
  NVCOMP_CHECK_NOT_NULL(device_compressed_chunk_ptrs);
  NVCOMP_CHECK_NOT_NULL(device_compressed_chunk_bytes);
  NVCOMP_CHECK_NOT_NULL(device_uncompressed_buffer_bytes);
  NVCOMP_CHECK_NOT_NULL(device_uncompressed_chunk_ptrs);
  if (device_uncompressed_chunk_bytes == nullptr && decompress_opts.bitshuffle_mode != NVCOMP_BITSHUFFLE_NONE)
  {
    LOG_ERROR("device_uncompressed_chunk_bytes is required when bitshuffle is enabled.");
    return nvcompErrorInvalidValue;
  }

  // Device status array must be provided for correctness checks
  if (check_lz4_correctness and (device_statuses == nullptr or device_temp_ptr == nullptr))
  {
    LOG_ERROR("For correctness checks, device_statuses and device_temp_ptr must be provided");
    return nvcompErrorInvalidValue;
  }
  LZ4_CHECK_DECOMPRESS_OPTS(decompress_opts);

  // Check device pointer alignment
  NVCOMP_CHECK_ALIGNMENT(device_compressed_chunk_ptrs);
  NVCOMP_CHECK_ALIGNMENT(device_compressed_chunk_bytes);
  NVCOMP_CHECK_ALIGNMENT(device_uncompressed_buffer_bytes);
  NVCOMP_CHECK_ALIGNMENT(device_uncompressed_chunk_bytes);
  NVCOMP_CHECK_ALIGNMENT(device_uncompressed_chunk_ptrs);
  NVCOMP_CHECK_ALIGNMENT(device_statuses);

  // Note:
  // We generally perform a NVCOMP_CHECK_BATCH_SIZE(...) due to potential DE->SM fallback,
  // even though the DE is seemingly not putting a constraint on the batch size being
  // maximum std::numeric_limits<int>::max().
  NVCOMP_CHECK_BATCH_SIZE(num_chunks, std::numeric_limits<int>::max());

  auto launch_hw_decomp = [&](const bool force_hw_decomp) -> bool {
    if (device_uncompressed_chunk_bytes == nullptr)
    {
      return false;
    }

    auto fill_params = [&device_uncompressed_buffer_bytes,
                        host_comp_chunk_buffers,
                        host_setup_mode,
                        &temp_bytes,
                        &device_temp_ptr,
                        &decompress_opts,
                        force_sync](
                         const void *const *device_compressed_chunk_ptrs,
                         const size_t *device_compressed_chunk_bytes,
                         void *const *device_uncompressed_chunk_ptrs,
                         CUmemDecompressParams *de_params,
                         size_t *device_uncompressed_chunk_bytes,
                         size_t num_chunks,
                         nvcompStatus_t *device_statuses,
                         cudaStream_t stream
                       ) {
      if (host_setup_mode)
      {
        FillCuDEParamsOnHost(
          device_compressed_chunk_ptrs,
          device_compressed_chunk_bytes,
          device_uncompressed_chunk_ptrs,
          de_params,
          device_uncompressed_chunk_bytes,
          device_uncompressed_buffer_bytes,
          num_chunks,
          device_statuses,
          false,
          decompress_opts.sort_before_hw_decompress,
          CU_MEM_DECOMPRESS_ALGORITHM_LZ4,
          host_comp_chunk_buffers,
          stream,
          force_sync
        );
      }
      else
      {
        LZ4FillCuDecompParams(
          device_compressed_chunk_ptrs,
          device_compressed_chunk_bytes,
          device_uncompressed_chunk_ptrs,
          de_params,
          device_uncompressed_chunk_bytes,
          device_uncompressed_buffer_bytes,
          num_chunks,
          device_statuses,
          reinterpret_cast<uint8_t *>(device_temp_ptr),
          temp_bytes,
          decompress_opts.sort_before_hw_decompress,
          stream
        );
      }
    };

    return do_hw_decomp(
      device_uncompressed_chunk_ptrs,
      device_uncompressed_chunk_bytes,
      num_chunks,
      device_compressed_chunk_ptrs,
      device_compressed_chunk_bytes,
      device_uncompressed_buffer_bytes,
      device_statuses,
      stream,
      fill_params,
      nvcompFormatType_t::LZ4,
      force_hw_decomp
    );
  };

  auto launch_sm_decomp = [&]() -> void {
    auto lz4_decompress_func = check_lz4_correctness ? lz4BatchDecompress<true> : lz4BatchDecompress<false>;

    void *device_correctness_ptrs = check_lz4_correctness
                                      ? roundUpToAlignment<LZ4CorrectnessChecker<true>>(device_temp_ptr)
                                      : nullptr;

    lz4_decompress_func(
      reinterpret_cast<const uint8_t *const *>(device_compressed_chunk_ptrs),
      device_compressed_chunk_bytes,
      device_uncompressed_buffer_bytes,
      num_chunks,
      reinterpret_cast<uint8_t *const *>(device_uncompressed_chunk_ptrs),
      device_uncompressed_chunk_bytes,
      device_statuses,
      device_correctness_ptrs,
      stream
    );
  };

  const auto env_var_value = nvcomp::getenv(USE_HW_DECOMPRESSION_ENV);
  if (!env_var_value.empty())
  {
    decompress_opts.backend = static_cast<bool>(std::stoi(env_var_value)) ? NVCOMP_DECOMPRESS_BACKEND_HARDWARE
                                                                          : NVCOMP_DECOMPRESS_BACKEND_CUDA;
  }
  if (check_lz4_correctness)
  {
    decompress_opts.backend = NVCOMP_DECOMPRESS_BACKEND_CUDA;
  }

  try
  {
    bool exec_on_hw = false;
    switch (decompress_opts.backend)
    {
      case NVCOMP_DECOMPRESS_BACKEND_DEFAULT:
        // try HW decompression first, if it fails,
        // fallback to cuda implementation
        LOG_INFO("Trying HW decompression");
        exec_on_hw = launch_hw_decomp(false /* force_hw_decomp */);
        if (not exec_on_hw)
        {
          LOG_INFO("HW Decompression launch failed, launching SM decompression");
          launch_sm_decomp();
        }
        else
        {
          LOG_INFO("HW Decompression successful");
        }
        break;
      case NVCOMP_DECOMPRESS_BACKEND_HARDWARE:
        LOG_INFO("Launching HW decompression");
        // try HW decomp only, if it fails return an error
        exec_on_hw = launch_hw_decomp(true /* force_hw_decomp */);
        if (not exec_on_hw)
        {
          LOG_ERROR("Failed to execute decompression on HW or HW engine not available");
          return nvcompErrorCannotDecompress;
        }
        break;
      case NVCOMP_DECOMPRESS_BACKEND_CUDA:
        LOG_INFO("Launching SM decompression");
        launch_sm_decomp();
        break;
      default:
        LOG_ERROR(
          "Decompress backend is invalid. Acceptable values are NVCOMP_DECOMPRESS_BACKEND_DEFAULT, "
          "NVCOMP_DECOMPRESS_BACKEND_HARDWARE, or NVCOMP_DECOMPRESS_BACKEND_CUDA."
        );
        assert(0);
        return nvcompErrorInvalidValue;
    }

    // Optionally do an inverse bitshuffle on the data
    if (decompress_opts.bitshuffle_mode == NVCOMP_BITSHUFFLE_MSB_FIRST ||
        decompress_opts.bitshuffle_mode == NVCOMP_BITSHUFFLE_LSB_FIRST)
    {
      doInverseBitshuffle(
        reinterpret_cast<uint8_t *const *>(device_uncompressed_chunk_ptrs),
        device_uncompressed_chunk_bytes,
        num_chunks,
        decompress_opts.data_type,
        decompress_opts.bitshuffle_mode == NVCOMP_BITSHUFFLE_MSB_FIRST,
        stream
      );
    }
  }
  catch (const std::exception &e)
  {
    LOG_ERROR("{}", e.what());
    return Check::exception_to_error(e, "nvcompBatchedLZ4DecompressAsync()");
  }

  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedLZ4DecompressAsync(
  const void *const *device_compressed_chunk_ptrs,
  const size_t *device_compressed_chunk_bytes,
  const size_t *device_uncompressed_buffer_bytes,
  size_t *device_uncompressed_chunk_bytes, // optional
  size_t num_chunks,
  void *const device_temp_ptr,
  size_t temp_bytes,
  void *const *device_uncompressed_chunk_ptrs,
  nvcompBatchedLZ4DecompressOpts_t decompress_opts,
  nvcompStatus_t *device_statuses,
  cudaStream_t stream
)
{
  return nvcompBatchedLZ4DecompressAsyncEx(
    device_compressed_chunk_ptrs,
    device_compressed_chunk_bytes,
    device_uncompressed_buffer_bytes,
    device_uncompressed_chunk_bytes,
    num_chunks,
    device_temp_ptr,
    temp_bytes,
    device_uncompressed_chunk_ptrs,
    device_statuses,
    nvcompBatchedLZ4CompressDefaultOpts,
    decompress_opts,
    stream,
    nullptr /*host_comp_chunk_buffers*/
  );
}

nvcompStatus_t nvcompBatchedLZ4GetDecompressSizeAsync(
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

  try
  {
    lz4BatchGetDecompressSizes(
      reinterpret_cast<const uint8_t *const *>(device_compressed_chunk_ptrs),
      device_compressed_chunk_bytes,
      device_uncompressed_chunk_bytes,
      num_chunks,
      stream
    );
  }
  catch (const std::exception &e)
  {
    LOG_ERROR("{}", e.what());
    return Check::exception_to_error(e, "nvcompBatchedLZ4GetDecompressSizeAsync()");
  }

  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedLZ4CompressGetRequiredAlignments(
  nvcompBatchedLZ4CompressOpts_t format_opts,
  nvcompAlignmentRequirements_t *alignment_requirements
)
{
  LZ4_LOG_WITH_COMPRESS_OPTS(nvcomp::logBatchedCompressGetRequiredAlignments, format_opts, alignment_requirements);
  LZ4_CHECK_COMPRESS_OPTS(format_opts);
  NVCOMP_CHECK_NOT_NULL(alignment_requirements);

  // Explicit assert in lz4CompressBatchKernel.
  alignment_requirements->input = nvcomp::sizeOfnvcompType(format_opts.data_type);
  // Only ever cast to uint8_t*.
  alignment_requirements->output = 1;
  // Cast to offset_type* (aka uint16_t*) in lz4BatchCompress.
  if (format_opts.bitshuffle_mode == NVCOMP_BITSHUFFLE_NONE)
  {
    alignment_requirements->temp = 2;
  }
  else
  {
    // If bitshuffle is enabled, the temp buffer must be aligned to the input type because
    // the bitshuffle kernel will store the bitshuffled data in the temp buffer for
    // consumption by the LZ4 compressor.
    alignment_requirements->temp = std::max(nvcomp::sizeOfnvcompType(format_opts.data_type), size_t(2));
  }

  static_assert(
    ((nvcompLZ4RequiredCompressionAlignment) & (nvcompLZ4RequiredCompressionAlignment - 1)) == 0,
    "Minimum compression alignment is invalid."
  );
  assert(
    std::max(std::max(alignment_requirements->input, alignment_requirements->output), alignment_requirements->temp) <=
    nvcompLZ4RequiredCompressionAlignment
  );

  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedLZ4CompressGetTempSizeAsync(
  const size_t num_chunks,
  const size_t max_uncompressed_chunk_bytes,
  const nvcompBatchedLZ4CompressOpts_t format_opts, // unused, except for logging
  size_t *const temp_bytes,
  const size_t max_total_uncompressed_bytes
) // unused, except for logging
{
  LZ4_LOG_WITH_COMPRESS_OPTS(
    nvcomp::logBatchedCompressGetTempSizeAsync,
    format_opts,
    num_chunks,
    max_uncompressed_chunk_bytes,
    temp_bytes,
    max_total_uncompressed_bytes
  );
  LZ4_CHECK_COMPRESS_OPTS(format_opts);

  NVCOMP_CHECK_NOT_NULL(temp_bytes);
  NVCOMP_CHECK_ALIGNMENT(temp_bytes);

  NVCOMP_CHECK_CHUNK_SIZE(max_uncompressed_chunk_bytes, nvcompLZ4CompressionMaxAllowedChunkSize);

  try
  {
    *temp_bytes = lz4BatchCompressComputeTempSize(max_uncompressed_chunk_bytes, num_chunks);
    // Add the size of the uncompressed data to the temp bytes for storing output of bitshuffle operation.
    // User must provide temp buffer with alignment per GetRequiredAlignments.
    if (format_opts.bitshuffle_mode != NVCOMP_BITSHUFFLE_NONE)
    {
      *temp_bytes += num_chunks * roundUpTo(max_uncompressed_chunk_bytes, LZ4_BITSHUFFLE_BYTES_PER_THREAD);
    }
  }
  catch (const std::exception &e)
  {
    LOG_ERROR("{}", e.what());
    return Check::exception_to_error(e, "nvcompBatchedLZ4CompressGetTempSizeAsync()");
  }
  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedLZ4CompressGetTempSizeSync(
  [[maybe_unused]] const void *const *const device_uncompressed_chunk_ptrs,
  [[maybe_unused]] const size_t *const device_uncompressed_chunk_bytes,
  size_t num_chunks,
  size_t max_uncompressed_chunk_bytes,
  nvcompBatchedLZ4CompressOpts_t compress_opts,
  size_t *temp_bytes,
  size_t max_total_uncompressed_bytes,
  [[maybe_unused]] cudaStream_t stream
)
{
  return nvcompBatchedLZ4CompressGetTempSizeAsync(
    num_chunks,
    max_uncompressed_chunk_bytes,
    compress_opts, // unused, except for logging
    temp_bytes,
    max_total_uncompressed_bytes
  );
}

nvcompStatus_t nvcompBatchedLZ4CompressGetMaxOutputChunkSize(
  const size_t max_uncompressed_chunk_bytes,
  const nvcompBatchedLZ4CompressOpts_t format_opts, // unused, except for logging
  size_t *const max_compressed_chunk_bytes
)
{
  LZ4_LOG_WITH_COMPRESS_OPTS(
    nvcomp::logBatchedCompressGetMaxOutputChunkSize,
    format_opts,
    max_uncompressed_chunk_bytes,
    max_compressed_chunk_bytes
  );
  LZ4_CHECK_COMPRESS_OPTS(format_opts);

  NVCOMP_CHECK_NOT_NULL(max_compressed_chunk_bytes);
  NVCOMP_CHECK_ALIGNMENT(max_compressed_chunk_bytes);

  NVCOMP_CHECK_CHUNK_SIZE(max_uncompressed_chunk_bytes, nvcompLZ4CompressionMaxAllowedChunkSize);

  try
  {
    *max_compressed_chunk_bytes = lz4ComputeMaxSize(max_uncompressed_chunk_bytes);
  }
  catch (const std::exception &e)
  {
    LOG_ERROR("{}", e.what());
    return Check::exception_to_error(e, "nvcompBatchedLZ4CompressGetOutputSize()");
  }

  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedLZ4CompressAsync(
  const void *const *const device_uncompressed_chunk_ptrs,
  const size_t *const device_uncompressed_chunk_bytes,
  const size_t max_uncompressed_chunk_bytes,
  const size_t num_chunks,
  void *const device_temp_ptr,
  const size_t temp_bytes,
  void *const *const device_compressed_chunk_ptrs,
  size_t *const device_compressed_chunk_bytes,
  const nvcompBatchedLZ4CompressOpts_t format_opts,
  nvcompStatus_t *device_statuses, // optional
  cudaStream_t stream
)
{
  // NOTE: if we start using `max_uncompressed_chunk_bytes`, we need to check
  // to make sure it is not zero, as we have notified users to supply zero if
  // they are not finding the maximum size.

  LZ4_LOG_WITH_COMPRESS_OPTS(
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
  LZ4_CHECK_COMPRESS_OPTS(format_opts);

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
  NVCOMP_WRAP_CHECK_FUNC(nvcompBatchedLZ4CompressGetRequiredAlignments, format_opts, &align_reqs);
  cuLibLogger::Logger::Instance().SetMostRecentApi(__func__);

  // Check device pointer alignment
  // TODO: Allow 2-byte alignment if format_opts.data_type isn't INT or UINT.
  NVCOMP_CHECK_ALIGNMENT(device_uncompressed_chunk_ptrs);
  NVCOMP_CHECK_ALIGNMENT(device_uncompressed_chunk_bytes);
  NVCOMP_CHECK_ALIGNMENT(device_temp_ptr, align_reqs.temp);
  NVCOMP_CHECK_ALIGNMENT(device_compressed_chunk_ptrs);
  NVCOMP_CHECK_ALIGNMENT(device_compressed_chunk_bytes);
  NVCOMP_CHECK_ALIGNMENT(device_statuses);

  NVCOMP_CHECK_CHUNK_SIZE(max_uncompressed_chunk_bytes, nvcompLZ4CompressionMaxAllowedChunkSize);

  // Batch size is used as grid size, which CUDA limits to 2^31 - 1
  NVCOMP_CHECK_BATCH_SIZE(num_chunks, std::numeric_limits<int>::max());

  bool do_bitshuffle = format_opts.bitshuffle_mode != NVCOMP_BITSHUFFLE_NONE && max_uncompressed_chunk_bytes > 0;
  if (not do_bitshuffle && format_opts.bitshuffle_mode != NVCOMP_BITSHUFFLE_NONE)
  {
    LOG_ERROR(
      "Bitshuffle mode is set to {}, but max_uncompressed_chunk_bytes is 0. "
      "Please set max_uncompressed_chunk_bytes to a valid non-zero value.",
      format_opts.bitshuffle_mode
    );
    return nvcompErrorInvalidValue;
  }

  static_assert(nvcompLZ4CompressionMaxAllowedChunkSize < (1ULL << 32));

  try
  {
    if (do_bitshuffle)
    {
      doBitshuffle(
        reinterpret_cast<const uint8_t *const *>(device_uncompressed_chunk_ptrs),
        device_uncompressed_chunk_bytes,
        device_temp_ptr,
        num_chunks,
        static_cast<uint32_t>(max_uncompressed_chunk_bytes),
        format_opts.data_type,
        format_opts.bitshuffle_mode == NVCOMP_BITSHUFFLE_MSB_FIRST,
        stream
      );
    }

    lz4BatchCompress(
      reinterpret_cast<const uint8_t *const *>(device_uncompressed_chunk_ptrs),
      device_uncompressed_chunk_bytes,
      static_cast<unsigned int>(max_uncompressed_chunk_bytes),
      num_chunks,
      device_temp_ptr,
      temp_bytes,
      reinterpret_cast<uint8_t *const *>(device_compressed_chunk_ptrs),
      device_compressed_chunk_bytes,
      format_opts.data_type,
      do_bitshuffle, // reads input from the temp buffer if bitshuffle is enabled
      device_statuses,
      stream
    );
  }
  catch (const std::exception &e)
  {
    LOG_ERROR("{}", e.what());
    return Check::exception_to_error(e, "nvcompBatchedLZ4CompressAsync()");
  }

  return nvcompSuccess;
}
