/*
 * Copyright (c) 2017-2021, NVIDIA CORPORATION. All rights reserved.
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
#include "exception.hpp"
#include "gdeflate/common.h"
#include "gdeflate/deflate.h"
#include "gdeflate/gdeflate.h"
#include "gdeflate/gdeflateKernels.h"
#include "HWDecompress.hpp"
#include "Logging.h"
#include "lowlevel/nvcomp_private.h"
#include "nvcomp.h"
#include "nvcomp/deflate.h"
#include "nvcomp/gzip.h"
#include "type_macros.h"

using nvcomp::Check;
using namespace nvcomp;

// Note: no "catch" statements are triggered during testing.
namespace
{

constexpr const char *deflate_compress_opts_format_str = "{{algo={:d}}}";
constexpr const char *deflate_decompress_opts_format_str = "{{backend={:d}}}";

#define DEFLATE_WITH_TRAILING_COMPRESS_OPTS_ARGS(callable, opts, ...) callable(__VA_ARGS__, opts.algorithm)

#define DEFLATE_WITH_TRAILING_DECOMPRESS_OPTS_ARGS(callable, opts, ...) callable(__VA_ARGS__, int(opts.backend))

#define DEFLATE_LOG_WITH_COMPRESS_OPTS(log_func, opts, ...)                                                            \
  DEFLATE_WITH_TRAILING_COMPRESS_OPTS_ARGS(log_func, opts, __func__, __VA_ARGS__, deflate_compress_opts_format_str)

#define DEFLATE_LOG_WITH_DECOMPRESS_OPTS(log_func, opts, ...)                                                          \
  DEFLATE_WITH_TRAILING_DECOMPRESS_OPTS_ARGS(log_func, opts, __func__, __VA_ARGS__, deflate_decompress_opts_format_str)

#define DEFLATE_CHECK_COMPRESS_OPTS(opts) NVCOMP_WRAP_CHECK_FUNC(check_compress_opts, opts)

#define DEFLATE_CHECK_DECOMPRESS_OPTS(opts) NVCOMP_WRAP_CHECK_FUNC(check_decompress_opts, opts)

nvcompStatus_t check_compress_opts(const nvcompBatchedDeflateCompressOpts_t &opts)
{
  const bool supported = opts.algorithm >= 0 && opts.algorithm <= 5;
  if (supported)
  {
    return nvcompSuccess;
  }

  DEFLATE_WITH_TRAILING_COMPRESS_OPTS_ARGS(
    LOG_ERROR,
    opts,
    std::string("Unsupported options: ") + deflate_compress_opts_format_str
  );
  return nvcompErrorNotSupported;
}

nvcompStatus_t check_decompress_opts(const nvcompBatchedDeflateDecompressOpts_t &opts)
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
    return nvcompSuccess;
  }

  // Other errors
  DEFLATE_WITH_TRAILING_DECOMPRESS_OPTS_ARGS(
    LOG_ERROR,
    opts,
    std::string("Unsupported options: ") + deflate_decompress_opts_format_str
  );
  return nvcompErrorNotSupported;
}

} // namespace

namespace nvcomp_deflate
{

nvcompStatus_t DetailDeflateDecompress(
  const void *const *device_compressed_chunk_ptrs,
  const size_t *device_compressed_chunk_bytes,
  const size_t *device_uncompressed_buffer_bytes,
  size_t *device_uncompressed_chunk_bytes, // optional
  size_t num_chunks,
  [[maybe_unused]] void *const device_temp_ptr,
  [[maybe_unused]] size_t temp_bytes,
  void *const *device_uncompressed_chunk_ptrs,
  nvcompStatus_t *device_statuses,
  nvcompBatchedDeflateDecompressOpts_t *deflate_decompress_opts,
  nvcompBatchedGzipDecompressOpts_t *gzip_decompress_opts,
  bool gzip,
  cudaStream_t stream,
  const void *const *host_comp_chunk_buffers
)
{
  bool host_setup_mode = host_comp_chunk_buffers != nullptr;
  bool force_sync = host_setup_mode;

  NVCOMP_CHECK_NOT_NULL(device_compressed_chunk_ptrs);
  NVCOMP_CHECK_NOT_NULL(device_compressed_chunk_bytes);
  NVCOMP_CHECK_NOT_NULL(device_uncompressed_buffer_bytes);
  NVCOMP_CHECK_NOT_NULL(device_uncompressed_chunk_ptrs);
  if ((device_uncompressed_chunk_bytes == nullptr) != (device_statuses == nullptr))
  {
    LOG_ERROR("Both device_uncompressed_chunk_bytes and device_statuses should be valid or nullptr");
    return nvcompErrorInvalidValue;
  }
  nvcompDecompressBackend_t backend = NVCOMP_DECOMPRESS_BACKEND_DEFAULT;
  bool use_sorting = false;
  if (deflate_decompress_opts)
  {
    backend = deflate_decompress_opts->backend;
    use_sorting = deflate_decompress_opts->sort_before_hw_decompress;
  }
  else if (gzip_decompress_opts)
  {
    backend = gzip_decompress_opts->backend;
    use_sorting = gzip_decompress_opts->sort_before_hw_decompress;
  }

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

  const auto deflate_correctness_env = nvcomp::getenv(CHECK_DEFLATE_CORRECTNESS_ENV);
  const bool check_deflate_correctness = not(deflate_correctness_env.empty() or deflate_correctness_env == "0");

  if (check_deflate_correctness and (device_statuses == nullptr or device_temp_ptr == nullptr))
  {
    LOG_ERROR("For correctness checks, device_statuses and device_temp_ptr must be provided");
    return nvcompErrorInvalidValue;
  }

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
                        gzip,
                        use_sorting,
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
          gzip,
          use_sorting,
          CU_MEM_DECOMPRESS_ALGORITHM_DEFLATE,
          host_comp_chunk_buffers,
          stream,
          force_sync
        );
      }
      else
      {
        nvcomp_deflate::DeflateFillCuDecompParams(
          de_params,
          device_compressed_chunk_ptrs,
          device_compressed_chunk_bytes,
          device_uncompressed_chunk_bytes,
          device_uncompressed_buffer_bytes,
          device_uncompressed_chunk_ptrs,
          gzip,
          num_chunks,
          device_statuses,
          reinterpret_cast<uint8_t *>(device_temp_ptr),
          temp_bytes,
          use_sorting,
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
      (gzip ? nvcompFormatType_t::Gzip : nvcompFormatType_t::Deflate),
      force_hw_decomp
    );
  };

  auto launch_sm_decomp = [&]() -> void {
    // Use device_status_ptrs as temp space to store deflate statuses
    static_assert(
      sizeof(nvcompStatus_t) == sizeof(nvcomp_deflate::deflateStatus_t),
      "Mismatched sizes of nvcompStatus_t and deflateStatus_t"
    );
    auto decompress_func = check_deflate_correctness ? DeflateDecompressAsync<true> : DeflateDecompressAsync<false>;
    void *device_correctness_ptrs = check_deflate_correctness
                                      ? roundUpToAlignment<DeflateCorrectnessChecker<true>>(device_temp_ptr)
                                      : nullptr;
    decompress_func(
      device_compressed_chunk_ptrs,
      device_compressed_chunk_bytes,
      device_uncompressed_chunk_ptrs,
      device_uncompressed_buffer_bytes,
      device_uncompressed_chunk_bytes,
      reinterpret_cast<nvcomp_deflate::deflateStatus_t *>(device_statuses),
      device_correctness_ptrs,
      num_chunks,
      stream,
      gzip
    );
  };

  const auto env_var_value = nvcomp::getenv(USE_HW_DECOMPRESSION_ENV);
  if (!env_var_value.empty())
  {
    backend = static_cast<bool>(std::stoi(env_var_value)) ? NVCOMP_DECOMPRESS_BACKEND_HARDWARE
                                                          : NVCOMP_DECOMPRESS_BACKEND_CUDA;
  }
  if (check_deflate_correctness)
  {
    backend = NVCOMP_DECOMPRESS_BACKEND_CUDA;
  }

  bool exec_on_hw = false;
  switch (backend)
  {
    case NVCOMP_DECOMPRESS_BACKEND_DEFAULT:
      // try HW decompression first, if it fails,
      // fallback to cuda implementation
      LOG_INFO("Trying HW decompression");
      exec_on_hw = launch_hw_decomp(false /* force_hw_decomp */);
      if (not exec_on_hw)
      {
        LOG_INFO("HW Decompression failed, launching SM decompression");
        launch_sm_decomp();
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
  return nvcompSuccess;
}

} // namespace nvcomp_deflate

static_assert(sizeof(nvcompBatchedDeflateCompressOpts_t) == 64);
static_assert(sizeof(nvcompBatchedDeflateDecompressOpts_t) == 64);

gdeflate::gdeflate_compression_algo getDeflateEnumFromFormatOpts(nvcompBatchedDeflateCompressOpts_t format_opts)
{
  switch (format_opts.algorithm)
  {
    case (0):
      return gdeflate::ENTROPY_ONLY;
    case (1):
      return gdeflate::HIGH_THROUGHPUT;
    case (2):
      return gdeflate::MEDIUM_COMPRESSION;
    //placeholder for further compression level support.
    //Will fall into MEDIUM_COMPRESSION at this point.
    case (3):
      return gdeflate::MEDIUM_COMPRESSION;
    case (4):
      return gdeflate::L6_OP_COMPRESSION;
    case (5):
      return gdeflate::HIGH_COMPRESSION;
    default:
      throw nvcomp::NVCompException(
        nvcompErrorInvalidValue,
        "Invalid compress_opts.algorithm value (not 1, 2, 3, 4, 5)"
      );
  }
  return gdeflate::gdeflate_compression_algo(-1);
}

nvcompStatus_t nvcompBatchedDeflateDecompressGetRequiredAlignments(
  nvcompBatchedDeflateDecompressOpts_t decompress_opts,
  nvcompAlignmentRequirements_t *alignment_requirements
)
{
  DEFLATE_LOG_WITH_DECOMPRESS_OPTS(
    nvcomp::logBatchedDecompressGetRequiredAlignments,
    decompress_opts,
    alignment_requirements
  );
  DEFLATE_CHECK_DECOMPRESS_OPTS(decompress_opts);
  NVCOMP_CHECK_NOT_NULL(alignment_requirements);

  // Note: The decompress engine (DE) requires 1-byte alignment.

  // For input alignment, see NOTE below.
  // Output is only ever cast to uint8_t* and so has an alignment of 1.
  // The temp buffer is not used, implying the smallest valid alignment, 1.
  // NOTE: As of September 2024, inflate_kernel, which is called by
  // nvcompBatchedDeflateDecompressAsync, attempts to support arbitrary input alignment,
  // but results in hard-to-resolve undefined behavior in the absence of a GZIP
  // header (which is the case here) unless the input is aligned to the size of
  // std::uint32_t. When that issue gets resolved, this alignment should be updated.
  alignment_requirements->input = 4;
  alignment_requirements->output = 1;
  alignment_requirements->temp = 1;

  static_assert(
    ((nvcompDeflateRequiredDecompressionAlignment) & (nvcompDeflateRequiredDecompressionAlignment - 1)) == 0,
    "Minimum decompression alignment is invalid."
  );
  assert(
    std::max(std::max(alignment_requirements->input, alignment_requirements->output), alignment_requirements->temp) <=
    nvcompDeflateRequiredDecompressionAlignment
  );

  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedDeflateDecompressGetTempSizeAsync(
  const size_t num_chunks,
  const size_t max_uncompressed_chunk_bytes,
  nvcompBatchedDeflateDecompressOpts_t decompress_opts,
  size_t *const temp_bytes,
  size_t max_total_uncompressed_bytes
)
{
  DEFLATE_LOG_WITH_DECOMPRESS_OPTS(
    nvcomp::logBatchedDecompressGetTempSizeAsync,
    decompress_opts,
    num_chunks,
    max_uncompressed_chunk_bytes,
    temp_bytes,
    max_total_uncompressed_bytes
  );

  DEFLATE_CHECK_DECOMPRESS_OPTS(decompress_opts);
  NVCOMP_CHECK_NOT_NULL(temp_bytes);
  NVCOMP_CHECK_ALIGNMENT(temp_bytes);
  NVCOMP_CHECK_CHUNK_SIZE(max_uncompressed_chunk_bytes, nvcompDeflateDecompressionMaxAllowedChunkSize);

  try
  {
    gdeflate::decompressGetTempSize(num_chunks, max_uncompressed_chunk_bytes, temp_bytes);

    // Add sorting requirements for hardware decompression
    if (decompress_opts.sort_before_hw_decompress)
    {
      *temp_bytes += get_sort_scratch_req(num_chunks, max_uncompressed_chunk_bytes);
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
    return Check::exception_to_error(e, "nvcompBatchedDeflateDecompressGetTempSizeAsync()");
  }

  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedDeflateDecompressGetTempSizeSync(
  [[maybe_unused]] const void *const *const device_compressed_chunk_ptrs,
  [[maybe_unused]] const size_t *const device_compressed_chunk_bytes,
  size_t num_chunks,
  size_t max_uncompressed_chunk_bytes,
  size_t *temp_bytes,
  size_t max_total_uncompressed_bytes,
  nvcompBatchedDeflateDecompressOpts_t decompress_opts,
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

  return nvcompBatchedDeflateDecompressGetTempSizeAsync(
    num_chunks,
    max_uncompressed_chunk_bytes,
    decompress_opts,
    temp_bytes,
    max_total_uncompressed_bytes
  );
}

nvcompStatus_t nvcompBatchedDeflateDecompressAsync(
  const void *const *device_compressed_chunk_ptrs,
  const size_t *device_compressed_chunk_bytes,
  const size_t *device_uncompressed_buffer_bytes,
  size_t *device_uncompressed_chunk_bytes, // optional
  size_t num_chunks,
  void *const device_temp_ptr,
  size_t temp_bytes,
  void *const *device_uncompressed_chunk_ptrs,
  nvcompBatchedDeflateDecompressOpts_t decompress_opts,
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

  DEFLATE_CHECK_DECOMPRESS_OPTS(decompress_opts);

  try
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
      &decompress_opts,
      nullptr,
      false, /*gzip*/
      stream,
      nullptr /*host_comp_chunk_buffers*/
    );
  }
  catch (const std::exception &e)
  {
    LOG_ERROR("{}", e.what());
    return Check::exception_to_error(e, "nvcompBatchedDeflateDecompressAsync()");
  }
}

nvcompStatus_t nvcompBatchedDeflateDecompressAsyncEx(
  const void *const *device_compressed_chunk_ptrs,
  const size_t *device_compressed_chunk_bytes,
  const size_t *device_uncompressed_buffer_bytes,
  size_t *device_uncompressed_chunk_bytes,
  size_t num_chunks,
  void *const device_temp_ptr,
  size_t temp_bytes,
  void *const *device_uncompressed_chunk_ptrs,
  nvcompStatus_t *device_statuses,
  [[maybe_unused]] nvcompBatchedDeflateCompressOpts_t compress_opts,
  nvcompBatchedDeflateDecompressOpts_t decompress_opts,
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

  DEFLATE_CHECK_DECOMPRESS_OPTS(decompress_opts);

  try
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
      &decompress_opts,
      nullptr,
      false, /*gzip*/
      stream,
      host_comp_chunk_buffers
    );
  }
  catch (const std::exception &e)
  {
    LOG_ERROR("{}", e.what());
    return Check::exception_to_error(e, "nvcompBatchedDeflateDecompressAsyncEx()");
  }
}

nvcompStatus_t nvcompBatchedDeflateGetDecompressSizeAsync(
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
      num_chunks,
      stream
    );
  }
  catch (const std::exception &e)
  {
    LOG_ERROR("{}", e.what());
    return Check::exception_to_error(e, "nvcompBatchedDeflateGetDecompressSizeAsync()");
  }

  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedDeflateCompressGetRequiredAlignments(
  nvcompBatchedDeflateCompressOpts_t format_opts,
  nvcompAlignmentRequirements_t *alignment_requirements
)
{
  DEFLATE_LOG_WITH_COMPRESS_OPTS(nvcomp::logBatchedCompressGetRequiredAlignments, format_opts, alignment_requirements);
  DEFLATE_CHECK_COMPRESS_OPTS(format_opts);
  NVCOMP_CHECK_NOT_NULL(alignment_requirements);

  // Only ever cast to pointer to ((un)signed) char or uint8_t.
  alignment_requirements->input = 1;
  // Cast to unsigned long long* in warp_bitwriter_standard::standard_write with
  // default template parameters. Also cast to uint32_t* in compressAsync.
  alignment_requirements->output = sizeof(unsigned long long);
  // Cast to uint16_t** in compressAsync.
  alignment_requirements->temp = sizeof(std::uint16_t *);

  static_assert(
    ((nvcompDeflateRequiredCompressionAlignment) & (nvcompDeflateRequiredCompressionAlignment - 1)) == 0,
    "Minimum compression alignment is invalid."
  );
  assert(
    std::max(std::max(alignment_requirements->input, alignment_requirements->output), alignment_requirements->temp) <=
    nvcompDeflateRequiredCompressionAlignment
  );

  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedDeflateCompressGetTempSizeAsync(
  const size_t num_chunks,
  const size_t max_uncompressed_chunk_bytes,
  nvcompBatchedDeflateCompressOpts_t format_opts,
  size_t *const temp_bytes,
  const size_t max_total_uncompressed_bytes
) // unused, except for logging
{
  DEFLATE_LOG_WITH_COMPRESS_OPTS(
    nvcomp::logBatchedCompressGetTempSizeAsync,
    format_opts,
    num_chunks,
    max_uncompressed_chunk_bytes,
    temp_bytes,
    max_total_uncompressed_bytes
  );
  DEFLATE_CHECK_COMPRESS_OPTS(format_opts);

  NVCOMP_CHECK_NOT_NULL(temp_bytes);
  NVCOMP_CHECK_ALIGNMENT(temp_bytes);

  NVCOMP_CHECK_CHUNK_SIZE(max_uncompressed_chunk_bytes, nvcompDeflateCompressionMaxAllowedChunkSize);

  try
  {
    gdeflate::gdeflate_compression_algo algo = getDeflateEnumFromFormatOpts(format_opts);
    gdeflate::compressGetTempSize(num_chunks, max_uncompressed_chunk_bytes, temp_bytes, algo);
  }
  catch (const std::exception &e)
  {
    LOG_ERROR("{}", e.what());
    return Check::exception_to_error(e, "nvcompBatchedDeflateCompressGetTempSizeAsync()");
  }

  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedDeflateCompressGetTempSizeSync(
  [[maybe_unused]] const void *const *const device_uncompressed_chunk_ptrs,
  [[maybe_unused]] const size_t *const device_uncompressed_chunk_bytes,
  size_t num_chunks,
  size_t max_uncompressed_chunk_bytes,
  nvcompBatchedDeflateCompressOpts_t compress_opts,
  size_t *temp_bytes,
  size_t max_total_uncompressed_bytes,
  [[maybe_unused]] cudaStream_t stream
)
{
  return nvcompBatchedDeflateCompressGetTempSizeAsync(
    num_chunks,
    max_uncompressed_chunk_bytes,
    compress_opts,
    temp_bytes,
    max_total_uncompressed_bytes
  );
}

nvcompStatus_t nvcompBatchedDeflateCompressGetMaxOutputChunkSize(
  size_t max_uncompressed_chunk_bytes,
  nvcompBatchedDeflateCompressOpts_t format_opts, // unused, except for logging
  size_t *max_compressed_chunk_bytes
)
{
  DEFLATE_LOG_WITH_COMPRESS_OPTS(
    nvcomp::logBatchedCompressGetMaxOutputChunkSize,
    format_opts,
    max_uncompressed_chunk_bytes,
    max_compressed_chunk_bytes
  );
  DEFLATE_CHECK_COMPRESS_OPTS(format_opts);

  NVCOMP_CHECK_NOT_NULL(max_compressed_chunk_bytes);
  NVCOMP_CHECK_ALIGNMENT(max_compressed_chunk_bytes);

  NVCOMP_CHECK_CHUNK_SIZE(max_uncompressed_chunk_bytes, nvcompDeflateCompressionMaxAllowedChunkSize);

  try
  {
    nvcomp_deflate::DeflateCompressGetMaxOutputChunkSize(max_uncompressed_chunk_bytes, max_compressed_chunk_bytes);
  }
  catch (const std::exception &e)
  {
    LOG_ERROR("{}", e.what());
    return Check::exception_to_error(e, "nvcompBatchedDeflateCompressGetMaxOutputChunkSize()");
  }

  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedDeflateCompressAsync(
  const void *const *const device_uncompressed_chunk_ptrs,
  const size_t *const device_uncompressed_chunk_bytes,
  const size_t max_uncompressed_chunk_bytes,
  const size_t num_chunks,
  void *const device_temp_ptr,
  const size_t temp_bytes,
  void *const *const device_compressed_chunk_ptrs,
  size_t *const device_compressed_chunk_bytes,
  nvcompBatchedDeflateCompressOpts_t format_opts,
  nvcompStatus_t *device_statuses, // optional
  cudaStream_t stream
)
{
  DEFLATE_LOG_WITH_COMPRESS_OPTS(
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
  DEFLATE_CHECK_COMPRESS_OPTS(format_opts);

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
  NVCOMP_WRAP_CHECK_FUNC(nvcompBatchedDeflateCompressGetRequiredAlignments, format_opts, &align_reqs);
  cuLibLogger::Logger::Instance().SetMostRecentApi(__func__);

  // Check device pointer alignment
  NVCOMP_CHECK_ALIGNMENT(device_uncompressed_chunk_ptrs);
  NVCOMP_CHECK_ALIGNMENT(device_uncompressed_chunk_bytes);
  NVCOMP_CHECK_ALIGNMENT(device_temp_ptr, align_reqs.temp);
  NVCOMP_CHECK_ALIGNMENT(device_compressed_chunk_ptrs);
  NVCOMP_CHECK_ALIGNMENT(device_compressed_chunk_bytes);
  NVCOMP_CHECK_ALIGNMENT(device_statuses);

  NVCOMP_CHECK_CHUNK_SIZE(max_uncompressed_chunk_bytes, nvcompDeflateCompressionMaxAllowedChunkSize);

  // Note:
  // - Batch size is used as grid size, which CUDA limits to 2^31 - 1
  // - It is not an exact limit, but a good enough threshold
  NVCOMP_CHECK_BATCH_SIZE(num_chunks, std::numeric_limits<int>::max());

  try
  {
    gdeflate::gdeflate_compression_algo algo = getDeflateEnumFromFormatOpts(format_opts);
    bool standard = true;
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
      stream,
      standard
    );
  }
  catch (const std::exception &e)
  {
    LOG_ERROR("{}", e.what());
    return Check::exception_to_error(e, "nvcompBatchedDeflateCompressAsync()");
  }
  return nvcompSuccess;
}
