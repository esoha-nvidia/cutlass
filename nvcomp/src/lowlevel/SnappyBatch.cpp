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

#ifdef USE_NVTX
#include <nvtx3/nvtx3.hpp>
#endif // USE_NVTX

#include "Check.h"
#include "common.h"
#include "CorrectnessChecks.cuh"
#include "CudaUtils.h"
#include "Environment.hpp"
#include "HWDecompress.hpp"
#include "Logging.h"
#include "lowlevel/nvcomp_private.h"
#include "nvcomp.h"
#include "nvcomp/snappy.h"
#include "snappy/constants.cuh"
#include "SnappyBatchKernels.h"
#include "type_macros.h"

using nvcomp::Check;
using namespace nvcomp;

namespace
{

constexpr const char *snappy_decompress_opts_format_str = "{{backend={:d}}}";

#define SNAPPY_WITH_TRAILING_DECOMPRESS_OPTS_ARGS(callable, opts, ...) callable(__VA_ARGS__, int(opts.backend))

#define SNAPPY_LOG_WITH_DECOMPRESS_OPTS(log_func, opts, ...)                                                           \
  SNAPPY_WITH_TRAILING_DECOMPRESS_OPTS_ARGS(log_func, opts, __func__, __VA_ARGS__, snappy_decompress_opts_format_str)

#define SNAPPY_CHECK_COMPRESS_OPTS(opts) NVCOMP_WRAP_CHECK_FUNC(check_compress_opts, opts)

#define SNAPPY_CHECK_DECOMPRESS_OPTS(opts) NVCOMP_WRAP_CHECK_FUNC(check_decompress_opts, opts)

nvcompStatus_t check_compress_opts([[maybe_unused]] const nvcompBatchedSnappyCompressOpts_t &opts)
{
  return nvcompSuccess;
}

// Note: This will always return success, since "supported" will always be true
nvcompStatus_t check_decompress_opts(const nvcompBatchedSnappyDecompressOpts_t &opts)
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
  SNAPPY_WITH_TRAILING_DECOMPRESS_OPTS_ARGS(
    LOG_ERROR,
    opts,
    std::string("Unsupported options: ") + snappy_decompress_opts_format_str
  );
  return nvcompErrorNotSupported;
}

size_t snappy_get_max_compressed_length(size_t source_bytes)
{
  // This is an estimate from the original snappy library
  return 32 + source_bytes + source_bytes / 6;
}

} // namespace

static_assert(sizeof(nvcompBatchedSnappyCompressOpts_t) == 64);
static_assert(sizeof(nvcompBatchedSnappyDecompressOpts_t) == 64);

nvcompStatus_t nvcompBatchedSnappyDecompressGetRequiredAlignments(
  nvcompBatchedSnappyDecompressOpts_t decompress_opts,
  nvcompAlignmentRequirements_t *alignment_requirements
)
{
  SNAPPY_LOG_WITH_DECOMPRESS_OPTS(
    nvcomp::logBatchedDecompressGetRequiredAlignments,
    decompress_opts,
    alignment_requirements
  );
  SNAPPY_CHECK_DECOMPRESS_OPTS(decompress_opts);
  NVCOMP_CHECK_NOT_NULL(alignment_requirements);

  // Note: The decompress engine (DE) requires 1-byte alignment.

  // Input and output are only ever cast to uint8_t*, so 1.
  // The temp buffer is not used, implying the smallest valid alignment, 1.
  alignment_requirements->input = 1;
  alignment_requirements->output = 1;
  alignment_requirements->temp = 1;

  static_assert(
    ((nvcompSnappyRequiredDecompressionAlignment) & (nvcompSnappyRequiredDecompressionAlignment - 1)) == 0,
    "Minimum decompression alignment is invalid."
  );
  assert(
    std::max(std::max(alignment_requirements->input, alignment_requirements->output), alignment_requirements->temp) <=
    nvcompSnappyRequiredDecompressionAlignment
  );

  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedSnappyDecompressGetTempSize(
  size_t num_chunks,
  size_t max_uncompressed_chunk_bytes,
  nvcompBatchedSnappyDecompressOpts_t decompress_opts,
  size_t *temp_bytes,
  size_t max_total_uncompressed_bytes, // unused, except for logging
  cudaStream_t stream // unused, except for logging
)
{
  SNAPPY_LOG_WITH_DECOMPRESS_OPTS(
    nvcomp::logBatchedDecompressGetTempSize,
    decompress_opts,
    num_chunks,
    max_uncompressed_chunk_bytes,
    temp_bytes,
    max_total_uncompressed_bytes,
    stream
  );

  SNAPPY_CHECK_DECOMPRESS_OPTS(decompress_opts);
  NVCOMP_CHECK_NOT_NULL(temp_bytes);
  NVCOMP_CHECK_ALIGNMENT(temp_bytes);
  NVCOMP_CHECK_CHUNK_SIZE(max_uncompressed_chunk_bytes, nvcompSnappyDecompressionMaxAllowedChunkSize);

  // Snappy doesn't need any workspace in GPU memory for decompression itself
  *temp_bytes = 0;

  // Allocating temporary space for sorting
  if (decompress_opts.sort_before_hw_decompress)
  {
    *temp_bytes += get_sort_scratch_req(num_chunks, max_uncompressed_chunk_bytes);
  }

  // Allocating temporary space for correctness checks
  const auto snappy_correctness_env = nvcomp::getenv(CHECK_SNAPPY_CORRECTNESS_ENV);
  const bool check_snappy_correctness = not(snappy_correctness_env.empty() or snappy_correctness_env == "0");
  if (check_snappy_correctness)
  {
    // If correctness checks are enabled, we need to allocate space for
    // the device status array.
    *temp_bytes += sizeof(snappy::SnappyCorrectnessChecker<true>) * num_chunks +
                   alignof(snappy::SnappyCorrectnessChecker<true>) - 1;
  }

  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedSnappyDecompressGetTempSizeSync(
  [[maybe_unused]] const void *const *const device_compressed_chunk_ptrs,
  [[maybe_unused]] const size_t *const device_compressed_chunk_bytes,
  size_t num_chunks,
  size_t max_uncompressed_chunk_bytes,
  size_t *temp_bytes,
  size_t max_total_uncompressed_bytes,
  nvcompBatchedSnappyDecompressOpts_t decompress_opts,
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

  return nvcompBatchedSnappyDecompressGetTempSize(
    num_chunks,
    max_uncompressed_chunk_bytes,
    decompress_opts,
    temp_bytes,
    max_total_uncompressed_bytes,
    stream
  );
}

nvcompStatus_t nvcompBatchedSnappyGetDecompressSizeAsync(
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
    nvcomp::gpu_get_uncompressed_sizes(
      device_compressed_chunk_ptrs,
      device_compressed_chunk_bytes,
      device_uncompressed_chunk_bytes,
      num_chunks,
      stream
    );

    // Note: In testing, the above function never throws
  }
  catch (const std::exception &e)
  {
    LOG_ERROR("{}", e.what());
    return Check::exception_to_error(e, "nvcompBatchedSnappyGetDecompressSizeAsync()");
  }

  return nvcompSuccess;
}

bool launch_hw_decomp(
  const void *const *device_compressed_chunk_ptrs,
  const size_t *device_compressed_chunk_bytes,
  const size_t *device_uncompressed_buffer_bytes,
  size_t *device_uncompressed_chunk_bytes,
  size_t num_chunks,
  void *const device_temp_ptr,
  const size_t temp_bytes,
  void *const *device_uncompressed_chunk_ptrs,
  nvcompBatchedSnappyDecompressOpts_t decompress_opts,
  nvcompStatus_t *device_statuses,
  cudaStream_t stream,
  const void *const *host_comp_chunk_buffers,
  const bool force_hw_decomp
)
{
  bool host_setup_mode = host_comp_chunk_buffers != nullptr;
  bool force_sync = host_setup_mode;
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
        true, // do_parse_header: Snappy frame has variable size header to skip
        decompress_opts.sort_before_hw_decompress,
        CU_MEM_DECOMPRESS_ALGORITHM_SNAPPY,
        host_comp_chunk_buffers,
        stream,
        force_sync
      );
    }
    else
    {
      gpu_fill_cudecomp_params(
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
    nvcompFormatType_t::Snappy,
    force_hw_decomp
  );
}

nvcompStatus_t nvcompBatchedSnappyDecompressAsyncEx(
  const void *const *device_compressed_chunk_ptrs,
  const size_t *device_compressed_chunk_bytes,
  const size_t *device_uncompressed_buffer_bytes,
  size_t *device_uncompressed_chunk_bytes,
  size_t num_chunks,
  void *const device_temp_ptr,
  size_t temp_bytes,
  void *const *device_uncompressed_chunk_ptrs,
  nvcompStatus_t *device_statuses,
  [[maybe_unused]] nvcompBatchedSnappyCompressOpts_t compress_opts,
  nvcompBatchedSnappyDecompressOpts_t decompress_opts,
  cudaStream_t stream,
  const void *const *host_comp_chunk_buffers
)
{
#ifdef USE_NVTX
  nvtx3::scoped_range snappy_decomp{"Snappy Decompress"};
#endif // USE_NVTX
  nvcomp::logBatchedDecompressAsync(
    __func__,
    device_compressed_chunk_ptrs,
    device_compressed_chunk_bytes,
    device_uncompressed_buffer_bytes,
    device_uncompressed_chunk_bytes,
    num_chunks,
    device_temp_ptr,
    temp_bytes,
    device_uncompressed_chunk_ptrs, // TODO add decompress_opts here
    decompress_opts.backend,
    device_statuses,
    stream
  );

  // error check inputs
  NVCOMP_CHECK_NOT_NULL(device_compressed_chunk_ptrs);
  NVCOMP_CHECK_NOT_NULL(device_compressed_chunk_bytes);
  NVCOMP_CHECK_NOT_NULL(device_uncompressed_buffer_bytes);
  NVCOMP_CHECK_NOT_NULL(device_uncompressed_chunk_ptrs);
  NVCOMP_CHECK_NOT_NULL(device_uncompressed_chunk_bytes);
  NVCOMP_CHECK_NOT_NULL(device_statuses);

  const auto snappy_correctness_env = nvcomp::getenv(CHECK_SNAPPY_CORRECTNESS_ENV);
  const bool check_snappy_correctness = not(snappy_correctness_env.empty() or snappy_correctness_env == "0");

  if (check_snappy_correctness && device_temp_ptr == nullptr)
  {
    LOG_ERROR("For correctness checks, device_temp_ptr must be provided");
    return nvcompErrorInvalidValue;
  }
  SNAPPY_CHECK_DECOMPRESS_OPTS(decompress_opts);

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

  auto launch_sm_decomp = [&]() -> void {
    auto snappy_decompress_func = check_snappy_correctness ? gpu_unsnap<true> : gpu_unsnap<false>;
    void *device_correctness_ptrs = check_snappy_correctness
                                      ? roundUpToAlignment<snappy::SnappyCorrectnessChecker<true>>(device_temp_ptr)
                                      : nullptr;
    snappy_decompress_func(
      device_compressed_chunk_ptrs,
      device_compressed_chunk_bytes,
      device_uncompressed_chunk_ptrs,
      device_uncompressed_buffer_bytes,
      device_statuses,
      device_correctness_ptrs,
      device_uncompressed_chunk_bytes,
      num_chunks,
      stream
    );
  };

  const auto env_var_value = nvcomp::getenv(USE_HW_DECOMPRESSION_ENV);
  if (!env_var_value.empty())
  {
    decompress_opts.backend = static_cast<bool>(std::stoi(env_var_value)) ? NVCOMP_DECOMPRESS_BACKEND_HARDWARE
                                                                          : NVCOMP_DECOMPRESS_BACKEND_CUDA;
  }
  if (check_snappy_correctness)
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
        exec_on_hw = launch_hw_decomp(
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
          stream,
          host_comp_chunk_buffers,
          false /* force_hw_decomp */
        );
        if (not exec_on_hw)
        {
          LOG_INFO("HW Decompression failed, launching SM decompression");
          launch_sm_decomp();
        }
        break;
      case NVCOMP_DECOMPRESS_BACKEND_HARDWARE:
        LOG_INFO("Launching HW decompression");
        // try HW decomp only, if it fails return an error
        exec_on_hw = launch_hw_decomp(
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
          stream,
          host_comp_chunk_buffers,
          true /* force_hw_decomp */
        );
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
  }
  catch (const std::exception &e)
  {
    LOG_ERROR("{}", e.what());
    return Check::exception_to_error(e, "nvcompBatchedSnappyDecompressAsync()");
  }
  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedSnappyDecompressAsync(
  const void *const *device_compressed_chunk_ptrs,
  const size_t *device_compressed_chunk_bytes,
  const size_t *device_uncompressed_buffer_bytes,
  size_t *device_uncompressed_chunk_bytes,
  size_t num_chunks,
  void *const device_temp_ptr,
  const size_t temp_bytes,
  void *const *device_uncompressed_chunk_ptrs,
  nvcompBatchedSnappyDecompressOpts_t decompress_opts,
  nvcompStatus_t *device_statuses,
  cudaStream_t stream
)
{
  return nvcompBatchedSnappyDecompressAsyncEx(
    device_compressed_chunk_ptrs,
    device_compressed_chunk_bytes,
    device_uncompressed_buffer_bytes,
    device_uncompressed_chunk_bytes,
    num_chunks,
    device_temp_ptr,
    temp_bytes,
    device_uncompressed_chunk_ptrs,
    device_statuses,
    nvcompBatchedSnappyCompressDefaultOpts,
    decompress_opts,
    stream,
    nullptr /*host_comp_chunk_buffers*/
  );
}

nvcompStatus_t nvcompBatchedSnappyCompressGetRequiredAlignments(
  [[maybe_unused]] nvcompBatchedSnappyCompressOpts_t compress_opts,
  nvcompAlignmentRequirements_t *alignment_requirements
)
{
  NVCOMP_CHECK_NOT_NULL(alignment_requirements);
  // Very simple, temporary buffer is not used and input and output are only
  // ever cast to uint8_t.
  alignment_requirements->input = 1;
  alignment_requirements->output = 1;
  alignment_requirements->temp = 1;

  static_assert(
    ((nvcompSnappyRequiredCompressionAlignment) & (nvcompSnappyRequiredCompressionAlignment - 1)) == 0,
    "Minimum compression alignment is invalid."
  );
  assert(
    std::max(std::max(alignment_requirements->input, alignment_requirements->output), alignment_requirements->temp) <=
    nvcompSnappyRequiredCompressionAlignment
  );

  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedSnappyCompressGetTempSize(
  const size_t num_chunks, // unused, except for logging
  const size_t max_uncompressed_chunk_bytes, // unused, except for logging
  const nvcompBatchedSnappyCompressOpts_t compress_opts,
  size_t *const temp_bytes,
  const size_t max_total_uncompressed_bytes, // unused, except for logging
  cudaStream_t stream // unused, except for logging
)
{
  nvcomp::logBatchedCompressGetTempSize(
    __func__,
    num_chunks,
    max_uncompressed_chunk_bytes,
    temp_bytes,
    max_total_uncompressed_bytes,
    stream
  );

  // error check inputs
  SNAPPY_CHECK_COMPRESS_OPTS(compress_opts);
  NVCOMP_CHECK_NOT_NULL(temp_bytes);
  NVCOMP_CHECK_ALIGNMENT(temp_bytes);

  // Snappy doesn't need any workspace in GPU memory
  *temp_bytes = 0;

  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedSnappyCompressGetTempSizeSync(
  [[maybe_unused]] const void *const *const device_uncompressed_chunk_ptrs,
  [[maybe_unused]] const size_t *const device_uncompressed_chunk_bytes,
  size_t num_chunks,
  size_t max_uncompressed_chunk_bytes,
  nvcompBatchedSnappyCompressOpts_t compress_opts,
  size_t *temp_bytes,
  size_t max_total_uncompressed_bytes,
  cudaStream_t stream
)
{
  return nvcompBatchedSnappyCompressGetTempSize(
    num_chunks,
    max_uncompressed_chunk_bytes,
    compress_opts,
    temp_bytes,
    max_total_uncompressed_bytes,
    stream
  );
}

nvcompStatus_t nvcompBatchedSnappyCompressGetMaxOutputChunkSize(
  const size_t max_uncompressed_chunk_bytes,
  [[maybe_unused]] const nvcompBatchedSnappyCompressOpts_t compress_opts,
  size_t *const max_compressed_chunk_bytes
)
{
  nvcomp::logBatchedCompressGetMaxOutputChunkSize(__func__, max_uncompressed_chunk_bytes, max_compressed_chunk_bytes);

  // error check inputs
  NVCOMP_CHECK_NOT_NULL(max_compressed_chunk_bytes);
  NVCOMP_CHECK_ALIGNMENT(max_compressed_chunk_bytes);

  NVCOMP_CHECK_CHUNK_SIZE(max_uncompressed_chunk_bytes, nvcompSnappyCompressionMaxAllowedChunkSize);

  try
  {
    *max_compressed_chunk_bytes = snappy_get_max_compressed_length(max_uncompressed_chunk_bytes);
  }
  catch (const std::exception &e)
  {
    LOG_ERROR("{}", e.what());
    return Check::exception_to_error(e, "nvcompBatchedSnappyCompressGetOutputSize()");
  }

  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedSnappyCompressAsync(
  const void *const *device_uncompressed_chunk_ptrs,
  const size_t *device_uncompressed_chunk_bytes,
  size_t max_uncompressed_chunk_bytes, // unused, except for logging
  size_t num_chunks,
  void *device_temp_ptr, // unused, except for logging
  size_t temp_bytes, // unused, except for logging
  void *const *device_compressed_chunk_ptrs,
  size_t *device_compressed_chunk_bytes,
  [[maybe_unused]] const nvcompBatchedSnappyCompressOpts_t compress_opts,
  nvcompStatus_t *device_statuses, // optional
  cudaStream_t stream
)
{
  nvcomp::logBatchedCompressAsync(
    __func__,
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

  // Check device pointer is not NULL
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

  try
  {
    size_t *device_out_available_bytes = nullptr;

    nvcomp::gpu_snap(
      device_uncompressed_chunk_ptrs,
      device_uncompressed_chunk_bytes,
      device_compressed_chunk_ptrs,
      device_out_available_bytes,
      device_statuses,
      device_compressed_chunk_bytes,
      num_chunks,
      stream
    );

    // Note: In testing, the above function never throws
  }
  catch (const std::exception &e)
  {
    LOG_ERROR("{}", e.what());
    return Check::exception_to_error(e, "nvcompBatchedSnappyCompressAsync()");
  }

  return nvcompSuccess;
}
