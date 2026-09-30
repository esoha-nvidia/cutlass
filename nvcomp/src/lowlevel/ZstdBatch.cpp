/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES.
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

#include "allocators/host_pinned.hpp"
#include "Check.h"
#include "common.h"
#include "CudaUtils.h"
#include "exception.hpp"
#include "Logging.h"
#include "lowlevel/nvcomp_private.h"
#include "nvcomp.h"
#include "nvcomp/zstd.h"
#include "zstd/constants.cuh"
#include "zstd/zstdKernels.cuh"

using namespace nvcomp;
using namespace zstd;

namespace
{

constexpr const char *zstd_decompress_opts_format_str = "{{backend={:d}}}";

#define ZSTD_PRINT_SYNC_DECOMP_TEMP_SIZE_INFO 0

#define ZSTD_WITH_TRAILING_DECOMPRESS_OPTS_ARGS(callable, opts, ...) callable(__VA_ARGS__, int(opts.backend))

#define ZSTD_LOG_WITH_DECOMPRESS_OPTS(log_func, opts, ...)                                                             \
  ZSTD_WITH_TRAILING_DECOMPRESS_OPTS_ARGS(log_func, opts, __func__, __VA_ARGS__, zstd_decompress_opts_format_str)

#define ZSTD_CHECK_COMPRESS_OPTS(opts) NVCOMP_WRAP_CHECK_FUNC(check_compress_opts, opts)

#define ZSTD_CHECK_DECOMPRESS_OPTS(opts) NVCOMP_WRAP_CHECK_FUNC(check_decompress_opts, opts)

nvcompStatus_t check_compress_opts([[maybe_unused]] const nvcompBatchedZstdCompressOpts_t &opts)
{
  return nvcompSuccess;
}

nvcompStatus_t check_decompress_opts(const nvcompBatchedZstdDecompressOpts_t &opts)
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
      LOG_ERROR("Hardware backend is not available for Zstd.");
      return nvcompErrorInvalidValue;
    default:
      LOG_ERROR(
        "Decompress backend invalid. Choose either NVCOMP_DECOMPRESS_BACKEND_DEFAULT or NVCOMP_DECOMPRESS_BACKEND_CUDA."
      );
      return nvcompErrorInvalidValue;
  }

  // Other errors
  ZSTD_WITH_TRAILING_DECOMPRESS_OPTS_ARGS(
    LOG_ERROR,
    opts,
    std::string("Unsupported options: ") + zstd_decompress_opts_format_str
  );
  return nvcompErrorNotSupported;
}

} // namespace

static_assert(sizeof(nvcompBatchedZstdCompressOpts_t) == 64);
static_assert(sizeof(nvcompBatchedZstdDecompressOpts_t) == 64);

nvcompStatus_t nvcompBatchedZstdDecompressGetRequiredAlignments(
  nvcompBatchedZstdDecompressOpts_t decompress_opts,
  nvcompAlignmentRequirements_t *alignment_requirements
)
{
  ZSTD_LOG_WITH_DECOMPRESS_OPTS(
    nvcomp::logBatchedDecompressGetRequiredAlignments,
    decompress_opts,
    alignment_requirements
  );
  ZSTD_CHECK_DECOMPRESS_OPTS(decompress_opts);
  NVCOMP_CHECK_NOT_NULL(alignment_requirements);

  // Input and output are only ever cast to uint8_t*, so 1.
  // Temp is cast to uint64_t in DecompressionScratchHandle::setup_buffer, so 8.
  alignment_requirements->input = 1;
  alignment_requirements->output = 1;
  alignment_requirements->temp = 8;

  static_assert(
    ((nvcompZstdRequiredDecompressionAlignment) & (nvcompZstdRequiredDecompressionAlignment - 1)) == 0,
    "Minimum decompression alignment is invalid."
  );
  assert(
    std::max(std::max(alignment_requirements->input, alignment_requirements->output), alignment_requirements->temp) <=
    nvcompZstdRequiredDecompressionAlignment
  );

  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedZstdDecompressGetTempSize(
  size_t num_chunks,
  size_t max_uncompressed_chunk_bytes,
  nvcompBatchedZstdDecompressOpts_t decompress_opts,
  size_t *temp_bytes,
  size_t max_total_uncompressed_bytes,
  cudaStream_t stream // unused, except for logging
)
{
  ZSTD_LOG_WITH_DECOMPRESS_OPTS(
    nvcomp::logBatchedDecompressGetTempSize,
    decompress_opts,
    num_chunks,
    max_uncompressed_chunk_bytes,
    temp_bytes,
    max_total_uncompressed_bytes,
    stream
  );

  // Error check inputs
  ZSTD_CHECK_DECOMPRESS_OPTS(decompress_opts);
  NVCOMP_CHECK_NOT_NULL(temp_bytes);
  NVCOMP_CHECK_ALIGNMENT(temp_bytes);
  NVCOMP_CHECK_CHUNK_SIZE(max_uncompressed_chunk_bytes, nvcompZstdDecompressionMaxAllowedChunkSize);

  try
  {
    // Estimating the temporary scratch space required heuristically
    *temp_bytes = zstd::get_reqd_tmp_buffer_size(nvcomp::narrow_cast<int>(num_chunks), max_total_uncompressed_bytes);

    // Note: this is never accessed, as "get_reqd_tmp_buffer_size" cannot throw.
  }
  catch (const std::exception &e)
  {
    LOG_ERROR("{}", e.what());
    return Check::exception_to_error(e, "nvcompBatchedZstdDecompressGetTempSize()");
  }

  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedZstdDecompressGetTempSizeSync(
  const void *const *const device_compressed_chunk_ptrs,
  const size_t *const device_compressed_chunk_bytes, // unused, except for logging
  size_t num_chunks,
  size_t max_uncompressed_chunk_bytes,
  size_t *temp_bytes,
  size_t max_total_uncompressed_bytes, // unused, except for logging
  nvcompBatchedZstdDecompressOpts_t decompress_opts,
  nvcompStatus_t *device_statuses,
  cudaStream_t stream
)
{
  nvcomp::logBatchedDecompressGetTempSizeSync(
    __func__,
    device_compressed_chunk_ptrs,
    device_compressed_chunk_bytes,
    num_chunks,
    max_uncompressed_chunk_bytes,
    temp_bytes,
    max_total_uncompressed_bytes,
    decompress_opts.backend,
    device_statuses,
    stream
  );

  ZSTD_CHECK_DECOMPRESS_OPTS(decompress_opts);

  // Check inputs
  NVCOMP_CHECK_NOT_NULL(device_compressed_chunk_ptrs);
  NVCOMP_CHECK_NOT_NULL(temp_bytes);

  // Check pointer alignment
  NVCOMP_CHECK_ALIGNMENT(device_compressed_chunk_ptrs);
  NVCOMP_CHECK_ALIGNMENT(temp_bytes);
  NVCOMP_CHECK_ALIGNMENT(device_statuses);

  // Note:
  // - Batch size is used as grid size, which CUDA limits to 2^31 - 1
  // - The limit is not entirely exact, but it is a good threshold
  NVCOMP_CHECK_BATCH_SIZE(num_chunks, std::numeric_limits<int>::max());
  NVCOMP_CHECK_CHUNK_SIZE(max_uncompressed_chunk_bytes, nvcompZstdDecompressionMaxAllowedChunkSize);

  try
  {
    // If there are no chunks to process,
    // just return the static size without turning to the device.
    if (num_chunks == 0)
    {
      // Setting required scratch size
      *temp_bytes = DecompressionScratchHandle::total_static_size(num_chunks);
      // Clearing device statuses (if available)
      try_clear_device_statuses(num_chunks, device_statuses, stream);
      return nvcompSuccess;
    }

    // Allocate pinned memory for the 2 outputs:
    // - total block count within all available zstd frames
    // - required scratch space for decompressing all blocks (note: no waves are assumed)
    HostPinnedGuard guard(sizeof(size_t) * 2, alignof(size_t), stream);
    size_t *block_count = static_cast<size_t *>(guard.get_ptr());
    size_t *total_dynamic_size = static_cast<size_t *>(block_count + 1);

    // Stream ordered memset to avoid pinned_memory_resource out-of-stream access
    CUDA_CHECK(cudaLaunchHostLambda(
      stream,
      [block_count, total_dynamic_size]() {
        *block_count = 0;
        *total_dynamic_size = 0;
      },
      true /* force_sync -- this is anyway a sync function*/
    ));

    // Note: device_statuses is set by the kernel
    zstd::gather_frame_blocks_api(
      reinterpret_cast<const uint8_t *const *>(device_compressed_chunk_ptrs),
      num_chunks,
      block_count,
      total_dynamic_size,
      device_statuses,
      stream
    );

    CUDA_CHECK(cudaStreamSynchronize(stream));

    size_t block_count_cached = *block_count;
    size_t total_dynamic_size_cached = *total_dynamic_size;

#if ZSTD_PRINT_SYNC_DECOMP_TEMP_SIZE_INFO
    std::cout << "Zstd frames (1): " << num_chunks << std::endl;
    std::cout << "Zstd blocks (1): " << block_count_cached << std::endl;
    std::cout << "Dynamic scratch (B): " << total_dynamic_size_cached << std::endl;
#endif // ZSTD_PRINT_SYNC_DECOMP_TEMP_SIZE_INFO

    // Sanity check(s)
    // Note:
    // There can be certain Zstd blocks that do not require dynamic scratch space at all.
    // If there are as many Zstd frames as many Zstd blocks, then we don't require extra
    // scratch space for storing more DeviceBlockShare structs.
    if (block_count_cached < num_chunks)
    {
      LOG_ERROR("The determined block count is invalid.");
      return nvcompErrorCannotDecompress;
    }

    *temp_bytes = DecompressionScratchHandle::total_static_size(num_chunks) + total_dynamic_size_cached;
    // Note: this is currently never accessed in testing
  }
  catch (const std::exception &e)
  {
    LOG_ERROR("{}", e.what());
    return Check::exception_to_error(e, "nvcompBatchedZstdDecompressGetTempSizeSync()");
  }

  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedZstdGetDecompressSizeAsync(
  const void *const *device_compressed_chunk_ptrs,
  const size_t *device_compressed_chunk_bytes, // unused, except for logging
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
  NVCOMP_CHECK_NOT_NULL(device_uncompressed_chunk_bytes);

  // Check device pointer alignment
  NVCOMP_CHECK_ALIGNMENT(device_compressed_chunk_ptrs);
  NVCOMP_CHECK_ALIGNMENT(device_uncompressed_chunk_bytes);

  try
  {
    zstd::get_frame_sizes_api(
      reinterpret_cast<const uint8_t *const *>(device_compressed_chunk_ptrs),
      device_uncompressed_chunk_bytes,
      nvcomp::narrow_cast<int>(num_chunks),
      stream
    );

    // Note: this is currently never accessed in testing
  }
  catch (const std::exception &e)
  {
    LOG_ERROR("{}", e.what());
    return Check::exception_to_error(e, "nvcompBatchedZstdGetDecompressSizeAsync()");
  }

  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedZstdCompressGetRequiredAlignments(
  [[maybe_unused]] nvcompBatchedZstdCompressOpts_t compress_opts,
  nvcompAlignmentRequirements_t *alignment_requirements
)
{
  NVCOMP_CHECK_NOT_NULL(alignment_requirements);
  // Cast to uint32_t* in lz_compress_tile_greedy_hash.
  alignment_requirements->input = 4;
  // Only every cast to uint8_t*.
  alignment_requirements->output = 1;
  // Temp buffer is explicitly aligned in zstdBatchCompress.
  alignment_requirements->temp = 1;

  static_assert(
    ((nvcompZstdRequiredCompressionAlignment) & (nvcompZstdRequiredCompressionAlignment - 1)) == 0,
    "Minimum compression alignment is invalid."
  );
  assert(
    std::max(std::max(alignment_requirements->input, alignment_requirements->output), alignment_requirements->temp) <=
    nvcompZstdRequiredCompressionAlignment
  );

  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedZstdCompressGetTempSize(
  const size_t num_chunks,
  const size_t max_uncompressed_chunk_bytes,
  const nvcompBatchedZstdCompressOpts_t compress_opts,
  size_t *temp_bytes,
  size_t max_total_uncompressed_bytes,
  cudaStream_t stream
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

  ZSTD_CHECK_COMPRESS_OPTS(compress_opts);
  NVCOMP_CHECK_NOT_NULL(temp_bytes);
  NVCOMP_CHECK_ALIGNMENT(temp_bytes);
  NVCOMP_CHECK_CHUNK_SIZE(max_uncompressed_chunk_bytes, nvcompZstdCompressionMaxAllowedChunkSize);

  try
  {
    *temp_bytes = zstd::compress_compute_temp_size(
      num_chunks,
      max_uncompressed_chunk_bytes,
      max_total_uncompressed_bytes,
      CudaUtils::get_sm_count(stream)
    );
  }
  catch (const std::exception &e)
  {
    LOG_ERROR("{}", e.what());
    return Check::exception_to_error(e, "nvcompBatchedZstdCompressGetTempSize()");
  }

  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedZstdCompressGetTempSizeSync(
  [[maybe_unused]] const void *const *const device_uncompressed_chunk_ptrs,
  [[maybe_unused]] const size_t *const device_uncompressed_chunk_bytes,
  size_t num_chunks,
  size_t max_uncompressed_chunk_bytes,
  nvcompBatchedZstdCompressOpts_t compress_opts,
  size_t *temp_bytes,
  size_t max_total_uncompressed_bytes,
  cudaStream_t stream
)
{
  return nvcompBatchedZstdCompressGetTempSize(
    num_chunks,
    max_uncompressed_chunk_bytes,
    compress_opts,
    temp_bytes,
    max_total_uncompressed_bytes,
    stream
  );
}

nvcompStatus_t nvcompBatchedZstdCompressGetMaxOutputChunkSize(
  const size_t max_uncompressed_chunk_bytes,
  [[maybe_unused]] const nvcompBatchedZstdCompressOpts_t compress_opts,
  size_t *const max_compressed_chunk_bytes
)
{
  nvcomp::logBatchedCompressGetMaxOutputChunkSize(__func__, max_uncompressed_chunk_bytes, max_compressed_chunk_bytes);

  NVCOMP_CHECK_CHUNK_SIZE(max_uncompressed_chunk_bytes, nvcompZstdCompressionMaxAllowedChunkSize);
  NVCOMP_CHECK_NOT_NULL(max_compressed_chunk_bytes);
  NVCOMP_CHECK_ALIGNMENT(max_compressed_chunk_bytes);

  try
  {
    *max_compressed_chunk_bytes = zstd::compute_max_comp_output_size(max_uncompressed_chunk_bytes);
    // Note: this is never accessed in testing
  }
  catch (const std::exception &e)
  {
    LOG_ERROR("{}", e.what());
    return Check::exception_to_error(e, "nvcompBatchedZstdCompressGetMaxOutputChunkSize()");
  }

  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedZstdDecompressAsync(
  const void *const *device_compressed_chunk_ptrs,
  const size_t *device_compressed_chunk_bytes, // unused, except for logging
  const size_t *device_uncompressed_buffer_bytes,
  size_t *device_uncompressed_chunk_bytes,
  size_t num_chunks,
  void *device_temp_ptr,
  const size_t temp_bytes,
  void *const *device_uncompressed_chunk_ptrs,
  nvcompBatchedZstdDecompressOpts_t decompress_opts,
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

  // error check inputs
  NVCOMP_CHECK_NOT_NULL(device_compressed_chunk_ptrs);
  NVCOMP_CHECK_NOT_NULL(device_uncompressed_buffer_bytes);
  if (temp_bytes > 0)
  {
    NVCOMP_CHECK_NOT_NULL(device_temp_ptr);
  }
  NVCOMP_CHECK_NOT_NULL(device_uncompressed_chunk_ptrs);
  NVCOMP_CHECK_NOT_NULL(device_uncompressed_chunk_bytes);
  NVCOMP_CHECK_NOT_NULL(device_statuses);

  ZSTD_CHECK_DECOMPRESS_OPTS(decompress_opts);

  nvcompAlignmentRequirements_t align_reqs{};
  NVCOMP_WRAP_CHECK_FUNC(nvcompBatchedZstdDecompressGetRequiredAlignments, decompress_opts, &align_reqs);
  cuLibLogger::Logger::Instance().SetMostRecentApi(__func__);

  // Check device pointer alignment
  NVCOMP_CHECK_ALIGNMENT(device_compressed_chunk_ptrs);
  NVCOMP_CHECK_ALIGNMENT(device_uncompressed_buffer_bytes);
  NVCOMP_CHECK_ALIGNMENT(device_uncompressed_chunk_bytes);
  NVCOMP_CHECK_ALIGNMENT(device_temp_ptr, align_reqs.temp);
  NVCOMP_CHECK_ALIGNMENT(device_uncompressed_chunk_ptrs);
  NVCOMP_CHECK_ALIGNMENT(device_statuses);

  // Note:
  // - Batch size is used as grid size, which CUDA limits to 2^31 - 1
  // - The limit is not entirely exact, but it is a good threshold
  NVCOMP_CHECK_BATCH_SIZE(num_chunks, std::numeric_limits<int>::max());

  try
  {
    LOG_INFO("Launching SM decompression");
    // The temporary buffer includes the "global ix frame" and "tmp_buffer_loc" variables. These need to be set asynchronously
    // before the start of the kernel.
    zstd::DecompressionScratchHandle buffers(reinterpret_cast<uint8_t *>(device_temp_ptr), num_chunks, temp_bytes);
    buffers.init_values(stream);

    // For now, no error checking.
    try_clear_device_statuses(num_chunks, device_statuses, stream);

    zstd::classify_frames_api(
      reinterpret_cast<const uint8_t *const *>(device_compressed_chunk_ptrs),
      num_chunks,
      buffers,
      device_uncompressed_buffer_bytes,
      stream
    );

    zstd::init_tables_api(num_chunks, buffers, stream);

    zstd::decompress_frames_api(
      reinterpret_cast<uint8_t *const *>(device_uncompressed_chunk_ptrs),
      device_uncompressed_chunk_bytes,
      device_uncompressed_buffer_bytes,
      num_chunks,
      buffers,
      stream
    );

    // Note: this is never accessed in testing
  }
  catch (const std::exception &e)
  {
    LOG_ERROR("{}", e.what());
    return Check::exception_to_error(e, "nvcompBatchedZstdDecompressAsync()");
  }

  return nvcompSuccess;
}
nvcompStatus_t nvcompBatchedZstdDecompressAsyncEx(
  const void *const *device_compressed_chunk_ptrs,
  const size_t *device_compressed_chunk_bytes,
  const size_t *device_uncompressed_buffer_bytes,
  size_t *device_uncompressed_chunk_bytes,
  size_t num_chunks,
  void *const device_temp_ptr,
  size_t temp_bytes,
  void *const *device_uncompressed_chunk_ptrs,
  nvcompStatus_t *device_statuses,
  [[maybe_unused]] nvcompBatchedZstdCompressOpts_t compress_opts,
  nvcompBatchedZstdDecompressOpts_t decompress_opts,
  cudaStream_t stream,
  [[maybe_unused]] const void *const *host_comp_chunk_buffers
)
{
  return nvcompBatchedZstdDecompressAsync(
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
nvcompStatus_t nvcompBatchedZstdCompressAsync(
  const void *const *const device_uncompressed_chunk_ptrs,
  const size_t *const device_uncompressed_chunk_bytes,
  const size_t max_uncompressed_chunk_bytes,
  const size_t num_chunks,
  void *const device_temp_ptr,
  const size_t temp_bytes,
  void *const *const device_compressed_chunk_ptrs,
  size_t *const device_compressed_chunk_bytes,
  [[maybe_unused]] const nvcompBatchedZstdCompressOpts_t compress_opts,
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
  if (temp_bytes > 0)
  {
    NVCOMP_CHECK_NOT_NULL(device_temp_ptr);
  }
  NVCOMP_CHECK_NOT_NULL(device_compressed_chunk_ptrs);
  NVCOMP_CHECK_NOT_NULL(device_compressed_chunk_bytes);

  nvcompAlignmentRequirements_t align_reqs{};
  NVCOMP_WRAP_CHECK_FUNC(nvcompBatchedZstdCompressGetRequiredAlignments, {}, &align_reqs);
  cuLibLogger::Logger::Instance().SetMostRecentApi(__func__);

  // Check device pointer alignment
  NVCOMP_CHECK_ALIGNMENT(device_uncompressed_chunk_ptrs);
  NVCOMP_CHECK_ALIGNMENT(device_uncompressed_chunk_bytes);
  NVCOMP_CHECK_ALIGNMENT(device_temp_ptr, align_reqs.temp);
  NVCOMP_CHECK_ALIGNMENT(device_compressed_chunk_ptrs);
  NVCOMP_CHECK_ALIGNMENT(device_compressed_chunk_bytes);
  NVCOMP_CHECK_ALIGNMENT(device_statuses);

  NVCOMP_CHECK_CHUNK_SIZE(max_uncompressed_chunk_bytes, nvcompZstdCompressionMaxAllowedChunkSize);

  // Note:
  // - Batch size is used as grid size, which CUDA limits to 2^31 - 1
  // - The limit is not entirely exact, but it is a good threshold
  NVCOMP_CHECK_BATCH_SIZE(num_chunks, std::numeric_limits<int>::max());

  try
  {
    zstdBatchCompress(
      reinterpret_cast<const uint8_t *const *>(device_uncompressed_chunk_ptrs),
      device_uncompressed_chunk_bytes,
      max_uncompressed_chunk_bytes,
      num_chunks,
      reinterpret_cast<uint8_t *>(device_temp_ptr),
      temp_bytes,
      reinterpret_cast<uint8_t *const *>(device_compressed_chunk_ptrs),
      device_compressed_chunk_bytes,
      device_statuses,
      stream
    );

    // Note: this is never accessed in testing
  }
  catch (const std::exception &e)
  {
    LOG_ERROR("{}", e.what());
    return Check::exception_to_error(e, "nvcompBatchedZstdCompressAsync()");
  }

  return nvcompSuccess;
}
