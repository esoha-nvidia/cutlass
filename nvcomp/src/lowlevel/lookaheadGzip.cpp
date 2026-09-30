/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
*/

#include <cuda/atomic>

#include <algorithm>
#include <cassert>
#include <fstream>

#include "Check.h"
#include "CudaUtils.h"
#include "exception.hpp"
#include "gzip/GzipKernels.cuh"
#include "lookahead_gzip/include/lookahead_gzip.h"
#include "lookahead_gzip/include/lookahead_gzip_client.h"
#include "nvcomp/gzip.h"
#include "nvcomp/shared_types.h"

#include <nvcomp/native/streaming_gzip.hpp>

using namespace nvcomp;
using namespace lookahead_gzip;

nvcompStatus_t nvcompLookaheadGzipDecompressGetRequiredAlignments(nvcompAlignmentRequirements_t *alignment_requirements)
{
  NVCOMP_CHECK_NOT_NULL(alignment_requirements);
  // Lookahead gzip processes data as 4-byte integers
  alignment_requirements->input = 4;
  alignment_requirements->output = 1;
  alignment_requirements->temp = 8;

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

nvcompStatus_t nvcompLookaheadGzipDecompressGetTempSize(size_t num_chunks, size_t *temp_bytes)
{
  NVCOMP_CHECK_NOT_NULL(temp_bytes);

  try
  {
    lookaheadGzipConfig_t config;
    LookaheadGzipOneshotClient::createConfig(num_chunks, nullptr, &config);

    LookaheadGzipOneshotClient::decompressGetTempSize(&config, temp_bytes);
  }
  catch (const std::exception &e)
  {
    LOG_ERROR("{}", e.what());
    return nvcomp::Check::exception_to_error(e, "nvcompLookaheadGzipDecompressGetTempSize()");
  }
  return nvcompSuccess;
}

nvcompStatus_t nvcompLookaheadGzipGetDecompressSizeAsync(
  const void *const *device_compressed_ptrs,
  const size_t *device_compressed_bytes,
  size_t *device_uncompressed_chunk_bytes,
  size_t num_chunks,
  nvcompStatus_t *device_statuses,
  cudaStream_t stream
)
{

  // error check inputs
  NVCOMP_CHECK_NOT_NULL(device_compressed_ptrs);
  NVCOMP_CHECK_NOT_NULL(device_compressed_bytes);
  NVCOMP_CHECK_NOT_NULL(device_uncompressed_chunk_bytes);

  try
  {
    return LookaheadGzipOneshotClient::getGzipSize(
      reinterpret_cast<const uint8_t *const *>(device_compressed_ptrs),
      device_compressed_bytes,
      device_uncompressed_chunk_bytes,
      num_chunks,
      device_statuses,
      stream
    );
  }
  catch (const std::exception &e)
  {
    LOG_ERROR("{}", e.what());
    return nvcomp::Check::exception_to_error(e, "nvcompLookaheadGzipGetDecompressSizeAsync()");
  }
  return nvcompSuccess;
}

nvcompStatus_t nvcompLookaheadGzipDecompressAsync(
  const void *const *device_compressed_ptrs,
  const size_t *device_compressed_bytes,
  void *const *device_uncompressed_ptrs,
  const size_t *device_uncompressed_buffer_bytes,
  size_t *device_uncompressed_chunk_bytes,
  void *const device_temp_ptr,
  const size_t temp_bytes,
  const size_t num_chunks,
  nvcompStatus_t *device_statuses,
  cudaStream_t stream
)
{
  // error check inputs
  NVCOMP_CHECK_NOT_NULL(device_compressed_bytes);
  NVCOMP_CHECK_NOT_NULL(device_compressed_ptrs);
  NVCOMP_CHECK_NOT_NULL(device_uncompressed_buffer_bytes);
  NVCOMP_CHECK_NOT_NULL(device_uncompressed_chunk_bytes);
  NVCOMP_CHECK_NOT_NULL(device_uncompressed_ptrs);
  NVCOMP_CHECK_NOT_NULL(device_temp_ptr);
  NVCOMP_CHECK_NOT_NULL(device_statuses);

  NVCOMP_CHECK_BATCH_SIZE(num_chunks, std::numeric_limits<int>::max());

  // Check device pointer alignment
  nvcompAlignmentRequirements_t align_reqs{};
  NVCOMP_WRAP_CHECK_FUNC(nvcompLookaheadGzipDecompressGetRequiredAlignments, &align_reqs);

  NVCOMP_CHECK_ALIGNMENT(device_uncompressed_ptrs);
  NVCOMP_CHECK_ALIGNMENT(device_uncompressed_buffer_bytes);
  NVCOMP_CHECK_ALIGNMENT(device_uncompressed_chunk_bytes);
  NVCOMP_CHECK_ALIGNMENT(device_temp_ptr, align_reqs.temp);
  NVCOMP_CHECK_ALIGNMENT(device_compressed_ptrs);
  NVCOMP_CHECK_ALIGNMENT(device_compressed_bytes);
  NVCOMP_CHECK_ALIGNMENT(device_statuses);

  try
  {
    lookaheadGzipConfig_t config;
    LookaheadGzipOneshotClient::createConfig(num_chunks, stream, &config);

    LookaheadGzipOneshotClient client(&config, temp_bytes, reinterpret_cast<uint8_t *>(device_temp_ptr));

    client.decompress(
      reinterpret_cast<const uint32_t *const *>(device_compressed_ptrs),
      device_compressed_bytes,
      reinterpret_cast<char *const *>(device_uncompressed_ptrs),
      device_uncompressed_buffer_bytes,
      device_uncompressed_chunk_bytes,
      reinterpret_cast<uint8_t *>(device_temp_ptr),
      device_statuses,
      stream
    );
  }
  catch (const std::exception &e)
  {
    LOG_ERROR("{}", e.what());
    return nvcomp::Check::exception_to_error(e, "nvcompLookaheadGzipDecompressAsync()");
  }
  return nvcompSuccess;
}
