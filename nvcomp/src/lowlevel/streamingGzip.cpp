/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2024 NVIDIA CORPORATION & AFFILIATES.
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

#include <fstream>

#include "Check.h"
#include "CudaUtils.h"
#include "exception.hpp"
#include "gzip/GzipKernels.cuh"
#include "lookahead_gzip/include/lookahead_gzip.h"
#include "lookahead_gzip/include/lookahead_gzip_client.h"
#include "nvcomp/shared_types.h"

#include <nvcomp/native/streaming_gzip.hpp>

using namespace nvcomp;
using namespace lookahead_gzip;

namespace
{
// Builds a lookahead streaming-decompress config (CUDA streams + cached device props) and tears it down
// when it leaves scope.
struct LookaheadConfigGuard
{
  lookaheadGzipConfig_t config{};

  explicit LookaheadConfigGuard(cudaStream_t stream) { LookaheadGzipStreamingClient::createConfig(&config, stream); }

  ~LookaheadConfigGuard()
  {
    try
    {
      LookaheadGzipStreamingClient::freeConfig(&config);
    }
    catch (const std::exception &e)
    {
      // Logging routes through fmt, which can itself throw.
      try
      {
        LOG_ERROR("{}", e.what());
      }
      catch (...)
      {}
    }
    catch (...)
    {}
  }

  LookaheadConfigGuard(const LookaheadConfigGuard &) = delete;
  LookaheadConfigGuard &operator=(const LookaheadConfigGuard &) = delete;
};
} // namespace

nvcompStatus_t nvcompGzipStreamingDecompressGetTempSize(size_t *temp_bytes)
{
  NVCOMP_CHECK_NOT_NULL(temp_bytes);

  try
  {
    // this will get an upper bound for the required memory, in reality this
    // might be more than required but passing the stream all the way here would
    // require an API change
    LookaheadConfigGuard cfg(nullptr);
    LookaheadGzipStreamingClient::decompressGetTempSize(&cfg.config, temp_bytes);
  }
  catch (const NVCompException &e)
  {
    LOG_ERROR("{}", e.what());
    return e.get_error();
  }
  catch (const std::exception &e)
  {
    LOG_ERROR("{}", e.what());
    return nvcompErrorInvalidValue;
  }
  return nvcompSuccess;
}

nvcompStatus_t nvcompGzipStreamingDecompress(
  std::istream &input_stream,
  std::ostream &output_stream,
  const size_t temp_bytes,
  void *const device_temp_ptr,
  cudaStream_t stream
)
{
  // error check inputs
  NVCOMP_CHECK_NOT_NULL(device_temp_ptr);

  // Check device pointer alignment
  nvcompAlignmentRequirements_t align_reqs{};
  NVCOMP_WRAP_CHECK_FUNC(nvcompLookaheadGzipDecompressGetRequiredAlignments, &align_reqs);

  NVCOMP_CHECK_ALIGNMENT(device_temp_ptr, align_reqs.temp);

  try
  {
    LookaheadConfigGuard cfg(stream);
    LookaheadGzipStreamingClient client(&cfg.config, temp_bytes, reinterpret_cast<uint8_t *>(device_temp_ptr));
    client.reset(stream);

    // Note:
    // We need to write a bit of data to the ring buffer, so that the
    // GPU can start cracking on it right away when we start the kernel.
    if (client.writeStart(input_stream) == 0)
    {
      return nvcompErrorCannotDecompress;
    }
    client.decompress(reinterpret_cast<uint8_t *>(device_temp_ptr), nullptr, stream);
    client.writeRemainder(input_stream);

    while (!client.is_done())
    {
      size_t iter_bytes_decompressed = 0;
      while (iter_bytes_decompressed == 0 && !client.is_done())
      {
        if (client.read(output_stream, iter_bytes_decompressed) != nvcompSuccess)
        {
          throw std::runtime_error("Failed to decompress input stream");
        }
      }
      client.advance(iter_bytes_decompressed);
    }
  }
  catch (const NVCompException &e)
  {
    LOG_ERROR("{}", e.what());
    return e.get_error();
  }
  catch (const std::exception &e)
  {
    LOG_ERROR("{}", e.what());
    return nvcompErrorInvalidValue;
  }
  return nvcompSuccess;
}

nvcompStatus_t nvcompGzipStreamingCompressGetTempSize(nvcompBatchedGzipCompressOpts_t opts, size_t *temp_bytes)
{
  NVCOMP_CHECK_NOT_NULL(temp_bytes);

  try
  {
    *temp_bytes = gzipStreamingCompressTempSize(opts);
  }
  catch (const NVCompException &e)
  {
    LOG_ERROR("{}", e.what());
    return e.get_error();
  }
  catch (const std::exception &e)
  {
    LOG_ERROR("{}", e.what());
    return nvcompErrorInvalidValue;
  }
  return nvcompSuccess;
}

nvcompStatus_t nvcompGzipStreamingCompress(
  std::istream &input_stream,
  std::ostream &output_stream,
  const size_t temp_bytes,
  void *const device_temp_ptr,
  nvcompBatchedGzipCompressOpts_t opts,
  cudaStream_t stream
)
{
  NVCOMP_CHECK_NOT_NULL(device_temp_ptr);

  // Check device pointer alignment
  nvcompAlignmentRequirements_t align_reqs{};
  NVCOMP_WRAP_CHECK_FUNC(nvcompBatchedGzipCompressGetRequiredAlignments, opts, &align_reqs);

  NVCOMP_CHECK_ALIGNMENT(device_temp_ptr, align_reqs.temp);

  try
  {
    gzipStreamingCompress(input_stream, output_stream, opts, device_temp_ptr, temp_bytes, stream);
  }
  catch (const NVCompException &e)
  {
    LOG_ERROR("{}", e.what());
    return e.get_error();
  }
  catch (const std::exception &e)
  {
    LOG_ERROR("{}", e.what());
    return nvcompErrorInvalidValue;
  }
  return nvcompSuccess;
}
