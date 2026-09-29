/*
 * Copyright (c) 2021-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <cassert>
#include <stdexcept>

#include "ans.h"
#include "ans/ans_utils.cuh"
#include "ans/compress_kernels_llif.cuh"
#include "ans/defrag.cuh"
#include "common.h"
#include "CudaUtils.h"
#include "device_guard.h"
#include "exception.hpp"
#include "Logging.h"
#include "nvcomp/shared_types.h"
#include "nvcomp_common_deps/hlif_shared_types.hpp"

using namespace ans_gpu_lib;
using namespace ans_gpu_lib::detail;
using nvcomp::cuda_dim_cast;
using nvcomp::CudaUtils;
using nvcomp::DeviceGuard;
using nvcomp::narrow_cast;

namespace ans
{

void compressGetTempSize(
  [[maybe_unused]] size_t num_chunks,
  [[maybe_unused]] size_t max_uncompressed_chunk_size,
  [[maybe_unused]] nvcompBatchedANSCompressOpts_t format_opts,
  size_t *temp_bytes
)
{
  *temp_bytes = 0;
}

void compressGetMaxOutputChunkSize(size_t max_chunk_size, size_t *max_compressed_size)
{
  assert(max_compressed_size != nullptr);
  // make this a multiple of 8 so that we can easily get 8-byte aligned chunks
  // from one allocation. We choose a factor of 1.3 because the size of the output
  // can be approximately upper bounded by DEFAULT_TABLELOG/8 = 1.25.
  *max_compressed_size = size_t(1.3 * max_chunk_size + 4096) & (~7);
}

void get_sub_chunking_config(
  const size_t max_chunk_size,
  const size_t batch_size,
  cudaStream_t stream,
  int &max_sub_chunk_size,
  int requested_max_sub_chunk_count = 0
)
{

  if (requested_max_sub_chunk_count > 0)
  {
    // Convert the requested max sub chunk count into a sub chunk size: round the
    // per-sub-chunk byte size up to the next power of 2, then clamp to the
    // minimum. This is a MAX count: if the user requested more sub chunks than
    // fit at the minimum size, we use the minimum (so the actual count ends up
    // below the request).
    int unrounded_sub_chunk_size =
      nvcomp::narrow_cast<int>(nvcomp::roundUpDiv(max_chunk_size, requested_max_sub_chunk_count));
    max_sub_chunk_size = max(MIN_SUB_CHUNK_SIZE, nvcomp::roundUpPow2(unrounded_sub_chunk_size));
  }
  else
  {
    // find the maximum number of warps that can be resident on the device and then make sure
    // there are enough waves to avoid tail effects
    int num_sms, max_threads_per_sm;
    {
      DeviceGuard device_guard(stream);
      int device_id;
      CUDA_CHECK(cudaGetDevice(&device_id));
      num_sms = CudaUtils::get_sm_count(stream);
      CUDA_CHECK(cudaDeviceGetAttribute(&max_threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, device_id));
    }

    int num_warps = num_sms * (max_threads_per_sm / WARP_SIZE);

    int target_num_warps_per_chunk =
      narrow_cast<int>(nvcomp::roundUpDiv(static_cast<size_t>(num_warps) * NUM_WAVES_PER_SM, batch_size));
    target_num_warps_per_chunk = min(target_num_warps_per_chunk, MAX_SUB_CHUNKS_PER_CHUNK);
    target_num_warps_per_chunk = max(target_num_warps_per_chunk, NUM_COMP_WARPS_PER_CTA);

    // now divide max size by warps per chunk to get the subchunk size and round up to nearest power of 2
    // that allows the number of sub-chunks to fall into allowable bounds
    int unrounded_sub_chunk_size =
      nvcomp::narrow_cast<int>(nvcomp::roundUpDiv(max_chunk_size, target_num_warps_per_chunk));

    uint32_t sub_chunk_size_power = 0;
    while (unrounded_sub_chunk_size >>= 1)
    {
      sub_chunk_size_power++;
    }

    int lower_eq_power_of_2_size = 1 << sub_chunk_size_power;
    int higher_power_of_2_size = 1 << (1 + sub_chunk_size_power);

    if (nvcomp::roundUpDiv(max_chunk_size, lower_eq_power_of_2_size) <= MAX_SUB_CHUNKS_PER_CHUNK)
    {
      max_sub_chunk_size = lower_eq_power_of_2_size;
    }
    else
    {
      max_sub_chunk_size = higher_power_of_2_size;
    }

    // At this point, the target number of sub chunks per chunk is less than or equal to the max, so if the subchunk becomes
    // larger here, it won't take the number of sub chunks per chunk over the max. It is OK if the number of sub chunks per
    // chunk goes below NUM_COMP_WARPS_PER_CTA. The excess warps in the CTA will simply exit. Bad for performance, but we would
    // get a terrible compression ratio anyway if the chunks were small enough for this to happen.
    max_sub_chunk_size = max(MIN_SUB_CHUNK_SIZE, max_sub_chunk_size);
  }
}

void compressAsync(
  const void *const *uncomp_chunks,
  const size_t *uncomp_chunk_sizes,
  const size_t max_chunk_size_bytes,
  size_t batch_size,
  [[maybe_unused]] void *temp_ptr,
  [[maybe_unused]] size_t temp_bytes,
  void *const *comp_chunks,
  size_t *comp_chunk_sizes,
  nvcompBatchedANSCompressOpts_t format_opts,
  nvcompStatus_t *device_statuses,
  cudaStream_t stream
)
{
  int max_sub_chunk_size;
  get_sub_chunking_config(max_chunk_size_bytes, batch_size, stream, max_sub_chunk_size, format_opts.max_sub_chunk_count);

  // mark compression successful
  nvcomp::try_clear_device_statuses(batch_size, device_statuses, stream);

  // Worst-case bytes for one sub-chunk's compressed output (the slot stride the
  // encoders write into). The fused kernels take it as a uint32 word count (slot_words).
  const uint32_t slot_words = static_cast<uint32_t>(get_max_comp_sub_chunk_size(max_sub_chunk_size) / sizeof(uint32_t));
  using ans_gpu_lib::detail::CharEncodePolicy;
  using ans_gpu_lib::detail::Fp16EncodePolicy;
  using ans_gpu_lib::detail::Fp8EncodePolicy;
  decltype(&ans_gpu_lib::detail::compress_kernel<CharEncodePolicy>) kernel = nullptr;
  switch (format_opts.data_type)
  {
    case NVCOMP_TYPE_FLOAT16:
      kernel = ans_gpu_lib::detail::compress_kernel<Fp16EncodePolicy>;
      break;
    case NVCOMP_TYPE_FLOAT8_E4M3:
      kernel = ans_gpu_lib::detail::compress_kernel<Fp8EncodePolicy>;
      break;
    default:
      kernel = ans_gpu_lib::detail::compress_kernel<CharEncodePolicy>;
      break;
  }

  dim3 fused_grid(cuda_dim_cast(batch_size), 1);
  dim3 fused_block(NUM_COMP_WARPS_PER_CTA * WARP_SIZE);
  kernel<<<fused_grid, fused_block, 0, stream>>>(
    comp_chunks,
    (const void *const *)uncomp_chunks,
    uncomp_chunk_sizes,
    max_sub_chunk_size,
    comp_chunk_sizes,
    slot_words
  );
  CUDA_CHECK(cudaGetLastError());
}

} // namespace ans
