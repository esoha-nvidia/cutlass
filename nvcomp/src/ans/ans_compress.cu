/*
 * Copyright (c) 2021-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <algorithm>
#include <cassert>
#include <string>

#include "ans.h"
#include "ans/ans_utils.cuh"
#include "ans/compress_kernels_llif.cuh"
#include "common.h"
#include "exception.hpp"
#include "Logging.h"
#include "nvcomp.hpp"
#include "nvcomp/shared_types.h"
#include "nvcomp_common_deps/hlif_shared_types.hpp"

using namespace ans_gpu_lib;
using namespace ans_gpu_lib::detail;
using nvcomp::cuda_dim_cast;
using nvcomp::narrow_cast;
using nvcomp::roundUpDiv;

namespace ans
{

namespace
{

uint32_t encode_bytes_per_symbol(nvcompType_t data_type)
{
  return ans_bytes_per_symbol(ans_stream_type_from_data_type(data_type));
}

// Every type defaults to two states. Explicit 1/2 always wins.
uint32_t compress_states_per_lane(nvcompType_t data_type, uint8_t states_per_lane)
{
  if (states_per_lane == 1 || states_per_lane == 2)
  {
    return states_per_lane;
  }
  return ans_default_states_per_lane(ans_stream_type_from_data_type(data_type));
}

// Matches compressAsync: power-of-two sub-chunk size, not smaller than
// MIN_SUB_CHUNK_SIZE, so the actual count never exceeds the request.
uint32_t sub_chunk_size_symbols_from_requested_count(uint32_t chunk_symbols, uint32_t requested_count)
{
  const uint32_t unrounded = nvcomp::roundUpDiv(chunk_symbols, requested_count);
  return std::max(MIN_SUB_CHUNK_SIZE, nvcomp::roundUpPow2(unrounded));
}

} // namespace

void compressGetTempSize(
  [[maybe_unused]] size_t num_chunks,
  [[maybe_unused]] size_t max_uncompressed_chunk_size,
  [[maybe_unused]] nvcompBatchedANSCompressOpts_t format_opts,
  size_t *temp_bytes
)
{
  *temp_bytes = 0;
}

void compressGetMaxOutputChunkSize(
  size_t max_chunk_size,
  nvcompBatchedANSCompressOpts_t format_opts,
  size_t *max_compressed_size
)
{
  assert(max_compressed_size != nullptr);

  const uint32_t max_chunk = static_cast<uint32_t>(max_chunk_size);
  const AnsStreamType stream_type = ans_stream_type_from_data_type(format_opts.data_type);
  const uint32_t bytes_per_symbol = ans_bytes_per_symbol(stream_type);
  const uint32_t symbols = nvcomp::roundUpDiv(max_chunk, bytes_per_symbol);
  const uint32_t states = compress_states_per_lane(format_opts.data_type, format_opts.states_per_lane);

  const uint32_t max_sub_chunk_size_symbols =
    sub_chunk_size_symbols_from_requested_count(symbols, resolve_max_sub_chunk_count(format_opts.max_sub_chunk_count));
  const uint32_t nsc = std::min<uint32_t>(
    MAX_SUB_CHUNKS_PER_CHUNK,
    ans_derived_num_sub_chunks(max_chunk, max_sub_chunk_size_symbols * bytes_per_symbol)
  );
  const uint32_t header_bytes =
    ans_sub_chunk_0_offset(stream_type, max_chunk, nsc, 0, static_cast<uint8_t>(NV_MAX_SYMBOL_VALUE));
  *max_compressed_size = nvcomp::roundUpTo(
    header_bytes + nsc * get_max_comp_sub_chunk_size(max_sub_chunk_size_symbols, states, ans_tablelog(stream_type)),
    ANS_SUB_CHUNK_SLOT_ALIGN
  );
}

namespace
{

void launchCompressKernel(
  const void *const *uncomp_chunks,
  const size_t *uncomp_chunk_sizes,
  const size_t max_chunk_size_bytes,
  size_t batch_size,
  void *const *comp_chunks,
  size_t *comp_chunk_sizes,
  nvcompBatchedANSCompressOpts_t format_opts,
  nvcompStatus_t *device_statuses,
  cudaStream_t stream,
  dim3 fused_grid,
  const float *pack_C,
  int pack_ldc,
  int pack_M,
  int pack_N,
  int pack_tile_m,
  int pack_tile_n,
  int pack_mn_swapped
)
{
  const uint32_t bytes_per_symbol = encode_bytes_per_symbol(format_opts.data_type);
  const uint32_t max_chunk_size_symbols = roundUpDiv(narrow_cast<uint32_t>(max_chunk_size_bytes), bytes_per_symbol);
  const uint32_t max_sub_chunk_size = sub_chunk_size_symbols_from_requested_count(
    max_chunk_size_symbols,
    resolve_max_sub_chunk_count(format_opts.max_sub_chunk_count)
  );

  nvcomp::try_clear_device_statuses(batch_size, device_statuses, stream);

  const uint32_t states_per_lane = compress_states_per_lane(format_opts.data_type, format_opts.states_per_lane);

  const uint32_t subchunk_comp_buffer_size = get_max_comp_sub_chunk_size(
    max_sub_chunk_size,
    states_per_lane,
    ans_tablelog(ans_stream_type_from_data_type(format_opts.data_type))
  );
  using ans_gpu_lib::detail::CharEncodePolicy;
  using ans_gpu_lib::detail::CharX1EncodeImpl;
  using ans_gpu_lib::detail::CharX2EncodeImpl;
  using ans_gpu_lib::detail::FP16EncodePolicy;
  using ans_gpu_lib::detail::FP16X1EncodeImpl;
  using ans_gpu_lib::detail::FP16X2EncodeImpl;
  using ans_gpu_lib::detail::FP32EncodePolicy;
  using ans_gpu_lib::detail::FP32X1EncodeImpl;
  using ans_gpu_lib::detail::FP32X2EncodeImpl;
  using ans_gpu_lib::detail::FP8EncodePolicy;
  using ans_gpu_lib::detail::FP8X1EncodeImpl;
  using ans_gpu_lib::detail::FP8X2EncodeImpl;
  decltype(&ans_gpu_lib::detail::compress_kernel<CharEncodePolicy<CharX1EncodeImpl>, false>) kernel = nullptr;

  const uint32_t histogram_reduction_log2 = format_opts.histogram_reduction_log2;
  const bool sampled = histogram_reduction_log2 != 0;
  switch (format_opts.data_type)
  {
    case NVCOMP_TYPE_FLOAT16:
      if (states_per_lane == 1)
      {
        kernel = sampled ? ans_gpu_lib::detail::compress_kernel<FP16EncodePolicy<FP16X1EncodeImpl>, true>
                         : ans_gpu_lib::detail::compress_kernel<FP16EncodePolicy<FP16X1EncodeImpl>, false>;
      }
      else
      {
        kernel = sampled ? ans_gpu_lib::detail::compress_kernel<FP16EncodePolicy<FP16X2EncodeImpl>, true>
                         : ans_gpu_lib::detail::compress_kernel<FP16EncodePolicy<FP16X2EncodeImpl>, false>;
      }
      break;
    case NVCOMP_TYPE_FLOAT8_E4M3:
      if (states_per_lane == 1)
      {
        kernel = sampled ? ans_gpu_lib::detail::compress_kernel<FP8EncodePolicy<FP8X1EncodeImpl>, true>
                         : ans_gpu_lib::detail::compress_kernel<FP8EncodePolicy<FP8X1EncodeImpl>, false>;
      }
      else
      {
        kernel = sampled ? ans_gpu_lib::detail::compress_kernel<FP8EncodePolicy<FP8X2EncodeImpl>, true>
                         : ans_gpu_lib::detail::compress_kernel<FP8EncodePolicy<FP8X2EncodeImpl>, false>;
      }
      break;
    case NVCOMP_TYPE_FLOAT32:
      if (states_per_lane == 1)
      {
        kernel = sampled ? ans_gpu_lib::detail::compress_kernel<FP32EncodePolicy<FP32X1EncodeImpl>, true>
                         : ans_gpu_lib::detail::compress_kernel<FP32EncodePolicy<FP32X1EncodeImpl>, false>;
      }
      else
      {
        kernel = sampled ? ans_gpu_lib::detail::compress_kernel<FP32EncodePolicy<FP32X2EncodeImpl>, true>
                         : ans_gpu_lib::detail::compress_kernel<FP32EncodePolicy<FP32X2EncodeImpl>, false>;
      }
      break;
    case NVCOMP_TYPE_CHAR:
    case NVCOMP_TYPE_UCHAR:
      if (states_per_lane == 1)
      {
        kernel = sampled ? ans_gpu_lib::detail::compress_kernel<CharEncodePolicy<CharX1EncodeImpl>, true>
                         : ans_gpu_lib::detail::compress_kernel<CharEncodePolicy<CharX1EncodeImpl>, false>;
      }
      else
      {
        kernel = sampled ? ans_gpu_lib::detail::compress_kernel<CharEncodePolicy<CharX2EncodeImpl>, true>
                         : ans_gpu_lib::detail::compress_kernel<CharEncodePolicy<CharX2EncodeImpl>, false>;
      }
      break;
    default:
      throw nvcomp::NVCompException(
        nvcompErrorNotSupported,
        "Unsupported ANS compression data_type: " + std::to_string(static_cast<int>(format_opts.data_type))
      );
  }

  dim3 fused_block(NUM_COMP_WARPS_PER_CTA * WARP_SIZE);
#ifndef NDEBUG
  ans_assert_smem_within_estimate(kernel, [](int arch_id) { return ans_compress_smem_bytes(arch_id); });
#endif
  kernel<<<fused_grid, fused_block, 0, stream>>>(
    comp_chunks,
    (const void *const *)uncomp_chunks,
    uncomp_chunk_sizes,
    max_sub_chunk_size,
    comp_chunk_sizes,
    device_statuses,
    subchunk_comp_buffer_size,
    histogram_reduction_log2,
    pack_C,
    pack_ldc,
    pack_M,
    pack_N,
    pack_tile_m,
    pack_tile_n,
    pack_mn_swapped
  );
  CUDA_CHECK(cudaGetLastError());
}

} // namespace

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
  if (batch_size == 0)
  {
    return;
  }
  launchCompressKernel(
    uncomp_chunks,
    uncomp_chunk_sizes,
    max_chunk_size_bytes,
    batch_size,
    comp_chunks,
    comp_chunk_sizes,
    format_opts,
    device_statuses,
    stream,
    dim3(cuda_dim_cast(batch_size), 1),
    nullptr,
    0,
    0,
    0,
    0,
    0,
    0
  );
}

void compressFromColMajorTilesAsync(
  const float *C,
  int ldc,
  int M,
  int N,
  int tile_m,
  int tile_n,
  int pack_mn_swapped,
  const void *const *uncomp_chunks,
  const size_t *uncomp_chunk_sizes,
  const size_t max_chunk_size_bytes,
  size_t batch_size,
  void *const *comp_chunks,
  size_t *comp_chunk_sizes,
  nvcompBatchedANSCompressOpts_t format_opts,
  nvcompStatus_t *device_statuses,
  cudaStream_t stream
)
{
  if (batch_size == 0)
  {
    return;
  }
  if (C == nullptr || tile_m <= 0 || tile_n <= 0 || M <= 0 || N <= 0 || ldc < M)
  {
    throw nvcomp::NVCompException(
      nvcompErrorInvalidValue,
      "compressFromColMajorTilesAsync requires a column-major C with positive tile sizes"
    );
  }
  const int tiles_m = (M + tile_m - 1) / tile_m;
  const int tiles_n = (N + tile_n - 1) / tile_n;
  if (static_cast<size_t>(tiles_m) * static_cast<size_t>(tiles_n) != batch_size)
  {
    throw nvcomp::NVCompException(
      nvcompErrorInvalidValue,
      "batch_size must equal the number of tile_m x tile_n tiles covering M x N"
    );
  }
  const dim3 fused_grid = pack_mn_swapped
    ? dim3(cuda_dim_cast(static_cast<size_t>(tiles_n)), cuda_dim_cast(static_cast<size_t>(tiles_m)))
    : dim3(cuda_dim_cast(static_cast<size_t>(tiles_m)), cuda_dim_cast(static_cast<size_t>(tiles_n)));
  launchCompressKernel(
    uncomp_chunks,
    uncomp_chunk_sizes,
    max_chunk_size_bytes,
    batch_size,
    comp_chunks,
    comp_chunk_sizes,
    format_opts,
    device_statuses,
    stream,
    fused_grid,
    C,
    ldc,
    M,
    N,
    tile_m,
    tile_n,
    pack_mn_swapped
  );
}

} // namespace ans
