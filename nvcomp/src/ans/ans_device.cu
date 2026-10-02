/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Relocatable device entry for in-kernel ANS LLIF compress. Kept in its own
 * translation unit so the rest of nvCOMP can be compiled without -rdc (gdeflate
 * headers define non-inline __device__ tables that nvlink otherwise duplicates).
 */

#include <algorithm>

#include "nvcomp/ans.h"
#include "nvcomp/ans_device.cuh"
#include "ans/ans_device_chunk.cuh"
#include "ans/ans_utils.cuh"
#include "nvcomp/utils.hpp"

using namespace ans_gpu_lib;

__device__ void nvcompDeviceANSCompressChunk(
  void *compressed,
  const void *uncompressed,
  size_t uncompressed_bytes,
  size_t *compressed_size,
  int max_sub_chunk_size,
  uint32_t slot_words,
  void *smem
)
{
  using Policy = ans_gpu_lib::detail::CharEncodePolicy<ans_gpu_lib::detail::CharX2EncodeImpl>;
  static_assert(
    NVCOMP_DEVICE_ANS_COMPRESS_BLOCK_THREADS == static_cast<int>(ans_gpu_lib::NUM_COMP_THREADS_PER_CTA),
    "device ANS compress CTA width must match the LLIF compressor"
  );
  using Workspace = ans_gpu_lib::detail::DeviceCompressSmem<Policy>;
  auto &workspace = *reinterpret_cast<Workspace *>(smem);
  ans_gpu_lib::detail::compress_chunk<Policy, /*Sampled=*/false>(
    compressed,
    uncompressed,
    static_cast<ans_gpu_lib::IndexT>(uncompressed_bytes),
    compressed_size,
    max_sub_chunk_size,
    nullptr,
    slot_words,
    /*histogram_reduction_log2=*/0u,
    workspace.workspace,
    workspace.packed_chunk_size_bytes
  );
}

extern "C" nvcompStatus_t nvcompBatchedANSCompressGetDeviceLaunchParams(
  size_t /*num_chunks*/,
  size_t max_uncompressed_chunk_bytes,
  nvcompBatchedANSCompressOpts_t compress_opts,
  int *max_sub_chunk_size,
  uint32_t *slot_words,
  size_t *smem_bytes,
  size_t *smem_alignment,
  int *block_threads
)
{
  if (max_sub_chunk_size == nullptr || slot_words == nullptr || smem_bytes == nullptr ||
      smem_alignment == nullptr || block_threads == nullptr)
  {
    return nvcompErrorInvalidValue;
  }
  if (max_uncompressed_chunk_bytes > nvcompANSCompressionMaxAllowedChunkSize)
  {
    return nvcompErrorInvalidValue;
  }

  const uint32_t bytes_per_symbol =
    ans_bytes_per_symbol(ans_stream_type_from_data_type(compress_opts.data_type));
  const uint32_t max_chunk_size_symbols =
    nvcomp::roundUpDiv(static_cast<uint32_t>(max_uncompressed_chunk_bytes), bytes_per_symbol);
  const uint32_t requested = resolve_max_sub_chunk_count(compress_opts.max_sub_chunk_count);
  const uint32_t unrounded = nvcomp::roundUpDiv(max_chunk_size_symbols, requested);
  const uint32_t max_sub = std::max(MIN_SUB_CHUNK_SIZE, nvcomp::roundUpPow2(unrounded));

  uint32_t states_per_lane = compress_opts.states_per_lane;
  if (states_per_lane != 1 && states_per_lane != 2)
  {
    states_per_lane = ans_default_states_per_lane(ans_stream_type_from_data_type(compress_opts.data_type));
  }
  const uint32_t subchunk_comp_buffer_size = get_max_comp_sub_chunk_size(
    max_sub,
    states_per_lane,
    ans_tablelog(ans_stream_type_from_data_type(compress_opts.data_type))
  );

  using DeviceWorkspace =
    ans_gpu_lib::detail::DeviceCompressSmem<ans_gpu_lib::detail::CharEncodePolicy<ans_gpu_lib::detail::CharX2EncodeImpl>>;

  *max_sub_chunk_size = static_cast<int>(max_sub);
  *slot_words = subchunk_comp_buffer_size;
  *smem_bytes = sizeof(DeviceWorkspace);
  *smem_alignment = alignof(DeviceWorkspace);
  *block_threads = static_cast<int>(ans_gpu_lib::NUM_COMP_THREADS_PER_CTA);
  return nvcompSuccess;
}
