/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Device-side one-chunk compress used by nvcompDeviceANSCompressChunk.
 * Kept out of stock compress_kernels_llif.cuh so nvCOMP source drops do not
 * wipe the CUTLASS in-kernel entry.
 */

#pragma once

#include <limits>

#include "ans/compress_kernels_llif.cuh"
#include "nvcomp/ans.h"

namespace ans_gpu_lib
{
namespace detail
{

// Workspace for nvcompDeviceANSCompressChunk: CompressSmem plus the uint32
// packed size that compress_kernel otherwise keeps as a separate __shared__.
template <typename EncodePolicy>
struct DeviceCompressSmem
{
  CompressSmem<EncodePolicy> workspace;
  uint32_t packed_chunk_size_bytes;
};

// One-chunk compress used by nvcompDeviceANSCompressChunk.
// Must be called by every thread of a NUM_COMP_THREADS_PER_CTA CTA.
template <typename EncodePolicy, bool Sampled>
__device__ __forceinline__ void compress_chunk(
  void *comp_chunk,
  const void *uncomp_chunk,
  IndexT bytes,
  size_t *comp_chunk_size_out,
  int max_sub_chunk_size,
  nvcompStatus_t *device_status,
  uint32_t subchunk_comp_buffer_size,
  uint32_t histogram_reduction_log2,
  CompressSmem<EncodePolicy> &smem,
  uint32_t &packed_chunk_size_bytes
)
{
  static_assert(
    nvcompANSCompressionMaxAllowedChunkSize <= std::numeric_limits<IndexT>::max(),
    "ANS chunk size exceeds IndexT range"
  );
  constexpr IndexT INPUT_BYTES_PER_VALUE = EncodePolicy::STREAM_TYPE == AnsStreamType::Fp16   ? sizeof(uint16_t)
                                           : EncodePolicy::STREAM_TYPE == AnsStreamType::Fp32 ? sizeof(uint32_t)
                                                                                              : sizeof(uint8_t);
  if (bytes % INPUT_BYTES_PER_VALUE != 0)
  {
    if (threadIdx.x == 0)
    {
      *comp_chunk_size_out = 0;
      if (device_status != nullptr)
      {
        *device_status = nvcompErrorCannotCompress;
      }
    }
    return;
  }

  const IndexT symbols = EncodePolicy::num_symbols(bytes);

  if (symbols == 0)
  {
    if (threadIdx.x == 0)
    {
      EncodePolicy::write_chunk_tail(
        static_cast<uint8_t *>(comp_chunk),
        uncomp_chunk,
        bytes,
        /*num_sub_chunks=*/1
      );
    }
    __syncthreads();
    finish_zero_ans_symbol_chunk<EncodePolicy>(
      comp_chunk,
      bytes,
      max_sub_chunk_size,
      packed_chunk_size_bytes,
      subchunk_comp_buffer_size
    );
    if (threadIdx.x == 0)
    {
      *comp_chunk_size_out = packed_chunk_size_bytes;
    }
    return;
  }

  const uint32_t max_sub_chunk_size_bytes = static_cast<uint32_t>(max_sub_chunk_size) *
                                            EncodePolicy::INPUT_BYTES_PER_SYMBOL;
  const int num_sub_chunks_per_chunk = static_cast<int>(ans_derived_num_sub_chunks(bytes, max_sub_chunk_size_bytes));

  // Defrag copies whole uint4s from each sub-chunk slot, including tail bytes the
  // encoder never wrote. Zero the chunk so those padding bytes are deterministic
  // (otherwise device vs host LLIF bitstreams differ while still decompressing).
  {
    const uint32_t nsc = static_cast<uint32_t>(num_sub_chunks_per_chunk);
    const uint32_t header_bytes = ans_sub_chunk_0_offset(
      EncodePolicy::STREAM_TYPE,
      bytes,
      nsc,
      /*min_symbol=*/0,
      static_cast<uint8_t>(NV_MAX_SYMBOL_VALUE)
    );
    const uint32_t total_bytes = header_bytes + nsc * subchunk_comp_buffer_size;
    uint32_t *words = static_cast<uint32_t *>(comp_chunk);
    const uint32_t nwords = total_bytes / sizeof(uint32_t);
    for (uint32_t i = threadIdx.x; i < nwords; i += static_cast<uint32_t>(blockDim.x))
    {
      words[i] = 0;
    }
    __syncthreads();
  }

  constexpr bool SAMPLED_HIST = Sampled && EncodePolicy::HIST_FLOOR != HistFloor::None;
  constexpr bool DETECT = SAMPLED_HIST && EncodePolicy::HIST_FLOOR == HistFloor::ObservedBand;
  const bool can_sample = Sampled && symbols >= MIN_SAMPLED_HIST_SYMBOLS;

  if (threadIdx.x == 0)
  {
    smem.enc.uncomp_chunk = uncomp_chunk;
    smem.enc.comp_chunk = comp_chunk;
    smem.enc.bytes = bytes;
    smem.enc.symbols = symbols;
    smem.enc.num_sub_chunks = static_cast<uint32_t>(num_sub_chunks_per_chunk);
    smem.enc.sample_shift = can_sample ? histogram_reduction_log2 : 0u;
    smem.enc.max_sub_chunk_size = max_sub_chunk_size;
  }
  __syncthreads();

  if constexpr (DETECT)
  {
    if (prepare_and_encode_chunk<EncodePolicy, true, /*SampledHist=*/true>(smem))
    {
      prepare_and_encode_chunk<EncodePolicy, /*Detect=*/false, /*SampledHist=*/false>(smem);
    }
  }
  else
  {
    prepare_and_encode_chunk<EncodePolicy, false, SAMPLED_HIST>(smem);
  }

  ANS_fixed_header *const defrag_header = static_cast<ANS_fixed_header *>(smem.enc.comp_chunk);

  simple_defrag_chunk_cta<NUM_COMP_THREADS_PER_CTA>(defrag_header, subchunk_comp_buffer_size, packed_chunk_size_bytes);
  if (threadIdx.x == 0)
  {
    *comp_chunk_size_out = packed_chunk_size_bytes;
  }
}

} // namespace detail
} // namespace ans_gpu_lib
