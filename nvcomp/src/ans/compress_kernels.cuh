/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include <ciso646>

#include <ans/ans_utils.cuh>
#include <ans/EncodePolicy.hpp>
#include <ans/symbol_encoder.cuh>
#include <stdio.h>

namespace ans_gpu_lib
{
namespace detail
{

// =====================================================================
// Shared encode scaffold (used by all modes)
// =====================================================================

template <typename EncodePolicy>
inline __device__ void begin_sub_chunk_encode(
  int sub_chunk_idx,
  IndexT uncomp_chunk_size, // symbols
  int max_sub_chunk_size,
  EncodeChunkSmem &enc
)
{
  const int wid = static_cast<int>(threadIdx.x / WARP_SIZE_U);
  assert(reinterpret_cast<uintptr_t>(enc.comp_chunk) % ANS_SUB_CHUNK_SLOT_ALIGN == 0);
  ANS_fixed_header *const header = static_cast<ANS_fixed_header *>(enc.comp_chunk);
  const IndexT in_start_idx = static_cast<IndexT>(sub_chunk_idx) * max_sub_chunk_size;
  const int sub_chunk_size =
    static_cast<int>(min(uncomp_chunk_size - in_start_idx, static_cast<IndexT>(max_sub_chunk_size)));

  const IndexT max_comp_sub_chunk_size = get_max_comp_sub_chunk_size(
    static_cast<uint32_t>(max_sub_chunk_size),
    EncodePolicy::STATES_PER_LANE,
    EncodePolicy::TABLELOG
  );
  uint8_t *const out = get_comp_sub_chunk_output_ptr(
    static_cast<uint32_t>(sub_chunk_idx),
    header->get_sub_chunk_0_start(),
    max_comp_sub_chunk_size
  );

  // A warp reuses warp_enc[wid] across the sub-chunks it owns, so the previous
  // sub-chunk's readers (encode_body, finish_sub_chunk_encode) must be done before it is
  // overwritten.
  __syncwarp();
  volatile WarpEncodeOff &w = enc.warp_enc[wid];
  w.subchunk_output = byte_off(enc.comp_chunk, out);
  w.mantissas_offset = enc.mantissas_chunk_offset == 0u
                         ? 0u
                         : enc.mantissas_chunk_offset + in_start_idx * EncodePolicy::MANTISSA_BYTES_PER_SYMBOL;
  w.in_start_idx = in_start_idx;
  w.sub_chunk_size = sub_chunk_size;
  __syncwarp();

  // Requires uncomp_chunk_size > 0.
  assert(uncomp_chunk_size > 0);
  assert(in_start_idx < uncomp_chunk_size);
}

// Flush the rANS state and record this sub-chunk's compressed size in the header. Paired with
// begin_sub_chunk_encode; used at every encoder's finalize site(s).
template <typename Encoder>
inline __device__ void finish_sub_chunk_encode(EncodeChunkSmem &enc, int sub_chunk_idx, Encoder &se)
{
  se.encode_final_state();
  if (get_lane_id() == 0)
  {
    const uint32_t out_off = enc.warp_enc[threadIdx.x / WARP_SIZE_U].subchunk_output;
    static_cast<ANS_fixed_header *>(enc.comp_chunk)->get_sub_chunk_sizes()[sub_chunk_idx] =
      se.get_comp_stream_size(enc.comp_at(out_off));
  }
}

// Encode one sub-chunk. Returns true when Detect is on and some lane in this warp hit a
// symbol the (sampled) model does not cover, which tells the caller to redo the chunk
// from an exact histogram. Always false when Detect is off.
template <typename EncodePolicy, bool Detect = false>
__device__ __forceinline__ bool encode_sub_chunk(
  int sub_chunk_idx,
  IndexT uncomp_chunk_size, // symbols
  int max_sub_chunk_size,
  const uint2 *shared_table,
  EncodeChunkSmem &enc
)
{
  // Byte-derived nsc can be one past the last symbol (FLOAT8 leftover byte). Write a
  // zero size and skip the encoder; begin_sub_chunk_encode requires a nonempty range.
  if (static_cast<IndexT>(sub_chunk_idx) * static_cast<IndexT>(max_sub_chunk_size) >= uncomp_chunk_size)
  {
    if (get_lane_id() == 0)
    {
      static_cast<ANS_fixed_header *>(enc.comp_chunk)->get_sub_chunk_sizes()[sub_chunk_idx] = 0;
    }
    return false;
  }

  begin_sub_chunk_encode<EncodePolicy>(sub_chunk_idx, uncomp_chunk_size, max_sub_chunk_size, enc);
  const uint32_t out_off = enc.warp_enc[threadIdx.x / WARP_SIZE_U].subchunk_output;
  typename EncodePolicy::Encoder se(get_lane_id(), shared_table, enc.comp_at(out_off));
  EncodePolicy::template encode_body<Detect>(se, enc);
  finish_sub_chunk_encode(enc, sub_chunk_idx, se);
  if constexpr (Detect)
  {
    return se.uncovered_any();
  }
  else
  {
    return false;
  }
}

} // end namespace detail
} // end namespace ans_gpu_lib
