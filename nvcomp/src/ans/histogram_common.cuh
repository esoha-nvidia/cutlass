/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
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

#pragma once

#include <cuda_pipeline.h>

#include "ans/ans_utils.cuh"
#include "ans/types.cuh" // Fp16EncodePolicy::split_float (fp16 DX histogram path)
#include "common_utils.hpp" // nvcomp::bytesUntilAlignmentBoundary

namespace ans_gpu_lib::detail
{

template <typename EncodePolicy>
inline __device__ uint8_t *
histogram_sideband_ptr([[maybe_unused]] uint8_t *sideband, [[maybe_unused]] IndexT symbol_offset)
{
  if constexpr (EncodePolicy::SIDEBAND_BYTES_PER_SYMBOL == 0)
  {
    return nullptr;
  }
  else
  {
    return sideband + symbol_offset * EncodePolicy::SIDEBAND_BYTES_PER_SYMBOL;
  }
}

// Generic per-warp histogram engine shared by char / fp16 / fp8. The load width,
// symbols-per-word, software-pipeline depth, optional alignment peel, per-register
// split + side-band store, and scalar tail are all supplied by EncodePolicy (see
// ans/types.cuh).
template <typename EncodePolicy, bool FIX_ALIGNMENT>
inline __device__ void
histogram_warp_impl(int idx_in_warp, uint32_t *counts, const uint8_t *ip, int size, uint8_t *sideband)
{
  using LoadT = typename EncodePolicy::LoadT;
  // Symbols (or FP8 pairs) per load word, derived from the load width and the input
  // bytes per symbol: char 8/1=8, fp16/fp8 8/2=4.
  constexpr int SYMBOLS_PER_WORD = sizeof(LoadT) / EncodePolicy::INPUT_BYTES_PER_SYMBOL;
  // LOADS_PER_ITER is per lane; WARP_WORDS_PER_ITER is the warp-wide total.
  constexpr int LOADS_PER_ITER = EncodePolicy::LOADS_PER_ITER;
  constexpr int WARP_WORDS_PER_ITER = LOADS_PER_ITER * WARP_SIZE;

  // Optional alignment peel (char: byte-granular to the uint2 boundary; fp16/fp8:
  // none, their input is already >= 8-byte aligned). Discarded at compile time otherwise.
  if constexpr (FIX_ALIGNMENT && EncodePolicy::HAS_PEEL)
  {
    const int peeled = EncodePolicy::histogram_peel_head_warp(idx_in_warp, counts, ip, size, sideband);
    if (size <= peeled)
    {
      return;
    }
    ip += peeled * EncodePolicy::INPUT_BYTES_PER_SYMBOL;
    size -= peeled;
    if constexpr (EncodePolicy::SIDEBAND_BYTES_PER_SYMBOL != 0)
    {
      sideband += static_cast<IndexT>(peeled) * EncodePolicy::SIDEBAND_BYTES_PER_SYMBOL;
    }
  }

  const LoadT *ip_word = reinterpret_cast<const LoadT *>(ip);

  // Complete words this warp owns; the final < SYMBOLS_PER_WORD leftover go to the tail.
  const int num_words = size / SYMBOLS_PER_WORD;

  // Main loop: LOADS_PER_ITER words per lane per iteration, software-pipelined --
  // issue all the coalesced loads before consuming any.
  const int nloop_iter = num_words / WARP_WORDS_PER_ITER;
  for (int i = 0; i < nloop_iter; i++)
  {
    const int word_iter_base = i * WARP_WORDS_PER_ITER;

    LoadT reg[LOADS_PER_ITER];
#pragma unroll
    for (int s = 0; s < LOADS_PER_ITER; ++s)
    {
      reg[s] = ip_word[word_iter_base + s * WARP_SIZE + idx_in_warp];
    }
#pragma unroll
    for (int s = 0; s < LOADS_PER_ITER; ++s)
    {
      const int w = word_iter_base + s * WARP_SIZE + idx_in_warp;
      EncodePolicy::template histogram_process_word<FIX_ALIGNMENT>(
        reg[s],
        counts,
        histogram_sideband_ptr<EncodePolicy>(sideband, static_cast<IndexT>(w) * SYMBOLS_PER_WORD)
      );
    }
  }

  // Remainder words: full warp-rows (one word/lane, coalesced), then the final
  // < WARP_SIZE words one per lane.
  const int rem_word_base = nloop_iter * WARP_WORDS_PER_ITER;
  const int rem_words = num_words - rem_word_base;
  const int rem_full_rows = rem_words / WARP_SIZE;
  for (int r = 0; r < rem_full_rows; ++r)
  {
    const int w = rem_word_base + r * WARP_SIZE + idx_in_warp;
    LoadT reg = ip_word[w];
    EncodePolicy::template histogram_process_word<FIX_ALIGNMENT>(
      reg,
      counts,
      histogram_sideband_ptr<EncodePolicy>(sideband, static_cast<IndexT>(w) * SYMBOLS_PER_WORD)
    );
  }
  const int rem_row_words = rem_words - rem_full_rows * WARP_SIZE;
  if (idx_in_warp < rem_row_words)
  {
    const int w = rem_word_base + rem_full_rows * WARP_SIZE + idx_in_warp;
    LoadT reg = ip_word[w];
    EncodePolicy::template histogram_process_word<FIX_ALIGNMENT>(
      reg,
      counts,
      histogram_sideband_ptr<EncodePolicy>(sideband, static_cast<IndexT>(w) * SYMBOLS_PER_WORD)
    );
  }

  // Scalar tail: the final < SYMBOLS_PER_WORD symbols (pairs for fp8) that do not
  // form a complete word. One lane per symbol (tail < WARP_SIZE, so no loop).
  const int tail_base = num_words * SYMBOLS_PER_WORD;
  const int tail_size = size - tail_base;
  const uint8_t *tail_in = ip + static_cast<IndexT>(tail_base) * EncodePolicy::INPUT_BYTES_PER_SYMBOL;
  uint8_t *tail_sideband = histogram_sideband_ptr<EncodePolicy>(sideband, static_cast<IndexT>(tail_base));
  EncodePolicy::histogram_process_scalar_tail_warp(idx_in_warp, counts, tail_in, tail_sideband, tail_size);
}

// ===========================================================================
// nvcompDX duplicates.

template <typename CG>
inline __device__ void reduce_max_symbol_in_block_dx(
  uint32_t idx_in_warp,
  uint32_t thread_max_symbol_value,
  uint32_t *shared_reduction_buffer,
  uint32_t &block_max_symbol_value,
  CG &group
)
{
  uint32_t warp_max_symbol_value = reduce_max(thread_max_symbol_value);

  if (group.size() != WARP_SIZE)
  {
    *shared_reduction_buffer = 0;
    group.sync();
    if (idx_in_warp == 0)
    {
      atomicMax(shared_reduction_buffer, warp_max_symbol_value);
    }
    group.sync();
    block_max_symbol_value = *shared_reduction_buffer;
  }
  else
  {
    block_max_symbol_value = warp_max_symbol_value;
  }
}

template <bool FIX_ALIGNMENT>
inline __device__ void histogram_uint8_data_dx(
  uint32_t idx_in_warp,
  uint32_t *shared_cta_counts,
  const uint8_t *ip,
  int histogram_size_per_warp,
  uint32_t &thread_max_symbol_value
)
{

  constexpr uint32_t NUM_HIST_SYMBOLS_PER_THREAD = 16;

  if constexpr (FIX_ALIGNMENT)
  {
    // Handle the first few bytes so that we can read 16-byte (uint4) aligned words
    const int alignment_rem = static_cast<int>(nvcomp::bytesUntilAlignmentBoundary(ip, sizeof(uint4)));
    if (idx_in_warp < min(alignment_rem, histogram_size_per_warp))
    {
      uint8_t symbol = ip[idx_in_warp];
      atomicAdd(&shared_cta_counts[symbol], 1);
      thread_max_symbol_value = max(thread_max_symbol_value, symbol);
    }

    if (histogram_size_per_warp <= alignment_rem)
    {
      return;
    }

    ip += alignment_rem;
    histogram_size_per_warp -= alignment_rem;
  }

  int nloop = histogram_size_per_warp / (NUM_HIST_SYMBOLS_PER_THREAD * WARP_SIZE_U);

  // We use a union in this manner to avoid register-swizzling
  // instructions such as PRMT that are needed when directly loading uchar4
  union reg_val
  {
    uint4 reg;
    uchar4 bytes[4];
  };
  const uint4 *ip_u4 = reinterpret_cast<const uint4 *>(ip);

  reg_val reg_next;
  if (nloop > 0)
  {
    reg_next.reg = ip_u4[idx_in_warp];
    ip_u4 += WARP_SIZE_U;
  }

  for (int i = 0; i < nloop - 1; ++i)
  {
    reg_val reg;
    reg.reg = reg_next.reg;

    // prefetch the next 16 bytes to hide some of the latency of the gmem read with shmem atomics
    reg_next.reg = ip_u4[idx_in_warp];
    ip_u4 += WARP_SIZE_U;

    for (int ix = 0; ix < 4; ++ix)
    {
      atomicAdd(&shared_cta_counts[reg.bytes[ix].x], 1);
      atomicAdd(&shared_cta_counts[reg.bytes[ix].y], 1);
      atomicAdd(&shared_cta_counts[reg.bytes[ix].z], 1);
      atomicAdd(&shared_cta_counts[reg.bytes[ix].w], 1);
      thread_max_symbol_value = max(thread_max_symbol_value, reg.bytes[ix].x);
      thread_max_symbol_value = max(thread_max_symbol_value, reg.bytes[ix].y);
      thread_max_symbol_value = max(thread_max_symbol_value, reg.bytes[ix].z);
      thread_max_symbol_value = max(thread_max_symbol_value, reg.bytes[ix].w);
    }
  }

  if (nloop > 0)
  {
    reg_val reg;
    reg.reg = reg_next.reg;

    for (int ix = 0; ix < 4; ++ix)
    {
      atomicAdd(&shared_cta_counts[reg.bytes[ix].x], 1);
      atomicAdd(&shared_cta_counts[reg.bytes[ix].y], 1);
      atomicAdd(&shared_cta_counts[reg.bytes[ix].z], 1);
      atomicAdd(&shared_cta_counts[reg.bytes[ix].w], 1);
      thread_max_symbol_value = max(thread_max_symbol_value, reg.bytes[ix].x);
      thread_max_symbol_value = max(thread_max_symbol_value, reg.bytes[ix].y);
      thread_max_symbol_value = max(thread_max_symbol_value, reg.bytes[ix].z);
      thread_max_symbol_value = max(thread_max_symbol_value, reg.bytes[ix].w);
    }
  }

  // take care of last < 4*block_size bytes
  const uint8_t *remaining_bytes = reinterpret_cast<const uint8_t *>(ip_u4);
  int rem = histogram_size_per_warp % (NUM_HIST_SYMBOLS_PER_THREAD * WARP_SIZE_U);

#pragma unroll
  for (uint32_t i = 0; i < NUM_HIST_SYMBOLS_PER_THREAD * WARP_SIZE_U; i += WARP_SIZE_U)
  {
    bool active = idx_in_warp + i < rem;
    if (!active)
    {
      break;
    }
    uint8_t symbol = remaining_bytes[idx_in_warp + i];
    atomicAdd(&shared_cta_counts[symbol], 1);
    thread_max_symbol_value = max(thread_max_symbol_value, symbol);
  }
}

template <bool FIX_ALIGNMENT>
inline __device__ void histogram_uint16_data_dx(
  uint32_t idx_in_warp,
  uint32_t *shared_cta_counts,
  uint8_t *shared_staging_buf,
  const uint8_t *ip,
  int histogram_size_per_warp,
  uint8_t *exponents,
  uint8_t *comp_chunk,
  uint32_t &thread_max_symbol_value
)
{
  constexpr uint32_t NUM_HIST_SYMBOLS_PER_THREAD = 4;

  if constexpr (FIX_ALIGNMENT)
  {
    // Handle the first few bytes so that we can read 8-byte (uint2) aligned words
    const int alignment_rem = static_cast<int>(nvcomp::bytesUntilAlignmentBoundary(ip, sizeof(uint2)));
    const int alignment_rem_symbols = alignment_rem / static_cast<int>(sizeof(uint16_t));
    if (idx_in_warp < min(alignment_rem_symbols, histogram_size_per_warp))
    {
      const uint16_t fp16 = reinterpret_cast<const uint16_t *>(ip)[idx_in_warp];
      uint8_t exponent, mantissa;
      Fp16EncodePolicy::split_float(fp16, exponent, mantissa);
      exponents[idx_in_warp] = exponent;
      comp_chunk[idx_in_warp] = mantissa;
      atomicAdd(&shared_cta_counts[exponent], 1);
      thread_max_symbol_value = max(thread_max_symbol_value, exponent);
    }

    if (histogram_size_per_warp <= alignment_rem_symbols)
    {
      return;
    }

    ip += alignment_rem;
    histogram_size_per_warp -= alignment_rem_symbols;
    exponents += alignment_rem_symbols;
    comp_chunk += alignment_rem_symbols;
  }

  uint2 *shared_staging_buf_uint2 = reinterpret_cast<uint2 *>(shared_staging_buf);
  const uint2 *ip_uint2 = reinterpret_cast<const uint2 *>(ip);

  int nloop = histogram_size_per_warp / (NUM_HIST_SYMBOLS_PER_THREAD * WARP_SIZE_U);

#pragma unroll
  for (int i = 0; i < nloop; ++i)
  {
    shared_staging_buf_uint2[idx_in_warp] = ip_uint2[idx_in_warp];
    __syncwarp();
    ip_uint2 += WARP_SIZE_U;

    for (int j = 0; j < NUM_HIST_SYMBOLS_PER_THREAD; ++j)
    {
      uint16_t float_symbol = reinterpret_cast<uint16_t *>(shared_staging_buf)[WARP_SIZE * j + idx_in_warp];

      uint8_t exponent, mantissa;
      Fp16EncodePolicy::split_float(float_symbol, exponent, mantissa);

      // exponent is used for histogramming purposes and also written to a
      // tmp buffer that is used by the compressor
      exponents[idx_in_warp] = exponent;
      exponents += WARP_SIZE_U;

      // Mantissa is written in its raw state to the initial part of the output buffer
      comp_chunk[idx_in_warp] = mantissa;
      comp_chunk += WARP_SIZE_U;

      // We're only compressing the exponents, so that's what we're histogramming
      atomicAdd(&shared_cta_counts[exponent], 1);
      thread_max_symbol_value = max(thread_max_symbol_value, exponent);
    }
    __syncwarp();
  }

  // take care of last < 4*block_size bytes
  const uint16_t *remaining_bytes = reinterpret_cast<const uint16_t *>(ip_uint2);
  uint32_t rem = histogram_size_per_warp % (NUM_HIST_SYMBOLS_PER_THREAD * WARP_SIZE_U);

#pragma unroll
  for (uint32_t i = 0; i < NUM_HIST_SYMBOLS_PER_THREAD * WARP_SIZE_U; i += WARP_SIZE_U)
  {
    bool active = idx_in_warp + i < rem;
    if (!active)
    {
      break;
    }
    uint16_t float_symbol = remaining_bytes[idx_in_warp + i];

    const uint8_t exponent = static_cast<uint8_t>(float_symbol >> 8);
    const uint8_t mantissa = static_cast<uint8_t>(float_symbol);

    // exponent is used for histogramming purposes and also written to a
    // tmp buffer that is used by the compressor
    exponents[idx_in_warp] = exponent;
    exponents += WARP_SIZE_U;

    // Mantissa is written in its raw state to the initial part of the output buffer
    comp_chunk[idx_in_warp] = mantissa;
    comp_chunk += WARP_SIZE_U;

    atomicAdd(&shared_cta_counts[exponent], 1);

    thread_max_symbol_value = max(thread_max_symbol_value, exponent);
  }
}

} // namespace ans_gpu_lib::detail
