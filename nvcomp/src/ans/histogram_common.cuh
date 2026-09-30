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
#include "ans/EncodePolicy.hpp"

namespace ans_gpu_lib::detail
{

template <typename EncodePolicy>
inline __device__ uint8_t *
histogram_mantissas_ptr([[maybe_unused]] uint8_t *mantissas, [[maybe_unused]] IndexT symbol_offset)
{
  if constexpr (EncodePolicy::MANTISSA_BYTES_PER_SYMBOL == 0 || EncodePolicy::ENCODE_WRITES_MANTISSAS)
  {
    return nullptr;
  }
  else
  {
    return mantissas + symbol_offset * EncodePolicy::MANTISSA_BYTES_PER_SYMBOL;
  }
}

// Generic per-warp histogram engine shared by char / fp16 / fp8 / fp32. The load width,
// prefetch depth, optional alignment peel, per-register split + mantissa store, and
// scalar tail are all supplied by EncodePolicy (see ans/EncodePolicy.hpp).
//
// Loads run through a DEPTH-deep cp.async pipeline into per-warp shared memory
// (`warp_stage`) when one is provided: the same latency hiding as a register window, but
// without the LoadT registers, which spill under the fused kernel's launch bounds. Callers
// with no stage buffer (the unit tests) pass nullptr and get a plain LDG loop.
template <typename EncodePolicy>
inline __device__ void histogram_warp_impl(
  int idx_in_warp,
  uint32_t *counts,
  const uint8_t *ip,
  int size,
  uint8_t *mantissas,
  uint8_t *warp_stage = nullptr // [HIST_PREFETCH_DEPTH * WARP_SIZE * sizeof(LoadT)]
)
{
  using LoadT = typename EncodePolicy::LoadT;
  // Symbols (or FP8 pairs) per load word, derived from the load width and the input
  // bytes per symbol: char 16/1=16, fp8/fp16 16/2=8, fp32 16/4=4.
  constexpr int SYMBOLS_PER_WORD = sizeof(LoadT) / EncodePolicy::INPUT_BYTES_PER_SYMBOL;
  constexpr int PREFETCH_DEPTH = EncodePolicy::HIST_PREFETCH_DEPTH;

  const LoadT *ip_word = reinterpret_cast<const LoadT *>(ip);

  // Complete words this warp owns; the final < SYMBOLS_PER_WORD leftover go to the tail.
  const int num_words = size / SYMBOLS_PER_WORD;

  auto process_word = [&](LoadT reg, int w) {
    EncodePolicy::histogram_process_word(
      reg,
      counts,
      histogram_mantissas_ptr<EncodePolicy>(mantissas, static_cast<IndexT>(w) * SYMBOLS_PER_WORD)
    );
  };

  // Main loop: one coalesced word per lane per warp-row, PREFETCH_DEPTH rows in flight.
  constexpr int stride = WARP_SIZE;
  const int num_rows = num_words / stride;
  int row = 0;
  if (warp_stage != nullptr && num_rows > 0)
  {
    LoadT *stage = reinterpret_cast<LoadT *>(warp_stage); // [DEPTH][WARP], row-major

    // Prime up to DEPTH rows, one commit each so a wait can drain them individually. A slice
    // shorter than DEPTH rows -- what a sampled histogram usually hands us -- primes only
    // what it owns and drains below, rather than dropping to the serial LDG path and putting
    // the contended atomics on the critical path with nothing to overlap them.
    const int primed = min(num_rows, PREFETCH_DEPTH);
#pragma unroll
    for (int g = 0; g < PREFETCH_DEPTH; ++g)
    {
      if (g < primed)
      {
        __pipeline_memcpy_async(&stage[g * stride + idx_in_warp], &ip_word[g * stride + idx_in_warp], sizeof(LoadT));
        __pipeline_commit();
      }
    }

    // Steady state: wait for the oldest commit only, consume that slot, refill it. Waiting
    // for one commit (rather than all) is what keeps DEPTH-1 rows in flight across the
    // atomics -- draining the whole pipeline here would serialize fetch behind process.
    for (; row + 2 * PREFETCH_DEPTH <= num_rows; row += PREFETCH_DEPTH)
    {
#pragma unroll
      for (int g = 0; g < PREFETCH_DEPTH; ++g)
      {
        __pipeline_wait_prior(PREFETCH_DEPTH - 1);
        __syncwarp();
        const int cur_row = row + g;
        process_word(stage[g * stride + idx_in_warp], cur_row * stride + idx_in_warp);
        __syncwarp(); // WAR: every lane must read slot g before it is refilled
        __pipeline_memcpy_async(
          &stage[g * stride + idx_in_warp],
          &ip_word[(cur_row + PREFETCH_DEPTH) * stride + idx_in_warp],
          sizeof(LoadT)
        );
        __pipeline_commit();
      }
    }

    // Drain the rows still staged (up to DEPTH), then fall through to LDG for any leftover.
    __pipeline_wait_prior(0);
    __syncwarp();
#pragma unroll
    for (int g = 0; g < PREFETCH_DEPTH; ++g)
    {
      const int cur_row = row + g;
      if (cur_row < num_rows)
      {
        process_word(stage[g * stride + idx_in_warp], cur_row * stride + idx_in_warp);
      }
    }
    row += PREFETCH_DEPTH;
  }

  // No stage buffer (callers that pass none), or the rows the drain above left over.
  for (; row < num_rows; ++row)
  {
    const int w = row * stride + idx_in_warp;
    process_word(ip_word[w], w);
  }

  // Final partial warp-row (< WARP_SIZE words, one per lane).
  const int rem_word_base = num_rows * stride;
  const int rem_words = num_words - rem_word_base;
  if (idx_in_warp < rem_words)
  {
    const int w = rem_word_base + idx_in_warp;
    process_word(ip_word[w], w);
  }

  // Scalar tail: the final < SYMBOLS_PER_WORD symbols (pairs for fp8) that do not
  // form a complete word. One lane per symbol (tail < WARP_SIZE, so no loop).
  const int tail_base = num_words * SYMBOLS_PER_WORD;
  const int tail_size = size - tail_base;
  const uint8_t *tail_in = ip + static_cast<IndexT>(tail_base) * EncodePolicy::INPUT_BYTES_PER_SYMBOL;
  uint8_t *tail_mantissas = histogram_mantissas_ptr<EncodePolicy>(mantissas, static_cast<IndexT>(tail_base));
  EncodePolicy::histogram_process_scalar_tail_warp(idx_in_warp, counts, tail_in, tail_mantissas, tail_size);
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

inline __device__ void histogram_uint8_data_dx(
  uint32_t idx_in_warp,
  uint32_t *shared_cta_counts,
  const uint8_t *ip,
  int histogram_size_per_warp,
  uint32_t &thread_max_symbol_value
)
{

  constexpr uint32_t NUM_HIST_SYMBOLS_PER_THREAD = 16;

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

// nvcompDx fp16 histogram. Still uses FP16EncodePolicy::split_float (high byte = symbol).
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
      FP16EncodePolicy<FP16X2EncodeImpl>::split_float(float_symbol, exponent, mantissa);

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
    uint8_t exponent, mantissa;
    FP16EncodePolicy<FP16X2EncodeImpl>::split_float(float_symbol, exponent, mantissa);

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
