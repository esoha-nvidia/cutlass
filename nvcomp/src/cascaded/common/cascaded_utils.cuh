/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

#include <cuda_runtime.h>

#include <cassert>
#include <cstdint>
#include <type_traits>

#include "cascaded_warp_reductions.cuh"
#include "CudaConstants.h"
#include "nvcomp.h"

namespace nvcomp::cascaded
{

inline __device__ int thread_warp_ix() { return threadIdx.x % WARP_SIZE; }

constexpr uint32_t log2_pow2(uint32_t x) { return (x <= 1) ? 0 : 1 + log2_pow2(x >> 1); }

// Vectorized block copy assuming `src` and `dst` share the same alignment offset
template <int num_bytes_aligned, typename uintN, uint32_t num_threads>
__device__ void block_copy_aligned(const uint8_t *__restrict__ src, uint8_t *__restrict__ dst, int num_bytes)
{
  static_assert(
    num_threads >= WARP_SIZE
  ); // Current impl requires >= 16, but this function will require at least 32 thds in the future
  static_assert(num_threads % WARP_SIZE == 0); // Future proofing

  assert(blockDim.x == num_threads && blockDim.y == 1 && blockDim.z == 1);

  const uint32_t tid = threadIdx.x;

  // --- Head: bytes before the next num_bytes_aligned boundary ---
  constexpr int alignment = num_bytes_aligned - 1;
  int head = static_cast<int>((num_bytes_aligned - (reinterpret_cast<uintptr_t>(src) & alignment)) & alignment);
  if (head > num_bytes)
  {
    head = num_bytes;
  }

  if (tid < static_cast<uint32_t>(head))
  {
    dst[tid] = src[tid];
  }

  int remaining = num_bytes - head;
  const uint8_t *src2 = src + head;
  uint8_t *dst2 = dst + head;

  // --- Middle: vectorized num_bytes_aligned bytes copies ---
  int num_vec = remaining / num_bytes_aligned;
  const uintN *src_v = reinterpret_cast<const uintN *>(src2);
  uintN *dst_v = reinterpret_cast<uintN *>(dst2);

  for (int i = tid; i < num_vec; i += num_threads)
  {
    dst_v[i] = src_v[i];
  }

  // --- Tail: leftover bytes (< num_bytes_aligned) ---
  int tail_start = num_vec * num_bytes_aligned;
  int tail = remaining - tail_start;
  if (tid < static_cast<uint32_t>(tail))
  {
    dst2[tail_start + tid] = src2[tail_start + tid];
  }
}

template <uint32_t num_threads>
__device__ void block_copy(const uint8_t *__restrict__ src, uint8_t *__restrict__ dst, int num_bytes)
{
  const uint32_t tid = threadIdx.x;

  // Fast path: both pointers share the same 16-byte alignment offset.
  // We can align to 16B, do a vectorized middle, and handle head/tail scalars.
  uintptr_t src_addr = reinterpret_cast<uintptr_t>(src);
  uintptr_t dst_addr = reinterpret_cast<uintptr_t>(dst);

  // Fast path: both pointers share the same 16-byte alignment offset.
  if (((src_addr ^ dst_addr) & 0xF) == 0)
  {
    block_copy_aligned<16, uint4, num_threads>(src, dst, num_bytes);
  }
  // Fallback: try 4-byte aligned vectorized copy
  else if (((src_addr ^ dst_addr) & 0x3) == 0)
  {
    block_copy_aligned<4, uint32_t, num_threads>(src, dst, num_bytes);
  }
  // Worst case: byte-wise copy
  else
  {
    for (int i = tid; i < num_bytes; i += num_threads)
    {
      dst[i] = src[i];
    }
  }
}

template <typename T>
inline __device__ nvcompType_t d_TypeOf()
{
  if (std::is_same<T, int8_t>::value)
  {
    return NVCOMP_TYPE_CHAR;
  }
  else if (std::is_same<T, uint8_t>::value)
  {
    return NVCOMP_TYPE_UCHAR;
  }
  else if (std::is_same<T, int16_t>::value)
  {
    return NVCOMP_TYPE_SHORT;
  }
  else if (std::is_same<T, uint16_t>::value)
  {
    return NVCOMP_TYPE_USHORT;
  }
  else if (std::is_same<T, int32_t>::value)
  {
    return NVCOMP_TYPE_INT;
  }
  else if (std::is_same<T, uint32_t>::value)
  {
    return NVCOMP_TYPE_UINT;
  }
  else if (std::is_same<T, int64_t>::value)
  {
    return NVCOMP_TYPE_LONGLONG;
  }
  else if (std::is_same<T, uint64_t>::value)
  {
    return NVCOMP_TYPE_ULONGLONG;
  }
  else
  {
    return NVCOMP_TYPE_CHAR;
  }

  // TODO - perform error checking and notify user if incorrect type is given
}

} // namespace nvcomp::cascaded
