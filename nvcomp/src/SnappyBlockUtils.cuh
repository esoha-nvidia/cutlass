/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include "LZ77_decomp.cuh"

#include <stdint.h>

#if (__CUDACC_VER_MAJOR__ >= 9)
#define SHFL0(v) __shfl_sync(WARP_ALL, v, 0)
#define SHFL(v, t) __shfl_sync(WARP_ALL, v, t)
#define SHFL_XOR(v, m) __shfl_xor_sync(WARP_ALL, v, m)
#define SYNCWARP() __syncwarp(WARP_ALL)
#define BALLOT(v) __ballot_sync(WARP_ALL, v)
#else
#define SHFL0(v) __shfl(v, 0)
#define SHFL(v, t) __shfl(v, t)
#define SHFL_XOR(v, m) __shfl_xor(v, m)
#define SYNCWARP()
#define BALLOT(v) __ballot(v)
#endif

// This reads exactly 4 bytes starting at p, so is less restrictive than
// unaligned_load32, but slower.
inline __device__ uint32_t unaligned_load32_slow(const uint8_t *p)
{
  uint32_t v = p[0] | (p[1] << 8) | (p[2] << 16) | (p[3] << 24);
  return v;
}

// This requires that two 32-bit values are in bounds if p
// is not a multiple of 4.  A sufficient check is to ensure that there are
// 7 bytes in bounds starting at p.
template <bool potentially_underaddressing>
inline __device__ uint32_t unaligned_load32(const uint8_t *p, uint32_t p_offset)
{
  uint32_t ofs = 3 & reinterpret_cast<uintptr_t>(p); // [0, 3]
  // Note: If the implicit buffer alignment is not guaranteed,
  //       then we must fall back to slow loading to make sure,
  //       that we do not underaddress the buffer.
  //       In every other case, we rely on the faster 4-byte loading.
  if constexpr (potentially_underaddressing)
  {
    if (ofs > p_offset)
    {
      // Prevent loading from before the beginning of a memory buffer
      return unaligned_load32_slow(p);
    }
  }
  const uint32_t *p32 = reinterpret_cast<const uint32_t *>(p - ofs);
  uint32_t v = p32[0];
  return (ofs) ? __funnelshift_r(v, p32[1], ofs * 8) : v;
}