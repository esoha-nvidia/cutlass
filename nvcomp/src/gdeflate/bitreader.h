/*
 * Copyright (c) 2021-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include <cuda.h>

#include <cassert>
#include <cstdint>

#include "common.h"

namespace gdeflate
{

// Class to cooperatively read the swizzled bitstream
template <
  typename T = uint32_t, // Type of the input gdeflate stream
  typename Tb = uint64_t, // Type of the internal bitstream state in registers
  unsigned int N = WARP_SIZE_U> // Number of SIMD lanes
class warp_bitreader
{

  const T *input;
  Tb buf;
  unsigned int cnt;
  const T *const input_end;
  bool corrupted;
  bool end_pos;

  inline __device__ void refill(bool active = true)
  {
    __syncwarp();
    active = active && (cnt < width);
    unsigned int ballot = __ballot_sync(WARP_ALL, active);
    unsigned int offset = __popc(ballot & ltMask()) - 1;
    if (active)
    {
      corrupted |= (input + offset >= input_end);
      buf |= corrupted ? 0 : ((Tb)(input[offset]) << cnt);
      cnt += width;
    }
    // Advance the input pointer for all threads
    input += __popc(ballot);
    corrupted = (__any_sync(WARP_ALL, corrupted) != 0);
  }

public:
  static constexpr unsigned int width = sizeof(T) * 8;

  __device__ warp_bitreader(const T *base, const T *base_end)
      : input(base)
      , input_end(base_end)
      , buf{0}
      , cnt{0}
      , corrupted{false}
      , end_pos{false}
  {
    corrupted |= (input + WARP_SIZE_U > input_end);
    cnt = width;
    buf = corrupted ? 0 : (Tb)input[threadIdx.x];
    input += WARP_SIZE_U;
  }

  inline __device__ T read(unsigned int n, bool active = true)
  {
    assert(n <= width);
    T bits = active ? buf & mask<T>(n) : 0;
    eat(n, active);
    return bits;
  }

  inline __device__ T read_aligned(unsigned int n, bool active = true)
  {
    assert(n + (cnt & 7) <= width);
    T bits = 0;
    if (active)
    {
      buf >>= cnt & 7;
      bits = buf & mask<T>(n);
      cnt &= ~7;
    }
    eat(n + (cnt & 7), active);
    return bits;
  }

  inline __device__ T peek(unsigned int n, bool active = true)
  {
    assert(n <= width);
    return active ? (T)buf & mask<T>(n) : 0;
  }

  inline __device__ T peek(bool active = true) { return active ? (T)buf : 0; }

  inline __device__ void eat(unsigned int n, bool active = true)
  {
    assert(n <= width);
    if (active)
    {
      buf >>= n;
      cnt -= n;
    }
    refill(active);
  }

  inline __device__ bool is_corrupted() { return corrupted; }
};

} // namespace gdeflate
