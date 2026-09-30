/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
*/

#pragma once

#include <cassert>
#include <cstdint>
#include <type_traits>
#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#endif // __CUDACC__
#include <stdexcept>
#include <string>

#include "CudaConstants.h"

#ifdef __CUDACC__
#define HDI __host__ __device__ inline
#else
#define HDI inline
#endif

namespace gdeflate
{

typedef enum
{
  OPTIMAL_PARSE,
  HASH_BASED,
  HUFFMAN_ONLY,
  HASH_WITH_CHAIN,
  OPTIMAL_PARSE_L6
} gdeflate_compression_internal_algo;

template <typename T>
HDI T mask(unsigned int n)
{
  static_assert(std::is_unsigned_v<T>, "Type must be unsigned");
  if (n == 0)
  {
    return 0;
  }

  constexpr int TYPE_SIZE = sizeof(T) * 8;
  int shift = TYPE_SIZE - n;
  return (T)-1 >> shift;
}

template <typename T>
HDI T ceilDiv(const T num, const T den)
{
  return (num + den - 1) / den;
}

template <typename T>
HDI T roundUpTo(const T in, const T quot)
{
  return quot * ceilDiv(in, quot);
}

// Number of padding bits in the final byte of a bitstream of `total_bits` bits.
// Zero when the stream is already byte-aligned.
template <typename T>
HDI uint8_t padBitsToByte(const T total_bits)
{
  return static_cast<uint8_t>((8 - (total_bits & 0x7)) & 0x7);
}

#ifdef __CUDACC__
template <typename T>
__inline__ __device__ T warpReduceSum(T val, unsigned int am)
{
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
  {
    val += __shfl_down_sync(am, val, offset);
  }
  val = __shfl_sync(am, val, 0);
  return val;
}

__inline__ __device__ int warpReduceMax(unsigned int val, unsigned int am)
{
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
  {
    val = max(val, __shfl_down_sync(am, val, offset));
  }
  val = __shfl_sync(am, val, 0);
  return val;
}

inline __device__ unsigned int ltMask()
{
  assert(threadIdx.x < 32);
  return -1u >> (32 - threadIdx.x - 1);
}

inline __device__ unsigned int postfixSum(unsigned int in, unsigned int am)
{
  unsigned int t0 = __shfl_up_sync(am, in, 1);
  if (threadIdx.x >= 1)
  {
    in += t0;
  }
  unsigned int t1 = __shfl_up_sync(am, in, 2);
  if (threadIdx.x >= 2)
  {
    in += t1;
  }
  unsigned int t2 = __shfl_up_sync(am, in, 4);
  if (threadIdx.x >= 4)
  {
    in += t2;
  }
  unsigned int t3 = __shfl_up_sync(am, in, 8);
  if (threadIdx.x >= 8)
  {
    in += t3;
  }
  unsigned int t4 = __shfl_up_sync(am, in, 16);
  if (threadIdx.x >= 16)
  {
    in += t4;
  }
  return in;
}

inline __device__ unsigned int prefixSum(unsigned int in, unsigned int am) { return postfixSum(in, am) - in; }

#endif // __CUDACC__

} // namespace gdeflate
