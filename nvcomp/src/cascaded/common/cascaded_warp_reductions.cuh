/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or its
 * affiliates is strictly prohibited.
 */

#pragma once

#include <cuda_runtime.h>

#include "CudaConstants.h"

namespace nvcomp::cascaded
{

/**
 * Warp-reduction contract:
 *
 * Every lane in a converged warp must call these functions. Warp-wide results
 * are returned to every lane. Half-warp results are returned independently to
 * lanes [0, 15] and [16, 31].
 */

enum class warp_reduction_op
{
  sum,
  bitwise_and,
  bitwise_or,
  min,
  max,
};

template <uint32_t Width, warp_reduction_op Op, typename T>
inline __device__ T warp_reduce(T value)
{
  static_assert(Width > 0u && Width <= WARP_SIZE && (Width & (Width - 1u)) == 0u);
#pragma unroll
  for (uint32_t offset = Width / 2u; offset > 0u; offset >>= 1u)
  {
    const T other = __shfl_xor_sync(WARP_ALL, value, offset, Width);
    if constexpr (Op == warp_reduction_op::sum)
    {
      value += other;
    }
    else if constexpr (Op == warp_reduction_op::bitwise_and)
    {
      value &= other;
    }
    else if constexpr (Op == warp_reduction_op::bitwise_or)
    {
      value |= other;
    }
    else if constexpr (Op == warp_reduction_op::min)
    {
      value = other < value ? other : value;
    }
    else
    {
      static_assert(Op == warp_reduction_op::max);
      value = value < other ? other : value;
    }
  }
  return value;
}

// 64-bit values are shuffled as two 32-bit words. ParamT matches the differing third-parameter
// type between __shfl_up_sync (unsigned int delta) and __shfl_sync/__shfl_xor_sync (int).
template <typename ParamT, uint32_t Shfl32(uint32_t, uint32_t, ParamT, int32_t)>
inline __device__ uint64_t warp_shuffle_u64_words(const uint64_t bits, const ParamT param)
{
  const uint32_t lo = Shfl32(WARP_ALL, static_cast<uint32_t>(bits), param, WARP_SIZE);
  const uint32_t hi = Shfl32(WARP_ALL, static_cast<uint32_t>(bits >> 32u), param, WARP_SIZE);
  return static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32u);
}

template <typename T>
inline __device__ T warp_shuffle_up(const T value, const uint32_t offset = 1u)
{
  static_assert(sizeof(T) == sizeof(uint32_t) || sizeof(T) == sizeof(uint64_t));
  if constexpr (sizeof(T) == sizeof(uint32_t))
  {
    return static_cast<T>(__shfl_up_sync(WARP_ALL, static_cast<uint32_t>(value), offset));
  }
  else
  {
    return static_cast<T>(warp_shuffle_u64_words<uint32_t, __shfl_up_sync>(static_cast<uint64_t>(value), offset));
  }
}

template <typename T>
inline __device__ T warp_shuffle(const T value, const uint32_t source_lane)
{
  static_assert(sizeof(T) == sizeof(uint32_t) || sizeof(T) == sizeof(uint64_t));
  if constexpr (sizeof(T) == sizeof(uint32_t))
  {
    return static_cast<T>(__shfl_sync(WARP_ALL, static_cast<uint32_t>(value), source_lane));
  }
  else
  {
    return static_cast<T>(
      warp_shuffle_u64_words<int32_t, __shfl_sync>(static_cast<uint64_t>(value), static_cast<int32_t>(source_lane))
    );
  }
}

template <typename T>
inline __device__ T warp_shuffle_xor(const T value, const uint32_t lane_mask)
{
  static_assert(sizeof(T) == sizeof(uint32_t) || sizeof(T) == sizeof(uint64_t));
  if constexpr (sizeof(T) == sizeof(uint32_t))
  {
    return static_cast<T>(__shfl_xor_sync(WARP_ALL, static_cast<uint32_t>(value), lane_mask));
  }
  else
  {
    return static_cast<T>(
      warp_shuffle_u64_words<int32_t, __shfl_xor_sync>(static_cast<uint64_t>(value), static_cast<int32_t>(lane_mask))
    );
  }
}

template <typename T>
inline __device__ T warp_exclusive_sum(const T value)
{
  T inclusive = value;
  const uint32_t lane = threadIdx.x % WARP_SIZE;
#pragma unroll
  for (int offset = 1; offset < WARP_SIZE; offset <<= 1)
  {
    const T preceding = warp_shuffle_up(inclusive, static_cast<uint32_t>(offset));
    if (lane >= static_cast<uint32_t>(offset))
    {
      inclusive += preceding;
    }
  }
  return inclusive - value;
}

} // namespace nvcomp::cascaded
