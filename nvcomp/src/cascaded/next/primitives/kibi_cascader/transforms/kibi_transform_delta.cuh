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

#include <cstdint>
#include <type_traits>

#include "cascaded/next/primitives/kibi_cascader/kibi_cascader_constants.cuh"
#include "cascaded/next/primitives/kibi_cascader/kibi_cascader_utils.cuh"
#include "cascaded/next/transforms/transform_delta.cuh"
#include "CudaConstants.h"

namespace nvcomp::cascaded::next::kibi_cascader::warp_transforms
{

// Replaces each value with its unsigned delta from the preceding warp value and returns lane zero's input.
template <typename T>
inline __device__ T warp_transform_delta(uint32_t values[KC_WORDS_PER_THREAD])
{
  static_assert(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>);
  constexpr uint32_t VALUES_PER_THREAD = KC_BYTES_PER_THREAD / sizeof(T);
  const uint32_t lane = threadIdx.x % WARP_SIZE;
  const T first = thread_value<T>(values, 0u);
  const T preceding_lane_value = warp_shuffle_up(thread_value<T>(values, VALUES_PER_THREAD - 1u));
#pragma unroll
  for (uint32_t value = VALUES_PER_THREAD; value > 0u; --value)
  {
    const uint32_t index = value - 1u;
    const T preceding = index == 0u ? preceding_lane_value : thread_value<T>(values, index - 1u);
    const T transformed = lane == 0u && index == 0u
                            ? T{0}
                            : transforms::transform_delta(thread_value<T>(values, index), preceding);
    store_thread_value<T>(values, index, transformed);
  }
  return warp_shuffle(first, 0u);
}

// Collectively reconstructs values from unsigned deltas using the original first value.
template <typename T>
inline __device__ void warp_inverse_transform_delta(uint32_t values[KC_WORDS_PER_THREAD], const T first_value)
{
  static_assert(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>);
  constexpr uint32_t VALUES_PER_THREAD = KC_BYTES_PER_THREAD / sizeof(T);
  T running = 0u;
#pragma unroll
  for (uint32_t value = 0u; value < VALUES_PER_THREAD; ++value)
  {
    running = transforms::inverse_transform_delta(thread_value<T>(values, value), running);
    store_thread_value<T>(values, value, running);
  }
  const T preceding = warp_exclusive_sum(running) + first_value;
#pragma unroll
  for (uint32_t value = 0u; value < VALUES_PER_THREAD; ++value)
  {
    store_thread_value<T>(values, value, transforms::inverse_transform_delta(thread_value<T>(values, value), preceding));
  }
}

} // namespace nvcomp::cascaded::next::kibi_cascader::warp_transforms
