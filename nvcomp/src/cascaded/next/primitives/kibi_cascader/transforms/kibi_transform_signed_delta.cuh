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
#include <limits>
#include <type_traits>

#include "cascaded/next/primitives/kibi_cascader/kibi_cascader_constants.cuh"
#include "cascaded/next/primitives/kibi_cascader/kibi_cascader_utils.cuh"
#include "cascaded/next/transforms/transform_delta.cuh"
#include "cascaded/next/transforms/transform_zigzag.cuh"
#include "CudaConstants.h"
#include "nvcomp_device_common.cuh"

namespace nvcomp::cascaded::next::kibi_cascader::warp_transforms
{

template <typename T>
struct SignedRange
{
  using SignedT = std::make_signed_t<T>;
  SignedT minimum{std::numeric_limits<SignedT>::max()};
  SignedT maximum{std::numeric_limits<SignedT>::min()};
};

// Deltas are stored as unsigned bit patterns but represent signed values, so the
// min/max must be computed in the signed domain.
template <typename T>
inline __device__ SignedRange<T>
warp_signed_delta_range(const uint32_t values[KC_WORDS_PER_THREAD], const uint32_t input_size_bytes)
{
  static_assert(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>);
  using SignedT = typename SignedRange<T>::SignedT;
  constexpr uint32_t VALUES_PER_THREAD = KC_BYTES_PER_THREAD / sizeof(T);
  const uint32_t lane = lane_id();
  const uint32_t num_values = input_size_bytes / sizeof(T);
  SignedRange<T> range;
#pragma unroll
  for (uint32_t value = 0u; value < VALUES_PER_THREAD; ++value)
  {
    const uint32_t index = lane * VALUES_PER_THREAD + value;
    if (index > 0u && index < num_values)
    {
      const SignedT delta = static_cast<SignedT>(thread_value<T>(values, value));
      range.minimum = min(range.minimum, delta);
      range.maximum = max(range.maximum, delta);
    }
  }
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
  {
    const SignedT other_minimum = warp_shuffle_xor(range.minimum, static_cast<uint32_t>(offset));
    const SignedT other_maximum = warp_shuffle_xor(range.maximum, static_cast<uint32_t>(offset));
    range.minimum = min(range.minimum, other_minimum);
    range.maximum = max(range.maximum, other_maximum);
  }
  if (num_values <= 1u)
  {
    range.minimum = 0;
    range.maximum = 0;
  }
  return range;
}

template <typename T>
inline __device__ void warp_inverse_transform_signed_delta(uint32_t values[KC_WORDS_PER_THREAD], const T first_value)
{
  static_assert(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>);
  constexpr uint32_t VALUES_PER_THREAD = KC_BYTES_PER_THREAD / sizeof(T);
  T running = 0u;
#pragma unroll
  for (uint32_t value = 0u; value < VALUES_PER_THREAD; ++value)
  {
    running =
      transforms::inverse_transform_delta(transforms::inverse_transform_zigzag(thread_value<T>(values, value)), running);
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
