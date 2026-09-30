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
#include "nvcomp_device_common.cuh"

namespace nvcomp::cascaded::next::kibi_cascader::warp_transforms
{

namespace
{
template <typename T, uint32_t KC_BYTES_PER_THREAD>
constexpr uint32_t values_per_thread()
{
  return KC_BYTES_PER_THREAD / sizeof(T);
}
} // namespace

template <typename T>
inline __device__ void thread_transform_frame_of_reference(uint32_t values[KC_WORDS_PER_THREAD], const T reference)
{
  static_assert(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>);
#pragma unroll
  for (uint32_t value = 0u; value < values_per_thread<T, KC_BYTES_PER_THREAD>(); ++value)
  {
    store_thread_value<T>(values, value, transforms::transform_delta(thread_value<T>(values, value), reference));
  }
}

template <typename T>
inline __device__ void
thread_inverse_transform_frame_of_reference(uint32_t values[KC_WORDS_PER_THREAD], const T reference)
{
  static_assert(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>);
#pragma unroll
  for (uint32_t value = 0u; value < values_per_thread<T, KC_BYTES_PER_THREAD>(); ++value)
  {
    store_thread_value<T>(values, value, transforms::inverse_transform_delta(thread_value<T>(values, value), reference));
  }
}

template <typename T>
inline __device__ void warp_transform_delta_frame_of_reference(
  uint32_t values[KC_WORDS_PER_THREAD],
  const uint32_t input_size_bytes,
  const T reference
)
{
  static_assert(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>);
  constexpr uint32_t VALUES_PER_THREAD = values_per_thread<T, KC_BYTES_PER_THREAD>();
  const uint32_t lane = lane_id();
  const uint32_t num_values = input_size_bytes / sizeof(T);
  const T first_delta = warp_shuffle(num_values > 1u ? thread_value<T>(values, 1u) : T{0}, 0u);
#pragma unroll
  for (uint32_t value = 0u; value < VALUES_PER_THREAD; ++value)
  {
    const uint32_t index = lane * VALUES_PER_THREAD + value;
    const T input = index == 0u || index >= num_values ? first_delta : thread_value<T>(values, value);
    store_thread_value<T>(values, value, transforms::transform_delta(input, reference));
  }
}

inline __device__ void warp_store_frame_of_reference(
  uint32_t *shared_output,
  const uint32_t values[KC_WORDS_PER_THREAD],
  const uint32_t reference
)
{
  constexpr uint32_t VALUES_PER_THREAD = values_per_thread<uint32_t, KC_BYTES_PER_THREAD>();
  const uint32_t lane = lane_id();
#pragma unroll
  for (uint32_t value = 0u; value < VALUES_PER_THREAD; ++value)
  {
    const uint32_t index = lane * VALUES_PER_THREAD + value;
    const uint32_t transformed = transforms::transform_delta(thread_value<uint32_t>(values, value), reference);
    shared_output[shared_input_word_index(index)] = transformed;
  }
}

inline __device__ void warp_store_delta_frame_of_reference(
  uint32_t *shared_output,
  const uint32_t deltas[KC_WORDS_PER_THREAD],
  const uint32_t input_size_bytes,
  const uint32_t reference
)
{
  const uint32_t lane = lane_id();
  const uint32_t num_values = input_size_bytes / sizeof(uint32_t);
  const uint32_t first_delta = warp_shuffle(num_values > 1u ? deltas[1u] : 0u, 0u);
#pragma unroll
  for (uint32_t word = 0u; word < KC_WORDS_PER_THREAD; ++word)
  {
    const uint32_t index = lane * KC_WORDS_PER_THREAD + word;
    const uint32_t value = index == 0u || index >= num_values ? first_delta : deltas[word];
    shared_output[shared_input_word_index(index)] = transforms::transform_delta(value, reference);
  }
}

} // namespace nvcomp::cascaded::next::kibi_cascader::warp_transforms
