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
#include "cascaded/next/transforms/transform_zigzag.cuh"

namespace nvcomp::cascaded::next::kibi_cascader::warp_transforms
{

template <typename T>
inline __device__ void thread_transform_zigzag(uint32_t values[KC_WORDS_PER_THREAD])
{
  static_assert(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>);
  constexpr uint32_t VALUES_PER_THREAD = KC_BYTES_PER_THREAD / sizeof(T);
#pragma unroll
  for (uint32_t value = 0u; value < VALUES_PER_THREAD; ++value)
  {
    store_thread_value<T>(values, value, transforms::transform_zigzag(thread_value<T>(values, value)));
  }
}

inline __device__ void warp_store_delta_zigzag(uint32_t *shared_output, const uint32_t deltas[KC_WORDS_PER_THREAD])
{
  const uint32_t lane = threadIdx.x % WARP_SIZE;
#pragma unroll
  for (uint32_t word = 0u; word < KC_WORDS_PER_THREAD; ++word)
  {
    const uint32_t index = lane * KC_WORDS_PER_THREAD + word;
    shared_output[shared_input_word_index(index)] = transforms::transform_zigzag(deltas[word]);
  }
}

} // namespace nvcomp::cascaded::next::kibi_cascader::warp_transforms
