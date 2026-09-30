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

#include <cassert>
#include <cstdint>
#include <type_traits>

#include "cascaded/common/cascaded_warp_reductions.cuh"
#include "cascaded/next/primitives/kibi_cascader/kibi_cascader_constants.cuh"
#include "CudaConstants.h"
#include "kibi_cascader_packed_uint64_warp.cuh"
#include "nvcomp/utils.hpp"

namespace nvcomp::cascaded::next::kibi_cascader
{

using nvcomp::cascaded::warp_exclusive_sum;
using nvcomp::cascaded::warp_shuffle;
using nvcomp::cascaded::warp_shuffle_up;
using nvcomp::cascaded::warp_shuffle_xor;

// Maps a logical KC word index to its bank-conflict-free shared-memory index.
inline constexpr __host__ __device__ uint32_t shared_input_word_index(const uint32_t logical_word)
{
  return logical_word ^ (logical_word / KC_WORDS_PER_THREAD);
}

inline __device__ uint32_t mask_for_range(const uint32_t range)
{
  return range == 0u ? 0u : (0xFFFFFFFFu >> __clz(range));
}

template <typename T>
inline __device__ T thread_value(const uint32_t values[KC_WORDS_PER_THREAD], const uint32_t index)
{
  static_assert(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>);
  if constexpr (std::is_same_v<T, uint32_t>)
  {
    return values[index];
  }
  else
  {
    return unpack_uint64(PackedUint64(values + 2u * index));
  }
}

template <typename T>
inline __device__ void store_thread_value(uint32_t values[KC_WORDS_PER_THREAD], const uint32_t index, const T value)
{
  static_assert(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>);
  if constexpr (std::is_same_v<T, uint32_t>)
  {
    values[index] = value;
  }
  else
  {
    store_packed_uint64<KC_WORDS_PER_THREAD>(values, index, pack_uint64(value));
  }
}

template <typename T>
inline __device__ void
thread_normalize_padding(uint32_t values[KC_WORDS_PER_THREAD], const uint32_t input_size_bytes, const T padding)
{
  static_assert(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>);
  constexpr uint32_t VALUES_PER_THREAD = KC_BYTES_PER_THREAD / sizeof(T);
  const uint32_t lane = threadIdx.x % WARP_SIZE;
  const uint32_t num_values = input_size_bytes / sizeof(T);
#pragma unroll
  for (uint32_t value = 0u; value < VALUES_PER_THREAD; ++value)
  {
    if (lane * VALUES_PER_THREAD + value >= num_values)
    {
      store_thread_value<T>(values, value, padding);
    }
  }
}

// Returns the interleaved source word for a destination word in the uint64_t word-plane layout.
inline constexpr __host__ __device__ uint32_t uint64_word_plane_source_index(const uint32_t planed_word)
{
  constexpr uint32_t WORDS_PER_PLANE = MAX_KC_INPUT_SIZE_BYTES / sizeof(PackedUint64);
  return 2u * (planed_word % WORDS_PER_PLANE) + planed_word / WORDS_PER_PLANE;
}

/**
  Returns the number of active lanes in each type-specific bitpack input. For uint32_t this is the
  number of active lanes in the full-warp bitpack. For uint64_t this is the number of active lanes
  in each word plane. The caller must pad unused shared-memory words in the final active lane, but
  input_size_bytes must exclude that padding.

  @tparam T Data type of the input vector
  @param input_size_bytes Logical input size in bytes
  @return Number of active bitpack lanes
*/
template <typename T>
inline constexpr __host__ __device__ uint32_t num_active_bitpack_lanes(const uint32_t input_size_bytes)
{
  assert(input_size_bytes <= MAX_KC_INPUT_SIZE_BYTES); // 1KB is the maximum input size
  assert(input_size_bytes % sizeof(T) == 0u); // Input must contain complete values

  const uint32_t num_values = nvcomp::roundUpDiv(input_size_bytes, static_cast<uint32_t>(sizeof(T)));
  return nvcomp::roundUpDiv(num_values, KC_WORDS_PER_THREAD);
}

namespace detail
{
template <int LowBit>
inline __device__ void swap_word_index_bits(uint32_t values[KC_WORDS_PER_THREAD])
{
  static_assert(LowBit >= 0 && LowBit < 7);
  constexpr int HIGH_BIT = LowBit + 1;
  if constexpr (HIGH_BIT < 3)
  {
    constexpr uint32_t LOW_MASK = 1u << LowBit;
    constexpr uint32_t HIGH_MASK = 1u << HIGH_BIT;
#pragma unroll
    for (uint32_t word = 0u; word < KC_WORDS_PER_THREAD; ++word)
    {
      if ((word & LOW_MASK) == 0u && (word & HIGH_MASK) != 0u)
      {
        const uint32_t partner = word ^ LOW_MASK ^ HIGH_MASK;
        const uint32_t temporary = values[word];
        values[word] = values[partner];
        values[partner] = temporary;
      }
    }
  }
  else if constexpr (LowBit >= 3)
  {
    constexpr uint32_t LOW_MASK = 1u << (LowBit - 3);
    constexpr uint32_t HIGH_MASK = 1u << (HIGH_BIT - 3);
    const uint32_t lane = threadIdx.x % WARP_SIZE;
    const bool bits_differ = ((lane & LOW_MASK) == 0u) != ((lane & HIGH_MASK) == 0u);
    const uint32_t source_lane = bits_differ ? lane ^ LOW_MASK ^ HIGH_MASK : lane;
#pragma unroll
    for (uint32_t word = 0u; word < KC_WORDS_PER_THREAD; ++word)
    {
      values[word] = __shfl_sync(WARP_ALL, values[word], source_lane);
    }
  }
  else
  {
    constexpr uint32_t REGISTER_MASK = 1u << LowBit;
    constexpr uint32_t LANE_MASK = 1u << (HIGH_BIT - 3);
    const uint32_t lane = threadIdx.x % WARP_SIZE;
    const bool high_lane = (lane & LANE_MASK) != 0u;
#pragma unroll
    for (uint32_t word = 0u; word < KC_WORDS_PER_THREAD; ++word)
    {
      if ((word & REGISTER_MASK) == 0u)
      {
        const uint32_t partner = word | REGISTER_MASK;
        const uint32_t low = values[word];
        const uint32_t high = values[partner];
        const uint32_t low_from_partner_lane = __shfl_xor_sync(WARP_ALL, low, LANE_MASK);
        const uint32_t high_from_partner_lane = __shfl_xor_sync(WARP_ALL, high, LANE_MASK);
        values[word] = high_lane ? high_from_partner_lane : low;
        values[partner] = high_lane ? high : low_from_partner_lane;
      }
    }
  }
}
} // namespace detail

/**
 * Word-planes a 1KB uint64_t vector in place across a warp.
 *
 * Each lane initially owns four uint64_t values as eight interleaved uint32_t
 * words. Lanes [0, 15] receive the low words and lanes [16, 31] receive the
 * high words, with eight consecutive words per lane in each plane.
 *
 * Every lane in a converged warp must call this function.
 * Transforms that produce interleaved uint64_t words in registers can use this
 * helper; the untransformed plan path loads words directly into planed positions.
 */
inline __device__ void word_plane_uint64_in_place(uint32_t values[KC_WORDS_PER_THREAD])
{
  detail::swap_word_index_bits<0>(values);
  detail::swap_word_index_bits<1>(values);
  detail::swap_word_index_bits<2>(values);
  detail::swap_word_index_bits<3>(values);
  detail::swap_word_index_bits<4>(values);
  detail::swap_word_index_bits<5>(values);
  detail::swap_word_index_bits<6>(values);
}

/**
 * Reconstructs the original interleaved uint64_t layout from two word planes.
 *
 * This is the inverse of word_plane_uint64_in_place: each lane receives four
 * uint64_t values as eight interleaved low/high uint32_t words.
 *
 * Every lane in a converged warp must call this function.
 */
inline __device__ void unplane_uint64_in_place(uint32_t values[KC_WORDS_PER_THREAD])
{
  detail::swap_word_index_bits<6>(values);
  detail::swap_word_index_bits<5>(values);
  detail::swap_word_index_bits<4>(values);
  detail::swap_word_index_bits<3>(values);
  detail::swap_word_index_bits<2>(values);
  detail::swap_word_index_bits<1>(values);
  detail::swap_word_index_bits<0>(values);
}

} // namespace nvcomp::cascaded::next::kibi_cascader
