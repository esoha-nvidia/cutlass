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
#include <cstring>
#include <type_traits>

#include "cascaded/common/cascaded_warp_reductions.cuh"
#include "cascaded/next/primitives/kibi_cascader/kibi_cascader_constants.cuh"
#include "cascaded/next/primitives/kibi_cascader/kibi_cascader_encoding.cuh"
#include "cascaded/next/primitives/kibi_cascader/kibi_cascader_utils.cuh"
#include "CudaConstants.h"
#include "nvcomp_device_common.cuh"

namespace nvcomp::cascaded::next::kibi_cascader
{

// Number of run-start bits stored in each serialized RLE mask word.
inline constexpr uint32_t VALUES_PER_MASK_WORD = 32u;

template <typename T>
inline constexpr __host__ __device__ uint32_t rle_mask_size_bytes(const uint32_t input_size_bytes)
{
  static_assert(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>);
  const uint32_t num_values = input_size_bytes / sizeof(T);
  return nvcomp::roundUpDiv(num_values, VALUES_PER_MASK_WORD) * sizeof(uint32_t);
}

struct WarpRlePlan
{
  uint32_t local_run_mask;
  uint32_t num_runs;
  uint32_t thread_run_offset;
};

// Finds raw-value run starts and their compacted output offsets collectively across a warp.
template <typename T>
inline __device__ WarpRlePlan warp_find_runs(const uint32_t values[KC_WORDS_PER_THREAD], const uint32_t input_size_bytes)
{
  static_assert(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>);
  constexpr uint32_t VALUES_PER_THREAD = KC_BYTES_PER_THREAD / sizeof(T);
  const uint32_t lane = threadIdx.x % WARP_SIZE;
  const uint32_t num_values = input_size_bytes / sizeof(T);
  uint32_t local_run_mask = 0u;
  T previous = warp_shuffle_up(thread_value<T>(values, VALUES_PER_THREAD - 1u));
#pragma unroll
  for (uint32_t value = 0u; value < VALUES_PER_THREAD; ++value)
  {
    const uint32_t index = lane * VALUES_PER_THREAD + value;
    const T current = thread_value<T>(values, value);
    if (index < num_values && (index == 0u || current != previous))
    {
      local_run_mask |= 1u << value;
    }
    previous = current;
  }
  const uint32_t local_runs = __popc(local_run_mask);
  return {local_run_mask, warp_reduce<WARP_SIZE, warp_reduction_op::sum>(local_runs), warp_exclusive_sum(local_runs)};
}

// Collectively packs each lane's run-start bits into the serialized RLE mask.
template <typename T>
inline __device__ uint32_t warp_pack_run_mask_word(const uint32_t local_run_mask, const uint32_t input_size_bytes)
{
  static_assert(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>);
  constexpr uint32_t VALUES_PER_THREAD = KC_BYTES_PER_THREAD / sizeof(T);
  constexpr uint32_t LANES_PER_MASK_WORD = VALUES_PER_MASK_WORD / VALUES_PER_THREAD;
  const uint32_t lane = lane_id();
  const uint32_t num_mask_words = rle_mask_size_bytes<T>(input_size_bytes) / sizeof(uint32_t);
  const uint32_t output_mask_word = lane < num_mask_words ? lane : 0u;
  uint32_t packed_mask = 0u;
#pragma unroll
  for (uint32_t source = 0u; source < LANES_PER_MASK_WORD; ++source)
  {
    const uint32_t source_lane = output_mask_word * LANES_PER_MASK_WORD + source;
    packed_mask |= __shfl_sync(WARP_ALL, local_run_mask, source_lane) << (source * VALUES_PER_THREAD);
  }
  return lane < num_mask_words ? packed_mask : 0u;
}

// Collectively writes one value per run to swizzled shared memory and synchronizes the warp.
template <typename T>
inline __device__ void
warp_compact_runs(const uint32_t values[KC_WORDS_PER_THREAD], const WarpRlePlan plan, uint32_t *shared_output)
{
  static_assert(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>);
  constexpr uint32_t VALUES_PER_THREAD = KC_BYTES_PER_THREAD / sizeof(T);
  constexpr uint32_t WORDS_PER_VALUE = sizeof(T) / sizeof(uint32_t);
  uint32_t local_offset = 0u;
#pragma unroll
  for (uint32_t value = 0u; value < VALUES_PER_THREAD; ++value)
  {
    if ((plan.local_run_mask & (1u << value)) != 0u)
    {
      const uint32_t output_value = plan.thread_run_offset + local_offset++;
      const T run_value = thread_value<T>(values, value);
#pragma unroll
      for (uint32_t word = 0u; word < WORDS_PER_VALUE; ++word)
      {
        shared_output[shared_input_word_index(output_value * WORDS_PER_VALUE + word)] =
          static_cast<uint32_t>(run_value >> (word * 32u));
      }
    }
  }
  __syncwarp();
}

// Collectively reconstructs values from compacted runs and the serialized run-start mask.
template <typename T>
inline __device__ void warp_expand_runs(
  const uint8_t *payload,
  uint32_t values[KC_WORDS_PER_THREAD],
  uint32_t *shared_workspace,
  const uint32_t num_runs,
  const uint32_t output_size_bytes
)
{
  static_assert(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>);
  constexpr uint32_t VALUES_PER_THREAD = KC_BYTES_PER_THREAD / sizeof(T);
  const uint32_t lane = threadIdx.x % WARP_SIZE;
  const uint32_t num_values = output_size_bytes / sizeof(T);
  const uint32_t num_mask_words = rle_mask_size_bytes<T>(output_size_bytes) / sizeof(uint32_t);
#pragma unroll
  for (uint32_t value = 0u; value < VALUES_PER_THREAD; ++value)
  {
    const uint32_t index = lane * VALUES_PER_THREAD + value;
    if (index < num_runs)
    {
      const T run_value = thread_value<T>(values, value);
#pragma unroll
      for (uint32_t word = 0u; word < sizeof(T) / sizeof(uint32_t); ++word)
      {
        shared_workspace[shared_input_word_index(index * (sizeof(T) / sizeof(uint32_t)) + word)] =
          static_cast<uint32_t>(run_value >> (word * 32u));
      }
    }
  }
  // The compacted values occupy low logical workspace words while the mask
  // occupies high logical words. Both use the Kibi shared-memory swizzle.
  if (lane < num_mask_words)
  {
    memcpy(
      &shared_workspace[shared_input_word_index(MAX_KC_INPUT_SIZE_BYTES / sizeof(uint32_t) - num_mask_words + lane)],
      payload + lane * sizeof(uint32_t),
      sizeof(uint32_t)
    );
  }
  __syncwarp();

  const uint32_t mask_word_index = lane * VALUES_PER_THREAD / VALUES_PER_MASK_WORD;
  const uint32_t mask = mask_word_index < num_mask_words
                          ? shared_workspace[shared_input_word_index(
                              MAX_KC_INPUT_SIZE_BYTES / sizeof(uint32_t) - num_mask_words + mask_word_index
                            )]
                          : 0u;
  const uint32_t mask_runs =
    lane < num_mask_words
      ? __popc(
          shared_workspace[shared_input_word_index(MAX_KC_INPUT_SIZE_BYTES / sizeof(uint32_t) - num_mask_words + lane)]
        )
      : 0u;
  const uint32_t mask_run_offset = warp_exclusive_sum(mask_runs);
  const uint32_t preceding_runs = __shfl_sync(WARP_ALL, mask_run_offset, mask_word_index);
#pragma unroll
  for (uint32_t value = 0u; value < VALUES_PER_THREAD; ++value)
  {
    const uint32_t index = lane * VALUES_PER_THREAD + value;
    if (index < num_values)
    {
      const uint32_t bit = index % VALUES_PER_MASK_WORD;
      const uint32_t bits_through_value = bit == VALUES_PER_MASK_WORD - 1u ? ~0u : ((1u << (bit + 1u)) - 1u);
      const uint32_t run = preceding_runs + __popc(mask & bits_through_value) - 1u;
      T run_value{};
#pragma unroll
      for (uint32_t word = 0u; word < sizeof(T) / sizeof(uint32_t); ++word)
      {
        run_value |=
          static_cast<T>(shared_workspace[shared_input_word_index(run * (sizeof(T) / sizeof(uint32_t)) + word)])
          << (word * 32u);
      }
      store_thread_value<T>(values, value, run_value);
    }
  }
}

} // namespace nvcomp::cascaded::next::kibi_cascader
