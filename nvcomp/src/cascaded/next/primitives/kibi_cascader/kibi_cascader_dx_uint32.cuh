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
#include <limits>

#include "cascaded/bitpack/bitpack_full_warp_dx.cuh"
#include "cascaded/common/cascaded_warp_reductions.cuh"
#include "cascaded/next/primitives/kibi_cascader/decompositions/kibi_decompose_rle.cuh"
#include "cascaded/next/primitives/kibi_cascader/kibi_cascader_dx.cuh"
#include "cascaded/next/primitives/kibi_cascader/kibi_cascader_encoding.cuh"
#include "cascaded/next/primitives/kibi_cascader/kibi_cascader_utils.cuh"
#include "cascaded/next/primitives/kibi_cascader/transforms/kibi_transform_dispatch.cuh"
#include "CudaConstants.h"
#include "nvcomp_device_common.cuh"

namespace nvcomp::cascaded::next::kibi_cascader
{

template <>
struct KibiCascader<uint32_t>
{
  static_assert(KC_WORDS_PER_THREAD == bitpack::BITPACK_WORDS_PER_THREAD);

  static inline __device__ void
  load_input(const uint32_t *shared_input, uint32_t values[KC_WORDS_PER_THREAD], const uint32_t input_size_bytes)
  {
    const uint32_t lane = lane_id();
    const uint32_t num_values = input_size_bytes / sizeof(uint32_t);
    const uint32_t padding = shared_input[shared_input_word_index(0u)];
#pragma unroll
    for (uint32_t word = 0u; word < KC_WORDS_PER_THREAD; ++word)
    {
      const uint32_t index = lane * KC_WORDS_PER_THREAD + word;
      values[word] = index < num_values ? shared_input[shared_input_word_index(index)] : padding;
    }
  }

  static inline __device__ uint32_t
  warp_input_minimum(const uint32_t values[KC_WORDS_PER_THREAD], const uint32_t input_size_bytes)
  {
    const uint32_t lane = lane_id();
    const uint32_t num_values = input_size_bytes / sizeof(uint32_t);
    uint32_t local_min = std::numeric_limits<uint32_t>::max();
#pragma unroll
    for (uint32_t word = 0u; word < KC_WORDS_PER_THREAD; ++word)
    {
      if (lane * KC_WORDS_PER_THREAD + word < num_values)
      {
        local_min = min(local_min, values[word]);
      }
    }
    return warp_reduce<WARP_SIZE, warp_reduction_op::min>(local_min);
  }

  struct BitpackEstimate
  {
    uint32_t change_mask;
    uint32_t payload_size_bytes;
  };

  template <typename Transform>
  static inline __device__ BitpackEstimate
  bitpack_plan(const uint32_t values[KC_WORDS_PER_THREAD], const uint32_t input_size_bytes, Transform transform)
  {
    const uint32_t num_active = num_active_bitpack_lanes<uint32_t>(input_size_bytes);
    const uint32_t lane = lane_id();
    uint32_t local_or = 0u;
#pragma unroll
    for (uint32_t word = 0u; word < KC_WORDS_PER_THREAD; ++word)
    {
      local_or |= lane < num_active ? transform(values[word]) : 0u;
    }
    const uint32_t change_mask = warp_reduce<WARP_SIZE, warp_reduction_op::bitwise_or>(local_or);
    return {change_mask, bitpack::warp_bitpack_get_compressed_size(change_mask, num_active)};
  }

  static inline __device__ BitpackEstimate
  simple_bitpack_plan(const uint32_t values[KC_WORDS_PER_THREAD], const uint32_t input_size_bytes)
  {
    return bitpack_plan(values, input_size_bytes, [] __device__(uint32_t v) { return v; });
  }

  static inline __device__ BitpackEstimate frame_of_reference_plan(
    const uint32_t values[KC_WORDS_PER_THREAD],
    const uint32_t reference,
    const uint32_t input_size_bytes
  )
  {
    // reference is the warp minimum, so this unsigned subtract cannot wrap.
    return bitpack_plan(values, input_size_bytes, [reference] __device__(uint32_t v) { return v - reference; });
  }

  static inline __device__ BitpackEstimate
  delta_zigzag_plan(const uint32_t deltas[KC_WORDS_PER_THREAD], const uint32_t input_size_bytes)
  {
    return bitpack_plan(deltas, input_size_bytes, [] __device__(uint32_t v) {
      return transforms::transform_zigzag(v);
    });
  }

  static inline __device__ KibiCascaderCompressionPlan
  plan_bitpack_input(uint32_t values[KC_WORDS_PER_THREAD], uint32_t *shared_metadata, const uint32_t input_size_bytes)
  {
    const uint32_t num_active = num_active_bitpack_lanes<uint32_t>(input_size_bytes);
    const uint32_t lane = lane_id();
    const bitpack::WarpBitpackCompressionPlan bitpack_plan = bitpack::warp_bitpack_compress_init(values, num_active);
    if (lane == 0u)
    {
      shared_metadata[0u] = bitpack_plan.change_mask;
    }
    else if (lane == 1u)
    {
      shared_metadata[1u] = bitpack_plan.base_value;
    }
    __syncwarp();
    return {input_size_bytes, 0u, bitpack_plan.payload_size, 0u};
  }

  template <kibi_cascader_optimization_search_space SearchSpace>
  static inline __device__ KibiCascaderCompressionPlan plan_without_rle(
    uint32_t *shared_input,
    uint32_t bitpack_input[KC_WORDS_PER_THREAD],
    uint32_t *shared_metadata,
    const uint32_t input_size_bytes
  )
  {
    static_assert((SearchSpace & SEARCH_RLE) == 0u);
    const uint32_t num_active = num_active_bitpack_lanes<uint32_t>(input_size_bytes);
    load_input(shared_input, bitpack_input, input_size_bytes);

    [[maybe_unused]] PreBitpackTransform transform = PreBitpackTransform::None;
    const bitpack::WarpBitpackCompressionPlan baseline = bitpack::warp_bitpack_compress_init(bitpack_input, num_active);
    uint32_t change_mask = baseline.change_mask;
    uint32_t primary = baseline.base_value;
    [[maybe_unused]] uint32_t secondary = 0u;
    uint32_t payload_size_bytes = baseline.payload_size;

    if constexpr ((SearchSpace & SEARCH_FOR) != 0u)
    {
      const uint32_t reference = warp_input_minimum(bitpack_input, input_size_bytes);
      const BitpackEstimate candidate = frame_of_reference_plan(bitpack_input, reference, input_size_bytes);
      if (candidate.payload_size_bytes < payload_size_bytes)
      {
        transform = PreBitpackTransform::FrameOfReference;
        change_mask = candidate.change_mask;
        primary = reference;
        payload_size_bytes = candidate.payload_size_bytes;
        if constexpr ((SearchSpace & SEARCH_DELTA) != 0u)
        {
          warp_transforms::warp_store_frame_of_reference(shared_input, bitpack_input, reference);
          __syncwarp();
        }
      }
    }

    if constexpr ((SearchSpace & SEARCH_DELTA) != 0u)
    {
      const uint32_t first_value = warp_transforms::warp_transform_delta<uint32_t>(bitpack_input);
      const BitpackEstimate delta = simple_bitpack_plan(bitpack_input, input_size_bytes);
      if (delta.payload_size_bytes < payload_size_bytes)
      {
        transform = PreBitpackTransform::Delta;
        change_mask = delta.change_mask;
        primary = first_value;
        secondary = 0u;
        payload_size_bytes = delta.payload_size_bytes;
      }

      const auto [delta_minimum, delta_maximum] =
        warp_transforms::warp_signed_delta_range<uint32_t>(bitpack_input, input_size_bytes);
      const uint32_t range = static_cast<uint32_t>(delta_maximum) - static_cast<uint32_t>(delta_minimum);
      const uint32_t candidate_mask = mask_for_range(range);
      const uint32_t candidate_size = bitpack::warp_bitpack_get_compressed_size(candidate_mask, num_active);
      if (candidate_size < payload_size_bytes)
      {
        transform = PreBitpackTransform::DeltaFrameOfReference;
        change_mask = candidate_mask;
        primary = first_value;
        secondary = static_cast<uint32_t>(delta_minimum);
        payload_size_bytes = candidate_size;
        warp_transforms::warp_store_delta_frame_of_reference(
          shared_input,
          bitpack_input,
          input_size_bytes,
          static_cast<uint32_t>(delta_minimum)
        );
        __syncwarp();
      }

      const BitpackEstimate zigzag = delta_zigzag_plan(bitpack_input, input_size_bytes);
      if (zigzag.payload_size_bytes < payload_size_bytes)
      {
        transform = PreBitpackTransform::DeltaZigzag;
        change_mask = zigzag.change_mask;
        primary = first_value;
        secondary = 0u;
        payload_size_bytes = zigzag.payload_size_bytes;
        warp_transforms::warp_store_delta_zigzag(shared_input, bitpack_input);
        __syncwarp();
      }
    }

    // Shared input contains the current winner unless the in-register delta is
    // still best. Each lane reloads only the fixed locations it may have
    // overwritten, so no warp barrier is needed.
    if constexpr ((SearchSpace & SEARCH_DELTA) != 0u)
    {
      if (transform != PreBitpackTransform::Delta)
      {
        load_input(shared_input, bitpack_input, input_size_bytes);
      }
    }
    else if constexpr ((SearchSpace & SEARCH_FOR) != 0u)
    {
      if (transform == PreBitpackTransform::FrameOfReference)
      {
        warp_transforms::warp_apply_transform<uint32_t>(bitpack_input, transform, input_size_bytes, primary);
      }
    }

    const uint32_t lane = lane_id();
    if (lane == 0u)
    {
      shared_metadata[0u] = change_mask;
    }
    else if (lane == 1u)
    {
      shared_metadata[1u] = primary;
    }
    if constexpr ((SearchSpace & (SEARCH_DELTA | SEARCH_FOR)) != 0u)
    {
      if (lane == 2u)
      {
        shared_metadata[2u] = encoding_metadata(transform, RleMode::None);
      }
    }
    if constexpr ((SearchSpace & SEARCH_DELTA) != 0u)
    {
      if (lane == 3u)
      {
        shared_metadata[3u] = secondary;
      }
    }
    (void)transform;
    (void)secondary;
    __syncwarp();
    return {input_size_bytes, 0u, payload_size_bytes, 0u};
  }

  template <kibi_cascader_optimization_search_space SearchSpace>
  static inline __device__ KibiCascaderCompressionPlan plan(
    uint32_t *shared_input,
    uint32_t bitpack_input[KC_WORDS_PER_THREAD],
    uint32_t *shared_metadata,
    [[maybe_unused]] uint32_t *shared_workspace,
    const uint32_t input_size_bytes
  )
  {
    static_assert(get_num_metadata_words<uint32_t, SearchSpace>() <= MAX_KC_METADATA_WORDS);
    if constexpr ((SearchSpace & SEARCH_RLE) == 0u)
    {
      return plan_without_rle<SearchSpace>(shared_input, bitpack_input, shared_metadata, input_size_bytes);
    }
    else
    {
      constexpr auto TRANSFORM_SEARCH = SearchSpace & ~SEARCH_RLE;
      constexpr uint32_t RLE_SIZE_WORD = 3u + static_cast<uint32_t>((TRANSFORM_SEARCH & SEARCH_DELTA) != 0u);
      const uint32_t lane = lane_id();

      // Preserve the raw input while the direct plan uses shared_input to
      // materialize provisional transform winners.
#pragma unroll
      for (uint32_t word = 0u; word < KC_WORDS_PER_THREAD; ++word)
      {
        const uint32_t index = lane * KC_WORDS_PER_THREAD + word;
        shared_workspace[shared_input_word_index(index)] = shared_input[shared_input_word_index(index)];
      }
      __syncwarp();

      load_input(shared_input, bitpack_input, input_size_bytes);
      const WarpRlePlan raw_rle = warp_find_runs<uint32_t>(bitpack_input, input_size_bytes);
      const KibiCascaderCompressionPlan direct_plan =
        plan_without_rle<TRANSFORM_SEARCH>(shared_input, bitpack_input, shared_metadata, input_size_bytes);
      const PreBitpackTransform direct_transform = (TRANSFORM_SEARCH & (SEARCH_DELTA | SEARCH_FOR)) != 0u
                                                     ? get_pre_bitpack_transform(shared_metadata[2u])
                                                     : PreBitpackTransform::None;
      const uint32_t direct_primary = shared_metadata[1u];
      const uint32_t direct_secondary = (TRANSFORM_SEARCH & SEARCH_DELTA) != 0u ? shared_metadata[3u] : 0u;
      const uint32_t prefix_size_bytes = rle_mask_size_bytes<uint32_t>(input_size_bytes);
      uint32_t bitpack_input_size_bytes = input_size_bytes;
      uint32_t payload_size_bytes = direct_plan.payload_size_bytes;
      RleMode rle_mode = RleMode::None;
      const bool raw_fits_workspace = raw_rle.num_runs + prefix_size_bytes / sizeof(uint32_t) <=
                                      MAX_KC_INPUT_SIZE_BYTES / sizeof(uint32_t);
      const uint32_t raw_size_bytes = raw_rle.num_runs * sizeof(uint32_t);
      const uint32_t estimated_rle_payload_size_bytes = prefix_size_bytes +
                                                        bitpack::warp_bitpack_get_compressed_size(
                                                          shared_metadata[0u],
                                                          num_active_bitpack_lanes<uint32_t>(raw_size_bytes)
                                                        );
      if (raw_rle.num_runs < input_size_bytes / sizeof(uint32_t) && raw_fits_workspace &&
          estimated_rle_payload_size_bytes < direct_plan.payload_size_bytes)
      {
        const uint32_t direct_change_mask = shared_metadata[0u];
        const uint32_t direct_base_value = shared_metadata[1u];
        load_input(shared_workspace, bitpack_input, input_size_bytes);
        warp_compact_runs<uint32_t>(bitpack_input, raw_rle, shared_input);
        load_input(shared_input, bitpack_input, raw_size_bytes);
        warp_transforms::warp_apply_transform<uint32_t>(
          bitpack_input,
          direct_transform,
          raw_size_bytes,
          direct_primary,
          direct_secondary
        );
        uint32_t rle_change_mask = 0u;
        uint32_t rle_base_value = 0u;
        uint32_t rle_payload_size_bytes = 0u;
        if (direct_transform == PreBitpackTransform::None)
        {
          const auto bitpack_plan =
            bitpack::warp_bitpack_compress_init(bitpack_input, num_active_bitpack_lanes<uint32_t>(raw_size_bytes));
          rle_change_mask = bitpack_plan.change_mask;
          rle_base_value = bitpack_plan.base_value;
          rle_payload_size_bytes = prefix_size_bytes + bitpack_plan.payload_size;
        }
        else
        {
          const BitpackEstimate estimate = simple_bitpack_plan(bitpack_input, raw_size_bytes);
          rle_change_mask = estimate.change_mask;
          rle_payload_size_bytes = prefix_size_bytes + estimate.payload_size_bytes;
        }
        if (rle_payload_size_bytes < direct_plan.payload_size_bytes)
        {
          if (lane == 0u)
          {
            shared_metadata[0u] = rle_change_mask;
          }
          else if (direct_transform == PreBitpackTransform::None && lane == 1u)
          {
            shared_metadata[1u] = rle_base_value;
          }
          payload_size_bytes = rle_payload_size_bytes;
          bitpack_input_size_bytes = raw_size_bytes;
          rle_mode = RleMode::Before;
        }
        else
        {
          load_input(shared_workspace, bitpack_input, input_size_bytes);
          warp_transforms::warp_apply_transform<uint32_t>(
            bitpack_input,
            direct_transform,
            input_size_bytes,
            direct_primary,
            direct_secondary
          );
          if (lane == 0u)
          {
            shared_metadata[0u] = direct_change_mask;
          }
          else if (lane == 1u)
          {
            shared_metadata[1u] = direct_base_value;
          }
        }
      }

      __syncwarp();

      if (lane == 2u)
      {
        shared_metadata[2u] = encoding_metadata(direct_transform, rle_mode);
      }
      if (lane == RLE_SIZE_WORD)
      {
        shared_metadata[RLE_SIZE_WORD] = bitpack_input_size_bytes;
      }
      __syncwarp();
      const uint32_t payload_prefix_word =
        rle_mode == RleMode::None ? 0u : warp_pack_run_mask_word<uint32_t>(raw_rle.local_run_mask, input_size_bytes);
      return {
        bitpack_input_size_bytes,
        rle_mode == RleMode::None ? 0u : prefix_size_bytes,
        payload_size_bytes,
        payload_prefix_word
      };
    }
  }

  template <kibi_cascader_optimization_search_space SearchSpace>
  static inline __device__ void encode(
    uint32_t bitpack_input[KC_WORDS_PER_THREAD],
    const uint32_t *shared_metadata,
    uint8_t *payload,
    const KibiCascaderCompressionPlan &plan
  )
  {
    write_payload_prefix(payload, plan);
    const uint32_t num_active = num_active_bitpack_lanes<uint32_t>(plan.bitpack_input_size_bytes);
    bitpack::warp_bitpack_compress_apply(
      bitpack_input,
      payload + plan.payload_prefix_size_bytes,
      shared_metadata[0u],
      num_active
    );
  }

  template <kibi_cascader_optimization_search_space SearchSpace>
  static inline __device__ void decode(
    const uint8_t *payload,
    uint32_t output[KC_WORDS_PER_THREAD],
    const uint32_t *shared_metadata,
    uint32_t *shared_workspace,
    const uint32_t output_size_bytes
  )
  {
    PreBitpackTransform transform = PreBitpackTransform::None;
    RleMode rle_mode = RleMode::None;
    uint32_t secondary = 0u;
    uint32_t bitpack_size = output_size_bytes;
    if constexpr ((SearchSpace & (SEARCH_DELTA | SEARCH_FOR | SEARCH_RLE)) != 0u)
    {
      transform = get_pre_bitpack_transform(shared_metadata[2u]);
      rle_mode = get_rle_mode(shared_metadata[2u]);
    }
    if constexpr ((SearchSpace & SEARCH_DELTA) != 0u)
    {
      secondary = shared_metadata[3u];
    }
    if constexpr ((SearchSpace & SEARCH_RLE) != 0u)
    {
      if (rle_mode != RleMode::None)
      {
        constexpr uint32_t RLE_SIZE_WORD = 3u + static_cast<uint32_t>((SearchSpace & SEARCH_DELTA) != 0u);
        bitpack_size = shared_metadata[RLE_SIZE_WORD];
      }
    }
    const uint32_t prefix = rle_mode == RleMode::None ? 0u : rle_mask_size_bytes<uint32_t>(output_size_bytes);
    bitpack::warp_bitpack_decompress(
      payload + prefix,
      output,
      shared_metadata[0u],
      transform == PreBitpackTransform::None ? shared_metadata[1u] : 0u,
      num_active_bitpack_lanes<uint32_t>(bitpack_size)
    );
    if (rle_mode == RleMode::Before)
    {
      const uint32_t num_runs = bitpack_size / sizeof(uint32_t);
      warp_transforms::warp_inverse_transform<uint32_t, SearchSpace>(output, transform, shared_metadata[1u], secondary);
      warp_expand_runs<uint32_t>(payload, output, shared_workspace, num_runs, output_size_bytes);
    }
    else
    {
      warp_transforms::warp_inverse_transform<uint32_t, SearchSpace>(output, transform, shared_metadata[1u], secondary);
    }
  }
};

} // namespace nvcomp::cascaded::next::kibi_cascader
