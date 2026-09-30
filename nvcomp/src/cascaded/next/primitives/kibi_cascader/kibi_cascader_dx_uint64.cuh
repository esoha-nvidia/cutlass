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

#include "cascaded/bitpack/bitpack_split_warp_dx.cuh"
#include "cascaded/common/cascaded_warp_reductions.cuh"
#include "cascaded/next/primitives/kibi_cascader/decompositions/kibi_decompose_rle.cuh"
#include "cascaded/next/primitives/kibi_cascader/kibi_cascader_dx.cuh"
#include "cascaded/next/primitives/kibi_cascader/kibi_cascader_encoding.cuh"
#include "cascaded/next/primitives/kibi_cascader/kibi_cascader_utils.cuh"
#include "cascaded/next/primitives/kibi_cascader/transforms/kibi_transform_dispatch.cuh"
#include "CudaConstants.h"
#include "nvcomp_device_common.cuh"

/*
  Each thread loads 32B as four interleaved uint64_t values held in eight
  uint32_t registers. Integer transforms operate on those logical uint64_t
  values. The selected result is then word-planed in place so lanes [0, 15]
  bitpack the low words and lanes [16, 31] bitpack the high words.

  RLE is evaluated over complete uint64_t values before word-planing. The
  untransformed path preserves the four-word metadata format. Transform and
  RLE searches add a selector; Delta adds two secondary values.
*/

namespace nvcomp::cascaded::next::kibi_cascader
{

template <>
struct KibiCascader<uint64_t>
{
  static_assert(KC_WORDS_PER_THREAD == bitpack::BITPACK_WORDS_PER_THREAD);
  static_assert(KC_WORDS_PER_THREAD % 2u == 0u);

  static constexpr uint32_t VALUES_PER_THREAD = KC_WORDS_PER_THREAD / 2u;

  struct Candidate
  {
    PreBitpackTransform transform;
    PackedUint64 change_mask;
    PackedUint64 parameter;
    PackedUint64 secondary;
    uint32_t payload_size_bytes;
  };

  static inline __device__ PackedUint64 subtract_words(PackedUint64 value, const PackedUint64 subtract)
  {
    const uint32_t old_lo = value.lo;
    value.lo -= subtract.lo;
    value.hi -= subtract.hi + static_cast<uint32_t>(old_lo < subtract.lo);
    return value;
  }

  static inline __device__ uint32_t packed_payload_size(const PackedUint64 change_mask, const uint32_t num_active)
  {
    return (__popc(change_mask.lo) + __popc(change_mask.hi)) * num_active;
  }

  static inline __device__ void
  load_input(const uint32_t *shared_input, uint32_t values[KC_WORDS_PER_THREAD], const uint32_t input_size_bytes)
  {
    const uint32_t lane = threadIdx.x % WARP_SIZE;
    const uint32_t num_words = input_size_bytes / sizeof(uint32_t);
    const PackedUint64 padding =
      num_words == 0u
        ? PackedUint64{0u, 0u}
        : PackedUint64{shared_input[shared_input_word_index(0u)], shared_input[shared_input_word_index(1u)]};
#pragma unroll
    for (uint32_t word = 0u; word < KC_WORDS_PER_THREAD; ++word)
    {
      const uint32_t index = lane * KC_WORDS_PER_THREAD + word;
      values[word] = index < num_words ? shared_input[shared_input_word_index(index)]
                                       : ((word & 1u) == 0u ? padding.lo : padding.hi);
    }
  }

  static inline __device__ void
  load_word_planes(const uint32_t *shared_input, uint32_t values[KC_WORDS_PER_THREAD], const uint32_t input_size_bytes)
  {
    constexpr uint32_t WORDS_PER_PLANE = MAX_KC_INPUT_SIZE_BYTES / sizeof(PackedUint64);
    const uint32_t lane = threadIdx.x % WARP_SIZE;
    const uint32_t plane = lane / (WARP_SIZE / 2u);
    const uint32_t plane_lane = lane % (WARP_SIZE / 2u);
    const uint32_t num_values = input_size_bytes / sizeof(PackedUint64);
    const uint32_t first_source = uint64_word_plane_source_index(plane * WORDS_PER_PLANE);
    const uint32_t padding = num_values == 0u ? 0u : shared_input[shared_input_word_index(first_source)];
#pragma unroll
    for (uint32_t word = 0u; word < KC_WORDS_PER_THREAD; ++word)
    {
      const uint32_t plane_value = plane_lane * KC_WORDS_PER_THREAD + word;
      const uint32_t source = uint64_word_plane_source_index(plane * WORDS_PER_PLANE + plane_value);
      values[word] = plane_value < num_values ? shared_input[shared_input_word_index(source)] : padding;
    }
  }

  static inline __device__ Candidate evaluate(
    const uint32_t values[KC_WORDS_PER_THREAD],
    const PreBitpackTransform transform,
    const PackedUint64 parameter,
    const uint32_t input_size_bytes
  )
  {
    const uint32_t num_active = num_active_bitpack_lanes<uint64_t>(input_size_bytes);
    const uint32_t lane = lane_id();
    const bool active = lane < 2u * num_active;
    PackedUint64 local_or{0u, 0u};
    PackedUint64 local_and{~0u, ~0u};
#pragma unroll
    for (uint32_t value = 0u; value < VALUES_PER_THREAD; ++value)
    {
      const PackedUint64 word(values + 2u * value);
      if (active)
      {
        local_or |= word;
        local_and &= word;
      }
    }
    const PackedUint64 value_or = warp_reduce_or(local_or);
    const PackedUint64 value_and = num_active == 0u ? PackedUint64{0u, 0u} : warp_reduce_and(local_and);
    const PackedUint64 change_mask = change_mask_from_or_and(value_or, value_and);
    return {
      transform,
      change_mask,
      transform == PreBitpackTransform::None ? value_and : parameter,
      PackedUint64{0u, 0u},
      packed_payload_size(change_mask, num_active)
    };
  }

  static inline __device__ Candidate smaller_candidate(const Candidate best, const Candidate candidate)
  {
    return candidate.payload_size_bytes < best.payload_size_bytes ? candidate : best;
  }

  static inline __device__ PackedUint64
  input_minimum(const uint32_t values[KC_WORDS_PER_THREAD], const uint32_t input_size_bytes)
  {
    const uint32_t lane = lane_id();
    const uint32_t num_values = input_size_bytes / sizeof(uint64_t);
    PackedUint64 minimum{std::numeric_limits<uint32_t>::max(), std::numeric_limits<uint32_t>::max()};
#pragma unroll
    for (uint32_t value = 0u; value < VALUES_PER_THREAD; ++value)
    {
      const PackedUint64 word(values + 2u * value);
      if (lane * VALUES_PER_THREAD + value < num_values && word < minimum)
      {
        minimum = word;
      }
    }
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    {
      const PackedUint64 other = warp_shuffle_xor(minimum, static_cast<uint32_t>(offset));
      if (other < minimum)
      {
        minimum = other;
      }
    }
    if (num_values == 0u)
    {
      minimum = PackedUint64{0u, 0u};
    }
    return minimum;
  }

  static inline __device__ Candidate evaluate_frame_of_reference(
    const uint32_t values[KC_WORDS_PER_THREAD],
    const PackedUint64 reference,
    const uint32_t input_size_bytes
  )
  {
    const uint32_t lane = threadIdx.x % WARP_SIZE;
    const uint32_t num_active = num_active_bitpack_lanes<uint64_t>(input_size_bytes);
    const bool active = lane < 2u * num_active;
    PackedUint64 local_or{0u, 0u};
    PackedUint64 local_and{~0u, ~0u};
#pragma unroll
    for (uint32_t value = 0u; value < VALUES_PER_THREAD; ++value)
    {
      const PackedUint64 transformed = subtract_words(PackedUint64(values + 2u * value), reference);
      if (active)
      {
        local_or |= transformed;
        local_and &= transformed;
      }
    }
    const PackedUint64 value_or = warp_reduce_or(local_or);
    const PackedUint64 value_and = num_active == 0u ? PackedUint64{0u, 0u} : warp_reduce_and(local_and);
    const PackedUint64 change_mask = change_mask_from_or_and(value_or, value_and);
    return {
      PreBitpackTransform::FrameOfReference,
      change_mask,
      reference,
      PackedUint64{0u, 0u},
      packed_payload_size(change_mask, num_active)
    };
  }

  static inline __device__ KibiCascaderCompressionPlan
  plan_bitpack_input(uint32_t values[KC_WORDS_PER_THREAD], uint32_t *shared_metadata, const uint32_t input_size_bytes)
  {
    const uint32_t num_active = num_active_bitpack_lanes<uint64_t>(input_size_bytes);
    const uint32_t lane = lane_id();
    const auto bitpack_plan = bitpack::warp_bitpack_split_compress_init(values, num_active, num_active);
    if (lane == 0u)
    {
      shared_metadata[0u] = bitpack_plan.change_mask_0;
    }
    else if (lane == 1u)
    {
      shared_metadata[1u] = bitpack_plan.change_mask_1;
    }
    else if (lane == 2u)
    {
      shared_metadata[2u] = bitpack_plan.base_value_0;
    }
    else if (lane == 3u)
    {
      shared_metadata[3u] = bitpack_plan.base_value_1;
    }
    __syncwarp();
    return {input_size_bytes, 0u, bitpack_plan.payload_size, 0u};
  }

  template <kibi_cascader_optimization_search_space SearchSpace>
  static inline __device__ KibiCascaderCompressionPlan plan_without_rle(
    uint32_t *shared_input,
    uint32_t values[KC_WORDS_PER_THREAD],
    uint32_t *shared_metadata,
    const uint32_t input_size_bytes
  )
  {
    static_assert((SearchSpace & SEARCH_RLE) == 0u);
    static_assert(get_num_metadata_words<uint64_t, SearchSpace>() <= MAX_KC_METADATA_WORDS);
    if constexpr ((SearchSpace & (SEARCH_DELTA | SEARCH_FOR)) == 0u)
    {
      load_word_planes(shared_input, values, input_size_bytes);
      return plan_bitpack_input(values, shared_metadata, input_size_bytes);
    }

    load_input(shared_input, values, input_size_bytes);
    Candidate best = evaluate(values, PreBitpackTransform::None, PackedUint64{0u, 0u}, input_size_bytes);

    if constexpr ((SearchSpace & SEARCH_FOR) != 0u)
    {
      const PackedUint64 minimum = input_minimum(values, input_size_bytes);
      best = smaller_candidate(best, evaluate_frame_of_reference(values, minimum, input_size_bytes));
    }
    if constexpr ((SearchSpace & SEARCH_DELTA) != 0u)
    {
      const uint64_t first = warp_transforms::warp_transform_delta<uint64_t>(values);
      const PackedUint64 first_value = pack_uint64(first);
      thread_normalize_padding<uint64_t>(values, input_size_bytes, 0u);
      best = smaller_candidate(best, evaluate(values, PreBitpackTransform::Delta, first_value, input_size_bytes));

      const auto [minimum, maximum] = warp_transforms::warp_signed_delta_range<uint64_t>(values, input_size_bytes);
      const uint64_t reference = static_cast<uint64_t>(minimum);
      const PackedUint64 range = pack_uint64(static_cast<uint64_t>(maximum) - reference);
      const PackedUint64 change_mask =
        {range.hi == 0u ? mask_for_range(range.lo) : 0xFFFFFFFFu, mask_for_range(range.hi)};
      best = smaller_candidate(
        best,
        {PreBitpackTransform::DeltaFrameOfReference,
         change_mask,
         first_value,
         pack_uint64(reference),
         packed_payload_size(change_mask, num_active_bitpack_lanes<uint64_t>(input_size_bytes))}
      );

      warp_transforms::thread_transform_zigzag<uint64_t>(values);
      best = smaller_candidate(best, evaluate(values, PreBitpackTransform::DeltaZigzag, first_value, input_size_bytes));
    }

    if constexpr ((SearchSpace & SEARCH_DELTA) != 0u)
    {
      load_input(shared_input, values, input_size_bytes);
    }
    warp_transforms::warp_apply_transform<uint64_t>(
      values,
      best.transform,
      input_size_bytes,
      unpack_uint64(best.parameter),
      unpack_uint64(best.secondary)
    );
    word_plane_uint64_in_place(values);

    const uint32_t lane = threadIdx.x % WARP_SIZE;
    if (lane == 0u)
    {
      shared_metadata[0u] = best.change_mask.lo;
    }
    else if (lane == 1u)
    {
      shared_metadata[1u] = best.change_mask.hi;
    }
    else if (lane == 2u)
    {
      shared_metadata[2u] = best.parameter.lo;
    }
    else if (lane == 3u)
    {
      shared_metadata[3u] = best.parameter.hi;
    }
    else if (lane == 4u)
    {
      shared_metadata[4u] = static_cast<uint32_t>(best.transform);
    }
    if constexpr ((SearchSpace & SEARCH_DELTA) != 0u)
    {
      if (lane == 5u)
      {
        shared_metadata[5u] = best.secondary.lo;
      }
      else if (lane == 6u)
      {
        shared_metadata[6u] = best.secondary.hi;
      }
    }
    __syncwarp();
    return {input_size_bytes, 0u, best.payload_size_bytes, 0u};
  }

  template <kibi_cascader_optimization_search_space SearchSpace>
  static inline __device__ KibiCascaderCompressionPlan plan(
    uint32_t *shared_input,
    uint32_t values[KC_WORDS_PER_THREAD],
    uint32_t *shared_metadata,
    [[maybe_unused]] uint32_t *shared_workspace,
    const uint32_t input_size_bytes
  )
  {
    static_assert(get_num_metadata_words<uint64_t, SearchSpace>() <= MAX_KC_METADATA_WORDS);
    if constexpr ((SearchSpace & SEARCH_RLE) == 0u)
    {
      return plan_without_rle<SearchSpace>(shared_input, values, shared_metadata, input_size_bytes);
    }
    else
    {
      constexpr auto TRANSFORM_SEARCH = SearchSpace & ~SEARCH_RLE;
      constexpr uint32_t RLE_SIZE_WORD = 5u + 2u * static_cast<uint32_t>((TRANSFORM_SEARCH & SEARCH_DELTA) != 0u);
      const uint32_t lane = threadIdx.x % WARP_SIZE;

#pragma unroll
      for (uint32_t word = 0u; word < KC_WORDS_PER_THREAD; ++word)
      {
        const uint32_t index = lane * KC_WORDS_PER_THREAD + word;
        shared_workspace[shared_input_word_index(index)] = shared_input[shared_input_word_index(index)];
      }
      __syncwarp();

      load_input(shared_input, values, input_size_bytes);
      const WarpRlePlan raw_rle = warp_find_runs<uint64_t>(values, input_size_bytes);
      const KibiCascaderCompressionPlan direct_plan =
        plan_without_rle<TRANSFORM_SEARCH>(shared_input, values, shared_metadata, input_size_bytes);
      const PreBitpackTransform direct_transform = (TRANSFORM_SEARCH & (SEARCH_DELTA | SEARCH_FOR)) != 0u
                                                     ? static_cast<PreBitpackTransform>(shared_metadata[4u])
                                                     : PreBitpackTransform::None;
      const PackedUint64 direct_parameter(shared_metadata + 2u);
      const PackedUint64 direct_secondary = (TRANSFORM_SEARCH & SEARCH_DELTA) != 0u ? PackedUint64(shared_metadata + 5u)
                                                                                    : PackedUint64{0u, 0u};
      const uint32_t prefix_size_bytes = rle_mask_size_bytes<uint64_t>(input_size_bytes);
      uint32_t bitpack_input_size_bytes = input_size_bytes;
      uint32_t payload_size_bytes = direct_plan.payload_size_bytes;
      RleMode rle_mode = RleMode::None;
      const bool raw_fits_workspace = raw_rle.num_runs * (sizeof(uint64_t) / sizeof(uint32_t)) +
                                        prefix_size_bytes / sizeof(uint32_t) <=
                                      MAX_KC_INPUT_SIZE_BYTES / sizeof(uint32_t);
      const uint32_t raw_size_bytes = raw_rle.num_runs * sizeof(uint64_t);
      const uint32_t estimated_rle_payload_size_bytes = prefix_size_bytes +
                                                        (__popc(shared_metadata[0u]) + __popc(shared_metadata[1u])) *
                                                          num_active_bitpack_lanes<uint64_t>(raw_size_bytes);
      if (raw_rle.num_runs < input_size_bytes / sizeof(uint64_t) && raw_fits_workspace &&
          estimated_rle_payload_size_bytes < direct_plan.payload_size_bytes)
      {
        const uint32_t direct_mask_lo = shared_metadata[0u];
        const uint32_t direct_mask_hi = shared_metadata[1u];
        const uint32_t direct_base_lo = shared_metadata[2u];
        const uint32_t direct_base_hi = shared_metadata[3u];
        load_input(shared_workspace, values, input_size_bytes);
        warp_compact_runs<uint64_t>(values, raw_rle, shared_input);
        load_input(shared_input, values, raw_size_bytes);
        warp_transforms::warp_apply_transform<uint64_t>(
          values,
          direct_transform,
          raw_size_bytes,
          unpack_uint64(direct_parameter),
          unpack_uint64(direct_secondary)
        );
        uint32_t rle_payload_size_bytes = 0u;
        PackedUint64 rle_change_mask{0u, 0u};
        if (direct_transform == PreBitpackTransform::None)
        {
          word_plane_uint64_in_place(values);
          const KibiCascaderCompressionPlan rle_plan = plan_bitpack_input(values, shared_metadata, raw_size_bytes);
          rle_payload_size_bytes = prefix_size_bytes + rle_plan.payload_size_bytes;
        }
        else
        {
          const Candidate rle_candidate = evaluate(values, direct_transform, direct_parameter, raw_size_bytes);
          rle_change_mask = rle_candidate.change_mask;
          word_plane_uint64_in_place(values);
          rle_payload_size_bytes = prefix_size_bytes + rle_candidate.payload_size_bytes;
        }
        if (rle_payload_size_bytes < direct_plan.payload_size_bytes)
        {
          if (direct_transform != PreBitpackTransform::None)
          {
            if (lane == 0u)
            {
              shared_metadata[0u] = rle_change_mask.lo;
            }
            else if (lane == 1u)
            {
              shared_metadata[1u] = rle_change_mask.hi;
            }
          }
          payload_size_bytes = rle_payload_size_bytes;
          bitpack_input_size_bytes = raw_size_bytes;
          rle_mode = RleMode::Before;
        }
        else
        {
          load_input(shared_workspace, values, input_size_bytes);
          warp_transforms::warp_apply_transform<uint64_t>(
            values,
            direct_transform,
            input_size_bytes,
            unpack_uint64(direct_parameter),
            unpack_uint64(direct_secondary)
          );
          word_plane_uint64_in_place(values);
          if (lane == 0u)
          {
            shared_metadata[0u] = direct_mask_lo;
          }
          else if (lane == 1u)
          {
            shared_metadata[1u] = direct_mask_hi;
          }
          else if (lane == 2u)
          {
            shared_metadata[2u] = direct_base_lo;
          }
          else if (lane == 3u)
          {
            shared_metadata[3u] = direct_base_hi;
          }
        }
      }
      __syncwarp();

      if (lane == 4u)
      {
        shared_metadata[4u] = encoding_metadata(direct_transform, rle_mode);
      }
      if (lane == RLE_SIZE_WORD)
      {
        shared_metadata[RLE_SIZE_WORD] = bitpack_input_size_bytes;
      }
      __syncwarp();
      const uint32_t payload_prefix_word =
        rle_mode == RleMode::None ? 0u : warp_pack_run_mask_word<uint64_t>(raw_rle.local_run_mask, input_size_bytes);
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
    uint32_t values[KC_WORDS_PER_THREAD],
    const uint32_t *shared_metadata,
    uint8_t *payload,
    const KibiCascaderCompressionPlan &plan
  )
  {
    write_payload_prefix(payload, plan);
    const uint32_t num_active = num_active_bitpack_lanes<uint64_t>(plan.bitpack_input_size_bytes);
    bitpack::warp_bitpack_split_compress_apply(
      values,
      payload + plan.payload_prefix_size_bytes,
      shared_metadata[0u],
      shared_metadata[1u],
      num_active,
      num_active
    );
  }

  template <kibi_cascader_optimization_search_space SearchSpace>
  static inline __device__ void decode(
    const uint8_t *payload,
    uint32_t values[KC_WORDS_PER_THREAD],
    const uint32_t *shared_metadata,
    [[maybe_unused]] uint32_t *shared_workspace,
    const uint32_t output_size_bytes
  )
  {
    PreBitpackTransform transform = PreBitpackTransform::None;
    RleMode rle_mode = RleMode::None;
    uint32_t bitpack_size_bytes = output_size_bytes;
    if constexpr ((SearchSpace & (SEARCH_DELTA | SEARCH_FOR | SEARCH_RLE)) != 0u)
    {
      transform = get_pre_bitpack_transform(shared_metadata[4u]);
      rle_mode = get_rle_mode(shared_metadata[4u]);
    }
    if constexpr ((SearchSpace & SEARCH_RLE) != 0u)
    {
      if (rle_mode != RleMode::None)
      {
        constexpr uint32_t RLE_SIZE_WORD = 5u + 2u * static_cast<uint32_t>((SearchSpace & SEARCH_DELTA) != 0u);
        bitpack_size_bytes = shared_metadata[RLE_SIZE_WORD];
      }
    }

    const uint32_t prefix_size_bytes = rle_mode == RleMode::None ? 0u
                                                                 : rle_mask_size_bytes<uint64_t>(output_size_bytes);
    const uint32_t num_active = num_active_bitpack_lanes<uint64_t>(bitpack_size_bytes);
    bitpack::warp_bitpack_split_decompress(
      payload + prefix_size_bytes,
      values,
      shared_metadata[0u],
      shared_metadata[1u],
      transform == PreBitpackTransform::None ? shared_metadata[2u] : 0u,
      transform == PreBitpackTransform::None ? shared_metadata[3u] : 0u,
      num_active,
      num_active
    );
    unplane_uint64_in_place(values);
    PackedUint64 secondary{0u, 0u};
    if constexpr ((SearchSpace & SEARCH_DELTA) != 0u)
    {
      secondary = PackedUint64(shared_metadata + 5u);
    }
    const uint64_t parameter = unpack_uint64(PackedUint64(shared_metadata + 2u));
    const uint64_t secondary_value = unpack_uint64(secondary);
    if (rle_mode == RleMode::Before)
    {
      const uint32_t num_runs = bitpack_size_bytes / sizeof(uint64_t);
      warp_transforms::warp_inverse_transform<uint64_t, SearchSpace>(values, transform, parameter, secondary_value);
      warp_expand_runs<uint64_t>(payload, values, shared_workspace, num_runs, output_size_bytes);
    }
    else
    {
      warp_transforms::warp_inverse_transform<uint64_t, SearchSpace>(values, transform, parameter, secondary_value);
    }
  }
};

} // namespace nvcomp::cascaded::next::kibi_cascader
