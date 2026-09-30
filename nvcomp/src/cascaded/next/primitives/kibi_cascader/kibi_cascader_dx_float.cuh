/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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

#include <cuda_runtime.h>

#include <cstdint>

#include "cascaded/next/primitives/kibi_cascader/kibi_cascader_dx.cuh"
#include "cascaded/next/primitives/kibi_cascader/kibi_cascader_dx_uint32.cuh"
#include "cascaded/next/primitives/kibi_cascader/kibi_cascader_dx_uint64.cuh"
#include "cascaded/next/primitives/kibi_cascader/kibi_cascader_utils.cuh"
#include "cascaded/next/transforms/alp/alp_dx.cuh"
#include "cascaded/next/transforms/alp/alp_dx_float32.cuh"
#include "cascaded/next/transforms/alp/alp_dx_float64.cuh"
#include "nvcomp_device_common.cuh"

/*
  Floating-point Kibi Cascader:

  float and double vectors are compressed as the integers the ALP-inspired transform in alp_dx.cuh
  produces. Everything after the transform is bitpacked by the same-width unsigned integer
  cascader's plain (non-searching) plan_bitpack_input path -- delta/frame-of-reference/RLE are not
  searched here, so the integer half always contributes a fixed NUM_PLANES * 2 metadata words
  (change_mask and base_value per 32-bit plane) regardless of SearchSpace. The transform adds a
  single metadata word immediately after those.

  ALP works on values in their original layout, so the 8B path plans the exponent before word
  planing rather than gathering straight into word-planed positions the way the uint64_t cascader
  does. Word planing is the only step here that depends on the type.

  A search space without SEARCH_ALP generates no ALP metadata word, so the metadata word count
  depends on SEARCH_ALP as well as the type -- but not on the other search bits, since they have no
  effect on this bitpack-only path.
*/

namespace nvcomp::cascaded::next::kibi_cascader
{

template <typename T, typename IntegerT>
struct FloatingPointKibiCascader
{
  template <kibi_cascader_optimization_search_space SearchSpace>
  static inline __device__ KibiCascaderCompressionPlan plan(
    const uint32_t *shared_input_buffer,
    uint32_t bitpack_input[KC_WORDS_PER_THREAD],
    uint32_t *shared_metadata,
    [[maybe_unused]] uint32_t *shared_workspace,
    const uint32_t input_size_bytes
  )
  {
    static_assert(sizeof(T) == sizeof(IntegerT));
    static_assert(get_num_metadata_words<T, SearchSpace>() <= MAX_KC_METADATA_WORDS);
    const uint32_t lane = lane_id();

#pragma unroll
    for (uint32_t word = 0u; word < KC_WORDS_PER_THREAD; ++word)
    {
      const uint32_t logical_word = lane * KC_WORDS_PER_THREAD + word;
      bitpack_input[word] = shared_input_buffer[shared_input_word_index(logical_word)];
    }
    if constexpr ((SearchSpace & SEARCH_ALP) != 0u)
    {
      // plan_bitpack_input below never searches delta/frame-of-reference/RLE, so the integer half
      // always writes exactly get_num_metadata_words<IntegerT, SEARCH_NONE>() words: ALP's word
      // goes right after those, regardless of which search bits the caller requested.
      constexpr uint32_t ALP_METADATA_WORD = get_num_metadata_words<IntegerT, SEARCH_NONE>();
      const transforms::alp::AlpPlan alp_plan =
        transforms::alp::encode_words_in_place<T, KC_WORDS_PER_THREAD>(bitpack_input, input_size_bytes / sizeof(T));
      if (lane == ALP_METADATA_WORD)
      {
        shared_metadata[ALP_METADATA_WORD] = transforms::alp::serialize_plan(alp_plan);
      }
      __syncwarp();
    }
    if constexpr (sizeof(T) == sizeof(uint64_t))
    {
      word_plane_uint64_in_place(bitpack_input);
    }

    return KibiCascader<IntegerT>::plan_bitpack_input(bitpack_input, shared_metadata, input_size_bytes);
  }

  template <kibi_cascader_optimization_search_space SearchSpace>
  static inline __device__ void encode(
    uint32_t values[KC_WORDS_PER_THREAD],
    const uint32_t *metadata,
    uint8_t *payload,
    const KibiCascaderCompressionPlan &plan
  )
  {
    // plan() above always used plan_bitpack_input, so encode must pair with the same SEARCH_NONE
    // metadata layout regardless of SearchSpace.
    KibiCascader<IntegerT>::template encode<SEARCH_NONE>(values, metadata, payload, plan);
  }

  template <kibi_cascader_optimization_search_space SearchSpace>
  static inline __device__ void decode(
    const uint8_t *payload,
    uint32_t output[KC_WORDS_PER_THREAD],
    const uint32_t *metadata,
    uint32_t *workspace,
    const uint32_t output_size_bytes
  )
  {
    // Must mirror plan()/encode() above: the integer half was always planned and encoded with
    // SEARCH_NONE, so decode has to read that same fixed metadata layout, not SearchSpace's.
    KibiCascader<IntegerT>::template decode<SEARCH_NONE>(payload, output, metadata, workspace, output_size_bytes);
    if constexpr ((SearchSpace & SEARCH_ALP) != 0u)
    {
      constexpr uint32_t ALP_METADATA_WORD = get_num_metadata_words<IntegerT, SEARCH_NONE>();
      transforms::alp::decode_words_in_place<T, KC_WORDS_PER_THREAD>(
        output,
        transforms::alp::deserialize_plan(metadata[ALP_METADATA_WORD]),
        output_size_bytes / sizeof(T)
      );
    }
  }
};

template <>
struct KibiCascader<float> : FloatingPointKibiCascader<float, uint32_t>
{};

template <>
struct KibiCascader<double> : FloatingPointKibiCascader<double, uint64_t>
{};

} // namespace nvcomp::cascaded::next::kibi_cascader
