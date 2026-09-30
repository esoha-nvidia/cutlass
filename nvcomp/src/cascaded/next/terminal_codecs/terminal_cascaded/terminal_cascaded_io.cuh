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

#include "cascaded/next/primitives/kibi_cascader/kibi_cascader_encoding.cuh"
#include "cascaded/next/primitives/kibi_cascader/kibi_cascader_utils.cuh"
#include "cascaded/next/terminal_codecs/terminal_cascaded/terminal_cascaded_utils.cuh"
#include "CudaConstants.h"

namespace nvcomp::cascaded::next::terminal_cascaded
{

template <kibi_cascader::kibi_cascader_optimization_search_space SearchSpace>
inline __device__ uint32_t get_encoding_selector(const uint32_t *const shared_metadata)
{
  [[maybe_unused]] constexpr uint32_t IX_UNPACKED_METADATA_SELECTOR = 2u;
  constexpr bool HAS_SELECTOR =
    (SearchSpace & (kibi_cascader::SEARCH_DELTA | kibi_cascader::SEARCH_FOR | kibi_cascader::SEARCH_RLE)) != 0u;
  if constexpr (HAS_SELECTOR)
  {
    return shared_metadata[IX_UNPACKED_METADATA_SELECTOR];
  }
  else
  {
    return kibi_cascader::encoding_metadata(kibi_cascader::PreBitpackTransform::None, kibi_cascader::RleMode::None);
  }
}

template <kibi_cascader::kibi_cascader_optimization_search_space SearchSpace>
inline __device__ uint32_t serialized_metadata_size_bytes(const uint32_t selector)
{
  // Every transform stores a selector, bitpack mask, and primary value.
  // Delta-FOR adds a secondary value, and RLE adds the compact stream size.
  uint32_t size = U32_PACKED_METADATA_BYTES;
  switch (kibi_cascader::get_pre_bitpack_transform(selector))
  {
    case kibi_cascader::PreBitpackTransform::None:
    case kibi_cascader::PreBitpackTransform::FrameOfReference:
    case kibi_cascader::PreBitpackTransform::Delta:
    case kibi_cascader::PreBitpackTransform::DeltaZigzag:
      break;
    case kibi_cascader::PreBitpackTransform::DeltaFrameOfReference:
      size = U32_PACKED_DELTA_FOR_METADATA_BYTES;
      break;
  }
  if constexpr ((SearchSpace & kibi_cascader::SEARCH_RLE) != 0u)
  {
    if (kibi_cascader::get_rle_mode(selector) != kibi_cascader::RleMode::None)
    {
      size += sizeof(uint32_t);
    }
  }
  return size;
}

template <kibi_cascader::kibi_cascader_optimization_search_space SearchSpace>
inline __device__ uint32_t get_serialized_metadata_size_bytes(const uint32_t *const shared_metadata)
{
  return serialized_metadata_size_bytes<SearchSpace>(get_encoding_selector<SearchSpace>(shared_metadata));
}

template <kibi_cascader::kibi_cascader_optimization_search_space SearchSpace>
inline __device__ void serialize_u32_kc_metadata(const uint32_t *const shared_metadata, uint8_t *const output)
{
  // Word indices into unpacked shared metadata.
  constexpr uint32_t IX_UNPACKED_METADATA_BITPACK_MASK = 0u;
  constexpr uint32_t IX_UNPACKED_METADATA_PRIMARY = 1u;
  [[maybe_unused]] constexpr uint32_t IX_UNPACKED_METADATA_SELECTOR = 2u;
  constexpr uint32_t IX_UNPACKED_METADATA_DELTA_SECONDARY = 3u;
  [[maybe_unused]] constexpr uint32_t IX_UNPACKED_METADATA_RLE_VALUES_SIZE =
    IX_UNPACKED_METADATA_DELTA_SECONDARY + static_cast<uint32_t>((SearchSpace & kibi_cascader::SEARCH_DELTA) != 0u);
  // Byte offsets into packed metadata.
  constexpr uint32_t IX_PACKED_METADATA_SELECTOR = 0u;
  constexpr uint32_t IX_PACKED_METADATA_BITPACK_MASK = IX_PACKED_METADATA_SELECTOR + sizeof(uint8_t);
  constexpr uint32_t IX_PACKED_METADATA_PRIMARY = IX_PACKED_METADATA_BITPACK_MASK + sizeof(uint32_t);
  constexpr uint32_t IX_PACKED_METADATA_OPTIONAL = IX_PACKED_METADATA_PRIMARY + sizeof(uint32_t);

  if (lane_id() == 0u)
  {
    const uint32_t selector = get_encoding_selector<SearchSpace>(shared_metadata);
    output[IX_PACKED_METADATA_SELECTOR] = static_cast<uint8_t>(selector);
    store_unaligned<uint32_t>(
      output + IX_PACKED_METADATA_BITPACK_MASK,
      shared_metadata[IX_UNPACKED_METADATA_BITPACK_MASK]
    );
    store_unaligned<uint32_t>(output + IX_PACKED_METADATA_PRIMARY, shared_metadata[IX_UNPACKED_METADATA_PRIMARY]);

    const bool delta_for = kibi_cascader::get_pre_bitpack_transform(selector) ==
                           kibi_cascader::PreBitpackTransform::DeltaFrameOfReference;
    if (delta_for)
    {
      store_unaligned<uint32_t>(
        output + IX_PACKED_METADATA_OPTIONAL,
        shared_metadata[IX_UNPACKED_METADATA_DELTA_SECONDARY]
      );
    }

    if constexpr ((SearchSpace & kibi_cascader::SEARCH_RLE) != 0u)
    {
      if (kibi_cascader::get_rle_mode(selector) != kibi_cascader::RleMode::None)
      {
        const uint32_t ix_packed_metadata_rle_values_size = IX_PACKED_METADATA_OPTIONAL +
                                                            (delta_for ? sizeof(uint32_t) : 0u);
        store_unaligned<uint32_t>(
          output + ix_packed_metadata_rle_values_size,
          shared_metadata[IX_UNPACKED_METADATA_RLE_VALUES_SIZE]
        );
      }
    }
  }
}

template <typename T, kibi_cascader::kibi_cascader_optimization_search_space SearchSpace>
inline __device__ void
deserialize_kc_metadata(const uint8_t *const input, uint32_t *const shared_metadata, const uint32_t metadata_size_bytes)
{
  // lane_id() seems to increase register pressure here. TODO: investigate
  const uint32_t lane = threadIdx.x % WARP_SIZE;

  if constexpr (std::is_same_v<T, uint32_t>)
  {
    // Word indices into unpacked shared metadata.
    constexpr uint32_t IX_UNPACKED_METADATA_BITPACK_MASK = 0u;
    constexpr uint32_t IX_UNPACKED_METADATA_PRIMARY = 1u;
    [[maybe_unused]] constexpr uint32_t IX_UNPACKED_METADATA_SELECTOR = 2u;
    constexpr uint32_t IX_UNPACKED_METADATA_DELTA_SECONDARY = 3u;
    constexpr bool SEARCHES_TRANSFORMS = (SearchSpace & (kibi_cascader::SEARCH_DELTA | kibi_cascader::SEARCH_FOR)) !=
                                         0u;
    constexpr bool SEARCHES_RLE = (SearchSpace & kibi_cascader::SEARCH_RLE) != 0u;
    constexpr bool SEARCHES_DELTA = (SearchSpace & kibi_cascader::SEARCH_DELTA) != 0u;
    [[maybe_unused]] constexpr uint32_t IX_UNPACKED_METADATA_RLE_VALUES_SIZE = IX_UNPACKED_METADATA_DELTA_SECONDARY +
                                                                               static_cast<uint32_t>(SEARCHES_DELTA);
    // Byte offsets into packed metadata.
    constexpr uint32_t IX_PACKED_METADATA_SELECTOR = 0u;
    constexpr uint32_t IX_PACKED_METADATA_BITPACK_MASK = IX_PACKED_METADATA_SELECTOR + sizeof(uint8_t);
    constexpr uint32_t IX_PACKED_METADATA_PRIMARY = IX_PACKED_METADATA_BITPACK_MASK + sizeof(uint32_t);
    [[maybe_unused]] constexpr uint32_t IX_PACKED_METADATA_OPTIONAL = IX_PACKED_METADATA_PRIMARY + sizeof(uint32_t);

    // Load serialized metadata cooperatively, four bytes per lane. Since its size
    // may not be word-aligned, the final lane assembles only the remaining bytes.
    // The following shuffles realign the one-byte-prefixed serialized layout into
    // uint32_t shared metadata words.
    uint32_t lane_word = 0u;
    const uint32_t lane_offset = lane * sizeof(uint32_t);
    if (lane_offset + sizeof(uint32_t) <= metadata_size_bytes)
    {
      lane_word = load_unaligned<uint32_t>(input + lane_offset);
    }
    else if (lane_offset < metadata_size_bytes)
    {
      const uint32_t nbytes = metadata_size_bytes - lane_offset;
      for (uint32_t byte = 0u; byte < nbytes; ++byte)
      {
        lane_word |= static_cast<uint32_t>(input[lane_offset + byte]) << (8u * byte);
      }
    }

    const uint32_t word0 = __shfl_sync(0xFFFFFFFFu, lane_word, 0u);
    const uint32_t word1 = __shfl_sync(0xFFFFFFFFu, lane_word, 1u);
    const uint32_t word2 = __shfl_sync(0xFFFFFFFFu, lane_word, 2u);
    const uint32_t word3 = __shfl_sync(0xFFFFFFFFu, lane_word, 3u);
    if (lane == 0u)
    {
      shared_metadata[IX_UNPACKED_METADATA_BITPACK_MASK] = (word0 >> 8u) | (word1 << 24u);
      shared_metadata[IX_UNPACKED_METADATA_PRIMARY] = (word1 >> 8u) | (word2 << 24u);
      if constexpr (SEARCHES_TRANSFORMS || SEARCHES_RLE)
      {
        shared_metadata[IX_UNPACKED_METADATA_SELECTOR] = word0 & 0xFFu;
      }
      if constexpr (SEARCHES_DELTA)
      {
        shared_metadata[IX_UNPACKED_METADATA_DELTA_SECONDARY] = (word2 >> 8u) | (word3 << 24u);
      }
    }

    if constexpr (SEARCHES_RLE)
    {
      // The RLE values size must be located from the selector, not from
      // metadata_size_bytes: a nine-byte encoding plus the four-byte run size
      // is the same length as DeltaFrameOfReference without RLE.
      const uint32_t selector = word0 & 0xFFu;
      if (kibi_cascader::get_rle_mode(selector) != kibi_cascader::RleMode::None && lane == 0u)
      {
        const bool delta_for = kibi_cascader::get_pre_bitpack_transform(selector) ==
                               kibi_cascader::PreBitpackTransform::DeltaFrameOfReference;
        const uint32_t ix_packed_metadata_rle_values_size = IX_PACKED_METADATA_OPTIONAL +
                                                            (delta_for ? sizeof(uint32_t) : 0u);
        shared_metadata[IX_UNPACKED_METADATA_RLE_VALUES_SIZE] =
          load_unaligned<uint32_t>(input + ix_packed_metadata_rle_values_size);
      }
    }
  }
  else
  {
    // Word indices into unpacked shared metadata.
    constexpr uint32_t IX_UNPACKED_METADATA_BITPACK_MASK_LOW = 0u;
    [[maybe_unused]] constexpr uint32_t IX_UNPACKED_METADATA_BITPACK_MASK_HIGH = 1u;
    [[maybe_unused]] constexpr uint32_t IX_UNPACKED_METADATA_PRIMARY_LOW = 2u;
    [[maybe_unused]] constexpr uint32_t IX_UNPACKED_METADATA_PRIMARY_HIGH = 3u;
    constexpr uint32_t IX_UNPACKED_METADATA_SELECTOR = 4u;
    constexpr uint32_t IX_UNPACKED_METADATA_DELTA_SECONDARY_LOW = 5u;
    [[maybe_unused]] constexpr uint32_t IX_UNPACKED_METADATA_DELTA_SECONDARY_HIGH = 6u;
    constexpr uint32_t METADATA_WORDS_PER_VALUE = sizeof(uint64_t) / sizeof(uint32_t);
    constexpr bool SEARCHES_RLE = (SearchSpace & kibi_cascader::SEARCH_RLE) != 0u;
    constexpr uint32_t IX_UNPACKED_METADATA_RLE_VALUES_SIZE =
      IX_UNPACKED_METADATA_DELTA_SECONDARY_LOW +
      METADATA_WORDS_PER_VALUE * static_cast<uint32_t>((SearchSpace & kibi_cascader::SEARCH_DELTA) != 0u);
    // Byte offsets into packed metadata.
    constexpr uint32_t IX_PACKED_METADATA_SELECTOR = 0u;
    constexpr uint32_t IX_PACKED_METADATA_BITPACK_MASK = IX_PACKED_METADATA_SELECTOR + sizeof(uint8_t);
    constexpr uint32_t IX_PACKED_METADATA_PRIMARY = IX_PACKED_METADATA_BITPACK_MASK + sizeof(uint64_t);
    constexpr uint32_t IX_PACKED_METADATA_OPTIONAL = IX_PACKED_METADATA_PRIMARY + sizeof(uint64_t);

    const uint32_t encoding_meta = input[IX_PACKED_METADATA_SELECTOR];
    const bool delta_for = kibi_cascader::get_pre_bitpack_transform(encoding_meta) ==
                           kibi_cascader::PreBitpackTransform::DeltaFrameOfReference;
    if (lane < 4u)
    {
      shared_metadata[IX_UNPACKED_METADATA_BITPACK_MASK_LOW + lane] =
        load_unaligned<uint32_t>(input + IX_PACKED_METADATA_BITPACK_MASK + lane * sizeof(uint32_t));
    }
    if (delta_for && lane < 2u)
    {
      shared_metadata[IX_UNPACKED_METADATA_DELTA_SECONDARY_LOW + lane] =
        load_unaligned<uint32_t>(input + IX_PACKED_METADATA_OPTIONAL + lane * sizeof(uint32_t));
    }
    if (lane == 0u)
    {
      shared_metadata[IX_UNPACKED_METADATA_SELECTOR] = encoding_meta;
      if constexpr (SEARCHES_RLE)
      {
        if (kibi_cascader::get_rle_mode(encoding_meta) != kibi_cascader::RleMode::None)
        {
          const uint32_t ix_packed_metadata_rle_values_size = IX_PACKED_METADATA_OPTIONAL +
                                                              (delta_for ? sizeof(uint64_t) : 0u);
          shared_metadata[IX_UNPACKED_METADATA_RLE_VALUES_SIZE] =
            load_unaligned<uint32_t>(input + ix_packed_metadata_rle_values_size);
        }
      }
    }
  }
  __syncwarp();
}

inline __device__ void load_kc(
  const uint8_t *const in,
  const uint32_t uncompressed_bytes,
  const uint32_t kc_idx,
  const uint32_t lane,
  const uint32_t num_active,
  uint32_t my_words[WORDS_PER_THREAD]
)
{
  if (lane >= num_active)
  {
#pragma unroll
    for (uint32_t i = 0u; i < WORDS_PER_THREAD; ++i)
    {
      my_words[i] = ~0u;
    }
    return;
  }
  const uint32_t kc_base_byte = kc_idx * KC_SIZE_BYTES;
  // size_t promotion seems to increase register pressure here. TODO: investigate
  constexpr uint32_t WORD_BYTES = static_cast<uint32_t>(sizeof(uint32_t));
  uint32_t ix_last_valid_input_word = 0u;
  bool has_valid_input = false;
#pragma unroll
  for (uint32_t i = 0u; i < WORDS_PER_THREAD; ++i)
  {
    // uint32_t arithmetic seems to reduce register pressure here. TODO: investigate
    const uint32_t byte0 = kc_base_byte + (lane * WORDS_PER_THREAD + static_cast<uint32_t>(i)) * WORD_BYTES;
    if (byte0 + WORD_BYTES <= uncompressed_bytes)
    {
      my_words[i] = load_unaligned<uint32_t>(in + byte0);
      ix_last_valid_input_word = i;
      has_valid_input = true;
    }
    else if (byte0 < uncompressed_bytes)
    {
      const uint32_t nbytes = uncompressed_bytes - byte0;
      uint32_t v = 0u;
      uint8_t msb = 0u;
      for (uint32_t j = 0u; j < nbytes; ++j)
      {
        const uint8_t bj = in[byte0 + j];
        v |= static_cast<uint32_t>(bj) << (8u * j);
        msb = bj;
      }
      // A uint32_t bound seems to reduce register pressure here. TODO: investigate
      for (uint32_t j = nbytes; j < WORD_BYTES; ++j)
      {
        v |= static_cast<uint32_t>(msb) << (8u * j);
      }
      my_words[i] = v;
      ix_last_valid_input_word = i;
      has_valid_input = true;
    }
    else
    {
      my_words[i] = has_valid_input ? my_words[ix_last_valid_input_word] : ~0u;
    }
  }
}

inline __device__ void store_kc(
  uint8_t *const out,
  const uint32_t uncompressed_bytes,
  const uint32_t kc_idx,
  const uint32_t lane,
  const uint32_t num_active,
  const uint32_t my_words[WORDS_PER_THREAD]
)
{
  if (lane >= num_active)
  {
    return;
  }
  const uint32_t kc_base_byte = kc_idx * KC_SIZE_BYTES;
  // size_t promotion seems to increase register pressure here. TODO: investigate
  constexpr uint32_t WORD_BYTES = static_cast<uint32_t>(sizeof(uint32_t));
#pragma unroll
  for (uint32_t i = 0u; i < WORDS_PER_THREAD; ++i)
  {
    // uint32_t arithmetic seems to reduce register pressure here. TODO: investigate
    const uint32_t byte0 = kc_base_byte + (lane * WORDS_PER_THREAD + static_cast<uint32_t>(i)) * WORD_BYTES;
    if (byte0 + WORD_BYTES <= uncompressed_bytes)
    {
      reinterpret_cast<uint32_t *>(out + byte0)[0u] = my_words[i];
    }
    else if (byte0 < uncompressed_bytes)
    {
      const uint32_t nbytes = uncompressed_bytes - byte0;
      for (uint32_t j = 0u; j < nbytes; ++j)
      {
        out[byte0 + j] = static_cast<uint8_t>(my_words[i] >> (8u * j));
      }
    }
  }
}

inline __device__ uint32_t swizzle_word(const uint32_t word) { return kibi_cascader::shared_input_word_index(word); }

// Match pipeline-0's load_full: stage a bank-conflict-free shared copy and leave
// the lane-major words in registers so plan() does not need a second reload.
inline __device__ void load_full_kc_coalesced(
  const uint8_t *const in,
  const uint32_t kc_idx,
  const uint32_t lane,
  uint32_t lane_words[WORDS_PER_THREAD],
  uint32_t scratch_words[WORDS_PER_KC]
)
{
  const uint32_t *const kc_input = reinterpret_cast<const uint32_t *>(in) + kc_idx * WORDS_PER_KC;
#pragma unroll
  for (uint32_t group = 0u; group < WORDS_PER_THREAD; ++group)
  {
    const uint32_t logical_word = group * WARP_SIZE + lane;
    scratch_words[swizzle_word(logical_word)] = kc_input[logical_word];
  }
  __syncwarp();
#pragma unroll
  for (uint32_t word = 0u; word < WORDS_PER_THREAD; ++word)
  {
    const uint32_t logical_word = lane * WORDS_PER_THREAD + word;
    lane_words[word] = scratch_words[swizzle_word(logical_word)];
  }
}

inline __device__ uint32_t uint64_pair_num_values(const uint32_t uncompressed_bytes, const uint32_t first_kc)
{
  const uint32_t pair_start = first_kc * KC_SIZE_BYTES;
  const uint32_t remaining = uncompressed_bytes - pair_start;
  return min(2u * KC_SIZE_BYTES, remaining) / sizeof(uint64_t);
}

inline __device__ void stage_uint64_pair_input(
  const uint8_t *const input,
  const uint32_t uncompressed_bytes,
  const uint32_t kc,
  const uint32_t lane,
  uint32_t pair_scratch[2u * WORDS_PER_KC]
)
{
  const uint32_t first_kc = kc & ~1u;
  const uint32_t source_half = kc & 1u;
  const uint32_t num_values = uint64_pair_num_values(uncompressed_bytes, first_kc);
  const uint32_t source_value_base = source_half * UINT64_VALUES_PER_KC;
  const uint32_t source_values =
    min(UINT64_VALUES_PER_KC, num_values > source_value_base ? num_values - source_value_base : 0u);
  const uint64_t *const source = reinterpret_cast<const uint64_t *>(input + kc * KC_SIZE_BYTES);

#pragma unroll
  for (uint32_t group = 0u; group < UINT64_VALUES_PER_KC / WARP_SIZE; ++group)
  {
    const uint32_t source_value = group * WARP_SIZE + lane;
    if (source_value < source_values)
    {
      const uint64_t value = source[source_value];
      const uint32_t plane_value = source_value_base + source_value;
      const uint32_t plane_word = swizzle_word(plane_value);
      pair_scratch[plane_word] = static_cast<uint32_t>(value);
      pair_scratch[WORDS_PER_KC + plane_word] = static_cast<uint32_t>(value >> 32u);
    }
  }
}

inline __device__ void load_uint64_pair_plane(
  const uint32_t pair_scratch[2u * WORDS_PER_KC],
  const uint32_t num_values,
  const uint32_t plane,
  const uint32_t lane,
  uint32_t words[WORDS_PER_THREAD]
)
{
  const uint32_t num_active = num_active_for(num_values);
  if (lane >= num_active)
  {
#pragma unroll
    for (uint32_t word = 0u; word < WORDS_PER_THREAD; ++word)
    {
      words[word] = ~0u;
    }
    return;
  }

  const uint32_t plane_base = plane * WORDS_PER_KC;
#pragma unroll
  for (uint32_t word = 0u; word < WORDS_PER_THREAD; ++word)
  {
    const uint32_t plane_value = lane * WORDS_PER_THREAD + word;
    const uint32_t source_value = min(plane_value, num_values - 1u);
    words[word] = pair_scratch[plane_base + swizzle_word(source_value)];
  }
}

inline __device__ void load_uint64_unpaired_contiguous(
  const uint8_t *const input,
  const uint32_t uncompressed_bytes,
  const uint32_t kc,
  const uint32_t lane,
  uint32_t scratch[WORDS_PER_KC],
  uint32_t words[WORDS_PER_THREAD]
)
{
  const uint32_t num_values = kc_size_bytes(uncompressed_bytes, kc) / sizeof(uint64_t);
  const uint64_t *const source = reinterpret_cast<const uint64_t *>(input + kc * KC_SIZE_BYTES);
#pragma unroll
  for (uint32_t group = 0u; group < UINT64_VALUES_PER_KC / WARP_SIZE; ++group)
  {
    const uint32_t value_index = group * WARP_SIZE + lane;
    if (value_index < num_values)
    {
      const uint64_t value = source[value_index];
      scratch[swizzle_word(value_index)] = static_cast<uint32_t>(value);
      scratch[swizzle_word(num_values + value_index)] = static_cast<uint32_t>(value >> 32u);
    }
  }
  __syncwarp();

  const uint32_t num_words = 2u * num_values;
#pragma unroll
  for (uint32_t word = 0u; word < WORDS_PER_THREAD; ++word)
  {
    const uint32_t logical_word = lane * WORDS_PER_THREAD + word;
    words[word] = scratch[swizzle_word(min(logical_word, num_words - 1u))];
  }
}

inline __device__ void stage_uint64_pair_plane(
  uint32_t pair_scratch[2u * OUTPUT_TRANSPOSE_WORDS_PER_WARP],
  const uint32_t num_values,
  const uint32_t plane,
  const uint32_t lane,
  const uint32_t words[WORDS_PER_THREAD]
)
{
  const uint32_t plane_base = plane * WORDS_PER_KC;
#pragma unroll
  for (uint32_t word = 0u; word < WORDS_PER_THREAD; ++word)
  {
    const uint32_t plane_value = lane * WORDS_PER_THREAD + word;
    if (plane_value < num_values)
    {
      pair_scratch[plane_base + swizzle_word(plane_value)] = words[word];
    }
  }
}

inline __device__ void load_uint64_kc_from_contiguous(
  const uint32_t contiguous[WORDS_PER_THREAD],
  const uint32_t num_values,
  const uint32_t lane,
  uint32_t scratch[OUTPUT_TRANSPOSE_WORDS_PER_WARP],
  uint32_t words[WORDS_PER_THREAD]
)
{
#pragma unroll
  for (uint32_t word = 0u; word < WORDS_PER_THREAD; ++word)
  {
    const uint32_t logical_word = lane * WORDS_PER_THREAD + word;
    if (logical_word < 2u * num_values)
    {
      scratch[swizzle_word(logical_word)] = contiguous[word];
    }
  }
  __syncwarp();

#pragma unroll
  for (uint32_t value = 0u; value < WORDS_PER_THREAD / 2u; ++value)
  {
    const uint32_t value_index = lane * (WORDS_PER_THREAD / 2u) + value;
    if (value_index < num_values)
    {
      words[2u * value] = scratch[swizzle_word(value_index)];
      words[2u * value + 1u] = scratch[swizzle_word(num_values + value_index)];
    }
    else
    {
      words[2u * value] = 0u;
      words[2u * value + 1u] = 0u;
    }
  }
}

inline __device__ void load_uint64_kc_from_pair(
  const uint32_t pair_scratch[2u * OUTPUT_TRANSPOSE_WORDS_PER_WARP],
  const uint32_t num_values,
  const uint32_t source_half,
  const uint32_t lane,
  uint32_t words[WORDS_PER_THREAD]
)
{
  const uint32_t source_value_base = source_half * UINT64_VALUES_PER_KC;
#pragma unroll
  for (uint32_t value = 0u; value < WORDS_PER_THREAD / 2u; ++value)
  {
    const uint32_t pair_value = source_value_base + lane * (WORDS_PER_THREAD / 2u) + value;
    if (pair_value < num_values)
    {
      const uint32_t plane_word = swizzle_word(pair_value);
      words[2u * value] = pair_scratch[plane_word];
      words[2u * value + 1u] = pair_scratch[WORDS_PER_KC + plane_word];
    }
    else
    {
      words[2u * value] = 0u;
      words[2u * value + 1u] = 0u;
    }
  }
}

inline __device__ void
copy_compressed_sub_unit(uint8_t *const global_output, const uint8_t *const shared_input, const uint32_t num_bytes)
{
  constexpr uint32_t ALIGNMENT = alignof(uint4);
  const uint32_t thread = threadIdx.x;
  const uint32_t misalignment = reinterpret_cast<uintptr_t>(global_output) & (ALIGNMENT - 1u);
  const uint32_t head = min(num_bytes, (ALIGNMENT - misalignment) & (ALIGNMENT - 1u));

  if (thread < head)
  {
    global_output[thread] = shared_input[thread];
  }

  const uint32_t middle_bytes = num_bytes - head;
  const uint32_t vectors = middle_bytes / ALIGNMENT;
  for (uint32_t vector = thread; vector < vectors; vector += THREADS_PER_CTA)
  {
    const uint8_t *const source = shared_input + head + vector * ALIGNMENT;
    uint4 value;
    value.x = load_unaligned<uint32_t>(source);
    value.y = load_unaligned<uint32_t>(source + sizeof(uint32_t));
    value.z = load_unaligned<uint32_t>(source + 2u * sizeof(uint32_t));
    value.w = load_unaligned<uint32_t>(source + 3u * sizeof(uint32_t));
    reinterpret_cast<uint4 *>(global_output + head)[vector] = value;
  }

  const uint32_t tail_start = head + vectors * ALIGNMENT;
  const uint32_t tail = num_bytes - tail_start;
  if (thread < tail)
  {
    global_output[tail_start + thread] = shared_input[tail_start + thread];
  }
}

inline __device__ void store_full_kc_coalesced(
  uint8_t *const out,
  const uint32_t kc_idx,
  const uint32_t lane,
  const uint32_t lane_words[WORDS_PER_THREAD],
  uint32_t transpose_words[OUTPUT_TRANSPOSE_WORDS_PER_WARP]
)
{
  // lane_words are lane-major: lane L owns output words [L * 8, L * 8 + 8).
  // Stage the 32x8 tile through padded shared memory, then read it in eight
  // 32-word stripes. The padding makes each lane's write start in a different
  // bank, avoiding the 8-way conflicts of a tightly packed lane-major tile.
#pragma unroll
  for (uint32_t word = 0u; word < WORDS_PER_THREAD; ++word)
  {
    transpose_words[lane * OUTPUT_TRANSPOSE_STRIDE + word] = lane_words[word];
  }
  __syncwarp();

  uint32_t *const kc_output = reinterpret_cast<uint32_t *>(out) + kc_idx * WORDS_PER_KC;
  const uint32_t source_lane_in_group = lane / WORDS_PER_THREAD;
  const uint32_t source_word = lane & (WORDS_PER_THREAD - 1u);
#pragma unroll
  for (uint32_t group = 0u; group < WORDS_PER_THREAD; ++group)
  {
    const uint32_t source_lane = group * (WARP_SIZE / WORDS_PER_THREAD) + source_lane_in_group;
    kc_output[group * WARP_SIZE + lane] = transpose_words[source_lane * OUTPUT_TRANSPOSE_STRIDE + source_word];
  }
}

} // namespace nvcomp::cascaded::next::terminal_cascaded
