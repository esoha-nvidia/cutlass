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

#include <cuda/std/utility>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "cascaded/next/primitives/kibi_cascader/decompositions/kibi_decompose_rle.cuh"
#include "cascaded/next/primitives/kibi_cascader/kibi_cascader_encoding.cuh"
#include "cascaded/next/primitives/kibi_cascader/kibi_cascader_utils.cuh"
#include "cascaded/next/terminal_codecs/terminal_cascaded/terminal_cascaded_io.cuh"
#include "cascaded/next/terminal_codecs/terminal_cascaded/terminal_cascaded_utils.cuh"
#include "CudaConstants.h"

namespace nvcomp::cascaded::next::terminal_cascaded
{

inline __device__ cuda::std::pair<bool, uint32_t>
checked_add_within(const uint32_t offset, const uint32_t size, const size_t limit)
{
  // Use subtraction so the bounds check cannot overflow before comparison.
  if (static_cast<size_t>(offset) > limit || static_cast<size_t>(size) > limit - offset)
  {
    return {false, 0u};
  }
  return {true, offset + size};
}

template <kibi_cascader::kibi_cascader_optimization_search_space SearchSpace>
inline __device__ bool selector_is_valid(const uint32_t selector)
{
  const auto transform = kibi_cascader::get_pre_bitpack_transform(selector);
  const auto rle_mode = kibi_cascader::get_rle_mode(selector);

  const bool transform_is_valid = transform == kibi_cascader::PreBitpackTransform::None ||
                                  (((SearchSpace & kibi_cascader::SEARCH_FOR) != 0u) &&
                                   transform == kibi_cascader::PreBitpackTransform::FrameOfReference) ||
                                  (((SearchSpace & kibi_cascader::SEARCH_DELTA) != 0u) &&
                                   (transform == kibi_cascader::PreBitpackTransform::Delta ||
                                    transform == kibi_cascader::PreBitpackTransform::DeltaZigzag ||
                                    transform == kibi_cascader::PreBitpackTransform::DeltaFrameOfReference));
  const bool rle_is_valid = rle_mode == kibi_cascader::RleMode::None ||
                            (((SearchSpace & kibi_cascader::SEARCH_RLE) != 0u) &&
                             rle_mode == kibi_cascader::RleMode::Before);
  return transform_is_valid && rle_is_valid;
}

// A kc_buffer is the compressed metadata and payload produced by Kibi Cascaded.
// Terminal Cascaded prefixes it with a one-byte metadata length. The minimum
// metadata is a one-byte encoding selector, a four-byte bitpack change mask,
// and a four-byte primary encoding parameter. Optional Delta-FOR and RLE
// metadata follows. Validate all regions and bounds before decompression.
template <kibi_cascader::kibi_cascader_optimization_search_space SearchSpace>
inline __device__ bool thread_validate_kc_buffer(
  const uint8_t *const kc_buffer,
  const uint32_t kc_buffer_size_bytes,
  const uint32_t bitpack_output_size_bytes
)
{
  if (kc_buffer_size_bytes < KC_BUFFER_PREFIX_BYTES + U32_PACKED_METADATA_BYTES)
  {
    return false;
  }

  const uint32_t metadata_size_bytes = kc_buffer[0u];
  if (metadata_size_bytes > kc_buffer_size_bytes - KC_BUFFER_PREFIX_BYTES ||
      metadata_size_bytes < U32_PACKED_METADATA_BYTES)
  {
    return false;
  }

  const uint8_t *const metadata = kc_buffer + KC_BUFFER_PREFIX_BYTES;
  const uint32_t selector = metadata[0u];
  if (!selector_is_valid<SearchSpace>(selector) ||
      metadata_size_bytes != serialized_metadata_size_bytes<SearchSpace>(selector))
  {
    return false;
  }

  uint32_t compact_size_bytes = bitpack_output_size_bytes;
  uint32_t payload_prefix_bytes = 0u;
  if (kibi_cascader::get_rle_mode(selector) != kibi_cascader::RleMode::None)
  {
    const uint32_t rle_size_offset = kibi_cascader::get_pre_bitpack_transform(selector) ==
                                         kibi_cascader::PreBitpackTransform::DeltaFrameOfReference
                                       ? U32_PACKED_DELTA_FOR_METADATA_BYTES
                                       : U32_PACKED_METADATA_BYTES;
    compact_size_bytes = load_unaligned<uint32_t>(metadata + rle_size_offset);
    if (compact_size_bytes == 0u || compact_size_bytes % sizeof(uint32_t) != 0u ||
        compact_size_bytes > bitpack_output_size_bytes)
    {
      return false;
    }
    const uint32_t num_runs = compact_size_bytes / sizeof(uint32_t);
    const uint32_t num_values = bitpack_output_size_bytes / sizeof(uint32_t);
    if (num_runs == 0u || num_runs > num_values)
    {
      return false;
    }
    payload_prefix_bytes = kibi_cascader::rle_mask_size_bytes<uint32_t>(bitpack_output_size_bytes);
  }

  const uint32_t change_mask = load_unaligned<uint32_t>(metadata + 1u);
  const uint32_t active_lanes = kibi_cascader::num_active_bitpack_lanes<uint32_t>(compact_size_bytes);
  const uint32_t bitpack_payload_bytes = static_cast<uint32_t>(__popc(change_mask)) * active_lanes;
  return checked_add_within(
           payload_prefix_bytes,
           bitpack_payload_bytes,
           kc_buffer_size_bytes - KC_BUFFER_PREFIX_BYTES - metadata_size_bytes
  )
    .first;
}

template <typename T, kibi_cascader::kibi_cascader_optimization_search_space SearchSpace>
inline __device__ bool cta_validate_compressed_chunk(
  const uint8_t *const compressed,
  const size_t compressed_bytes,
  const uint32_t uncompressed_bytes,
  uint32_t &validation_failed
)
{
  static_assert(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>);
  const uint32_t num_kc_batches = num_kc_batches_for(uncompressed_bytes);
  const uint32_t num_sub_units = num_sub_units_for(num_kc_batches);
  const uint32_t preamble_bytes = preamble_size(num_sub_units);

  if (threadIdx.x == 0u)
  {
    validation_failed = 0u;
    uint32_t sub_unit_start = preamble_bytes;
    for (uint32_t sub_unit = 0u; sub_unit < num_sub_units && validation_failed == 0u; ++sub_unit)
    {
      const uint32_t num_kc_buffers = min(WARPS_PER_CTA, num_kc_batches - sub_unit * WARPS_PER_CTA);
      const uint32_t sub_unit_size =
        load_unaligned<uint16_t>(compressed + universal_header::HEADER_SIZE_BYTES + sub_unit * sizeof(uint16_t));
      const auto [sub_unit_fits, sub_unit_end] = checked_add_within(sub_unit_start, sub_unit_size, compressed_bytes);
      if (sub_unit_size < num_kc_buffers * KC_BUFFER_OFFSET_BYTES || !sub_unit_fits)
      {
        validation_failed = 1u;
      }
      sub_unit_start = sub_unit_end;
    }
    if (validation_failed == 0u && static_cast<size_t>(sub_unit_start) != compressed_bytes)
    {
      validation_failed = 1u;
    }
  }
  __syncthreads();
  if (validation_failed != 0u)
  {
    return false;
  }

  const uint32_t warp_id = threadIdx.x / WARP_SIZE;
  const uint32_t lane = lane_id();
  constexpr auto KC_BUFFER_SEARCH_SPACE = SearchSpace;
  uint32_t sub_unit_start = preamble_bytes;
  for (uint32_t sub_unit = 0u; sub_unit < num_sub_units; ++sub_unit)
  {
    const uint32_t first_kc_batch = sub_unit * WARPS_PER_CTA;
    const uint32_t num_kc_buffers = min(WARPS_PER_CTA, num_kc_batches - first_kc_batch);
    const uint32_t sub_unit_size =
      load_unaligned<uint16_t>(compressed + universal_header::HEADER_SIZE_BYTES + sub_unit * sizeof(uint16_t));
    if (lane == 0u && warp_id < num_kc_buffers)
    {
      const uint32_t kc_buffer_offset =
        load_unaligned<uint16_t>(compressed + sub_unit_start + warp_id * KC_BUFFER_OFFSET_BYTES);
      const uint32_t kc_buffer_end =
        warp_id + 1u < num_kc_buffers
          ? load_unaligned<uint16_t>(compressed + sub_unit_start + (warp_id + 1u) * KC_BUFFER_OFFSET_BYTES)
          : sub_unit_size;
      const bool valid_offsets = kc_buffer_offset >= num_kc_buffers * KC_BUFFER_OFFSET_BYTES &&
                                 kc_buffer_offset < kc_buffer_end && kc_buffer_end <= sub_unit_size;
      bool valid_kc_buffer = valid_offsets;
      if (valid_kc_buffer)
      {
        const uint32_t ix_batch = first_kc_batch + warp_id;
        uint32_t bitpack_output_size_bytes = kc_size_bytes(uncompressed_bytes, ix_batch);
        if constexpr (std::is_same_v<T, uint64_t>)
        {
          if ((ix_batch | 1u) < num_kc_batches)
          {
            const uint32_t pair_start = (ix_batch & ~1u) * KC_SIZE_BYTES;
            const uint32_t pair_bytes = min(2u * KC_SIZE_BYTES, uncompressed_bytes - pair_start);
            bitpack_output_size_bytes = pair_bytes / sizeof(uint64_t) * sizeof(uint32_t);
          }
        }
        valid_kc_buffer = thread_validate_kc_buffer<KC_BUFFER_SEARCH_SPACE>(
          compressed + sub_unit_start + kc_buffer_offset,
          kc_buffer_end - kc_buffer_offset,
          bitpack_output_size_bytes
        );
      }
      if (!valid_kc_buffer)
      {
        atomicExch(&validation_failed, 1u);
      }
    }
    sub_unit_start += sub_unit_size;
  }
  __syncthreads();
  return validation_failed == 0u;
}

} // namespace nvcomp::cascaded::next::terminal_cascaded
