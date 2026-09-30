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
#include <cstring>

#include "cascaded/bitpack/bitpack_constants.cuh"

namespace nvcomp::cascaded::bitpack::detail
{

// Writes one contiguous lane group's bitpacked payload.
inline __device__ void group_bitpack_compress_apply(
  const uint32_t *my_words,
  uint8_t *output,
  const uint32_t change_mask,
  const uint32_t group_lane,
  const uint32_t num_active
)
{
  const bool active = group_lane < num_active;

  if (change_mask == 0xFFFFFFFFu)
  {
    if (active)
    {
#pragma unroll
      for (uint32_t i = 0u; i < BITPACK_WORDS_PER_THREAD; ++i)
      {
        memcpy(output + (i * num_active + group_lane) * sizeof(uint32_t), &my_words[i], sizeof(uint32_t));
      }
    }
    return;
  }

  // Keep inactive lanes in the uniform payload-control flow and predicate only
  // their writes. Hoisting this condition around the payload loop causes a
  // significant throughput regression for partial lane groups.
  uint32_t output_offset = 0u;

  // Unroll the four fixed byte positions; the set-bit loop remains runtime-dependent.
#pragma unroll
  for (uint32_t byte = 0u; byte < static_cast<uint32_t>(sizeof(uint32_t)); ++byte)
  {
    const uint32_t byte_shift = byte * 8u;
    const uint32_t mask_byte = (change_mask >> byte_shift) & 0xFFu;

    if (mask_byte == 0xFFu)
    {
#pragma unroll
      for (uint32_t i = 0u; i < BITPACK_WORDS_PER_THREAD; ++i)
      {
        if (active)
        {
          output[output_offset + group_lane] = static_cast<uint8_t>(my_words[i] >> byte_shift);
        }
        output_offset += num_active;
      }
    }
    else
    {
      uint32_t remaining_mask = mask_byte;
      while (remaining_mask != 0u)
      {
        const uint32_t bit_in_byte = static_cast<uint32_t>(__ffs(remaining_mask) - 1);
        remaining_mask &= remaining_mask - 1u;
        const uint32_t bit_position = byte_shift + bit_in_byte;

        if (active)
        {
          uint8_t packed = 0u;
#pragma unroll
          for (uint32_t i = 0u; i < BITPACK_WORDS_PER_THREAD; ++i)
          {
            packed |= ((my_words[i] >> bit_position) & 1u) << i;
          }
          output[output_offset + group_lane] = packed;
        }
        output_offset += num_active;
      }
    }
  }
}

/** Reconstructs one contiguous lane group's words from a bitpacked payload. */
inline __device__ void group_bitpack_decompress(
  const uint8_t *input,
  uint32_t *my_words,
  const uint32_t change_mask,
  const uint32_t base_value,
  const uint32_t group_lane,
  const uint32_t num_active
)
{
  const bool active = group_lane < num_active;

  if (change_mask == 0xFFFFFFFFu)
  {
#pragma unroll
    for (uint32_t i = 0u; i < BITPACK_WORDS_PER_THREAD; ++i)
    {
      if (active)
      {
        memcpy(&my_words[i], input + (i * num_active + group_lane) * sizeof(uint32_t), sizeof(uint32_t));
      }
      else
      {
        my_words[i] = base_value;
      }
    }
    return;
  }

  // Keep inactive lanes in the uniform payload-control flow and predicate only
  // their reads. Hoisting this condition around the payload loop causes a
  // significant throughput regression for partial lane groups.
#pragma unroll
  for (uint32_t i = 0u; i < BITPACK_WORDS_PER_THREAD; ++i)
  {
    my_words[i] = base_value;
  }

  uint32_t input_offset = 0u;

  // Unroll the four fixed byte positions; the set-bit loop remains runtime-dependent.
#pragma unroll
  for (uint32_t byte = 0u; byte < static_cast<uint32_t>(sizeof(uint32_t)); ++byte)
  {
    const uint32_t byte_shift = byte * 8u;
    const uint32_t mask_byte = (change_mask >> byte_shift) & 0xFFu;

    if (mask_byte == 0xFFu)
    {
#pragma unroll
      for (uint32_t i = 0u; i < BITPACK_WORDS_PER_THREAD; ++i)
      {
        if (active)
        {
          const uint8_t as_is = input[input_offset + group_lane];
          my_words[i] |= static_cast<uint32_t>(as_is) << byte_shift;
        }
        input_offset += num_active;
      }
    }
    else
    {
      uint32_t remaining_mask = mask_byte;
      while (remaining_mask != 0u)
      {
        const uint32_t bit_in_byte = static_cast<uint32_t>(__ffs(remaining_mask) - 1);
        remaining_mask &= remaining_mask - 1u;
        const uint32_t bit_position = byte_shift + bit_in_byte;

        if (active)
        {
          const uint8_t packed = input[input_offset + group_lane];
#pragma unroll
          for (uint32_t i = 0u; i < BITPACK_WORDS_PER_THREAD; ++i)
          {
            my_words[i] |= static_cast<uint32_t>((packed >> i) & 1u) << bit_position;
          }
        }
        input_offset += num_active;
      }
    }
  }
}

} // namespace nvcomp::cascaded::bitpack::detail
