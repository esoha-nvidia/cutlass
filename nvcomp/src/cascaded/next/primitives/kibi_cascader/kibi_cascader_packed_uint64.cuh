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

#include <cstdint>

namespace nvcomp::cascaded::next
{

struct PackedUint64
{
  uint32_t lo;
  uint32_t hi;

  inline constexpr __host__ __device__ PackedUint64(const uint32_t lo_, const uint32_t hi_)
      : lo(lo_)
      , hi(hi_)
  {}

  inline constexpr __host__ __device__ explicit PackedUint64(const uint32_t *words)
      : lo(words[0])
      , hi(words[1])
  {}

  inline constexpr __host__ __device__ bool operator<(const PackedUint64 other) const
  {
    return hi < other.hi || (hi == other.hi && lo < other.lo);
  }

  inline constexpr __host__ __device__ PackedUint64 &operator|=(const PackedUint64 other)
  {
    lo |= other.lo;
    hi |= other.hi;
    return *this;
  }

  inline constexpr __host__ __device__ PackedUint64 &operator&=(const PackedUint64 other)
  {
    lo &= other.lo;
    hi &= other.hi;
    return *this;
  }
};

static_assert(sizeof(PackedUint64) == sizeof(uint64_t));

inline constexpr __host__ __device__ PackedUint64 pack_uint64(const uint64_t value)
{
  return {static_cast<uint32_t>(value), static_cast<uint32_t>(value >> 32u)};
}

inline constexpr __host__ __device__ uint64_t unpack_uint64(const PackedUint64 value)
{
  return static_cast<uint64_t>(value.lo) | (static_cast<uint64_t>(value.hi) << 32u);
}

template <uint32_t WORDS_PER_THREAD>
inline __device__ void
store_packed_uint64(uint32_t values[WORDS_PER_THREAD], const uint32_t index, const PackedUint64 value)
{
  values[2u * index] = value.lo;
  values[2u * index + 1u] = value.hi;
}

// Calculate the change mask represented by the bitwise OR and AND of an input vector
// \returns true if bits set in OR but not set in AND
inline constexpr __host__ __device__ uint32_t change_mask_from_or_and(const uint32_t value_or, const uint32_t value_and)
{
  return (~value_and & value_or);
}

inline constexpr __host__ __device__ PackedUint64
change_mask_from_or_and(const PackedUint64 value_or, const PackedUint64 value_and)
{
  return {change_mask_from_or_and(value_or.lo, value_and.lo), change_mask_from_or_and(value_or.hi, value_and.hi)};
}

} // namespace nvcomp::cascaded::next
