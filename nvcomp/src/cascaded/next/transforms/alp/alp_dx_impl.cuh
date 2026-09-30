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

#include <cstddef>
#include <cstdint>

#include "cascaded/next/transforms/alp/alp_constants.cuh"
#include "cascaded/next/transforms/alp/alp_utils.cuh"
#include "nvcomp_device_common.cuh"

/*
  ALP control flow shared by every floating-point type:

  Only the warp maximum of the per-value exponents is necessary, so encoding never computes the
  individual ones. Each value is tested against the running exponent, and a failure raises it for
  the whole warp. The result is the smallest exponent valid for the whole vector, and it is applied
  only if every lane also agrees the values scale within the exact integer range for the data type.

  The types differ only in leaf operations: how a value maps to its words, which power of ten
  tables to read, and how a scaled integer converts back to a value. AlpTypeOps supplies those.
*/

namespace nvcomp::cascaded::next::transforms::alp
{

/**
 * Leaf operations that differ between the floating-point types ALP supports, specialized in
 * alp_dx_float32.cuh and alp_dx_float64.cuh. load and store hide each type's word layout, so the
 * search below indexes values rather than words.
 */
template <typename T>
struct AlpTypeOps;

namespace detail
{

template <typename T, size_t WordsPerThread>
inline constexpr size_t VALUES_PER_THREAD = WordsPerThread * sizeof(uint32_t) / sizeof(T);

/**
 * Shared body of AlpTransform<T>::encode_words_in_place.
 */
template <typename T, size_t WordsPerThread>
inline __device__ AlpPlan encode_words_in_place_impl(uint32_t *words, const uint32_t num_values)
{
  using Ops = AlpTypeOps<T>;
  static_assert((WordsPerThread * sizeof(uint32_t)) % sizeof(T) == 0u);
  constexpr size_t VALUES = VALUES_PER_THREAD<T, WordsPerThread>;

  const uint32_t lane_base = lane_id() * VALUES;
  T max_abs = static_cast<T>(0);

  // Warp uniform: only ever advanced under an __all_sync result, so its table reads are broadcasts.
  uint32_t exponent = 0u;
  T power = Ops::exponent_at(0u);
  typename Ops::Inverse inverse = Ops::inverse_at(0u);

#pragma unroll
  for (uint32_t i = 0u; i < VALUES; ++i)
  {
    const bool active = lane_base + i < num_values;
    T value = static_cast<T>(0);
    if (active)
    {
      value = Ops::load(words, i);
      max_abs = Ops::max_value(max_abs, Ops::abs_value(value));
    }

    while (exponent <= Ops::MAX_EXPONENT)
    {
      bool round_trips = true;
      if (active)
      {
        round_trips = Ops::same_bits(Ops::decode(Ops::scale(value, power), inverse), value);
      }
      if (__all_sync(WARP_ALL, round_trips) != 0)
      {
        break;
      }
      ++exponent;
      power = Ops::exponent_at(exponent);
      inverse = Ops::inverse_at(exponent);
    }
  }

  const bool lane_fits = exponent <= Ops::MAX_EXPONENT &&
                         max_abs * Ops::exponent_at(exponent) <= Ops::EXACT_INTEGER_LIMIT;
  const bool applied = __all_sync(WARP_ALL, lane_fits) != 0;
  if (!applied)
  {
    return AlpPlan{false, 0u};
  }

  const T selected_exponent = Ops::exponent_at(exponent);
#pragma unroll
  for (uint32_t i = 0u; i < VALUES; ++i)
  {
    if (lane_base + i < num_values)
    {
      const T value = Ops::load(words, i);
      Ops::store_encoded(words, i, Ops::scale(value, selected_exponent));
    }
  }
  return AlpPlan{true, exponent};
}

/**
 * Shared body of AlpTransform<T>::decode_words_in_place.
 */
template <typename T, size_t WordsPerThread>
inline __device__ void decode_words_in_place_impl(uint32_t *words, const AlpPlan plan, const uint32_t num_values)
{
  using Ops = AlpTypeOps<T>;
  static_assert((WordsPerThread * sizeof(uint32_t)) % sizeof(T) == 0u);
  constexpr size_t VALUES = VALUES_PER_THREAD<T, WordsPerThread>;

  if (!plan.applied)
  {
    return;
  }

  const uint32_t lane_base = lane_id() * VALUES;
  const typename Ops::Inverse inverse_exponent = Ops::inverse_at(plan.exponent);
#pragma unroll
  for (uint32_t i = 0u; i < VALUES; ++i)
  {
    if (lane_base + i < num_values)
    {
      const typename Ops::Encoded encoded = Ops::load_encoded(words, i);
      Ops::store(words, i, Ops::decode(encoded, inverse_exponent));
    }
  }
}

} // namespace detail

} // namespace nvcomp::cascaded::next::transforms::alp
