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

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "cascaded/next/transforms/alp/alp_constants.cuh"
#include "cascaded/next/transforms/alp/alp_dx.cuh"
#include "cascaded/next/transforms/alp/alp_dx_impl.cuh"

/*
  float64 ALP:

  Each lane owns four doubles as eight interleaved uint32_t words, low word of a value first. The 
  exponent search and the warp-wide agreement are the shared ones in alp_dx_impl.cuh.
*/

namespace nvcomp::cascaded::next::transforms::alp
{

static_assert(sizeof(double) == sizeof(uint64_t));
static_assert(std::numeric_limits<double>::is_iec559);

template <>
struct AlpTypeOps<double>
{
  using Encoded = int64_t;
  using Inverse = DoubleDouble;

  static constexpr uint32_t MAX_EXPONENT = DOUBLE_MAX_EXPONENT;
  static constexpr double EXACT_INTEGER_LIMIT = DOUBLE_EXACT_INTEGER_LIMIT;

  static inline __device__ double load(const uint32_t *words, const uint32_t index)
  {
    return __hiloint2double(static_cast<int32_t>(words[2u * index + 1u]), static_cast<int32_t>(words[2u * index]));
  }

  static inline __device__ void store(uint32_t *words, const uint32_t index, const double value)
  {
    words[2u * index] = static_cast<uint32_t>(__double2loint(value));
    words[2u * index + 1u] = static_cast<uint32_t>(__double2hiint(value));
  }

  static inline __device__ Encoded load_encoded(const uint32_t *words, const uint32_t index)
  {
    const uint64_t bits = static_cast<uint64_t>(words[2u * index]) |
                          (static_cast<uint64_t>(words[2u * index + 1u]) << 32u);
    return static_cast<Encoded>(bits);
  }

  static inline __device__ void store_encoded(uint32_t *words, const uint32_t index, const Encoded encoded)
  {
    const uint64_t bits = static_cast<uint64_t>(encoded);
    words[2u * index] = static_cast<uint32_t>(bits);
    words[2u * index + 1u] = static_cast<uint32_t>(bits >> 32u);
  }

  static inline __device__ double exponent_at(const uint32_t exponent) { return DOUBLE_EXPONENTS[exponent]; }

  static inline __device__ Inverse inverse_at(const uint32_t exponent) { return INVERSE_EXPONENTS[exponent]; }

  static inline __device__ Encoded scale(const double value, const double exponent)
  {
    return __double2ll_rn(value * exponent);
  }

  // Scaling by a higher precision inverse power of ten carries enough of the product to round back to the
  // value the encoder started from in all cases.
  static inline __device__ double decode(const Encoded encoded, const Inverse inverse_exponent)
  {
    const double encoded_double = __ll2double_rn(encoded);
    return __fma_rn(encoded_double, inverse_exponent.high, __dmul_rn(encoded_double, inverse_exponent.low));
  }

  static inline __device__ double abs_value(const double value) { return fabs(value); }

  static inline __device__ double max_value(const double lhs, const double rhs) { return fmax(lhs, rhs); }

  // Compares bit patterns rather than values, because an == check would accept
  // an exponent whose decode drops the sign bit of a negative zero.
  static inline __device__ bool same_bits(const double lhs, const double rhs)
  {
    return __double_as_longlong(lhs) == __double_as_longlong(rhs);
  }
};

template <>
struct AlpTransform<double>
{
  template <size_t WordsPerThread>
  static inline __device__ AlpPlan encode_words_in_place(uint32_t *words, const uint32_t num_values)
  {
    return detail::encode_words_in_place_impl<double, WordsPerThread>(words, num_values);
  }

  template <size_t WordsPerThread>
  static inline __device__ void decode_words_in_place(uint32_t *words, const AlpPlan plan, const uint32_t num_values)
  {
    detail::decode_words_in_place_impl<double, WordsPerThread>(words, plan, num_values);
  }
};

} // namespace nvcomp::cascaded::next::transforms::alp
