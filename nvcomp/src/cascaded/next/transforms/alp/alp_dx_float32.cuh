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
  float32 ALP:

  One value per word, so a lane's eight words are eight independent floats and no interleaving is
  needed. The exponent search and the warp-wide agreement are the shared ones in alp_dx_impl.cuh,
  running over the shorter FLOAT_MAX_EXPONENT range. Decoding scales in double then narrows to float.
*/

namespace nvcomp::cascaded::next::transforms::alp
{

static_assert(sizeof(float) == sizeof(uint32_t));
static_assert(std::numeric_limits<float>::is_iec559);

template <>
struct AlpTypeOps<float>
{
  using Encoded = int32_t;
  using Inverse = double;

  static constexpr uint32_t MAX_EXPONENT = FLOAT_MAX_EXPONENT;
  static constexpr float EXACT_INTEGER_LIMIT = FLOAT_EXACT_INTEGER_LIMIT;

  static inline __device__ float load(const uint32_t *words, const uint32_t index)
  {
    return __uint_as_float(words[index]);
  }

  static inline __device__ void store(uint32_t *words, const uint32_t index, const float value)
  {
    words[index] = __float_as_uint(value);
  }

  static inline __device__ Encoded load_encoded(const uint32_t *words, const uint32_t index)
  {
    return static_cast<Encoded>(words[index]);
  }

  static inline __device__ void store_encoded(uint32_t *words, const uint32_t index, const Encoded encoded)
  {
    words[index] = static_cast<uint32_t>(encoded);
  }

  static inline __device__ float exponent_at(const uint32_t exponent) { return FLOAT_EXPONENTS[exponent]; }

  // Only the high limb is needed. The product is formed in double and then narrowed, so the low
  // limb lands well below the bits float keeps.
  static inline __device__ Inverse inverse_at(const uint32_t exponent) { return INVERSE_EXPONENTS[exponent].high; }

  static inline __device__ Encoded scale(const float value, const float exponent)
  {
    return __float2int_rn(value * exponent);
  }

  static inline __device__ float decode(const Encoded encoded, const Inverse inverse_exponent)
  {
    return __double2float_rn(__dmul_rn(__int2double_rn(encoded), inverse_exponent));
  }

  static inline __device__ float abs_value(const float value) { return fabsf(value); }

  static inline __device__ float max_value(const float lhs, const float rhs) { return fmaxf(lhs, rhs); }

  // Compares bit patterns rather than values, because an == check would accept
  // an exponent whose decode drops the sign bit of a negative zero.
  static inline __device__ bool same_bits(const float lhs, const float rhs)
  {
    return __float_as_uint(lhs) == __float_as_uint(rhs);
  }
};

template <>
struct AlpTransform<float>
{
  template <size_t WordsPerThread>
  static inline __device__ AlpPlan encode_words_in_place(uint32_t *words, const uint32_t num_values)
  {
    return detail::encode_words_in_place_impl<float, WordsPerThread>(words, num_values);
  }

  template <size_t WordsPerThread>
  static inline __device__ void decode_words_in_place(uint32_t *words, const AlpPlan plan, const uint32_t num_values)
  {
    detail::decode_words_in_place_impl<float, WordsPerThread>(words, plan, num_values);
  }
};

} // namespace nvcomp::cascaded::next::transforms::alp
