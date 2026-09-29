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

#include <cuda/std/bit>
#include <cuda/std/cmath>
#include <cuda_fp16.h>

#include <cassert>
#include <type_traits>

#include "nvcomp/native/bitcomp.h"
#include "utilities.h"

namespace bitcomp
{

// Always-false helper — must depend on T to defer evaluation
template <typename T>
inline constexpr bool always_false = false;

// Unsupported-type sentinel using the helper
template <typename T>
struct UnsupportedSize
{
  static_assert(always_false<T>, "InputSelector: unsupported type size (must be 2, 4, or 8 bytes)");
};

template <typename T>
using InputSelector = std::conditional_t<
  sizeof(T) == 2,
  half2,
  std::conditional_t<
    sizeof(T) == 4,
    float,
    std::conditional_t<
      sizeof(T) == 8,
      double,
      UnsupportedSize<T> // fallback: unsupported size
      >>>;

template <typename T, bitcompMode_t mode>
using OutputSelector = std::conditional_t<
  mode == BITCOMP_LOSSY_FP_TO_SIGNED,
  std::conditional_t<
    sizeof(T) == 2,
    short2,
    std::conditional_t<
      sizeof(T) == 4,
      int32_t,
      std::conditional_t<
        sizeof(T) == 8,
        int64_t,
        UnsupportedSize<T> // fallback: unsupported size
        >>>,
  std::conditional_t<
    sizeof(T) == 2,
    ushort2,
    std::conditional_t<
      sizeof(T) == 4,
      uint32_t,
      std::conditional_t<
        sizeof(T) == 8,
        uint64_t,
        UnsupportedSize<T> // fallback: unsupported size
        >>>>;

template <typename T>
inline __host__ __device__ bool inf_check(const T orig, const T reconst)
{
  return cuda::std::isinf(reconst) != cuda::std::isinf(orig);
}

// If val is inf or NaN, we consider it an overflow.
template <typename T, typename OutputIntegerType>
inline __host__ __device__ bool is_overflow(const T val)
{
  if constexpr (cuda::std::is_unsigned_v<OutputIntegerType>)
  {
    return !(val >= 0.0 && val < static_cast<T>(std::numeric_limits<OutputIntegerType>::max()));
  }
  else
  {
    return !(cuda::std::fabs(val) < static_cast<T>(std::numeric_limits<OutputIntegerType>::max()));
  }
}

template <typename T, std::enable_if_t<std::is_same_v<T, ushort2> || std::is_same_v<T, short2>, bool> = true>
inline __host__ __device__ half2 dequantize(const T input, const half delta)
{
  float tmplo = static_cast<float>(input.x);
  float tmphi = static_cast<float>(input.y);
  return make_half2(tmplo * __half2float(delta), tmphi * __half2float(delta));
}

template <typename T>
using FloatT = typename std::conditional<sizeof(T) == 4, float, double>::type;

template <
  typename T,
  bool PowTwoDelta,
  std::enable_if_t<
    std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t> || std::is_same_v<T, int32_t> ||
      std::is_same_v<T, int64_t>,
    bool> = true>
inline __host__ __device__ FloatT<T> dequantize(const T input, const FloatT<T> delta)
{
  FloatT<T> tmp = static_cast<FloatT<T>>(input);

  if constexpr (PowTwoDelta)
  {
    assert(utilities::zeroMantissaBits(delta) == delta && "Delta must have zeroed mantissa bits");
    assert(
      static_cast<FloatT<T>>(input - static_cast<T>(tmp)) == FloatT<T>(0.0) &&
      "Since delta has zeroed mantissa bits, the dequantized value should be exact with no remainder"
    );
    return tmp * delta;
  }
  else
  {
    FloatT<T> rem = static_cast<FloatT<T>>(input - static_cast<T>(tmp));
    return tmp * delta + rem * delta;
  }
}

template <class T, std::enable_if_t<std::is_same_v<T, ushort2> || std::is_same_v<T, short2>, bool> = true>
inline __host__ __device__ T quantize(const half2 input, const half delta, int &overflow)
{
  using OutputType = std::conditional_t<std::is_same_v<T, short2>, int16_t, uint16_t>;
  assert(utilities::zeroMantissaBits(delta) == delta && "Delta must have zeroed mantissa bits");
  const float invdelta = 1.0f / __half2float(delta);
  float tmplo = cuda::std::rint(__half2float(input.x) * invdelta);
  float tmphi = cuda::std::rint(__half2float(input.y) * invdelta);
  T output{static_cast<OutputType>(tmplo), static_cast<OutputType>(tmphi)};
  half2 reconstructed = dequantize(output, delta);
  // Need to perform this check since due to the rounding and low max FP16_MAX (65504), quantizing a large number could
  // result in an inf.
  if (is_overflow<float, OutputType>(tmplo) || is_overflow<float, OutputType>(tmphi) ||
      inf_check(input.x, reconstructed.x) || inf_check(input.y, reconstructed.y))
  {
    overflow = 1;
  }

  return output;
}

// We can use a simpler quantization method since we are using a power-of-two
// delta, which gives exact multiplication and division.
template <
  typename T,
  std::enable_if_t<
    std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t> || std::is_same_v<T, int32_t> ||
      std::is_same_v<T, int64_t>,
    bool> = true>
inline __host__ __device__ T quantize(const FloatT<T> input, const FloatT<T> delta, int &overflow)
{
  assert(utilities::zeroMantissaBits(delta) == delta && "Delta must have zeroed mantissa bits");
  const FloatT<T> invdelta = FloatT<T>(1.0) / delta;
  FloatT<T> quant = cuda::std::rint(input * invdelta);

  // Since the previous operation are all exact, "tmp2" from the previous quantization method is always 0.
  assert(
    (cuda::std::rint((input - quant * delta) * invdelta) == FloatT<T>(0.0) || is_overflow<FloatT<T>, T>(quant)) &&
    "Rounding error in quantization, check that delta is a power of two "
    "and that input values are not too large"
  );
  if (is_overflow<FloatT<T>, T>(quant))
  {
    overflow = 1;
  }
  return static_cast<T>(quant);
}
} // namespace bitcomp