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
#include <limits>

namespace nvcomp::cascaded::next::transforms::alp
{

// The derivations below assume IEEE-754 binary32 and binary64.
static_assert(std::numeric_limits<double>::is_iec559 && std::numeric_limits<double>::digits == 53);
static_assert(std::numeric_limits<float>::is_iec559 && std::numeric_limits<float>::digits == 24);

// Scaling an integer by a power of ten is only guaranteed to round trip if the
// result stays below the limit where the type represents every integer
// exactly. Checking this is necessary during ALP transformation.
inline constexpr double DOUBLE_EXACT_INTEGER_LIMIT = static_cast<double>(1ull << std::numeric_limits<double>::digits);
inline constexpr float FLOAT_EXACT_INTEGER_LIMIT = static_cast<float>(1ull << std::numeric_limits<float>::digits);

// A negative power of ten split across two doubles: high is the nearest
// double to 10^-e, low the nearest double to the remainder 10^-e - high. Using
// this to convert from integer to double prevents rounding error in ALP decode.
struct DoubleDouble
{
  double high;
  double low;
};

// A constexpr function cannot fill a raw array, so the generated tables are
// wrapped in a type that still reads as TABLE[e] at the call sites.
template <typename T, uint32_t N>
struct Table
{
  T values[N];

  constexpr __host__ __device__ T operator[](const uint32_t index) const { return values[index]; }
};

namespace detail
{

// 10^e = 2^e * 5^e and the power of two costs no significand bits, so 10^e is
// exact while 5^e stays below 2^digits. Gives 22 for double and 10 for float.
template <typename T>
constexpr uint32_t max_exact_power_of_ten()
{
  uint32_t e = 0u;
  for (uint64_t five_pow = 1u; five_pow * 5u < (1ull << std::numeric_limits<T>::digits); five_pow *= 5u)
  {
    ++e;
  }
  return e;
}

// Dekker's product used to recover the part that the significand drops. Every
// operation is a single IEEE one on values inside exponent range, so each one is
// exact. These are only ever constant evaluated to keep FMA contraction from
// fusing the subtractions the algorithm depends on.
struct TwoProduct
{
  double product; // fl(a * b)
  double error; // a * b - product
};

constexpr double split_high(const double a)
{
  const double t = 0x1p27 * a + a; // (2^27 + 1) * a, exact in the first product
  return t - (t - a);
}

constexpr TwoProduct two_product(const double a, const double b)
{
  const double product = a * b;
  const double a_high = split_high(a);
  const double a_low = a - a_high;
  const double b_high = split_high(b);
  const double b_low = b - b_high;
  return {product, ((a_high * b_high - product) + a_high * b_low + a_low * b_high) + a_low * b_low};
}

// Powers of ten ascending. Each product is exact because both operands and the
// result stay representable.
template <typename T, uint32_t N>
constexpr Table<T, N> make_power_table()
{
  Table<T, N> table{};
  T power_of_ten = 1;
  for (uint32_t e = 0u; e + 1u < N; ++e)
  {
    table.values[e] = power_of_ten;
    power_of_ten *= 10;
  }
  return table;
}

// high is fl(1/10^e), which is the nearest double to 10^-e because division is
// correctly rounded. low is the residual 10^-e - high:
// writing high * 10^e as product + error, that residual is
// (1 - product - error) / 10^e, and 1 - product is itself exact.
constexpr DoubleDouble inverse_power_of_ten(const double power_of_ten)
{
  const double high = 1.0 / power_of_ten;
  const TwoProduct scaled = two_product(high, power_of_ten);
  const double residual = (1.0 - scaled.product) - scaled.error;
  return {high, residual / power_of_ten};
}

template <uint32_t N>
constexpr Table<DoubleDouble, N> make_inverse_table(const Table<double, N> &powers)
{
  Table<DoubleDouble, N> table{};
  for (uint32_t e = 0u; e + 1u < N; ++e)
  {
    table.values[e] = inverse_power_of_ten(powers[e]);
  }
  return table;
}

// Both limbs together should invert their power of ten to about 2^-106.
template <uint32_t N>
constexpr bool inverses_are_sane(const Table<DoubleDouble, N> &inverses)
{
  double power_of_ten = 1.0;
  for (uint32_t e = 0u; e + 1u < N; ++e)
  {
    const TwoProduct scaled = two_product(inverses[e].high, power_of_ten);
    const double residue = (scaled.product - 1.0) + (scaled.error + inverses[e].low * power_of_ten);
    if (!(residue > -0x1p-100 && residue < 0x1p-100))
    {
      return false;
    }
    power_of_ten *= 10.0;
  }
  return true;
}

} // namespace detail

// Largest exponent whose power of ten the type represents exactly.
template <typename T>
inline constexpr uint32_t MAX_EXACT_POWER_OF_TEN = detail::max_exact_power_of_ten<T>();

// One slot per exponent 0..MAX_EXACT_POWER_OF_TEN, plus the trailing slot that
// keeps the encoder's read of index candidate + 1 in bounds.
template <typename T>
inline constexpr uint32_t TABLE_SIZE = MAX_EXACT_POWER_OF_TEN<T> + 2u;

// The largest power of ten ALP transformation searches. ALP's reference
// implementation stops at 18 for double:
// https://github.com/cwida/ALP/blob/main/include/alp/constants.hpp
// 15 is the largest exponent found to be useful empirically. Double can be
// raised without extending its table, float is already at the bound.
inline constexpr uint32_t DOUBLE_MAX_EXPONENT = 15u;
inline constexpr uint32_t FLOAT_MAX_EXPONENT = 10u;

static_assert(DOUBLE_MAX_EXPONENT + 1u < TABLE_SIZE<double>);
static_assert(FLOAT_MAX_EXPONENT + 1u < TABLE_SIZE<float>);

// The float path uses only the high limb.
constexpr __device__ auto INVERSE_EXPONENTS =
  detail::make_inverse_table(detail::make_power_table<double, TABLE_SIZE<double>>());

constexpr __device__ auto DOUBLE_EXPONENTS = detail::make_power_table<double, TABLE_SIZE<double>>();
constexpr __device__ auto FLOAT_EXPONENTS = detail::make_power_table<float, TABLE_SIZE<float>>();

static_assert(MAX_EXACT_POWER_OF_TEN<double> == 22u && MAX_EXACT_POWER_OF_TEN<float> == 10u);
static_assert(DOUBLE_EXPONENTS[MAX_EXACT_POWER_OF_TEN<double>] == 1e22);
static_assert(FLOAT_EXPONENTS[MAX_EXACT_POWER_OF_TEN<float>] == 1e10f);
static_assert(INVERSE_EXPONENTS[0].high == 1.0 && INVERSE_EXPONENTS[0].low == 0.0);
static_assert(INVERSE_EXPONENTS[1].high == 0x1.999999999999ap-4);
static_assert(INVERSE_EXPONENTS[1].low == -0x1.999999999999ap-58);
static_assert(detail::inverses_are_sane(INVERSE_EXPONENTS));

} // namespace nvcomp::cascaded::next::transforms::alp
