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
#include <type_traits>

namespace nvcomp::cascaded::next::transforms
{

// https://en.wikipedia.org/wiki/Signed_number_representations
// Zig-zag encoding is used to make signed integers more bitpack-friendly.

template <typename T>
inline __host__ __device__ T transform_zigzag(const T value)
{
  static_assert(std::is_unsigned_v<T>);
  using SignedT = std::make_signed_t<T>;
  constexpr uint32_t SIGN_SHIFT = sizeof(T) * 8u - 1u;
  return (value << 1u) ^ static_cast<T>(static_cast<SignedT>(value) >> SIGN_SHIFT);
}

template <typename T>
inline __host__ __device__ T inverse_transform_zigzag(const T value)
{
  static_assert(std::is_unsigned_v<T>);
  return (value >> 1u) ^ (T{0} - (value & T{1}));
}

} // namespace nvcomp::cascaded::next::transforms
