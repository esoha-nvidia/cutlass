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

#include <type_traits>

namespace nvcomp::cascaded::next::transforms
{

// delta treating values as unsigned integers

template <typename T>
inline __host__ __device__ T transform_delta(const T value, const T previous)
{
  static_assert(std::is_unsigned_v<T>);
  return value - previous;
}

template <typename T>
inline __host__ __device__ T inverse_transform_delta(const T delta, const T previous)
{
  static_assert(std::is_unsigned_v<T>);
  return previous + delta;
}

} // namespace nvcomp::cascaded::next::transforms
