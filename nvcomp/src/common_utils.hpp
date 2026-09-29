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

#include <cstddef> // size_t
#include <cstdint> // uintptr_t

#include "nvcomp/utils.hpp" // NVCOMP_HOST_DEVICE_FUNCTION

namespace nvcomp
{

/**
 * @brief Number of bytes from `ptr` forward to the next `align`-byte boundary.
 *
 * Returns 0 when `ptr` is already aligned, so this is the length of the scalar
 * "peel" needed before a vectorized loop can use `align`-wide accesses.
 */
template <typename T>
constexpr NVCOMP_HOST_DEVICE_FUNCTION size_t bytesUntilAlignmentBoundary(const T *const ptr, const size_t align)
{
  return (align - (reinterpret_cast<uintptr_t>(ptr) % align)) % align;
}

} // namespace nvcomp
