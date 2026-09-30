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

#include <algorithm>
#include <cassert>
#include <cstddef> // size_t
#include <cstdint> // uintptr_t
#include <iterator>
#include <string>

#include "nvcomp.hpp"
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

template <typename T, size_t N>
inline bool reserved_bytes_all_zero(const T (&reserved)[N])
{
  static_assert(sizeof(T) == 1);
  return std::all_of(std::begin(reserved), std::end(reserved), [](const T byte) { return byte == 0; });
}

/**
 * @brief Validate that a user-supplied buffer pointer satisfies the required alignment.
 *
 * @param ptr The buffer pointer to validate.
 * @param alignment The required alignment in bytes. Alignment must be larger than 0.
 * @param what Human-readable description of the buffer for the error message.
 *
 * @throws NVCompException(nvcompErrorAlignment) if @p ptr is not a multiple of @p alignment.
 */
inline void check_buffer_alignment(const void *ptr, size_t alignment, const char *what)
{
  assert(alignment > 0);
  if (reinterpret_cast<uintptr_t>(ptr) % alignment != 0)
  {
    throw NVCompException(
      nvcompErrorAlignment,
      std::string(what) + " buffer is not aligned to the required alignment of " + std::to_string(alignment) + " bytes."
    );
  }
}

} // namespace nvcomp
