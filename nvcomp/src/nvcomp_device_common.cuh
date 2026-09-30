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

#include <cassert>
#include <cstdint>

namespace nvcomp
{

/**
 * @brief Load 4 bytes from a 4-byte aligned pointer p.
**/
inline __device__ uint32_t loadUpTo4Bytes(const uint8_t *p, const ptrdiff_t &bytes_available)
{
#if 1
  if (bytes_available >= 4)
  {
    return *reinterpret_cast<const uint32_t *>(p);
  }
  else if (bytes_available == 3)
  {
    return static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(p)) | (static_cast<uint32_t>(p[2]) << 16);
  }
  else if (bytes_available == 2)
  {
    return *reinterpret_cast<const uint16_t *>(p);
  }
  else if (bytes_available == 1)
  {
    return p[0];
  }
  return 0u;
#else
  // Opportunistic load, when we don't care about over-reading the input buffer
  return bytes_available > 0 ? *reinterpret_cast<const uint32_t *>(p) : 0u;
#endif
}

// fast way of returning the rank of the current thread while respecting multi-
// dimensional thread grids
inline __device__ uint32_t lane_id()
{
  uint32_t ret;
  asm("mov.u32 %0, %laneid;" : "=r"(ret));
  return ret;
}

} // namespace nvcomp
