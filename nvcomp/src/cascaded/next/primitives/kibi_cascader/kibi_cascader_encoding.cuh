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

namespace nvcomp::cascaded::next::kibi_cascader
{

// Bitpack always terminates the encoding, so it is not a transformation; None
// means the values are bitpacked without any preceding transformation.
enum class PreBitpackTransform : uint32_t
{
  None = 0u,
  FrameOfReference = 1u,
  Delta = 2u,
  DeltaZigzag = 3u,
  DeltaFrameOfReference = 4u
};

enum class RleMode : uint32_t
{
  None = 0u,
  Before = 1u
};

// The encoding selector is serialized as one byte. Keep the RLE mode in the
// selector's two high bits so the complete selection survives serialization.
inline constexpr uint32_t RLE_MODE_SHIFT = 6u;
inline constexpr uint32_t RLE_MODE_MASK = 0x3u << RLE_MODE_SHIFT;

inline constexpr __host__ __device__ uint32_t
encoding_metadata(const PreBitpackTransform transform, const RleMode rle_mode)
{
  return static_cast<uint32_t>(transform) | (static_cast<uint32_t>(rle_mode) << RLE_MODE_SHIFT);
}

inline constexpr __host__ __device__ PreBitpackTransform get_pre_bitpack_transform(const uint32_t metadata)
{
  return static_cast<PreBitpackTransform>(metadata & ~RLE_MODE_MASK);
}

inline constexpr __host__ __device__ RleMode get_rle_mode(const uint32_t metadata)
{
  return static_cast<RleMode>((metadata & RLE_MODE_MASK) >> RLE_MODE_SHIFT);
}

} // namespace nvcomp::cascaded::next::kibi_cascader
