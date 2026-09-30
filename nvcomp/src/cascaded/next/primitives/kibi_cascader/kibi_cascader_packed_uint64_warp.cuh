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

#include "cascaded/common/cascaded_warp_reductions.cuh"
#include "cascaded/next/primitives/kibi_cascader/kibi_cascader_packed_uint64.cuh"
#include "CudaConstants.h"

// Warp-collective PackedUint64 operations, split out from kibi_cascader_packed_uint64.cuh because
// these require a full warp's participation (they call warp_reduce/__shfl_xor_sync), unlike the
// plain __host__ __device__ arithmetic in that header, which has no device-only dependencies.

namespace nvcomp::cascaded::next
{

inline __device__ PackedUint64 warp_reduce_or(const PackedUint64 value)
{
  return {
    warp_reduce<WARP_SIZE, warp_reduction_op::bitwise_or>(value.lo),
    warp_reduce<WARP_SIZE, warp_reduction_op::bitwise_or>(value.hi)
  };
}

inline __device__ PackedUint64 warp_reduce_and(const PackedUint64 value)
{
  return {
    warp_reduce<WARP_SIZE, warp_reduction_op::bitwise_and>(value.lo),
    warp_reduce<WARP_SIZE, warp_reduction_op::bitwise_and>(value.hi)
  };
}

inline __device__ PackedUint64 warp_shuffle_xor(const PackedUint64 value, const uint32_t lane_mask)
{
  return {__shfl_xor_sync(WARP_ALL, value.lo, lane_mask), __shfl_xor_sync(WARP_ALL, value.hi, lane_mask)};
}

} // namespace nvcomp::cascaded::next
