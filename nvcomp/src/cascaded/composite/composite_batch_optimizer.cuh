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

#include <cstddef>
#include <cstdint>

#include "nvcomp/utils.hpp"

namespace nvcomp::cascaded::composite
{

template <typename data_type, typename size_type>
__host__ __device__ __forceinline__ size_type
get_adaptive_record_size(const uintptr_t record_start, const uintptr_t final_start, const size_type final_bytes)
{
  const uintptr_t final_end = roundUpTo(final_start + roundUpTo(final_bytes, sizeof(uint32_t)), sizeof(data_type));
  return static_cast<size_type>(final_end - record_start);
}

__host__ __device__ __forceinline__ uint32_t
get_adaptive_stage_mask(const uint32_t first_stage, const uint32_t num_stages = 1)
{
  return ((1u << num_stages) - 1u) << first_stage;
}

} // namespace nvcomp::cascaded::composite
