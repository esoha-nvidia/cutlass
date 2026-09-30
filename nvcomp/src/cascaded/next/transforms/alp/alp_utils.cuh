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

#include "cascaded/next/transforms/alp/alp_constants.cuh"

namespace nvcomp::cascaded::next::transforms::alp
{

/**
 * The outcome of an ALP exponent search, uniform across the warp. When applied is false the
 * transform was declined, the values keep their original bit patterns, and exponent is zero.
 */
struct AlpPlan
{
  bool applied;
  uint32_t exponent;
};

/**
 * Packs a plan into the single metadata word ALP contributes: bit 0 is applied and bits 1
 * through 4 the exponent. Four bits cover every exponent a search can select, because
 * DOUBLE_MAX_EXPONENT bounds both types.
 */
inline constexpr __host__ __device__ uint32_t serialize_plan(const AlpPlan plan)
{
  return static_cast<uint32_t>(plan.applied) | (plan.exponent << 1u);
}

/**
 * Unpacks the metadata word written by serialize_plan, with exponent clamped to the max allowed.
 */
inline constexpr __host__ __device__ AlpPlan deserialize_plan(const uint32_t value)
{
  const uint32_t exponent = (value >> 1u) & 0xFu;
  const bool in_range = exponent <= DOUBLE_MAX_EXPONENT;
  return AlpPlan{(value & 1u) != 0u && in_range, in_range ? exponent : 0u};
}

} // namespace nvcomp::cascaded::next::transforms::alp
