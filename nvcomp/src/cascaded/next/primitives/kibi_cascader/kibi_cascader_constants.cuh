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

#include <cstdint>

#include "cascaded/bitpack/bitpack_constants.cuh"
#include "CudaConstants.h"

namespace nvcomp::cascaded::next::kibi_cascader
{

// The kibi cascader is designed to brute force the best
// compression graph (given a user-defined search space)
// for an input vector of size <= 1024B.
inline constexpr uint32_t MAX_KC_INPUT_SIZE_BYTES = bitpack::BITPACK_BATCH_SIZE_BYTES;

// Each warp lane holds 32 bytes of bitpack input in registers.
inline constexpr uint32_t KC_BYTES_PER_THREAD = MAX_KC_INPUT_SIZE_BYTES / WARP_SIZE;
inline constexpr uint32_t KC_WORDS_PER_THREAD = KC_BYTES_PER_THREAD / sizeof(uint32_t);

// The data type and user-defined search space determine how much metadata is
// generated per kibi-cascader call.
inline constexpr uint32_t MAX_KC_METADATA_SIZE_BYTES = 32u;
inline constexpr uint32_t MAX_KC_METADATA_WORDS = MAX_KC_METADATA_SIZE_BYTES / sizeof(uint32_t);

// The search space is determined by a bit mask. Each bit position
// determines whether the corresponding encoding can be considered.
using kibi_cascader_optimization_search_space = uint32_t;
inline constexpr kibi_cascader_optimization_search_space SEARCH_NONE = 0u;
inline constexpr kibi_cascader_optimization_search_space SEARCH_DELTA = 1u << 0u;
inline constexpr kibi_cascader_optimization_search_space SEARCH_FOR = 1u << 1u;
inline constexpr kibi_cascader_optimization_search_space SEARCH_RLE = 1u << 2u;
inline constexpr kibi_cascader_optimization_search_space SEARCH_ALP = 1u << 3u;

inline constexpr kibi_cascader_optimization_search_space SEARCH_ALL = SEARCH_DELTA | SEARCH_FOR | SEARCH_RLE |
                                                                      SEARCH_ALP;

} // namespace nvcomp::cascaded::next::kibi_cascader
