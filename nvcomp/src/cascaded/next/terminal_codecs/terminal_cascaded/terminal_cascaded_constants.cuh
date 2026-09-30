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

#include <cstddef>
#include <cstdint>

#include "cascaded/next/primitives/kibi_cascader/kibi_cascader_constants.cuh"
#include "CudaConstants.h"
#include "nvcomp/cascaded.h"

namespace nvcomp::cascaded::next::terminal_cascaded
{

inline constexpr uint32_t WARPS_PER_CTA = 8u;
inline constexpr uint32_t THREADS_PER_CTA = WARPS_PER_CTA * WARP_SIZE;
inline constexpr auto TERMINAL_CASCADED_SEARCH_ALL = kibi_cascader::SEARCH_DELTA | kibi_cascader::SEARCH_FOR |
                                                     kibi_cascader::SEARCH_RLE;

// KC abbreviates Kibi Cascaded. A kc_buffer is a compressed buffer produced by
// Kibi Cascaded.
inline constexpr uint32_t KC_SIZE_BYTES = kibi_cascader::MAX_KC_INPUT_SIZE_BYTES;
inline constexpr uint32_t WORDS_PER_KC = KC_SIZE_BYTES / sizeof(uint32_t);
inline constexpr uint32_t WORDS_PER_THREAD = kibi_cascader::KC_WORDS_PER_THREAD;
inline constexpr uint32_t UINT64_VALUES_PER_KC = KC_SIZE_BYTES / sizeof(uint64_t);

inline constexpr uint32_t KC_BUFFER_OFFSET_BYTES = sizeof(uint16_t);
inline constexpr uint32_t KC_BUFFER_PREFIX_BYTES = sizeof(uint8_t);
inline constexpr uint32_t KC_METADATA_PREFIX_BYTES = sizeof(uint8_t);
inline constexpr uint32_t MAX_KC_BUFFER_BYTES = KC_BUFFER_PREFIX_BYTES + kibi_cascader::MAX_KC_METADATA_SIZE_BYTES +
                                                KC_SIZE_BYTES;
inline constexpr uint32_t COMPRESS_SCRATCH_BYTES_PER_WARP = KC_BUFFER_OFFSET_BYTES + MAX_KC_BUFFER_BYTES;
inline constexpr uint32_t COMPRESS_SCRATCH_BYTES = WARPS_PER_CTA * COMPRESS_SCRATCH_BYTES_PER_WARP;

inline constexpr uint32_t OUTPUT_TRANSPOSE_STRIDE = WORDS_PER_THREAD + 1u;
inline constexpr uint32_t OUTPUT_TRANSPOSE_WORDS_PER_WARP = WARP_SIZE * OUTPUT_TRANSPOSE_STRIDE;

static_assert(KC_SIZE_BYTES == WORDS_PER_KC * sizeof(uint32_t));
static_assert(COMPRESS_SCRATCH_BYTES_PER_WARP == 1059u);
static_assert(COMPRESS_SCRATCH_BYTES == 8472u);

} // namespace nvcomp::cascaded::next::terminal_cascaded
