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

#include <cstdint>

#include "CudaConstants.h"

namespace nvcomp::cascaded::bitpack
{

// A full warp processes one 1 KiB batch. The packed payload uses one uint8_t
// per (lane, set-bit), so each lane's word count equals the byte bit width.
inline constexpr uint32_t BITPACK_BATCH_SIZE_BYTES = 1024u;
inline constexpr uint32_t BITPACK_WORDS_PER_THREAD = BITPACK_BATCH_SIZE_BYTES / sizeof(uint32_t) / WARP_SIZE;

static_assert(BITPACK_WORDS_PER_THREAD == 8u);

} // namespace nvcomp::cascaded::bitpack
