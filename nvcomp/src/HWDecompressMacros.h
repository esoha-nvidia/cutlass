/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

// Base DE support comes with CTK 12.8 & r570
#define CTK_DE_BASE_SUPPORTED(CTK_VERSION) ((CTK_VERSION) >= 12080)

// LZ4 DE support comes with CTK 12.9 & r575
#define CTK_DE_LZ4_SUPPORTED(CTK_VERSION) ((CTK_VERSION) >= 12090)
