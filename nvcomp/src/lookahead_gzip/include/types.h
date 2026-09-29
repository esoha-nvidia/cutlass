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

namespace lookahead_gzip
{

enum class bufferType
{
  RING,
  LINEAR
};
using bufferType_t = bufferType;

enum class operatingMode
{
  ONESHOT,
  STREAMING
};
using operatingMode_t = operatingMode;

} // namespace lookahead_gzip
