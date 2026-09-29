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

namespace nvcomp
{
/**
 * @brief nvCOMP supported compression formats.
*/
enum class nvcompFormatType_t : uint8_t
{
  LZ4 = 0,
  Snappy = 1,
  ANS = 2,
  GDeflate = 3,
  Cascaded = 4,
  Bitcomp = 5,
  Zstd = 6,
  Deflate = 7,
  Gzip = 8,
  NotSupportedError = 255
};
} // namespace nvcomp
