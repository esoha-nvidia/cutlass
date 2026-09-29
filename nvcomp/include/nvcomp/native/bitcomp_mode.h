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

/**
 * @brief Bitcomp compression mode.
 */
typedef enum bitcompMode_t
{
  BITCOMP_LOSSLESS = 0,
  /** Lossy compression of floating-point data quantized to signed integers. */
  BITCOMP_LOSSY_FP_TO_SIGNED,
  /** Lossy compression of floating-point data quantized to unsigned integers. */
  BITCOMP_LOSSY_FP_TO_UNSIGNED
} bitcompMode_t;
