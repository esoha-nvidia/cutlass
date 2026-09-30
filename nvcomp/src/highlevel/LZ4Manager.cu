/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
*/

#include "LZ4Manager.hpp"

namespace nvcomp
{

template struct ManagerBase<
  LZ4FormatSpecHeader,
  decltype(nvcompBatchedLZ4DecompressAsyncEx) *,
  decltype(nvcompBatchedLZ4DecompressGetTempSize) *,
  decltype(nvcompBatchedLZ4GetDecompressSizeAsync) *,
  decltype(nvcompBatchedLZ4CompressAsync) *,
  decltype(nvcompBatchedLZ4CompressGetTempSize) *,
  decltype(nvcompBatchedLZ4CompressGetMaxOutputChunkSize) *,
  nvcompBatchedLZ4CompressOpts_t,
  nvcompBatchedLZ4DecompressOpts_t,
  nvcompFormatType_t::LZ4>;

} // namespace nvcomp
