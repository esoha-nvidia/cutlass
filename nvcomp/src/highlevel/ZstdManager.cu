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

#include "ZstdManager.hpp"

namespace nvcomp
{

template struct ManagerBase<
  ZstdFormatSpecHeader,
  decltype(nvcompBatchedZstdDecompressAsyncEx) *,
  decltype(nvcompBatchedZstdDecompressGetTempSize) *,
  decltype(nvcompBatchedZstdGetDecompressSizeAsync) *,
  decltype(nvcompBatchedZstdCompressAsync) *,
  decltype(nvcompBatchedZstdCompressGetTempSize) *,
  decltype(nvcompBatchedZstdCompressGetMaxOutputChunkSize) *,
  nvcompBatchedZstdCompressOpts_t,
  nvcompBatchedZstdDecompressOpts_t,
  nvcompFormatType_t::Zstd>;

} // namespace nvcomp
