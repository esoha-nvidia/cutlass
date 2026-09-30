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

#include "DeflateManager.hpp"

namespace nvcomp
{

template struct ManagerBase<
  DeflateFormatSpecHeader,
  decltype(nvcompBatchedDeflateDecompressAsyncEx) *,
  decltype(nvcompBatchedDeflateDecompressGetTempSize) *,
  decltype(nvcompBatchedDeflateGetDecompressSizeAsync) *,
  decltype(nvcompBatchedDeflateCompressAsync) *,
  decltype(nvcompBatchedDeflateCompressGetTempSize) *,
  decltype(nvcompBatchedDeflateCompressGetMaxOutputChunkSize) *,
  nvcompBatchedDeflateCompressOpts_t,
  nvcompBatchedDeflateDecompressOpts_t,
  nvcompFormatType_t::Deflate>;

} // namespace nvcomp
