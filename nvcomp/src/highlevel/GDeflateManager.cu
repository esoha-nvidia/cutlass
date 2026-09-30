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

#include "GDeflateManager.hpp"

namespace nvcomp
{

template struct ManagerBase<
  GdeflateFormatSpecHeader,
  decltype(nvcompBatchedGdeflateDecompressAsyncEx) *,
  decltype(nvcompBatchedGdeflateDecompressGetTempSize) *,
  decltype(nvcompBatchedGdeflateGetDecompressSizeAsync) *,
  decltype(nvcompBatchedGdeflateCompressAsync) *,
  decltype(nvcompBatchedGdeflateCompressGetTempSize) *,
  decltype(nvcompBatchedGdeflateCompressGetMaxOutputChunkSize) *,
  nvcompBatchedGdeflateCompressOpts_t,
  nvcompBatchedGdeflateDecompressOpts_t,
  nvcompFormatType_t::GDeflate>;

} // namespace nvcomp
