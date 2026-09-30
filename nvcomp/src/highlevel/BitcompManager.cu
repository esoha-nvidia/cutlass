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

#include "BitcompManager.hpp"

namespace nvcomp
{

template struct ManagerBase<
  BitcompFormatSpecHeader,
  decltype(nvcompBatchedBitcompDecompressAsyncEx) *,
  decltype(nvcompBatchedBitcompDecompressGetTempSize) *,
  decltype(nvcompBatchedBitcompGetDecompressSizeAsync) *,
  decltype(nvcompBatchedBitcompCompressAsync) *,
  decltype(nvcompBatchedBitcompCompressGetTempSize) *,
  decltype(nvcompBatchedBitcompCompressGetMaxOutputChunkSize) *,
  nvcompBatchedBitcompCompressOpts_t,
  nvcompBatchedBitcompDecompressOpts_t,
  nvcompFormatType_t::Bitcomp>;

} // namespace nvcomp
