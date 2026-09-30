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

#include "ANSManager.hpp"

namespace nvcomp
{

template struct ManagerBase<
  ANSFormatSpecHeader,
  decltype(nvcompBatchedANSDecompressAsyncEx) *,
  decltype(nvcompBatchedANSDecompressGetTempSize) *,
  decltype(nvcompBatchedANSGetDecompressSizeAsync) *,
  decltype(nvcompBatchedANSCompressAsync) *,
  decltype(nvcompBatchedANSCompressGetTempSize) *,
  decltype(nvcompBatchedANSCompressGetMaxOutputChunkSize) *,
  nvcompBatchedANSCompressOpts_t,
  nvcompBatchedANSDecompressOpts_t,
  nvcompFormatType_t::ANS>;

} // namespace nvcomp
