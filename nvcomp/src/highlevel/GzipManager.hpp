/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

#include "CRC32.hpp"
#include "highlevel/ManagerBase.hpp"
#include "lowlevel/nvcomp_private.h"
#include "nvcomp/gzip.h"
#include "nvcomp/gzip.hpp"
#include "nvcomp_common_deps/hlif_shared_types.hpp"

#include <assert.h>

namespace nvcomp
{

struct GzipManagerImpl
    : ManagerBase<
        GzipFormatSpecHeader,
        decltype(nvcompBatchedGzipDecompressAsyncEx) *,
        decltype(nvcompBatchedGzipDecompressGetTempSizeAsync) *,
        decltype(nvcompBatchedGzipGetDecompressSizeAsync) *,
        decltype(nvcompBatchedGzipCompressAsync) *,
        decltype(nvcompBatchedGzipCompressGetTempSizeAsync) *,
        decltype(nvcompBatchedGzipCompressGetMaxOutputChunkSize) *,
        nvcompBatchedGzipCompressOpts_t,
        nvcompBatchedGzipDecompressOpts_t,
        nvcompFormatType_t::Gzip>
{
  GzipManagerImpl(
    size_t uncomp_chunk_size,
    nvcompBatchedGzipCompressOpts_t format_opts,
    nvcompBatchedGzipDecompressOpts_t decompress_opts,
    cudaStream_t user_stream,
    ChecksumPolicy checksum_policy,
    BitstreamKind bitstream_kind
  )
      : ManagerBase(
          uncomp_chunk_size,
          format_opts,
          decompress_opts,
          user_stream,
          checksum_policy,
          bitstream_kind,
          nvcompBatchedGzipDecompressAsyncEx,
          nvcompBatchedGzipDecompressGetTempSizeAsync,
          nvcompBatchedGzipGetDecompressSizeAsync,
          nvcompBatchedGzipCompressAsync,
          nvcompBatchedGzipCompressGetTempSizeAsync,
          nvcompBatchedGzipCompressGetMaxOutputChunkSize,
          query_alignment_requirements(nvcompBatchedGzipCompressGetRequiredAlignments, format_opts),
          query_alignment_requirements(nvcompBatchedGzipDecompressGetRequiredAlignments, decompress_opts)
        )
  {
    static_assert(offsetof(GzipFormatSpecHeader, algorithm) == offsetof(nvcompBatchedGzipCompressOpts_t, algorithm));
  }

  ~GzipManagerImpl() {}
};

extern template struct ManagerBase<
  GzipFormatSpecHeader,
  decltype(nvcompBatchedGzipDecompressAsyncEx) *,
  decltype(nvcompBatchedGzipDecompressGetTempSizeAsync) *,
  decltype(nvcompBatchedGzipGetDecompressSizeAsync) *,
  decltype(nvcompBatchedGzipCompressAsync) *,
  decltype(nvcompBatchedGzipCompressGetTempSizeAsync) *,
  decltype(nvcompBatchedGzipCompressGetMaxOutputChunkSize) *,
  nvcompBatchedGzipCompressOpts_t,
  nvcompBatchedGzipDecompressOpts_t,
  nvcompFormatType_t::Gzip>;

GzipManager::GzipManager(
  size_t uncomp_chunk_size,
  const nvcompBatchedGzipCompressOpts_t &format_opts,
  const nvcompBatchedGzipDecompressOpts_t &decompress_opts,
  cudaStream_t user_stream,
  ChecksumPolicy checksum_policy,
  BitstreamKind bitstream_kind
)
{
  impl = std::make_unique<GzipManagerImpl>(
    uncomp_chunk_size,
    format_opts,
    decompress_opts,
    user_stream,
    checksum_policy,
    bitstream_kind
  );
}

GzipManager::~GzipManager() {}

} // namespace nvcomp
