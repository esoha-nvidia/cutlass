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

#pragma once

#include "CRC32.hpp"
#include "highlevel/ManagerBase.hpp"
#include "lowlevel/nvcomp_private.h"
#include "nvcomp/zstd.h"
#include "nvcomp/zstd.hpp"
#include "nvcomp_common_deps/hlif_shared_types.hpp"

#include <assert.h>

namespace nvcomp
{

struct ZstdManagerImpl
    : ManagerBase<
        ZstdFormatSpecHeader,
        decltype(nvcompBatchedZstdDecompressAsyncEx) *,
        decltype(nvcompBatchedZstdDecompressGetTempSize) *,
        decltype(nvcompBatchedZstdGetDecompressSizeAsync) *,
        decltype(nvcompBatchedZstdCompressAsync) *,
        decltype(nvcompBatchedZstdCompressGetTempSize) *,
        decltype(nvcompBatchedZstdCompressGetMaxOutputChunkSize) *,
        nvcompBatchedZstdCompressOpts_t,
        nvcompBatchedZstdDecompressOpts_t,
        nvcompFormatType_t::Zstd>
{
  ZstdManagerImpl(
    size_t uncomp_chunk_size,
    const nvcompBatchedZstdCompressOpts_t &format_opts,
    const nvcompBatchedZstdDecompressOpts_t &decompress_opts,
    cudaStream_t user_stream,
    ChecksumPolicy checksum_policy,
    ExecutionPolicy execution_policy,
    BitstreamKind bitstream_kind
  )
      : ManagerBase(
          uncomp_chunk_size,
          format_opts,
          decompress_opts,
          user_stream,
          checksum_policy,
          execution_policy,
          bitstream_kind,
          nvcompBatchedZstdDecompressAsyncEx,
          nvcompBatchedZstdDecompressGetTempSize,
          nvcompBatchedZstdGetDecompressSizeAsync,
          nvcompBatchedZstdCompressAsync,
          nvcompBatchedZstdCompressGetTempSize,
          nvcompBatchedZstdCompressGetMaxOutputChunkSize,
          query_alignment_requirements(nvcompBatchedZstdCompressGetRequiredAlignments, format_opts),
          query_alignment_requirements(nvcompBatchedZstdDecompressGetRequiredAlignments, decompress_opts)
        )
  {}

  ~ZstdManagerImpl() {};
};

// C++ does not allow extern-template declarations through a type alias, so the
// specialization's argument list must be repeated here.
extern template struct ManagerBase<
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

ZstdManager::ZstdManager(
  size_t uncomp_chunk_size,
  const nvcompBatchedZstdCompressOpts_t &format_opts,
  const nvcompBatchedZstdDecompressOpts_t &decompress_opts,
  cudaStream_t user_stream,
  ChecksumPolicy checksum_policy,
  ExecutionPolicy execution_policy,
  BitstreamKind bitstream_kind
)
{
  impl = std::make_unique<ZstdManagerImpl>(
    uncomp_chunk_size,
    format_opts,
    decompress_opts,
    user_stream,
    checksum_policy,
    execution_policy,
    bitstream_kind
  );
}

ZstdManager::~ZstdManager() {}

} // namespace nvcomp
