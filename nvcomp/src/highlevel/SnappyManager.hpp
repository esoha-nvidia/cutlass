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
#include "nvcomp/snappy.h"
#include "nvcomp/snappy.hpp"
#include "nvcomp_common_deps/hlif_shared_types.hpp"

#include <assert.h>

namespace nvcomp
{

struct SnappyManagerImpl
    : ManagerBase<
        SnappyFormatSpecHeader,
        decltype(nvcompBatchedSnappyDecompressAsyncEx) *,
        decltype(nvcompBatchedSnappyDecompressGetTempSize) *,
        decltype(nvcompBatchedSnappyGetDecompressSizeAsync) *,
        decltype(nvcompBatchedSnappyCompressAsync) *,
        decltype(nvcompBatchedSnappyCompressGetTempSize) *,
        decltype(nvcompBatchedSnappyCompressGetMaxOutputChunkSize) *,
        nvcompBatchedSnappyCompressOpts_t,
        nvcompBatchedSnappyDecompressOpts_t,
        nvcompFormatType_t::Snappy>
{
  SnappyManagerImpl(
    size_t uncomp_chunk_size,
    const nvcompBatchedSnappyCompressOpts_t &format_opts,
    const nvcompBatchedSnappyDecompressOpts_t &decompress_opts,
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
          nvcompBatchedSnappyDecompressAsyncEx,
          nvcompBatchedSnappyDecompressGetTempSize,
          nvcompBatchedSnappyGetDecompressSizeAsync,
          nvcompBatchedSnappyCompressAsync,
          nvcompBatchedSnappyCompressGetTempSize,
          nvcompBatchedSnappyCompressGetMaxOutputChunkSize,
          query_alignment_requirements(nvcompBatchedSnappyCompressGetRequiredAlignments, format_opts),
          query_alignment_requirements(nvcompBatchedSnappyDecompressGetRequiredAlignments, decompress_opts)
        )
  {}

  ~SnappyManagerImpl() {};
};

// C++ does not allow extern-template declarations through a type alias, so the
// specialization's argument list must be repeated here.
extern template struct ManagerBase<
  SnappyFormatSpecHeader,
  decltype(nvcompBatchedSnappyDecompressAsyncEx) *,
  decltype(nvcompBatchedSnappyDecompressGetTempSize) *,
  decltype(nvcompBatchedSnappyGetDecompressSizeAsync) *,
  decltype(nvcompBatchedSnappyCompressAsync) *,
  decltype(nvcompBatchedSnappyCompressGetTempSize) *,
  decltype(nvcompBatchedSnappyCompressGetMaxOutputChunkSize) *,
  nvcompBatchedSnappyCompressOpts_t,
  nvcompBatchedSnappyDecompressOpts_t,
  nvcompFormatType_t::Snappy>;

SnappyManager::SnappyManager(
  size_t uncomp_chunk_size,
  const nvcompBatchedSnappyCompressOpts_t &format_opts,
  const nvcompBatchedSnappyDecompressOpts_t &decompress_opts,
  cudaStream_t user_stream,
  ChecksumPolicy checksum_policy,
  ExecutionPolicy execution_policy,
  BitstreamKind bitstream_kind
)
{
  impl = std::make_unique<SnappyManagerImpl>(
    uncomp_chunk_size,
    format_opts,
    decompress_opts,
    user_stream,
    checksum_policy,
    execution_policy,
    bitstream_kind
  );
}

SnappyManager::~SnappyManager() {}

} // namespace nvcomp
