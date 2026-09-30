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
#include "nvcomp/deflate.h"
#include "nvcomp/deflate.hpp"
#include "nvcomp_common_deps/hlif_shared_types.hpp"

#include <assert.h>

namespace nvcomp
{

struct DeflateManagerImpl
    : ManagerBase<
        DeflateFormatSpecHeader,
        decltype(nvcompBatchedDeflateDecompressAsyncEx) *,
        decltype(nvcompBatchedDeflateDecompressGetTempSize) *,
        decltype(nvcompBatchedDeflateGetDecompressSizeAsync) *,
        decltype(nvcompBatchedDeflateCompressAsync) *,
        decltype(nvcompBatchedDeflateCompressGetTempSize) *,
        decltype(nvcompBatchedDeflateCompressGetMaxOutputChunkSize) *,
        nvcompBatchedDeflateCompressOpts_t,
        nvcompBatchedDeflateDecompressOpts_t,
        nvcompFormatType_t::Deflate>
{
  DeflateManagerImpl(
    size_t uncomp_chunk_size,
    nvcompBatchedDeflateCompressOpts_t format_opts,
    nvcompBatchedDeflateDecompressOpts_t decompress_opts,
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
          nvcompBatchedDeflateDecompressAsyncEx,
          nvcompBatchedDeflateDecompressGetTempSize,
          nvcompBatchedDeflateGetDecompressSizeAsync,
          nvcompBatchedDeflateCompressAsync,
          nvcompBatchedDeflateCompressGetTempSize,
          nvcompBatchedDeflateCompressGetMaxOutputChunkSize,
          query_alignment_requirements(nvcompBatchedDeflateCompressGetRequiredAlignments, format_opts),
          query_alignment_requirements(nvcompBatchedDeflateDecompressGetRequiredAlignments, decompress_opts)
        )
  {
    static_assert(
      offsetof(DeflateFormatSpecHeader, algorithm) == offsetof(nvcompBatchedDeflateCompressOpts_t, algorithm)
    );
  }

  ~DeflateManagerImpl() {}
};

// C++ does not allow extern-template declarations through a type alias, so the
// specialization's argument list must be repeated here.
extern template struct ManagerBase<
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

DeflateManager::DeflateManager(
  size_t uncomp_chunk_size,
  const nvcompBatchedDeflateCompressOpts_t &format_opts,
  const nvcompBatchedDeflateDecompressOpts_t &decompress_opts,
  cudaStream_t user_stream,
  ChecksumPolicy checksum_policy,
  ExecutionPolicy execution_policy,
  BitstreamKind bitstream_kind
)
{
  impl = std::make_unique<DeflateManagerImpl>(
    uncomp_chunk_size,
    format_opts,
    decompress_opts,
    user_stream,
    checksum_policy,
    execution_policy,
    bitstream_kind
  );
}

DeflateManager::~DeflateManager() {}

} // namespace nvcomp
