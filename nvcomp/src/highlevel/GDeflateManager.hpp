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
#include "nvcomp/gdeflate.h"
#include "nvcomp/gdeflate.hpp"
#include "nvcomp_common_deps/hlif_shared_types.hpp"

#include <assert.h>

namespace nvcomp
{

struct GdeflateManagerImpl
    : ManagerBase<
        GdeflateFormatSpecHeader,
        decltype(nvcompBatchedGdeflateDecompressAsyncEx) *,
        decltype(nvcompBatchedGdeflateDecompressGetTempSize) *,
        decltype(nvcompBatchedGdeflateGetDecompressSizeAsync) *,
        decltype(nvcompBatchedGdeflateCompressAsync) *,
        decltype(nvcompBatchedGdeflateCompressGetTempSize) *,
        decltype(nvcompBatchedGdeflateCompressGetMaxOutputChunkSize) *,
        nvcompBatchedGdeflateCompressOpts_t,
        nvcompBatchedGdeflateDecompressOpts_t,
        nvcompFormatType_t::GDeflate>
{
  GdeflateManagerImpl(
    size_t uncomp_chunk_size,
    nvcompBatchedGdeflateCompressOpts_t format_opts,
    nvcompBatchedGdeflateDecompressOpts_t decompress_opts,
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
          nvcompBatchedGdeflateDecompressAsyncEx,
          nvcompBatchedGdeflateDecompressGetTempSize,
          nvcompBatchedGdeflateGetDecompressSizeAsync,
          nvcompBatchedGdeflateCompressAsync,
          nvcompBatchedGdeflateCompressGetTempSize,
          nvcompBatchedGdeflateCompressGetMaxOutputChunkSize,
          query_alignment_requirements(nvcompBatchedGdeflateCompressGetRequiredAlignments, format_opts),
          query_alignment_requirements(nvcompBatchedGdeflateDecompressGetRequiredAlignments, decompress_opts)
        )
  {
    static_assert(
      offsetof(GdeflateFormatSpecHeader, algorithm) == offsetof(nvcompBatchedGdeflateCompressOpts_t, algorithm)
    );
  }

  ~GdeflateManagerImpl() {}
};

// C++ does not allow extern-template declarations through a type alias, so the
// specialization's argument list must be repeated here.
extern template struct ManagerBase<
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

GdeflateManager::GdeflateManager(
  size_t uncomp_chunk_size,
  const nvcompBatchedGdeflateCompressOpts_t &format_opts,
  const nvcompBatchedGdeflateDecompressOpts_t &decompress_opts,
  cudaStream_t user_stream,
  ChecksumPolicy checksum_policy,
  ExecutionPolicy execution_policy,
  BitstreamKind bitstream_kind
)
{
  impl = std::make_unique<GdeflateManagerImpl>(
    uncomp_chunk_size,
    format_opts,
    decompress_opts,
    user_stream,
    checksum_policy,
    execution_policy,
    bitstream_kind
  );
}

GdeflateManager::~GdeflateManager() {}

} // namespace nvcomp
