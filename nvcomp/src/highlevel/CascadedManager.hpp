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
#include "nvcomp/cascaded.h"
#include "nvcomp/cascaded.hpp"
#include "nvcomp_common_deps/hlif_shared_types.hpp"

#include <assert.h>

namespace nvcomp
{

struct CascadedManagerImpl
    : ManagerBase<
        CascadedFormatSpecHeader,
        decltype(nvcompBatchedCascadedDecompressAsyncEx) *,
        decltype(nvcompBatchedCascadedDecompressGetTempSize) *,
        decltype(nvcompBatchedCascadedGetDecompressSizeAsync) *,
        decltype(nvcompBatchedCascadedCompressAsync) *,
        decltype(nvcompBatchedCascadedCompressGetTempSize) *,
        decltype(nvcompBatchedCascadedCompressGetMaxOutputChunkSize) *,
        nvcompBatchedCascadedCompressOpts_t,
        nvcompBatchedCascadedDecompressOpts_t,
        nvcompFormatType_t::Cascaded>
{
  static nvcompBatchedCascadedDecompressOpts_t make_targeted_decompress_opts(
    const nvcompBatchedCascadedCompressOpts_t &format_opts,
    nvcompBatchedCascadedDecompressOpts_t decompress_opts
  )
  {
    decompress_opts.common_opts = format_opts.common_opts;
    return decompress_opts;
  }

  CascadedManagerImpl(
    size_t uncomp_chunk_size,
    nvcompBatchedCascadedCompressOpts_t format_opts,
    nvcompBatchedCascadedDecompressOpts_t decompress_opts,
    cudaStream_t user_stream,
    ChecksumPolicy checksum_policy,
    ExecutionPolicy execution_policy,
    BitstreamKind bitstream_kind
  )
      : ManagerBase(
          uncomp_chunk_size,
          format_opts,
          make_targeted_decompress_opts(format_opts, decompress_opts),
          user_stream,
          checksum_policy,
          execution_policy,
          bitstream_kind,
          nvcompBatchedCascadedDecompressAsyncEx,
          nvcompBatchedCascadedDecompressGetTempSize,
          nvcompBatchedCascadedGetDecompressSizeAsync,
          nvcompBatchedCascadedCompressAsync,
          nvcompBatchedCascadedCompressGetTempSize,
          nvcompBatchedCascadedCompressGetMaxOutputChunkSize,
          query_alignment_requirements(nvcompBatchedCascadedCompressGetRequiredAlignments, format_opts),
          query_alignment_requirements(
            nvcompBatchedCascadedDecompressGetRequiredAlignments,
            make_targeted_decompress_opts(format_opts, decompress_opts)
          )
        )
  {
    static_assert(
      offsetof(CascadedFormatSpecHeader, common_opts) == offsetof(nvcompBatchedCascadedCompressOpts_t, common_opts)
    );
    static_assert(
      offsetof(CascadedFormatSpecHeader, compression_level) ==
      offsetof(nvcompBatchedCascadedCompressOpts_t, compression_level)
    );
    static_assert(
      offsetof(CascadedFormatSpecHeader, fine_grained_encoding_flags) ==
      offsetof(nvcompBatchedCascadedCompressOpts_t, fine_grained_encoding_flags)
    );
  }

  ~CascadedManagerImpl() noexcept {}
};

// C++ does not allow extern-template declarations through a type alias, so the
// specialization's argument list must be repeated here.
extern template struct ManagerBase<
  CascadedFormatSpecHeader,
  decltype(nvcompBatchedCascadedDecompressAsyncEx) *,
  decltype(nvcompBatchedCascadedDecompressGetTempSize) *,
  decltype(nvcompBatchedCascadedGetDecompressSizeAsync) *,
  decltype(nvcompBatchedCascadedCompressAsync) *,
  decltype(nvcompBatchedCascadedCompressGetTempSize) *,
  decltype(nvcompBatchedCascadedCompressGetMaxOutputChunkSize) *,
  nvcompBatchedCascadedCompressOpts_t,
  nvcompBatchedCascadedDecompressOpts_t,
  nvcompFormatType_t::Cascaded>;

CascadedManager::CascadedManager(
  size_t uncomp_chunk_size,
  const nvcompBatchedCascadedCompressOpts_t &format_opts,
  const nvcompBatchedCascadedDecompressOpts_t &decompress_opts,
  cudaStream_t user_stream,
  ChecksumPolicy checksum_policy,
  ExecutionPolicy execution_policy,
  BitstreamKind bitstream_kind
)
{
  impl = std::make_unique<CascadedManagerImpl>(
    uncomp_chunk_size,
    format_opts,
    decompress_opts,
    user_stream,
    checksum_policy,
    execution_policy,
    bitstream_kind
  );
}

CascadedManager::~CascadedManager() noexcept {}

} // namespace nvcomp
