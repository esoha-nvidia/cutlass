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
#include "nvcomp/bitcomp.h"
#include "nvcomp/bitcomp.hpp"
#include "nvcomp_common_deps/hlif_shared_types.hpp"

#include <assert.h>

namespace nvcomp
{

struct BitcompManagerImpl
    : ManagerBase<
        BitcompFormatSpecHeader,
        decltype(nvcompBatchedBitcompDecompressAsyncEx) *,
        decltype(nvcompBatchedBitcompDecompressGetTempSize) *,
        decltype(nvcompBatchedBitcompGetDecompressSizeAsync) *,
        decltype(nvcompBatchedBitcompCompressAsync) *,
        decltype(nvcompBatchedBitcompCompressGetTempSize) *,
        decltype(nvcompBatchedBitcompCompressGetMaxOutputChunkSize) *,
        nvcompBatchedBitcompCompressOpts_t,
        nvcompBatchedBitcompDecompressOpts_t,
        nvcompFormatType_t::Bitcomp>
{
  BitcompManagerImpl(
    size_t uncomp_chunk_size,
    nvcompBatchedBitcompCompressOpts_t format_opts,
    nvcompBatchedBitcompDecompressOpts_t decompress_opts,
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
          nvcompBatchedBitcompDecompressAsyncEx,
          nvcompBatchedBitcompDecompressGetTempSize,
          nvcompBatchedBitcompGetDecompressSizeAsync,
          nvcompBatchedBitcompCompressAsync,
          nvcompBatchedBitcompCompressGetTempSize,
          nvcompBatchedBitcompCompressGetMaxOutputChunkSize,
          query_alignment_requirements(nvcompBatchedBitcompCompressGetRequiredAlignments, format_opts),
          query_alignment_requirements(nvcompBatchedBitcompDecompressGetRequiredAlignments, decompress_opts)
        )
  {
    static_assert(
      offsetof(BitcompFormatSpecHeader, algorithm) == offsetof(nvcompBatchedBitcompCompressOpts_t, algorithm)
    );
    static_assert(
      offsetof(BitcompFormatSpecHeader, data_type) == offsetof(nvcompBatchedBitcompCompressOpts_t, data_type)
    );
    static_assert(offsetof(BitcompFormatSpecHeader, delta) == offsetof(nvcompBatchedBitcompCompressOpts_t, delta));
    static_assert(offsetof(BitcompFormatSpecHeader, mode) == offsetof(nvcompBatchedBitcompCompressOpts_t, mode));
    static_assert(offsetof(BitcompFormatSpecHeader, reserved) == offsetof(nvcompBatchedBitcompCompressOpts_t, reserved));
  }

  ~BitcompManagerImpl() {}
};

// C++ does not allow extern-template declarations through a type alias, so the
// specialization's argument list must be repeated here.
extern template struct ManagerBase<
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

BitcompManager::BitcompManager(
  size_t uncomp_chunk_size,
  const nvcompBatchedBitcompCompressOpts_t &format_opts,
  const nvcompBatchedBitcompDecompressOpts_t &decompress_opts,
  cudaStream_t user_stream,
  ChecksumPolicy checksum_policy,
  ExecutionPolicy execution_policy,
  BitstreamKind bitstream_kind
)
{
  impl = std::make_unique<BitcompManagerImpl>(
    uncomp_chunk_size,
    format_opts,
    decompress_opts,
    user_stream,
    checksum_policy,
    execution_policy,
    bitstream_kind
  );
}

BitcompManager::~BitcompManager() {}

} // namespace nvcomp
