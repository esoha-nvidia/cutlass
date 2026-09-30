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
#include "nvcomp/ans.h"
#include "nvcomp/ans.hpp"
#include "nvcomp_common_deps/hlif_shared_types.hpp"

#include <assert.h>

namespace nvcomp
{

namespace
{
// Fill decompress fields that are still at their defaults from the compressor.
nvcompBatchedANSDecompressOpts_t build_decomp_opts(
  const nvcompBatchedANSDecompressOpts_t &orig_opts,
  uint8_t max_sub_chunk_count,
  nvcompType_t data_type,
  uint8_t states_per_lane
)
{
  auto new_opts = orig_opts;
  if (orig_opts.data_type == nvcompBatchedANSDecompressDefaultOpts.data_type)
  {
    new_opts.data_type = data_type;
  }
  if (orig_opts.max_sub_chunk_count == nvcompBatchedANSDecompressDefaultOpts.max_sub_chunk_count)
  {
    new_opts.max_sub_chunk_count = max_sub_chunk_count;
  }
  if (orig_opts.states_per_lane == nvcompBatchedANSDecompressDefaultOpts.states_per_lane)
  {
    new_opts.states_per_lane = states_per_lane;
  }
  return new_opts;
}
} // namespace

struct ANSManagerImpl
    : ManagerBase<
        ANSFormatSpecHeader,
        decltype(nvcompBatchedANSDecompressAsyncEx) *,
        decltype(nvcompBatchedANSDecompressGetTempSize) *,
        decltype(nvcompBatchedANSGetDecompressSizeAsync) *,
        decltype(nvcompBatchedANSCompressAsync) *,
        decltype(nvcompBatchedANSCompressGetTempSize) *,
        decltype(nvcompBatchedANSCompressGetMaxOutputChunkSize) *,
        nvcompBatchedANSCompressOpts_t,
        nvcompBatchedANSDecompressOpts_t,
        nvcompFormatType_t::ANS>
{
  ANSManagerImpl(
    size_t uncomp_chunk_size,
    const nvcompBatchedANSCompressOpts_t &format_opts,
    const nvcompBatchedANSDecompressOpts_t &decompress_opts,
    cudaStream_t user_stream,
    ChecksumPolicy checksum_policy,
    ExecutionPolicy execution_policy,
    BitstreamKind bitstream_kind
  )
      : ManagerBase(
          uncomp_chunk_size,
          format_opts,
          build_decomp_opts(
            decompress_opts,
            format_opts.max_sub_chunk_count,
            format_opts.data_type,
            format_opts.states_per_lane
          ),
          user_stream,
          checksum_policy,
          execution_policy,
          bitstream_kind,
          nvcompBatchedANSDecompressAsyncEx,
          nvcompBatchedANSDecompressGetTempSize,
          nvcompBatchedANSGetDecompressSizeAsync,
          nvcompBatchedANSCompressAsync,
          nvcompBatchedANSCompressGetTempSize,
          nvcompBatchedANSCompressGetMaxOutputChunkSize,
          query_alignment_requirements(nvcompBatchedANSCompressGetRequiredAlignments, format_opts),
          query_alignment_requirements(nvcompBatchedANSDecompressGetRequiredAlignments, decompress_opts)
        )
  {
    static_assert(offsetof(ANSFormatSpecHeader, type) == offsetof(nvcompBatchedANSCompressOpts_t, type));
    static_assert(offsetof(ANSFormatSpecHeader, data_type) == offsetof(nvcompBatchedANSCompressOpts_t, data_type));
    static_assert(
      offsetof(ANSFormatSpecHeader, max_sub_chunk_count) ==
      offsetof(nvcompBatchedANSCompressOpts_t, max_sub_chunk_count)
    );
    static_assert(
      offsetof(ANSFormatSpecHeader, states_per_lane) == offsetof(nvcompBatchedANSCompressOpts_t, states_per_lane)
    );
    static_assert(
      offsetof(ANSFormatSpecHeader, histogram_reduction_log2) ==
      offsetof(nvcompBatchedANSCompressOpts_t, histogram_reduction_log2)
    );
    static_assert(offsetof(ANSFormatSpecHeader, reserved) == offsetof(nvcompBatchedANSCompressOpts_t, reserved));
  }

  ~ANSManagerImpl() {};
};

// C++ does not allow extern-template declarations through a type alias, so the
// specialization's argument list must be repeated here.
extern template struct ManagerBase<
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

ANSManager::ANSManager(
  size_t uncomp_chunk_size,
  const nvcompBatchedANSCompressOpts_t &format_opts,
  const nvcompBatchedANSDecompressOpts_t &decompress_opts,
  cudaStream_t user_stream,
  ChecksumPolicy checksum_policy,
  ExecutionPolicy execution_policy,
  BitstreamKind bitstream_kind
)
{
  impl = std::make_unique<ANSManagerImpl>(
    uncomp_chunk_size,
    format_opts,
    decompress_opts,
    user_stream,
    checksum_policy,
    execution_policy,
    bitstream_kind
  );
}

ANSManager::~ANSManager() {}

} // namespace nvcomp
