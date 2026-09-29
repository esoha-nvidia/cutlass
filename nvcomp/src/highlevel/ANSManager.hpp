#pragma once

/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

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
// Carry the compressor's max sub chunk count and data type into the decompress
// opts so the decompress launch grid matches what was used at compression time
// and the data type selects the type-specialized decode kernel (fp8/fp16); the
// bitstream itself is self-describing.
nvcompBatchedANSDecompressOpts_t
build_decomp_opts(const nvcompBatchedANSDecompressOpts_t &orig_opts, uint8_t max_sub_chunk_count, nvcompType_t data_type)
{
  auto new_opts = orig_opts;
  new_opts.max_sub_chunk_count = max_sub_chunk_count;
  new_opts.data_type = data_type;
  return new_opts;
}
} // namespace

struct ANSManagerImpl
    : ManagerBase<
        ANSFormatSpecHeader,
        decltype(nvcompBatchedANSDecompressAsyncEx) *,
        decltype(nvcompBatchedANSDecompressGetTempSizeAsync) *,
        decltype(nvcompBatchedANSGetDecompressSizeAsync) *,
        decltype(nvcompBatchedANSCompressAsync) *,
        decltype(nvcompBatchedANSCompressGetTempSizeAsync) *,
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
    BitstreamKind bitstream_kind
  )
      : ManagerBase(
          uncomp_chunk_size,
          format_opts,
          build_decomp_opts(decompress_opts, format_opts.max_sub_chunk_count, format_opts.data_type),
          user_stream,
          checksum_policy,
          bitstream_kind,
          nvcompBatchedANSDecompressAsyncEx,
          nvcompBatchedANSDecompressGetTempSizeAsync,
          nvcompBatchedANSGetDecompressSizeAsync,
          nvcompBatchedANSCompressAsync,
          nvcompBatchedANSCompressGetTempSizeAsync,
          nvcompBatchedANSCompressGetMaxOutputChunkSize,
          query_alignment_requirements(nvcompBatchedANSCompressGetRequiredAlignments, format_opts),
          query_alignment_requirements(nvcompBatchedANSDecompressGetRequiredAlignments, decompress_opts)
        )
  {}

  ~ANSManagerImpl() {};
};

extern template struct ManagerBase<
  ANSFormatSpecHeader,
  decltype(nvcompBatchedANSDecompressAsyncEx) *,
  decltype(nvcompBatchedANSDecompressGetTempSizeAsync) *,
  decltype(nvcompBatchedANSGetDecompressSizeAsync) *,
  decltype(nvcompBatchedANSCompressAsync) *,
  decltype(nvcompBatchedANSCompressGetTempSizeAsync) *,
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
  BitstreamKind bitstream_kind
)
{
  impl = std::make_unique<ANSManagerImpl>(
    uncomp_chunk_size,
    format_opts,
    decompress_opts,
    user_stream,
    checksum_policy,
    bitstream_kind
  );
}

ANSManager::~ANSManager() {}

} // namespace nvcomp
