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
        decltype(nvcompBatchedBitcompDecompressGetTempSizeAsync) *,
        decltype(nvcompBatchedBitcompGetDecompressSizeAsync) *,
        decltype(nvcompBatchedBitcompCompressAsync) *,
        decltype(nvcompBatchedBitcompCompressGetTempSizeAsync) *,
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
    BitstreamKind bitstream_kind
  )
      : ManagerBase(
          uncomp_chunk_size,
          format_opts,
          decompress_opts,
          user_stream,
          checksum_policy,
          bitstream_kind,
          nvcompBatchedBitcompDecompressAsyncEx,
          nvcompBatchedBitcompDecompressGetTempSizeAsync,
          nvcompBatchedBitcompGetDecompressSizeAsync,
          nvcompBatchedBitcompCompressAsync,
          nvcompBatchedBitcompCompressGetTempSizeAsync,
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

extern template struct ManagerBase<
  BitcompFormatSpecHeader,
  decltype(nvcompBatchedBitcompDecompressAsyncEx) *,
  decltype(nvcompBatchedBitcompDecompressGetTempSizeAsync) *,
  decltype(nvcompBatchedBitcompGetDecompressSizeAsync) *,
  decltype(nvcompBatchedBitcompCompressAsync) *,
  decltype(nvcompBatchedBitcompCompressGetTempSizeAsync) *,
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
  BitstreamKind bitstream_kind
)
{
  impl = std::make_unique<BitcompManagerImpl>(
    uncomp_chunk_size,
    format_opts,
    decompress_opts,
    user_stream,
    checksum_policy,
    bitstream_kind
  );
}

BitcompManager::~BitcompManager() {}

} // namespace nvcomp
