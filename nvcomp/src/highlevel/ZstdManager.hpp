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
        decltype(nvcompBatchedZstdDecompressGetTempSizeAsync) *,
        decltype(nvcompBatchedZstdGetDecompressSizeAsync) *,
        decltype(nvcompBatchedZstdCompressAsync) *,
        decltype(nvcompBatchedZstdCompressGetTempSizeAsync) *,
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
    BitstreamKind bitstream_kind
  )
      : ManagerBase(
          uncomp_chunk_size,
          format_opts,
          decompress_opts,
          user_stream,
          checksum_policy,
          bitstream_kind,
          nvcompBatchedZstdDecompressAsyncEx,
          nvcompBatchedZstdDecompressGetTempSizeAsync,
          nvcompBatchedZstdGetDecompressSizeAsync,
          nvcompBatchedZstdCompressAsync,
          nvcompBatchedZstdCompressGetTempSizeAsync,
          nvcompBatchedZstdCompressGetMaxOutputChunkSize,
          query_alignment_requirements(nvcompBatchedZstdCompressGetRequiredAlignments, format_opts),
          query_alignment_requirements(nvcompBatchedZstdDecompressGetRequiredAlignments, decompress_opts)
        )
  {
    static_assert(std::is_empty<ZstdFormatSpecHeader>::value);
  }

  ~ZstdManagerImpl() {};
};

extern template struct ManagerBase<
  ZstdFormatSpecHeader,
  decltype(nvcompBatchedZstdDecompressAsyncEx) *,
  decltype(nvcompBatchedZstdDecompressGetTempSizeAsync) *,
  decltype(nvcompBatchedZstdGetDecompressSizeAsync) *,
  decltype(nvcompBatchedZstdCompressAsync) *,
  decltype(nvcompBatchedZstdCompressGetTempSizeAsync) *,
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
  BitstreamKind bitstream_kind
)
{
  impl = std::make_unique<ZstdManagerImpl>(
    uncomp_chunk_size,
    format_opts,
    decompress_opts,
    user_stream,
    checksum_policy,
    bitstream_kind
  );
}

ZstdManager::~ZstdManager() {}

} // namespace nvcomp
