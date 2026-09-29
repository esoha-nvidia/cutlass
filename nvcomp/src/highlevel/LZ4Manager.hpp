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
#include "nvcomp/lz4.h"
#include "nvcomp/lz4.hpp"
#include "nvcomp_common_deps/hlif_shared_types.hpp"

#include <assert.h>

namespace nvcomp
{

struct LZ4ManagerImpl
    : ManagerBase<
        LZ4FormatSpecHeader,
        decltype(nvcompBatchedLZ4DecompressAsyncEx) *,
        decltype(nvcompBatchedLZ4DecompressGetTempSizeAsync) *,
        decltype(nvcompBatchedLZ4GetDecompressSizeAsync) *,
        decltype(nvcompBatchedLZ4CompressAsync) *,
        decltype(nvcompBatchedLZ4CompressGetTempSizeAsync) *,
        decltype(nvcompBatchedLZ4CompressGetMaxOutputChunkSize) *,
        nvcompBatchedLZ4CompressOpts_t,
        nvcompBatchedLZ4DecompressOpts_t,
        nvcompFormatType_t::LZ4>
{
  LZ4ManagerImpl(
    size_t uncomp_chunk_size,
    nvcompBatchedLZ4CompressOpts_t format_opts,
    nvcompBatchedLZ4DecompressOpts_t decompress_opts,
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
          nvcompBatchedLZ4DecompressAsyncEx,
          nvcompBatchedLZ4DecompressGetTempSizeAsync,
          nvcompBatchedLZ4GetDecompressSizeAsync,
          nvcompBatchedLZ4CompressAsync,
          nvcompBatchedLZ4CompressGetTempSizeAsync,
          nvcompBatchedLZ4CompressGetMaxOutputChunkSize,
          query_alignment_requirements(nvcompBatchedLZ4CompressGetRequiredAlignments, format_opts),
          query_alignment_requirements(nvcompBatchedLZ4DecompressGetRequiredAlignments, decompress_opts)
        )
  {
    static_assert(offsetof(LZ4FormatSpecHeader, data_type) == offsetof(nvcompBatchedLZ4CompressOpts_t, data_type));
  }

  ~LZ4ManagerImpl() {}
};

extern template struct ManagerBase<
  LZ4FormatSpecHeader,
  decltype(nvcompBatchedLZ4DecompressAsyncEx) *,
  decltype(nvcompBatchedLZ4DecompressGetTempSizeAsync) *,
  decltype(nvcompBatchedLZ4GetDecompressSizeAsync) *,
  decltype(nvcompBatchedLZ4CompressAsync) *,
  decltype(nvcompBatchedLZ4CompressGetTempSizeAsync) *,
  decltype(nvcompBatchedLZ4CompressGetMaxOutputChunkSize) *,
  nvcompBatchedLZ4CompressOpts_t,
  nvcompBatchedLZ4DecompressOpts_t,
  nvcompFormatType_t::LZ4>;

LZ4Manager::LZ4Manager(
  size_t uncomp_chunk_size,
  const nvcompBatchedLZ4CompressOpts_t &format_opts,
  const nvcompBatchedLZ4DecompressOpts_t &decompress_opts,
  cudaStream_t user_stream,
  ChecksumPolicy checksum_policy,
  BitstreamKind bitstream_kind
)
{
  if (format_opts.bitshuffle_mode != decompress_opts.bitshuffle_mode)
  {
    throw NVCompException(
      nvcompErrorInvalidValue,
      "Bitshuffle mode is not consistent between compression and "
      "decompression options."
    );
  }
  if (format_opts.bitshuffle_mode != NVCOMP_BITSHUFFLE_NONE && decompress_opts.data_type != format_opts.data_type)
  {
    throw NVCompException(
      nvcompErrorInvalidValue,
      "Data type is not consistent between compression and "
      "decompression options when bitshuffle is enabled."
    );
  }
  impl = std::make_unique<LZ4ManagerImpl>(
    uncomp_chunk_size,
    format_opts,
    decompress_opts,
    user_stream,
    checksum_policy,
    bitstream_kind
  );
}

LZ4Manager::~LZ4Manager() {}

} // namespace nvcomp
