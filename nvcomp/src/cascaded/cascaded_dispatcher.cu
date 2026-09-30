/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

// This is a separate file from cascaded_api.cpp, because with both CUDA and spdlog
// in the same file, nvcc sometimes uses too much memory for the
// build machines.

#include <cassert>

#include "composite/composite_comp_kernels.cuh"
#include "composite/composite_decomp_kernels.cuh"
#include "next/terminal_codecs/terminal_cascaded/terminal_cascaded_dispatcher.cuh"
#include "universal/universal_decomp.cuh"

namespace nvcomp
{
namespace
{

cascaded::next::kibi_cascader::kibi_cascader_optimization_search_space
get_terminal_cascaded_search_space(const nvcompBatchedCascadedCompressOpts_t &format_opts)
{
  auto search_space = cascaded::next::kibi_cascader::SEARCH_NONE;
  if ((format_opts.fine_grained_encoding_flags & NVCOMP_CASCADED_FINE_GRAINED_ENCODING_DELTA) != 0u)
  {
    search_space |= cascaded::next::kibi_cascader::SEARCH_DELTA;
  }
  if ((format_opts.fine_grained_encoding_flags & NVCOMP_CASCADED_FINE_GRAINED_ENCODING_FOR) != 0u)
  {
    search_space |= cascaded::next::kibi_cascader::SEARCH_FOR;
  }
  if ((format_opts.fine_grained_encoding_flags & NVCOMP_CASCADED_FINE_GRAINED_ENCODING_RLE) != 0u)
  {
    search_space |= cascaded::next::kibi_cascader::SEARCH_RLE;
  }
  return search_space;
}

} // namespace

nvcompStatus_t cascadedCompressGetMaxOutputChunkSizeDispatch(
  const size_t max_uncompressed_chunk_bytes,
  const nvcompBatchedCascadedCompressOpts_t format_opts,
  size_t *max_compressed_chunk_bytes
)
{
  if (format_opts.common_opts.mode == NVCOMP_CASCADED_MODE_SYMMETRIC)
  {
    return cascaded::next::terminal_cascaded::get_max_output_chunk_size(
      max_uncompressed_chunk_bytes,
      format_opts.common_opts.data_type,
      max_compressed_chunk_bytes
    );
  }
  else if (format_opts.common_opts.mode == NVCOMP_CASCADED_MODE_ASYMMETRIC)
  {
    // Asymmetric compression may store the input verbatim. Reserve the universal
    // header plus enough space for the uncompressed payload rounded up to a 32-bit word.
    *max_compressed_chunk_bytes = roundUpTo(max_uncompressed_chunk_bytes, sizeof(uint32_t)) +
                                  cascaded::universal_header::HEADER_SIZE_BYTES;
    return nvcompSuccess;
  }
  else
  {
    assert(false && "Unsupported Cascaded compression mode.");
    return nvcompErrorInvalidValue;
  }
}

nvcompStatus_t cascadedCompressAsyncDispatch(
  const void *const *device_uncompressed_chunk_ptrs,
  const size_t *device_uncompressed_chunk_bytes,
  size_t num_chunks,
  void *const *device_compressed_chunk_ptrs,
  size_t *device_compressed_chunk_bytes,
  const nvcompBatchedCascadedCompressOpts_t format_opts,
  nvcompStatus_t *device_statuses,
  cudaStream_t stream
)
{
  if (num_chunks == 0)
  {
    return nvcompSuccess;
  }
  if (format_opts.common_opts.mode == NVCOMP_CASCADED_MODE_SYMMETRIC)
  {
    return cascaded::next::terminal_cascaded::compress_async(
      device_uncompressed_chunk_ptrs,
      device_uncompressed_chunk_bytes,
      nvcompCascadedCompressionMaxAllowedChunkSize,
      num_chunks,
      nullptr,
      0u,
      device_compressed_chunk_ptrs,
      device_compressed_chunk_bytes,
      format_opts.common_opts.data_type,
      get_terminal_cascaded_search_space(format_opts),
      device_statuses,
      stream
    );
  }
  else if (format_opts.common_opts.mode == NVCOMP_CASCADED_MODE_ASYMMETRIC)
  {
    return cascaded::composite::compressAsync(
      device_uncompressed_chunk_ptrs,
      device_uncompressed_chunk_bytes,
      num_chunks,
      device_compressed_chunk_ptrs,
      device_compressed_chunk_bytes,
      format_opts,
      device_statuses,
      stream
    );
  }
  else
  {
    assert(false && "Unsupported Cascaded compression mode.");
    return nvcompErrorInvalidValue;
  }
}

nvcompStatus_t cascadedDecompressAsyncDispatch(
  const void *const *device_compressed_chunk_ptrs,
  const size_t *device_compressed_chunk_bytes,
  const size_t *device_uncompressed_buffer_bytes,
  size_t *device_uncompressed_chunk_bytes,
  size_t num_chunks,
  void *const *device_uncompressed_chunk_ptrs,
  nvcompStatus_t *device_statuses,
  cudaStream_t stream
)
{
  if (num_chunks == 0)
  {
    return nvcompSuccess;
  }
  const nvcompStatus_t universal_status = cascaded::universal_header::DecompressAsync(
    device_compressed_chunk_ptrs,
    device_compressed_chunk_bytes,
    device_uncompressed_buffer_bytes,
    device_uncompressed_chunk_bytes,
    num_chunks,
    device_uncompressed_chunk_ptrs,
    device_statuses,
    stream
  );
  if (universal_status != nvcompSuccess)
  {
    return universal_status;
  }
  return cascaded::next::terminal_cascaded::decompress_self_dispatch_async<false>(
    device_compressed_chunk_ptrs,
    device_compressed_chunk_bytes,
    device_uncompressed_buffer_bytes,
    device_uncompressed_chunk_bytes,
    num_chunks,
    device_uncompressed_chunk_ptrs,
    NVCOMP_TYPE_BITS,
    device_statuses,
    stream
  );
}

nvcompStatus_t cascadedDecompressAsyncDispatch(
  const void *const *device_compressed_chunk_ptrs,
  const size_t *device_compressed_chunk_bytes,
  const size_t *device_uncompressed_buffer_bytes,
  size_t *device_uncompressed_chunk_bytes,
  size_t num_chunks,
  void *const *device_uncompressed_chunk_ptrs,
  nvcompStatus_t *device_statuses,
  nvcompCascadedCommonOpts_t common_opts,
  cudaStream_t stream
)
{
  if (num_chunks == 0)
  {
    return nvcompSuccess;
  }
  if (common_opts.mode == NVCOMP_CASCADED_MODE_SYMMETRIC)
  {
    return cascaded::next::terminal_cascaded::decompress_self_dispatch_async(
      device_compressed_chunk_ptrs,
      device_compressed_chunk_bytes,
      device_uncompressed_buffer_bytes,
      device_uncompressed_chunk_bytes,
      num_chunks,
      device_uncompressed_chunk_ptrs,
      common_opts.data_type,
      device_statuses,
      stream
    );
  }
  return cascaded::composite::DecompressAsync(
    device_compressed_chunk_ptrs,
    device_compressed_chunk_bytes,
    device_uncompressed_buffer_bytes,
    device_uncompressed_chunk_bytes,
    num_chunks,
    device_uncompressed_chunk_ptrs,
    device_statuses,
    common_opts.data_type,
    stream
  );
}

nvcompStatus_t cascadedGetDecompressSizeAsyncDispatch(
  const void *const *device_compressed_chunk_ptrs,
  const size_t *device_compressed_chunk_bytes,
  size_t *device_uncompressed_chunk_bytes,
  size_t num_chunks,
  cudaStream_t stream
)
{
  // comp opts / decomp opts not provided
  // All configurations must support the legacy GetDecompressSizeAsync function
  return cascaded::universal_header::GetDecompressSizeAsync(
    device_compressed_chunk_ptrs,
    device_compressed_chunk_bytes,
    device_uncompressed_chunk_bytes,
    num_chunks,
    stream
  );
}

} // namespace nvcomp
