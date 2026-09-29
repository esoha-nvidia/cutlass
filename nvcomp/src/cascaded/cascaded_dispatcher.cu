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

#include "composite/composite_comp_kernels.cuh"
#include "composite/composite_decomp_kernels.cuh"
#include "universal/universal_decomp.cuh"

namespace nvcomp
{
nvcompStatus_t cascadedCompressAsyncPart2(
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
  return composite::compressAsync(
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

nvcompStatus_t cascadedDecompressAsyncPart2(
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
  // Compression settings are not provided, so we must use the universal decomp pipeline
  return universal_header::DecompressAsync(
    device_compressed_chunk_ptrs,
    device_compressed_chunk_bytes,
    device_uncompressed_buffer_bytes,
    device_uncompressed_chunk_bytes,
    num_chunks,
    device_uncompressed_chunk_ptrs,
    device_statuses,
    stream
  );
}

nvcompStatus_t cascadedDecompressAsyncPart2(
  const void *const *device_compressed_chunk_ptrs,
  const size_t *device_compressed_chunk_bytes,
  const size_t *device_uncompressed_buffer_bytes,
  size_t *device_uncompressed_chunk_bytes,
  size_t num_chunks,
  void *const *device_uncompressed_chunk_ptrs,
  nvcompStatus_t *device_statuses,
  nvcompType_t type,
  cudaStream_t stream
)
{
  return composite::DecompressAsync(
    device_compressed_chunk_ptrs,
    device_compressed_chunk_bytes,
    device_uncompressed_buffer_bytes,
    device_uncompressed_chunk_bytes,
    num_chunks,
    device_uncompressed_chunk_ptrs,
    device_statuses,
    type,
    stream
  );
}

nvcompStatus_t cascadedGetDecompressSizeAsyncPart2(
  const void *const *device_compressed_chunk_ptrs,
  const size_t *device_compressed_chunk_bytes,
  size_t *device_uncompressed_chunk_bytes,
  size_t num_chunks,
  cudaStream_t stream
)
{
  // comp opts / decomp opts not provided
  // All configurations must support the legacy GetDecompressSizeAsync function
  return universal_header::GetDecompressSizeAsync(
    device_compressed_chunk_ptrs,
    device_compressed_chunk_bytes,
    device_uncompressed_chunk_bytes,
    num_chunks,
    stream
  );
}

} // namespace nvcomp
