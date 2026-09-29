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

#pragma once

#include "cascaded/composite/composite_decomp_kernels.cuh"
#include "common.h"
#include "CudaUtils.h"
#include "exception.hpp"
#include "lowlevel/Check.h"
#include "nvcomp.h"
#include "nvcomp/cascaded.h"
#include "type_macros.h"
#include "universal_header.cuh"

using nvcomp::Check;
using nvcomp::CudaUtils;
using nvcomp::isAligned;
using nvcomp::roundUpDiv;
using nvcomp::roundUpTo;
using nvcomp::roundUpToAlignment;

namespace universal_header
{

__global__ void get_decompress_size_kernel(
  const void *const *device_compressed_ptrs,
  const size_t *device_compressed_bytes,
  size_t *device_uncompressed_bytes,
  size_t num_chunks
)
{
  for (size_t ix_chunk = blockIdx.x * blockDim.x + threadIdx.x; ix_chunk < num_chunks;
       ix_chunk += gridDim.x * blockDim.x)
  {
    if (device_compressed_bytes[ix_chunk] < universal_header::header_size_bytes)
    {
      // The compressed buffer should always have enough space for metadata. If
      // not, we report error.
      device_uncompressed_bytes[ix_chunk] = 0;
    }
    else
    {
      const uint8_t *comp_buffer = reinterpret_cast<const uint8_t *>(device_compressed_ptrs[ix_chunk]);
      device_uncompressed_bytes[ix_chunk] = universal_header::get_uncompressed_size(comp_buffer);
    }
  }
}

inline nvcompStatus_t GetDecompressSizeAsync(
  const void *const *device_compressed_chunk_ptrs,
  const size_t *device_compressed_chunk_bytes,
  size_t *device_uncompressed_chunk_bytes,
  size_t num_chunks,
  cudaStream_t stream
)
{
  try
  {
    constexpr int universal_get_decomp_size_threadblock_size = 128;

    get_decompress_size_kernel<<<
      nvcomp::cuda_dim_cast(roundUpDiv(num_chunks, universal_get_decomp_size_threadblock_size)),
      universal_get_decomp_size_threadblock_size,
      0,
      stream>>>(
      device_compressed_chunk_ptrs,
      device_compressed_chunk_bytes,
      device_uncompressed_chunk_bytes,
      num_chunks
    );
    CUDA_CHECK(cudaGetLastError());
  }
  catch (const std::exception &e)
  {
    LOG_ERROR("{}", e.what());
    return Check::exception_to_error(e, "nvcompBatchedCascadedGetDecompressSizeAsync()");
  }

  return nvcompSuccess;
}

nvcompStatus_t DecompressAsync(
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
  // Compression settings are not provided, so we have to use trial & error
  // Current strategy: launch kernel for each compression configuration that
  // MIGHT be in the current batch

  // TODO: pass decomp options into this function (already given in API)
  // TODO: let users prune the number of kernel calls down to what is necessary

  try
  {
    // Just call kernel to perform compression. Macro for datatype happens
    // within kernel
    constexpr int threadblock_size = composite::composite_decompress_threadblock_size;

    // call for all 4 possible sizes, all except the correct one will
    // immediately exit.

    // CHAR or UCHAR
    composite::type_checked_composite_decompression_kernel<1, size_t, threadblock_size>
      <<<nvcomp::cuda_dim_cast(num_chunks), threadblock_size, 0, stream>>>(
        nvcomp::narrow_cast<int>(num_chunks),
        device_compressed_chunk_ptrs,
        device_compressed_chunk_bytes,
        device_uncompressed_chunk_ptrs,
        device_uncompressed_buffer_bytes,
        device_uncompressed_chunk_bytes,
        device_statuses
      );
    CUDA_CHECK(cudaGetLastError());

    // SHORT or USHORT
    composite::type_checked_composite_decompression_kernel<2, size_t, threadblock_size>
      <<<nvcomp::cuda_dim_cast(num_chunks), threadblock_size, 0, stream>>>(
        nvcomp::narrow_cast<int>(num_chunks),
        device_compressed_chunk_ptrs,
        device_compressed_chunk_bytes,
        device_uncompressed_chunk_ptrs,
        device_uncompressed_buffer_bytes,
        device_uncompressed_chunk_bytes,
        device_statuses
      );
    CUDA_CHECK(cudaGetLastError());

    // INT or UINT
    composite::type_checked_composite_decompression_kernel<4, size_t, threadblock_size>
      <<<nvcomp::cuda_dim_cast(num_chunks), threadblock_size, 0, stream>>>(
        nvcomp::narrow_cast<int>(num_chunks),
        device_compressed_chunk_ptrs,
        device_compressed_chunk_bytes,
        device_uncompressed_chunk_ptrs,
        device_uncompressed_buffer_bytes,
        device_uncompressed_chunk_bytes,
        device_statuses
      );
    CUDA_CHECK(cudaGetLastError());

    // LONGLONG or ULONGLONG
    composite::type_checked_composite_decompression_kernel<8, size_t, threadblock_size>
      <<<nvcomp::cuda_dim_cast(num_chunks), threadblock_size, 0, stream>>>(
        nvcomp::narrow_cast<int>(num_chunks),
        device_compressed_chunk_ptrs,
        device_compressed_chunk_bytes,
        device_uncompressed_chunk_ptrs,
        device_uncompressed_buffer_bytes,
        device_uncompressed_chunk_bytes,
        device_statuses
      );
    CUDA_CHECK(cudaGetLastError());
  }
  catch (const std::exception &e)
  {
    LOG_ERROR("{}", e.what());
    return Check::exception_to_error(e, "nvcompBatchedCascadedDecompressAsync()");
  }

  return nvcompSuccess;
}

} // namespace universal_header
