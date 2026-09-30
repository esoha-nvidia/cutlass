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
#include "CudaUtils.h"
#include "exception.hpp"
#include "lowlevel/Check.h"
#include "universal_header.cuh"

namespace nvcomp::cascaded::universal_header
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
    if (device_compressed_bytes[ix_chunk] < universal_header::HEADER_SIZE_BYTES)
    {
      // The compressed buffer should always have enough space for metadata. If
      // not, we report error.
      device_uncompressed_bytes[ix_chunk] = 0;
    }
    else
    {
      const uint8_t *comp_buffer = reinterpret_cast<const uint8_t *>(device_compressed_ptrs[ix_chunk]);
      device_uncompressed_bytes[ix_chunk] = !universal_header::is_legacy_compressed(comp_buffer) &&
                                                universal_header::has_supported_preamble(comp_buffer)
                                              ? universal_header::get_uncompressed_size(comp_buffer)
                                              : 0u;
    }
  }
}

// Initialize the decompression outputs of every partition to not
// decompressed.
__global__ void initialize_decompression_outputs_kernel(
  const size_t *device_compressed_bytes,
  size_t *device_uncompressed_bytes, ///< [out]
  nvcompStatus_t *device_statuses, ///< [out]
  const size_t num_chunks
)
{
  for (size_t ix_chunk = blockIdx.x * blockDim.x + threadIdx.x; ix_chunk < num_chunks;
       ix_chunk += gridDim.x * blockDim.x)
  {
    composite::set_empty_chunk_outputs(
      device_compressed_bytes[ix_chunk],
      device_uncompressed_bytes[ix_chunk],
      device_statuses[ix_chunk]
    );
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
  if (num_chunks == 0)
  {
    // An empty grid is an invalid launch configuration, and there are no sizes
    // to report either way.
    return nvcompSuccess;
  }

  try
  {
    constexpr int GET_DECOMP_THREADBLOCK_SIZE = 128;

    get_decompress_size_kernel<<<
      nvcomp::cuda_dim_cast(roundUpDiv(num_chunks, GET_DECOMP_THREADBLOCK_SIZE)),
      GET_DECOMP_THREADBLOCK_SIZE,
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

  if (num_chunks == 0)
  {
    // The batch size doubles as the grid size, and an empty grid is an invalid
    // launch configuration. There is nothing to decompress either way.
    return nvcompSuccess;
  }

  try
  {
    // Just call kernel to perform compression. Macro for datatype happens
    // within kernel
    constexpr int THREADBLOCK_SIZE = composite::composite_decompress_threadblock_size;

    // Report every partition as not decompressed up front, so that partitions
    // none of the launches below claims fail rather than being left
    // uninitialized.
    constexpr int INITIALIZE_OUTPUTS_THREADBLOCK_SIZE = 128;
    initialize_decompression_outputs_kernel<<<
      nvcomp::cuda_dim_cast(roundUpDiv(num_chunks, INITIALIZE_OUTPUTS_THREADBLOCK_SIZE)),
      INITIALIZE_OUTPUTS_THREADBLOCK_SIZE,
      0,
      stream>>>(device_compressed_chunk_bytes, device_uncompressed_chunk_bytes, device_statuses, num_chunks);
    CUDA_CHECK(cudaGetLastError());

    // call for all 4 possible sizes, all except the correct one will
    // immediately exit.

    // CHAR or UCHAR
    composite::
      type_checked_composite_decompression_kernel<1, size_t, THREADBLOCK_SIZE, /*SKIP_TYPE_MISMATCHED_BATCH=*/true>
      <<<nvcomp::cuda_dim_cast(num_chunks), THREADBLOCK_SIZE, 0, stream>>>(
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
    composite::
      type_checked_composite_decompression_kernel<2, size_t, THREADBLOCK_SIZE, /*SKIP_TYPE_MISMATCHED_BATCH=*/true>
      <<<nvcomp::cuda_dim_cast(num_chunks), THREADBLOCK_SIZE, 0, stream>>>(
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
    composite::
      type_checked_composite_decompression_kernel<4, size_t, THREADBLOCK_SIZE, /*SKIP_TYPE_MISMATCHED_BATCH=*/true>
      <<<nvcomp::cuda_dim_cast(num_chunks), THREADBLOCK_SIZE, 0, stream>>>(
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
    composite::
      type_checked_composite_decompression_kernel<8, size_t, THREADBLOCK_SIZE, /*SKIP_TYPE_MISMATCHED_BATCH=*/true>
      <<<nvcomp::cuda_dim_cast(num_chunks), THREADBLOCK_SIZE, 0, stream>>>(
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

} // namespace nvcomp::cascaded::universal_header
