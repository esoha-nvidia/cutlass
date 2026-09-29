/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#include "CorrectnessChecks.cuh"
#include "gdeflate/gdeflate_constants.h"
#include "HWDecompress.hpp"
#include "nvcomp/deflate.h"
#include "nvcomp/gzip.h"
#include "nvcomp/shared_types.h"

namespace nvcomp_deflate
{

/**
 * @brief Get the maximum size a chunk could compress to in the batch. That is, the minimum amount of output memory required to be given compressAsync() for this particular chunk.
 *
 * @param decomp_chunk_size The size of the chunk to be compressed.
 * @param max_compressed_size The maximum compressed size of the largest chunk (output).
 *
 */
inline __host__ __device__ void
DeflateCompressGetMaxOutputChunkSize(size_t decomp_chunk_size, size_t *max_compressed_size)
{

#ifdef __CUDA_ARCH__
  assert(decomp_chunk_size <= gdeflate::gdeflateMaxChunkSize);
#else
  if (decomp_chunk_size > gdeflate::gdeflateMaxChunkSize)
  {
    throw nvcomp::NVCompException(
      nvcompErrorInvalidValue,
      std::string("Maximum allowed chunk size for Deflate is ") + gdeflate::gdeflateMaxChunkSizeText
    );
  }
#endif // __CUDA_ARCH__
  // TODO: reduce temp_factor to make upper bound tighter
  constexpr float temp_factor = 2.f;

  /* upper bound for fixed blocks with 9-bit literals and length 255
    (memLevel == 2, which is the lowest that may not use stored blocks) --
    ~13% overhead plus a small constant */
  size_t fixedlen = decomp_chunk_size + (decomp_chunk_size >> 3) + (decomp_chunk_size >> 8) + (decomp_chunk_size >> 9) +
                    4;

  /* upper bound for stored blocks with length 127 (memLevel == 1) --
      ~4% overhead plus a small constant */
  size_t storelen = decomp_chunk_size + (decomp_chunk_size >> 5) + (decomp_chunk_size >> 7) +
                    (decomp_chunk_size >> 11) + 7;

  size_t max_overhead = std::max(fixedlen, storelen);

  // Calculate the rounded maximum compressed size
  size_t non_aligned_max_compressed_size = static_cast<size_t>(ceilf(temp_factor * max_overhead + 18));
  size_t rounded_max_compressed_size = nvcomp::roundUpTo(non_aligned_max_compressed_size, sizeof(uint64_t));

  *max_compressed_size = std::max(rounded_max_compressed_size, size_t(64));
}

typedef enum
{
  deflateSuccess = 0,
  deflateErrorInvalidValue = 10,
  deflateErrorNotSupported = 11,
  deflateErrorCannotDecompress = 12,
  deflateErrorWrongHeaderLength = 13,
  deflateErrorOutputBufferTooSmall = 14,
  deflateErrorCudaError = 1000,
  deflateErrorInternal = 10000,
} deflateStatus_t;

void DeflateFillCuDecompParams(
  CUmemDecompressParams *params,
  const void *const *device_compressed_chunk_ptrs,
  const size_t *device_compressed_chunk_bytes,
  size_t *device_uncompressed_chunk_bytes,
  const size_t *device_uncompressed_buffer_bytes,
  void *const *device_uncompressed_chunk_ptrs,
  bool do_gzip_header_parse,
  const size_t num_chunks,
  nvcompStatus_t *device_statuses,
  uint8_t *scratch_allocation,
  size_t scratch_size,
  bool use_sorting,
  cudaStream_t stream
);

template <bool CORRECTNESS_CHECK>
cudaError_t __host__ DeflateDecompressAsync(
  const void *const *device_compressed_ptrs,
  const size_t *device_compressed_bytes,
  void *const *device_uncompressed_ptrs,
  const size_t *device_uncompressed_bytes,
  size_t *device_actual_uncompressed_bytes,
  deflateStatus_t *device_statuses,
  void *device_correctness_ptrs,
  size_t batch_size,
  cudaStream_t stream,
  bool gzip_header_parser = false
);

nvcompStatus_t DetailDeflateDecompress(
  const void *const *device_compressed_chunk_ptrs,
  const size_t *device_compressed_chunk_bytes,
  const size_t *device_uncompressed_buffer_bytes,
  size_t *device_uncompressed_chunk_bytes,
  size_t num_chunks,
  void *const device_temp_ptr,
  size_t temp_bytes,
  void *const *device_uncompressed_chunk_ptrs,
  nvcompStatus_t *device_statuses,
  nvcompBatchedDeflateDecompressOpts_t *deflate_decompress_opts,
  nvcompBatchedGzipDecompressOpts_t *gzip_decompress_opts,
  bool gzip,
  cudaStream_t stream,
  const void *const *host_comp_chunk_buffers = nullptr
);

cudaError_t __host__ DeflateDecompressSizeAsync(
  const void *const *device_compressed_ptrs,
  const size_t *device_compressed_bytes,
  void *const *device_uncompressed_ptrs,
  const size_t *device_uncompressed_bytes,
  size_t *device_actual_uncompressed_bytes,
  deflateStatus_t *device_statuses,
  size_t batch_size,
  cudaStream_t stream,
  bool gzip_header_parser = false
);

} // namespace nvcomp_deflate
