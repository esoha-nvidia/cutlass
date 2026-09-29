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

#include <cuda_runtime.h>

#include <cstdint>

#include "nvcomp/shared_types.h"

namespace gdeflate
{

typedef enum
{
  HIGH_COMPRESSION,
  HIGH_THROUGHPUT,
  ENTROPY_ONLY,
  MEDIUM_COMPRESSION,
  L6_OP_COMPRESSION
} gdeflate_compression_algo;

typedef enum
{
  gdeflateSuccess = 0,
  gdeflateErrorInvalidValue = 10,
  gdeflateErrorNotSupported = 11,
  gdeflateErrorCannotDecompress = 12,
  gdeflateErrorCudaError = 1000,
  gdeflateErrorInternal = 10000,
} gdeflateStatus_t;

// Maps the public compress-opts algorithm selector (0-5) to the internal
// gdeflate compression algorithm. Throws NVCompException on an invalid value.
gdeflate_compression_algo getCompressionAlgo(int algorithm);

/**
 * @brief Get the amount of temp space required on the GPU for decompression.
 *
 * @param num_chunks The number of items in the batch.
 * @param max_uncompressed_chunk_size The size of the largest chunk when uncompressed.
 * @param temp_bytes The amount of temporary GPU space that will be required to
 * decompress.
 *
 */
void decompressGetTempSize(size_t num_chunks, size_t max_uncompressed_chunk_size, size_t *temp_bytes);

/**
 * @brief Perform decompression.
 *
 * @param device_compressed_ptr The pointers on the GPU, to the compressed chunks.
 * @param device_compressed_bytes The size of each compressed chunk on the GPU.
 * @param device_uncompressed_bytes The size of each uncompressed chunk on the GPU (max available space).
 * @param device_actual_uncompressed_bytes Actual bytes of uncompressed chunk data
 * @param batch_size The number of batch items.
 * @param device_uncompressed_ptr The pointers on the GPU, to where to uncompress each chunk (output).
 * @param device_statuses Pointer to per chunk decompression status (success or failure)
 * @param stream The stream to operate on.
 *
 */
void decompressAsync(
  const void *const *device_compressed_ptr,
  const size_t *device_compressed_bytes,
  const size_t *device_uncompressed_bytes,
  size_t *device_actual_uncompressed_bytes,
  size_t batch_size,
  void *const *device_uncompressed_ptr,
  gdeflateStatus_t *device_statuses,
  cudaStream_t stream
);

/**
 * @brief Calculates the decompressed size of each chunk asynchronously. This is
 * needed when we do not know the expected output size. All pointers must be GPU
 * accessible. Note, if the stream is corrupt, the sizes will be garbage.
 *
 * @param device_compressed_ptr The compressed chunks of data. List of pointers
 * must be GPU accessible along with each chunk.
 * @param device_compressed_bytes The size of each compressed chunk. Must be GPU
 * accessible.
 * @param device_actual_uncompressed_bytes The calculated decompressed size of each
 * chunk. Must be GPU accessible.
 * @param batch_size The number of chunks
 * @param stream The stream to operate on.
 *
 */
void getDecompressSizeAsync(
  const void *const *device_compressed_ptr,
  const size_t *device_compressed_bytes,
  size_t *device_actual_uncompressed_bytes,
  size_t batch_size,
  cudaStream_t stream
);

/**
 * @brief Get temporary space required for compression.
 *
 * @param batch_size The number of items in the batch.
 * @param max_chunk_size The maximum size of a chunk in the batch.
 * @param temp_bytes The size of the required GPU workspace for compression
 * (output).
 * @param algo Algorithm to use for compression
 *
 */
void compressGetTempSize(size_t batch_size, size_t max_chunk_size, size_t *temp_bytes, gdeflate_compression_algo algo);

/**
 * @brief Get the maximum size any chunk could compress to in the batch. That is, the minimum amount of output memory required to be given compressAsync() for each batch item.
 *
 * @param max_chunk_size The maximum size of a chunk in the batch.
 * @param max_compressed_size The maximum compressed size of the largest chunk (output).
 *
 */
void compressGetMaxOutputChunkSize(size_t max_chunk_size, size_t *max_compressed_size);

/**
 * @brief Perform compression.
 *
 * @param device_uncompressed_ptr The pointers on the GPU, to uncompressed batched items.
 * @param device_uncompressed_bytes The size of each uncompressed batch item on the GPU.
 * @param max_chunk_size The maximum size of a chunk.
 * @param batch_size The number of batch items.
 * @param temp_ptr The temporary GPU workspace.
 * @param temp_bytes The size of the temporary GPU workspace.
 * @param device_compressed_ptr The pointers on the GPU, to the output location for each compressed batch item (output).
 * @param device_compressed_bytes The compressed size of each chunk on the GPU (output).
 * @param algo Algorithm to use for compression
 * @param device_statuses The status of each compression batch item on the GPU.
 * @param stream The stream to operate on.
 * @param standard true for deflate, false for gdeflate
 */
void compressAsync(
  const void *const *device_uncompressed_ptr,
  const size_t *device_uncompressed_bytes,
  const size_t max_chunk_size,
  size_t batch_size,
  void *temp_ptr,
  size_t temp_bytes,
  void *const *device_compressed_ptr,
  size_t *device_compressed_bytes,
  gdeflate_compression_algo algo,
  nvcompStatus_t *device_statuses,
  cudaStream_t stream,
  bool standard = false,
  uint8_t *device_compressed_pad_bits = nullptr,
  // Optional device pointer to the actual batch_size to use for runtime bounds.
  // When non-null, `batch_size` (host) is used only for grid sizing & temp space,
  // and kernels read this device-side value as the actual count.
  const int *device_num_deflate_blocks = nullptr
);

} // namespace gdeflate
