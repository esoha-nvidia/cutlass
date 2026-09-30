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

#include "nvcomp/ans.h"

namespace ans
{

/**
 * @brief Get temporary space required for compression.
 *
 * @param batch_size The number of items in the batch.
 * @param max_chunk_size The maximum size of a chunk in the batch.
 * @param temp_bytes The size of the required GPU workspace for compression
 * (output).
 *
 */
void compressGetTempSize(
  size_t num_chunks,
  size_t max_uncompressed_chunk_size,
  nvcompBatchedANSCompressOpts_t format_opts,
  size_t *temp_bytes
);

/**
 * @brief Get the maximum size any chunk could compress to in the batch. That is, the minimum amount of output memory required to be given compressAsync() for each batch item.
 *
 * @param max_chunk_size The maximum size of a chunk in the batch.
 * @param max_compressed_size The maximum compressed size of the largest chunk (output).
 */
void compressGetMaxOutputChunkSize(size_t max_chunk_size, size_t *max_compressed_size);

/**
 * @brief Launch parameters for the in-kernel (device) char-ANS compressor.
 *        Block width is 256 threads so it can run in the CUTLASS 128x128 SIMT CTA.
 */
void compressGetDeviceLaunchParams(
  size_t num_chunks,
  size_t max_uncompressed_chunk_size,
  nvcompBatchedANSCompressOpts_t format_opts,
  cudaStream_t stream,
  int *max_sub_chunk_size,
  uint32_t *slot_words,
  size_t *smem_bytes,
  size_t *smem_alignment,
  int *block_threads
);

/**
 * @brief Perform compression.
 *
 * @param type The ANS compression algorithm type.
 * @param device_in_ptr The pointers on the GPU, to uncompressed batched items.
 * @param device_in_bytes The size of each uncompressed batch item on the GPU.
 * @param max_chunk_size The maximum size of a chunk.
 * @param batch_size The number of batch items.
 * @param temp_ptr The temporary GPU workspace.
 * @param temp_bytes The size of the temporary GPU workspace.
 * @param device_out_ptr The pointers on the GPU, to the output location for each compressed batch item (output).
 * Each pointer must be aligned to a 8-byte boundary.
 * @param device_out_bytes The compressed size of each chunk on the GPU (output).
 * @param stream The stream to operate on.
 */
void compressAsync(
  const void *const *device_in_ptr,
  const size_t *device_in_bytes,
  const size_t max_chunk_size,
  size_t batch_size,
  void *temp_ptr,
  size_t temp_bytes,
  void *const *device_out_ptr,
  size_t *device_out_bytes,
  nvcompBatchedANSCompressOpts_t format_opts,
  nvcompStatus_t *device_statuses,
  cudaStream_t stream
);

/**
 * @brief Get the amount of temp space required on the GPU for decompression.
 *
 * @param num_chunks The number of items in the batch.
 * @param max_uncompressed_chunk_size The size of the largest chunk when uncompressed.
 * @param temp_bytes The amount of temporary GPU space that will be required to
 * decompress.
 */
void decompressGetTempSize(size_t num_chunks, size_t max_uncompressed_chunk_size, size_t *temp_bytes);

/**
 * @brief Perform decompression.
 *
 * @param type The ANS compression algorithm type.
 * @param device_in_ptrs The pointers on the GPU, to the compressed chunks. Each compressed
 * chunk must be aligned to a 8-byte boundary.
 * @param device_in_bytes The size of each compressed chunk on the GPU.
 * @param device_out_bytes The size of each uncompressed chunk buffer on the GPU.
 * @param device_actual_out_bytes The return sizes of each uncompressed chunk on the GPU.
 * @param max_uncompressed_chunk_size The maximum size of an uncompressed chunk in the batch.
 * @param batch_size The number of batch items.
 * @param temp_ptr The temporary GPU space.
 * @param temp_bytes The size of the temporary GPU space.
 * @param device_out_ptrs The pointers on the GPU, to where to uncompress each chunk (output).
 * @param device_statuses The status of each chunk's decompression on the GPU (output).
 * @param max_sub_chunk_count The maximum number of sub-chunks per chunk.
 * @param data_type The ANS data type to decode (must match compression).
 * @param stream The stream to operate on.
 *
 */
void decompressAsync(
  const void *const *device_in_ptrs,
  const size_t *device_in_bytes,
  const size_t *device_out_bytes,
  size_t *device_actual_out_bytes,
  size_t batch_size,
  void *const temp_ptr,
  const size_t temp_bytes,
  void *const *device_out_ptrs,
  nvcompStatus_t *device_statuses,
  uint8_t max_sub_chunk_count,
  nvcompType_t data_type,
  cudaStream_t stream
);

/**
 * @brief Calculates the decompressed size of each chunk asynchronously. This is
 * needed when we do not know the expected output size.
 *
 * @param device_compressed_ptrs The compressed chunks of data. List of pointers
 * must be GPU accessible along with each chunk.
 * @param device_uncompressed_bytes The calculated decompressed size of each
 * chunk. Must be GPU accessible.
 * @param batch_size The number of chunks.
 * @param stream The CUDA stream to operate on.
 */
void getDecompressSizeAsync(
  const void *const *device_compressed_ptrs,
  const size_t *device_compressed_bytes,
  size_t *device_uncompressed_bytes,
  size_t batch_size,
  cudaStream_t stream
);
} // namespace ans
