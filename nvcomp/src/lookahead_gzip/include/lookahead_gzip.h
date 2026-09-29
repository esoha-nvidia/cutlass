/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <fstream>

#include "nvcomp/shared_types.h"

/**
 *  NOTE: This is an internal C API for testing the lookahead_gzip algorithm. This will not be exposed to the end-user.
 *        Internally, one can chose between using this C API, or the C++ API via the LookaheadGzip<XXX>Client classes.
 */

// Minimum number of CTAs that are needed to decompress a single deflate block.
constexpr size_t LOOKAHEAD_GZIP_MIN_CTA_AMOUNT = 7;

// Minimum size of chunk in bytes that will trigger the Lookahead Gzip algorithm.
constexpr size_t LOOKAHEAD_GZIP_MIN_CHUNK_SIZE = (1 << 16) + 1;

/**
 * @brief LookaheadGzip decompression config for the low-level API.
 */
typedef struct
{
  /**
     *  @brief CUDA stream for input prefetching. Must be different from
     *         the decompression stream.
     */
  cudaStream_t prefetch_input_stream;
  /**
     *  @brief CUDA stream for output prefetching. Must be different from
     *         the decompression stream.
     */
  cudaStream_t prefetch_output_stream;
  /**
     *  @brief Number of SMs used for decompression.
     */
  int SM_count;
  /**
     *  @brief The total number of chunks to be processed. Some of them will be processed in parallel.
     */
  int num_chunks;
  /**
     *  @brief GPU device id
     */
  int device_id;
} lookaheadGzipConfig_t;

/**
 * @brief Get alignment requirements for lookahead gzip decompression.
 *
 *  @param[out] alignment_requirements Struct to be filled with alignment requirements.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompStatus_t nvcompLookaheadGzipDecompressGetRequiredAlignments(nvcompAlignmentRequirements_t *alignment_requirements);

/**
 * @brief Compute the number of bytes to allocate for decompression workspace.
 *
 *  @param[in] num_chunks Number of chunks
 *  @param[out] temp_bytes Number of bytes needed for scratch space.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
nvcompStatus_t nvcompLookaheadGzipDecompressGetTempSize(size_t num_chunks, size_t *temp_bytes);

/**
 * @brief Compute the number of bytes of uncompressed data
 *
 * This is needed when we do not know the expected output size.
 * Internally it fetches the ISIZE GZip footer. Note that any file
 * bigger than 2^32 bytes (4 GiB) will have this value as modulo 2^32.
 *  @param[in] device_compressed_ptrs Array (in device memory) of pointers to compressed data chunks in device memory.
 *  @param[in] device_compressed_bytes Array (in device memory) of compressed chunk sizes.
 *  @param[out] device_uncompressed_chunk_bytes Array (in device memory) to be filled with uncompressed sizes.
 *  @param[in] num_chunks Number of chunks.
 *  @param[out] device_statuses Array of length \p num_chunks in device memory for per-chunk status.
 *  Set to nvcompSuccess on success, or nvcompErrorNotSupported if the chunk lacks a valid gzip header.
 *  Can be nullptr if not needed.
 *  @param[in] stream The CUDA stream to operate on.
 *
 * @return nvcompSuccess if successfully launched, and an error code otherwise.
 */
nvcompStatus_t nvcompLookaheadGzipGetDecompressSizeAsync(
  const void *const *device_compressed_ptrs,
  const size_t *device_compressed_bytes,
  size_t *device_uncompressed_chunk_bytes,
  size_t num_chunks,
  nvcompStatus_t *device_statuses,
  cudaStream_t stream
);

/**
 * @brief Perform asynchronous decompression.
 *
 *  @param[in] device_compressed_ptrs Pointer to device memory allocated with compressed data for each chunk.
 *  @param[in] device_compressed_bytes Expected amount of bytes of compressed data.
 *  @param[out] device_uncompressed_ptrs Pointer to device memory allocated to store decompressed data for each chunk.
 *  @param[in] device_uncompressed_buffer_bytes Number of bytes allocated for holding each uncompressed chunk.
 *  @param[out] device_uncompressed_chunk_bytes Array of length \p num_chunks in device memory where the actual
 *  uncompressed size for each chunk will be written. Can be nullptr if not needed.
 *  @param[in] device_temp_ptr Pointer to device memory allocated to store scratch data.
 *  @param[in] temp_bytes Size of internal buffer for scratch space.
 *  @param[in] num_chunks Number of chunks to decompress.
 *  @param[out] device_status Array of length \p num_chunks in preallocated device memory for per-chunk
 *  status values. On successful decompression the entry is set to `nvcompSuccess`.
 *  On detected errors (e.g., invalid block header, output buffer too small) the entry may be set to
 *  an error code such as `nvcompErrorCannotDecompress` or `nvcompErrorOutputBufferTooSmall`.
 *  Not all corruption is guaranteed to be detected.
 *  @param[in] stream The CUDA stream to operate on.
 *
 * @return nvcompSuccess if successfully launched, and an error code otherwise.
 */
nvcompStatus_t nvcompLookaheadGzipDecompressAsync(
  const void *const *device_compressed_ptrs,
  const size_t *device_compressed_bytes,
  void *const *device_uncompressed_ptrs,
  const size_t *device_uncompressed_buffer_bytes,
  size_t *device_uncompressed_chunk_bytes,
  void *const device_temp_ptr,
  const size_t temp_bytes,
  const size_t num_chunks,
  nvcompStatus_t *device_status,
  cudaStream_t stream
);
