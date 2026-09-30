/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
*/

#ifndef NVCOMP_ANS_H
#define NVCOMP_ANS_H

#include "nvcomp.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Available ANS types
 */
typedef enum nvcompANSType_t
{
  nvcomp_rANS,
} nvcompANSType_t;

/**
 * @brief ANS compression options for the low-level API
 */
typedef struct
{
  /**
   * @brief ANS algorithm to use.
   */
  nvcompANSType_t type;
  /**
   * @brief ANS data type to use.
   *
   * - NVCOMP_TYPE_(U)CHAR: 1-byte, generic data type
   * - NVCOMP_TYPE_FLOAT16: 2-byte floating-point data type. Applicable to all half-precision data formats.
   * - NVCOMP_TYPE_FLOAT8_E4M3: 1-byte FP8 (E4M3) floating-point data type.
   * - NVCOMP_TYPE_FLOAT32: 4-byte IEEE-754 single-precision floating-point data type.
   */
  nvcompType_t data_type;
  /**
   * @brief Maximum sub chunk count override for compression.
   * 0: default of 8. Nonzero must be a power-of-2 between 4 and 64.
   * Leave zero unless you can tune the performance of your e2e application based on this value.
   */
  uint8_t max_sub_chunk_count;
  /**
   * @brief Number of interleaved rANS states per lane.
   * 0: auto (default): two interleaved streams for every data type.
   * 1: single stream.
   * 2: two interleaved streams.
   * "2 states" improves performance but adds an overhead of 128B per sub chunk, impacting
   * compression ratio; the relative cost grows as the sub chunk shrinks.
   */
  uint8_t states_per_lane;
  /**
   * @brief Reduces the amount of data used to build the histogram by this log2 factor.
   * 0: exact model (default). N keeps 1/2^N of the slice (1 = 1/2, 2 = 1/4, 3 = 1/8,
   * up to nvcompANSMaxHistogramReductionLog2). Useful to speed up compression when
   * full chunks are known to be sampled from a common distribution.
   * Applies to every data type. Chunks below 4096 ANS symbols are always histogrammed
   * exactly, whatever the requested reduction.
   */
  uint8_t histogram_reduction_log2;
  /**
   * @brief These bytes are unused and must be zeroed. This ensures
   *        compatibility if additional fields are added in the future.
   */
  char reserved[53];
} nvcompBatchedANSCompressOpts_t;

/**
 * @brief ANS decompression options for the low-level API
 */
typedef struct
{
  /**
   * @brief Decompression backend to use.
   */
  nvcompDecompressBackend_t backend;
  /**
   * @brief ANS data type to use for decompression.
   *
   * - NVCOMP_TYPE_BITS: unknown (the default). The type is read from the bitstream, so any
   *   ANS stream decodes; naming a type instead must match the bitstream.
   * - NVCOMP_TYPE_(U)CHAR: 1-byte, generic data type
   * - NVCOMP_TYPE_FLOAT16: 2-byte floating-point data type. Applicable to all half-precision data formats.
   * - NVCOMP_TYPE_FLOAT8_E4M3: 1-byte FP8 (E4M3) floating-point data type.
   * - NVCOMP_TYPE_FLOAT32: 4-byte IEEE-754 single-precision floating-point data type.
   */
  nvcompType_t data_type;
  /**
   * @brief Maximum sub chunk count override for decompression.
   * 0: infer the value from the compressed bitstream.
   * Nonzero must be a power-of-2 between 4 and 64; smaller than the stream's
   * count fails decompression.
   */
  uint8_t max_sub_chunk_count;
  /**
   * @brief Number of interleaved rANS states per lane used during compression.
   * 0: infer the value from the compressed bitstream. Otherwise it must match the bitstream,
   *    and setting it lets a more specialized kernel be launched once data_type names a type.
   * 1: single stream.
   * 2: two interleaved streams.
   */
  uint8_t states_per_lane;

  /**
   * @brief Skip per-chunk validation in the decompress kernel.
   */
  uint8_t skip_validate;
  /**
   * @brief These bytes are unused and must be zeroed. This ensures
   *        compatibility if additional fields are added in the future.
   */
  char reserved[53];
} nvcompBatchedANSDecompressOpts_t;

/**
 * @brief Default ANS compression options
 */
static const nvcompBatchedANSCompressOpts_t nvcompBatchedANSCompressDefaultOpts =
  {nvcomp_rANS, NVCOMP_TYPE_CHAR, 0, 0, 0, {0}};

/**
 * @brief Default ANS decompression options
 */
static const nvcompBatchedANSDecompressOpts_t nvcompBatchedANSDecompressDefaultOpts =
  {NVCOMP_DECOMPRESS_BACKEND_DEFAULT, NVCOMP_TYPE_BITS, 0, 0, 0, {0}};

/**
 * @brief The maximum supported uncompressed chunk size in bytes for the ANS compressor.
 */
static const size_t nvcompANSCompressionMaxAllowedChunkSize = 1 << 24;

/**
 * @brief The maximum supported compressed and decompressed chunk size in bytes for the ANS decompressor.
 * @note To maximize decompression performance, users are encouraged to compress in smaller chunks, for example 64KiB.
 */
static const size_t nvcompANSDecompressionMaxAllowedChunkSize = 1ull << 25;

/**
 * @brief Maximum `histogram_reduction_log2` for ANS compression. N keeps 1/2^N of each
 * warp's histogram slice, so this cap is 1/256.
 */
static const uint8_t nvcompANSMaxHistogramReductionLog2 = 8;

/**
 * @brief The most restrictive of the minimum alignment requirements for void-type CUDA memory buffers
 * used for input, output, or temporary memory, passed to compression functions.
 *
 * @note In all cases, typed memory buffers must still be aligned to their type's size,
 * e.g., 4 bytes for `int`.
 */
static const size_t nvcompANSRequiredCompressionAlignment = 16;

/**
 * @brief Get the minimum buffer alignment requirements for compression.
 *
 * @note Providing buffers with alignments above the minimum requirements
 * (e.g., 16- or 32-byte alignment) may help improve performance.
 *
 * @param[in] compress_opts Compression options.
 * @param[out] alignment_requirements The minimum buffer alignment requirements
 * for compression.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
NVCOMP_EXPORT
nvcompStatus_t nvcompBatchedANSCompressGetRequiredAlignments(
  nvcompBatchedANSCompressOpts_t compress_opts,
  nvcompAlignmentRequirements_t *alignment_requirements
);

/**
 * @brief Get the amount of temporary memory required on the GPU for compression.
 *
 * @note This function does not enqueue asynchronous work on the stream; its result can be used immediately.
 *
 * @param[in] num_chunks The number of chunks of memory in the batch.
 * @param[in] max_uncompressed_chunk_bytes The maximum size of a chunk in the
 * batch.
 * @param[in] compress_opts Compression options.
 * @param[out] temp_bytes The amount of GPU memory that will be temporarily
 * required during compression. The value is returned on the host side.
 * @param[in] max_total_uncompressed_bytes Upper bound on the total uncompressed
 * size of all chunks
 *
 * @param[in] stream The CUDA stream associated with the operation.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
NVCOMP_EXPORT
nvcompStatus_t nvcompBatchedANSCompressGetTempSize(
  size_t num_chunks,
  size_t max_uncompressed_chunk_bytes,
  nvcompBatchedANSCompressOpts_t compress_opts,
  size_t *temp_bytes,
  size_t max_total_uncompressed_bytes,
  cudaStream_t stream
);

/**
 * @brief Get the amount of temporary memory required on the GPU for compression
 * synchronously.
 * 
 * @note This function may perform operations on the stream; if so, it will synchronize it internally.
 * Therefore, it does not require additional synchronization after it returns,
 * and the result can be used immediately.
 *
 * @param[in] device_uncompressed_chunk_ptrs Array with size \p num_chunks of pointers
 * to the uncompressed data chunks. Both the pointers and the uncompressed data
 * should reside in device-accessible memory.
 * Each chunk must be aligned to the value in the `input` member of the
 * \ref nvcompAlignmentRequirements_t object output by
 * `nvcompBatchedANSCompressGetRequiredAlignments` when called with the same
 * \p compress_opts.
 * @param[in] device_uncompressed_chunk_bytes Array with size \p num_chunks of
 * sizes of the uncompressed chunks in bytes.
 * The sizes should reside in device-accessible memory.
 * @param[in] num_chunks The number of chunks of memory in the batch.
 * @param[in] max_uncompressed_chunk_bytes The maximum size of a chunk in the
 * batch.
 * @param[in] compress_opts Compression options.
 * @param[out] temp_bytes The amount of GPU memory that will be temporarily
 * required during compression. The value is returned on the host side.
 * @param[in] max_total_uncompressed_bytes Upper bound on the total uncompressed
 * size of all chunks
 * @param[in] stream The CUDA stream to operate on.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
NVCOMP_EXPORT
nvcompStatus_t nvcompBatchedANSCompressGetTempSizeSync(
  const void *const *const device_uncompressed_chunk_ptrs,
  const size_t *const device_uncompressed_chunk_bytes,
  size_t num_chunks,
  size_t max_uncompressed_chunk_bytes,
  nvcompBatchedANSCompressOpts_t compress_opts,
  size_t *temp_bytes,
  size_t max_total_uncompressed_bytes,
  cudaStream_t stream
);

/**
 * @brief Get the maximum size that a chunk of size at most max_uncompressed_chunk_bytes
 * could compress to. That is, the minimum amount of output memory required to be given
 * nvcompBatchedANSCompressAsync() for each chunk.
 *
 * @param[in] max_uncompressed_chunk_bytes The maximum size of a chunk before compression.
 * @param[in] compress_opts Compression options.
 * @param[out] max_compressed_chunk_bytes The maximum possible compressed size of the chunk.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
NVCOMP_EXPORT
nvcompStatus_t nvcompBatchedANSCompressGetMaxOutputChunkSize(
  size_t max_uncompressed_chunk_bytes,
  nvcompBatchedANSCompressOpts_t compress_opts,
  size_t *max_compressed_chunk_bytes
);

/**
 * @brief Perform batched asynchronous compression.
 *
 * @warning Violating any of the conditions listed in the parameter descriptions
 * below may result in undefined behaviour.
 *
 * @note This function performs operations on the stream, and does not synchronize it,
 * therefore, it requires synchronization or stream-ordered operations to use its results.
 *
 * @param[in] device_uncompressed_chunk_ptrs Array with size \p num_chunks of pointers
 * to the uncompressed data chunks. Both the pointers and the uncompressed data
 * should reside in device-accessible memory.
 * Each chunk must be aligned to the value in the `input` member of the
 * \ref nvcompAlignmentRequirements_t object output by
 * `nvcompBatchedANSCompressGetRequiredAlignments` when called with the same
 * \p compress_opts.
 * @param[in] device_uncompressed_chunk_bytes Array with size \p num_chunks of
 * sizes of the uncompressed chunks in bytes.
 * The sizes should reside in device-accessible memory.
 * Each size must be a multiple of the byte width of \p compress_opts.data_type
 * (four bytes for `NVCOMP_TYPE_FLOAT32`).
 * @param[in] max_uncompressed_chunk_bytes The size of the largest uncompressed chunk.
 * @param[in] num_chunks Number of chunks of data to compress.
 * @param[in] device_temp_ptr The temporary GPU workspace, could be NULL in case
 * temporary memory is not needed.
 * Must be aligned to the value in the `temp` member of the
 * \ref nvcompAlignmentRequirements_t object output by
 * `nvcompBatchedANSCompressGetRequiredAlignments` when called with the same
 * \p compress_opts.
 * @param[in] temp_bytes The size of the temporary GPU memory pointed to by
 * `device_temp_ptr`.
 * @param[out] device_compressed_chunk_ptrs Array with size \p num_chunks of pointers
 * to the output compressed buffers. Both the pointers and the compressed
 * buffers should reside in device-accessible memory. Each compressed buffer
 * should be preallocated with the size given by
 * `nvcompBatchedANSCompressGetMaxOutputChunkSize`.
 * Each compressed buffer must be aligned to the value in the `output` member of the
 * \ref nvcompAlignmentRequirements_t object output by
 * `nvcompBatchedANSCompressGetRequiredAlignments` when called with the same
 * \p compress_opts.
 * @param[out] device_compressed_chunk_bytes Array with size \p num_chunks,
 * to be filled with the compressed sizes of each chunk.
 * The buffer should be preallocated in device-accessible memory.
 * @param[in] compress_opts Compression options.
 * @param[out] device_statuses Array with size \p num_chunks of statuses in
 * device-accessible memory. This argument needs to be preallocated. For each
 * chunk, if the compression is successful, the status will be set to
 * `nvcompSuccess`, and an error code otherwise.
 * Can be NULL if desired, in which case error status is not reported.
 * @param[in] stream The CUDA stream to operate on.
 *
 * @return nvcompSuccess if successfully launched, and an error code otherwise.
 */
NVCOMP_EXPORT
nvcompStatus_t nvcompBatchedANSCompressAsync(
  const void *const *device_uncompressed_chunk_ptrs,
  const size_t *device_uncompressed_chunk_bytes,
  size_t max_uncompressed_chunk_bytes,
  size_t num_chunks,
  void *device_temp_ptr,
  size_t temp_bytes,
  void *const *device_compressed_chunk_ptrs,
  size_t *device_compressed_chunk_bytes,
  nvcompBatchedANSCompressOpts_t compress_opts,
  nvcompStatus_t *device_statuses,
  cudaStream_t stream
);

/**
 * @brief The most restrictive of the minimum alignment requirements for void-type CUDA memory buffers
 * used for input, output, or temporary memory, passed to decompression functions.
 *
 * @note In all cases, typed memory buffers must still be aligned to their type's size,
 * e.g., 4 bytes for `int`.
 */
static const size_t nvcompANSRequiredDecompressionAlignment = 16;

/**
 * @brief Get the minimum buffer alignment requirements for decompression.
 *
 * @note Providing buffers with alignments above the minimum requirements
 * (e.g., 16- or 32-byte alignment) may help improve performance.
 *
 * @param[in] decompress_opts Decompression options.
 * @param[out] alignment_requirements The minimum buffer alignment requirements
 * for decompression.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
NVCOMP_EXPORT
nvcompStatus_t nvcompBatchedANSDecompressGetRequiredAlignments(
  nvcompBatchedANSDecompressOpts_t decompress_opts,
  nvcompAlignmentRequirements_t *alignment_requirements
);

/**
 * @brief Get the amount of temporary memory required on the GPU for decompression.
 *
 * @note This function does not enqueue asynchronous work on the stream; its result can be used immediately.
 *
 * @param[in] num_chunks Number of chunks of data to be decompressed.
 * @param[in] max_uncompressed_chunk_bytes The size of the largest chunk in bytes
 * when uncompressed.
 * @param[in] decompress_opts Decompression options.
 * @param[out] temp_bytes The amount of GPU memory that will be temporarily required
 * during decompression. The value is returned on the host side.
 * @param[in] max_total_uncompressed_bytes The total decompressed size of all the chunks.
 *
 * @param[in] stream The CUDA stream associated with the operation.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
NVCOMP_EXPORT
nvcompStatus_t nvcompBatchedANSDecompressGetTempSize(
  size_t num_chunks,
  size_t max_uncompressed_chunk_bytes,
  nvcompBatchedANSDecompressOpts_t decompress_opts,
  size_t *temp_bytes,
  size_t max_total_uncompressed_bytes,
  cudaStream_t stream
);

/**
 * @brief Get the amount of temporary memory required on the GPU for decompression
 * synchronously.
 *
 * @note This function may perform operations on the stream; if so, it will synchronize it internally.
 * Therefore, it does not require additional synchronization after it returns,
 * and the result can be used immediately.
 *
 * @param[in] device_compressed_chunk_ptrs Array with size \p num_chunks of pointers
 * in device-accessible memory to device-accessible compressed buffers.
 * Each chunk must be aligned to the value in the `input` member of the
 * \ref nvcompAlignmentRequirements_t object output by
 * `nvcompBatchedANSDecompressGetRequiredAlignments`.
 * @param[in] device_compressed_chunk_bytes Array with size \p num_chunks of sizes of
 * the compressed buffers in bytes. The sizes should reside in device-accessible memory.
 * @param[in] num_chunks Number of chunks of data to be decompressed.
 * @param[in] max_uncompressed_chunk_bytes The size of the largest chunk in bytes
 * when uncompressed.
 * @param[out] temp_bytes The amount of GPU memory that will be temporarily required
 * during decompression. The value is returned on the host side.
 * @param[in] max_total_uncompressed_bytes  The total decompressed size of all the chunks.
 * @param[in] decompress_opts Decompression options.
 * @param[out] device_statuses Array with size \p num_chunks of statuses in
 * device-accessible memory. This argument needs to be preallocated. For each
 * chunk, if the data can be parsed successfully, the status will be set to
 * `nvcompSuccess`, and an error code otherwise.
 * Can be NULL if desired, in which case error status is not reported.
 * @param[in] stream The CUDA stream to operate on.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
NVCOMP_EXPORT
nvcompStatus_t nvcompBatchedANSDecompressGetTempSizeSync(
  const void *const *const device_compressed_chunk_ptrs,
  const size_t *const device_compressed_chunk_bytes,
  size_t num_chunks,
  size_t max_uncompressed_chunk_bytes,
  size_t *temp_bytes,
  size_t max_total_uncompressed_bytes,
  nvcompBatchedANSDecompressOpts_t decompress_opts,
  nvcompStatus_t *device_statuses,
  cudaStream_t stream
);

/**
 * @brief Asynchronously compute the number of bytes of uncompressed data for
 * each compressed chunk.
 *
 * @warning Violating any of the conditions listed in the parameter descriptions
 * below may result in undefined behaviour.
 *
 * @note This function performs operations on the stream, and does not synchronize it,
 * therefore, it requires synchronization or stream-ordered operations to use its results.
 *
 * @param[in] device_compressed_chunk_ptrs Array with size \p num_chunks of
 * pointers in device-accessible memory to compressed buffers.
 * Each chunk must be aligned to the value in the `input` member of the
 * \ref nvcompAlignmentRequirements_t object output by
 * `nvcompBatchedANSDecompressGetRequiredAlignments`.
 * @param[in] device_compressed_chunk_bytes Array with size \p num_chunks of sizes
 * of the compressed buffers in bytes. The sizes should reside in device-accessible memory.
 * @param[out] device_uncompressed_chunk_bytes Array with size \p num_chunks
 * to be filled with the sizes, in bytes, of each uncompressed data chunk.
 * If there is an error when retrieving the size of a chunk, the
 * uncompressed size of that chunk will be set to 0. This argument needs to
 * be preallocated in device-accessible memory.
 * @param[in] num_chunks Number of data chunks to compute sizes of.
 * @param[in] stream The CUDA stream to operate on.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
NVCOMP_EXPORT
nvcompStatus_t nvcompBatchedANSGetDecompressSizeAsync(
  const void *const *device_compressed_chunk_ptrs,
  const size_t *device_compressed_chunk_bytes,
  size_t *device_uncompressed_chunk_bytes,
  size_t num_chunks,
  cudaStream_t stream
);

/**
 * @brief Perform batched asynchronous decompression.
 *
 * This function is used to decompress compressed buffers produced by
 * \ref nvcompBatchedANSCompressAsync .
 *
 * @warning Violating any of the conditions listed in the parameter descriptions
 * below may result in undefined behaviour.
 *
 * @note This function performs operations on the stream, and does not synchronize it,
 * therefore, it requires synchronization or stream-ordered operations to use its results.
 *
 * @param[in] device_compressed_chunk_ptrs Array with size \p num_chunks of pointers
 * in device-accessible memory to device-accessible compressed buffers.
 * Each chunk must be aligned to the value in the `input` member of the
 * \ref nvcompAlignmentRequirements_t object output by
 * `nvcompBatchedANSDecompressGetRequiredAlignments`.
 * @param[in] device_compressed_chunk_bytes Array with size \p num_chunks of sizes of
 * the compressed buffers in bytes. The sizes should reside in device-accessible memory.
 * @param[in] device_uncompressed_buffer_bytes Array with size \p num_chunks of sizes,
 * in bytes, of the output buffers to be filled with uncompressed data for each chunk.
 * The sizes should reside in device-accessible memory. If a
 * size is not large enough to hold all decompressed data, the decompressor
 * will set the status in \p device_statuses corresponding to the
 * overflow chunk to `nvcompErrorCannotDecompress`.
 * @param[out] device_uncompressed_chunk_bytes Array with size \p num_chunks to
 * be filled with the actual number of bytes decompressed for every chunk.
 * This argument needs to be preallocated in device-accessible memory.
 * @param[in] num_chunks Number of chunks of data to decompress.
 * @param[in] device_temp_ptr The temporary GPU space, could be NULL in case temporary space is not needed.
 * Must be aligned to the value in the `temp` member of the
 * \ref nvcompAlignmentRequirements_t object output by
 * `nvcompBatchedANSDecompressGetRequiredAlignments`.
 * @param[in] temp_bytes The size of the temporary GPU space.
 * @param[out] device_uncompressed_chunk_ptrs Array with size \p num_chunks of
 * pointers in device-accessible memory to decompressed data. Each uncompressed
 * buffer needs to be preallocated in device-accessible memory, have the size
 * specified by the corresponding entry in \p device_uncompressed_buffer_bytes,
 * and be aligned to the value in the `output` member of the
 * \ref nvcompAlignmentRequirements_t object output by
 * `nvcompBatchedANSDecompressGetRequiredAlignments`.
 * @param[in] decompress_opts Decompression options.
 * @param[out] device_statuses Array with size \p num_chunks of statuses in
 * device-accessible memory. This argument needs to be preallocated. For each
 * chunk, if the decompression is successful, the status will be set to
 * `nvcompSuccess`. If the decompression is not successful, for example due to
 * the corrupted input or out-of-bound errors, the status will be set to
 * `nvcompErrorCannotDecompress`.
 * @param[in] stream The CUDA stream to operate on.
 *
 * @return nvcompSuccess if successfully launched, and an error code otherwise.
 */
NVCOMP_EXPORT
nvcompStatus_t nvcompBatchedANSDecompressAsync(
  const void *const *device_compressed_chunk_ptrs,
  const size_t *device_compressed_chunk_bytes,
  const size_t *device_uncompressed_buffer_bytes,
  size_t *device_uncompressed_chunk_bytes,
  size_t num_chunks,
  void *const device_temp_ptr,
  size_t temp_bytes,
  void *const *device_uncompressed_chunk_ptrs,
  nvcompBatchedANSDecompressOpts_t decompress_opts,
  nvcompStatus_t *device_statuses,
  cudaStream_t stream
);

#ifdef __cplusplus
}
#endif

#endif // NVCOMP_ANS_H
