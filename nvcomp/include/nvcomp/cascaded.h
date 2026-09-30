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

#ifndef NVCOMP_CASCADED_H
#define NVCOMP_CASCADED_H

#include "nvcomp.h"

#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif // __cplusplus

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Selects the Cascaded compression mode.
 */
typedef enum nvcompCascadedMode_t
{
  /**
   * @brief Cascaded common options are not specified.
   *
   * This value is only valid for decompression. An unspecified mode maximizes
   * decompression support at the cost of decompression performance.
   */
  NVCOMP_CASCADED_MODE_UNSPECIFIED = 0u,
  /**
   * @brief Select adaptive asymmetric Cascaded compression.
   */
  NVCOMP_CASCADED_MODE_ASYMMETRIC = 1u,
  /**
   * @brief Select Terminal Cascaded compression and matching decompression.
   *
   * Supports signed and unsigned 32- and 64-bit integer types.
   */
  NVCOMP_CASCADED_MODE_SYMMETRIC = 2u
} nvcompCascadedMode_t;

/**
 * @brief Terminal codecs considered during compression and supported during
 * decompression.
 *
 * Values may be combined as a bit mask using bitwise OR. An enabled codec is
 * considered by the compressor, but is not guaranteed to be used.
 *
 * NVCOMP_CASCADED_TERMINAL_CODEC_UNSPECIFIED is invalid for compression and
 * for decompression when the mode is specified. It is ignored when the mode is
 * NVCOMP_CASCADED_MODE_UNSPECIFIED.
 *
 * During decompression, the mask identifies the terminal codecs that may have
 * been used to produce the input. If an input uses a terminal codec not included
 * in this mask, decompression reports `nvcompErrorCannotDecompress` for that
 * input.
 */
typedef enum
{
  NVCOMP_CASCADED_TERMINAL_CODEC_UNSPECIFIED = 0u,
  NVCOMP_CASCADED_TERMINAL_CODEC_CASCADED_BITPACK = 1u << 0u
} nvcompCascadedTerminalCodec_t;

/**
 * @brief Fine-grained encodings considered when Cascaded bitpacking is used as
 * the terminal codec.
 *
 * Values may be combined as a bit mask using bitwise OR. For example,
 * NVCOMP_CASCADED_FINE_GRAINED_ENCODING_DELTA | NVCOMP_CASCADED_FINE_GRAINED_ENCODING_RLE
 * enables both encodings.
 *
 * Asymmetric mode currently requires
 * NVCOMP_CASCADED_FINE_GRAINED_ENCODING_FOR to be enabled. In symmetric mode,
 * Frame Of Reference encoding is optional.
 */
typedef enum nvcompCascadedFineGrainedEncoding_t
{
  NVCOMP_CASCADED_FINE_GRAINED_ENCODING_NONE = 0u,
  NVCOMP_CASCADED_FINE_GRAINED_ENCODING_DELTA = 1u << 0u,
  NVCOMP_CASCADED_FINE_GRAINED_ENCODING_RLE = 1u << 1u,
  NVCOMP_CASCADED_FINE_GRAINED_ENCODING_FOR = 1u << 2u,
  NVCOMP_CASCADED_FINE_GRAINED_ENCODING_ALL = NVCOMP_CASCADED_FINE_GRAINED_ENCODING_DELTA |
                                              NVCOMP_CASCADED_FINE_GRAINED_ENCODING_RLE |
                                              NVCOMP_CASCADED_FINE_GRAINED_ENCODING_FOR
} nvcompCascadedFineGrainedEncoding_t;

/**
 * @brief Coarse-grained encodings considered during compression and supported
 * during decompression for the complete input stream.
 *
 * No coarse-grained encodings are currently supported.
 */
typedef enum nvcompCascadedCoarseGrainedEncoding_t
{
  NVCOMP_CASCADED_COARSE_GRAINED_ENCODING_NONE = 0u
} nvcompCascadedCoarseGrainedEncoding_t;

/**
 * @brief Cascaded options required by compression and optionally used to narrow
 * decompression support.
 *
 * Providing these options allows nvCOMP to improve decompression performance by
 * only launching decompression implementations that are required by the input.
 * If \p mode is NVCOMP_CASCADED_MODE_UNSPECIFIED, the remaining fields are
 * ignored to maximize decompression support.
 */
typedef struct
{
  /**
   * @brief The logical data type of the input.
   *
   * Asymmetric mode supports signed and unsigned 8-, 16-, 32-, and 64-bit
   * integer types. Symmetric mode supports signed and unsigned 32- and 64-bit
   * integer types. This field is ignored during universal decompression.
   */
  nvcompType_t data_type;
  /**
   * @brief The Cascaded mode used for compression and decompression.
   */
  nvcompCascadedMode_t mode;
  /**
   * @brief Mask of terminal codecs considered during compression or supported
   * during decompression.
   */
  uint64_t terminal_codec_flags;
  /**
   * @brief Mask of coarse-grained encodings considered during compression or
   * supported during decompression.
   */
  uint64_t coarse_grained_encoding_flags;
} nvcompCascadedCommonOpts_t;

/**
 * @brief Cascaded compression options for the low-level API
 */
typedef struct
{
  /**
   * @brief Options that may also be provided during decompression to improve
   * performance.
   */
  nvcompCascadedCommonOpts_t common_opts;
  /**
   * @brief Mask of fine-grained encodings to consider.
   */
  uint64_t fine_grained_encoding_flags;
  /**
   * @brief Balances compression throughput and compression ratio.
   *
   * In asymmetric mode, higher levels consider more optional encoding stages.
   * The current maximum useful level is 7, or 3 when only Delta is enabled.
   * Higher values select the highest supported level. Symmetric mode currently
   * ignores this option. The default is 3.
   */
  uint8_t compression_level;
  /**
   * @brief These bytes are unused and must be zeroed. This ensures
   *        compatibility if additional fields are added in the future.
   */
  char reserved[31];
} nvcompBatchedCascadedCompressOpts_t;

/**
 * @brief Cascaded decompression options for the low-level API
 */
typedef struct
{
  /**
   * @brief Decompression backend to use.
   */
  nvcompDecompressBackend_t backend;
  /**
   * @brief Options to narrow decompression support in favor of performance.
   *
   * Set \p common_opts.mode to NVCOMP_CASCADED_MODE_UNSPECIFIED to maximize
   * decompression support.
   */
  nvcompCascadedCommonOpts_t common_opts;
  /**
   * @brief These bytes are unused and must be zeroed. This ensures
   *        compatibility if additional fields are added in the future.
   */
  char reserved[32];
} nvcompBatchedCascadedDecompressOpts_t;

/**
 * @brief Default Cascaded compression options
 */
static const nvcompBatchedCascadedCompressOpts_t nvcompBatchedCascadedCompressDefaultOpts = {
  {NVCOMP_TYPE_UINT,
   NVCOMP_CASCADED_MODE_ASYMMETRIC,
   NVCOMP_CASCADED_TERMINAL_CODEC_CASCADED_BITPACK,
   NVCOMP_CASCADED_COARSE_GRAINED_ENCODING_NONE},
  NVCOMP_CASCADED_FINE_GRAINED_ENCODING_ALL,
  3u,
  {0}
};

/**
 * @brief Default Cascaded decompression options
 */
static const nvcompBatchedCascadedDecompressOpts_t nvcompBatchedCascadedDecompressDefaultOpts = {
  NVCOMP_DECOMPRESS_BACKEND_DEFAULT,
  {NVCOMP_TYPE_BITS,
   NVCOMP_CASCADED_MODE_UNSPECIFIED,
   NVCOMP_CASCADED_TERMINAL_CODEC_UNSPECIFIED,
   NVCOMP_CASCADED_COARSE_GRAINED_ENCODING_NONE},
  {0}
};

/**
 * @brief The maximum supported uncompressed chunk size in bytes for the Cascaded compressor.
 */
static const size_t nvcompCascadedCompressionMaxAllowedChunkSize = 1 << 24;

/**
 * @brief The maximum supported compressed and decompressed chunk size in bytes for the Cascaded decompressor.
 * @note To maximize decompression performance, users are encouraged to compress in smaller chunks, for example 64KiB.
 */
static const size_t nvcompCascadedDecompressionMaxAllowedChunkSize = (1 << 24) + 8;

/**
 * @brief The most restrictive of the minimum alignment requirements for void-type CUDA memory buffers
 * used for input, output, or temporary memory, passed to compression functions.
 *
 * @note In all cases, typed memory buffers must still be aligned to their type's size,
 * e.g., 4 bytes for `int`.
 */
static const size_t nvcompCascadedRequiredCompressionAlignment = 8;

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
nvcompStatus_t nvcompBatchedCascadedCompressGetRequiredAlignments(
  nvcompBatchedCascadedCompressOpts_t compress_opts,
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
nvcompStatus_t nvcompBatchedCascadedCompressGetTempSize(
  size_t num_chunks,
  size_t max_uncompressed_chunk_bytes,
  nvcompBatchedCascadedCompressOpts_t compress_opts,
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
 * `nvcompBatchedCascadedCompressGetRequiredAlignments` when called with the same
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
nvcompStatus_t nvcompBatchedCascadedCompressGetTempSizeSync(
  const void *const *const device_uncompressed_chunk_ptrs,
  const size_t *const device_uncompressed_chunk_bytes,
  size_t num_chunks,
  size_t max_uncompressed_chunk_bytes,
  nvcompBatchedCascadedCompressOpts_t compress_opts,
  size_t *temp_bytes,
  size_t max_total_uncompressed_bytes,
  cudaStream_t stream
);

/**
 * @brief Get the maximum size that a chunk of size at most max_uncompressed_chunk_bytes
 * could compress to. That is, the minimum amount of output memory required to be given
 * \ref nvcompBatchedCascadedCompressAsync for each chunk.
 *
 * @param[in] max_uncompressed_chunk_bytes The maximum size of a chunk before compression.
 * @param[in] compress_opts The Cascaded compression options to use.
 * @param[out] max_compressed_chunk_bytes The maximum possible compressed size of the chunk.
 *
 * @return nvcompSuccess if successful, and an error code otherwise.
 */
NVCOMP_EXPORT
nvcompStatus_t nvcompBatchedCascadedCompressGetMaxOutputChunkSize(
  size_t max_uncompressed_chunk_bytes,
  nvcompBatchedCascadedCompressOpts_t compress_opts,
  size_t *max_compressed_chunk_bytes
);

/**
 * @brief Perform batched asynchronous compression.
 *
 * @note The current implementation does not support uncompressed size larger
 * than 4,294,967,295 bytes (max uint32_t).
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
 * `nvcompBatchedCascadedCompressGetRequiredAlignments` when called with the same
 * \p compress_opts.
 * @param[in] device_uncompressed_chunk_bytes Array with size \p num_chunks of
 * sizes of the uncompressed chunks in bytes.
 * The sizes should reside in device-accessible memory.
 * Each chunk size must be a multiple of the size of the data type specified by
 * compress_opts.common_opts.data_type, else this may crash or produce invalid
 * output.
 * @param[in] max_uncompressed_chunk_bytes The size of the largest uncompressed chunk.
 * This parameter is currently unused. Set it to either the actual value
 * or zero.
 * @param[in] num_chunks Number of chunks of data to compress.
 * @param[in] device_temp_ptr This argument is not used.
 * @param[in] temp_bytes This argument is not used.
 * @param[out] device_compressed_chunk_ptrs Array with size \p num_chunks of pointers
 * to the output compressed buffers. Both the pointers and the compressed
 * buffers should reside in device-accessible memory. Each compressed buffer
 * should be preallocated with the size given by
 * `nvcompBatchedCascadedCompressGetMaxOutputChunkSize`.
 * Each compressed buffer must be aligned to the value in the `output` member of the
 * \ref nvcompAlignmentRequirements_t object output by
 * `nvcompBatchedCascadedCompressGetRequiredAlignments` when called with the same
 * \p compress_opts.
 * @param[out] device_compressed_chunk_bytes Array with size \p num_chunks,
 * to be filled with the compressed sizes of each chunk.
 * The buffer should be preallocated in device-accessible memory.
 * @param[in] compress_opts The cascaded format options. The format must be valid.
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
nvcompStatus_t nvcompBatchedCascadedCompressAsync(
  const void *const *device_uncompressed_chunk_ptrs,
  const size_t *device_uncompressed_chunk_bytes,
  size_t max_uncompressed_chunk_bytes, // not used
  size_t num_chunks,
  void *device_temp_ptr, // not used
  size_t temp_bytes, // not used
  void *const *device_compressed_chunk_ptrs,
  size_t *device_compressed_chunk_bytes,
  nvcompBatchedCascadedCompressOpts_t compress_opts,
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
static const size_t nvcompCascadedRequiredDecompressionAlignment = 8;

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
nvcompStatus_t nvcompBatchedCascadedDecompressGetRequiredAlignments(
  nvcompBatchedCascadedDecompressOpts_t decompress_opts,
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
nvcompStatus_t nvcompBatchedCascadedDecompressGetTempSize(
  size_t num_chunks,
  size_t max_uncompressed_chunk_bytes,
  nvcompBatchedCascadedDecompressOpts_t decompress_opts,
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
 * `nvcompBatchedCascadedDecompressGetRequiredAlignments`.
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
nvcompStatus_t nvcompBatchedCascadedDecompressGetTempSizeSync(
  const void *const *const device_compressed_chunk_ptrs,
  const size_t *const device_compressed_chunk_bytes,
  size_t num_chunks,
  size_t max_uncompressed_chunk_bytes,
  size_t *temp_bytes,
  size_t max_total_uncompressed_bytes,
  nvcompBatchedCascadedDecompressOpts_t decompress_opts,
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
 * `nvcompBatchedCascadedDecompressGetRequiredAlignments`.
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
nvcompStatus_t nvcompBatchedCascadedGetDecompressSizeAsync(
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
 * \ref nvcompBatchedCascadedCompressAsync.
 *
 * @note All compressed buffers in a batch must use data types with the same
 * element width; mixed-width batches are not supported.
 *
 * @warning Violating any of the conditions listed in the parameter descriptions
 * below may result in undefined behaviour.
 *
 * @warning Providing a corrupt buffer for decompression will result in undefined
 * behavior.
 *
 * @note This function performs operations on the stream, and does not synchronize it,
 * therefore, it requires synchronization or stream-ordered operations to use its results.
 *
 * @param[in] device_compressed_chunk_ptrs Array with size \p num_chunks of pointers
 * in device-accessible memory to device-accessible compressed buffers.
 * Each chunk must be aligned to the value in the `input` member of the
 * \ref nvcompAlignmentRequirements_t object output by
 * `nvcompBatchedCascadedDecompressGetRequiredAlignments`.
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
 * @param[in] device_temp_ptr This argument is not used.
 * @param[in] temp_bytes This argument is not used.
 * @param[out] device_uncompressed_chunk_ptrs Array with size \p num_chunks of
 * pointers in device-accessible memory to decompressed data. Each uncompressed
 * buffer needs to be preallocated in device-accessible memory, have the size
 * specified by the corresponding entry in \p device_uncompressed_buffer_bytes,
 * and be aligned to the value in the `output` member of the
 * \ref nvcompAlignmentRequirements_t object output by
 * `nvcompBatchedCascadedDecompressGetRequiredAlignments`.
 * @param[in] decompress_opts Decompression options.
 * @param[out] device_statuses Array with size \p num_chunks of statuses in
 * device-accessible memory. This argument needs to be preallocated. For each
 * chunk, if the decompression is successful, the status will be set to
 * `nvcompSuccess`. Passing corrupt, invalid, or insufficient data leads to
 * undefined behavior or out-of-bound errors. Error reporting cannot be guaranteed
 * in this scenario as only a limited validation is performed to maintain performance.
 * @param[in] stream The CUDA stream to operate on.
 *
 * @return nvcompSuccess if successfully launched, and an error code otherwise.
 */
NVCOMP_EXPORT
nvcompStatus_t nvcompBatchedCascadedDecompressAsync(
  const void *const *device_compressed_chunk_ptrs,
  const size_t *device_compressed_chunk_bytes,
  const size_t *device_uncompressed_buffer_bytes,
  size_t *device_uncompressed_chunk_bytes,
  size_t num_chunks,
  void *const device_temp_ptr, // not used
  size_t temp_bytes, // not used
  void *const *device_uncompressed_chunk_ptrs,
  nvcompBatchedCascadedDecompressOpts_t decompress_opts,
  nvcompStatus_t *device_statuses,
  cudaStream_t stream
);

#ifdef __cplusplus
}
#endif

#endif // NVCOMP_CASCADED_H
