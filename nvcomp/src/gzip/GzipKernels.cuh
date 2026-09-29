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

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <limits>
#include <string>

#include "common.h"
#include "exception.hpp"
#include "gdeflate/gdeflate.h"
#include "GzipConstants.cuh"
#include "nvcomp/gzip.h"
#include "nvcomp/shared_types.h"
#include "nvcomp/utils.hpp"

namespace nvcomp
{

// Upper bound on the total number of DEFLATE sub-blocks across the whole batch
// (every chunk treated as the maximum size). This value sizes the scratch arrays
// and is used as a CUDA grid dimension / int index, so it must fit in an int.
inline size_t gzip_max_total_deflate_blocks(size_t max_chunk_size, size_t batch_size)
{
  const size_t max_total_deflate_blocks = roundUpDiv(max_chunk_size, static_cast<size_t>(DEFLATE_BLOCK_SIZE)) *
                                          batch_size;
  if (max_total_deflate_blocks > static_cast<size_t>(std::numeric_limits<int>::max()))
  {
    throw NVCompException(
      nvcompErrorChunkSizeTooLarge,
      "gzip compression requires max_uncompressed_chunk_bytes * num_chunks to not exceed " +
        std::to_string(nvcompGzipCompressionMaxAllowedChunkSize) + " bytes."
    );
  }
  return max_total_deflate_blocks;
}

struct CompressionScratchHandle
{
  const uint8_t **deflate_block_src_ptrs; // [max_total_deflate_blocks]
  size_t *deflate_block_src_sizes; // [max_total_deflate_blocks]
  uint8_t **deflate_block_dst_ptrs; // [max_total_deflate_blocks]
  size_t *deflate_block_dst_sizes; // [max_total_deflate_blocks]
  int *num_deflates_per_chunk; // [batch_size]
  int *chunk_start_offsets; // [batch_size]
  int *total_deflate_blocks; // single int
  uint8_t *deflate_block_pad_bits; // [max_total_deflate_blocks]
  uint32_t *crc32_per_chunk; // [batch_size]
  nvcompStatus_t *deflate_block_statuses; // [max_total_deflate_blocks]
  uint64_t *deflate_byte_offsets; // [max_total_deflate_blocks]
  uint64_t *chunk_tile_counts; // [batch_size] defrag tiles per chunk
  uint64_t *chunk_tile_starts; // [batch_size] exclusive prefix sum of chunk_tile_counts
  unsigned char *read_done; // [read_done_bytes] per-tile "source staged" flags for the defrag frontier
  size_t read_done_bytes; // number of tile flags (upper bound on total defrag tiles across the batch)

  uint8_t *scan_tmp_ptr; // cub::DeviceScan scratch, reused for both exclusive sums in the gzip layer
  size_t scan_tmp_bytes; // size of the scan scratch region (already rounded up to MEMBER_ALIGNMENT)

  // Number of bytes consumed from tmp_storage by this carve-out.
  size_t bytes_used = 0;

  static constexpr size_t MEMBER_ALIGNMENT = 8;

  CompressionScratchHandle(uint8_t *tmp_storage, size_t batch_size, size_t max_total_deflate_blocks, size_t slot_stride);

  static size_t required_tmp_bytes(size_t batch_size, size_t max_total_deflate_blocks, size_t slot_stride);
};

// Batch gzip compressor for chunked inputs.
// Subdivides each input buffer into uniformly sized deflate blocks and wraps them around with required gzip frame.
void gzipBatchCompress(
  const uint8_t *const *input_buffers,
  const size_t *input_decomp_sizes,
  size_t max_chunk_size,
  const size_t batch_size,
  uint8_t *tmp_buffer,
  size_t tmp_buffer_size,
  uint8_t *const *compressed_buffers,
  size_t *compressed_sizes,
  nvcompStatus_t *device_statuses,
  gdeflate::gdeflate_compression_algo algorithm,
  cudaStream_t stream
);

size_t gzipStreamingCompressTempSize(nvcompBatchedGzipCompressOpts_t opts);

// Streaming gzip compressor for inputs larger than GPU/host memory. Reads `input` in fixed-size
// windows, compresses each on the GPU as one non-final deflate block, and emits a single gzip member
// (header + concatenated payloads + terminal block + CRC32/ISIZE footer).
void gzipStreamingCompress(
  std::istream &input,
  std::ostream &output,
  nvcompBatchedGzipCompressOpts_t opts,
  void *device_temp,
  size_t device_temp_bytes,
  cudaStream_t stream
);

} // namespace nvcomp
