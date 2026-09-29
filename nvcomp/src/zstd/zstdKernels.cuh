/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

#include <cstdint>

#include "assert.h"
#include "nvcomp/shared_types.h"
#include "simple_types.cuh"

namespace zstd
{

struct ZstdFrame;
struct DeviceBlockShare;

struct DecompressionScratchHandle
{
  uint64_t *tmp_buffer_loc; // Amount of used space in the global scratch buffer [bytes]
  int *global_block_count;
  int *global_huff_block_count;
  int *global_ans_block_count;
  int *global_ix_block_huff;
  int *global_ix_block_ans;
  int *global_ix_frame_lz;
  int *global_ix_block_lz;
  int *global_ix_block_ans_init;
  int *global_ix_block_huff_init;

  ZstdFrame *zstd_frames; // Array of Zstd frames

  // Note:
  // In case the Zstd frames contain more Zstd blocks,
  // the additional Zstd blocks will be allocated at the
  // memory location right after the preallocated array of Zstd blocks.
  DeviceBlockShare *block_shares; // Array of Zstd blocks

  uint8_t *offset_tmp_buffer; // Next free address in the global scratch buffer
  size_t tmp_buffer_size; // Remaining free space in the global scratch buffer [bytes]
  size_t zstd_frame_count; // Number of Zstd frames (or nvCOMP chunks) this handle was intended for

  DecompressionScratchHandle(uint8_t *tmp_buffer, size_t frame_count, size_t input_tmp_buffer_size);
  DecompressionScratchHandle(const DecompressionScratchHandle &other) = delete;
  DecompressionScratchHandle(DecompressionScratchHandle &&other) = delete;

  // Initialize the device pointers before the actual decompression starts
  void init_values(cudaStream_t stream);

  // Determine the total static scratch buffer size needed for the decompression
  // Note: This only includes space that is known a priori to decompression
  //       and does not include space required for extra blocks (beyond frame_count)
  //       and space needed for decompressing huffman and FSE blocks.
  static size_t total_static_size(size_t frame_count);
};

void decompress_frames_api(
  uint8_t *const *output_frames,
  size_t *frame_decomp_sizes,
  const size_t *input_decomp_sizes,
  size_t frame_count,
  DecompressionScratchHandle &buffers,
  cudaStream_t &stream
);

void gather_frame_blocks_api(
  const uint8_t *const *input_frames,
  size_t frame_count,
  size_t *block_count,
  size_t *scratch_space_required,
  nvcompStatus_t *device_statuses,
  cudaStream_t stream
);

void classify_frames_api(
  const uint8_t *const *input_frames,
  size_t frame_count,
  DecompressionScratchHandle &buffers,
  const size_t *input_decomp_sizes,
  cudaStream_t stream
);

void init_tables_api(size_t frame_count, DecompressionScratchHandle &buffers, cudaStream_t &stream);

void lower_bound_test_api(int *ix_result, uint16_t search_val, uint16_t *search_array, int array_size);

void get_frame_sizes_api(const uint8_t *const *input_frames, size_t *frame_sizes, int frame_count, cudaStream_t &stream);

size_t get_reqd_tmp_buffer_size(int num_frames, size_t max_frame_size);

void zstdBatchCompress(
  const uint8_t *const *input_frames,
  const size_t *const decomp_sizes_device,
  const size_t max_chunk_size,
  const size_t batch_size,
  uint8_t *tmp_buffer,
  size_t tmp_buffer_size,
  uint8_t *const *compressed_data,
  size_t *compressed_sizes,
  nvcompStatus_t *device_statuses,
  cudaStream_t stream
);

size_t compute_max_comp_output_size(const size_t max_chunk_size);
size_t compress_compute_temp_size(
  const size_t batch_size,
  const size_t max_chunk_size,
  const size_t max_total_uncomp_size,
  int num_sms
);

void snappy_transcode_api(
  uint8_t *const *output_frames,
  size_t *frame_decomp_sizes,
  const size_t *device_uncompressed_bytes,
  size_t frame_count,
  DecompressionScratchHandle &buffers,
  cudaStream_t &stream
);

} // end namespace zstd