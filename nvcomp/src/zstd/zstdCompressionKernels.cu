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

#include "common.h"
#include "constants.cuh"
#include "CudaUtils.h"
#include "device_guard.h"
#include "EntropyTables.cuh"
#include "Environment.hpp"
#include "exception.hpp"
#include "io.cuh"
#include "lz_hash.cuh"
#include "utils.cuh"
#include "zstd_fse_compression.cuh"
#include "zstd_huff_compression.cuh"
#include "zstdKernels.cuh"

using namespace nvcomp;
using nvcomp::DeviceGuard;

// #define ZSTD_LOGGING 1

namespace zstd
{

// Minimum number of LZ compression CTAs per SM
constexpr int min_num_lz_ctas(int cuda_arch = 0)
{
  switch (cuda_arch)
  {
    case 800:
    case 900:
    case 1000:
      return 28;
    case 1200:
      // The value was obtained experimentally as
      // the highest value for which ptxas would not throw a warning
      // "value of minnctapersm out of range" and performance is recovered for
      // zstd compression when comparing nvCOMP v5.0/ctk13.0/r580 versus nvCOMP
      // v4.2/ctk12.8/r575 on GB20x.
      return 24;
    default:
      return 16;
  }
}

__device__ void write_block_header(uint8_t *output, int comp_block_size, const bool is_raw, const bool last_block)
{
  // Block header is 3 bytes.
  // Bit 0 indicates whether this is the last block in the frame
  // bits 1-2 specifiy the block type
  // Bits 3-23 specify the block size

  // First 3 bits are: 101 // [2-1]: block type [0]: last block
  uint8_t block_type = is_raw ? 0 : 2;
  uint8_t block_type_byte = block_type << 1;
  output[0] = ((comp_block_size & 0x1f) << 3) | block_type_byte | last_block;
  output[1] = (comp_block_size >> 5) & 0xff;
  output[2] = (comp_block_size >> 13) & 0xff;
}

inline __host__ __device__ int compute_frame_header_size(const size_t uncomp_frame_size)
{
  constexpr int MAGIC_NUMBER_SIZE = 4;
  constexpr int HEADER_BYTES = 1;
  int window_descriptor_bytes = 1;
  int frame_content_size_bytes = 0;
  if (uncomp_frame_size <= ZSTD_FRAME_CONTENT_SIZE_2BYTE_THRESHOLD)
  {
    frame_content_size_bytes = 1;
    window_descriptor_bytes = 0;
  }
  else if (uncomp_frame_size <= ZSTD_FRAME_CONTENT_SIZE_4BYTE_THRESHOLD)
  {
    frame_content_size_bytes = 2;
  }
  else
  {
    frame_content_size_bytes = 4;
  }
  return MAGIC_NUMBER_SIZE + HEADER_BYTES + window_descriptor_bytes + frame_content_size_bytes;
}

__device__ int write_frame_header(uint8_t *output, const size_t uncomp_frame_size)
{
  int ix_output = 0;
  // ZSTD magic number
  output[ix_output++] = 0x28;
  output[ix_output++] = 0xb5;
  output[ix_output++] = 0x2f;
  output[ix_output++] = 0xfd;

  // Header byte has bits as follows:
  // 7-6	Frame_Content_Size_flag - compute the needed number of bytes based on the uncomp frame size.
  //      Flag value set based on the format spec.
  // 5	Single_Segment_flag - set to zero except for very small frames
  // 4	Unused_bit -- set to zero
  // 3	Reserved_bit -- must be zero
  // 2	Content_Checksum_flag - no checksum -- zero
  // 1-0	Dictionary_ID_flag -- dictionary field size -- zero here

  // Note: when both the fcs_flag and the ss_flag are 0, this indicates to the decompressor that fcs is not
  //       provided in the frame header. Currently the compressor always writes fcs.
  uint8_t fcs_flag = 0;
  uint8_t ss_flag = 0;
  if (uncomp_frame_size <= ZSTD_FRAME_CONTENT_SIZE_2BYTE_THRESHOLD)
  {
    fcs_flag = 0;
    // If fcs flag is zero, we need to set single segment flag to be able to provide the
    // frame content size in the header. In this case, the window descriptor is not present.
    ss_flag = 1;
  }
  else if (uncomp_frame_size <= ZSTD_FRAME_CONTENT_SIZE_4BYTE_THRESHOLD)
  {
    fcs_flag = 1;
  }
  else
  {
    fcs_flag = 2;
  }

  output[ix_output++] = (fcs_flag << 6) + (ss_flag << 5);

  if (not ss_flag)
  {
    // Write window descriptor
    // Bits 7-3: Exponent
    // Bits 2-0: Mantissa
    // windowLog = 10 + Exponent;
    // windowBase = 1 << windowLog;
    // windowAdd = (windowBase / 8) * Mantissa;
    // Window_Size = windowBase + windowAdd;
    // i.e. Window size of 1MB: exponent 20 mantissa 0.
    uint8_t exponent = OFFSET_BIT_COUNT - 10;
    output[ix_output++] = exponent << 3;
  }

  if (uncomp_frame_size <= ZSTD_FRAME_CONTENT_SIZE_2BYTE_THRESHOLD)
  {
    output[ix_output++] = uncomp_frame_size;
  }
  else if (uncomp_frame_size <= ZSTD_FRAME_CONTENT_SIZE_4BYTE_THRESHOLD)
  {
    size_t output_frame_size = uncomp_frame_size - 256;
    output[ix_output++] = output_frame_size & 0xff;
    output[ix_output++] = output_frame_size >> 8;
  }
  else
  {
    output[ix_output++] = uncomp_frame_size & 0xff;
    output[ix_output++] = (uncomp_frame_size >> 8) & 0xff;
    output[ix_output++] = (uncomp_frame_size >> 16) & 0xff;
    output[ix_output++] = (uncomp_frame_size >> 24) & 0xff;
  }
  return ix_output;
}

#ifdef __CUDA_ARCH__
__launch_bounds__(32, min_num_lz_ctas(__CUDA_ARCH__))
#else
__launch_bounds__(32, min_num_lz_ctas())
#endif
  __global__ void lz_compression_kernel(
    CompressBlockShare *block_shares,
    uint8_t *stage_buffer,
    int *global_ix_block,
    const int *global_block_count,
    uint8_t *seq_lit_buffer,
    const size_t seq_lit_buffer_size,
    uint64_t *seq_lit_buffer_loc,
    size_t max_block_size
  )
{
  max_block_size = roundUpTo(min(max_block_size, COMPRESS_NOMINAL_BLOCK_SIZE), size_t{4});
  TmpBufferManager tmp_buffer_manager{seq_lit_buffer_loc, seq_lit_buffer, seq_lit_buffer_size};

  int *hash_tables = reinterpret_cast<int *>(stage_buffer);
  stage_buffer += roundUpTo(gridDim.x * HASH_TABLE_SIZE * sizeof(int), SCRATCH_ALIGNMENT_REQ);

  assert(blockDim.x == WARP_SIZE_U);

  int *hash_table = hash_tables + HASH_TABLE_SIZE * blockIdx.x;
  const int block_count = *global_block_count;

  // Each CTA has a staging lit / sequence buffer that we write to before copying to the final loc
  // We need a per-CTA staging buffer that is 2.5x the maximum block size. This allows for
  // the entire block to be represented by 4 byte sequence copies (6 bytes each, hence 1.5x), or for the entire
  // block to be represented by literal bytes (1 byte each literal, hence 1x)

  // The per-staging CTA staging buffer allows us to use a smaller output buffer
  // ("output" being the output of the lz kernel that's ingested by the seq/lit kernels)
  // The maximum final allocation requirement is 1.5x, because the biggest final result is
  // the case where every byte is represented by a 4-byte sequence
  //
  // The per-cta staging buffer is needed because we don't know the number of literals and sequences
  // before the lz process is complete.

  uint8_t *literal_staging_buffer = stage_buffer + blockIdx.x * max_block_size;
  stage_buffer += roundUpTo(gridDim.x * max_block_size, SCRATCH_ALIGNMENT_REQ);

  const size_t max_seq_count = compute_max_sequence_count(max_block_size);
  const size_t comp_seq_buff_size = CompressSequenceBuffer::get_buff_size(max_seq_count);
  uint8_t *seq_data_buff_loc = stage_buffer + comp_seq_buff_size * blockIdx.x;
  __shared__ CompressSequenceBuffer seq_staging_buffer;
  seq_staging_buffer = CompressSequenceBuffer{seq_data_buff_loc, static_cast<int>(max_seq_count)};

  __shared__ int repeat_offsets[2]; // Maintain the two most recent offsets
  /*
   * Each CTA in the kernel works as follows:
   * 1) Get the next block, skip any block whose index in the frame isn't a multiple of BLOCKS_PER_LZ_COMP_TASK,
   * 2) if this is the first CTA to reach this block start iterating blocks in the frame
   *  until reaching the end or reaching a block that another CTA has already started to work on"
   */

  // Work-stealing loop
  __shared__ int ix_block;
  while (true)
  {
    __syncwarp();
    if (threadIdx.x == 0)
    {
      ix_block = atomicAdd(global_ix_block, 1);
    }
    __syncwarp();
    if (ix_block >= block_count)
    {
      return;
    }
#ifdef ZSTD_LZ_LOGGING
    if (ix_frame != 0)
    {
      continue;
    }
#endif

    CompressBlockShare *this_block_share = &block_shares[ix_block];

    if (this_block_share->ix_block % BLOCKS_PER_LZ_COMP_TASK > 0)
    {
      continue;
    }

    // Are we the first?
    int already_started = 0; // Default value for Coverity. Real value will be computed by 0th thread.
    if (threadIdx.x == 0)
    {
      already_started = this_block_share->already_started.fetch_or(1, cuda::std::memory_order_relaxed);
    }
    already_started = __shfl_sync(WARP_ALL, already_started, 0);
    if (already_started)
    {
      continue;
    }

    // Initialize the hash table
    for (uint32_t i = threadIdx.x; i < HASH_TABLE_SIZE; i += WARP_SIZE_U)
    {
      hash_table[i] = NULL_OFFSET;
    }
    __syncwarp();

    // Then warmstart the hash table if this is not block 0 in the frame
    uint32_t start_offset = 0;
    const uint8_t *input_buffer = this_block_share->input_buffer;
    if (this_block_share->ix_block > 0)
    {
      // During warmstart, we'll also need to disable repeat offset logic...
      repeat_offsets[0] = -1;
      repeat_offsets[1] = -2;
      input_buffer -= LZ_BACKFILL_HASH_SIZE;
      start_offset = LZ_BACKFILL_HASH_SIZE;
      fill_lz_hash_table(input_buffer, LZ_BACKFILL_HASH_SIZE, hash_table);
    }
    else
    {
      repeat_offsets[0] = ZSTD_INITIAL_REPEAT_OFFSETS[0];
      repeat_offsets[1] = ZSTD_INITIAL_REPEAT_OFFSETS[1];
    }

    int iter_block_count = 0;
    // Iterate within the frame until we reach a block that another CTA has started to work on
    while (this_block_share != nullptr)
    {
      if (iter_block_count > 0 and iter_block_count % BLOCKS_PER_LZ_COMP_TASK == 0)
      {
        // Are we the first?
        int already_started;
        if (threadIdx.x == 0)
        {
          already_started = this_block_share->already_started.fetch_or(1, cuda::std::memory_order_relaxed);
        }
        already_started = __shfl_sync(WARP_ALL, already_started, 0);
        if (already_started)
        {
          break;
        }
      }
      auto &block_share = *this_block_share;

      // Disable repeat offset logic between blocks.
      // Note:
      // This is a workaround for the case when the compressor falls back
      // to a raw uncompressed block if the size of the compressed output is larger.
      if (block_share.ix_block > 0)
      {
        repeat_offsets[0] = -1;
        repeat_offsets[1] = -2;
      }
      const uint32_t stop_ix = start_offset + block_share.uncompressed_size;

      seq_staging_buffer.reinit();

      int num_literals;
      int num_sequences;
      lz_compress_tile_greedy_hash(
        input_buffer,
        stop_ix,
        hash_table,
        seq_staging_buffer,
        literal_staging_buffer,
        num_sequences,
        num_literals,
        repeat_offsets,
        start_offset
      );
      start_offset = stop_ix;

      // Then copy over lits / sequences
      if (thread_warp_ix() == 0)
      {
        block_share.literals = tmp_buffer_manager.allocate<uint8_t>(num_literals);
        block_share.num_literals = num_literals;
        block_share.sequence_buffer =
          tmp_buffer_manager.allocate<uint8_t, 4>(CompressSequenceBuffer::get_buff_size(num_sequences));
        block_share.num_sequences = num_sequences;
      }
      __syncwarp();

      // Note, that literal_staging_buffer points to a large temporary buffer where we can
      // read beyond `num_literals` without commiting out-of-bounds reads.
      uint8_t *literal_output = block_share.literals;
      warp_input_4byte_aligned_copy(literal_staging_buffer, literal_output, num_literals);

      CompressSequenceBuffer compact_buffer{block_share.sequence_buffer, num_sequences};

      compact_buffer.set_sequence_count(num_sequences);
      seq_staging_buffer.set_sequence_count(num_sequences);

      seq_staging_buffer.copy(compact_buffer);
      __syncwarp();

#ifdef ZSTD_LOGGING
      print0(
        "frame %d block %d num seq %d num lit %d first match len %d\n",
        block_share.ix_frame,
        block_share.ix_block,
        block_share.num_sequences,
        block_share.num_literals,
        compact_buffer.get_match_length(0)
      );
#endif

      this_block_share = block_share.next_block_share;

      ++iter_block_count;
    }
  }
}

__global__ void literal_compression_kernel(
  CompressBlockShare *block_shares,
  CompressHuffmanBuffers *all_comp_huff_buffers,
  int *global_ix_block,
  const int *global_block_count_ptr
)
{
  assert(blockDim.x == WARP_SIZE_U);
  const int global_block_count = *global_block_count_ptr;

  CompressHuffmanBuffers &comp_huff_buffers = all_comp_huff_buffers[blockIdx.x];
  alloc_comp_huff_ans_encoder(comp_huff_buffers.encoder);

  __shared__ CompressHuffmanTable huff_table;
  int ix_block = 0; // Default value for Coverity. Real value will be computed by 0th thread.
  while (true)
  {
    if (threadIdx.x == 0)
    {
      ix_block = atomicAdd(global_ix_block, 1);
    }
    ix_block = __shfl_sync(WARP_ALL, ix_block, 0);

    if (ix_block >= global_block_count)
    {
      return;
    }

    auto &block_share = block_shares[ix_block];

    bool non_ans_block = block_share.num_sequences == 0;
    int num_literals = non_ans_block ? block_share.uncompressed_size : block_share.num_literals;
    const uint8_t *literals = non_ans_block ? block_share.input_buffer : block_share.literals;
    // Skipping the block header for now.

    bool do_huffman_compression = true;
    int comp_literal_bytes = 0;
    if (num_literals < MIN_LITERALS_FOR_HUFF_COMPRESSION)
    {
      do_huffman_compression = false;
    }
    else
    {
      uint8_t *literal_block_output = block_share.output_buffer + ZSTD_BLOCK_HEADER_SIZE;
      do_huffman_compression = zstd_compress_literal_block(
        literals,
        HUF_MAX_SYMBOLS - 1, // max symbol value
        num_literals,
        &comp_literal_bytes,
        comp_huff_buffers,
        huff_table,
        literal_block_output
      );
    }

    bool is_raw_block = false;

    if (not do_huffman_compression)
    {
      // Check whether we should output a raw block
      if (block_share.num_sequences == 0)
      {
        is_raw_block = true;
      }
    }

    if (is_raw_block)
    {
#ifdef ZSTD_LOGGING
      print0("block %d raw\n", ix_block);
#endif
      uint8_t *output_block = block_share.output_buffer;
      size_t input_size = block_share.uncompressed_size;
      const uint8_t *input_buffer = block_share.input_buffer;
      int comp_block_size = input_size;
      bool last_block = block_share.ix_block == block_share.num_blocks - 1;
      write_block_header(output_block, comp_block_size, true /*raw*/, last_block);
      output_block += ZSTD_BLOCK_HEADER_SIZE;

      // Then copy the raw bytes to the output block
      for (int ix_byte = threadIdx.x; ix_byte < comp_block_size; ix_byte += WARP_SIZE)
      {
        output_block[ix_byte] = input_buffer[ix_byte];
      }
      block_share.num_sequences = -1;
      block_share.compressed_size = ZSTD_BLOCK_HEADER_SIZE + comp_block_size;
      continue;
    }
    else if (not do_huffman_compression)
    {
      // Overwrite the output with raw literals
      uint8_t *literal_block_output = block_share.output_buffer + ZSTD_BLOCK_HEADER_SIZE;
      // Fallback to raw bytes for literals.
      int header_bytes = generate_raw_huffman_header(literal_block_output, num_literals);
      literal_block_output += header_bytes;
      for (int ix_byte = threadIdx.x; ix_byte < num_literals; ix_byte += WARP_SIZE)
      {
        literal_block_output[ix_byte] = literals[ix_byte];
      }
      comp_literal_bytes = header_bytes + num_literals;
#ifdef ZSTD_LOGGING
      print0("block %d raw literals\n", ix_block);
#endif
    }
#ifdef ZSTD_LOGGING
    else
    {
      print0("block %d compressed, num literals %d\n", ix_block, num_literals);
    }
#endif

    block_share.compressed_size = ZSTD_BLOCK_HEADER_SIZE + comp_literal_bytes;
  }
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900 || __CUDA_ARCH__ == 800 || __CUDA_ARCH__ == 1000)
const int SEQ_NUM_CTAS = 32;
#else
const int SEQ_NUM_CTAS = 16;
#endif
__launch_bounds__(32, SEQ_NUM_CTAS) __global__ void sequence_compression_kernel(
  CompressBlockShare *block_shares,
  SequenceCompressBuffers *all_seq_buffers,
  int *global_ix_block,
  const int *global_block_count_ptr,
  uint8_t *all_symbol_arrays,
  const size_t max_chunk_size,
  const size_t batch_size
)
{
  assert(blockDim.x == WARP_SIZE_U);

  const int global_block_count = *global_block_count_ptr;

  __shared__ SymbolEncoder encoders[3];

  auto &seq_buffers = all_seq_buffers[blockIdx.x];

  allocate_comp_fse_tables(encoders);

  const int max_block_size = min(max_chunk_size, COMPRESS_NOMINAL_BLOCK_SIZE);
  const int max_sequence_count = compute_max_sequence_count(max_block_size);

  uint8_t *symbol_array = all_symbol_arrays + max_sequence_count * 3 * blockIdx.x; // 3 is number of ANS streams

  __shared__ SharedConstantTables constant_tables;
  constant_tables.init();

  while (true)
  {
    int ix_block = 0; // Default value for Coverity. Real value will be computed by 0th thread.
    if (threadIdx.x == 0)
    {
      ix_block = atomicAdd(global_ix_block, 1);
    }
    ix_block = __shfl_sync(WARP_ALL, ix_block, 0);
    if (ix_block >= global_block_count)
    {
      return;
    }

    auto &block_share = block_shares[ix_block];

    int num_sequences = block_share.num_sequences;
    if (num_sequences == -1)
    {
// This means it's a raw block. Skip it.
#ifdef ZSTD_LOGGING
      print0("skipping block %d size %lu\n", ix_block, block_share.uncompressed_size);
#endif

      continue;
    }

    CompressSequenceBuffer sequences{block_share.sequence_buffer, num_sequences, num_sequences};

    // this_comp_size is the compressed size after the literals kernel finished
    size_t this_comp_size = block_share.compressed_size;

    uint8_t *seq_output_buffer = block_share.output_buffer + this_comp_size;

    // Assumes that a compressed block would never be larger than the uncompressed size
    const int max_output_size = block_share.uncompressed_size + ZSTD_BLOCK_HEADER_SIZE - this_comp_size;

    int seq_comp_buffer_size = 0;
    compress_sequences(
      num_sequences,
      sequences,
      seq_output_buffer,
      &seq_comp_buffer_size,
      seq_buffers,
      encoders,
      symbol_array,
      max_output_size,
      constant_tables
    );

    this_comp_size += seq_comp_buffer_size;

    // Now write out the block size
    uint8_t *output_block = block_share.output_buffer;

    bool last_block = block_share.ix_block == block_share.num_blocks - 1;

    int comp_block_size = this_comp_size - ZSTD_BLOCK_HEADER_SIZE;
    const size_t input_size = block_share.uncompressed_size;
    if (comp_block_size >= input_size)
    {
#ifdef ZSTD_LOGGING
      print0("block %d raw after seq processing\n", ix_block);
#endif

      // Fallback to raw block
      const uint8_t *input_buffer = block_share.input_buffer;
      int comp_block_size = input_size;
      write_block_header(output_block, comp_block_size, true /*raw*/, last_block);
      output_block += ZSTD_BLOCK_HEADER_SIZE;

      // Then copy the raw bytes to the output block
      for (int ix_byte = threadIdx.x; ix_byte < comp_block_size; ix_byte += WARP_SIZE)
      {
        output_block[ix_byte] = input_buffer[ix_byte];
      }
      block_share.compressed_size = ZSTD_BLOCK_HEADER_SIZE + comp_block_size;
    }
    else
    {
      block_share.compressed_size = this_comp_size;
      write_block_header(output_block, comp_block_size, false /*raw*/, last_block);
    }
  }
}

__global__ void setup_frame_compress(
  const uint8_t *const *input_frames,
  const size_t *input_decomp_sizes,
  uint8_t *const *output_frames,
  const size_t frame_count,
  CompressBlockShare *block_shares,
  const size_t max_chunk_size,
  int *global_ix_block
)
{
  const int ix_frame = threadIdx.x + blockIdx.x * blockDim.x;

  if (ix_frame >= frame_count)
  {
    return;
  }

  uint8_t *output_buffer = output_frames[ix_frame];
  int num_header_bytes = write_frame_header(output_buffer, input_decomp_sizes[ix_frame]);

  output_buffer += num_header_bytes;

  const int input_size = narrow_cast<int>(input_decomp_sizes[ix_frame]);
  const int num_blocks = max(roundUpDiv(input_size, COMPRESS_NOMINAL_BLOCK_SIZE), 1);
  const uint8_t *input_buffer = input_frames[ix_frame];

  const size_t max_block_size = min(max_chunk_size, COMPRESS_NOMINAL_BLOCK_SIZE);
  const size_t max_sequence_count = compute_max_sequence_count(max_block_size);

  int base_ix_block;
  if (num_blocks > 1)
  {
    base_ix_block = atomicAdd(global_ix_block, num_blocks - 1);
  }

  const int rem_bytes = input_size % COMPRESS_NOMINAL_BLOCK_SIZE;
  const int last_block_size = rem_bytes == 0 ? COMPRESS_NOMINAL_BLOCK_SIZE : rem_bytes;

  CompressBlockShare *block_share_ptr;
  for (size_t ix_block = 0; ix_block < num_blocks; ++ix_block)
  {
    size_t true_ix_block;
    if (ix_block == 0)
    {
      true_ix_block = ix_frame;
      block_share_ptr = &block_shares[ix_frame];
      block_share_ptr->uncompressed_size = num_blocks == 1 ? input_size : COMPRESS_NOMINAL_BLOCK_SIZE;
      block_share_ptr->input_buffer = input_buffer;
    }
    else
    {
      // Assumes that the max size of a block is UNCOMP_SIZE + BLOCK_HEADER_SIZE
      output_buffer += COMPRESS_NOMINAL_BLOCK_SIZE + ZSTD_BLOCK_HEADER_SIZE;
      input_buffer += COMPRESS_NOMINAL_BLOCK_SIZE;
      true_ix_block = base_ix_block + ix_block - 1;
      const auto tmp_block_ptr = &block_shares[true_ix_block];
      block_share_ptr->next_block_share = tmp_block_ptr;
      block_share_ptr = tmp_block_ptr;
      block_share_ptr->uncompressed_size = ix_block == num_blocks - 1 ? last_block_size : COMPRESS_NOMINAL_BLOCK_SIZE;
    }

    block_share_ptr->input_buffer = input_buffer;
    block_share_ptr->output_buffer = output_buffer;
    block_share_ptr->ix_block = ix_block;
    block_share_ptr->num_blocks = num_blocks;
    block_share_ptr->ix_frame = ix_frame;
    block_share_ptr->already_started = 0;
    if (ix_block == num_blocks - 1)
    {
      block_share_ptr->next_block_share = nullptr;
    }
  }
}

inline __device__ void copy_with_cache(uint8_t *output, const uint8_t *input, uint8_t *cache, const uint32_t num_bytes)
{
  const bool have_same_alignment = (reinterpret_cast<uintptr_t>(output) % sizeof(uint32_t)) ==
                                   (reinterpret_cast<uintptr_t>(input) % sizeof(uint32_t));
  if (have_same_alignment)
  {
    const uint8_t unaligned_bytes = sizeof(uint32_t) - reinterpret_cast<uintptr_t>(input) % sizeof(uint32_t);

    // Copy unaligned bytes directly
    bool is_active = threadIdx.x < unaligned_bytes and threadIdx.x < num_bytes;
    uint8_t byte = is_active ? input[threadIdx.x] : 0;
    __syncwarp();
    if (is_active)
    {
      output[threadIdx.x] = byte;
    }
    if (num_bytes <= unaligned_bytes)
    {
      return;
    }
    input += unaligned_bytes;
    output += unaligned_bytes;

    // Copy aligned bytes via cache
    uint32_t *cache32 = reinterpret_cast<uint32_t *>(cache);
    uint32_t *output32 = reinterpret_cast<uint32_t *>(output);
    const uint32_t *input32 = reinterpret_cast<const uint32_t *>(input);
    const uint32_t num_elems = (num_bytes - unaligned_bytes) / sizeof(uint32_t);
    for (int ix = threadIdx.x; ix < num_elems; ix += blockDim.x)
    {
      cache32[ix] = input32[ix];
    }
    __syncthreads();
    for (int ix = threadIdx.x; ix < num_elems; ix += blockDim.x)
    {
      output32[ix] = cache32[ix];
    }

    // Copy remaining bytes directly
    const uint8_t remaining_bytes = (num_bytes - unaligned_bytes) % sizeof(uint32_t);
    input += num_elems * sizeof(uint32_t);
    output += num_elems * sizeof(uint32_t);
    is_active = threadIdx.x < remaining_bytes;
    byte = is_active ? input[threadIdx.x] : 0;
    __syncwarp();
    if (is_active)
    {
      output[threadIdx.x] = byte;
    }
  }
  else
  {
    // Copy byte-by-byte
    for (int ix = threadIdx.x; ix < num_bytes; ix += blockDim.x)
    {
      cache[ix] = input[ix];
    }
    __syncthreads();
    for (int ix = threadIdx.x; ix < num_bytes; ix += blockDim.x)
    {
      output[ix] = cache[ix];
    }
  }
}

__global__ void compact_compressed_frames(
  const CompressBlockShare *block_shares,
  const size_t *const input_decomp_sizes,
  size_t *compressed_sizes
)
{
  // Note:
  // 1 zstd frame = [Block_Header, 3 bytes][ Block_Content, n bytes], where
  // Block_Content is limited by Block_Maximum_Size .
  // Block_Maximum_Size = min(Window_Size, 128 KiB) is applicable to both the compressed and the decompressed sizes
  // of any block in the frame.
  // However, the nvCOMP Zstd compressor compresses up to COMPRESS_NOMINAL_BLOCK_SIZE bytes.
  constexpr int SHMEM_SIZE = (COMPRESS_NOMINAL_BLOCK_SIZE + ZSTD_BLOCK_HEADER_SIZE + 1) / 2;
  __shared__ __align__(4) uint8_t cache[SHMEM_SIZE];
  const int ix_frame = blockIdx.x;
  const int input_size = input_decomp_sizes[ix_frame];
  const int num_blocks = roundUpDiv(input_size, COMPRESS_NOMINAL_BLOCK_SIZE);

  auto block_share_ptr = &block_shares[ix_frame];
  size_t compressed_size = compute_frame_header_size(input_size) + block_share_ptr->compressed_size;
  uint8_t *output_buffer = block_share_ptr->output_buffer + block_share_ptr->compressed_size;

  // If there are multiple blocks, we'll compact the result
  for (int ix_block = 1; ix_block < num_blocks; ++ix_block)
  {
    block_share_ptr = block_share_ptr->next_block_share;
    compressed_size += block_share_ptr->compressed_size;
    uint8_t *input_buffer = block_share_ptr->output_buffer;
    if (block_share_ptr->compressed_size <= SHMEM_SIZE)
    {
      copy_with_cache(output_buffer, input_buffer, cache, block_share_ptr->compressed_size);
    }
    else
    {
      copy_with_cache(output_buffer, input_buffer, cache, SHMEM_SIZE);
      __syncthreads();
      copy_with_cache(
        output_buffer + SHMEM_SIZE,
        input_buffer + SHMEM_SIZE,
        cache,
        block_share_ptr->compressed_size - SHMEM_SIZE
      );
    }
    output_buffer += block_share_ptr->compressed_size;
    __syncthreads();
  }

  if (threadIdx.x == 0)
  {
    compressed_sizes[ix_frame] = compressed_size;
  }
}

struct CompressionScratchHandle
{
  size_t current_size;
  int32_t *num_literals;
  int32_t *num_sequences;
  int32_t *global_ix_block_lz;
  int32_t *global_ix_block_huff;
  int32_t *global_ix_block_ans;
  int32_t *global_block_count;
  uint64_t *seq_lit_buffer_loc;
  uint8_t *all_buffer_ptr; // we'll use this pointer for both huffman and ANS
  uint8_t *stage_buffer;
  uint8_t *seq_lit_buffer;
  size_t seq_lit_buffer_size;
  CompressBlockShare *block_shares;

  CompressionScratchHandle(
    uint8_t *tmp_buffer,
    const size_t batch_size,
    const size_t lz_grid_dim,
    const size_t ans_grid_dim,
    const size_t huff_grid_dim,
    const size_t max_chunk_size,
    const size_t tmp_buffer_size,
    size_t total_uncompressed_bytes,
    bool compute_total_uncomp_bytes
  )
      : current_size(0)
      , num_literals(nullptr)
      , num_sequences(nullptr)
  {
    total_uncompressed_bytes = roundUpTo(total_uncompressed_bytes, size_t{4});
    const size_t max_block_size =
      roundUpTo(max(min(COMPRESS_NOMINAL_BLOCK_SIZE, max_chunk_size), size_t{1}), size_t{4});
    const size_t max_num_blocks = max(roundUpDiv(max_chunk_size, COMPRESS_NOMINAL_BLOCK_SIZE), size_t{1}) * batch_size;

    // Beginning of memory to be zero'd before kernel calls
    global_ix_block_lz = reinterpret_cast<int32_t *>(tmp_buffer + current_size);
    current_size += roundUpTo(sizeof(int32_t), SCRATCH_ALIGNMENT_REQ);

    global_ix_block_huff = reinterpret_cast<int32_t *>(tmp_buffer + current_size);
    current_size += roundUpTo(sizeof(int32_t), SCRATCH_ALIGNMENT_REQ);

    global_ix_block_ans = reinterpret_cast<int32_t *>(tmp_buffer + current_size);
    current_size += roundUpTo(sizeof(int32_t), SCRATCH_ALIGNMENT_REQ);

    global_block_count = reinterpret_cast<int32_t *>(tmp_buffer + current_size);
    current_size += roundUpTo(sizeof(int32_t), SCRATCH_ALIGNMENT_REQ);

    seq_lit_buffer_loc = reinterpret_cast<uint64_t *>(tmp_buffer + current_size);
    current_size += roundUpTo(sizeof(uint64_t), SCRATCH_ALIGNMENT_REQ);

    const size_t all_buffer_bytes =
      max(sizeof(CompressHuffmanBuffers) * huff_grid_dim, sizeof(SequenceCompressBuffers) * ans_grid_dim);
    all_buffer_ptr = reinterpret_cast<uint8_t *>(tmp_buffer + current_size);
    current_size += roundUpTo(all_buffer_bytes, SCRATCH_ALIGNMENT_REQ);

    // Combine seq symbol arrays and hash table scratch space as well
    const size_t lz_hash_table_bytes = HASH_TABLE_SIZE * sizeof(int) * lz_grid_dim;
    const size_t lz_staging_seq_result_bytes =
      lz_grid_dim * CompressSequenceBuffer::get_buff_size(compute_max_sequence_count(max_block_size));

    const size_t lz_staging_lit_result_bytes = lz_grid_dim * max_block_size;
    const size_t lz_stage_bytes = lz_hash_table_bytes + lz_staging_seq_result_bytes + lz_staging_lit_result_bytes +
                                  3 * (SCRATCH_ALIGNMENT_REQ - 1);

    const size_t seq_stage_bytes = size_t{3} * ans_grid_dim * compute_max_sequence_count(max_block_size);

    const size_t stage_buffer_bytes = max(seq_stage_bytes, lz_stage_bytes);

    stage_buffer = reinterpret_cast<uint8_t *>(tmp_buffer + current_size);
    current_size += roundUpTo(stage_buffer_bytes, SCRATCH_ALIGNMENT_REQ);

    block_shares = reinterpret_cast<CompressBlockShare *>(tmp_buffer + current_size);
    current_size += roundUpTo(sizeof(CompressBlockShare) * max_num_blocks, SCRATCH_ALIGNMENT_REQ);

    seq_lit_buffer = reinterpret_cast<uint8_t *>(tmp_buffer + current_size);
    // Note, this section needs to be the last part of the temp buffer for proper operation.
    // Add further requirements before the below block.
    if (compute_total_uncomp_bytes)
    {
      seq_lit_buffer_size = tmp_buffer_size - current_size;
    }
    else
    {
      // There's a per-buffer overhead and a per-sequence ovehead.
      // The per-buffer overhead results from
      // 1) Alignment requirement of 4 bytes for each buffer
      // 2) Rounding error in the offset bits requirement
      // 3) Rounding error in number of sequences -- we require an even number of sequences in the allocation
      //    to improve copy speed, so in the worst case we could have an extra sequence allocated for every block
      const int per_buff_alignment_req = 3;
      const int per_buff_rounding_req = 4;
      const int per_buff_seq_rounding_req = 8; // 4B offset, 2B lit len, 2B mat len
      const int per_buffer_overhead = per_buff_alignment_req + per_buff_rounding_req + per_buff_seq_rounding_req;

      size_t max_num_sequences = compute_max_sequence_count(total_uncompressed_bytes);
      seq_lit_buffer_size = roundUpTo(
        CompressSequenceBuffer::get_buff_size(max_num_sequences) + per_buffer_overhead * max_num_blocks,
        SCRATCH_ALIGNMENT_REQ
      );
      current_size += SCRATCH_ALIGNMENT_REQ - 1; // provide enough extra storage to allow alignment to 8 bytes
    }

    current_size += seq_lit_buffer_size;

    assert(tmp_buffer_size == 0 or current_size <= tmp_buffer_size);
  }
};

void get_grid_dims(
  const size_t batch_size,
  const size_t num_sms,
  size_t &lz_grid_dim,
  size_t &huff_grid_dim,
  size_t &ans_grid_dim
)
{
  int num_blocks_per_sm;
  const size_t block_size = WARP_SIZE_U; // All 3 kernels require a single warp per CTA

  // How many CTAs did we intend to launch minimum simultaneously?
  const auto min_cta_count = min_num_lz_ctas(get_cuda_arch());

  // grid dim ideally would be based on the total size of the batch to allow strong scaling,
  // but we don't have this information in the compression API
  // We'll just launch full occupancy kernels for now

  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &num_blocks_per_sm,
    lz_compression_kernel,
    block_size,
    0 /*dynamic shmem size*/
  ));
#ifdef ZSTD_LZ_LOGGING
  lz_grid_dim = 1;
#else
  lz_grid_dim = num_sms * narrow_cast<size_t>(std::min(num_blocks_per_sm, min_cta_count));
#endif
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &num_blocks_per_sm,
    literal_compression_kernel,
    block_size,
    0 /*dynamic shmem size*/
  ));
  huff_grid_dim = num_sms * narrow_cast<size_t>(num_blocks_per_sm);

  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &num_blocks_per_sm,
    sequence_compression_kernel,
    block_size,
    0 /*dynamic shmem size*/
  ));
  ans_grid_dim = num_sms * narrow_cast<size_t>(num_blocks_per_sm);
}

__global__ void init_buffers(
  int32_t *global_ix_block_lz,
  int32_t *global_ix_block_huff,
  int32_t *global_ix_block_ans,
  int32_t *global_block_count,
  uint64_t *seq_lit_buffer_loc,
  const size_t batch_size
)
{
  *global_ix_block_lz = 0;
  *global_ix_block_huff = 0;
  *global_ix_block_ans = 0;
  *global_block_count = batch_size;
  *seq_lit_buffer_loc = 0;
}

void zstdBatchCompress(
  const uint8_t *const *input_frames,
  const size_t *const input_decomp_sizes,
  const size_t max_chunk_size,
  const size_t batch_size,
  uint8_t *tmp_buffer,
  size_t tmp_buffer_size,
  uint8_t *const *compressed_frames,
  size_t *compressed_sizes,
  nvcompStatus_t *device_statuses,
  cudaStream_t stream
)
{
  DeviceGuard device_guard(stream);

  const size_t rounded_max_chunk_size = roundUpTo(max_chunk_size, size_t{4});
  const int block_size = WARP_SIZE; // All 3 kernels require a single warp per CTA

  const int num_sms = CudaUtils::get_sm_count(stream);

  size_t lz_grid_dim, ans_grid_dim, huff_grid_dim;
  get_grid_dims(batch_size, num_sms, lz_grid_dim, huff_grid_dim, ans_grid_dim);

  // mark compression successful
  try_clear_device_statuses(batch_size, device_statuses, stream);

  // Can back out the total comp size from the tmp buffer size. The tmp buffer requirement was
  // padded by `SCRATCH_ALIGNMENT_REQ - 1` bytes to allow alignment to 8 bytes
  tmp_buffer = reinterpret_cast<uint8_t *>(roundUpTo((uintptr_t)tmp_buffer, size_t{8}));
  tmp_buffer_size -= (SCRATCH_ALIGNMENT_REQ - 1);
  CompressionScratchHandle tmp_buffers{
    tmp_buffer,
    batch_size,
    lz_grid_dim,
    ans_grid_dim,
    huff_grid_dim,
    rounded_max_chunk_size,
    tmp_buffer_size,
    0 /*total uncomp bytes*/,
    true /*compute total uncomp bytes*/
  };

  // Set the global ix frames to zero
  init_buffers<<<1, 1, 0, stream>>>(
    tmp_buffers.global_ix_block_lz,
    tmp_buffers.global_ix_block_huff,
    tmp_buffers.global_ix_block_ans,
    tmp_buffers.global_block_count,
    tmp_buffers.seq_lit_buffer_loc,
    batch_size
  );
  CUDA_CHECK(cudaGetLastError());

  const int setup_block_dim = 32;
  const int setup_grid_dim = narrow_cast<int>(roundUpDiv(batch_size, setup_block_dim));

  setup_frame_compress<<<setup_grid_dim, 32, 0, stream>>>(
    input_frames,
    input_decomp_sizes,
    compressed_frames,
    batch_size,
    tmp_buffers.block_shares,
    rounded_max_chunk_size,
    tmp_buffers.global_block_count
  );
  CUDA_CHECK(cudaGetLastError());

  const auto single_cta_env_var = nvcomp::getenv(ZSTD_USE_SINGLE_CTA_FOR_LZ_COMPRESS_ENV);
  bool single_cta = single_cta_env_var != "" and single_cta_env_var != "0";
  if (single_cta)
  {
    lz_grid_dim = 1;
  }
  lz_compression_kernel<<<cuda_dim_cast(lz_grid_dim), block_size, 0, stream>>>(
    // lz_compression_kernel<<<1, block_size, 0, stream>>>(
    tmp_buffers.block_shares,
    tmp_buffers.stage_buffer,
    tmp_buffers.global_ix_block_lz,
    tmp_buffers.global_block_count,
    tmp_buffers.seq_lit_buffer,
    tmp_buffers.seq_lit_buffer_size,
    tmp_buffers.seq_lit_buffer_loc,
    rounded_max_chunk_size
  );
  CUDA_CHECK(cudaGetLastError());

  // Now call a kernel to fill in literals.
  literal_compression_kernel<<<cuda_dim_cast(huff_grid_dim), block_size, 0, stream>>>(
    tmp_buffers.block_shares,
    reinterpret_cast<zstd::CompressHuffmanBuffers *>(tmp_buffers.all_buffer_ptr),
    tmp_buffers.global_ix_block_huff,
    tmp_buffers.global_block_count
  );
  CUDA_CHECK(cudaGetLastError());

  // Then a kernel to fill in sequences. Sequence kernel will fill in the final compressed size.
  sequence_compression_kernel<<<cuda_dim_cast(ans_grid_dim), block_size, 0, stream>>>(
    tmp_buffers.block_shares,
    reinterpret_cast<zstd::SequenceCompressBuffers *>(tmp_buffers.all_buffer_ptr),
    tmp_buffers.global_ix_block_ans,
    tmp_buffers.global_block_count,
    reinterpret_cast<uint8_t *>(tmp_buffers.stage_buffer),
    rounded_max_chunk_size,
    batch_size
  );
  CUDA_CHECK(cudaGetLastError());

  compact_compressed_frames<<<narrow_cast<int>(batch_size), 512, 0, stream>>>(
    tmp_buffers.block_shares,
    input_decomp_sizes,
    compressed_sizes
  );
  CUDA_CHECK(cudaGetLastError());
}

size_t compress_compute_temp_size(
  const size_t batch_size,
  const size_t max_chunk_size,
  const size_t max_total_uncomp_size,
  const int num_sms
)
{
  const size_t rounded_max_chunk_size = roundUpTo(max_chunk_size, size_t{4});

  size_t lz_grid_dim, ans_grid_dim, huff_grid_dim;
  get_grid_dims(batch_size, num_sms, lz_grid_dim, huff_grid_dim, ans_grid_dim);

  CompressionScratchHandle tmp_buffers{
    nullptr /*tmp buffer*/,
    batch_size,
    lz_grid_dim,
    ans_grid_dim,
    huff_grid_dim,
    rounded_max_chunk_size,
    0 /*tmp buffer size*/,
    max_total_uncomp_size,
    false /*compute total uncomp size*/
  };

  return tmp_buffers.current_size;
}

size_t compute_max_comp_output_size(const size_t max_chunk_size)
{
  // If not compressible, the maximum chunk will be split into blocks and stored raw
  const size_t num_chunks = max(roundUpDiv(max_chunk_size, COMPRESS_NOMINAL_BLOCK_SIZE), size_t{1});

  // This computes an upper bound, and we're always writing the fcs header.
  return max_chunk_size + compute_frame_header_size(max_chunk_size) + ZSTD_BLOCK_HEADER_SIZE * num_chunks;
}

} // end namespace zstd
