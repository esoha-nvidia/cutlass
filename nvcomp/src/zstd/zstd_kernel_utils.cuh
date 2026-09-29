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

#pragma once

#include "ANSSequenceDecoder.cuh"
#include "constants.cuh"
#include "EntropyTables.cuh"
#include "HuffDecoder.cuh"
#include "huffman.cuh"
#include "io.cuh"
#include "nvcomp/shared_types.h"
#include "types.cuh"
#include "utils.cuh"

// #define PRINT_TIMING 1
// #define HUFF_LOGGING 1

namespace zstd
{

// Note: this is currently only used for testing Huffman encoding in "zstd_huffman_test.cpp"
inline __device__ void decompress_huffman_single_stream(
  const uint8_t *src,
  const HuffmanTable &huf_table,
  uint8_t *literal_buffer,
  const unsigned len,
  unsigned mask,
  unsigned num_literals
)
{
  const int padding = 8 - highest_set_bit(src[len - 1]);
  int32_t bit_offset = len * 8 - padding;

  __shared__ unsigned shared_val[WARP_SIZE];
  HuffCache huff_cache(shared_val[thread_warp_ix()]);
  huff_cache.init(src, bit_offset);

  uint16_t state = huff_cache.read(huf_table.max_bits, mask);

  const uint16_t *basecodes = huf_table.basecodes;
  const uint8_t *offsets = huf_table.offsets;
  const uint8_t *symbols = huf_table.symbols;
  const int max_bits = huf_table.max_bits;

  // Can pre-load these registers to avoid many shared memory loads during tight loop binary search
  const unsigned bc2 = basecodes[2] | (basecodes[1] << 16);
  const unsigned bc3 = basecodes[3];
  const unsigned bc5 = basecodes[5] | (basecodes[4] << 16);
  const unsigned bc6 = basecodes[6];
  const unsigned bc8 = basecodes[8] | (basecodes[7] << 16);
  const unsigned bc9 = basecodes[9];
  const unsigned bc11 = basecodes[11] | (basecodes[10] << 16);
  const unsigned of0 = (offsets[11]) + (offsets[5] << 8) + (offsets[8] << 16) + (offsets[2] << 24);
  const unsigned of25 = (offsets[4]) + (offsets[3] << 8) + (offsets[1] << 16) + (offsets[0] << 24);
  const unsigned of811 = (offsets[10]) + (offsets[9] << 8) + (offsets[7] << 16) + (offsets[6] << 24);

  int ix_lit = 0;
  while (true)
  {
    bool finished = ix_lit >= num_literals;
    mask = __ballot_sync(mask, not finished);
    if (finished)
    {
      return;
    }
    literal_buffer[ix_lit++] = decode_huffman_symbol(
      state,
      src,
      mask,
      symbols,
      max_bits,
      huff_cache,
      bc2,
      bc3,
      bc5,
      bc6,
      bc8,
      bc9,
      bc11,
      of0,
      of25,
      of811,
      true /*valid*/
    );
#ifdef HUFF_LOGGING
    if (ix_lit < 32)
    {
      printf(
        "thread %d ix lit %d bit offset %d symbol %u\n",
        threadIdx.x,
        ix_lit,
        bit_offset,
        literal_buffer[ix_lit - 1]
      );
    }
#endif
  }
}

template <bool partial_parse = false>
inline __device__ void decode_literal_header(
  BitReader &bit_reader,
  BlockHeader *literal_block_header,
  LiteralBlockType *literal_block_type = nullptr,
  unsigned int *literal_block_regenerated_size = nullptr,
  unsigned int *literal_block_compressed_size = nullptr
)
{
  unsigned initial_rem_bytes = bit_reader.rem_bytes();
  uint8_t block_type = bit_reader.read_bits(2);
  uint8_t size_format = bit_reader.read_bits(2);
  [[maybe_unused]] uint8_t num_streams = 4;

  // Initialization for coverity
  unsigned regen_size{0};
  unsigned compressed_size{0};
  if (block_type <= 1)
  {
    // Raw or RLE literals block
    switch (size_format)
    {
      case 0:
      case 2:
        // Size_Format uses 1 bit. Rewind a bit since we already read 2 bits to get here.
        bit_reader.rewind_bits(1);
        regen_size = bit_reader.read_bits(5);
        break;
      case 1:
        regen_size = bit_reader.read_bits(12);
        break;
      case 3:
        regen_size = bit_reader.read_bits(20);
        break;
      default:
        assert(false); // not possible.
        break;
    }

    if (block_type == 0)
    {
      // Raw block
      compressed_size = regen_size;
    }
    else
    {
      // RLE block
      compressed_size = 1;
    }
  }
  else
  {
    // Huffman compressed literals
    switch (size_format)
    {
      case 0:
        num_streams = 1;
        [[fallthrough]];
      case 1:
        regen_size = bit_reader.read_bits(10);
        compressed_size = bit_reader.read_bits(10);
        break;
      case 2:
        regen_size = bit_reader.read_bits(14);
        compressed_size = bit_reader.read_bits(14);
        break;
      case 3:
        regen_size = bit_reader.read_bits(18);
        compressed_size = bit_reader.read_bits(18);
        break;
      default:
        assert(false); // not possible.
        break;
    }
  }

  assert(regen_size <= MAX_LITERALS_SIZE);
  assert(compressed_size <= MAX_LITERALS_SIZE);
  assert(bit_reader.is_stream_aligned());

  if constexpr (partial_parse)
  {
    *literal_block_type = static_cast<LiteralBlockType>(block_type);
    *literal_block_regenerated_size = regen_size;
    *literal_block_compressed_size = compressed_size;
  }
  else
  {
    unsigned final_rem_bytes = bit_reader.rem_bytes();
    unsigned literal_header_size = initial_rem_bytes - final_rem_bytes;

    if (thread_warp_ix() == 0)
    {
      literal_block_header->seq_section_start = literal_header_size + compressed_size + 3;
      literal_block_header->literal_header_size = literal_header_size;
      literal_block_header->literal_header.compressed_size = compressed_size;
      literal_block_header->literal_header.regenerated_size = regen_size;
      literal_block_header->literal_header.block_type = static_cast<LiteralBlockType>(block_type);
      literal_block_header->literal_header.num_streams = num_streams;
    }
  }
}

constexpr size_t compute_per_cta_tmp_buffer() { return compute_fse_table_alloc() + compute_huff_table_alloc(); }

inline __device__ bool
gather_block_scratch_requirement(const uint8_t *&block_input, size_t &scratch_bytes, nvcompStatus_t *device_status)
{
  // Note:
  // The size of Block_Content is limited by Block_Maximum_Size,
  // which is min(Window_Size, 128 KiB).
  BitReader bit_reader{block_input, 128 * 1024};
  bool last_block = static_cast<bool>(bit_reader.read_bits(1));
  BlockType block_type = static_cast<BlockType>(bit_reader.read_bits(2));
  const unsigned comp_block_size = bit_reader.read_bits(21) + ZSTD_BLOCK_HEADER_SIZE;

  if (block_type == BlockType::Raw)
  {
    // Raw block
    // Note: neither an FSE, nor a Huffman literal block
    block_input += comp_block_size;
  }
  else if (block_type == BlockType::RLE)
  {
    // RLE Block. Outputs a byte N times.
    // Note: neither an FSE, nor a Huffman literal block
    block_input += 1 + 3;
  }
  else if (block_type == BlockType::Compressed)
  {
    // Regular compressed block:
    // Consists of 2 sections:
    // - Literals section
    // - Sequences section

    // 1) Process `Literals` section
    LiteralBlockType literals_block_type;
    unsigned int literals_regenerated_size;
    unsigned int literals_compressed_size;
    decode_literal_header<true>(
      bit_reader,
      nullptr,
      &literals_block_type,
      &literals_regenerated_size,
      &literals_compressed_size
    );

    // Is it a Huffman literal block?
    switch (literals_block_type)
    {
      case LiteralBlockType::Raw:
        // Nope, simply fast-forward bit reader, nothing to do
        bit_reader.get_read_pointer(literals_compressed_size);
        break;
      case LiteralBlockType::RLE:
        // Nope, simply fast-forward bit reader by a single byte (block content)
        bit_reader.get_read_pointer(1);
        break;
      case LiteralBlockType::Compressed:
      case LiteralBlockType::Treeless:
        // Yes, it contains Huffman prefix code.
        scratch_bytes += literals_regenerated_size + sizeof(HuffmanTable) + (alignof(HuffmanTable) - 1);
        // Fast-forward bit reader with the compressed size of the literals
        // section.
        bit_reader.get_read_pointer(literals_compressed_size);
        break;
      default:
        assert(false); // not possible.
        break;
    }

    // 2) Process `Sequences` section
    // Is it an FSE block? Are there any sequences?
    int num_sequences = decode_num_sequences(bit_reader);
    if (num_sequences > 0)
    {
      // Note:
      // The compressed bitstream actually contains the `Accuracy_Log`,
      // that determines the fse table needed for literal length,
      // offset, and match length. This could be additionally parsed to
      // lower our scratch space needs even further.
      // TODO(bnagy): parse the `Accuracy_log` (1st byte of FSE tables)
      //              and allocate accordingly.
      scratch_bytes +=
        sizeof(sequence) * num_sequences + (alignof(sequence) - 1) + sizeof(FSETables) + (alignof(FSETables) - 1) +
        sizeof(uint32_t) * fse_table_size_words(LITERAL_LENGTH_MAX_ACCURACY, 0) + (alignof(uint32_t) - 1) +
        sizeof(uint32_t) * fse_table_size_words(OFFSET_MAX_ACCURACY, 1) + (alignof(uint32_t) - 1) +
        sizeof(uint32_t) * fse_table_size_words(MATCH_LENGTH_MAX_ACCURACY, 2) + (alignof(uint32_t) - 1);
    }
    block_input += comp_block_size;
  }
  else
  {
    // Any other block type (Reserved) currently is considered corrupted data.
    assert(false);
    if (device_status)
    {
      *device_status = nvcompErrorCannotDecompress;
    }
  }

  return last_block;
}

inline __device__ bool classify_blocks(const uint8_t *block_input, DeviceBlockShare &block_share)
{
  // Note:
  // The size of Block_Content is limited by Block_Maximum_Size,
  // which is min(Window_Size, 128 KiB).
  BitReader bit_reader{block_input, 128 * 1024};
  BlockHeader &block_header = block_share.block_header;
  // Compute a shared block header
  bool last_block = static_cast<bool>(bit_reader.read_bits(1)); // Already know how many blocks there are.
  uint8_t block_type = static_cast<uint8_t>(bit_reader.read_bits(2));
  const unsigned comp_block_size = bit_reader.read_bits(21) + ZSTD_BLOCK_HEADER_SIZE;

  if (thread_warp_ix() == 0)
  {
    block_header.block_type = block_type;
    block_share.lz_status = 0;
    block_share.compressed_buffer = block_input;
    block_share.comp_block_size = comp_block_size;
    block_share.decode_seq_count = 0;
    block_share.decode_lit_count = 0;
    block_share.rle_byte = 0;
    block_share.copy_buffer_ptr = nullptr;
    block_share.literal_buffer = nullptr;
  }
  __syncwarp();

  switch (block_type)
  {
    case 0: {
      // Raw block
      // Just a warp copy of the input to the output.
      const uint8_t *copy_buffer_ptr = bit_reader.get_read_pointer(comp_block_size - 3);
      if (thread_warp_ix() == 0)
      {
        block_share.copy_buffer_ptr = copy_buffer_ptr;
        block_share.decomp_block_size = comp_block_size - 3;

        block_share.is_compressed_block = false;
        block_share.is_fse_block = false;
        block_share.is_huff_literal_block = false;
        block_share.is_rle_block = false;
        block_share.is_literal_rle_block = false;
        block_share.is_raw_block = true;
      }
    }
    break;
    case 1:
      // RLE Block. Outputs a byte N times.
      {
        uint8_t input_byte = bit_reader.read_bits(8);
        if (thread_warp_ix() == 0)
        {
          block_share.rle_byte = input_byte;
          block_share.is_compressed_block = false;
          // For RLE blocks, the comp_block_size read above (using 3 bytes)
          // indicates the number of times to repeat the RLE byte. The compressed
          // block size is actually 1 byte (the RLE byte to copy). The comp block size in
          // the block_share includes the 3 bytes read in as part of the header, for a total
          // of 4 bytes for this block.
          block_share.decomp_block_size = comp_block_size - 3;
          block_share.comp_block_size = 1 + 3;
          block_share.is_fse_block = false;
          block_share.is_huff_literal_block = false;
          block_share.is_rle_block = true;
          block_share.is_literal_rle_block = false;
          block_share.is_raw_block = false;
        }
      }
      break;
    case 2:
      // Regular compressed block
      // Parse the LiteralHeader, find the start of the sequence section.
      decode_literal_header(bit_reader, &block_header);
      if (thread_warp_ix() == 0)
      {
        if (block_header.literal_header.block_type == LiteralBlockType::Compressed)
        {
          block_share.is_huff_literal_block = true;
        }
        else if (block_header.literal_header.block_type == LiteralBlockType::Treeless)
        {
          block_share.is_huff_literal_block = true;
          // Check whether the table is repeated. If so, use the last valid.
        }
        else
        {
          block_share.is_huff_literal_block = false;
        }
        block_share.is_compressed_block = true;
        block_share.is_literal_rle_block = false;
        block_share.is_raw_block = false;
        block_share.is_rle_block = false;
      }

      __syncwarp(); // Need to have set this already
      if (not block_share.is_huff_literal_block)
      {
        if (block_header.literal_header.block_type == LiteralBlockType::Raw)
        {
          // Raw literals
          const uint8_t *copy_buffer_ptr = bit_reader.get_read_pointer(block_header.literal_header.compressed_size);
          if (thread_warp_ix() == 0)
          {
            block_share.num_literals = block_header.literal_header.compressed_size;
            block_share.copy_buffer_ptr = copy_buffer_ptr;
          }
        }
        else
        {
          uint8_t rle_byte = bit_reader.read_bits(8);
          if (thread_warp_ix() == 0)
          {
            block_share.is_literal_rle_block = true;
            block_share.num_literals = block_header.literal_header.regenerated_size;
            block_share.rle_byte = rle_byte;
          }
        }
        if (thread_warp_ix() == 0)
        {
          block_share.decode_lit_count = block_share.num_literals;
        }
      }
      else
      {
        if (thread_warp_ix() == 0)
        {
          block_share.num_literals = block_header.literal_header.regenerated_size;
        }
        bit_reader.get_read_pointer(block_header.literal_header.compressed_size);
      }

      // Check how many sequences
      const int prev_rem_bytes = bit_reader.rem_bytes();
      const int num_sequences = decode_num_sequences(bit_reader);
      const int new_rem_bytes = bit_reader.rem_bytes();
      const int num_seq_count_bytes = prev_rem_bytes - new_rem_bytes;

      if (thread_warp_ix() == 0)
      {
        block_share.block_header.seq_section_start += num_seq_count_bytes;

        block_share.is_fse_block = num_sequences > 0; // if zero, no sequences, no fse decoding to do.
        if (not block_share.is_fse_block)
        {
          block_share.total_seq_bytes = 0;
          block_share.decomp_block_size = block_header.literal_header.regenerated_size;
        }

        block_share.num_sequences = num_sequences;
      }
  }

  return last_block;
}

// This is necessary to safely increment the shared value
inline __device__ void increment_ix(int &ix_block, int *global_ix_block)
{
  __syncwarp(); // ensure threads are done with prior value of ix_block
  if (thread_warp_ix() == 0)
  {
    ix_block = atomicAdd(global_ix_block, 1);
  }
  __syncwarp(); // ensure threads have the value of ix_block prior to continuing
}

inline __device__ void warp_entropy_sequence_decoding(
  DeviceBlockShare *block_shares,
  uint32_t *table_buff,
  SharedExtraBitTables &extra_tables,
  int num_ans_tables,
  int *global_ix_block,
  const int num_blocks
)
{
  int ix_fse_warp = ix_warp() < NUM_FSE_WARPS_PER_CTA ? ix_warp() : NUM_FSE_WARPS_PER_CTA;

  __shared__ int shared_shared_ix_block[MAX_FSE_WARPS_PER_CTA];
  int &shared_ix_block = shared_shared_ix_block[ix_fse_warp];

  __shared__ Uint8WarpScan::TempStorage uint8_storage[MAX_FSE_WARPS_PER_CTA];
  Uint8WarpScan uint8_warp_scan{uint8_storage[ix_fse_warp]};

  __shared__ Uint16WarpScan::TempStorage uint16_storage[MAX_FSE_WARPS_PER_CTA];
  Uint16WarpScan uint16_warp_scan{uint16_storage[ix_fse_warp]};

  __shared__ WarpReduceUint16::TempStorage uint16_reduce_storage[MAX_FSE_WARPS_PER_CTA];
  WarpReduceUint16 uint16_warp_reduce{uint16_reduce_storage[ix_fse_warp]};

  const int ix_thread_zstd_block = thread_warp_ix() / NUM_THREADS_PER_ANS_BLOCK;
  const int ix_in_block = thread_warp_ix() % NUM_THREADS_PER_ANS_BLOCK;
  const int ix_base_thread = ix_thread_zstd_block * NUM_THREADS_PER_ANS_BLOCK;

  ANSCache this_cache;
  if (ix_thread_zstd_block < num_ans_tables)
  {
    this_cache = ANSCache{ix_thread_zstd_block};
  }

  __shared__ unsigned shared_shared_words[MAX_FSE_WARPS_PER_CTA][2 * MAX_ANS_TABLES];
  unsigned *shared_words = shared_shared_words[ix_fse_warp];

  ANSSequenceDecoder decoder{this_cache, ix_in_block, ix_base_thread, extra_tables, ix_thread_zstd_block, shared_words};

  uint8_t num_ans_blocks = 0;
  bool active = false;
  bool more_blocks_to_decode = true;
  FSETables fse_tables;
  while (num_ans_blocks < num_ans_tables)
  {

    increment_ix(shared_ix_block, global_ix_block);

    if (shared_ix_block >= num_blocks)
    {
      more_blocks_to_decode = false;
      break;
    }

    auto &block_share = block_shares[shared_ix_block];
    if (block_share.is_fse_block)
    {
      uint32_t *this_table_buff = table_buff + max_fse_tables_words * num_ans_blocks;
      FSETables new_fse_tables{*block_share.global_fse_tables, this_table_buff};
      if (ix_thread_zstd_block == num_ans_blocks)
      {
        fse_tables = new_fse_tables;
        decoder.this_block_share = &block_share;
      }

      // Construct the tables, construct a sequence decoder. This is done per thread.
      ++num_ans_blocks;
    }
  }

  __syncwarp();

  if (num_ans_blocks <= 0)
  {
    return;
  }

  int seq_count = std::numeric_limits<int>::max();
  if (ix_thread_zstd_block < num_ans_blocks)
  {
    decoder.init(fse_tables);
    active = true;
    seq_count = decoder.num_sequences;
  }
  __syncwarp();

  while (true)
  {
    // Do partial decoding, refill as necessary
    int ix_next = -1; // initialize for coverity
    int num_sequences;

    // If we don't have more blocks to decode, loop until we do
    unsigned active_mask = __ballot_sync(WARP_ALL, active);
    assert(
      active_mask
    ); // There is at least one active thread. Makes sure ix_next will be initialized inside the while loop
    while (active_mask)
    {
#ifdef STAGE_LOGGING
      print0(
        "bid %d warp %d decoding, active count %d clock %lu\n",
        blockIdx.x,
        ix_warp(),
        __popc(active_mask) / NUM_THREADS_PER_ANS_BLOCK,
        cuda::std::chrono::system_clock::now()
      );
#endif
      minANSBlockIndex(ix_next, num_sequences, seq_count);
      __syncwarp();
      decoder.decode_sequences(num_sequences, active);
      seq_count -= num_sequences;

      active_mask ^= ((1 << NUM_THREADS_PER_ANS_BLOCK) - 1)
                     << (NUM_THREADS_PER_ANS_BLOCK * ix_next); // Null out the appropriate bits

      if (ix_thread_zstd_block == ix_next)
      {
        assert(active == true); // Makes sure the thread is active and decoder was initialized
        decoder.finish_block();
        assert(seq_count == 0);
        active = false;
        seq_count = std::numeric_limits<int>::max();
      }

      if (more_blocks_to_decode)
      {
        break;
      }
    }

    if (not more_blocks_to_decode)
    {
      return;
    }

    --num_ans_blocks;
    assert(ix_next >= 0);

    // Try to get another
    while (true)
    {
      increment_ix(shared_ix_block, global_ix_block);

      if (shared_ix_block >= num_blocks)
      {
        more_blocks_to_decode = false;
        break;
      }

      auto &block_share = block_shares[shared_ix_block];
      if (block_share.is_fse_block)
      {
        uint32_t *this_table_buff = table_buff + max_fse_tables_words * ix_next;
        FSETables fse_tables{*block_share.global_fse_tables, this_table_buff};

        __syncwarp();

        // Construct the tables, construct a sequence decoder. This is done per thread.
        ++num_ans_blocks;

        __syncwarp();
        if (ix_thread_zstd_block == ix_next)
        {
          active = true;
          decoder.this_block_share = &block_share;
          decoder.init(fse_tables);
          seq_count = decoder.num_sequences;
        }

        __syncwarp();
        break;
      }
    }
  }
}

inline __device__ void warp_entropy_literal_decoding(
  DeviceBlockShare *block_shares,
  MetaHuffmanTable &meta_huff_table,
  int *global_ix_block,
  const int num_blocks
)
{
  __shared__ DeviceBlockShare *shared_huff_block_shares[NUM_HUFF_WARPS_PER_CTA][MAX_HUFF_TABLES];
  DeviceBlockShare **huff_block_shares = shared_huff_block_shares[(ix_warp() - NUM_FSE_WARPS_PER_CTA)];
  __shared__ int shared_ix_block[NUM_HUFF_WARPS_PER_CTA];
  int &ix_block = shared_ix_block[ix_warp() - NUM_FSE_WARPS_PER_CTA];
  __shared__ unsigned shared_shared_vals[NUM_HUFF_WARPS_PER_CTA][WARP_SIZE];
  unsigned *shared_vals = shared_shared_vals[ix_warp() - NUM_FSE_WARPS_PER_CTA];
  constexpr int IMPOSSIBLY_LARGE_LIT_COUNT = std::numeric_limits<int>::max();
  constexpr unsigned FIRST_THREAD_MASK = 0x11111111;
  unsigned &shared_val = shared_vals[thread_warp_ix()];

  const int ix_thread_zstd_block = thread_warp_ix() / NUM_THREADS_PER_HUFF_BLOCK;
  const int ix_in_block = thread_warp_ix() % NUM_THREADS_PER_HUFF_BLOCK;
  int num_huff_blocks = 0;

  while (num_huff_blocks < MAX_HUFF_TABLES)
  {
    increment_ix(ix_block, global_ix_block);

    if (ix_block >= num_blocks)
    {
      break;
    }

    auto &block_share = block_shares[ix_block];

    if (not block_share.is_huff_literal_block)
    {
      continue;
    }

    huff_block_shares[num_huff_blocks] = &block_share;

    meta_huff_table.deepcopy(*block_share.global_huff_table, num_huff_blocks);
    ++num_huff_blocks;
  }

  __syncwarp();

  HuffDecoder decoder{ix_in_block, shared_val};

  // Initialize those that were found
  if (ix_thread_zstd_block < num_huff_blocks)
  {
    decoder.init(*huff_block_shares[ix_thread_zstd_block], meta_huff_table, ix_thread_zstd_block);
  }

  // We've filled the queue as much as possible
  // Note:
  // Given we have NUM_HUFF_WARPS_PER_CTA warps simultaneously competing for global_ix_block,
  // we have a benign race reading *global_ix_block. Given global_ix_block is well-aligned, and
  // 32-bit wide, it is considered an atomic (non-torn) read, but it is potentially served from L1.
  bool final_iters = num_huff_blocks != MAX_HUFF_TABLES or *global_ix_block >= num_blocks;

  while (num_huff_blocks > 0)
  {
    unsigned full_mask = __ballot_sync(WARP_ALL, decoder.is_active());
    // Do partial decoding, refill as necessary
    int lit_count = decoder.is_active() and decoder.num_rem_literals() > 0 ? decoder.num_rem_literals()
                                                                           : IMPOSSIBLY_LARGE_LIT_COUNT;
    int num_literals = reduce_min(lit_count);

    if (decoder.is_active())
    {
      // We don't want to finish *part* of a block. We also don't want to back out of decoding
      // and then immediately back out again. Here, as long as any thread will have < 3 literals remaining
      // after the specified number of literals is computed, increment the number of literals by 3.
      while (__any_sync(
        full_mask,
        decoder.num_rem_literals() > num_literals and decoder.num_rem_literals() - num_literals <= 3
      ))
      {
        num_literals += 3;
      }

      decoder.decode_literals(num_literals, full_mask);
    }

    full_mask = __ballot_sync(WARP_ALL, decoder.is_active() and decoder.num_rem_literals() > 0);
    if (not final_iters)
    {
      // Note:
      // Given we have NUM_HUFF_WARPS_PER_CTA warps simultaneously competing for global_ix_block,
      // we have a benign race reading *global_ix_block. Given global_ix_block is well-aligned, and
      // 32-bit wide, it is considered an atomic (non-torn) read, but it is potentially served from L1.
      if (*global_ix_block >= num_blocks)
      {
        final_iters = true;
      }
    }

    unsigned ix_unfilled = (~full_mask) & FIRST_THREAD_MASK;

    if (decoder.is_active() and (ix_unfilled & (1 << (ix_thread_zstd_block * NUM_THREADS_PER_HUFF_BLOCK))))
    {
      decoder.finish_block();
    }
    num_huff_blocks = __popc(full_mask & FIRST_THREAD_MASK);

    // Try to reload
    while (not final_iters and ix_unfilled)
    {
      increment_ix(ix_block, global_ix_block);

      if (ix_block >= num_blocks)
      {
        final_iters = true;
        break;
      }

      auto &block_share = block_shares[ix_block];

      if (not block_share.is_huff_literal_block)
      {
        continue;
      }

      // Get the next value
      int ix_thread_next = __ffs(ix_unfilled) - 1;
      int ix_block_next = ix_thread_next / NUM_THREADS_PER_HUFF_BLOCK;
      ix_unfilled ^= 1U << ix_thread_next;

      huff_block_shares[ix_block_next] = &block_share;

      meta_huff_table.deepcopy(*block_share.global_huff_table, ix_block_next);

      __syncwarp();
      if (ix_thread_zstd_block == ix_block_next)
      {
        assert(not decoder.is_active());
        decoder.init(*huff_block_shares[ix_block_next], meta_huff_table, ix_block_next);
      }

      ++num_huff_blocks;
    }
  }
}

} // end namespace zstd
