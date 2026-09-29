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
// #define PRINT_TIMING
// #define ANS_PRECOMPUTE_OFFSET

#include <cub/cub.cuh>

#include <cuda/atomic>
#include <cuda/barrier>
#include <cuda_pipeline.h>

#include <memory>

#include "constants.cuh"
#include "io.cuh"
#include "simple_types.cuh"
#include "utils.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

namespace cg = cooperative_groups;

namespace zstd
{

enum class SequenceType
{
  LiteralLength = 0,
  Offset,
  MatchLength,
  Invalid
};

enum class BlockType : uint8_t
{
  Raw = 0,
  RLE,
  Compressed,
  Reserved
};

enum class LiteralBlockType : uint8_t
{
  Raw = 0,
  RLE,
  Compressed,
  Treeless
};

typedef nvcomp::cub::WarpReduce<uint16_t> WarpReduceUint16;
typedef nvcomp::cub::WarpReduce<unsigned> WarpReduceUnsigned;
typedef nvcomp::cub::WarpReduce<int> WarpReduceInt;
typedef nvcomp::cub::WarpReduce<int16_t> WarpReduceInt16;
typedef nvcomp::cub::WarpScan<int32_t> IntWarpScan;
typedef nvcomp::cub::WarpScan<unsigned> UnsignedWarpScan;
typedef nvcomp::cub::WarpScan<uint8_t> Uint8WarpScan;
typedef nvcomp::cub::WarpScan<uint16_t> Uint16WarpScan;

class HuffCache
{
private:
  uint16_t size_bits_;
  const uint32_t *src_;
  int src_offset_;
  uint64_t buf_;
  unsigned &shared_val;

public:
  __device__ HuffCache(unsigned &shared_val)
      : size_bits_(0)
      , src_(nullptr)
      , src_offset_(0)
      , buf_(0)
      , shared_val(shared_val)
  {}

  __device__ void init(const uint8_t *src, size_t bit_offset)
  {
    src += (bit_offset + 7) / 8;

    uint8_t loadrem = reinterpret_cast<uintptr_t>(src) % 4;
    uint8_t n_bytes = 4 + loadrem;
    src -= n_bytes;

    buf_ = 0;
    for (int i = 0; i < n_bytes; ++i)
    {
      buf_ |= static_cast<uint64_t>(src[i]) << (i * 8);
    }

    size_bits_ = n_bytes * 8;
    if (bit_offset % 8)
    {
      size_bits_ -= 8 - (bit_offset % 8);
    }

    src_ = reinterpret_cast<const uint32_t *>(src);
    src_offset_ = bit_offset - size_bits_;

    if (src_offset_ > 0)
    {
      __pipeline_memcpy_async(&shared_val, --src_, sizeof(unsigned));
      __pipeline_commit();
    }
  }

  __device__ uint16_t read(uint8_t n_bits, uint32_t mask)
  {
    bool can_read = src_offset_ > 0 and size_bits_ <= 32;
    bool must_read = can_read and size_bits_ < n_bits;
    bool any_must_read = __any_sync(mask, must_read);
    if (can_read and any_must_read)
    {
      __pipeline_wait_prior(0);
      buf_ <<= 32;
      buf_ |= shared_val;
      src_offset_ -= 32;
      if (src_offset_ > 0)
      {
        __pipeline_memcpy_async(&shared_val, --src_, sizeof(unsigned));
        __pipeline_commit();
      }

      size_bits_ += 32;
    }

    uint8_t n_read = min(n_bits, size_bits_ + src_offset_);

    uint16_t bits = ((buf_ >> (size_bits_ - n_read)) & ((1 << n_read) - 1)) << (n_bits - n_read);

    size_bits_ -= n_read;

    return bits;
  };
};

class ANSCache
{
private:
  const uint8_t *src; // No guaranteed alignment
  const uint32_t *input_buffer;
  int ix_last_word;

public:
  uint32_t num_bits_of_front_padding;
  uint32_t num_bits_end_padding;
  int total_bits_remaining;
  int buffer_size;

public:
  __device__ ANSCache(int ix_block)
      : src(nullptr)
      , total_bits_remaining(-1)
      , num_bits_of_front_padding(0)
      , ix_last_word(0)
  {}

  ANSCache() = default;
  ~ANSCache() = default;

  __device__ void init_cache(
    const uint8_t *src_ /* buffer pointer (unaligned) */,
    const int buffer_size_ /* in bytes */,
    int ix_in_block /* unused */
  )
  {
    assert(buffer_size_ != 0);
    buffer_size = buffer_size_;
    src = src_;
    num_bits_end_padding = 8 - highest_set_bit(src[buffer_size - 1]);

    const uint8_t src_alignment = (uintptr_t)src % 4;
    num_bits_of_front_padding = src_alignment * 8;

    total_bits_remaining = buffer_size * 8 + num_bits_of_front_padding - num_bits_end_padding;
    input_buffer = reinterpret_cast<const uint32_t *>(src - src_alignment);
    ix_last_word = (total_bits_remaining + 31) / 32 - 1;
  }

  // Read `num_bits` number of bits from the location specified by reg1 and reg2
  // Notes:
  // - reg1: control register that defines how many bits of data thread [0, 1, 2, 3] is going to read
  // - reg2: control register that defines how many bits of data thread [4, 5] is going to read
  //   ^ We have these for calculating a sum scan, so each of the threads know where to read
  // - num_bits: provisional number of bits the current particular thread needs to read
  __device__ uint32_t read(unsigned reg1, unsigned reg2, int num_bits, int ix_in_block)
  {
    assert(num_bits <= 32);

    uint8_t full_num_bits = 0; // same for all (6) threads: the sum of the 8-bit sections of reg1 and reg2
    uint8_t this_start_bit = custom_scan_num_bits(full_num_bits, reg1, reg2, ix_in_block);
    int bit_offset = total_bits_remaining - this_start_bit; // read from the back, hence the inclusive sum scan

    // ZSTD allows bits to be read outside of the buffer. They must be 0
    // The actual number of bits read is shrunk when reading at the edge of the buffer
    int actual_num_bits = num_bits;
    if (bit_offset < num_bits_of_front_padding)
    {
      actual_num_bits += (bit_offset - num_bits_of_front_padding); // reducing the actual number of bits to be read
      bit_offset = num_bits_of_front_padding;
      actual_num_bits = max(actual_num_bits, 0);
    }

    int word_offset = bit_offset / 32;
    // This complexity is needed because the last word might be at the end of the buffer.
    // Doing a word-level read could OOB
    uint32_t word1 = 0;
    uint32_t word2 = 0;
    if (word_offset < ix_last_word - 1)
    {
      word1 = input_buffer[word_offset];
      word2 = input_buffer[word_offset + 1];
    }
    else
    {
      // We need to do special handling for the last word
      uint32_t last_word = 0;
      int bytes_in_last_word = (buffer_size + num_bits_of_front_padding / 8 - num_bits_end_padding / 8) % 4;
      if (bytes_in_last_word == 0)
      {
        bytes_in_last_word = 4;
      }

      const uint32_t *last_word_ptr = &input_buffer[ix_last_word];
      if (bytes_in_last_word == 4)
      {
        last_word = *last_word_ptr;
      }
      else
      {
        const uint8_t *last_bytes_ptr = reinterpret_cast<const uint8_t *>(last_word_ptr);
        for (int ix = 0; ix < bytes_in_last_word; ++ix)
        {
          last_word = last_word + ((last_bytes_ptr[ix]) << (ix * 8));
        }
      }

      if (word_offset == ix_last_word)
      {
        word1 = last_word;
      }
      else
      {
        word1 = input_buffer[ix_last_word - 1];
        word2 = last_word;
      }
    }

    uint32_t mask = (1 << actual_num_bits) - 1;
    uint32_t read_val = mask & __funnelshift_r(word1, word2, bit_offset % 32);

    total_bits_remaining -= full_num_bits;

    return read_val;
  }
};

struct LiteralHeader
{
  LiteralBlockType block_type;
  unsigned regenerated_size;
  unsigned compressed_size;
  uint8_t num_streams;
  unsigned table_desc_size;
};

struct BlockHeader
{
  uint8_t block_type;
  unsigned seq_section_start;
  LiteralHeader literal_header;
  unsigned literal_header_size;
  unsigned block_seq_table_size;
};

struct FSETables;
struct HuffmanTable;

struct DeviceBlockShare
{
  uint8_t *literal_buffer;
  sequence *sequence_buffer;
  const uint8_t *copy_buffer_ptr;
  // When Block_Type == Compressed_Block && the literals are compressed
  bool is_huff_literal_block;
  // When Block_Type == Compressed_Block && the literals consist of a single byte (.rle_byte) repeated (.num_literals) times
  bool is_literal_rle_block;
  // When Block_Type == Compressed_Block
  bool is_compressed_block;
  // When Block_Type == Compressed_Block && more than 0 sequences to decode
  bool is_fse_block;
  cuda::atomic<int, cuda::thread_scope_device> lz_status;
  cuda::atomic<int, cuda::thread_scope_device> decode_seq_count;
  cuda::atomic<int, cuda::thread_scope_device> decode_lit_count;
  int total_seq_bytes;
  // When Block_Type == Raw_Block
  bool is_raw_block;
  // When Block_Type == RLE_Block
  bool is_rle_block;
  const uint8_t *compressed_buffer;
  int32_t comp_block_size;
  int32_t num_sequences;
  int32_t num_literals;
  const uint8_t *input_frame;
  uint8_t *output_frame;
  int32_t ix_frame;
  int32_t ix_block;
  // RLE byte to duplicate in the decompressed stream
  uint8_t rle_byte;
  BlockHeader block_header;
  uint8_t *block_gpu_dst;
  int decomp_block_size; // A decompressed block cannot be larger than 128KiB
  DeviceBlockShare *next_block_share;
  FSETables *global_fse_tables;
  HuffmanTable *global_huff_table;
  int repeat_offsets[3];
};

// Allows for a custom number of bits to be used for the offset
struct CompressSequenceBuffer
{
  // Note:
  // The underlying buffer structure looks as follows:
  // <all match lens> <all lit lens> <all offsets>
  //                  ^
  //
  // To make copying faster, we copy in 4-byte units, and this
  // requires that the beginning of the literal length section (^) must
  // start at a 4-byte boundary. We can easily achieve this with
  // a maximum sequence count that is even.

  int max_sequence_count; // maximum sequence count storable in this buffer
  int sequence_count; // current sequence count

  uint8_t *data_buffer; // buffer holding `max_sequence_count` match and literal lengths
  uint32_t *base_offset; // buffer holding `max_sequence_count` offsets

  CompressSequenceBuffer() = default;

  inline __device__ CompressSequenceBuffer(uint8_t *data_buffer, const int max_sequence_count_)
      : max_sequence_count(((max_sequence_count_ + 1) / 2) * 2)
      , sequence_count(0)
      , data_buffer(data_buffer)
      , base_offset(reinterpret_cast<uint32_t *>(data_buffer + max_sequence_count * 2 * sizeof(uint16_t)))
  {
    assert((uintptr_t)data_buffer % 4 == 0); // 4 byte alignment required
    reinit(max_sequence_count);
  }

  inline __device__ CompressSequenceBuffer(uint8_t *data_buffer, const int max_sequence_count_, const int sequence_count)
      : max_sequence_count(((max_sequence_count_ + 1) / 2) * 2)
      , sequence_count(sequence_count)
      , data_buffer(data_buffer)
      , base_offset(reinterpret_cast<uint32_t *>(data_buffer + max_sequence_count * 2 * sizeof(uint16_t)))
  {
    assert((uintptr_t)data_buffer % 4 == 0); // 4 byte alignment required
  }

  inline __device__ void reinit(int reinit_sequences = -1)
  {
    if (reinit_sequences == -1)
    {
      reinit_sequences = sequence_count;
    }

    const unsigned num_words = ((reinit_sequences * OFFSET_BIT_COUNT + 31) / 32);
    for (unsigned ix = threadIdx.x; ix < num_words; ix += WARP_SIZE_U)
    {
      base_offset[ix] = 0;
    }

    __syncwarp();
    sequence_count = 0;
  }

  static size_t __host__ __device__ get_buff_size(size_t num_sequences)
  {
    return get_compacted_seq_buff_size(num_sequences);
  }

  inline __device__ uint16_t get_match_length(const uint32_t ix) const
  {
    return *reinterpret_cast<uint16_t *>(data_buffer + ix * sizeof(uint16_t));
  }

  inline __device__ uint16_t get_literal_length(const uint32_t ix) const
  {
    return *reinterpret_cast<uint16_t *>(data_buffer + max_sequence_count * sizeof(uint16_t) + ix * sizeof(uint16_t));
  }

  __device__ uint32_t operator()(const uint32_t ix_seq, const int ix_seq_type) const
  {
    if (ix_seq_type == 0)
    {
      return get_literal_length(ix_seq);
    }
    else if (ix_seq_type == 1)
    {
      return get_offset(ix_seq);
    }
    else
    {
      return get_match_length(ix_seq);
    }
  }

  inline __device__ uint32_t get_offset(const uint32_t ix) const
  {
    const uint32_t bit_offset = ix * OFFSET_BIT_COUNT;
    const uint32_t word_offset = bit_offset % 32;
    const uint32_t word1_idx = bit_offset >> 5;
    const uint32_t word1 = base_offset[word1_idx];
    // Note: if the 1st word contains all the bits, there's no need to read out the next word
    const uint32_t word2 = 32 < (word_offset + OFFSET_BIT_COUNT) ? base_offset[word1_idx + 1] : 0;
    uint32_t full_offset = __funnelshift_r(word1, word2, word_offset) & ((1 << OFFSET_BIT_COUNT) - 1);
    return full_offset;
  }

  inline __device__ void set_match_length(const uint32_t ix, const uint16_t match_length)
  {
    reinterpret_cast<uint16_t *>(data_buffer)[ix] = match_length;
  }

  inline __device__ void set_literal_length(const uint32_t ix, const uint16_t literal_length)
  {
    reinterpret_cast<uint16_t *>(data_buffer)[ix + max_sequence_count] = literal_length;
  }

  inline __device__ void set_offset(const uint32_t ix, const uint32_t offset)
  {
    const uint32_t bit_offset = ix * OFFSET_BIT_COUNT;

    const uint32_t word1_idx = bit_offset >> 5;
    const uint32_t word_offset = bit_offset % 32;
    uint64_t shifted_val = uint64_t{offset} << word_offset;
    cuda::atomic_ref<uint32_t, cuda::thread_scope_block> word1_atomic{base_offset[word1_idx]};
    word1_atomic.fetch_or(static_cast<uint32_t>(shifted_val), cuda::std::memory_order_relaxed);
    shifted_val >>= 32;
    if (shifted_val > 0)
    {
      cuda::atomic_ref<uint32_t, cuda::thread_scope_block> word2_atomic{base_offset[word1_idx + 1]};
      word2_atomic.fetch_or(static_cast<uint32_t>(shifted_val), cuda::std::memory_order_relaxed);
    }
  }

  inline __device__ void set_sequence_count(const uint32_t seq_count) { sequence_count = seq_count; }

  // This is used to consolidate buffers into a new "raw buffer"
  inline __device__ void copy(CompressSequenceBuffer &new_buffer)
  {
    assert(new_buffer.max_sequence_count >= sequence_count);
    const uint32_t *match_len_input = reinterpret_cast<uint32_t *>(data_buffer);
    uint32_t *match_len_output = reinterpret_cast<uint32_t *>(new_buffer.data_buffer);
    const uint32_t *lit_len_input =
      reinterpret_cast<const uint32_t *>(data_buffer + max_sequence_count * sizeof(uint16_t));
    uint32_t *lit_len_output =
      reinterpret_cast<uint32_t *>(new_buffer.data_buffer + new_buffer.max_sequence_count * sizeof(uint16_t));
    uint32_t *offset_output = new_buffer.base_offset;
    const uint32_t *offset_input = base_offset;
    const int seq_count_words = (sequence_count + 1) / 2;
    const int offset_word_count = (sequence_count * OFFSET_BIT_COUNT + 31) / 32;
    for (int ix_copy = threadIdx.x; ix_copy < offset_word_count; ix_copy += WARP_SIZE)
    {
      uint32_t match_word, lit_word, off_word;
      if (ix_copy < seq_count_words)
      {
        match_word = match_len_input[ix_copy];
        lit_word = lit_len_input[ix_copy];
      }

      off_word = offset_input[ix_copy];

      if (ix_copy < seq_count_words)
      {
        match_len_output[ix_copy] = match_word;
        lit_len_output[ix_copy] = lit_word;
      }
      offset_output[ix_copy] = off_word;
    }
  }
};

struct CompressBlockShare
{
  uint8_t *output_buffer;
  uint8_t *sequence_buffer;
  uint8_t *literals;
  CompressBlockShare *next_block_share;
  const uint8_t *input_buffer;
  int ix_block;
  unsigned compressed_size;
  unsigned uncompressed_size;
  int num_sequences;
  int num_literals;
  int num_blocks;
  int ix_frame;
  cuda::atomic<int, cuda::thread_scope_device> already_started;
};

struct ZstdFrame
{
  DeviceBlockShare *block_share;
};

// symbol translation cell
class st_cell
{
public:
  int16_t state_offset;
  uint16_t n_bits;
};

} // namespace zstd
