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
#include "ans.cuh"
#include "EntropyTables.cuh"
#include "Reduction.cuh"
#include "utils.cuh"

namespace zstd
{

inline __device__ void compute_basecodes(uint16_t *basecodes, uint32_t *rank_counts, const int max_symbol_bit_count)
{
  if (thread_warp_ix() == 0)
  {
    basecodes[HUF_MAX_BITS] = 0;
    for (int ix_bit = HUF_MAX_BITS; ix_bit >= 1; ix_bit--)
    {
      basecodes[ix_bit - 1] = basecodes[ix_bit] + rank_counts[ix_bit] * (1 << (max_symbol_bit_count - ix_bit));
    }
  }
}

inline __device__ unsigned compute_weight_sum(uint8_t *weights, uint8_t num_weights)
{
  unsigned weight_sum = 0;
  for (int ix_symbol = thread_warp_ix(); ix_symbol < num_weights; ix_symbol += WARP_SIZE)
  {
    if (weights[ix_symbol] > 0)
    {
      weight_sum += (unsigned)1 << (weights[ix_symbol] - 1);
    }
  }

  __shared__ WarpReduceUnsigned::TempStorage reduce_temp_storage;
  weight_sum = WarpReduceUnsigned(reduce_temp_storage).Sum(weight_sum);
  weight_sum = __shfl_sync(WARP_ALL, weight_sum, 0);
  return weight_sum;
}

inline __device__ int compute_per_symbol_bitcounts(uint8_t *weights, uint8_t num_weights, uint8_t *bits)
{
  __syncwarp();
  unsigned weight_sum = compute_weight_sum(weights, num_weights);

  const int max_bits = highest_set_bit(weight_sum) + 1;
  const uint16_t left_over = ((uint16_t)1 << max_bits) - weight_sum;
  assert(not(left_over & (left_over - 1))); // corruption -- left_over needs to be a power of 2
  bits[num_weights] = max_bits - highest_set_bit(left_over); // The last weight isn't transmitted / stored

  int max_symbol_bit_count = 0;

  for (int ix_symbol = thread_warp_ix(); ix_symbol < num_weights; ix_symbol += WARP_SIZE)
  {
    if (weights[ix_symbol] > 0)
    {
      uint8_t this_bits = max_bits + 1 - weights[ix_symbol];
      bits[ix_symbol] = this_bits;
      max_symbol_bit_count = max(max_symbol_bit_count, this_bits);
    }
    else
    {
      bits[ix_symbol] = 0;
    }
  }
  __syncwarp();

  __shared__ WarpReduceInt::TempStorage temp_storage_int;
  max_symbol_bit_count = WarpReduceInt(temp_storage_int).Reduce(max_symbol_bit_count, cub_maximum());
  max_symbol_bit_count = __shfl_sync(WARP_ALL, max_symbol_bit_count, 0);
  return max_symbol_bit_count;
}

inline __device__ void compute_rank_counts(uint8_t *bits, uint32_t *rank_counts, uint16_t num_symbols)
{
  if (thread_warp_ix() <= HUF_MAX_BITS)
  {
    rank_counts[thread_warp_ix()] = 0;
  }
  __syncwarp();

  for (int ix_symbol = thread_warp_ix(); ix_symbol < num_symbols; ix_symbol += WARP_SIZE)
  {
    atomicAdd(&rank_counts[bits[ix_symbol]], unsigned{1});
  }
}

inline __device__ void init_compress_huff_table(
  CompressHuffmanTable &huf_table,
  CompressHuffmanBuffers &huff_buffers,
  uint8_t *weights,
  uint8_t num_weights
)
{
  uint16_t num_symbols = num_weights + 1;
  int max_symbol_bit_count = compute_per_symbol_bitcounts(weights, num_weights, huf_table.bits);
  huf_table.max_bits = max_symbol_bit_count;
  huf_table.num_symbols = num_symbols;
  uint32_t *rank_counts = huff_buffers.rank_counts;
  uint16_t *basecodes = huff_buffers.basecodes;

  compute_rank_counts(huf_table.bits, rank_counts, num_symbols);
  __syncwarp(); // ensure rank counts loop is finished before continuing

  compute_basecodes(basecodes, rank_counts, max_symbol_bit_count);
  __syncwarp(); // ensure basecodes loop is finished before continuing

  auto in_warp_idx = thread_warp_ix();
  assert(max_symbol_bit_count <= HUF_MAX_OFFSET_SIZE);
  if (in_warp_idx <= max_symbol_bit_count)
  {
    rank_counts[in_warp_idx] = 0;
  }
  __syncwarp(); // Ensure rank_counts is set before continuing

  // Now, assign symbols in natural order
  for (int warp_ix_symbol = 0; warp_ix_symbol < num_symbols; warp_ix_symbol += WARP_SIZE)
  {
    int ix_symbol = warp_ix_symbol + in_warp_idx;
    bool active = ix_symbol < num_symbols;
    unsigned this_mask = __ballot_sync(WARP_ALL, active);
    if (active)
    {
      unsigned bits = huf_table.bits[ix_symbol];
      unsigned bit_match = __match_any_sync(this_mask, bits);
      unsigned total_match = __popc(bit_match);
      unsigned prev_match = __popc(bit_match & ((1 << in_warp_idx) - 1));
      // Compute how many before and after this thread
      // Assign code
      if (bits > 0)
      {
        // huf_table.code[ix_symbol] = basecodes[bits] + (rank_counts[bits] + prev_match) * (1 << (max_symbol_bit_count - bits));
        uint16_t raw_code = basecodes[bits] + (rank_counts[bits] + prev_match) * (1 << (max_symbol_bit_count - bits));
        uint16_t right_shift_code = raw_code >> (huf_table.max_bits - bits);
        huf_table.code[ix_symbol] = right_shift_code;
      }

      __syncwarp(this_mask);
      if (bits > 0 and total_match == prev_match + 1)
      {
        // If this is the last one for this bitcount, update the shmem total
        rank_counts[bits] += total_match;
      }
    }
    __syncwarp();
  }
}

// Table creation methods
inline __device__ void init_huff_table(
  HuffmanTable &huf_table,
  uint8_t *weights,
  uint8_t num_weights,
  HuffmanTableConstructionBuffers &huff_buffers
)
{
  uint16_t num_symbols = num_weights + 1;

  uint8_t *bits = huff_buffers.bits;

  // for (int ix = thread_warp_ix(); ix < num_weights; ix+= 32) {
  //   printf("weight %d = %u num weights %d\n", ix, weights[ix], num_weights);
  // }
  int max_symbol_bit_count = compute_per_symbol_bitcounts(weights, num_weights, bits);

  uint32_t *rank_counts = huff_buffers.rank_counts;

  compute_rank_counts(bits, rank_counts, num_symbols);

  huf_table.max_bits = max_symbol_bit_count;

  __syncwarp(); // Necessary to have rank_counts filled in

  compute_basecodes(huf_table.basecodes, rank_counts, max_symbol_bit_count);

  // Need to do an exclusive sum on the rank counts
  __shared__ UnsignedWarpScan::TempStorage scan_temp_storage;
  UnsignedWarpScan unsigned_cub_scan{scan_temp_storage};
  unsigned sum_result;
  unsigned scan_input_val = 0;

  auto in_warp_idx = thread_warp_ix();
  if (in_warp_idx < HUF_MAX_OFFSET_SIZE)
  {
    scan_input_val = rank_counts[in_warp_idx];
  }

  unsigned_cub_scan.ExclusiveSum(scan_input_val, sum_result);
  if (in_warp_idx < HUF_MAX_OFFSET_SIZE)
  {
    huf_table.offsets[in_warp_idx] = sum_result;
    rank_counts[in_warp_idx] = 0;
  }

  __syncwarp(); // Necessary to finish the above before continuing

  uint32_t *rank_idx = rank_counts; // Reuse rank_counts
  // Allocate codes and fill in the table
  unsigned active_mask = WARP_ALL;
  for (int ix_base_symbol = 0; ix_base_symbol < num_symbols; ix_base_symbol += WARP_SIZE)
  {
    int ix_symbol = ix_base_symbol + in_warp_idx;

    const int remaining_symbols = num_symbols - ix_base_symbol;
    if (remaining_symbols < 32)
    {
      active_mask = (1 << remaining_symbols) - 1;
    }
    if (ix_symbol < num_symbols)
    {
      uint8_t num_bits = bits[ix_symbol];
      const unsigned match = __match_any_sync(active_mask, num_bits);
      const int rank_offset = __popc(match & ((1 << in_warp_idx) - 1));
      huf_table.symbols[huf_table.offsets[num_bits] + rank_idx[num_bits] + rank_offset] = ix_symbol;
      __syncwarp(active_mask); // rank idx reads need to happen before update below
      if (__ffs(match) - 1 == in_warp_idx)
      {
        rank_idx[num_bits] += __popc(match);
      }
      __syncwarp(active_mask); // rank_idx write needs to happen before next iteration
    }
  }
}

inline __device__ void fse_decode_hufweights(
  uint8_t *weights,
  BitReader &bit_reader,
  HuffmanTableConstructionBuffers &huff_buffers,
  uint8_t &num_symbols
)
{
  int16_t *frequencies = huff_buffers.frequencies;

  __shared__ Uint8WarpScan::TempStorage uint8_storage;
  Uint8WarpScan uint8_warp_scan{uint8_storage};

  __shared__ Uint16WarpScan::TempStorage uint16_storage;
  Uint16WarpScan uint16_warp_scan{uint16_storage};

  __shared__ WarpReduceUint16::TempStorage uint16_reduce_storage;
  WarpReduceUint16 uint16_warp_reduce{uint16_reduce_storage};

  decode_fse_header(
    huff_buffers.fse_table,
    bit_reader,
    frequencies,
    HUF_FSE_WEIGHT_MAX_ACCURACY_LOG,
    HUF_MAX_SYMBOLS,
    3 /*3 is huffman weights mode */,
    uint8_warp_scan,
    uint16_warp_scan,
    uint16_warp_reduce,
    huff_buffers.ans_buffers,
    true /*build_table*/
  );

  // Decode the weights
  int symbols_written = fse_decompress_interleaved2(huff_buffers.fse_table, weights, bit_reader);
  num_symbols = __shfl_sync(WARP_ALL, symbols_written, 0);

  __syncwarp();
}

// returns ther number of remaining bytes after the work
inline __device__ unsigned decode_weights(
  const uint8_t *input,
  int block_rem_size,
  uint8_t &num_symbols,
  HuffmanTableConstructionBuffers &huff_buffers
)
{
  BitReader bit_reader{input, block_rem_size};

  uint8_t header = bit_reader.read_bits(8);

  uint8_t *weights = huff_buffers.weights;
  if (header >= 128)
  {
    num_symbols = header - 127;
    const size_t num_bytes = (num_symbols + 1) / 2;

    const uint8_t *const weight_src = bit_reader.get_read_pointer(num_bytes);

    for (int ix_byte = thread_warp_ix(); ix_byte < num_bytes; ix_byte += WARP_SIZE)
    {
      // OK here if we go past the last symbol. Just won't use the result.
      int ix_symbol = ix_byte * 2;
      weights[ix_symbol] = weight_src[ix_byte] >> 4;

      ++ix_symbol;
      weights[ix_symbol] = weight_src[ix_byte] & 0x0f;
    }
  }
  else
  {
    // FSE encoded weights
    // Make a new bitreader with a stream that is the size of the FSE bitstream
    auto new_bitreader = bit_reader.make_subreader(header);
    fse_decode_hufweights(weights, new_bitreader, huff_buffers, num_symbols);
  }

  return bit_reader.rem_bytes();
}

template <bool copy_from_ptr = false>
inline __device__ void get_huffman_table(
  const uint8_t *input,
  const unsigned comp_block_size,
  HuffmanTable &huf_table,
  HuffmanTableConstructionBuffers &huff_buffers,
  DeviceBlockShare &block_share
)
{
  auto &block_header = block_share.block_header;

  int offset =
    3 +
    block_share.block_header.literal_header_size; // 3 bytes before this is the overall block header. literal header is
  unsigned init_rem_bytes = comp_block_size - offset;
  uint8_t num_symbols;
  unsigned post_rem_bytes = decode_weights(input + offset, init_rem_bytes, num_symbols, huff_buffers);

  block_header.literal_header.table_desc_size = init_rem_bytes - post_rem_bytes;

  init_huff_table(huf_table, huff_buffers.weights, num_symbols, huff_buffers);
}

inline __device__ uint8_t decode_huffman_symbol(
  uint16_t &state,
  const uint8_t *const src,
  const unsigned mask,
  const uint8_t *symbols,
  const int max_bits,
  HuffCache &huff_cache,
  const unsigned bc2,
  const unsigned bc3,
  const unsigned bc5,
  const unsigned bc6,
  const unsigned bc8,
  const unsigned bc9,
  const unsigned bc11,
  const unsigned of0,
  const unsigned of25,
  const unsigned of811,
  bool valid
)
{
  // Starts with the state filled in with max_bits bits
  // First, find the basecode.

  assert(max_bits <= 11);
  uint16_t basecode;
  uint8_t offset;
  int num_bits =
    lower_bound_descending_11(state, bc2, bc3, bc5, bc6, bc8, bc9, bc11, of0, of25, of811, offset, basecode);

  // Need the symbol.
  const uint8_t symbol = symbols[offset + ((state - basecode) >> (max_bits - num_bits))];
  if (not valid)
  {
    num_bits = 0;
  }

  const uint16_t rest = huff_cache.read(num_bits, mask);

  // Shift `bits` bits out of the state, keeping the low order bits that
  // weren't necessary to determine this symbol.  Then add in the new bits
  // that were read from the stream.
  state = ((state << num_bits) + rest) & ((1 << max_bits) - 1);
  return symbol;
}

inline __device__ void update_huffman_4way_pointers(
  const uint8_t *&src,
  uint8_t *&literal_buffer,
  int &regenerated_size,
  int ix_stream,
  int &this_comp_size,
  const unsigned this_block_mask
)
{
  const int NUM_COMP_SIZE_BITS = 16;

  if (ix_stream < 3)
  {
    this_comp_size = read_bits_single_thread(src, NUM_COMP_SIZE_BITS, ix_stream * NUM_COMP_SIZE_BITS);
  }

  int input_offset = 0;
  int incr = __shfl_up_sync(this_block_mask, this_comp_size, 1);
  if (ix_stream > 0)
  {
    input_offset += incr;
  }
  incr = __shfl_up_sync(this_block_mask, this_comp_size, 2);
  if (ix_stream > 1)
  {
    input_offset += incr;
  }
  incr = __shfl_up_sync(this_block_mask, this_comp_size, 3);
  if (ix_stream > 2)
  {
    input_offset += incr;
  }

  constexpr int TOTAL_COMP_SIZE_BYTES = 6;
  if (ix_stream == 3)
  {
    this_comp_size -= (input_offset + TOTAL_COMP_SIZE_BYTES);
  }
  assert(this_comp_size > 0);

  src += TOTAL_COMP_SIZE_BYTES + input_offset;

  unsigned this_regen_size = (regenerated_size + 3) / 4;
  literal_buffer += ix_stream * this_regen_size;
  if (ix_stream == 3)
  {
    this_regen_size = regenerated_size - 3 * this_regen_size;
  }
  regenerated_size = this_regen_size;
}

} // namespace zstd
