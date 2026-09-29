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

#include <cuda_runtime.h>

#include <cstdint>

#include "ans/ans_tools.cuh"
#include "EntropyTables.cuh"
#include "huffman.cuh"
#include "io.cuh"
#include "types.cuh"
#include "utils.cuh"

// #define HUFF_LITERAL_LOGGING 1
// #define HUFF_FSE_LOGGING 1

namespace zstd
{

/*
    Fills in the header, returns the number of bytes advanced
*/
template <int NUM_STREAMS>
inline __device__ int reserve_compressed_huffman_header(BitWriter &bit_writer, unsigned num_literals)
{
  int output_bytes;

  if constexpr (NUM_STREAMS == 1)
  {
    output_bytes = 3;
  }
  else
  {
    if (num_literals < (1 << 10))
    {
      output_bytes = 3;
    }
    else if (num_literals < (1 << 14))
    {
      output_bytes = 4;
    }
    else
    {
      output_bytes = 5;
    }
  }

  bit_writer.increment_bytes(output_bytes);

  return output_bytes;
}

template <int NUM_STREAMS>
inline __device__ void generate_compressed_huffman_header(uint8_t *output, unsigned num_literals, unsigned comp_size)
{
  uint8_t block_type = 0x2; // Takes first 2 bits
  uint8_t size_format;
  // Note: according to the Zstd standard, it is only the number of literals that
  //       determines the header size and not the compressed size. Hence the
  //       comparison with only `num_literals`.
  if (NUM_STREAMS == 1 || num_literals < (1 << 10))
  {
    assert(comp_size < (1 << 10));
    assert(num_literals < (1 << 10));

    // 10 bits for the regen size
    size_format = NUM_STREAMS == 1 ? 0x0 : 0x1;

    // 10 bits for the regen size
    *(output++) = (size_format << 2) | block_type | ((num_literals & 0xf) << 4);
    *(output++) = num_literals >> 4 |
                  ((comp_size & 0x3) << 6); // final 6 bits of the literals, first 2 bits of comp size
    *output = comp_size >> 2; // last 8 bits
  }
  else if (num_literals < (1 << 14))
  {
    assert(comp_size < (1 << 14));

    size_format = 0x2;
    *(output++) = (size_format << 2) | block_type | ((num_literals & 0xf) << 4); // first 4 bits
    *(output++) = (num_literals >> 4) & 0xff; // next 8 bits of the literals
    *(output++) = (num_literals >> 12) |
                  ((comp_size & 0x3f) << 2); // final 2 bits of the literals and first 6 bits of comp size
    *output = (comp_size >> 6); // last 8 bits
  }
  else
  {
    assert(num_literals < (1 << 18));
    assert(comp_size < (1 << 18));

    size_format = 0x3;
    *(output++) = (size_format << 2) | block_type | ((num_literals & 0xf) << 4); // first 4 bits
    *(output++) = (num_literals >> 4) & 0xff; // next 8 bits of the literals
    *(output++) = (num_literals >> 12) | (comp_size & 0x3)
                                           << 6; // First 2 bits of comp size, final 6 bits of the literals
    *(output++) = (comp_size >> 2) & 0xff;
    *output = (comp_size >> 10) & 0xff;
  }
}

inline __device__ int generate_raw_huffman_header(uint8_t *output, unsigned num_literals)
{
  // uint8_t block_type = 0x0; // Takes first 2 bits

  int output_bytes = 0;

  // Note: in testing this is always false
  if (num_literals < (1 << 5))
  {
    output[output_bytes++] = num_literals << 3; // First 3 bits zero in this config
    // Note: in testing this is always true
  }
  else if (num_literals < (1 << 12))
  {
    output[output_bytes++] = ((num_literals << 4) & 0xff) | (0x1 << 2); // 0x1 is the size format
    output[output_bytes++] = num_literals >> 4; // 12 bits for the literal size
  }
  else
  {
    output[output_bytes++] = ((num_literals << 4) & 0xff) | (0x3 << 2); // 0x3 is the size format
    output[output_bytes++] = (num_literals >> 4) & 0xff; // 12 bits for the literal size
    output[output_bytes++] = num_literals >> 12; // 20 bits for the literal size
  }

  return output_bytes;
}

inline __device__ int generate_rle_huffman_header(uint8_t *output, unsigned num_literals, uint8_t rle_val)
{
  uint8_t block_type = 0x1; // Takes first 2 bits

  int output_bytes = 0;

  if (num_literals < (1 << 5))
  {
    output[output_bytes++] = num_literals << 3 | block_type;
  }
  else if (num_literals < (1 << 12))
  {
    output[output_bytes++] = ((num_literals << 4) & 0xff) | (0x1 << 2) | block_type; // 0x1 is the size format
    output[output_bytes++] = num_literals >> 4; // 12 bits for the literal size
  }
  else
  {
    output[output_bytes++] = ((num_literals << 4) & 0xff) | (0x3 << 2) | block_type; // 0x3 is the size format
    output[output_bytes++] = (num_literals >> 4) & 0xff; // 12 bits for the literal size
    output[output_bytes++] = num_literals >> 12; // 20 bits for the literal size
  }

  output[output_bytes++] = rle_val;

  return output_bytes;
}

inline __device__ void huff_compress_literals(
  BitWriter &bit_writer,
  const uint8_t *input,
  const int num_literals,
  CompressHuffmanTable &huff_table,
  IntWarpScan &int_warp_scan,
  int ix_total_byte
)
{
  int scan_result;

  for (int warp_ix_byte = num_literals - 1; warp_ix_byte >= 0; warp_ix_byte -= WARP_SIZE)
  {
    int ix_byte = warp_ix_byte - threadIdx.x;
    int bits = 0;
    uint8_t symbol = 0;
    if (ix_byte >= 0)
    {
      symbol = input[ix_byte];
      bits = huff_table.bits[symbol];
    }
    int_warp_scan.ExclusiveSum(bits, scan_result);
    uint16_t write_val = 0;
    if (bits > 0)
    {
      write_val = huff_table.code[symbol];
    }
    bit_writer.write_bits(write_val, bits, threadIdx.x, scan_result, WARP_ALL, WARP_SIZE, true /*shuffle end*/);

#ifdef HUFF_LITERAL_LOGGING
    if (ix_total_byte + ix_byte < 32 and ix_total_byte + ix_byte >= 0)
    {
      printf(
        "thread %d ix byte %d symbol %u bits %u code %u write val %u\n",
        threadIdx.x,
        ix_total_byte + ix_byte,
        symbol,
        bits,
        huff_table.code[symbol],
        write_val
      );
    }
#endif
  }

  // Write a padding bit
  if (threadIdx.x == 0)
  {
    bit_writer.write_bits(1, 1);
    bit_writer.align_bytewise();
  }
  __syncwarp();
}

inline __device__ void write_jump_offset_le(uint8_t *output, int jump_offset_val)
{
  output[0] = jump_offset_val & 0xff;
  output[1] = (jump_offset_val >> 8) & 0xff;
}

// Returns the size of the complete compressed stream
inline __device__ int huff_compress_literals_4stream(
  BitWriter &bit_writer,
  const uint8_t *input,
  const int num_literals,
  CompressHuffmanTable &huff_table,
  IntWarpScan &int_warp_scan
)
{
  uint8_t *orig_ptr =
    bit_writer.get_output_ptr(); // 6 Jump table bytes. This ptr is used to compute the number of bytes written
  __syncwarp();
  bit_writer.flush(threadIdx.x, WARP_ALL, WARP_SIZE);
  if (threadIdx.x == 0)
  {
    bit_writer.increment_bytes(6);
  }
  __syncwarp();

  int ix_total_byte = 0;
  int this_regen_size = (num_literals + 3) / 4;

  for (int ix_stream = 0; ix_stream < 4; ++ix_stream)
  {
    int stream_num_literals = ix_stream < 3 ? this_regen_size : num_literals - 3 * this_regen_size;
    uint8_t *block_start_ptr = bit_writer.get_output_ptr();
    huff_compress_literals(bit_writer, input, stream_num_literals, huff_table, int_warp_scan, ix_total_byte);
    ix_total_byte += stream_num_literals;
    input += stream_num_literals;
    if (ix_stream < 3)
    {
      // Write out the jump table as well
      bit_writer.flush(threadIdx.x, WARP_ALL, WARP_SIZE);
      const unsigned compress_bytes = bit_writer.compute_byte_offset(block_start_ptr);
      write_jump_offset_le(orig_ptr + ix_stream * 2, compress_bytes);

#ifdef HUFF_LITERAL_LOGGING
      print0("compressed bytes %u for stream %d stream literals %d\n", compress_bytes, ix_stream, stream_num_literals);
#endif
    }
  }

  bit_writer.flush(threadIdx.x, WARP_ALL, WARP_SIZE);
  return bit_writer.compute_byte_offset(orig_ptr);
}

// This is a slightly modified version of what's in GDeflate.
// TODO: merge the two.
template <
  int max_symbol_count, // Max number of symbols
  int max_code_len> // Max code length
class WarpHuffmanTree
{
  // Members
  int root;
  uint16_t n_symbols;

public:
  uint16_t left[max_symbol_count];
  uint16_t right[max_symbol_count];
  uint16_t depth[2 * max_symbol_count];
  uint16_t symbols[max_symbol_count];
  uint16_t codelen[max_symbol_count];
  __align__(4) uint8_t weights[max_symbol_count];
  uint32_t counts[2 * max_symbol_count];

public:
  WarpHuffmanTree() = default;

  __device__ void build(const int num_symbols)
  {
    int this_thread_n_symbols = 0;

    // Sort counts
    constexpr unsigned int items_per_thread = (max_symbol_count + WARP_SIZE - 1) / WARP_SIZE;
    typedef nvcomp::cub::BlockRadixSort<unsigned int, WARP_SIZE, items_per_thread, uint16_t> BlockRadixSort;
    __shared__ typename BlockRadixSort::TempStorage temp_storage;

    uint16_t s[items_per_thread];
    unsigned int c[items_per_thread];

// Load keys and values
#pragma unroll
    for (uint16_t i = 0; i < items_per_thread; ++i)
    {
      uint16_t start = items_per_thread * threadIdx.x;
      s[i] = start + i;
      c[i] = (start + i < num_symbols) ? counts[start + i] : 0;
      this_thread_n_symbols += (c[i] == 0) ? 0 : 1;
      c[i] = c[i] == 0 ? std::numeric_limits<unsigned int>::max() : c[i];
    }

    // Sort
    BlockRadixSort(temp_storage).Sort(c, s);
    __syncwarp(); // TODO: BlockRadixSort here limits this implementation to 1 warp per CTA

// Store keys and values
#pragma unroll
    for (uint16_t i = 0; i < items_per_thread; ++i)
    {
      uint16_t start = items_per_thread * threadIdx.x;
      if ((start + i) < max_symbol_count)
      {
        symbols[start + i] = s[i];
        counts[start + i] = c[i];
      }
    }
    __syncwarp();

    n_symbols = warpReduceSum(this_thread_n_symbols, WARP_ALL);
    assert(n_symbols > 0);
    __syncwarp();

    // Only thread 0 builds the tree in a single threaded fashion
    if (threadIdx.x == 0)
    {
      uint16_t leaf_counter = 0;
      uint16_t branch_counter = 0;
      uint16_t n_branches = 0;

      // Build tree by combining nodes with the least counts until only 1 node remains
      while ((leaf_counter < n_symbols) || (n_branches - branch_counter > 1))
      {
        unsigned int ind[2];

        // Get the smallest two nodes
        for (int i = 0; i < 2; ++i)
        {
          if (branch_counter >= n_branches)
          {
            ind[i] = leaf_counter++;
          }
          else if (leaf_counter >= n_symbols)
          {
            ind[i] = max_symbol_count + branch_counter++;
          }
          else
          {
            ind[i] = (counts[leaf_counter] <= counts[max_symbol_count + branch_counter])
                       ? leaf_counter++
                       : (max_symbol_count + branch_counter++);
          }
        }

        // Merge the two nodes
        counts[max_symbol_count + n_branches] = counts[ind[0]] + counts[ind[1]];
        // Put higher count node on the left
        left[n_branches] = ind[1];
        right[n_branches] = ind[0];
        n_branches++;
      }

      // Last remaining node is the root of the tree
      root = max_symbol_count + n_branches - 1;
      depth[root] = 0;

      // Now need to traverse the tree to get the codelengths of each symbol
      uint16_t q_size = 0, q_counter = 0;
      int *queue = reinterpret_cast<int *>(&counts[max_symbol_count]);

      // Add root to the queue and traverse the tree
      // TODO: Parallelize this tree traversal
      queue[q_size++] = root;
      uint16_t max_depth = 0;
      while (q_size > q_counter)
      {
        int node = queue[q_counter++];
        max_depth = depth[node] + 1;

        uint16_t l = left[node - max_symbol_count];
        uint16_t r = right[node - max_symbol_count];
        depth[l] = max_depth;
        depth[r] = max_depth;

        // Don't add leaf nodes to the queue
        if (l >= max_symbol_count)
        {
          queue[q_size++] = l;
        }
        if (r >= max_symbol_count)
        {
          queue[q_size++] = r;
        }
      }
    }
    __syncwarp();
  }

  __device__ void codelengths()
  {

    // Compute Kraft number scaled by 2^(max_code_len/2) for increased dynamic range
    float K = 0.;
    constexpr int norm_power = max_code_len / 2;

    for (unsigned int i = threadIdx.x; i < max_symbol_count; i += WARP_SIZE_U)
    {
      uint16_t symbol = symbols[i];
      uint16_t len = min(i < n_symbols ? depth[i] : 0, max_code_len); // Limit to max code length max_code_len
      codelen[symbol] = len;
      K += len > 0 ? scalbnf(1.f, norm_power - len) : 0; // K += 2^(norm_power - len)
    }
    K = warpReduceSum(K, WARP_ALL);
    __syncwarp();

    // Return if Kraft number <= 1
    if (K <= scalbnf(1.f, norm_power))
    {
      return;
    }

    // TODO: Add better depth limiting
    if (threadIdx.x == 0)
    {

      for (int i = 0; i < n_symbols; ++i)
      {
        uint16_t symbol = symbols[i];
        uint16_t len = codelen[symbol];
        if (len < max_code_len)
        {
          len += 1;
          K -= scalbnf(1.f, norm_power - len);
          codelen[symbol] = len;
        }
        if (K <= scalbnf(1.f, norm_power))
        {
          break;
        }
      }

      for (int i = n_symbols - 1; i >= 0; --i)
      {
        uint16_t symbol = symbols[i];
        uint16_t len = codelen[symbol];
        float K_ = K + scalbnf(1.f, norm_power - len);

        if (K_ > scalbnf(1.f, norm_power))
        {
          continue;
        }

        codelen[symbol] = len - 1;
        K = K_;

        if (K >= scalbnf(1.f, norm_power))
        {
          break;
        }
      }
    }
    __syncwarp();
  }

  __device__ uint16_t num_symbols() { return n_symbols; }
};

template <typename WarpHuffmanTree_t>
__device__ int compute_weights(WarpHuffmanTree_t &wht, const int num_symbols)
{
  unsigned max_codelen = 0;

  for (int ix = threadIdx.x; ix < num_symbols; ix += WARP_SIZE)
  {
    const unsigned symbol = wht.symbols[ix];
    const unsigned this_codelen = wht.codelen[symbol];
    max_codelen = max(max_codelen, this_codelen);
    if (symbol < num_symbols)
    {
      wht.weights[symbol] = this_codelen;
    }
  }

  __syncwarp();
  max_codelen = warpReduceMax(max_codelen, WARP_ALL);

  int max_nonzero_symbol = 0;
  for (int ix = threadIdx.x; ix < num_symbols; ix += WARP_SIZE)
  {
    if (wht.weights[ix] > 0)
    {
      wht.weights[ix] = max_codelen - wht.weights[ix] + 1;
      max_nonzero_symbol = max(max_nonzero_symbol, ix);
    }
  }

  max_nonzero_symbol = warpReduceMax(max_nonzero_symbol, WARP_ALL);

  return max_nonzero_symbol; // Number of valid weights
}

// This isn't currently used, but left in place for now.
template <typename WarpHuffmanTree_t>
inline __device__ int
compute_weight_repr_standard(const WarpHuffmanTree_t &wht, const int num_weights, BitWriter &bit_writer)
{
  // Write the header
  if (threadIdx.x == 0)
  {
    bit_writer.write_bits(num_weights + 127, 8);
  }
  __syncwarp();

  int upper_bound = (num_weights + 1) / 2;
  int base_ix = 0;
  for (; base_ix + WARP_SIZE < upper_bound; base_ix += WARP_SIZE)
  {
    int ix = base_ix + threadIdx.x;
    uint8_t weight1 = wht.weights[ix * 2];
    uint8_t weight2 = (ix * 2 + 1) < num_weights ? wht.weights[ix * 2 + 1] : 0;
    uint8_t byte = (weight1 << 4) | (weight2);
    int num_bits = 8;
    bit_writer.write_bits(byte, num_bits, threadIdx.x, 8 * threadIdx.x, WARP_ALL, WARP_SIZE);
  }

  // Finish the base
  if (base_ix + WARP_SIZE > upper_bound)
  {
    int ix = base_ix + threadIdx.x;
    bool active = ix < upper_bound;
    int num_active = __popc(__ballot_sync(WARP_ALL, active));
    if (active)
    {
      uint8_t weight1 = wht.weights[ix * 2];
      uint8_t weight2 = (ix * 2 + 1) < num_weights ? wht.weights[ix * 2 + 1] : 0;
      uint8_t byte = (weight1 << 4) | (weight2);
      int num_bits = 8;
      unsigned mask = (1 << num_active) - 1;
      bit_writer.write_bits(byte, num_bits, threadIdx.x, 8 * threadIdx.x, mask, num_active);
    }
  }

  return (num_weights + 1) / 2 + 1;
}

inline __device__ void compress_two_fse_streams(
  SymbolEncoder &shared_encoder,
  const uint8_t *symbols,
  size_t symbol_count,
  BitWriter &bit_writer
)
{
  // Starting with the last symbol, compress
  if (threadIdx.x >= 2)
  {
    return;
  }
  // Copy the state locally -- can share tables
  SymbolEncoder encoder = shared_encoder;

  const unsigned TWO_THREADS_MASK = 0x3;
  bool odd_count = symbol_count % 2 != 0;
  int ix_symbol;

  // Process the last symbol with the first thread
  if (odd_count)
  {
    ix_symbol = symbol_count - 1 - threadIdx.x;
  }
  else
  {
    ix_symbol = symbol_count - 1 - (threadIdx.x ^ 1);
  }

#ifdef HUFF_FSE_LOGGING
  printf("encoding ix %d symbol %u\n", ix_symbol, symbols[ix_symbol]);
#endif

  encoder.encode_first_symbol(symbols[ix_symbol]);
  ix_symbol -= 2;

  // if this is an odd count, first do a thread 0 symbol
  if (odd_count)
  {
    if (threadIdx.x == 0)
    {
      // One more symbol to write for the first thread (state)
      auto res = encoder.encode_symbol(symbols[ix_symbol]);
      bit_writer.write_bits(res.prev_state, res.num_bits);

#ifdef HUFF_FSE_LOGGING
      printf("ix %d symbol %u prev state %u num bits %u\n", ix_symbol, symbols[ix_symbol], res.prev_state, res.num_bits);
#endif

      ix_symbol -= 2;
    }
  }

  __syncwarp(TWO_THREADS_MASK);
  for (; ix_symbol >= 0; ix_symbol -= 2)
  {
    auto res = encoder.encode_symbol(symbols[ix_symbol]);
    int shuffle_val = __shfl_sync(TWO_THREADS_MASK, res.num_bits, 1);
    int scan_offset = threadIdx.x == 0 ? shuffle_val : 0;
    bit_writer.write_bits(
      res.prev_state,
      res.num_bits,
      threadIdx.x,
      scan_offset,
      TWO_THREADS_MASK,
      2 /* num threads */,
      false /*shuffle end*/
    );
#ifdef HUFF_FSE_LOGGING
    printf("ix %d symbol %u prev state %u num bits %u\n", ix_symbol, symbols[ix_symbol], res.prev_state, res.num_bits);
#endif
  }

  // Finally, write out the last state
  int scan_offset = threadIdx.x == 0 ? encoder.tablelog : 0;
#ifdef HUFF_FSE_LOGGING
  printf("write last state: %u\n", encoder.state & ((1 << encoder.tablelog) - 1));
#endif
  bit_writer.write_bits(
    encoder.state,
    encoder.tablelog,
    threadIdx.x,
    scan_offset,
    TWO_THREADS_MASK,
    2 /* num threads */,
    false /*shuffle end*/
  );
  __syncwarp(TWO_THREADS_MASK);

  // Then write a single bit -- zstd format requirement for finding the end of the stream
  if (threadIdx.x == 0)
  {
    bit_writer.write_bits(1, 1);
    bit_writer.align_bytewise();
  }
}

// returns the byte count of the weight representation
// returns -1 if the computation failed.
template <typename WarpHuffmanTree_t>
inline __device__ int compute_weight_repr_fse(
  const WarpHuffmanTree_t &wht,
  CompressHuffmanBuffers &huff_buffers,
  const int num_weights,
  BitWriter &bit_writer
)
{
  uint8_t *header_output;
  if (threadIdx.x == 0)
  {
    header_output = bit_writer.check_out_byte();
  }

  // Compute the maximum weight
  const int table_log = HUF_FSE_WEIGHT_MAX_ACCURACY_LOG; // For now, always use the maximum table log.
  uint32_t *weight_freqs = huff_buffers.ans_buffers.weight_freqs[0];
  int16_t *norm_weight_freqs = huff_buffers.ans_buffers.norm_weight_freqs;
  if (threadIdx.x <= HUF_MAX_BITS)
  {
    weight_freqs[threadIdx.x] = 0;
  }
  __syncwarp(); // ensure this has been zero'd before proceeding

  nvcomp::ans_shared::frequency_histogram(weight_freqs, wht.weights, num_weights);
  __syncwarp();

  int max_weight = 0;
  int nonzero_count = 0;
  int total_count = num_weights;

  for (int ix = 0; ix <= HUF_MAX_BITS; ++ix)
  {
    if (weight_freqs[ix] != 0)
    {
      max_weight = ix;
      ++nonzero_count;
    }
  }
  // Have to send 2 unique symbols.
  assert(max_weight != 0);

  if (nonzero_count == 1)
  {
    // For now, just inform the outside world that this failed
    // return -1;
    // TODO: All symbols have the same weight. In this case, we really want to use fse compression of the weights,
    // but it naturally doesn't work for FSE table descriptions. Give a nonzero count to zero weight to get
    // around this limitation.
    //
    weight_freqs[0] = 1;
    ++total_count;
  }

  bool valid =
    nvcomp::ans_shared::normalize_frequencies(norm_weight_freqs, table_log, weight_freqs, total_count, max_weight);
  assert(valid);

  // TODO: this limits us to 1 warp / CTA that can call this function.
  __shared__ Uint16WarpScan::TempStorage temp_storage;
  Uint16WarpScan uint16_warp_scan{temp_storage};

  tANS_construct_encoding_table(
    huff_buffers.encoder,
    norm_weight_freqs,
    huff_buffers.ans_buffers.starts,
    huff_buffers.ans_buffers.dst_table,
    max_weight,
    table_log,
    uint16_warp_scan
  );

  // Need to write the FSE header
  const int num_fse_weights = max_weight + 1;
  output_fse_header(bit_writer, table_log, norm_weight_freqs, num_fse_weights);

  compress_two_fse_streams(huff_buffers.encoder, wht.weights, num_weights, bit_writer);

  __syncwarp();

  bit_writer.flush(threadIdx.x, WARP_ALL, WARP_SIZE);
  __syncwarp();

  // Align the bitstream to the next byte boundary
  int num_bytes;
  if (threadIdx.x == 0)
  {
    num_bytes = bit_writer.compute_byte_offset(header_output) - 1; // -1 because bit_offset includes the header
    bit_writer.check_in_byte(header_output, num_bytes);
  }

  num_bytes = __shfl_sync(WARP_ALL, num_bytes, 0);
  return num_bytes + 1;
}

template <typename WarpHuffmanTree_t>
inline __device__ int compute_weight_repr(
  const WarpHuffmanTree_t &wht,
  CompressHuffmanBuffers &huff_buffers,
  const int num_weights,
  BitWriter &bit_writer
)
{
  if (num_weights == 1)
  {
    if (threadIdx.x == 0)
    {
      bit_writer.write_bits(num_weights + 127, 8); // Header for non-fse representation,
      unsigned write_val = wht.weights[0] << 4;
      bit_writer.write_bits(write_val, 8); // just a single byte
    }
    __syncwarp();
    return 2; // 2 bytes written
  }
  else
  {
    // Have to use the fse storage
    int num_bytes = compute_weight_repr_fse(wht, huff_buffers, num_weights, bit_writer);
    assert(num_bytes > 0); // TODO: Would need to fall back to raw literal storage
    assert(num_bytes <= 128); // TODO: Fall back to raw literal storage, but this can't happen
    return num_bytes;
  }
}

/* Returns boolean indicating whether compression was successful
   Currently compression is only unsuccessful if the size of the compressed 
   buffer would be larger than the uncompressed buffer.
   Note, we only call this function if we have "enough" literals for there to be opportunity.
   "Enough" is specified in zstdCompressionKernels using the constant 
   MIN_LITERALS_FOR_HUFF_COMPRESSION
*/
inline __device__ bool zstd_compress_literal_block(
  const uint8_t *symbols,
  const uint8_t max_symbol_value,
  const int num_literals,
  int *num_comp_bytes,
  CompressHuffmanBuffers &huff_buffers,
  CompressHuffmanTable &huff_table,
  uint8_t *comp_buffer
)
{
  __shared__ BitWriter bit_writer;
  bit_writer.init(comp_buffer, num_literals /* max output size */);

  const int cexpr_max_symbol_value = 255;
  const int cexpr_max_num_symbols = cexpr_max_symbol_value + 1;
  assert(max_symbol_value <= cexpr_max_symbol_value);

  __shared__ WarpHuffmanTree<cexpr_max_num_symbols, HUF_MAX_BITS> wht;

  const int num_symbols = max_symbol_value + 1;
  uint32_t *frequencies = wht.counts;

  for (int ix = threadIdx.x; ix <= max_symbol_value; ix += WARP_SIZE)
  {
    frequencies[ix] = 0;
  }
  __syncwarp();

  nvcomp::ans_shared::frequency_histogram(frequencies, symbols, num_literals);
  __syncwarp();

  // Check for RLE -- how many unique symbols are there?
  int per_thread_unique_symbol_count = 0;
  int nonzero_symbol = -1;
  for (int ix = threadIdx.x; ix <= max_symbol_value; ix += WARP_SIZE)
  {
    if (frequencies[ix] > 0)
    {
      ++per_thread_unique_symbol_count;
      nonzero_symbol = ix;
    }
  }

  int unique_symbol_count = warpReduceSum(per_thread_unique_symbol_count, WARP_ALL);
  if (unique_symbol_count == 1)
  {
    // Write out an RLE header
    unsigned ix_nonzero = __ffs(__ballot_sync(WARP_ALL, nonzero_symbol >= 0)) - 1;
    int rle_val = __shfl_sync(WARP_ALL, nonzero_symbol, ix_nonzero);
    *num_comp_bytes = generate_rle_huffman_header(comp_buffer, num_literals, rle_val);
    return true;
  }

  // for (int ix = threadIdx.x; ix <= max_symbol_value; ix += WARP_SIZE) {
  //   printf("freq[%d] = %u num literals %d\n", ix, frequencies[ix], num_literals);
  // }
  // Eventually this will be shmem that is re-used by the compressed huffman table
  // TODO: investigate items in WarpHuffmanTree that would be acceptable to place in global memory
  wht.build(num_symbols);

  wht.codelengths();

  const int max_nonzero_symbol = compute_weights(wht, num_symbols);

  const int num_weights = max_nonzero_symbol;

  constexpr int NUM_STREAMS = 4;
  if (threadIdx.x == 0)
  {
    reserve_compressed_huffman_header<NUM_STREAMS>(bit_writer, num_literals);
  }
  __syncwarp();

  int table_desc_size = compute_weight_repr(wht, huff_buffers, num_weights, bit_writer);

  // Compress the symbols
  __shared__ IntWarpScan::TempStorage warp_scan_storage;
  IntWarpScan int_cub_scan{warp_scan_storage};

  init_compress_huff_table(huff_table, huff_buffers, wht.weights, num_weights);

  // Go ahead and do this with the entire warp. Set into 4 channels.
  const int compressed_bytes =
    huff_compress_literals_4stream(bit_writer, symbols, num_literals, huff_table, int_cub_scan);

  // Note: If the total number of compressed bytes is larger or equal to the number of literals,
  //       we'll simply store the uncompressed literals instead, and the compressed output will be discarded.
  *num_comp_bytes = bit_writer.compute_byte_offset(comp_buffer);
  if (*num_comp_bytes >= num_literals)
  {
    return false;
  }

  __syncwarp();
  bit_writer
    .flush(threadIdx.x, WARP_ALL, WARP_SIZE); // BitWriter should flush before doing the manual writes in the below
  assert(bit_writer.check_fully_flushed());

  generate_compressed_huffman_header<NUM_STREAMS>(comp_buffer, num_literals, table_desc_size + compressed_bytes);

  return true;
}

} // namespace zstd
