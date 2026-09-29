/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include <cub/cub.cuh>

#include <cuda.h>

#include <cassert>
#include <cstdint>
#include <limits>

#include "gdeflate_constants.h"
#include "gdeflate_decompress.h"
#include "huffman.h"
#include "lz.h"
#include "lz_hash.h"
#include "nvcomp/utils.hpp"

#include <stdio.h>

namespace gdeflate
{

// Binary search to find the code for a given length
template <bool deflate64, typename offset_type>
inline __device__ uint16_t length2sym(const offset_type length)
{
  constexpr int minMatchLength = deflate64 ? gdeflateMinMatchLength : deflateMinMatchLength;
  constexpr int maxMatchLength = deflate64 ? gdeflateMaxMatchLength : deflateMaxMatchLength;
  assert(length >= minMatchLength);
  assert(length <= maxMatchLength);
  (void)minMatchLength; // For NDEBUG builds
  (void)maxMatchLength; // For NDEBUG builds
  constexpr int valid_length_symbols = deflate64 ? gdeflate_valid_length_symbols : deflate_valid_length_symbols;
  uint16_t *lenbase = deflate64 ? lenbase64 : lenbase32;
  uint16_t base = 0;
  if (length >= lenbase[16])
  {
    base = 16;
  }
  if ((base + 8 < valid_length_symbols) && (length >= lenbase[base + 8]))
  {
    base += 8;
  }
  if ((base + 4 < valid_length_symbols) && (length >= lenbase[base + 4]))
  {
    base += 4;
  }
  if ((base + 2 < valid_length_symbols) && (length >= lenbase[base + 2]))
  {
    base += 2;
  }
  if ((base + 1 < valid_length_symbols) && (length >= lenbase[base + 1]))
  {
    base += 1;
  }
  return gdeflate_literal_symbols + base;
}

// Binary search to find the code for a given distance
template <bool deflate64 = true, typename offset_type>
inline __device__ uint16_t distance2sym(const offset_type distance)
{
  assert(distance > 0);
  constexpr int valid_distance_symbols = deflate64 ? gdeflate_valid_distance_symbols : deflate_valid_distance_symbols;
  uint16_t base = 0;
  uint16_t *distanceTable = deflate64 ? distanceTable64 : distanceTable32;
  if (distance >= distanceTable[16])
  {
    base = 16;
  }
  if ((base + 8 < valid_distance_symbols) && (distance >= distanceTable[base + 8]))
  {
    base += 8;
  }
  if ((base + 4 < valid_distance_symbols) && (distance >= distanceTable[base + 4]))
  {
    base += 4;
  }
  if ((base + 2 < valid_distance_symbols) && (distance >= distanceTable[base + 2]))
  {
    base += 2;
  }
  if ((base + 1 < valid_distance_symbols) && (distance >= distanceTable[base + 1]))
  {
    base += 1;
  }
  return base;
}

template <bool deflate64>
class NaiveCostMetric
{
public:
  __device__ NaiveCostMetric(uint8_t avg_litlen_bits_, uint8_t avg_distance_bits_)
      : avg_litlen_bits(avg_litlen_bits_)
      , avg_distance_bits(avg_distance_bits_)
  {}

  inline __device__ uint8_t literal_cost(const uint8_t /*literal*/) const { return avg_litlen_bits; }

  template <typename offset_type>
  inline __device__ uint8_t length_cost(const offset_type length) const
  {
    // Compute length cost
    uint8_t *xlenbits = deflate64 ? xlenbits64 : xlenbits32;
    uint16_t sym = length2sym<deflate64>(length);
    uint8_t xlen = xlenbits[sym - gdeflate_literal_symbols];
    return avg_litlen_bits + xlen;
  }

  template <typename offset_type>
  inline __device__ uint8_t distance_cost(const offset_type distance) const
  {
    // Compute distance cost
    assert(distance > 0);
    uint8_t *xdistbits = deflate64 ? xdistbits64 : xdistbits32;
    uint16_t sym = distance2sym<deflate64>(distance);
    uint8_t xlen = xdistbits[sym];
    return avg_distance_bits + xlen;
  }

private:
  uint8_t avg_litlen_bits;
  uint8_t avg_distance_bits;
};

template <bool deflate64, typename offset_type>
__device__ uint16_t encode_litlen(
  const warp_encoder<gdeflate_litlen_symbols, gdeflate_max_codelen> &enc,
  const uint16_t sym,
  const offset_type length,
  uint32_t &bits
)
{
  assert(sym < gdeflate_valid_litlen_symbols);
  uint16_t len = enc.get_codelen(sym);
  assert(len <= gdeflate_max_codelen);
  bits = enc.get_code(sym);

  uint16_t *lenbase = deflate64 ? lenbase64 : lenbase32;
  uint8_t *xlenbits = deflate64 ? xlenbits64 : xlenbits32;
  unsigned char xlen = sym >= gdeflate_literal_symbols ? xlenbits[sym - gdeflate_literal_symbols] : 0;
  uint16_t lengthBase = sym >= gdeflate_literal_symbols ? lenbase[sym - gdeflate_literal_symbols] : 1;
  bits |= ((uint32_t)(length - lengthBase) & mask<uint32_t>(xlen)) << len;

  // if (sym >= gdeflate_literal_symbols) printf("length = %hu, sym = %hu, len = %hu, xlen = %hu, bits = %u\n", length, sym, len, xlen, bits);

  return len + xlen;
}

template <bool deflate64, typename offset_type>
__device__ uint16_t encode_distance(
  const warp_encoder<gdeflate_distance_symbols, gdeflate_max_codelen> &enc,
  const uint16_t sym,
  const offset_type distance,
  uint32_t &bits
)
{
  assert(sym < gdeflate_valid_distance_symbols);
  uint16_t len = enc.get_codelen(sym);
  assert(len <= gdeflate_max_codelen);
  bits = enc.get_code(sym);

  uint16_t *distanceTable = deflate64 ? distanceTable64 : distanceTable32;
  uint8_t *xdistbits = deflate64 ? xdistbits64 : xdistbits32;
  unsigned char xdist = xdistbits[sym];
  bits |= ((uint32_t)(distance - distanceTable[sym]) & mask<uint32_t>(xdist)) << len;
  return len + xdist;
}

template <bool deflate64 = true, typename offset_type>
__device__ void gdeflate_encode_tile(
  uint32_t *output,
  size_t *compressed_size,
  const offset_type *lengths,
  const offset_type *distances,
  const uint8_t *literals,
  const unsigned int num_symbols,
  const unsigned int num_literals
)
{
  // Can share these variables across literal and distance Huffman tree builders
  __shared__ uint16_t left[gdeflate_litlen_symbols];
  __shared__ uint16_t right[gdeflate_litlen_symbols];
  __shared__ uint16_t depth[2 * gdeflate_litlen_symbols];

  // Literal symbols and counts
  __shared__ uint16_t l_symbols[gdeflate_litlen_symbols];
  __shared__ unsigned int l_counts[2 * gdeflate_litlen_symbols];

  // Distance symbols and counts
  __shared__ uint16_t d_symbols[gdeflate_distance_symbols];
  __shared__ unsigned int d_counts[2 * gdeflate_distance_symbols];

  // Keep a contiguous buffer for literal and distance codelengths
  uint16_t *codelen = reinterpret_cast<uint16_t *>(l_counts); // Have 2*gdeflate_litlen_symbols > gdeflate_total_symbols
  uint16_t *l_codelen = codelen;
  uint16_t *d_codelen = codelen + gdeflate_litlen_symbols;
  uint32_t *l_codes = reinterpret_cast<uint32_t *>(&l_counts[gdeflate_litlen_symbols]);
  uint32_t *d_codes = reinterpret_cast<uint32_t *>(&d_counts[gdeflate_distance_symbols]);

  for (unsigned int i = threadIdx.x; i < gdeflate_litlen_symbols; i += WARP_SIZE_U)
  {
    l_counts[i] = 0;
  }
  for (unsigned int i = threadIdx.x; i < gdeflate_distance_symbols; i += WARP_SIZE_U)
  {
    d_counts[i] = 0;
  }
  __syncwarp();

  // Compute counts of each literal character
  update_histogram(l_counts, literals, num_literals);

  // Add end of block character
  if (threadIdx.x == 0)
  {
    l_counts[gdeflate_literal_symbols - 1] = 1;
  }

  // Counts of lengths and distances
  for (unsigned int i = threadIdx.x; i < num_symbols; i += WARP_SIZE_U)
  {
    offset_type len = lengths[i];
    offset_type dist = distances[i];
    // Check if this token is a match
    if (len >= gdeflateMinMatchLength)
    {
      // Update length histogram
      uint16_t b = length2sym<deflate64>(len);
      atomicAdd(&l_counts[b], 1);

      // Update distance histogram
      assert(dist > 0);
      uint16_t d = distance2sym<deflate64>(dist);
      atomicAdd(&d_counts[d], 1);
    }
  }

  // Handle edge case when there are no copies
  // Add dummy distance symbol
  if ((num_symbols == num_literals) && (threadIdx.x == 0))
  {
    d_counts[distance2sym<deflate64>(1)] = 1;
  }

  __syncwarp();

  // Literal tree builder
  WarpHuffmanTree<gdeflate_litlen_symbols, gdeflate_max_codelen> lht(l_counts, l_symbols);
  lht.build(l_counts, left, right, depth);
  lht.codelengths(l_codelen, l_symbols, depth);

  // Distance tree builder
  WarpHuffmanTree<gdeflate_distance_symbols, gdeflate_max_codelen> dht(d_counts, d_symbols);
  dht.build(d_counts, left, right, depth);
  dht.codelengths(d_codelen, d_symbols, depth);

  // Build the literal/length Huffman encoder
  warp_encoder<gdeflate_litlen_symbols, gdeflate_max_codelen> l_enc;
  l_enc.init(
    gdeflate_litlen_symbols,
    l_codelen,
    l_codes,
    reinterpret_cast<uint16_t *>(left),
    reinterpret_cast<unsigned int *>(right)
  );

  // Build the distance Huffman encoder
  warp_encoder<gdeflate_distance_symbols, gdeflate_max_codelen> d_enc;
  d_enc.init(
    gdeflate_distance_symbols,
    d_codelen,
    d_codes,
    reinterpret_cast<uint16_t *>(left),
    reinterpret_cast<unsigned int *>(right)
  );

  // Initialize the warp bitwriter
  warp_bitwriter<uint32_t> bw(output);

  // Write the stream header
  uint32_t bfinal = 1, btype = 2, header = 0;
  header = ((btype & 3) << 1) | (bfinal & 1);
  bw.write(header, 3, threadIdx.x == 0); // Only thread 0 writes to its stream

  // Huffman encode codelengths and write them to the output stream
  // Also writes the deflate block header
  pack_codelens(codelen, gdeflate_valid_litlen_symbols, gdeflate_valid_distance_symbols, bw);

  bool isCopy = false;
  unsigned int symbol_counter = 0;
  unsigned int distance = 0;
  do
  {
    // Get symbol offset by counting # of lower threads that are not copies
    unsigned int symbol_offset = __popc(__ballot_sync(WARP_ALL, !isCopy) & ltMask()) - (unsigned int)(!isCopy);
    unsigned int symbol_idx = symbol_counter + symbol_offset;
    bool active = isCopy | (symbol_idx <= num_symbols);
    bool isLitlen = (symbol_idx < num_symbols) && (!isCopy);

    // Read length and distance symbols unless this is a copy round
    offset_type length = isLitlen ? lengths[symbol_idx] : 0;
    distance = isLitlen ? distances[symbol_idx] : distance;

    bool isLiteral = isLitlen && (length < gdeflateMinMatchLength);
    bool isLength = isLitlen && (!isLiteral);

    uint16_t sym = 0;
    if (isLiteral)
    {
      sym = literals[distance];
    }
    else if (isLength)
    {
      sym = length2sym<deflate64>(length);
    }
    else if (isCopy)
    {
      assert(distance > 0);
      sym = distance2sym<deflate64>(distance);
    }
    else if (symbol_idx == num_symbols)
    {
      sym = gdeflate_literal_symbols - 1; // End of block symbol
    }

    // Get the encoded literal/length/distance bits and number of bits
    uint32_t bits = 0;
    uint16_t numbits = isCopy ? encode_distance<deflate64>(d_enc, sym, distance, bits)
                              : encode_litlen<deflate64>(l_enc, sym, length, bits);

    // Write to the output stream
    bw.write(bits, numbits, active);

    // Update the symbol counter for the next round
    symbol_counter = __shfl_sync(WARP_ALL, symbol_idx + (unsigned int)(!isCopy), 31);

    // Set copy flag for next round to write out the encoded distance
    isCopy = isLength;
  } while (symbol_counter <= num_symbols); // Equality for end of block symbol

  // Last round to write out remaining copy distances
  {
    assert(!isCopy || distance > 0);
    uint16_t sym = isCopy ? distance2sym<deflate64>(distance) : 0;

    uint32_t bits = 0;
    uint16_t numbits = isCopy ? encode_distance<deflate64>(d_enc, sym, distance, bits) : 0;
    bw.write(bits, numbits, isCopy); // Only thread 0 writes to its stream
  }

  // Write out final remainder state to the stream
  uint32_t *compressed_stream_end = bw.finalize();
  *compressed_size = (size_t)(compressed_stream_end - output) * sizeof(uint32_t);
}

template <bool deflate64 = false, typename offset_type>
__device__ void gdeflate_encode_tile_standard(
  uint32_t *output,
  size_t *compressed_size,
  uint8_t *compressed_pad_bits,
  const offset_type *lengths,
  const offset_type *distances,
  const uint8_t *literals,
  const unsigned int num_symbols,
  const unsigned int num_literals
)
{
  // Can share these variables across literal and distance Huffman tree builders
  __shared__ uint16_t left[gdeflate_litlen_symbols];
  __shared__ uint16_t right[gdeflate_litlen_symbols];
  __shared__ uint16_t depth[2 * gdeflate_litlen_symbols];

  // Literal symbols and counts
  __shared__ uint16_t l_symbols[gdeflate_litlen_symbols];
  __shared__ unsigned int l_counts[2 * gdeflate_litlen_symbols];

  // Distance symbols and counts
  __shared__ uint16_t d_symbols[gdeflate_distance_symbols];
  __shared__ unsigned int d_counts[2 * gdeflate_distance_symbols];

  __shared__ cuda::atomic<size_t, cuda::thread_scope_block> output_pos;
  if (threadIdx.x == 0)
  {
    output_pos.store(0, cuda::std::memory_order_relaxed);
  }

  // Keep a contiguous buffer for literal and distance codelengths
  uint16_t *codelen = reinterpret_cast<uint16_t *>(l_counts); // Have 2*gdeflate_litlen_symbols > gdeflate_total_symbols
  uint16_t *l_codelen = codelen;
  uint16_t *d_codelen = codelen + gdeflate_litlen_symbols;
  uint32_t *l_codes = reinterpret_cast<uint32_t *>(&l_counts[gdeflate_litlen_symbols]);
  uint32_t *d_codes = reinterpret_cast<uint32_t *>(&d_counts[gdeflate_distance_symbols]);

  for (unsigned int i = threadIdx.x; i < gdeflate_litlen_symbols; i += WARP_SIZE_U)
  {
    l_counts[i] = 0;
  }
  for (unsigned int i = threadIdx.x; i < gdeflate_distance_symbols; i += WARP_SIZE_U)
  {
    d_counts[i] = 0;
  }
  __syncwarp();

  // Compute counts of each literal character
  update_histogram(l_counts, literals, num_literals);

  // Add end of block character
  if (threadIdx.x == 0)
  {
    l_counts[gdeflate_literal_symbols - 1] = 1;
  }

  // Counts of lengths and distances
  for (unsigned int i = threadIdx.x; i < num_symbols; i += WARP_SIZE_U)
  {
    offset_type len = lengths[i];
    offset_type dist = distances[i];
    // Check if this token is a match
    if (len >= gdeflateMinMatchLength)
    {
      // Update length histogram
      uint16_t b = length2sym<deflate64>(len);
      atomicAdd(&l_counts[b], 1);

      // Update distance histogram
      assert(dist > 0);
      uint16_t d = distance2sym<deflate64>(dist);
      atomicAdd(&d_counts[d], 1);
    }
  }

  // Handle edge case when there are no copies
  // Add dummy distance symbol
  if ((num_symbols == num_literals) && (threadIdx.x == 0))
  {
    d_counts[distance2sym<deflate64>(1)] = 1;
  }

  __syncwarp();

  // Literal tree builder
  WarpHuffmanTree<gdeflate_litlen_symbols, gdeflate_max_codelen> lht(l_counts, l_symbols);
  lht.build(l_counts, left, right, depth);
  lht.codelengths(l_codelen, l_symbols, depth);

  // Distance tree builder
  WarpHuffmanTree<gdeflate_distance_symbols, gdeflate_max_codelen> dht(d_counts, d_symbols);
  dht.build(d_counts, left, right, depth);
  dht.codelengths(d_codelen, d_symbols, depth);

  // Build the literal/length Huffman encoder
  warp_encoder<gdeflate_litlen_symbols, gdeflate_max_codelen> l_enc;
  l_enc.init(
    gdeflate_litlen_symbols,
    l_codelen,
    l_codes,
    reinterpret_cast<uint16_t *>(left),
    reinterpret_cast<unsigned int *>(right)
  );

  // Build the distance Huffman encoder
  warp_encoder<gdeflate_distance_symbols, gdeflate_max_codelen> d_enc;
  d_enc.init(
    gdeflate_distance_symbols,
    d_codelen,
    d_codes,
    reinterpret_cast<uint16_t *>(left),
    reinterpret_cast<unsigned int *>(right)
  );

  // Initialize the warp bitwriter
  warp_bitwriter_standard<uint32_t> bw(output);

  // Write the stream header
  uint32_t bfinal = 1, btype = 2, header = 0;
  header = ((btype & 3) << 1) | (bfinal & 1);
  bw.standard_write_header(header, 3, output_pos, threadIdx.x == 0); // Only thread 0 writes to its stream
  // Huffman encode codelengths and write them to the output stream
  // Also writes the deflate block header
  pack_codelens(codelen, deflate_valid_litlen_symbols, deflate_valid_distance_symbols, bw, output_pos);

  unsigned int symbol_counter = 0;
  unsigned int distance = 0;
  do
  {
    unsigned int symbol_idx = symbol_counter + threadIdx.x;
    bool active = symbol_idx <= num_symbols;
    bool isLitlen = symbol_idx < num_symbols;

    // Read length and distance symbols unless this is a copy round
    offset_type length = isLitlen ? lengths[symbol_idx] : 0;
    distance = isLitlen ? distances[symbol_idx] : distance;

    bool isLiteral = isLitlen && (length < gdeflateMinMatchLength);
    bool isLength = isLitlen && (!isLiteral);
    uint16_t sym = 0, disnumbits = 0;
    uint32_t disbits = 0, lenbits = 0;
    uint64_t bits = 0;
    if (isLiteral)
    {
      sym = literals[distance];
    }
    else if (isLength)
    {
      assert(distance > 0);
      sym = distance2sym<deflate64>(distance);
      disnumbits = encode_distance<deflate64>(d_enc, sym, distance, disbits);
      sym = length2sym<deflate64>(length);
    }
    else if (symbol_idx == num_symbols)
    {
      sym = gdeflate_literal_symbols - 1; // End of block symbol
    }

    // Get the encoded literal/length/distance bits and number of bits

    uint16_t numbits = encode_litlen<deflate64>(l_enc, sym, length, lenbits);
    assert(numbits + disnumbits <= 64);

    bits = isLiteral ? (uint64_t)((uint64_t)(lenbits & mask<uint64_t>(numbits)))
                     : (((uint64_t)(disbits & mask<uint64_t>(disnumbits)) << numbits) |
                        ((uint64_t)(lenbits & mask<uint64_t>(numbits))));
    numbits = isLiteral ? numbits : numbits + disnumbits;

    int check = false;
    // Write to the output stream
    bw.standard_write(bits, numbits, output_pos, active, check);

    // Update the symbol counter for the next round
    symbol_counter += WARP_SIZE_U;

  } while (symbol_counter <= num_symbols); // Equality for end of block symbol

  size_t total_bits = output_pos.load(cuda::std::memory_order_relaxed);
  *compressed_size = roundUpDiv(total_bits, 8);
  if (compressed_pad_bits != nullptr && threadIdx.x == 0)
  {
    // save information about how many padding bits are in the last byte.
    *compressed_pad_bits = padBitsToByte(total_bits);
  }
}

// TODO: CUB BlockRadixSort limits this implementation to 1 warp per CTA
template <typename offset_type>
__global__ void __launch_bounds__(WARP_SIZE) gdeflate_encode(
  uint32_t *const *outputs,
  size_t *compressed_sizes,
  const offset_type *const *lengths,
  const offset_type *const *distances,
  const uint8_t *const *literals,
  const unsigned int *num_symbols,
  const unsigned int *num_literals,
  const size_t num_tiles_host,
  const int *device_num_tiles
)
{
  assert(blockDim.x == WARP_SIZE_U);
  const size_t num_tiles = device_num_tiles ? static_cast<size_t>(*device_num_tiles) : num_tiles_host;
  size_t stride = gridDim.x * blockDim.y;
  for (size_t bid = blockIdx.x * blockDim.y + threadIdx.y; bid < num_tiles; bid += stride)
  {
    gdeflate_encode_tile(
      outputs[bid],
      &compressed_sizes[bid],
      lengths[bid],
      distances[bid],
      literals[bid],
      num_symbols[bid],
      num_literals[bid]
    );
  }
}

// TODO: CUB BlockRadixSort limits this implementation to 1 warp per CTA
template <typename offset_type>
__global__ void __launch_bounds__(WARP_SIZE) gdeflate_encode_standard(
  uint32_t *const *outputs,
  size_t *compressed_sizes,
  uint8_t *compressed_pad_bits,
  const offset_type *const *lengths,
  const offset_type *const *distances,
  const uint8_t *const *literals,
  const unsigned int *num_symbols,
  const unsigned int *num_literals,
  const size_t num_tiles_host,
  const int *device_num_tiles
)
{
  assert(blockDim.x == WARP_SIZE_U);
  const size_t num_tiles = device_num_tiles ? static_cast<size_t>(*device_num_tiles) : num_tiles_host;
  size_t stride = gridDim.x * blockDim.y;
  for (size_t bid = blockIdx.x * blockDim.y + threadIdx.y; bid < num_tiles; bid += stride)
  {
    gdeflate_encode_tile_standard(
      outputs[bid],
      &compressed_sizes[bid],
      compressed_pad_bits ? &compressed_pad_bits[bid] : nullptr,
      lengths[bid],
      distances[bid],
      literals[bid],
      num_symbols[bid],
      num_literals[bid]
    );
  }
}

template <bool deflate64, typename offset_type>
__global__ void lz_compress_optimal_parse(
  const unsigned char *const *input_ptrs,
  const size_t *input_bytes,
  const unsigned int maxLength,
  unsigned int **cost_ptrs,
  offset_type **length_ptrs,
  offset_type **distance_ptrs,
  uint8_t **literal_ptrs,
  unsigned int *num_symbols,
  unsigned int *num_literals,
  const unsigned int num_tiles_host,
  const int *device_num_tiles
)
{
  constexpr uint8_t avg_length_bits = 6;
  constexpr uint8_t avg_distance_bits = 5;
  NaiveCostMetric<deflate64> cost_evaluator(avg_length_bits, avg_distance_bits);

  const unsigned int num_tiles = device_num_tiles ? static_cast<unsigned int>(*device_num_tiles) : num_tiles_host;
  unsigned int stride = gridDim.x;
  for (unsigned int bid = blockIdx.x; bid < num_tiles; bid += stride)
  {
    lz_compress_tile_optimal_parse(
      input_ptrs[bid],
      (unsigned int)input_bytes[bid],
      maxLength,
      cost_evaluator,
      cost_ptrs[bid],
      length_ptrs[bid],
      distance_ptrs[bid],
      literal_ptrs[bid]
    );
    write_chosen_match(
      input_ptrs[bid],
      (unsigned int)input_bytes[bid],
      length_ptrs[bid],
      distance_ptrs[bid],
      literal_ptrs[bid],
      num_symbols + bid,
      num_literals + bid
    );
  }
}

} // namespace gdeflate
