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

#include "prescan.cuh"

#include <stdint.h>
#include <stdio.h>

// Everything that you need to decode a bit stream into a Huffman
// symbol and other data about the symbol.
template <uint MAX_BITS, uint LENGTHS_COUNT>
struct HuffmanTable
{
  template <uint NEW_LENGTHS_COUNT>
  inline __device__ HuffmanTable<MAX_BITS, NEW_LENGTHS_COUNT> &slice()
  {
    return *reinterpret_cast<HuffmanTable<MAX_BITS, NEW_LENGTHS_COUNT> *>(this);
  }
  // Given the next 32 bits from the stream, compute the length of the
  // code, not including any extra_bits.  The bits should be reversed
  // already so that the first bit of the stream is in the MSB.
  inline __device__ uint bits_to_length(uint32_t bit_buffer) const
  {
    bit_buffer &= 0xfffffffe; // We need to make sure that this is smaller
    // than the values that are beyond
    // max_code_lengths.
    uint ret = 0;
    if (MAX_BITS > 8)
    {
      if (bit_buffer >= length_to_min_code[ret + 8])
      {
        ret += 8;
      }
    }
    if (bit_buffer >= length_to_min_code[ret + 4])
    {
      ret += 4;
    }
    if (bit_buffer >= length_to_min_code[ret + 2])
    {
      ret += 2;
    }
    if (bit_buffer >= length_to_min_code[ret + 1])
    {
      ret += 1;
    }
    return ret;
  }
  inline __device__ uint get_offset(uint bit_length, uint32_t bit_buffer) const
  {
    auto min_code = length_to_min_code[bit_length];
    auto offset = (bit_buffer - min_code) >> (32 - bit_length);
    offset += symbols_before_length[bit_length];
    return offset;
  }
  inline __device__ uint16_t get_symbol(uint offset) const { return symbols[offset]; }

  __host__ __device__ void print() const
  {
    if (is_non_compressed())
    {
      printf("non-compressed length: %d\n", get_non_compressed_length());
      return;
    }
    for (uint i = 0; i < MAX_BITS + 1; i++)
    {
      printf("%3d: ", i);
      for (size_t j = 0; j < MAX_BITS + 1; j++)
      {
        if (j < i)
        {
          printf("%d", (length_to_min_code[i] >> (31 - j)) & 1);
        }
        else
        {
          printf(" ");
        }
      }
      printf("symbols before: %3d\n", symbols_before_length[i]);
    }
    for (size_t i = 0; i < LENGTHS_COUNT; i++)
    {
      if (i % 16 == 0)
      {
        printf("\n");
      }
      printf("%3d ", symbols[i]);
    }
    printf("\n");
  }

  // How many symbols are there that have a length that is less than
  // `length`?
  inline __device__ uint get_symbols_before_length(uint length) const { return symbols_before_length[length]; }
  inline __device__ void set_symbols_before_length(uint length, uint count) { symbols_before_length[length] = count; }
  inline __device__ void prescan_symbols_before_length()
  {
    block_prescan<MAX_BITS + 1>(symbols_before_length, &symbols_count);
  }
  inline __device__ void set_symbol(uint index, uint symbol) { symbols[index] = symbol; }
  inline __device__ void set_length_to_min_code(uint length, uint min_code) { length_to_min_code[length] = min_code; }
  inline __device__ uint get_length_to_min_code(uint length) const { return length_to_min_code[length]; }
  inline __device__ uint get_symbols_count() const { return symbols_count; }
  inline __device__ void set_non_compressed() { symbols_count = 0; }
  inline __device__ bool is_non_compressed() const { return symbols_count == 0; }
  inline __device__ void set_non_compressed_length(uint16_t len) { non_compressed_length = len; }
  inline __device__ uint16_t get_non_compressed_length() const { return non_compressed_length; }
  inline __device__ void set_to_fixed_litlen()
  {
    static_assert(MAX_BITS == 15, "Required by the RFC.");
    static_assert(LENGTHS_COUNT == 286, "Required by the RFC.");
    // As described in RFC 1951 section 3.2.6.
    this->symbols_count = 286;
    for (uint i = 0; i < 280 - 256; i++)
    {
      this->symbols[i] = 256 + i;
    }
    for (uint i = 0; i < 144 - 0; i++)
    {
      this->symbols[i + 280 - 256] = i + 0;
    }
    for (uint i = 0; i < 286 - 280; i++)
    {
      this->symbols[i + 280 - 256 + 144 - 0] = i + 280;
    }
    for (uint i = 0; i < 256 - 144; i++)
    {
      this->symbols[i + 280 - 256 + 144 - 0 + 286 - 280] = i + 144;
    }
    for (uint i = 0; i < 8; i++)
    {
      this->symbols_before_length[i] = 0;
      this->length_to_min_code[i] = 0;
    }
    this->symbols_before_length[8] = this->symbols_before_length[7] + 280 - 256;
    this->symbols_before_length[9] = this->symbols_before_length[8] + 286 - 280 + 144 - 0;
    this->symbols_before_length[10] = this->symbols_before_length[9] + 256 - 144;
    this->length_to_min_code[8] = 0b00110000000000000000000000000000;
    this->length_to_min_code[9] = 0b11001000000000000000000000000000;
    this->length_to_min_code[10] = 0b11111111111111111111111111111111;
    for (uint i = 11; i < 16; i++)
    {
      this->symbols_before_length[i] = this->symbols_before_length[10];
      this->length_to_min_code[i] = 0xffffffff;
    }
  }
  inline __device__ void set_to_fixed_dist()
  {
    static_assert(MAX_BITS == 15, "Required by the RFC.");
    static_assert(LENGTHS_COUNT == 30, "Required by the RFC.");
    // As described in RFC 1951 section 3.2.6.
    this->symbols_count = 30;
    for (uint i = 0; i < 30; i++)
    {
      this->symbols[i] = i;
    }
    for (uint i = 0; i < 6; i++)
    {
      this->symbols_before_length[i] = 0;
      this->length_to_min_code[i] = 0;
    }
    for (uint i = 6; i < 16; i++)
    {
      this->symbols_before_length[i] = 30;
      this->length_to_min_code[i] = 0xffffffff;
    }
  }

private:
  // The order below is important so that slice continues to work.

  // Use this for a binary search to determine the length of the code.
  // bits2length[x] is the smallest value of symbol that has length x.
  uint32_t length_to_min_code[MAX_BITS + 1];
  // All the lengths of the same size are together in the symbol
  // array.  So first it's all the length 1 symbols, then all the
  // length 2 symbols, etc.  symbols_before_length[x] tells us how
  // many symbols there are with length less than x.
  union
  {
    uint16_t non_compressed_length;
    uint symbols_before_length[MAX_BITS + 1];
  };
  uint symbols_count;
  // The symbols themselves, sorted by length primarily and then by
  // value.
  uint16_t symbols[LENGTHS_COUNT];
};

struct HuffmanInfo
{
  uint header_bits_size; // How many bits were needed to represent
  // these Huffman trees.
  bool is_final; // Is this the final deflate block within the stream?
  struct
  {
    static_assert(
      LITLEN_COUNT >= DIST_COUNT,
      "It's important that the litlen comes before the dist "
      "because we're going to occasionally pretend like this "
      "struct is an array of huffman_litlen, even though we "
      "would walk off the end of the array if we were to treat "
      "the second element as if it were as big as the first."
    );
    HuffmanTable<MAX_SYMBOL_BITS, LITLEN_COUNT> huffman_litlen;
    HuffmanTable<MAX_SYMBOL_BITS, DIST_COUNT> huffman_dist;
  } huffmans;
};
