/*
 * Copyright (c) 2021-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

namespace gdeflate
{

#define GDEFLATE_ENABLE_DEFLATE64

static constexpr size_t gdeflateMaxChunkSize = 1u << 31;
static constexpr const char *const gdeflateMaxChunkSizeText = "2GB";

static constexpr int gdeflateMinMatchLength = 3;
#ifdef GDEFLATE_ENABLE_DEFLATE64
static constexpr int gdeflateMaxMatchLength = (1 << 16) + 3;
static constexpr int gdeflateMaxDictionaryLength = 1 << 16;
#else
static constexpr int gdeflateMaxMatchLength = 258;
static constexpr int gdeflateMaxDictionaryLength = 1 << 15;
#endif

static constexpr int deflateMinMatchLength = 3;
static constexpr int deflateMaxMatchLength = 258;
static constexpr int deflateL6MaxMatchLength = 32;
static constexpr int deflateMaxDictionaryLength = 1 << 15;

static constexpr unsigned int gdeflate_max_codelen = 15;
static constexpr unsigned int gdeflate_literal_symbols = 257;
static constexpr unsigned int gdeflate_length_symbols = 31; //29
static constexpr unsigned int gdeflate_litlen_symbols = gdeflate_literal_symbols +
                                                        gdeflate_length_symbols; // Total 288 literal+length symbols
static constexpr unsigned int gdeflate_distance_symbols = 32; //30
static constexpr unsigned int gdeflate_total_symbols = gdeflate_litlen_symbols +
                                                       gdeflate_distance_symbols; // Total 320 symbols
static constexpr unsigned int gdeflate_valid_length_symbols = 29;
static constexpr unsigned int gdeflate_valid_litlen_symbols = gdeflate_literal_symbols +
                                                              gdeflate_valid_length_symbols; //Total 286
#ifdef GDEFLATE_ENABLE_DEFLATE64
static constexpr unsigned int gdeflate_valid_distance_symbols = 32;
#else
static constexpr unsigned int gdeflate_valid_distance_symbols = 30;
#endif
static constexpr unsigned int deflate_valid_length_symbols = 29;
static constexpr unsigned int deflate_valid_litlen_symbols = gdeflate_literal_symbols +
                                                             gdeflate_valid_length_symbols; //Total 286
static constexpr unsigned int deflate_valid_distance_symbols = 30;
static constexpr unsigned int gdeflate_alphabet_size = 19;
static constexpr unsigned int gdeflate_codelen_max_codelen = 7;
static constexpr unsigned int gdeflate_total_symbols_smem = 320;

// Lookup tables shared by the GPU decoder and the CPU (host) reference paths.
// On device they live in constant memory (__constant__); on host the exact same
// definitions are exposed as constexpr so the CPU code in cpu/ can reuse them
// instead of carrying its own copies.
#ifdef __CUDACC__
#define CONSTANT_ARRAY static __constant__
#else
#define CONSTANT_ARRAY static constexpr
#endif // __CUDACC__

CONSTANT_ARRAY unsigned int map[] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15}; // 5 bits
CONSTANT_ARRAY unsigned int invmap[] = {3, 17, 15, 13, 11, 9, 7, 5, 4, 6, 8, 10, 12, 14, 16, 18, 0, 1, 2};

// Length and distance tables for symbol translation
// Pad all arrays to 32 (for warp_dual_decoder)
CONSTANT_ARRAY unsigned char xlenbits32[] = {
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  1,
  1,
  1,
  1,
  2,
  2,
  2,
  2,
  3,
  3,
  3,
  3,
  4,
  4,
  4,
  4,
  5,
  5,
  5,
  5,
  // in standard deflate, last litlen code (285) has no extra bits
  0,
  0,
  0,
  0
}; // 3 bits

CONSTANT_ARRAY uint16_t lenbase32[] = {
  3,
  4,
  5,
  6,
  7,
  8,
  9,
  10,
  11,
  13,
  15,
  17,
  19,
  23,
  27,
  31,
  35,
  43,
  51,
  59,
  67,
  83,
  99,
  115,
  131,
  163,
  195,
  227,
  // last litlen code (285) specifies a copy length of 258 bytes in standard deflate
  258,
  0,
  0,
  1
}; // 9 bits, last value set for literals

CONSTANT_ARRAY unsigned char xdistbits32[] = {0, 0, 0, 0, 1, 1, 2,  2,  3,  3,  4,  4,  5,  5,  6, 6,
                                              7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 0, 0}; // 4 bits

CONSTANT_ARRAY uint16_t distanceTable32[] = {1,    2,    3,    4,    5,    7,     9,     13,    17,  25,   33,
                                             49,   65,   97,   129,  193,  257,   385,   513,   769, 1025, 1537,
                                             2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577, 0,   0}; // 15 bits

// Length and distance tables for symbol translation
// Pad all arrays to 32 (for warp_dual_decoder)
CONSTANT_ARRAY unsigned char xlenbits64[] = {
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  1,
  1,
  1,
  1,
  2,
  2,
  2,
  2,
  3,
  3,
  3,
  3,
  4,
  4,
  4,
  4,
  5,
  5,
  5,
  5,
  // in deflate64, last litlen code (285) is followed by extra 16 bits of length
  16,
  0,
  0,
  0
}; // 5 bits

CONSTANT_ARRAY uint16_t lenbase64[] = {
  3,
  4,
  5,
  6,
  7,
  8,
  9,
  10,
  11,
  13,
  15,
  17,
  19,
  23,
  27,
  31,
  35,
  43,
  51,
  59,
  67,
  83,
  99,
  115,
  131,
  163,
  195,
  227,
  // last litlen code (285) specifies a "long" copy between 3 and 65538 bytes, sent as an additional 16 bit number
  3,
  0,
  0,
  1
}; // 9 bits, last value set for literals

CONSTANT_ARRAY unsigned char xdistbits64[] = {
  0,
  0,
  0,
  0,
  1,
  1,
  2,
  2,
  3,
  3,
  4,
  4,
  5,
  5,
  6,
  6,
  7,
  7,
  8,
  8,
  9,
  9,
  10,
  10,
  11,
  11,
  12,
  12,
  13,
  13,
  // deflate64 uses two extra distance codes (30 and 31) each having 14 extra bits of distance
  14,
  14
};

CONSTANT_ARRAY uint16_t distanceTable64[] = {
  1,   2,   3,   4,    5,    7,    9,    13,   17,   25,   33,    49,    65,    97,    129,  193, 257,
  385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577, 32769, 49153
}; // 16 bits, deflate64 uses two extra copy codes to cover distances up to 64K

#undef CONSTANT_ARRAY
#ifdef GDEFLATE_ENABLE_DEFLATE64
static constexpr unsigned int gdeflate_max_extra_bits = 16;
#else
static constexpr unsigned int gdeflate_max_extra_bits = 13;
#endif
} // namespace gdeflate
