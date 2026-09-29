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

#include "CudaConstants.h"

// #define STAGE_LOGGING 1

namespace zstd
{

constexpr int HUF_MAX_SYMBOLS = 256;
constexpr int HUF_MAX_BITS = 11;
constexpr int HUF_MAX_OFFSET_SIZE = HUF_MAX_BITS + 1;
constexpr int HUF_FSE_WEIGHT_MAX_ACCURACY_LOG = 6;

constexpr size_t ZSTD_BLOCK_SIZE_MAX = 128 * 1024;
constexpr size_t MAX_LITERALS_SIZE = ZSTD_BLOCK_SIZE_MAX;

constexpr uint8_t DEFAULT_LITERAL_LENGTH_ACCURACY = 6;
constexpr uint8_t DEFAULT_OFFSET_ACCURACY = 5;
constexpr uint8_t DEFAULT_MATCH_LENGTH_ACCURACY = 6;
constexpr uint8_t LITERAL_LENGTH_MAX_SYMBOLS = 36;
constexpr uint8_t OFFSET_MAX_SYMBOLS = 29;
constexpr uint8_t MATCH_LENGTH_MAX_SYMBOLS = 53;
constexpr uint8_t LITERAL_LENGTH_MAX_ACCURACY = 9;
constexpr uint8_t OFFSET_MAX_ACCURACY = 8;
constexpr uint8_t MATCH_LENGTH_MAX_ACCURACY = 9;
constexpr uint8_t MAX_ANS_ACCURACY = 9;
constexpr int ANS_ALGORITHM_MAX_TABLELOG = 11;
constexpr unsigned MAX_ANS_TABLE_SIZE = 1 << MAX_ANS_ACCURACY;
constexpr int ANS_MAX_SYMBOLS = 256;
constexpr int SEQ_ANS_MAX_SYMBOLS = 54;
constexpr int ZSTD_MAX_MATCH_LENGTH = (1 << 17) + 3;
constexpr int ZSTD_MIN_MATCH_LENGTH = 4;
constexpr int ZSTD_BLOCK_HEADER_SIZE = 3; // This is always the same
constexpr int ZSTD_FRAME_CONTENT_SIZE_2BYTE_THRESHOLD = 255;
constexpr int ZSTD_FRAME_CONTENT_SIZE_4BYTE_THRESHOLD = 65791;
constexpr int ZSTD_SEQ_ANS_STREAMS = 3;
constexpr int ZSTD_FSE_MIN_TABLELOG = 5;
constexpr int MAX_COMPRESS_NUM_SEQUENCES = ZSTD_BLOCK_SIZE_MAX / ZSTD_MIN_MATCH_LENGTH;
constexpr int MIN_LITERALS_FOR_HUFF_COMPRESSION = 100;
constexpr int MIN_MATCH_COPY_BYTES_FOR_SEQ_COMPRESSION = 100;
constexpr int HUF_FSE_MAX_TABLE_SIZE = 1 << HUF_FSE_WEIGHT_MAX_ACCURACY_LOG;
constexpr int ZSTD_MAX_DEFAULT_SEQUENCES = 5;
constexpr int OFFSET_BIT_COUNT = 20;
constexpr int MAX_OFFSET = (1 << OFFSET_BIT_COUNT) - 3; // Note: Non-repeat offsets get added 3 before storage
constexpr size_t COMPRESS_NOMINAL_BLOCK_SIZE = 64 << 10;
constexpr int BLOCKS_PER_LZ_COMP_TASK = 8;
constexpr int ZSTD_COMP_MAX_MATCH_LENGTH = (1 << 16) - 1;
constexpr int LZ_BACKFILL_HASH_SIZE = 4 << 10;

const size_t SCRATCH_ALIGNMENT_REQ = 8;

// Sleeps below are used instead of barrier arrive/wait because barrier uses an exponential backoff spin
// loop behavior that doesn't work well here. We can spend such a high % of the time sleeping that it's not
// helpful to do the backoff -- warps in the sleep loop are not taking a significant % of overall resources,
// and just because i.e. we needed to delay 100 us, doesn't mean we want to wait another 100 us which can happen
// with the exponential backoff behavior
// TODO: Coordinate with CCCL team to give API for barrier behavior which fits our use case better

// used when delaying the start of LZ warps
constexpr int ZSTD_VERY_LONG_SLEEP_NS = 1000000;
// used in cases where a shorter wait would just lead to more waits after the waiting warp caught up
constexpr int ZSTD_LONG_SLEEP_NS = 5000;
// Used in cases where a shorter wait is suitable
// (but an even shorter wait would cause too much instruction / scheduler pressure)
constexpr int ZSTD_SHORT_SLEEP_NS = 1000;

constexpr __device__ __constant__ int ZSTD_INITIAL_REPEAT_OFFSETS[3] = {1, 4, 8};

constexpr __device__ __constant__ int16_t SEQ_LITERAL_LENGTH_DEFAULT_DIST[LITERAL_LENGTH_MAX_SYMBOLS] = {
  4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 1, 1, 1, 1, 1, -1, -1, -1, -1
};
constexpr __device__ __constant__ int16_t SEQ_OFFSET_DEFAULT_DIST[OFFSET_MAX_SYMBOLS] = {1,  1,  1,  1,  1, 1, 2, 2,
                                                                                         2,  1,  1,  1,  1, 1, 1, 1,
                                                                                         1,  1,  1,  1,  1, 1, 1, 1,
                                                                                         -1, -1, -1, -1, -1};
constexpr __device__ __constant__ int16_t SEQ_MATCH_LENGTH_DEFAULT_DIST[MATCH_LENGTH_MAX_SYMBOLS] = {
  1, 4, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1
};

// Define constant memory on the device to use for this.
constexpr __device__ __constant__ uint32_t SEQ_LITERAL_LENGTH_BASELINES[LITERAL_LENGTH_MAX_SYMBOLS] = {
  0,  1,  2,  3,  4,  5,  6,  7,  8,   9,   10,  11,   12,   13,   14,   15,    16,    18,
  20, 22, 24, 28, 32, 40, 48, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536
};
constexpr __device__ __constant__ uint8_t SEQ_LITERAL_LENGTH_EXTRA_BITS[LITERAL_LENGTH_MAX_SYMBOLS] = {
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
};

constexpr __device__ __constant__ uint8_t SEQ_OFFSET_EXTRA_BITS[OFFSET_MAX_SYMBOLS] = {0,  1,  2,  3,  4,  5,  6,  7,
                                                                                       8,  9,  10, 11, 12, 13, 14, 15,
                                                                                       16, 17, 18, 19, 20, 21, 22, 23,
                                                                                       24, 25, 26, 27, 28};
constexpr __device__ __constant__ uint32_t SEQ_OFFSET_BASELINES[OFFSET_MAX_SYMBOLS] = {
  1 << 0,  1 << 1,  1 << 2,  1 << 3,  1 << 4,  1 << 5,  1 << 6,  1 << 7,  1 << 8,  1 << 9,
  1 << 10, 1 << 11, 1 << 12, 1 << 13, 1 << 14, 1 << 15, 1 << 16, 1 << 17, 1 << 18, 1 << 19,
  1 << 20, 1 << 21, 1 << 22, 1 << 23, 1 << 24, 1 << 25, 1 << 26, 1 << 27, 1 << 28
};

constexpr __device__ __constant__ uint32_t SEQ_MATCH_LENGTH_BASELINES[MATCH_LENGTH_MAX_SYMBOLS] = {
  3,  4,  5,  6,  7,  8,  9,  10,  11,  12,  13,   14,   15,   16,   17,    18,    19,   20,
  21, 22, 23, 24, 25, 26, 27, 28,  29,  30,  31,   32,   33,   34,   35,    37,    39,   41,
  43, 47, 51, 59, 67, 83, 99, 131, 259, 515, 1027, 2051, 4099, 8195, 16387, 32771, 65539
};

constexpr __device__ __constant__ uint8_t SEQ_MATCH_LENGTH_EXTRA_BITS[MATCH_LENGTH_MAX_SYMBOLS] = {
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0, 0,
  0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
};

constexpr __device__ __constant__ uint8_t
  max_accuracies[3] = {LITERAL_LENGTH_MAX_ACCURACY, OFFSET_MAX_ACCURACY, MATCH_LENGTH_MAX_ACCURACY};

constexpr __device__ __constant__ uint8_t
  default_accuracies[3] = {DEFAULT_LITERAL_LENGTH_ACCURACY, DEFAULT_OFFSET_ACCURACY, DEFAULT_MATCH_LENGTH_ACCURACY};

constexpr const __device__ __constant__ int16_t *const default_distributions[3] =
  {SEQ_LITERAL_LENGTH_DEFAULT_DIST, SEQ_OFFSET_DEFAULT_DIST, SEQ_MATCH_LENGTH_DEFAULT_DIST};

constexpr const __device__ __constant__ uint32_t *const seq_baselines[3] =
  {SEQ_LITERAL_LENGTH_BASELINES, SEQ_OFFSET_BASELINES, SEQ_MATCH_LENGTH_BASELINES};

constexpr const __device__ __constant__ uint8_t *const seq_extra_bits[3] =
  {SEQ_LITERAL_LENGTH_EXTRA_BITS, SEQ_OFFSET_EXTRA_BITS, SEQ_MATCH_LENGTH_EXTRA_BITS};

// The predefined FSE distribution tables for predefined mode
constexpr const __device__ __constant__ int SEQ_MAX_SYMBOLS[3] = {36, 29, 53};
constexpr const __device__ __constant__ int SEQ_MAX_LOGS[3] = {6, 5, 6};

// This is a matrix where cols are for the sequence type, rows are for symbol, baseline and next_bits, respectively
// the fourth column is for ANS decoding of huffman weights. It describes how many bits to use in the bitpacked FSE tables
// for each and is based on the dynamic range of the value.
// Need both of these because constexpr __device__ __constant__ can't be used outside a
// constexpr expression in CTK 10.2
constexpr __device__ __constant__ uint8_t SEQ_MAX_BITS[4][3] = {{6, 9, 4}, {5, 8, 4}, {6, 9, 4}, {8, 7, 3}};
constexpr uint8_t CEXPR_SEQ_MAX_BITS[4][3] = {{6, 9, 4}, {5, 8, 4}, {6, 9, 4}, {8, 7, 3}};

constexpr int __device__ compute_ll_table_size_word()
{
  constexpr int table_size = 1 << LITERAL_LENGTH_MAX_ACCURACY;
  constexpr int cell_size_bits = CEXPR_SEQ_MAX_BITS[0][0] + CEXPR_SEQ_MAX_BITS[0][1] + CEXPR_SEQ_MAX_BITS[0][2];
  return (cell_size_bits * table_size + 31) / 32;
}

constexpr int __device__ compute_of_table_size_word()
{
  constexpr int table_size = 1 << OFFSET_MAX_ACCURACY;
  constexpr int cell_size_bits = CEXPR_SEQ_MAX_BITS[1][0] + CEXPR_SEQ_MAX_BITS[1][1] + CEXPR_SEQ_MAX_BITS[1][2];
  return (cell_size_bits * table_size + 31) / 32;
}

constexpr int __device__ compute_ml_table_size_word()
{
  constexpr int table_size = 1 << MATCH_LENGTH_MAX_ACCURACY;
  constexpr int cell_size_bits = CEXPR_SEQ_MAX_BITS[2][0] + CEXPR_SEQ_MAX_BITS[2][1] + CEXPR_SEQ_MAX_BITS[2][2];
  return (cell_size_bits * table_size + 31) / 32;
}

constexpr int max_ll_words = compute_ll_table_size_word();
constexpr int max_of_words = compute_of_table_size_word();
constexpr int max_ml_words = compute_ml_table_size_word();

// +1 to allow us to read from NUM_WORDS without OOB checking
// (The extra value isn't used but flags racecheck, 4 bytes is an OK price to pay per table to avoid racecheck FAs)
constexpr int max_fse_tables_words = max_ll_words + max_of_words + max_ml_words + 1;

constexpr int INIT_FSE_WARPS_PER_CTA = 2;
constexpr int FRAME_SIZES_MAX_ANS_TABLES = 3;

constexpr int NUM_THREADS_PER_ANS_BLOCK = 6;
constexpr int NUM_THREADS_PER_HUFF_BLOCK = 4;

// Number of tables processed in parallel by each warp
constexpr int MAX_HUFF_TABLES = WARP_SIZE / NUM_THREADS_PER_HUFF_BLOCK;

// The following arrays are used to ensure the HOST is aware of __cuda_arch__ dependant constants
// Architecture specific configurations for decompression
constexpr int NUM_ARCH_IDS = 8;
constexpr int CONFIG_ARCH[NUM_ARCH_IDS] = {1200, 900, 890, 860, 800, 750, 700, 0}; // Used by host to allocate dyn shmem
constexpr int CONFIG_NUM_WARPS_PER_CTA[NUM_ARCH_IDS] = {32, 32, 16, 18, 32, 20, 20, 8};
constexpr int CONFIG_NUM_HUFF_WARPS_PER_CTA[NUM_ARCH_IDS] = {6, 7, 6, 6, 6, 6, 6, 3};
constexpr int CONFIG_NUM_FSE_WARPS_PER_CTA[NUM_ARCH_IDS] = {6, 7, 6, 6, 6, 5, 4, 3};
constexpr int CONFIG_MAX_FSE_WARPS_PER_CTA[NUM_ARCH_IDS] =
  {6, 7, 6, 6, 6, 6, 4, 3}; // if (MAX_FSE > NUM_FSE) then a huff warp will become an FSE warp
constexpr int CONFIG_MAX_TOTAL_FSE_TABLES[NUM_ARCH_IDS] = {
  24,
  35,
  24,
  24,
  24,
  16,
  16,
  8
}; // Determines dyn shmem usage. Generally equal to (MAX_ANS_TABLES * NUM_FSE_WARPS) unless shmem size is limited.
constexpr int CONFIG_MAX_ANS_TABLES[NUM_ARCH_IDS] = {4, 5, 4, 4, 4, 4, 4, 4};

// Translates __CUDA_ARCH into an index for the above arrays. MUST match the CONFIG_ARCH values
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 1200
constexpr int ARCH_ID = 0;
#elif __CUDA_ARCH__ >= 900
constexpr int ARCH_ID = 1;
#elif __CUDA_ARCH__ >= 890
constexpr int ARCH_ID = 2;
#elif __CUDA_ARCH__ >= 860
constexpr int ARCH_ID = 3;
#elif __CUDA_ARCH__ >= 800
constexpr int ARCH_ID = 4;
#elif __CUDA_ARCH__ >= 750
constexpr int ARCH_ID = 5;
#elif __CUDA_ARCH__ >= 700
constexpr int ARCH_ID = 6;
#else
constexpr int ARCH_ID = (NUM_ARCH_IDS - 1);
#endif
#else
constexpr int ARCH_ID = 0;
#endif

// These constants are only safe to use inside the kernel
constexpr int NUM_WARPS_PER_CTA = CONFIG_NUM_WARPS_PER_CTA[ARCH_ID];
constexpr int NUM_HUFF_WARPS_PER_CTA = CONFIG_NUM_HUFF_WARPS_PER_CTA[ARCH_ID];
constexpr int NUM_FSE_WARPS_PER_CTA = CONFIG_NUM_FSE_WARPS_PER_CTA[ARCH_ID];
constexpr int MAX_FSE_WARPS_PER_CTA = CONFIG_MAX_FSE_WARPS_PER_CTA[ARCH_ID];
constexpr int MAX_TOTAL_FSE_TABLES = CONFIG_MAX_TOTAL_FSE_TABLES[ARCH_ID];
constexpr int MAX_ANS_TABLES =
  CONFIG_MAX_ANS_TABLES[ARCH_ID]; // Intentionally use fewer intra-warp groups than possible

// Calculating shared memory requirement for init fse
constexpr int PER_BLOCK_ANS_SHARE_SIZE = (2985 + 3) / 4 * 4;
constexpr int ANS_SHARE_SIZE = PER_BLOCK_ANS_SHARE_SIZE * MAX_ANS_TABLES;
constexpr int SHARE_ARRAY_SIZE = ANS_SHARE_SIZE;

constexpr int LZ_STATUS_COPY_STARTED_BIT = 0x1;
constexpr int LZ_STATUS_FULL_STARTED_BIT = 0x2;
constexpr int LZ_STATUS_COPY_FINISHED_BIT = 0x4;

constexpr int COMPRESS_STATUS_FSE_START_BIT = 0x1;
constexpr int COMPRESS_STATUS_HUFF_START_BIT = 0x2;
constexpr int COMPRESS_STATUS_FSE_FINISH_BIT = 0x4;
constexpr int COMPRESS_STATUS_HUFF_FINISH_BIT = 0x8;
constexpr int COMPRESS_STATUS_ENTROPY_STARTED = COMPRESS_STATUS_HUFF_START_BIT | COMPRESS_STATUS_FSE_START_BIT;
constexpr int COMPRESS_STATUS_ENTROPY_FINISHED = COMPRESS_STATUS_HUFF_FINISH_BIT | COMPRESS_STATUS_FSE_FINISH_BIT;

constexpr int LZ_REPEAT_OFFSET_LENGTH_COMPUTE_BIT = 0x8;
constexpr int LZ_HASH_OFFSET_LENGTH_COMPUTE_BIT = 0x4;
constexpr int LZ_HAS_HASH_OFFSET_BIT = 0x1;
constexpr int LZ_HAS_REPEAT_OFFSET_BIT = 0x2;
constexpr int LZ_REPEAT_OFFSET_MASK = LZ_HAS_REPEAT_OFFSET_BIT | LZ_REPEAT_OFFSET_LENGTH_COMPUTE_BIT;
constexpr int LZ_HASH_OFFSET_MASK = LZ_HAS_HASH_OFFSET_BIT | LZ_HASH_OFFSET_LENGTH_COMPUTE_BIT;

} // namespace zstd
