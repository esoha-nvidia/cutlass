/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier:
 * LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <cassert>

#include "nvcomp/ans.h"
#include "nvcomp/cascaded.h"
#include "nvcomp/native/bitcomp_mode.h"
#include "nvcomp/shared_types.h"

namespace nvcomp
{

/**
 * @brief Format specification for ANS compression.
 *
 * Layout matches the prefix of nvcompBatchedANSCompressOpts_t. A zeroed spec
 * is the 6.0 default (rANS, CHAR, auto states_per_lane, default sub-chunk
 * count of 8, exact histogram). ANS 6.0 bitstreams are not compatible with
 * earlier nvCOMP.
 */
struct ANSFormatSpecHeader
{
  /**
   * @brief ANS algorithm to use.
   */
  nvcompANSType_t type;
  /**
   * @brief ANS data type to use.
   *
   * - NVCOMP_TYPE_(U)CHAR: 1-byte, generic data type
   * - NVCOMP_TYPE_FLOAT16: 2-byte floating-point data type. Applicable to all half-precision data formats.
   * - NVCOMP_TYPE_FLOAT8_E4M3: 1-byte FP8 (E4M3) floating-point data type.
   * - NVCOMP_TYPE_FLOAT32: 4-byte IEEE-754 single-precision floating-point data type.
   */
  nvcompType_t data_type;
  /**
   * @brief Maximum sub chunk count override for compression.
   * 0: default of 8. Nonzero must be a power-of-2 between 4 and 64.
   * Leave zero unless you can tune the performance of your e2e application based on this value.
   */
  uint8_t max_sub_chunk_count;
  /**
   * @brief Number of interleaved rANS states per lane.
   * 0: auto (default): two interleaved streams for every data type.
   * 1: single stream.
   * 2: two interleaved streams.
   * "2 states" improves performance but adds an overhead of 128B per sub chunk, impacting
   * compression ratio; the relative cost grows as the sub chunk shrinks.
   */
  uint8_t states_per_lane;
  /**
   * @brief Reduces the amount of data used to build the histogram by this log2 factor.
   * 0: exact model (default). N keeps 1/2^N of the slice (1 = 1/2, 2 = 1/4, 3 = 1/8,
   * up to nvcompANSMaxHistogramReductionLog2). Useful to speed up compression when
   * full chunks are known to be sampled from a common distribution.
   * Applies to every data type. Chunks below 4096 ANS symbols are always histogrammed
   * exactly, whatever the requested reduction.
   */
  uint8_t histogram_reduction_log2;
  /**
   * @brief Unused; must be zero. Reserved for future FormatSpec extensions.
   * 5 bytes to make the last padding byte explicit.
   */
  char reserved[5];
};

static_assert(
  sizeof(ANSFormatSpecHeader) == 16,
  "ANSFormatSpecHeader must serialize as nvcompANSType_t + nvcompType_t + 3 uint8_t + 5 reserved"
);

/**
 * @brief Format specification for Bitcomp compression
 */
struct BitcompFormatSpecHeader
{
  /**
   * @brief Bitcomp algorithm options.
   *
   * - 0 : Default algorithm, usually gives the best compression ratios
   * - 1 : "Sparse" algorithm, works well on sparse data (with lots of zeroes),
   *        and is usually faster than the default algorithm.
   */
  int algorithm;
  /**
   * @brief Data type of the uncompressed input.
   *
   * Lossless compression supports the integral nvcomp types as well as
   * NVCOMP_TYPE_FLOAT16, NVCOMP_TYPE_FLOAT32, and NVCOMP_TYPE_FLOAT64. Lossy
   * compression supports only those three floating-point types.
   */
  nvcompType_t data_type;
  /**
   * @brief Scalar quantization delta used for lossy compression.
   *
   * The maximum error between decompressed and original values is at most
   * delta / 2. The value is rounded down to the nearest power of two. Lossy
   * compression requires delta to remain finite, positive, and normal after
   * conversion to the selected data type. It is ignored in lossless mode.
   */
  double delta;
  /**
   * @brief Compression mode.
   */
  bitcompMode_t mode;
  /**
   * @brief Unused; must be zero. Reserved for future FormatSpec extensions.
   */
  char reserved[4];
};

static_assert(
  sizeof(BitcompFormatSpecHeader) == 24,
  "BitcompFormatSpecHeader must serialize as int + nvcompType_t + double + bitcompMode_t + 4 reserved"
);

/**
 * @brief Format specification for Cascaded compression
 */
struct CascadedFormatSpecHeader
{
  /**
   * @brief Common options shared by compression and decompression.
   */
  nvcompCascadedCommonOpts_t common_opts;
  /**
   * @brief Mask of fine-grained encodings considered during compression.
   */
  uint64_t fine_grained_encoding_flags;
  /**
   * @brief The requested compression level.
   */
  uint8_t compression_level;
  /**
   * @brief Unused; must be zero. Reserved for future FormatSpec extensions.
   */
  char reserved[4];
};

static_assert(
  sizeof(CascadedFormatSpecHeader) == 40,
  "CascadedFormatSpecHeader must serialize as common options + uint64_t + uint8_t + 4 reserved bytes + padding"
);

/**
 * @brief Format specification for Deflate compression
 */
struct DeflateFormatSpecHeader
{
  /**
   * @brief Compression algorithm to use.
   *
   * - 0: highest-throughput, entropy-only compression (use for symmetric
   * compression/decompression performance)
   * - 1: high-throughput, low compression ratio (default)
   * - 2: medium-throughput, medium compression ratio, beat Zlib level 1 on the
   * compression ratio
   * - 3: placeholder for further compression level support, will fall into
   * MEDIUM_COMPRESSION at this point
   * - 4: lower-throughput, higher compression ratio, beat Zlib level 6 on the
   * compression ratio
   * - 5: lowest-throughput, highest compression ratio
   */
  int algorithm;
  /**
   * @brief Unused; must be zero. Reserved for future FormatSpec extensions.
   */
  char reserved[4];
};

static_assert(
  sizeof(DeflateFormatSpecHeader) == 8,
  "DeflateFormatSpecHeader must serialize as int algorithm + 4 reserved"
);

/**
 * @brief Format specification for GDeflate compression
 */
struct GdeflateFormatSpecHeader
{
  /**
   * @brief Compression algorithm to use.
   *
   * - 0: highest-throughput, entropy-only compression (use for symmetric
   * compression/decompression performance)
   * - 1: high-throughput, low compression ratio (default)
   * - 2: medium-throughput, medium compression ratio, beat Zlib level 1 on the
   * compression ratio
   * - 3: placeholder for further compression level support, will fall into
   * MEDIUM_COMPRESSION at this point
   * - 4: lower-throughput, higher compression ratio, beat Zlib level 6 on the
   * compression ratio
   * - 5: lowest-throughput, highest compression ratio
   */
  int algorithm;
  /**
   * @brief Unused; must be zero. Reserved for future FormatSpec extensions.
   */
  char reserved[4];
};

static_assert(
  sizeof(GdeflateFormatSpecHeader) == 8,
  "GdeflateFormatSpecHeader must serialize as int algorithm + 4 reserved"
);

/**
 * @brief Format specification for Gzip compression
 */
struct GzipFormatSpecHeader
{
  /**
   * @brief Compression algorithm to use.
   *
   * - 0: highest-throughput, lowest compression ratio, entropy-only compression (use for symmetric
   * compression/decompression performance)
   * - 1: high-throughput, low compression ratio (default)
   * - 2: medium-throughput, medium compression ratio, beat Zlib level 1 on the
   * compression ratio
   * - 3: placeholder for further compression level support, will fall into
   * MEDIUM_COMPRESSION at this point
   * - 4: lower-throughput, higher compression ratio, beat Zlib level 6 on the
   * compression ratio
   * - 5: lowest-throughput, highest compression ratio
   */
  int algorithm;
  /**
   * @brief Unused; must be zero. Reserved for future FormatSpec extensions.
   */
  char reserved[4];
};

static_assert(sizeof(GzipFormatSpecHeader) == 8, "GzipFormatSpecHeader must serialize as int algorithm + 4 reserved");

/**
 * @brief Format specification for LZ4 compression
 */
struct LZ4FormatSpecHeader
{
  /**
   * @brief LZ4 data type to use.
   */
  nvcompType_t data_type;
  /**
   * @brief Bitshuffle mode used during compression.
   */
  nvcompBitshuffleMode_t bitshuffle_mode;
  /**
   * @brief Unused; must be zero. Reserved for future FormatSpec extensions.
   */
  char reserved[4];
};

static_assert(
  sizeof(LZ4FormatSpecHeader) == 12,
  "LZ4FormatSpecHeader must serialize as nvcompType_t + nvcompBitshuffleMode_t + 4 reserved"
);

/**
 * @brief Format specification for Snappy compression
 */
struct SnappyFormatSpecHeader
{
  /**
   * @brief Unused; must be zero. Reserved for future FormatSpec extensions.
   */
  char reserved[4];
};

static_assert(sizeof(SnappyFormatSpecHeader) == 4, "SnappyFormatSpecHeader must serialize as 4 reserved");

/**
 * @brief Format specification for Zstd compression
 */
struct ZstdFormatSpecHeader
{
  /**
   * @brief Unused; must be zero. Reserved for future FormatSpec extensions.
   */
  char reserved[4];
};

static_assert(sizeof(ZstdFormatSpecHeader) == 4, "ZstdFormatSpecHeader must serialize as 4 reserved");

} // namespace nvcomp
