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

#include "common.cuh"
#include "include/lookahead_gzip.h"

#include <stdint.h>

// How many blocks?  The number of blocks affects how long it may take
// to decode a file because if there aren't enough blocks then the
// code might need to loop more in order to find the end of block But
// if it's too many than there will be more wasted effort of decoding
// beyond the end of the block.  gzip usually creates blocks that have
// 32k symbols.

// TODO:
// O2O vectors per block were originally 21, but this was decreased
// to favor complete wraps. This should be investigated to match the
// thread count of the deflate block header decoder CTA.
#ifndef O2O_VECTORS_PER_BLOCK
#define O2O_VECTORS_PER_BLOCK 20
#endif

// This is the maximum size of a compressed block in a CSV file that I
// tried, roughly.
// Note, that a compressed stream exceeding this block size does not do any
// harm, the effect is that more internal iterations will be performed, but less
// shared memory will be used, and potentially more CTAs can be scheduled.
#define EXPECTED_BLOCK_COMPRESSED_SIZE 56862

constexpr uint MAX_DISTANCE_EXTRA_BITS = 13;
constexpr uint MAX_LEN_EXTRA_BITS = 5;
// The offset can be [0..MAX_OFFSET) bits.
constexpr uint MAX_OFFSET = MAX_SYMBOL_BITS + MAX_DISTANCE_EXTRA_BITS + MAX_SYMBOL_BITS + MAX_LEN_EXTRA_BITS;

// Number of threads to use when adding two O2O vectors together.
#ifndef THREADS_PER_O2O_ADD
#define THREADS_PER_O2O_ADD (MAX_OFFSET / 2)
#endif

// Number of threads to use when copying an O2O vector.
#ifndef THREADS_PER_O2O_COPY
#define THREADS_PER_O2O_COPY (MAX_OFFSET / 2)
#endif

// Number of threads to use when decoding an O2O vector into words.
// The threads are shared with global LZ77 resolution so it's
// important to find a balance.
#ifndef THREADS_PER_O2O_DECODE
#define THREADS_PER_O2O_DECODE 16
#endif

// Whether to use pruning to decrease work.
#ifndef PRUNE
#define PRUNE true
#endif

// How many threads to use for each O2O that we make by decoding.
// Using fewer means more O2O vectors per block.
#ifndef THREADS_PER_O2O_CREATION
#define THREADS_PER_O2O_CREATION (MAX_OFFSET / 2)
#endif

// Putting in runtime assertions slows the code but can detect errors
// that might otherwise be hard to notice.
#ifndef ASSERTIONS
#define ASSERTIONS false
#endif

// Putting in timing data slows the code but is useful for measuring the effects of changes.
#ifndef TIMING
#define TIMING false
#endif

// Allowing hazards gives better performance but causes compute-sanitizer to report errors.
#ifndef HAZARDS
#define HAZARDS false
#endif

#define OVERLAP_BASE_TYPE uint16_t
#define OVERLAP_TYPE UncompressedSizeOverlap<OVERLAP_BASE_TYPE>

// LOOKAHEAD_GZIP_MIN_CTA_AMOUNT / LOOKAHEAD_GZIP_MIN_CHUNK_SIZE live in include/lookahead_gzip.h.

// Size of input ring buffer used for streaming mode [elements]
constexpr size_t INPUT_RING_BUFFER_SIZE = 1024 * 1024 * 4;

// Size of output ring buffer used for streaming mode [elements]
constexpr size_t OUTPUT_RING_BUFFER_SIZE = 1024 * 1024 * 8;

using copy_type = uint64_t;

constexpr uint HEADER_STRIDE_BYTES = 4;
constexpr uint THREADS_PER_BLOCK = O2O_VECTORS_PER_BLOCK * THREADS_PER_O2O_CREATION;
constexpr uint BLOCK_HEADER_THREADS_PER_BLOCK = 512;
static_assert(THREADS_PER_BLOCK % WARP_SIZE_U == 0);
static_assert(BLOCK_HEADER_THREADS_PER_BLOCK % WARP_SIZE_U == 0);

constexpr uint UNCOMPRESSED_WORDS_COUNT = 1024 * 1024;
static_assert(
  UNCOMPRESSED_WORDS_COUNT > MAX_LOOKBACK,
  "Not enough bytes to store the 32kB length of LZ77.  "
  "Increase UNCOMPRESSED_WORDS_COUNT."
);
static_assert(
  UNCOMPRESSED_WORDS_COUNT * 2 <= OUTPUT_RING_BUFFER_SIZE,
  "Not enough space in output ring to send all the "
  "uncompressed_words.  Increase OUTPUT_RING_BUFFER_SIZE or "
  "decrease UNCOMPRESSED_WORDS_COUNT."
);

constexpr uint UNCOMPRESSED_SHARED_WORDS_COUNT = 4096;

// These are derived from above.  Do not adjust them.
constexpr uint HEADER_STRIDE_BITS = HEADER_STRIDE_BYTES * 8;
