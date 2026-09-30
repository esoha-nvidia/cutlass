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

#include <cstddef>
#include <cstdint>

#include "CudaConstants.h"

namespace ans_gpu_lib
{

// Type for in-chunk sizes, symbol counts and byte offsets in device code. A chunk
// is capped at nvcompANSCompressionMaxAllowedChunkSize (1 << 24) on compress and
// nvcompANSDecompressionMaxAllowedChunkSize (1 << 25) on decompress, so every
// intra-chunk quantity fits in 32 bits with room to spare. Values that cross the
// host API boundary stay size_t.
using IndexT = uint32_t;

// Note:
// symbols are collected on the byte level [0, ..., 255]
constexpr uint32_t NV_MAX_SYMBOL_VALUE = 255;
constexpr uint32_t NV_SYMBOL_COUNT = NV_MAX_SYMBOL_VALUE + 1;
constexpr uint32_t MIN_TABLELOG = 5;
constexpr uint32_t MAX_TABLELOG = 11;

// Host API ANS frequency-table precision, independently tunable by stream type.
// These defaults balance model precision against decoding-table footprint.
// Raising the maximum configured value can require retuning the decode occupancy
// budgets in ans_arch_profile.cuh.
constexpr uint32_t CHAR_TABLELOG = 11;
constexpr uint32_t FP16_TABLELOG = 10;
constexpr uint32_t FP8_TABLELOG = 11;
constexpr uint32_t FP32_TABLELOG = 10;

// Lowest tablelog the modes above ship. 8 is the hard floor (2^8 == NV_SYMBOL_COUNT), but
// normalization needs more slack than that.
constexpr uint32_t MIN_MODE_TABLELOG = 10;
constexpr bool is_valid_mode_tablelog(uint32_t tablelog)
{
  return tablelog >= MIN_MODE_TABLELOG && tablelog <= MAX_TABLELOG;
}
static_assert(
  is_valid_mode_tablelog(CHAR_TABLELOG) && is_valid_mode_tablelog(FP16_TABLELOG) &&
    is_valid_mode_tablelog(FP8_TABLELOG) && is_valid_mode_tablelog(FP32_TABLELOG),
  "host ANS mode tablelogs must be in [MIN_MODE_TABLELOG, MAX_TABLELOG]"
);

// nvcompDx has a separate stream format and remains fixed at its existing value.
constexpr uint32_t DX_DEFAULT_TABLELOG = 10;
constexpr uint32_t MIN_SUB_CHUNKS_PER_CHUNK = 4;
constexpr uint32_t MAX_SUB_CHUNKS_PER_CHUNK = 64;
constexpr uint32_t DEFAULT_SUB_CHUNKS_PER_CHUNK = 8;
static_assert(
  DEFAULT_SUB_CHUNKS_PER_CHUNK >= MIN_SUB_CHUNKS_PER_CHUNK && DEFAULT_SUB_CHUNKS_PER_CHUNK <= MAX_SUB_CHUNKS_PER_CHUNK,
  "DEFAULT_SUB_CHUNKS_PER_CHUNK must be in [MIN, MAX]"
);
constexpr uint32_t MIN_SUB_CHUNK_SIZE = 2048;

// 0 in the public compress max_sub_chunk_count option means this default.
inline constexpr uint8_t resolve_max_sub_chunk_count(uint8_t count)
{
  return count == 0 ? static_cast<uint8_t>(DEFAULT_SUB_CHUNKS_PER_CHUNK) : count;
}

inline constexpr uint8_t resolve_decomp_launch_sub_chunk_count(uint8_t count)
{
  return count == 0 ? static_cast<uint8_t>(DEFAULT_SUB_CHUNKS_PER_CHUNK) : count;
}

constexpr uint32_t DEFAULT_DEVICE_SUB_CHUNK_SIZE = 4096;
constexpr uint32_t NUM_COMP_WARPS_PER_CTA = 8;
constexpr uint32_t NUM_COMP_THREADS_PER_CTA = NUM_COMP_WARPS_PER_CTA * WARP_SIZE;
constexpr uint32_t NUM_DECOMP_WARPS_PER_CTA = 8;
constexpr uint32_t NUM_DECOMP_THREADS_PER_CTA = NUM_DECOMP_WARPS_PER_CTA * WARP_SIZE;

// uint4s each thread holds in registers per defrag barrier. Trades registers (live only
// in the defrag epilogue) for memory-level parallelism and a proportionally lower barrier
// count; 1 reproduces the one-copy-per-barrier behavior. Measured on silesia/B200: the
// step from 1 to 2 is worth ~0.4%, 3 and 4 are flat, and 6 costs ~2% as the epilogue's
// registers start bounding the encode loop's occupancy.
constexpr uint32_t ANS_DEFRAG_UINT4S_PER_THREAD = 2;

// Longest slot run a symbol's owning lane fills by itself during the decode-table
constexpr uint32_t ANS_TABLE_SELF_FILL_MAX = 16;

// Host knows the stream has at most one sub-chunk per decomp warp: launch one CTA
// and compile the per-warp decode as a single pass (no sub-chunk stride loop).
// 0 on decompress means "whatever compression used", so the count is unknown.
inline constexpr bool decomp_one_subchunk_per_warp(uint8_t max_sub_chunk_count)
{
  return max_sub_chunk_count != 0 && max_sub_chunk_count <= NUM_DECOMP_WARPS_PER_CTA;
}

constexpr uint32_t SYMBOL_TABLE_SIZE = 1024;
constexpr uint32_t CDF_TABLE_SIZE = 514;
// Shared encoding table size (shared across the CTA, irrespective of warp count)
constexpr uint32_t SHARED_ENCODING_TABLE_SIZE = 3072;
// nvCOMPDx FP16 per-warp staging buffer (WARP_SIZE * sizeof(uint2)).
constexpr uint32_t SHARED_STAGING_BUFFER_SIZE_PER_WARP = 256;
// Shared decoding buffer size (128 * sizeof(uint16_t))
constexpr uint32_t SHARED_DECODING_BUFFER_SIZE_PER_WARP = 256;
constexpr uint32_t NUM_ENCODES_PER_LOOP = 64;
constexpr uint32_t NUM_ENCODES_PER_THREAD = 2;
constexpr uint32_t BITS_PER_BYTE = 8;
constexpr uint32_t RENORM_SIZE = 2;
constexpr uint32_t TAIL_U16_PER_STATE = 2;

// Below this many symbols a chunk is histogrammed exactly, whatever the sampling shift.
constexpr uint32_t MIN_SAMPLED_HIST_SYMBOLS = 4096;

// Sampled-histogram SPAN dilation, shared by every HistFloor::ObservedBand policy. The
// shift itself is the public histogram_reduction_log2 option (0 = exact). The sampled counts
// give an observed symbol range [min_obs, max_obs]; the contiguous band
//   [min_obs - SAMPLED_HIST_DILATE_NEG, max_obs + SAMPLED_HIST_DILATE_POS]
// (clamped to [0, 255]) is floored to count >= 1, so a symbol just outside the sample still
// has a table slot. Independent radii match the usual BF16 weight asymmetry (long tail
// toward smaller exponents, short high side). Detect + redo still guarantees correctness
// for any symbol outside the band; either radius may be 0 to disable that side.
//
// The radius is in symbols, so its reach depends on the mode's symbol alphabet: BF16 and
// FP32 both code the full 8-bit exponent, making a radius one exponent, while fp16 codes
// eeeee plus 3 mantissa bits, making it an eighth of one. These values were tuned on BF16
// weights, so FP32 inherits them unchanged: the two alphabets are identical, and a BF16
// tensor and the FP32 values it was narrowed from have the same exponent distribution.
constexpr uint32_t SAMPLED_HIST_DILATE_NEG = 15;
constexpr uint32_t SAMPLED_HIST_DILATE_POS = 15;

} // namespace ans_gpu_lib
