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

#include <ans/ans_arch_profile.cuh> // ANS_HIST_STAGE_BYTES_PER_WARP (hist_prefetch_depth_for)
#include <ans/ans_utils.cuh>

namespace ans_gpu_lib
{
namespace detail
{

// Warp-rows of LoadT the histogram keeps in flight, from the arch's per-warp staging
// budget: a wider load buys proportionally fewer rows for the same shared memory.
template <typename LoadT>
inline constexpr __host__ __device__ int hist_prefetch_depth_for()
{
  constexpr int row_bytes = WARP_SIZE * static_cast<int>(sizeof(LoadT));
  static_assert(
    ANS_HIST_STAGE_BYTES_PER_WARP >= row_bytes,
    "histogram staging budget must cover at least one warp-row of the widest load"
  );
  return ANS_HIST_STAGE_BYTES_PER_WARP / row_bytes;
}

// CHAR symbols are raw bytes. Extract one from a packed word as a zero-extended
// encoding-table index without a shift/mask chain.
template <int SYMBOL_IN_WORD>
inline __device__ uint32_t char_get_symbol_from_word(uint32_t word)
{
  static_assert(SYMBOL_IN_WORD >= 0 && SYMBOL_IN_WORD < 4, "a uint32 word contains four CHAR symbols");
  return __byte_perm(word, 0u, 0x4440u + SYMBOL_IN_WORD);
}

// FP16 / BF16 exponent rotate.
//
// Rotate each 16-bit value LEFT by one bit so the exponent is byte-aligned and the
// sign leaves the symbol byte:
//   bf16  s eeeeeeee mmmmmmm  ->  symbol = eeeeeeee     mantissa = mmmmmmm s
//   fp16  s eeeee mmmmmmmmmm  ->  symbol = eeeee mmm     mantissa = mmmmmmm s
// Same operation for both; bf16 is passed as NVCOMP_TYPE_FLOAT16.
//
// Applied to the whole 32-bit load word. For w = [v1 (high half) | v0 (low half)]:
//   byte 0 = mantissa(v0)   byte 1 = symbol(v0)
//   byte 2 = mantissa(v1)   byte 3 = symbol(v1)
//
// A 32-bit rotate swaps the two signs: mantissa byte i carries sign(i ^ 1).
// Producers and consumers treat symbols as ALIGNED PAIRS (2k, 2k+1) relative to
// the chunk start. A trailing unpaired symbol (odd chunk length) pairs with itself.
inline __device__ uint32_t rotate_fp_pair_left1(uint32_t w) { return __funnelshift_l(w, w, 1); }

// Scalar form for the tails: rotate one aligned pair. Pass `lo` twice for a trailing
// unpaired value, which then keeps its own sign.
inline __device__ uint32_t rotate_fp_pair_left1(uint16_t lo, uint16_t hi)
{
  return rotate_fp_pair_left1(static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16));
}

template <int SYMBOL_IN_PAIR>
inline __device__ uint32_t fp16_get_symbol_from_rotated(uint32_t rotated)
{
  static_assert(SYMBOL_IN_PAIR == 0 || SYMBOL_IN_PAIR == 1, "an FP16 pair contains two symbols");
  constexpr uint32_t BYTE_SELECTOR = SYMBOL_IN_PAIR == 0 ? 0x4441u : 0x4443u;
  return __byte_perm(rotated, 0u, BYTE_SELECTOR);
}

template <int SYMBOL_IN_PAIR>
inline __device__ uint8_t fp16_get_mantissa_from_rotated(uint32_t rotated)
{
  static_assert(SYMBOL_IN_PAIR == 0 || SYMBOL_IN_PAIR == 1, "an FP16 pair contains two mantissas");
  constexpr uint32_t BIT_SHIFT = SYMBOL_IN_PAIR == 0 ? 0 : 16;
  return static_cast<uint8_t>(rotated >> BIT_SHIFT);
}

inline __device__ uint32_t fp16_pack_symbols_from_rotated_pairs(uint32_t rotated0, uint32_t rotated1)
{
  return __byte_perm(rotated0, rotated1, 0x7531u);
}

inline __device__ uint32_t fp16_pack_mantissas_from_rotated_pairs(uint32_t rotated0, uint32_t rotated1)
{
  return __byte_perm(rotated0, rotated1, 0x6420u);
}

// FP32 exponent rotate.
//
//   fp32  s eeeeeeee mmm..m (23)  ->  symbol = eeeeeeee     mantissa = mmm..m s
//
// A value fills the whole load word, so the transform never crosses a value boundary:
// byte 3 is the symbol and bytes 0..2 are the mantissa.
inline NVCOMP_HOST_DEVICE_FUNCTION uint32_t rotate_fp32_left1(uint32_t word)
{
#ifdef __CUDA_ARCH__
  return __funnelshift_l(word, word, 1);
#else
  return (word << 1) | (word >> 31);
#endif
}

inline NVCOMP_HOST_DEVICE_FUNCTION uint32_t rotate_fp32_right1(uint32_t word)
{
#ifdef __CUDA_ARCH__
  return __funnelshift_r(word, word, 1);
#else
  return (word >> 1) | (word << 31);
#endif
}

inline NVCOMP_HOST_DEVICE_FUNCTION uint32_t fp32_get_symbol_from_rotated(uint32_t rotated) { return rotated >> 24; }

template <int MANTISSA_BYTE>
inline __device__ uint32_t fp32_get_mantissa_from_rotated(uint32_t rotated)
{
  static_assert(MANTISSA_BYTE >= 0 && MANTISSA_BYTE < 3, "an FP32 mantissa contains three bytes");
  constexpr uint32_t BIT_SHIFT = MANTISSA_BYTE * BITS_PER_BYTE;
  return static_cast<uint8_t>(rotated >> BIT_SHIFT);
}

inline NVCOMP_HOST_DEVICE_FUNCTION uint32_t fp32_exponent_symbol(uint32_t word)
{
  return fp32_get_symbol_from_rotated(rotate_fp32_left1(word));
}

// FP32 tile geometry and mantissa layout, shared by encode and decode.
//
// A tile is two 128-value groups. In each group lane t owns the four adjacent
// values at 4t, loaded as one uint4. Their 12 mantissa bytes are split into three
// equal segments, allowing one coalesced uint2 access per lane and segment.
//
// The trailing partial tile is stored as three bytes per value in value order.
struct FP32Tile
{

  static constexpr int MANTISSA_BYTES_PER_SYMBOL = static_cast<int>(ans_mantissa_bytes_per_symbol(AnsStreamType::Fp32));
  static constexpr int LANE_SYMBOLS_PER_GROUP = 4;
  static constexpr int GROUP_SYMBOLS = WARP_SIZE * LANE_SYMBOLS_PER_GROUP;
  static constexpr int GROUPS_PER_TILE = 2;
  struct alignas(sizeof(uint2)) Segment
  {
    uint32_t values[GROUPS_PER_TILE];
  };
  static constexpr int TILE_SYMBOLS = GROUPS_PER_TILE * GROUP_SYMBOLS;
  static constexpr int LANE_SYMBOLS_PER_TILE = GROUPS_PER_TILE * LANE_SYMBOLS_PER_GROUP;
  static constexpr int LANE_PAIRS_PER_GROUP = LANE_SYMBOLS_PER_GROUP / 2;
  static constexpr int LANE_PAIRS_PER_TILE = LANE_SYMBOLS_PER_TILE / 2;

  static constexpr int LANE_SEGMENTS_PER_GROUP = (LANE_SYMBOLS_PER_GROUP * MANTISSA_BYTES_PER_SYMBOL) /
                                                 static_cast<int>(sizeof(uint32_t));
  static constexpr int SEGMENT_BYTES = WARP_SIZE * static_cast<int>(sizeof(Segment));
  static constexpr int TILE_MANTISSA_BYTES = LANE_SEGMENTS_PER_GROUP * SEGMENT_BYTES;

  static_assert(GROUPS_PER_TILE == 2, "FP32 stream geometry fixes two groups per tile");
  static_assert(LANE_SEGMENTS_PER_GROUP == 3, "four FP32 values carry three uint32 mantissa segments");
  static_assert(
    TILE_MANTISSA_BYTES == MANTISSA_BYTES_PER_SYMBOL * TILE_SYMBOLS,
    "the segmented tile must contain exactly three mantissa bytes per FP32 value"
  );
  static_assert(LANE_SYMBOLS_PER_GROUP % 2 == 0, "the two-state split requires aligned symbol pairs");

  static __device__ int lane_segment_offset(int tile, int segment, int lane_id)
  {
    return tile * TILE_MANTISSA_BYTES + segment * SEGMENT_BYTES + static_cast<int>(sizeof(Segment)) * lane_id;
  }

  static __device__ int lane_value_offset(int group, int lane_id)
  {
    return group * GROUP_SYMBOLS + LANE_SYMBOLS_PER_GROUP * lane_id;
  }

  static __device__ int lane_pair_value_offset(int pair, int lane_id)
  {
    const int group = pair / LANE_PAIRS_PER_GROUP;
    const int pair_in_group = pair - group * LANE_PAIRS_PER_GROUP;
    return lane_value_offset(group, lane_id) + 2 * pair_in_group;
  }
};

// FP8 (E4M3) pair packing.
//
// E4M3 byte: [sign:1 | exp:4 | mantissa:3]. Two adjacent values form one ANS symbol:
// packed_exp = the two 4-bit exponents, packed_signs_mantissas = (sign << 3) | mantissa
// for each. Even value in bits 0..3, odd in bits 4..7.
inline __device__ uint8_t fp8_exp_nibble(uint8_t b) { return (b >> 3) & 0x0F; }

inline __device__ uint8_t fp8_signs_mantissas_nibble(uint8_t b) { return ((b & 0x80) >> 4) | (b & 0x07); }

// Scalar form, used by the tails.
inline __device__ void split_fp8_pair(uint8_t b0, uint8_t b1, uint8_t &packed_exp, uint8_t &packed_signs_mantissas)
{
  packed_exp = static_cast<uint8_t>(fp8_exp_nibble(b0) | (fp8_exp_nibble(b1) << 4));
  packed_signs_mantissas = static_cast<uint8_t>(fp8_signs_mantissas_nibble(b0) | (fp8_signs_mantissas_nibble(b1) << 4));
}

// Adjacent 4-bit fields (bits 0..3 of each byte) packed into the low 16 bits.
inline __device__ uint32_t pack_adjacent_4bit_pairs_lo16(uint32_t low4_per_byte)
{
  const uint32_t packed = low4_per_byte | (low4_per_byte >> 4);
  return __byte_perm(packed, 0u, 0x4420u); // bytes {0,2} -> {0,1}
}

// Histogram counts packed_exp only; split_fp8_quad_e4m3 reuses this.
inline __device__ uint32_t fp8_quad_packed_exp_e4m3(uint32_t w)
{
  return pack_adjacent_4bit_pairs_lo16((w >> 3) & 0x0F0F0F0Fu);
}

// Vectorized split of 4 FP8 bytes (2 pairs). packed_exp_lo16 / packed_signs_mantissas_lo16
// each hold pair0 in byte 0 and pair1 in byte 1.
inline __device__ void split_fp8_quad_e4m3(uint32_t w, uint32_t &packed_exp_lo16, uint32_t &packed_signs_mantissas_lo16)
{
  packed_exp_lo16 = fp8_quad_packed_exp_e4m3(w);
  const uint32_t signs_mantissas4 = ((w & 0x80808080u) >> 4) | (w & 0x07070707u);
  packed_signs_mantissas_lo16 = pack_adjacent_4bit_pairs_lo16(signs_mantissas4);
}

inline __device__ void split_fp8_eight_e4m3(uint2 raw, uint32_t &packed_exp, uint32_t &packed_signs_mantissas)
{
  uint32_t exp_lo;
  uint32_t signs_mantissas_lo;
  uint32_t exp_hi;
  uint32_t signs_mantissas_hi;
  split_fp8_quad_e4m3(raw.x, exp_lo, signs_mantissas_lo);
  split_fp8_quad_e4m3(raw.y, exp_hi, signs_mantissas_hi);
  packed_exp = exp_lo | (exp_hi << (2 * BITS_PER_BYTE));
  packed_signs_mantissas = signs_mantissas_lo | (signs_mantissas_hi << (2 * BITS_PER_BYTE));
}

// Compress-side policies live in EncodePolicy.hpp.

// Per-chunk, per-CTA decoding table, built in-kernel by construct_decoding_table and read
// on the decode side. 4 bytes per slot:
//   bits  0..7  : sym
//   bits  8..19 : pdf
//   bits 20..31 : smcdf
// Single LDS.U32 + smart unpack on the decode side (sym = byte 0 consumed by
// __byte_perm; pdf = (entry >> 8) & 0xFFF; smcdf = entry >> 20 with no mask).
// Live across both the build and decode phases, so it stays out of the
// build/decode union below.
template <uint32_t TABLELOG>
struct DecodeTable
{
  uint32_t slots[1u << TABLELOG];
};

using TableBuildScan = nvcomp::cub::BlockScan<uint32_t, NUM_DECOMP_THREADS_PER_CTA>;

// Per-CTA scratch used only during the table-build phase. Overlaps with the
// decode buffers via the SetupAndDecodeSmem union (the two phases are never live
// at the same time).
struct TableBuildScratch
{
  uint2 pdfs_and_cdfs[NV_SYMBOL_COUNT];
  typename TableBuildScan::TempStorage scan_smem;
};

// One warp's decode region: its renorm prefetch window, then the refill state the window
// is refilled from. Adjacent so the decoder reaches the state at a constant offset from
// the window base it already holds, and so a read one u16 past the window stays inside
// the slice (the value is discarded; that lane is not renormalizing).
struct __align__(sizeof(uint4)) WarpDecodeSlice
{
  uint16_t renorm_buf[ANS_RENORM_BUF_UINT16];
  WarpRefillState refill;
};
static_assert(
  sizeof(WarpDecodeSlice) == ANS_RENORM_BUF_UINT16 * sizeof(uint16_t) + sizeof(WarpRefillState),
  "symbol_decoder reaches refill at RENORM_BUF_BYTES from the window base, so the slice must not pad"
);

// Per-CTA decode-phase buffers. Mantissas are a plain LDG; the per-warp buffer is the
// renorm prefetch window.
struct DecodeBuffers
{
  WarpDecodeSlice warp[NUM_DECOMP_WARPS_PER_CTA];
};

// The table-build scratch and the decode buffers are never live at the same
// time: construct_decoding_table finishes (writing only the separate `table`)
// and a __syncthreads separates it from the decode phase, which is the first
// thing to touch renorm_buf. Overlap them in a union so the build scratch costs
// no shared memory on top of the (larger) decode buffers.
union SetupAndDecodeSmem
{
  TableBuildScratch build;
  DecodeBuffers decode;
};

} // namespace detail
} // namespace ans_gpu_lib
