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

#include <type_traits>

#include <ans/symbol_decoder.cuh>
#include <ans/types.cuh>

namespace ans_gpu_lib
{
namespace detail
{

// Aggregate per-lane decode errors across the warp (lane 0 writes). Does not
// store Success: BOUNDS_CHECK pre-clears the status array before launch.
inline __device__ void write_decode_status(nvcompStatus_t err, nvcompStatus_t *status)
{
  const bool found_err = __ballot_sync(WARP_ALL, err != nvcompSuccess);
  if (get_lane_id() == 0 && found_err)
  {
    *status = nvcompErrorCannotDecompress;
  }
}

// Worst-case warp uint16s to decode symbols_per_lane symbols per lane, spread over
// states_per_lane rANS states. Each state emits one symbol per iteration and renormalizes
// at most tablelog bits; the 16-bit rounding is per state, so the same symbol count
// costs more when split across two of them. Every policy below reaches this through
// RenormRefillThreshold, passing whatever unit it refills before.
constexpr uint32_t
ans_worst_case_renorm_consumption(uint32_t symbols_per_lane, uint32_t states_per_lane, uint32_t tablelog)
{
  return WARP_SIZE_U * states_per_lane * nvcomp::roundUpDiv(tablelog * (symbols_per_lane / states_per_lane), 16u);
}

// advance_bitstream_pos re-anchors to the top of the buffer but keeps the unconsumed
// alignment leftover live there, so this many uint16 are unavailable after a refill.
constexpr uint32_t ANS_RENORM_BUF_ALIGN_SLACK_U16 = sizeof(uint4) / sizeof(uint16_t) - 1u;

// The invariant RenormRefillThreshold asserts: refill_if_needed only guarantees `threshold`
// words remain when it does not fire, so the buffer must still cover a full threshold on
// the iteration where it does.
constexpr bool ans_renorm_buf_covers(uint32_t threshold_u16, uint32_t buf_u16)
{
  return threshold_u16 + ANS_RENORM_BUF_ALIGN_SLACK_U16 <= buf_u16;
}

// Every policy's REFILL_THRESH_U16 is its consume unit costed by the two functions above,
// so they take it from here to assert the invariant once. SYMBOLS_PER_LANE is the unit each
// policy refills before, which its own comment at the use site names.
template <uint32_t SYMBOLS_PER_LANE, uint32_t STATES_PER_LANE, uint32_t TABLELOG>
struct RenormRefillThreshold
{
  static constexpr uint32_t value = ans_worst_case_renorm_consumption(SYMBOLS_PER_LANE, STATES_PER_LANE, TABLELOG);
  static_assert(
    ans_renorm_buf_covers(value, ANS_RENORM_BUF_UINT16),
    "a decode policy's consume unit exceeds its renorm buffer"
  );
};

template <typename Impl, AnsStreamType StreamType>
struct DecodePolicyBase
{
  static constexpr AnsStreamType STREAM_TYPE = StreamType;
  static constexpr uint32_t TABLELOG = ans_tablelog(STREAM_TYPE);
  static constexpr uint32_t REFILL_THRESH_U16 =
    RenormRefillThreshold<Impl::REFILL_SYMBOLS_PER_LANE, Impl::STATES_PER_LANE, TABLELOG>::value;
  static_assert(Impl::STATES_PER_LANE == 1 || Impl::STATES_PER_LANE == 2);

  template <bool BOUNDS_CHECK>
  using Decoder = std::conditional_t<
    Impl::STATES_PER_LANE == 1,
    symbol_decoder<BOUNDS_CHECK, ANS_RENORM_BUF_UINT16, REFILL_THRESH_U16, TABLELOG>,
    symbol_decoder_dual_state<BOUNDS_CHECK, ANS_RENORM_BUF_UINT16, REFILL_THRESH_U16, TABLELOG>>;
};

// ===========================================================================
// Decode policies. Char, fp16, fp8, and fp32 each share one policy across x1/x2, parameterized
// by an Impl (tile map, remainder, state count). An Impl repeats its
// geometry instead of sharing a helper: the constants are the contract with its encode mirror
// (CharX2Impl <-> CharX2EncodeImpl), so a pair is tuned as a unit, and ROWS_PER_TILE equalling
// STATES_PER_LANE is current tuning rather than a rule.

// Two-state block-interleave. Mirror of CharX2EncodeImpl: the dual-state decode_pair
// advances one state each, so a row's bytes alternate between the two.
struct CharX2Impl
{
  static constexpr int LANE_SYMBOLS_PER_ROW = 8;
  static constexpr int ROW_SYMBOLS = WARP_SIZE * LANE_SYMBOLS_PER_ROW;
  static constexpr int ROWS_PER_TILE = 2;
  static constexpr int TILE_SYMBOLS = ROWS_PER_TILE * ROW_SYMBOLS;
  static constexpr uint32_t STATES_PER_LANE = 2u;
  // decode_row refills before every row, so a row is the consume unit.
  static constexpr uint32_t REFILL_SYMBOLS_PER_LANE = LANE_SYMBOLS_PER_ROW;
};

// Single-state. Mirror of CharX1EncodeImpl: one-row (256-symbol) tiles.
struct CharX1Impl
{
  static constexpr int LANE_SYMBOLS_PER_ROW = 8;
  static constexpr int ROW_SYMBOLS = WARP_SIZE * LANE_SYMBOLS_PER_ROW;
  static constexpr int ROWS_PER_TILE = 1;
  static constexpr int TILE_SYMBOLS = ROWS_PER_TILE * ROW_SYMBOLS;
  static constexpr uint32_t STATES_PER_LANE = 1u;
  // decode_row refills before every row, and an x1 tile is one row.
  static constexpr uint32_t REFILL_SYMBOLS_PER_LANE = LANE_SYMBOLS_PER_ROW;
};

// Shared char decode (x1 and x2). 8-bit symbols: the decoded symbol is the output byte,
// and lane t owns 8 contiguous symbols per row. Impl owns the tile height and state count.
template <typename Impl>
struct CharDecodePolicy : DecodePolicyBase<Impl, AnsStreamType::Char>
{
  using out_t = uint8_t;

  static constexpr int LANE_SYMBOLS_PER_ROW = Impl::LANE_SYMBOLS_PER_ROW;
  static constexpr int ROW_SYMBOLS = Impl::ROW_SYMBOLS;
  static constexpr int ROWS_PER_TILE = Impl::ROWS_PER_TILE;
  static constexpr int TILE_SYMBOLS = Impl::TILE_SYMBOLS;

  template <bool BOUNDS_CHECK, typename SymbolDecoderT>
  static __device__ void
  decode_body(SymbolDecoderT &sd, void *uncomp_sub_chunk, const uint8_t *sub_chunk_mantissas, int num_decodes);

  static __device__ int get_lane_row_ix(int row, int lane_id)
  {
    return row * ROW_SYMBOLS + LANE_SYMBOLS_PER_ROW * lane_id;
  }

  // A row is the renorm consume unit, so each row refills itself. decode_pair gathers its
  // symbol bytes into the low bytes of one word (pack_pair_entries plus the SECOND
  // gather), so two pairs fill a word and the row lands as a single STG.64.
  template <typename SymbolDecoderT>
  static __device__ void decode_row(SymbolDecoderT &sd, out_t *out, int row)
  {
    sd.refill_if_needed(true);
    uint32_t w0, w1;
    sd.template decode_pair<false /*no fuse*/, false /*second symbol*/>(w0);
    sd.template decode_pair<false /*no fuse*/, true /*second symbol*/>(w0);
    sd.template decode_pair<false /*no fuse*/, false /*second symbol*/>(w1);
    sd.template decode_pair<false /*no fuse*/, true /*second symbol*/>(w1);
    *reinterpret_cast<uint2 *>(out + get_lane_row_ix(row, get_lane_id())) = make_uint2(w0, w1);
  }

  // 64 symbols (32 lanes x 1 pair); lane t owns the adjacent pair at window_base + 2t, the
  // same map for both geometries. Partial is the last iter (window_n < 64). Each window
  // refills itself: an x2 remainder runs to TILE_SYMBOLS-1, twice the per-row threshold.
  //
  // An unpaired decode_pair leaves its two symbols in bytes 2 and 3 (the SECOND gather
  // that moves them down is only issued for the paired row calls), so one shift brings
  // them to bytes 0/1 for the uint16 store. ix is even and out is 16 B-aligned.
  template <bool Partial, typename SymbolDecoderT>
  static __device__ void remainder_iter(SymbolDecoderT &sd, out_t *out, int window_base, int window_n)
  {
    sd.refill_if_needed(true);
    const int pair_ix = 2 * get_lane_id();
    const int ix = window_base + pair_ix;
    if constexpr (Partial)
    {
      const bool has_a = pair_ix < window_n;
      const bool has_b = (pair_ix + 1) < window_n;
      uint32_t a, b;
      sd.template decode_pair<false, false, true /*partial*/>(a, 0u, has_a, has_b, &b);
      if (has_a)
      {
        out[ix] = static_cast<out_t>(a);
      }
      if (has_b)
      {
        out[ix + 1] = static_cast<out_t>(b);
      }
    }
    else
    {
      uint32_t w;
      sd.template decode_pair<false, false>(w);
      *reinterpret_cast<uint16_t *>(out + ix) = static_cast<uint16_t>(w >> 16);
    }
  }
};

// Per-warp CHAR decode: full tiles walked as linear rows (the encoder's ROWS_PER_TILE does
// not change the output layout), then the leftover as 64-symbol remainder windows.
template <typename Impl>
template <bool BOUNDS_CHECK, typename SymbolDecoderT>
inline __device__ void CharDecodePolicy<Impl>::decode_body(
  SymbolDecoderT &sd,
  void *uncomp_sub_chunk,
  const uint8_t * /*sub_chunk_mantissas*/,
  int num_decodes
)
{
  out_t *out = reinterpret_cast<out_t *>(uncomp_sub_chunk);

  // Pre-loop: initial full renorm fill (the renorm buffer is a continuous cursor
  // across iters, only re-anchored + refetched on refill_if_needed, so most iters
  // issue no renorm LDGSTS).
  sd.wait_initial_prefetch();

  // Row countdown: the bound is a constant, and on exit row is the rows consumed.
  int rows_left = (num_decodes / TILE_SYMBOLS) * ROWS_PER_TILE;
  int row = 0;
#pragma unroll CHAR_DECODE_ROWS_PER_META
  for (; rows_left > 0; --rows_left, ++row)
  {
    decode_row(sd, out, row);
  }

  // Leftover 1..TILE_SYMBOLS-1 symbols as 64-symbol windows, the last possibly short.
  const int rem_base = (row / ROWS_PER_TILE) * TILE_SYMBOLS;
  const int rem = num_decodes - rem_base;
  if (rem != 0)
  {
    // Each remainder iteration decodes two symbols at a time.
    constexpr int STEP = WARP_SIZE * 2;
    const int nfull = rem / STEP;
    const int tail = rem - nfull * STEP;
    for (int j = 0; j < nfull; ++j)
    {
      remainder_iter<false>(sd, out, rem_base + j * STEP, STEP);
    }
    if (tail != 0)
    {
      remainder_iter<true>(sd, out, rem_base + nfull * STEP, tail);
    }
  }
}

// Shared fp16 decode (x1 and x2). Impl owns the tile/remainder map and state count.
template <typename Impl>
struct FP16DecodePolicy : DecodePolicyBase<Impl, AnsStreamType::Fp16>
{
  static constexpr int TILE_SYMBOLS = Impl::TILE_SYMBOLS;
  static constexpr int ROWS_PER_TILE = Impl::ROWS_PER_TILE;

  static __device__ int get_lane_row_ix(int row, int lane_id)
  {
    return row * Impl::ROW_SYMBOLS + Impl::SYMBOLS_PER_LANE * lane_id;
  }

  static __device__ uint2 load_mantissas(const uint8_t *mantissas, int row)
  {
    return *reinterpret_cast<const uint2 *>(mantissas + get_lane_row_ix(row, get_lane_id()));
  }

  // Undo the encoder's 1-bit left pair rotate (rotate_fp_pair_left1) for 4 decoded fp16
  // symbols and their 4 mantissa bytes. packed_exp holds the 4 symbol bytes (symbol j in
  // byte j); mantissas_four holds the 4 mantissa bytes, and symbol 0 must be at an EVEN
  // sub-chunk index. Each byte_perm reassembles one ALIGNED PAIR in exactly the layout the
  // rotate produced -- mantissa in the even bytes, symbol in the odd ones -- so one rotate
  // right per pair undoes it.
  static __device__ uint2 reconstruct_four(uint32_t packed_exp, uint32_t mantissas_four)
  {
    const uint32_t pair0 = __byte_perm(mantissas_four, packed_exp, 0x5140u); // {s0, e0, s1, e1}
    const uint32_t pair1 = __byte_perm(mantissas_four, packed_exp, 0x7362u); // {s2, e2, s3, e3}
    uint2 result;
    result.x = __funnelshift_r(pair0, pair0, 1);
    result.y = __funnelshift_r(pair1, pair1, 1);
    return result;
  }

  // Scalar (remainder) form: recover one aligned pair from its two decoded symbols and two
  // mantissa bytes. Pass mantissa_a twice for a trailing unpaired symbol, which then
  // carries its own sign -- the same rule the histogram's scalar tail applies.
  static NVCOMP_HOST_DEVICE_FUNCTION void reconstruct_pair(
    uint32_t sym_a,
    uint32_t sym_b,
    uint32_t mantissa_a,
    uint32_t mantissa_b,
    uint16_t &out_a,
    uint16_t &out_b
  )
  {
    const uint32_t packed = mantissa_a | (sym_a << 8) | (mantissa_b << 16) | (sym_b << 24);
#ifdef __CUDA_ARCH__
    const uint32_t both = __funnelshift_r(packed, packed, 1);
#else
    const uint32_t both = (packed >> 1) | (packed << 31);
#endif // __CUDA_ARCH__
    out_a = static_cast<uint16_t>(both & 0xFFFFu);
    out_b = static_cast<uint16_t>(both >> 16);
  }

  // Reconstruct 8 fp16 values (two reconstruct_four) and STG.128 at out16+base.
  static __device__ void store_eight(uint16_t *out16, int base, uint32_t ea, uint32_t eb, uint2 sb)
  {
    const uint2 first = reconstruct_four(ea, sb.x);
    const uint2 next = reconstruct_four(eb, sb.y);
    uint4 v;
    v.x = first.x;
    v.y = first.y;
    v.z = next.x;
    v.w = next.y;
    *reinterpret_cast<uint4 *>(out16 + base) = v;
  }

  // A row is the renorm consume unit, so each row refills itself. Callers hoist the
  // mantissa load above the call, keeping that LDG in flight across the refill.
  template <typename SymbolDecoderT>
  static __device__ void decode_packed_row(SymbolDecoderT &sd, uint16_t *out16, uint2 sb, int row)
  {
    sd.refill_if_needed(true);
    uint32_t out0, out1;
    sd.template decode_pair<false /*pack mantissas*/, false /*second symbol*/>(out0);
    sd.template decode_pair<false /*pack mantissas*/, true /*second symbol*/>(out0);
    sd.template decode_pair<false /*pack mantissas*/, false /*second symbol*/>(out1);
    sd.template decode_pair<false /*pack mantissas*/, true /*second symbol*/>(out1);
    store_eight(out16, get_lane_row_ix(row, get_lane_id()), out0, out1, sb);
  }

  template <typename SymbolDecoderT>
  static __device__ void decode_fused_row(SymbolDecoderT &sd, uint16_t *out16, uint2 sb, int row)
  {
    sd.refill_if_needed(true);
    uint32_t p0, p1, p2, p3;
    sd.template decode_pair<true /*fuse mantissas*/, false /*second symbol*/>(p0, sb.x);
    sd.template decode_pair<true /*fuse mantissas*/, true /*second symbol*/>(p1, sb.x);
    sd.template decode_pair<true /*fuse mantissas*/, false /*second symbol*/>(p2, sb.y);
    sd.template decode_pair<true /*fuse mantissas*/, true /*second symbol*/>(p3, sb.y);
    uint4 v;
    v.x = p0;
    v.y = p1;
    v.z = p2;
    v.w = p3;
    *reinterpret_cast<uint4 *>(out16 + get_lane_row_ix(row, get_lane_id())) = v;
  }

  // 64-symbol remainder windows (32 lanes x 1 aligned pair). Full windows fuse the
  // pair into one uint32; the last short window (Partial) skips packing.
  template <bool Partial, typename SymbolDecoderT>
  static __device__ void remainder_iter(
    SymbolDecoderT &sd,
    uint16_t *out16,
    const uint8_t *mantissas,
    int window_base,
    [[maybe_unused]] int window_n
  )
  {
    const int pair_ix = 2 * get_lane_id();
    const int ix = window_base + pair_ix;
    if constexpr (Partial)
    {
      const bool has_a = pair_ix < window_n;
      const bool has_b = (pair_ix + 1) < window_n;
      sd.refill_if_needed(true);
      uint32_t a, b;
      sd.template decode_pair<false /*pack mantissas*/, false /*second symbol*/, true /*partial*/>(
        a,
        0u,
        has_a,
        has_b,
        &b
      );
      if (has_a)
      {
        const uint8_t sb_a = mantissas[ix];
        const uint8_t sb_b = has_b ? mantissas[ix + 1] : sb_a;
        uint16_t out_a, out_b;
        reconstruct_pair(
          static_cast<uint8_t>(a),
          has_b ? static_cast<uint8_t>(b) : static_cast<uint8_t>(a),
          sb_a,
          sb_b,
          out_a,
          out_b
        );
        out16[ix] = out_a;
        if (has_b)
        {
          out16[ix + 1] = out_b;
        }
      }
    }
    else
    {
      const uint32_t sb = static_cast<uint32_t>(mantissas[ix]) | (static_cast<uint32_t>(mantissas[ix + 1]) << 8);
      sd.refill_if_needed(true);
      uint32_t both;
      sd.template decode_pair<true /*fuse mantissas*/, false /*second symbol*/>(both, sb);
      *reinterpret_cast<uint32_t *>(out16 + ix) = both;
    }
  }

  template <bool BOUNDS_CHECK, typename SymbolDecoderT>
  static __device__ void
  decode_body(SymbolDecoderT &sd, void *uncomp_sub_chunk, const uint8_t *sub_chunk_mantissas, int num_decodes);
};

struct FP32X2Impl
{
  static constexpr uint32_t STATES_PER_LANE = 2u;
  static constexpr uint32_t REFILL_SYMBOLS_PER_LANE = FP32Tile::LANE_SYMBOLS_PER_TILE;
};

struct FP32X1Impl
{
  static constexpr uint32_t STATES_PER_LANE = 1u;
  static constexpr uint32_t REFILL_SYMBOLS_PER_LANE = FP32Tile::LANE_SYMBOLS_PER_TILE;
};

template <typename Impl>
struct FP32DecodePolicy : DecodePolicyBase<Impl, AnsStreamType::Fp32>
{
  static constexpr int TILE_SYMBOLS = FP32Tile::TILE_SYMBOLS;

  struct TileMantissas
  {
    FP32Tile::Segment segments[FP32Tile::LANE_SEGMENTS_PER_GROUP];
  };

  static __device__ TileMantissas load_tile_mantissas(const uint8_t *mantissas, int tile, int lane_id)
  {
    TileMantissas result;
#pragma unroll
    for (int segment = 0; segment < FP32Tile::LANE_SEGMENTS_PER_GROUP; ++segment)
    {
      result.segments[segment] =
        *reinterpret_cast<const FP32Tile::Segment *>(mantissas + FP32Tile::lane_segment_offset(tile, segment, lane_id));
    }
    return result;
  }

  static NVCOMP_HOST_DEVICE_FUNCTION uint32_t
  reconstruct_one(uint32_t exponent, uint32_t mantissa0, uint32_t mantissa1, uint32_t mantissa2)
  {
    const uint32_t rotated = (exponent << 24) | (mantissa2 << 16) | (mantissa1 << 8) | mantissa0;
    return rotate_fp32_right1(rotated);
  }

  static __device__ uint4
  reconstruct_four(uint32_t packed_exponents, uint32_t mantissa0, uint32_t mantissa1, uint32_t mantissa2)
  {
    const uint32_t rotated0 = __byte_perm(mantissa0, packed_exponents, 0x4210u);
    const uint32_t rotated1 = __byte_perm(__byte_perm(mantissa0, mantissa1, 0x3543u), packed_exponents, 0x5210u);
    const uint32_t rotated2 = __byte_perm(__byte_perm(mantissa1, mantissa2, 0x2432u), packed_exponents, 0x6210u);
    const uint32_t rotated3 = __byte_perm(mantissa2, packed_exponents, 0x7321u);
    uint4 output;
    output.x = rotate_fp32_right1(rotated0);
    output.y = rotate_fp32_right1(rotated1);
    output.z = rotate_fp32_right1(rotated2);
    output.w = rotate_fp32_right1(rotated3);
    return output;
  }

  static __device__ uint32_t reconstruct_at(const uint8_t *mantissas, int value_offset, uint32_t table_entry)
  {
    const uint8_t *const value_mantissa = mantissas + FP32Tile::MANTISSA_BYTES_PER_SYMBOL * value_offset;
    return reconstruct_one(table_entry & 0xFFu, value_mantissa[0], value_mantissa[1], value_mantissa[2]);
  }

  template <bool BOUNDS_CHECK, typename SymbolDecoderT>
  static __device__ void
  decode_body(SymbolDecoderT &sd, void *uncomp_sub_chunk, const uint8_t *sub_chunk_mantissas, int num_decodes);
};

// Shared fp8 (E4M3) decode (x1 and x2), the mirror of FP8EncodePolicy and structured like
// FP16DecodePolicy: Impl owns the rANS state layout (state count and rows per tile) and
// everything else is shared.
//
// Each ANS symbol is a packed_exp byte; the matching mantissa byte is
// packed_signs_mantissas. One symbol reconstructs TWO FP8 output bytes (a uint16).
// For each of the two FP8 values in the pair, packed_exp holds a 4-bit exponent
// nibble and packed_signs_mantissas holds a nibble = (sign << 3) | mantissa(3 bits).
template <typename Impl>
struct FP8DecodePolicy : DecodePolicyBase<Impl, AnsStreamType::Fp8>
{
  static constexpr int TILE_SYMBOLS = Impl::TILE_SYMBOLS;
  static constexpr int ROWS_PER_TILE = Impl::ROWS_PER_TILE;
  static constexpr int STORE_SYMBOLS = 4;
  static constexpr int HALF_SYMBOLS = WARP_SIZE * STORE_SYMBOLS; // 128
  static_assert(Impl::LANE_SYMBOLS_PER_ROW == STORE_SYMBOLS * 2);
  static_assert(Impl::ROW_SYMBOLS == 2 * HALF_SYMBOLS);

  // Lane t's 4-symbol half h of a row: coalesced 8 B at half_base + 4t.
  static __device__ int get_lane_half_ix(int row, int half, int lane_id)
  {
    return row * Impl::ROW_SYMBOLS + half * HALF_SYMBOLS + STORE_SYMBOLS * lane_id;
  }

  // Two coalesced LDG.32 (4 mantissa bytes per half), packed as uint2 for the 4-row hoist.
  static __device__ uint2 load_mantissas(const uint8_t *mantissas, int row)
  {
    const int lane_id = get_lane_id();
    uint2 sm;
    sm.x = *reinterpret_cast<const uint32_t *>(mantissas + get_lane_half_ix(row, 0, lane_id));
    sm.y = *reinterpret_cast<const uint32_t *>(mantissas + get_lane_half_ix(row, 1, lane_id));
    return sm;
  }

  // Reconstruct one FP8 (E4M3) output byte from its exponent nibble (4 bits) and its
  // sign+mantissa nibble (sign in bit 3, mantissa in bits 0..2):
  //   output = sign(bit 7) | exponent(bits 6..3) | mantissa(bits 2..0)
  static NVCOMP_HOST_DEVICE_FUNCTION uint32_t reconstruct_byte(uint32_t exp_nibble, uint32_t signs_mantissas_nibble)
  {
    const uint32_t sign = (signs_mantissas_nibble & 0x8) << 4; // nibble bit 3 -> byte bit 7
    const uint32_t exponent = exp_nibble << 3; // -> byte bits 6..3
    const uint32_t mantissa = signs_mantissas_nibble & 0x7; // -> byte bits 2..0
    return sign | exponent | mantissa;
  }

  // Reconstruct the two FP8 bytes of one symbol (a packed pair) and pack them into
  // a uint16 (first FP8 byte in the low byte, second in the high byte). Scalar form,
  // used by the remainder tail.
  static NVCOMP_HOST_DEVICE_FUNCTION uint16_t reconstruct_pair(uint8_t packed_exp, uint8_t packed_signs_mantissas)
  {
    const uint32_t byte_lo = reconstruct_byte(packed_exp & 0xF, packed_signs_mantissas & 0xF);
    const uint32_t byte_hi = reconstruct_byte((packed_exp >> 4) & 0xF, (packed_signs_mantissas >> 4) & 0xF);
    return static_cast<uint16_t>(byte_lo | (byte_hi << 8));
  }

  // Reconstruct 4 FP8 output bytes from one SECOND=false decode_pair. That word is
  // {pdf1[11:0], s0, s1}; the matching two mantissa bytes are in lanes 0 and 1.
  static __device__ uint32_t reconstruct_four(uint32_t entry_pair, uint32_t packed_signs_mantissas)
  {
    // Only byte lanes 0 and 1 carry symbols here, so the upper half of reconstruct_eight
    // is dead and its PRMT folds away; the SIMD work over all 4 lanes costs the same.
    const uint32_t packed_exp = __byte_perm(entry_pair, 0u, 0x4432u);
    return reconstruct_eight(packed_exp, packed_signs_mantissas).x;
  }

  // Reconstruct 8 FP8 output bytes from 4 packed symbols (4 packed pairs) at once.
  // One SIMD pass over all 4 byte lanes, then two PRMTs to interleave lo/hi.
  static __device__ uint2 reconstruct_eight(uint32_t packed_exp, uint32_t packed_signs_mantissas)
  {
    const uint32_t sign_mantissa_lo = select_bits<0x80808080u>(packed_signs_mantissas << 4, packed_signs_mantissas);
    const uint32_t sign_mantissa_hi = select_bits<0x80808080u>(packed_signs_mantissas, packed_signs_mantissas >> 4);
    const uint32_t out_lo = select_bits<0x78787878u>(packed_exp << 3, sign_mantissa_lo);
    const uint32_t out_hi = select_bits<0x78787878u>(packed_exp >> 1, sign_mantissa_hi);
    uint2 result;
    result.x = __byte_perm(out_lo, out_hi, 0x5140u); // {lo0, hi0, lo1, hi1}
    result.y = __byte_perm(out_lo, out_hi, 0x7362u); // {lo2, hi2, lo3, hi3}
    return result;
  }

  template <typename SymbolDecoderT>
  static __device__ void decode_row(SymbolDecoderT &sd, uint16_t *out16, uint2 packed_signs_mantissas, int row)
  {
    sd.refill_if_needed(true);
    uint32_t packed_exp;
    sd.template decode_pair<false /*pack mantissas*/, false /*second symbol*/>(packed_exp);
    sd.template decode_pair<false /*pack mantissas*/, true /*second symbol*/>(packed_exp);
    *reinterpret_cast<uint2 *>(out16 + get_lane_half_ix(row, 0, get_lane_id())) =
      reconstruct_eight(packed_exp, packed_signs_mantissas.x);
    sd.template decode_pair<false /*pack mantissas*/, false /*second symbol*/>(packed_exp);
    sd.template decode_pair<false /*pack mantissas*/, true /*second symbol*/>(packed_exp);
    *reinterpret_cast<uint2 *>(out16 + get_lane_half_ix(row, 1, get_lane_id())) =
      reconstruct_eight(packed_exp, packed_signs_mantissas.y);
  }

  // 64-symbol remainder windows (32 lanes x 1 adjacent symbol pair), the mirror of
  // FP8EncodePolicy::remainder_iter. The last short window is Partial and skips packing.
  template <bool Partial, typename SymbolDecoderT>
  static __device__ void remainder_iter(
    SymbolDecoderT &sd,
    uint16_t *out16,
    const uint8_t *mantissas,
    int window_base,
    [[maybe_unused]] int window_n
  )
  {
    assert(WARP_SIZE * 2 >= window_n);
    const int pair_ix = 2 * get_lane_id();
    const int ix = window_base + pair_ix;
    if constexpr (Partial)
    {
      const bool has_even = pair_ix < window_n;
      const bool has_odd = (pair_ix + 1) < window_n;
      sd.refill_if_needed(true);
      uint32_t entry_even, entry_odd;
      sd.template decode_pair<false /*pack mantissas*/, false /*second symbol*/, true /*partial*/>(
        entry_even,
        0u,
        has_even,
        has_odd,
        &entry_odd
      );
      if (has_even)
      {
        out16[ix] = reconstruct_pair(static_cast<uint8_t>(entry_even), mantissas[ix]);
      }
      if (has_odd)
      {
        out16[ix + 1] = reconstruct_pair(static_cast<uint8_t>(entry_odd), mantissas[ix + 1]);
      }
    }
    else
    {
      const uint32_t signs_mantissas = static_cast<uint32_t>(mantissas[ix]) |
                                       (static_cast<uint32_t>(mantissas[ix + 1]) << 8);
      sd.refill_if_needed(true);
      uint32_t entry_pair;
      sd.template decode_pair<false /*pack mantissas*/, false /*second symbol*/>(entry_pair);
      // window_base is even, so out16 + ix is 4 B-aligned and both values go out at once.
      *reinterpret_cast<uint32_t *>(out16 + ix) = reconstruct_four(entry_pair, signs_mantissas);
    }
  }

  template <bool BOUNDS_CHECK, typename SymbolDecoderT>
  static __device__ void
  decode_body(SymbolDecoderT &sd, void *uncomp_sub_chunk, const uint8_t *sub_chunk_mantissas, int num_decodes);
};

// Two-state block-interleave. Mirror of FP16X2EncodeImpl.
struct FP16X2Impl
{
  static constexpr int SYMBOLS_PER_LANE = 8;
  static constexpr int ROW_SYMBOLS = WARP_SIZE * SYMBOLS_PER_LANE;
  static constexpr int ROWS_PER_TILE = 2;
  static constexpr int TILE_SYMBOLS = ROWS_PER_TILE * ROW_SYMBOLS;
  static constexpr uint32_t STATES_PER_LANE = 2u;
  // decode_body refills before every row, so a row is the consume unit -- the 3-row group
  // it decodes them in only exists to keep mantissa loads in flight.
  static constexpr uint32_t REFILL_SYMBOLS_PER_LANE = SYMBOLS_PER_LANE;
};

// Single-state. Mirror of FP16X1EncodeImpl.
struct FP16X1Impl
{
  static constexpr int SYMBOLS_PER_LANE = 8;
  static constexpr int ROW_SYMBOLS = WARP_SIZE * SYMBOLS_PER_LANE;
  static constexpr int ROWS_PER_TILE = 1;
  static constexpr int TILE_SYMBOLS = ROWS_PER_TILE * ROW_SYMBOLS;
  static constexpr uint32_t STATES_PER_LANE = 1u;
  // decode_body refills before every row, so a row is the consume unit.
  static constexpr uint32_t REFILL_SYMBOLS_PER_LANE = SYMBOLS_PER_LANE;
};

// Two-state block-interleave, the fp8 default.
struct FP8X2Impl
{
  static constexpr int LANE_SYMBOLS_PER_ROW = 8;
  static constexpr int ROW_SYMBOLS = WARP_SIZE * LANE_SYMBOLS_PER_ROW;
  static constexpr int ROWS_PER_TILE = 2;
  static constexpr int TILE_SYMBOLS = ROWS_PER_TILE * ROW_SYMBOLS;
  static constexpr uint32_t STATES_PER_LANE = 2u;
  // decode_body refills before every row, so a row is the consume unit -- the 4-row group
  // it decodes them in only exists to keep mantissa loads in flight.
  static constexpr uint32_t REFILL_SYMBOLS_PER_LANE = LANE_SYMBOLS_PER_ROW;
};

// Single-state fp8. Mirror of FP8X1EncodeImpl.
struct FP8X1Impl
{
  static constexpr int LANE_SYMBOLS_PER_ROW = 8;
  static constexpr int ROW_SYMBOLS = WARP_SIZE * LANE_SYMBOLS_PER_ROW;
  static constexpr int ROWS_PER_TILE = 1;
  static constexpr int TILE_SYMBOLS = ROWS_PER_TILE * ROW_SYMBOLS;
  static constexpr uint32_t STATES_PER_LANE = 1u;
  static constexpr uint32_t REFILL_SYMBOLS_PER_LANE = LANE_SYMBOLS_PER_ROW;
};

template <typename Impl>
template <bool BOUNDS_CHECK, typename SymbolDecoderT>
inline __device__ void FP16DecodePolicy<Impl>::decode_body(
  SymbolDecoderT &sd,
  void *uncomp_sub_chunk,
  const uint8_t *sub_chunk_mantissas,
  int num_decodes
)
{
  constexpr int TILE_SYMBOLS = Impl::TILE_SYMBOLS;
  static_assert(ROWS_PER_TILE == 1 || ROWS_PER_TILE == 2);

  uint16_t *out16 = reinterpret_cast<uint16_t *>(uncomp_sub_chunk);

  sd.wait_initial_prefetch();

  // Groups of 3 rows execute (pack, fuse, fuse)
  // The first row of a group should be packed to help hide the mantissa load stall
  // Fused rows require the mantissa to be loaded before the second pair can be decoded.
  int linear_row = 0;
  {
    int rows_left = (num_decodes / TILE_SYMBOLS) * ROWS_PER_TILE;
    for (; rows_left >= 3; rows_left -= 3, linear_row += 3)
    {
      const uint2 sb0 = load_mantissas(sub_chunk_mantissas, linear_row);
      const uint2 sb1 = load_mantissas(sub_chunk_mantissas, linear_row + 1);
      const uint2 sb2 = load_mantissas(sub_chunk_mantissas, linear_row + 2);
      decode_packed_row(sd, out16, sb0, linear_row);
      decode_fused_row(sd, out16, sb1, linear_row + 1);
      decode_fused_row(sd, out16, sb2, linear_row + 2);
    }
    if (rows_left >= 2)
    {
      const uint2 sb0 = load_mantissas(sub_chunk_mantissas, linear_row);
      const uint2 sb1 = load_mantissas(sub_chunk_mantissas, linear_row + 1);
      decode_packed_row(sd, out16, sb0, linear_row);
      decode_fused_row(sd, out16, sb1, linear_row + 1);
      linear_row += 2;
    }
    else if (rows_left >= 1)
    {
      const uint2 sb0 = load_mantissas(sub_chunk_mantissas, linear_row);
      decode_packed_row(sd, out16, sb0, linear_row);
      linear_row += 1;
    }
  }

  const int rem_base = (linear_row / ROWS_PER_TILE) * TILE_SYMBOLS;
  const int rem = num_decodes - rem_base;
  if (rem != 0)
  {
    constexpr int STEP = WARP_SIZE * 2;
    const int full_remainder_steps = rem / STEP;
    const int tail = rem - full_remainder_steps * STEP;
    for (int ix_step = 0; ix_step < full_remainder_steps; ++ix_step)
    {
      remainder_iter<false>(sd, out16, sub_chunk_mantissas, rem_base + ix_step * STEP, STEP);
    }
    if (tail != 0)
    {
      remainder_iter<true>(sd, out16, sub_chunk_mantissas, rem_base + full_remainder_steps * STEP, tail);
    }
  }
}

// FP8 (E4M3) decode, the exact mirror of FP8EncodePolicy::encode_body and shaped like
// FP16DecodePolicy::decode_body: full tiles forward (lane t owns two 4-symbol halves
// [row_base + 4t .. +3] and [row_base + 128 + 4t .. +3]), then the 64-symbol remainder
// windows, then the last short window.
//
// Rows are decoded in groups of 4 so eight mantissa LDGs are in flight while the first row
// decodes. Each row is two decode_pairs, coalesced STG.64, then the other half.
//
// num_decodes is the sub-chunk's symbol (pair) count.
template <typename Impl>
template <bool BOUNDS_CHECK, typename SymbolDecoderT>
inline __device__ void FP8DecodePolicy<Impl>::decode_body(
  SymbolDecoderT &sd,
  void *uncomp_sub_chunk,
  const uint8_t *sub_chunk_mantissas, // packed_signs_mantissas stream (1 byte/symbol, symbol order)
  int num_decodes
)
{
  static_assert(ROWS_PER_TILE == 1 || ROWS_PER_TILE == 2);

  uint16_t *out16 = reinterpret_cast<uint16_t *>(uncomp_sub_chunk);

  sd.wait_initial_prefetch();

  // Symbol countdown: both bounds are constants, and on exit syms_left is the remainder and
  // row the rows consumed.
  constexpr int GROUP_ROWS = 4;
  constexpr int GROUP_SYMBOLS = (GROUP_ROWS / ROWS_PER_TILE) * TILE_SYMBOLS;
  int row = 0;
  int syms_left = num_decodes;
  for (; syms_left >= GROUP_SYMBOLS; syms_left -= GROUP_SYMBOLS, row += GROUP_ROWS)
  {
    const uint2 sm0 = load_mantissas(sub_chunk_mantissas, row);
    const uint2 sm1 = load_mantissas(sub_chunk_mantissas, row + 1);
    const uint2 sm2 = load_mantissas(sub_chunk_mantissas, row + 2);
    const uint2 sm3 = load_mantissas(sub_chunk_mantissas, row + 3);
    decode_row(sd, out16, sm0, row);
    decode_row(sd, out16, sm1, row + 1);
    decode_row(sd, out16, sm2, row + 2);
    decode_row(sd, out16, sm3, row + 3);
  }
  for (; syms_left >= TILE_SYMBOLS; syms_left -= TILE_SYMBOLS, row += ROWS_PER_TILE)
  {
    const uint2 sm = load_mantissas(sub_chunk_mantissas, row);
    decode_row(sd, out16, sm, row);
    if constexpr (ROWS_PER_TILE == 2)
    {
      const uint2 sm1 = load_mantissas(sub_chunk_mantissas, row + 1);
      decode_row(sd, out16, sm1, row + 1);
    }
  }

  const int rem = syms_left;
  if (rem != 0)
  {
    // Each remainder iteration decodes two symbols at a time.
    constexpr int STEP = WARP_SIZE * 2;
    const int rem_base = (row / ROWS_PER_TILE) * TILE_SYMBOLS;
    const int full_remainder_steps = rem / STEP;
    const int tail = rem - full_remainder_steps * STEP;
    for (int ix_step = 0; ix_step < full_remainder_steps; ++ix_step)
    {
      remainder_iter<false>(sd, out16, sub_chunk_mantissas, rem_base + ix_step * STEP, STEP);
    }
    if (tail != 0)
    {
      remainder_iter<true>(sd, out16, sub_chunk_mantissas, rem_base + full_remainder_steps * STEP, tail);
    }
  }
}

// FP32 decode walks full tiles forward, followed by the mapped partial tile. Both
// state counts produce the same packed exponent order; only their decoder differs.
template <typename Impl>
template <bool BOUNDS_CHECK, typename SymbolDecoderT>
inline __device__ void FP32DecodePolicy<Impl>::decode_body(
  SymbolDecoderT &sd,
  void *uncomp_sub_chunk,
  const uint8_t *sub_chunk_mantissas,
  int num_decodes
)
{
  uint32_t *const output = reinterpret_cast<uint32_t *>(uncomp_sub_chunk);
  const int full_tiles = num_decodes / TILE_SYMBOLS;
  const int lane_id = get_lane_id();

  sd.wait_initial_prefetch();

  auto decode_tile = [&](int tile, const TileMantissas &mantissas) {
    const int tile_base = tile * TILE_SYMBOLS;
#pragma unroll
    for (int group = 0; group < FP32Tile::GROUPS_PER_TILE; ++group)
    {
      uint32_t packed_exponents;
      sd.template decode_pair<false /*fuse mantissas*/, false /*second symbol*/>(packed_exponents);
      sd.template decode_pair<false /*fuse mantissas*/, true /*second symbol*/>(packed_exponents);
      *reinterpret_cast<uint4 *>(output + tile_base + FP32Tile::lane_value_offset(group, lane_id)) = reconstruct_four(
        packed_exponents,
        mantissas.segments[0].values[group],
        mantissas.segments[1].values[group],
        mantissas.segments[2].values[group]
      );
    }
  };

  constexpr int TILES_PER_BATCH = 2;
  int tile = 0;
  for (; tile + TILES_PER_BATCH <= full_tiles; tile += TILES_PER_BATCH)
  {
    TileMantissas mantissas[TILES_PER_BATCH];
#pragma unroll
    for (int batch_index = 0; batch_index < TILES_PER_BATCH; ++batch_index)
    {
      mantissas[batch_index] = load_tile_mantissas(sub_chunk_mantissas, tile + batch_index, lane_id);
    }
#pragma unroll
    for (int batch_index = 0; batch_index < TILES_PER_BATCH; ++batch_index)
    {
      sd.refill_if_needed(true);
      decode_tile(tile + batch_index, mantissas[batch_index]);
    }
  }
  for (; tile < full_tiles; ++tile)
  {
    const TileMantissas mantissas = load_tile_mantissas(sub_chunk_mantissas, tile, lane_id);
    sd.refill_if_needed(true);
    decode_tile(tile, mantissas);
  }

  const int remainder_base = full_tiles * TILE_SYMBOLS;
  const int remainder = num_decodes - remainder_base;
  if (remainder != 0)
  {
    const uint8_t *const mantissas = sub_chunk_mantissas + FP32Tile::MANTISSA_BYTES_PER_SYMBOL * remainder_base;
#pragma unroll
    for (int pair = 0; pair < FP32Tile::LANE_PAIRS_PER_TILE; ++pair)
    {
      const int even_offset = FP32Tile::lane_pair_value_offset(pair, lane_id);
      const int odd_offset = even_offset + 1;
      const bool even_active = even_offset < remainder;
      const bool odd_active = odd_offset < remainder;

      sd.refill_if_needed(true);
      uint32_t even_entry;
      uint32_t odd_entry;
      sd.template decode_pair<false /*fuse mantissas*/, false /*second symbol*/, true /*partial*/>(
        even_entry,
        0u,
        even_active,
        odd_active,
        &odd_entry
      );
      if (even_active)
      {
        output[remainder_base + even_offset] = reconstruct_at(mantissas, even_offset, even_entry);
      }
      if (odd_active)
      {
        output[remainder_base + odd_offset] = reconstruct_at(mantissas, odd_offset, odd_entry);
      }
    }
  }
}

} // namespace detail
} // namespace ans_gpu_lib
