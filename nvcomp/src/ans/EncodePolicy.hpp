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

#include <ans/ans_arch_profile.cuh> // ANS_COMP_MIN_BLOCKS_PER_SM
#include <ans/simple_types.cuh> // EncodeChunkSmem, WarpEncodeOff
#include <ans/symbol_encoder.cuh>
#include <ans/types.cuh>

namespace ans_gpu_lib
{
namespace detail
{

inline __device__ uint32_t byte_off(const void *base, const void *p)
{
  return static_cast<uint32_t>(static_cast<const uint8_t *>(p) - static_cast<const uint8_t *>(base));
}

// Which symbols a sampled model is guaranteed to cover. A sample can miss a symbol the
// chunk does use, so a policy that opts into sampling picks one of these.
enum class HistFloor
{
  None, // never samples; histograms exactly whatever the shift asks for
  ObservedBand, // floor [-HIST_DILATE_NEG, +HIST_DILATE_POS] around the support; needs detect / redo
  WholeAlphabet, // floor every symbol; covers any chunk, so detect / redo compiles out
};

// Compress-side policies (char / fp16 / fp8 / fp32): type-specific histogram + encode
// hooks for compute_histogram / histogram_warp_impl / encode_sub_chunk.
template <typename Derived, AnsStreamType StreamType>
struct EncodePolicyBase
{
  static constexpr AnsStreamType STREAM_TYPE = StreamType;
  static constexpr uint32_t TABLELOG = ans_tablelog(STREAM_TYPE);

  // uint2 (LDG.64) loads.
  using LoadT = uint2;

  // Histogram cp.async depth: how many warp-rows each lane keeps in flight through the
  // shared staging window. A policy that widens LoadT must restate this so the row count
  // matches its own load width (the base cannot ask Derived, which is still incomplete).
  static constexpr int HIST_PREFETCH_DEPTH = hist_prefetch_depth_for<LoadT>();

  static __device__ IndexT num_symbols(IndexT bytes) { return bytes / Derived::INPUT_BYTES_PER_SYMBOL; }
  static __device__ IndexT input_byte_offset(IndexT in_start) { return in_start * Derived::INPUT_BYTES_PER_SYMBOL; }

  // Whether encode_body writes the per-symbol mantissas instead of the histogram. FP16
  // and FP8 enable this so a sampled histogram can skip symbols; CHAR has no mantissas.
  static constexpr bool ENCODE_WRITES_MANTISSAS = false;

  // Histogram sampling: when the public histogram_reduction_log2 is nonzero, each warp
  // counts only its leading (range >> shift) symbols.
  static constexpr HistFloor HIST_FLOOR = HistFloor::None;
  // Read only for HistFloor::ObservedBand.
  static constexpr uint32_t HIST_DILATE_NEG = 0;
  static constexpr uint32_t HIST_DILATE_POS = 0;

  // Occupancy target for the fused compress kernel. Every policy loads its encode tiles
  // straight from gmem, so none of them needs to lower this below the arch tier's value
  // to keep that many blocks resident.
  static constexpr int COMP_MIN_BLOCKS_PER_SM = ANS_COMP_MIN_BLOCKS_PER_SM;

  // rANS states each lane owns. An x1 implementation overrides this with 1 and an x2
  // implementation with 2. The public auto setting dispatches x2 for every type.
  static constexpr uint32_t STATES_PER_LANE = 1;

  // Mantissas base for this warp's first symbol (unused for char). num_sub_chunks is
  // known before the histogram, so the region follows the sizes array.
  static __device__ uint8_t *histogram_mantissas_warp_base(uint8_t *comp_chunk, IndexT in_start, uint8_t num_sub_chunks)
  {
    return comp_chunk + ans_mantissas_offset(STREAM_TYPE, num_sub_chunks) +
           in_start * Derived::MANTISSA_BYTES_PER_SYMBOL;
  }

  // Trailing chunk bytes not covered by the ANS stream (tid==0). Only fp8 needs one.
  static __device__ void
  write_chunk_tail(uint8_t * /*comp*/, const void * /*uncomp*/, IndexT /*bytes*/, uint8_t /*num_sub_chunks*/)
  {}
};

// Two-state block-interleave. Same tile geometry as FP16X2EncodeImpl (lane t owns 8
// symbols in each of a tile's 2 rows), but a symbol is one byte: the lane's row is an
// LDG.64 and each uint32 holds 4 symbols = 2 state pairs, low byte first.
struct CharX2EncodeImpl
{
  using Encoder = symbol_encoder_dual_state<ans_tablelog(AnsStreamType::Char)>;
  static constexpr uint32_t STATES_PER_LANE = 2;
  static constexpr int SYMBOLS_PER_LANE = 8;
  static constexpr int ROW_SYMBOLS = WARP_SIZE * SYMBOLS_PER_LANE;
  static constexpr int ROWS_PER_TILE = 2;
  static constexpr int TILE_SYMBOLS = ROWS_PER_TILE * ROW_SYMBOLS;
  // Already 16 symbols per lane per tile; the loop overhead is amortized as-is.
  static constexpr int TILES_PER_ITER = 1;

  struct Tile
  {
    uint2 raw0;
    uint2 raw1;
  };

  static __device__ void load_tile(const uint8_t *raw_base, int tile_idx, int idx_in_warp, Tile &tile)
  {
    const uint8_t *src = raw_base + tile_idx * TILE_SYMBOLS + SYMBOLS_PER_LANE * idx_in_warp;
    tile.raw0 = *reinterpret_cast<const uint2 *>(src);
    tile.raw1 = *reinterpret_cast<const uint2 *>(src + ROW_SYMBOLS);
  }

  // rANS is LIFO, so rows and pairs are emitted high to low; the decoder walks row 0's
  // four pairs first. One PRMT per symbol isolates its byte with no shift/mask chain.
  template <bool Detect>
  static __device__ void encode_tile(Encoder &se, const Tile &tile)
  {
    auto enc_word = [&](uint32_t w) {
      se.template encode_pair<Detect>(char_get_symbol_from_word<3>(w), char_get_symbol_from_word<2>(w));
      se.template encode_pair<Detect>(char_get_symbol_from_word<1>(w), char_get_symbol_from_word<0>(w));
    };
    enc_word(tile.raw1.y);
    enc_word(tile.raw1.x);
    enc_word(tile.raw0.y);
    enc_word(tile.raw0.x);
  }

  // 64 symbols (32 lanes x 2 states). Lane t owns the adjacent pair at window_base + 2t.
  // Partial is the last (short) window. All 32 lanes must call.
  template <bool Detect, bool Partial>
  static __device__ void
  remainder_iter(Encoder &se, const uint8_t *raw_base, int window_base, [[maybe_unused]] int window_n, int idx_in_warp)
  {
    const int pair_ix = 2 * idx_in_warp;
    const int ix = window_base + pair_ix;
    if constexpr (Partial)
    {
      const bool has_a = pair_ix < window_n;
      const bool has_b = (pair_ix + 1) < window_n;
      const uint32_t sa = has_a ? raw_base[ix] : 0u;
      const uint32_t sb = has_b ? raw_base[ix + 1] : 0u;
      se.template encode_pair<Detect, true>(sb, sa, has_b, has_a);
    }
    else
    {
      const uint32_t w = *reinterpret_cast<const uint16_t *>(raw_base + ix);
      se.template encode_pair<Detect>(char_get_symbol_from_word<1>(w), char_get_symbol_from_word<0>(w));
    }
  }
};

// Single-state for 8-bit symbols. Same tile geometry as FP16X1EncodeImpl: one-row
// (256-symbol) tiles with lane t owning 8 contiguous symbols, so there is no G0/G1
// interleave to mirror. A symbol is one byte, so the lane's row is an LDG.64 and each
// uint32 of it holds 4 symbols.
struct CharX1EncodeImpl
{
  using Encoder = symbol_encoder<ans_tablelog(AnsStreamType::Char)>;
  static constexpr uint32_t STATES_PER_LANE = 1;
  static constexpr int SYMBOLS_PER_LANE = 8;
  static constexpr int ROW_SYMBOLS = WARP_SIZE * SYMBOLS_PER_LANE;
  static constexpr int ROWS_PER_TILE = 1;
  static constexpr int TILE_SYMBOLS = ROWS_PER_TILE * ROW_SYMBOLS;
  // A 256-symbol tile only spans 8 symbols per lane, so the tile loop's index math,
  // bounds test and address setup land on half as many symbols as x2's 512-symbol tile.
  // Encoding two tiles per iteration amortizes that back down without touching the
  // bitstream (see CharEncodePolicy::encode_body).
  static constexpr int TILES_PER_ITER = 2;

  struct Tile
  {
    uint2 raw;
  };

  static __device__ void load_tile(const uint8_t *raw_base, int tile_idx, int idx_in_warp, Tile &tile)
  {
    tile.raw = *reinterpret_cast<const uint2 *>(raw_base + tile_idx * TILE_SYMBOLS + SYMBOLS_PER_LANE * idx_in_warp);
  }

  // rANS is LIFO, so the two words and the bytes within each are emitted high to low; the
  // decoder walks byte 0 of the low word first. One PRMT per symbol isolates its byte with
  // no shift/mask chain.
  template <bool Detect>
  static __device__ void encode_tile(Encoder &se, const Tile &tile)
  {
    auto enc_word = [&](uint32_t w) {
      se.template encode_symbol<false, Detect>(char_get_symbol_from_word<3>(w));
      se.template encode_symbol<false, Detect>(char_get_symbol_from_word<2>(w));
      se.template encode_symbol<false, Detect>(char_get_symbol_from_word<1>(w));
      se.template encode_symbol<false, Detect>(char_get_symbol_from_word<0>(w));
    };
    enc_word(tile.raw.y);
    enc_word(tile.raw.x);
  }

  // 64-symbol window; lane t owns the adjacent pair at window_base + 2t. One state, so the
  // pair is two encode_symbol calls (high then low). Partial is the last (short) window.
  // All 32 lanes must call.
  template <bool Detect, bool Partial>
  static __device__ void
  remainder_iter(Encoder &se, const uint8_t *raw_base, int window_base, int window_n, int idx_in_warp)
  {
    const int pair_ix = 2 * idx_in_warp;
    const int ix = window_base + pair_ix;
    const bool has_a = !Partial || pair_ix < window_n;
    const bool has_b = !Partial || (pair_ix + 1) < window_n;
    const uint32_t sa = has_a ? raw_base[ix] : 0u;
    const uint32_t sb = has_b ? raw_base[ix + 1] : 0u;
    se.template encode_symbol<Partial, Detect>(sb, has_b);
    se.template encode_symbol<Partial, Detect>(sa, has_a);
  }
};

// Char: no mantissas. Impl owns the tile/remainder map and the encoder; the histogram,
// alignment and chunk prefix are shared so x1 and x2 stay in step.
template <typename Impl>
struct CharEncodePolicy : EncodePolicyBase<CharEncodePolicy<Impl>, AnsStreamType::Char>
{
  using Base = EncodePolicyBase<CharEncodePolicy<Impl>, AnsStreamType::Char>;
  static constexpr int PARTITION_ALIGN = 16; // symbols
  static constexpr int INPUT_BYTES_PER_SYMBOL = static_cast<int>(ans_bytes_per_symbol(Base::STREAM_TYPE));
  static constexpr int MANTISSA_BYTES_PER_SYMBOL = 0;

  // A sampled slice can miss any byte value and char has no locality for a band to exploit.
  static constexpr HistFloor HIST_FLOOR = HistFloor::WholeAlphabet;

  // 16-byte histogram loads (uint4 = 16 symbols/word at 1 byte/symbol), which the 16-byte
  // input alignment and the 16-symbol PARTITION_ALIGN both guarantee are legal.
  using LoadT = uint4;
  static constexpr int HIST_PREFETCH_DEPTH = hist_prefetch_depth_for<LoadT>();

  // One uint4 = 16 symbols. memcpy avoids PRMT swizzles from a direct uchar4 load.
  static __device__ void histogram_process_word(uint4 reg, uint32_t *counts, uint8_t * /*mantissas_dst*/)
  {
    uchar4 bytes[4];
    memcpy(bytes, &reg, sizeof(uint4));
#pragma unroll
    for (int ix = 0; ix < 4; ++ix)
    {
      atomicAdd(&counts[bytes[ix].x], 1);
      atomicAdd(&counts[bytes[ix].y], 1);
      atomicAdd(&counts[bytes[ix].z], 1);
      atomicAdd(&counts[bytes[ix].w], 1);
    }
  }

  static __device__ void histogram_process_scalar_tail_warp(
    uint32_t idx_in_warp,
    uint32_t *counts,
    const uint8_t *tail_in,
    uint8_t * /*tail_mantissas*/,
    int tail_size
  )
  {
    if (static_cast<int>(idx_in_warp) < tail_size)
    {
      atomicAdd(&counts[tail_in[idx_in_warp]], 1);
    }
  }

  // ---- encode phase ----

  using Encoder = typename Impl::Encoder;
  static constexpr uint32_t STATES_PER_LANE = Impl::STATES_PER_LANE;

  // Char floors the whole alphabet, so nothing is ever uncovered: Detect is only ever false.
  template <bool Detect>
  static __device__ void encode_body(Encoder &se, EncodeChunkSmem &enc);
};

// Shared fp16 encode (x1 and x2). Impl owns the tile/remainder map and encoder knobs.
// Histogram, rotate, mantissa layout, and chunk prefix are shared so the two modes stay
// in step.
template <typename Impl>
struct FP16EncodePolicy : EncodePolicyBase<FP16EncodePolicy<Impl>, AnsStreamType::Fp16>
{
  using Base = EncodePolicyBase<FP16EncodePolicy<Impl>, AnsStreamType::Fp16>;
  static constexpr int PARTITION_ALIGN = 8; // symbols
  static constexpr int INPUT_BYTES_PER_SYMBOL = static_cast<int>(ans_bytes_per_symbol(Base::STREAM_TYPE));
  static constexpr int MANTISSA_BYTES_PER_SYMBOL = 1;
  // The encode pass already rotates every value, so it emits the mantissas as a byproduct;
  // that also frees the histogram to read only a sample of the chunk.
  static constexpr bool ENCODE_WRITES_MANTISSAS = true;

  // 16-byte histogram loads (uint4 = 8 values/word), which the 16-byte input alignment
  // and the 8-symbol PARTITION_ALIGN both guarantee are legal.
  using LoadT = uint4;
  static constexpr int HIST_PREFETCH_DEPTH = hist_prefetch_depth_for<LoadT>();

  // The exponents cluster, so a band is far cheaper than flooring all 256 symbols.
  static constexpr HistFloor HIST_FLOOR = HistFloor::ObservedBand;
  static constexpr uint32_t HIST_DILATE_NEG = SAMPLED_HIST_DILATE_NEG;
  static constexpr uint32_t HIST_DILATE_POS = SAMPLED_HIST_DILATE_POS;

  using Encoder = typename Impl::Encoder;
  static constexpr uint32_t STATES_PER_LANE = Impl::STATES_PER_LANE;

  // Pre-6.0 high-byte split. nvcompDx still histograms this way
  // (`histogram_uint16_data_dx`); the host path uses rotate_fp_pair_left1.
  static __device__ void split_float(uint16_t float_input, uint8_t &exponent, uint8_t &mantissa)
  {
    exponent = static_cast<uint8_t>(float_input >> 8);
    mantissa = static_cast<uint8_t>(float_input);
  }

  // Histogram the 4 symbols of one uint2 (= 4 values = 2 aligned pairs): rotate both
  // words and count the odd bytes of each. The mantissas are written during encode.
  static __device__ void histogram_count_uint2(uint2 reg, uint32_t *counts)
  {
    // After the rotate each word is [sb0, sym0, sb1, sym1].
    const uint32_t r0 = rotate_fp_pair_left1(reg.x);
    const uint32_t r1 = rotate_fp_pair_left1(reg.y);

    const uint32_t packed_sym = fp16_pack_symbols_from_rotated_pairs(r0, r1);

    const uint32_t e0 = packed_sym & 0xFF;
    const uint32_t e1 = __byte_perm(packed_sym, 0, 0x4441);
    const uint32_t e2 = __byte_perm(packed_sym, 0, 0x4442);
    const uint32_t e3 = packed_sym >> 24;

    atomicAdd(&counts[e0], 1);
    atomicAdd(&counts[e1], 1);
    atomicAdd(&counts[e2], 1);
    atomicAdd(&counts[e3], 1);
  }

  // One uint4 = 8 values = two uint2 halves = 4 aligned pairs.
  static __device__ void histogram_process_word(uint4 reg, uint32_t *counts, uint8_t * /*mantissas_dst*/)
  {
    histogram_count_uint2(make_uint2(reg.x, reg.y), counts);
    histogram_count_uint2(make_uint2(reg.z, reg.w), counts);
  }

  // Scalar tail: the final < 4 values, walked as ALIGNED PAIRS because the rotate swaps
  // the two signs of a pair. Lane j owns the pair (2j, 2j+1); a trailing odd value pairs
  // with itself and so keeps its own sign. tail_base is a multiple of 4, so these pairs
  // are aligned relative to the chunk, and only a chunk's very last symbol can be unpaired.
  static __device__ void histogram_process_scalar_tail_warp(
    uint32_t idx_in_warp,
    uint32_t *counts,
    const uint8_t *tail_in,
    uint8_t * /*tail_mantissas*/,
    int tail_size
  )
  {
    const int lo_ix = 2 * static_cast<int>(idx_in_warp);
    if (lo_ix >= tail_size)
    {
      return;
    }
    const int hi_ix = lo_ix + 1;
    const bool has_hi = hi_ix < tail_size;
    const uint16_t *in16 = reinterpret_cast<const uint16_t *>(tail_in);
    const uint32_t r = rotate_fp_pair_left1(in16[lo_ix], in16[has_hi ? hi_ix : lo_ix]);

    atomicAdd(&counts[fp16_get_symbol_from_rotated<0>(r)], 1);
    if (has_hi)
    {
      atomicAdd(&counts[fp16_get_symbol_from_rotated<1>(r)], 1);
    }
  }

  // Detect: compile-time uncovered-symbol check for the sampled-histogram fallback.
  template <bool Detect>
  static __device__ void encode_body(typename Impl::Encoder &se, EncodeChunkSmem &enc);
};

// The one- and two-state FP32 modes share the same value-to-lane map and segmented
// mantissa layout. Only the order in which a group's four symbols enter the rANS
// state(s) differs.
struct FP32X2EncodeImpl
{
  using Encoder = symbol_encoder_dual_state<ans_tablelog(AnsStreamType::Fp32)>;
  static constexpr uint32_t STATES_PER_LANE = 2;

  template <bool Detect>
  static __device__ __forceinline__ void
  encode_group(Encoder &se, uint32_t rotated0, uint32_t rotated1, uint32_t rotated2, uint32_t rotated3)
  {
    se.template encode_pair<Detect>(fp32_get_symbol_from_rotated(rotated3), fp32_get_symbol_from_rotated(rotated2));
    se.template encode_pair<Detect>(fp32_get_symbol_from_rotated(rotated1), fp32_get_symbol_from_rotated(rotated0));
  }

  template <bool Detect>
  static __device__ __forceinline__ void
  encode_remainder_pair(Encoder &se, uint32_t odd_rotated, uint32_t even_rotated, bool odd_active, bool even_active)
  {
    se.template encode_pair<Detect, true>(
      fp32_get_symbol_from_rotated(odd_rotated),
      fp32_get_symbol_from_rotated(even_rotated),
      odd_active,
      even_active
    );
  }
};

struct FP32X1EncodeImpl
{
  using Encoder = symbol_encoder<ans_tablelog(AnsStreamType::Fp32)>;
  static constexpr uint32_t STATES_PER_LANE = 1;

  template <bool Detect>
  static __device__ __forceinline__ void
  encode_group(Encoder &se, uint32_t rotated0, uint32_t rotated1, uint32_t rotated2, uint32_t rotated3)
  {
    se.template encode_symbol<false, Detect>(fp32_get_symbol_from_rotated(rotated3));
    se.template encode_symbol<false, Detect>(fp32_get_symbol_from_rotated(rotated2));
    se.template encode_symbol<false, Detect>(fp32_get_symbol_from_rotated(rotated1));
    se.template encode_symbol<false, Detect>(fp32_get_symbol_from_rotated(rotated0));
  }

  template <bool Detect>
  static __device__ __forceinline__ void
  encode_remainder_pair(Encoder &se, uint32_t odd_rotated, uint32_t even_rotated, bool odd_active, bool even_active)
  {
    se.template encode_symbol<true, Detect>(fp32_get_symbol_from_rotated(odd_rotated), odd_active);
    se.template encode_symbol<true, Detect>(fp32_get_symbol_from_rotated(even_rotated), even_active);
  }
};

// FP32 uses the exponent byte as its ANS symbol and stores the sign plus 23-bit
// mantissa in a three-byte side stream. Both state counts use FP32Tile verbatim.
template <typename Impl>
struct FP32EncodePolicy : EncodePolicyBase<FP32EncodePolicy<Impl>, AnsStreamType::Fp32>
{
  using Base = EncodePolicyBase<FP32EncodePolicy<Impl>, AnsStreamType::Fp32>;
  static constexpr int INPUT_BYTES_PER_SYMBOL = static_cast<int>(ans_bytes_per_symbol(Base::STREAM_TYPE));
  static constexpr int MANTISSA_BYTES_PER_SYMBOL = FP32Tile::MANTISSA_BYTES_PER_SYMBOL;
  static constexpr int PARTITION_ALIGN = 4; // one uint4 of FP32 values
  static constexpr bool ENCODE_WRITES_MANTISSAS = true;

  using LoadT = uint4;
  static constexpr int HIST_PREFETCH_DEPTH = hist_prefetch_depth_for<LoadT>();

  // Same 8-bit exponent alphabet as BF16, so the same band floor and radii apply.
  static constexpr HistFloor HIST_FLOOR = HistFloor::ObservedBand;
  static constexpr uint32_t HIST_DILATE_NEG = SAMPLED_HIST_DILATE_NEG;
  static constexpr uint32_t HIST_DILATE_POS = SAMPLED_HIST_DILATE_POS;

  using Encoder = typename Impl::Encoder;
  static constexpr uint32_t STATES_PER_LANE = Impl::STATES_PER_LANE;

  static __device__ void histogram_process_word(uint4 reg, uint32_t *counts, uint8_t * /*mantissas_dst*/)
  {
    atomicAdd(&counts[fp32_exponent_symbol(reg.x)], 1);
    atomicAdd(&counts[fp32_exponent_symbol(reg.y)], 1);
    atomicAdd(&counts[fp32_exponent_symbol(reg.z)], 1);
    atomicAdd(&counts[fp32_exponent_symbol(reg.w)], 1);
  }

  static __device__ void histogram_process_scalar_tail_warp(
    uint32_t idx_in_warp,
    uint32_t *counts,
    const uint8_t *tail_in,
    uint8_t * /*tail_mantissas*/,
    int tail_size
  )
  {
    if (static_cast<int>(idx_in_warp) < tail_size)
    {
      atomicAdd(&counts[fp32_exponent_symbol(reinterpret_cast<const uint32_t *>(tail_in)[idx_in_warp])], 1);
    }
  }

  // Detect: compile-time uncovered-symbol check for the sampled-histogram fallback.
  template <bool Detect>
  static __device__ void encode_body(Encoder &se, EncodeChunkSmem &enc);
};

template <typename Impl>
struct FP8EncodePolicy : EncodePolicyBase<FP8EncodePolicy<Impl>, AnsStreamType::Fp8>
{
  using Base = EncodePolicyBase<FP8EncodePolicy<Impl>, AnsStreamType::Fp8>;
  static constexpr int PARTITION_ALIGN = 8; // pairs
  static constexpr int INPUT_BYTES_PER_SYMBOL = static_cast<int>(ans_bytes_per_symbol(Base::STREAM_TYPE));
  static constexpr int MANTISSA_BYTES_PER_SYMBOL = 1;
  static constexpr bool ENCODE_WRITES_MANTISSAS = true;

  using LoadT = uint4;
  static constexpr int HIST_PREFETCH_DEPTH = hist_prefetch_depth_for<LoadT>();

  // Sampled model from the public histogram_reduction_log2 option, made safe by flooring
  // every packed_exp probability to 1.
  static constexpr HistFloor HIST_FLOOR = HistFloor::WholeAlphabet;

  using Encoder = typename Impl::Encoder;
  static constexpr uint32_t STATES_PER_LANE = Impl::STATES_PER_LANE;

  static constexpr int LANE_SYMBOLS_PER_ROW = 8;
  static constexpr int LOAD_STORE_SYMBOLS_PER_LANE = 4;
  static constexpr int HALVES_PER_ROW = LANE_SYMBOLS_PER_ROW / LOAD_STORE_SYMBOLS_PER_LANE;
  static constexpr int ROW_SYMBOLS = WARP_SIZE * LANE_SYMBOLS_PER_ROW;
  static constexpr int HALF_SYMBOLS = WARP_SIZE * LOAD_STORE_SYMBOLS_PER_LANE;
  static constexpr int ROWS_PER_TILE = Impl::ROWS_PER_TILE;
  static constexpr int TILE_SYMBOLS = ROWS_PER_TILE * ROW_SYMBOLS;
  static constexpr int ROW_RAW_BYTES = INPUT_BYTES_PER_SYMBOL * ROW_SYMBOLS;
  static constexpr int HALF_RAW_BYTES = INPUT_BYTES_PER_SYMBOL * HALF_SYMBOLS;
  static constexpr int TILE_RAW_BYTES = INPUT_BYTES_PER_SYMBOL * TILE_SYMBOLS;
  static_assert(LANE_SYMBOLS_PER_ROW == LOAD_STORE_SYMBOLS_PER_LANE * HALVES_PER_ROW);
  static_assert(ROW_SYMBOLS == HALVES_PER_ROW * HALF_SYMBOLS);

  // One uint32 = 4 raw FP8 bytes = 2 pairs = 2 packed_exp symbols.
  static __device__ void histogram_count_quad(uint32_t w, uint32_t *counts)
  {
    const uint32_t packed_exp_lo16 = fp8_quad_packed_exp_e4m3(w);
    atomicAdd(&counts[packed_exp_lo16 & 0xFFu], 1);
    atomicAdd(&counts[packed_exp_lo16 >> 8], 1);
  }

  // One uint4 = 16 raw FP8 bytes = 8 pairs. Counts only; the mantissas are written during
  // encode, which is what lets a sampled histogram skip most of the chunk.
  static __device__ void histogram_process_word(uint4 reg, uint32_t *counts, uint8_t * /*mantissas_dst*/)
  {
    histogram_count_quad(reg.x, counts);
    histogram_count_quad(reg.y, counts);
    histogram_count_quad(reg.z, counts);
    histogram_count_quad(reg.w, counts);
  }

  // Scalar tail: the final < 8 pairs that do not fill a uint4. One lane per pair.
  static __device__ void histogram_process_scalar_tail_warp(
    uint32_t idx_in_warp,
    uint32_t *counts,
    const uint8_t *tail_in,
    uint8_t * /*tail_mantissas*/,
    int tail_size
  )
  {
    if (static_cast<int>(idx_in_warp) < tail_size)
    {
      // Exponents only, like the vector path: the mantissa nibbles a full split would
      // also produce are written during encode, not here.
      const uint32_t packed_exp = static_cast<uint32_t>(fp8_exp_nibble(tail_in[2 * idx_in_warp + 0])) |
                                  (static_cast<uint32_t>(fp8_exp_nibble(tail_in[2 * idx_in_warp + 1])) << 4);
      atomicAdd(&counts[packed_exp], 1);
    }
  }

  // Odd-N trailing raw byte after the packed_signs_mantissas stream (tid==0).
  static __device__ void
  write_chunk_tail(uint8_t *comp_chunk, const void *uncomp_chunk, IndexT bytes, uint8_t num_sub_chunks)
  {
    const IndexT num_pairs = Base::num_symbols(bytes);
    if (bytes & 1)
    {
      (comp_chunk + ans_mantissas_offset(Base::STREAM_TYPE, num_sub_chunks))[num_pairs] =
        reinterpret_cast<const uint8_t *>(uncomp_chunk)[bytes - 1];
    }
  }

  // ---- encode phase ----

  // One tile's raw FP8 input for this lane: one coalesced LDG.64 per half (two per row).
  struct Tile
  {
    uint2 raw[ROWS_PER_TILE][HALVES_PER_ROW];
  };

  static __device__ void load_tile(const uint8_t *raw_base, int tile_idx, int idx_in_warp, Tile &tile)
  {
    const uint8_t *const src = raw_base + tile_idx * TILE_RAW_BYTES +
                               INPUT_BYTES_PER_SYMBOL * LOAD_STORE_SYMBOLS_PER_LANE * idx_in_warp;
#pragma unroll
    for (int r = 0; r < ROWS_PER_TILE; ++r)
    {
#pragma unroll
      for (int h = 0; h < HALVES_PER_ROW; ++h)
      {
        tile.raw[r][h] = *reinterpret_cast<const uint2 *>(src + r * ROW_RAW_BYTES + h * HALF_RAW_BYTES);
      }
    }
  }

  // One 4-symbol half: two pairs, 8 raw bytes, 4 mantissa bytes, one coalesced STG.32 of
  // mantissas. rANS reverse: the high pair (raw.y) first. Seed is that high pair.
  template <bool Seed>
  static __device__ __forceinline__ void encode_half(Encoder &se, uint8_t *mantissas_half, uint2 raw)
  {
    uint32_t packed_exp;
    uint32_t packed_signs_mantissas;
    split_fp8_eight_e4m3(raw, packed_exp, packed_signs_mantissas);
    *reinterpret_cast<uint32_t *>(mantissas_half) = packed_signs_mantissas;
    if constexpr (Seed)
    {
      Impl::encode_first_pair(se, packed_exp >> 24, (packed_exp >> 16) & 0xFFu);
    }
    else
    {
      Impl::encode_pair(se, packed_exp >> 24, (packed_exp >> 16) & 0xFFu);
    }
    Impl::encode_pair(se, (packed_exp >> 8) & 0xFFu, packed_exp & 0xFFu);
  }

  // Halves of a row, high -> low. Seed applies to the high half's high pair.
  template <bool Seed>
  static __device__ __forceinline__ void encode_row(Encoder &se, uint8_t *mantissas_half0, uint2 raw0, uint2 raw1)
  {
    encode_half<Seed>(se, mantissas_half0 + HALF_SYMBOLS, raw1);
    encode_half<false>(se, mantissas_half0, raw0);
  }

  // Rows of a tile, high -> low (rANS reverse). Seed applies to the top row's high-half
  // high pair.
  template <bool Seed>
  static __device__ __forceinline__ void
  encode_tile(Encoder &se, uint8_t *mantissas_base, int tile_idx, int idx_in_warp, const Tile &tile)
  {
    uint8_t *const dst = mantissas_base + tile_idx * TILE_SYMBOLS + LOAD_STORE_SYMBOLS_PER_LANE * idx_in_warp;
#pragma unroll
    for (int r = ROWS_PER_TILE - 1; r >= 0; --r)
    {
      const bool seed_row = Seed && (r == ROWS_PER_TILE - 1);
      uint8_t *const row_dst = dst + r * ROW_SYMBOLS;
      if (seed_row)
      {
        encode_row<Seed>(se, row_dst, tile.raw[r][0], tile.raw[r][1]);
      }
      else
      {
        encode_row<false>(se, row_dst, tile.raw[r][0], tile.raw[r][1]);
      }
    }
  }

  // 64 symbols (32 lanes x 1 adjacent pair). Lane t owns the pair (window_base + 2t, +1),
  // so the even symbol rides state 0 and the odd one state 1 exactly as in a tile row.
  // Partial is the last (short) window; all 32 lanes must call.
  template <bool Partial>
  static __device__ void remainder_iter(
    Encoder &se,
    uint8_t *mantissas_base,
    const uint8_t *raw_base,
    int idx_in_warp,
    int window_base,
    [[maybe_unused]] int window_n
  )
  {
    assert(WARP_SIZE * 2 >= window_n);
    const int pair_ix = 2 * idx_in_warp;
    const int ix = window_base + pair_ix;
    if constexpr (Partial)
    {
      const bool has_even = pair_ix < window_n;
      const bool has_odd = (pair_ix + 1) < window_n;
      uint8_t packed_exp_even = 0, signs_mantissas_even = 0;
      uint8_t packed_exp_odd = 0, signs_mantissas_odd = 0;
      if (has_even)
      {
        split_fp8_pair(raw_base[2 * ix], raw_base[2 * ix + 1], packed_exp_even, signs_mantissas_even);
      }
      if (has_odd)
      {
        split_fp8_pair(raw_base[2 * ix + 2], raw_base[2 * ix + 3], packed_exp_odd, signs_mantissas_odd);
      }
      Impl::encode_pair_partial(se, packed_exp_odd, packed_exp_even, has_odd, has_even);
      if (has_even)
      {
        mantissas_base[ix] = signs_mantissas_even;
      }
      if (has_odd)
      {
        mantissas_base[ix + 1] = signs_mantissas_odd;
      }
    }
    else
    {
      // Both pairs are the 4 contiguous raw bytes at raw_base + 2*ix; window_base is even,
      // so that address is 4 B-aligned.
      uint32_t packed_exp_lo16, signs_mantissas_lo16;
      split_fp8_quad_e4m3(*reinterpret_cast<const uint32_t *>(raw_base + 2 * ix), packed_exp_lo16, signs_mantissas_lo16);
      *reinterpret_cast<uint16_t *>(mantissas_base + ix) = static_cast<uint16_t>(signs_mantissas_lo16);
      Impl::encode_pair(se, packed_exp_lo16 >> 8, packed_exp_lo16 & 0xFFu);
    }
  }

  // fp8 floors its whole alphabet, so Detect is only ever instantiated false.
  template <bool Detect>
  static __device__ void encode_body(Encoder &se, EncodeChunkSmem &enc);
};

// Two-state block-interleave. Lane t owns 8 symbols in row 0 and 8 in row 1 per
// 512-symbol tile, both as one LDG.128 from gmem.
struct FP16X2EncodeImpl
{
  using Encoder = symbol_encoder_dual_state<ans_tablelog(AnsStreamType::Fp16)>;
  static constexpr uint32_t STATES_PER_LANE = 2;
  static constexpr int SYMBOLS_PER_LANE = 8;
  static constexpr int ROW_SYMBOLS = WARP_SIZE * SYMBOLS_PER_LANE;
  static constexpr int ROWS_PER_TILE = 2;
  static constexpr int TILE_SYMBOLS = ROWS_PER_TILE * ROW_SYMBOLS;
  struct Tile
  {
    uint4 raw0;
    uint4 raw1;
  };

  static __device__ void load_tile(const uint8_t *raw_base, int tile_idx, int idx_in_warp, Tile &tile)
  {
    constexpr int TILE_RAW_BYTES = 2 * TILE_SYMBOLS;
    const uint8_t *src = raw_base + tile_idx * TILE_RAW_BYTES + 2 * SYMBOLS_PER_LANE * idx_in_warp;
    tile.raw0 = *reinterpret_cast<const uint4 *>(src);
    tile.raw1 = *reinterpret_cast<const uint4 *>(src + 2 * ROW_SYMBOLS);
  }

  // Each uint32 word is an interleaved {A,B} pair. One SHF rotates the whole word so both
  // symbol bytes land in bytes 1 and 3; the transform helpers extract B/A for encode_pair
  // and pack the four mantissa bytes of two words in symbol order.
  template <bool Detect>
  static __device__ __forceinline__ void
  encode_tile(Encoder &se, uint8_t *mantissas_base, int tile_idx, int idx_in_warp, const Tile &tile)
  {
    const uint32_t r0x = rotate_fp_pair_left1(tile.raw0.x);
    const uint32_t r0y = rotate_fp_pair_left1(tile.raw0.y);
    const uint32_t r0z = rotate_fp_pair_left1(tile.raw0.z);
    const uint32_t r0w = rotate_fp_pair_left1(tile.raw0.w);
    const uint32_t r1x = rotate_fp_pair_left1(tile.raw1.x);
    const uint32_t r1y = rotate_fp_pair_left1(tile.raw1.y);
    const uint32_t r1z = rotate_fp_pair_left1(tile.raw1.z);
    const uint32_t r1w = rotate_fp_pair_left1(tile.raw1.w);

    uint8_t *const dst = mantissas_base + tile_idx * TILE_SYMBOLS + SYMBOLS_PER_LANE * idx_in_warp;
    *reinterpret_cast<uint2 *>(dst) =
      make_uint2(fp16_pack_mantissas_from_rotated_pairs(r0x, r0y), fp16_pack_mantissas_from_rotated_pairs(r0z, r0w));
    *reinterpret_cast<uint2 *>(dst + ROW_SYMBOLS) =
      make_uint2(fp16_pack_mantissas_from_rotated_pairs(r1x, r1y), fp16_pack_mantissas_from_rotated_pairs(r1z, r1w));

    auto enc_rot = [&](uint32_t r) {
      se.template encode_pair<Detect>(fp16_get_symbol_from_rotated<1>(r), fp16_get_symbol_from_rotated<0>(r));
    };
    enc_rot(r1w);
    enc_rot(r1z);
    enc_rot(r1y);
    enc_rot(r1x);
    enc_rot(r0w);
    enc_rot(r0z);
    enc_rot(r0y);
    enc_rot(r0x);
  }

  // 64 symbols (32 lanes x 2 states). Lane t owns the aligned pair at window_base + 2t.
  // Partial is the last (short) window; a trailing unpaired symbol pairs with itself.
  // All 32 lanes must call.
  template <bool Detect, bool Partial>
  static __device__ void remainder_iter(
    Encoder &se,
    uint8_t *mantissas_base,
    const uint8_t *raw_base,
    int idx_in_warp,
    int window_base,
    [[maybe_unused]] int window_n
  )
  {
    const int pair_ix = 2 * idx_in_warp;
    const int ix = window_base + pair_ix;
    if constexpr (Partial)
    {
      const uint16_t *fp16_base = reinterpret_cast<const uint16_t *>(raw_base);
      const bool has_a = pair_ix < window_n;
      const bool has_b = (pair_ix + 1) < window_n;
      const uint16_t va = has_a ? fp16_base[ix] : 0;
      const uint16_t vb = has_b ? fp16_base[ix + 1] : va;
      const uint32_t r = has_a ? rotate_fp_pair_left1(va, vb) : 0u;
      se.template encode_pair<Detect, true>(
        fp16_get_symbol_from_rotated<1>(r),
        fp16_get_symbol_from_rotated<0>(r),
        has_b,
        has_a
      );
      if (has_a)
      {
        mantissas_base[ix] = fp16_get_mantissa_from_rotated<0>(r);
      }
      if (has_b)
      {
        mantissas_base[ix + 1] = fp16_get_mantissa_from_rotated<1>(r);
      }
    }
    else
    {
      const uint32_t w = *reinterpret_cast<const uint32_t *>(raw_base + 2 * ix);
      const uint32_t r = rotate_fp_pair_left1(w);
      *reinterpret_cast<uint16_t *>(mantissas_base + ix) = static_cast<uint16_t>(__byte_perm(r, 0u, 0x20u));
      se.template encode_pair<Detect>(fp16_get_symbol_from_rotated<1>(r), fp16_get_symbol_from_rotated<0>(r));
    }
  }
};

// Single-state. One-row (256-symbol) tiles with lane t owning 8 contiguous symbols, so there
// is no A/B interleave to mirror. 8t is even, so a lane's 8 symbols are still 4 aligned
// pairs and the rotate rule is unchanged.
struct FP16X1EncodeImpl
{
  using Encoder = symbol_encoder<ans_tablelog(AnsStreamType::Fp16)>;
  static constexpr uint32_t STATES_PER_LANE = 1;
  static constexpr int SYMBOLS_PER_LANE = 8;
  static constexpr int ROW_SYMBOLS = WARP_SIZE * SYMBOLS_PER_LANE;
  static constexpr int ROWS_PER_TILE = 1;
  static constexpr int TILE_SYMBOLS = ROWS_PER_TILE * ROW_SYMBOLS;
  struct Tile
  {
    uint4 raw;
  };

  static __device__ void load_tile(const uint8_t *raw_base, int tile_idx, int idx_in_warp, Tile &tile)
  {
    constexpr int TILE_RAW_BYTES = 2 * TILE_SYMBOLS;
    tile.raw =
      *reinterpret_cast<const uint4 *>(raw_base + tile_idx * TILE_RAW_BYTES + 2 * SYMBOLS_PER_LANE * idx_in_warp);
  }

  // Each uint32 word is a pair of adjacent values. One SHF rotates the whole word so both
  // symbol bytes land in bytes 1 and 3; the transform helpers extract the high/low symbol
  // and pack the four mantissa bytes of two words in symbol order.
  template <bool Detect>
  static __device__ __forceinline__ void
  encode_tile(Encoder &se, uint8_t *mantissas_base, int tile_idx, int idx_in_warp, const Tile &tile)
  {
    const uint32_t r0 = rotate_fp_pair_left1(tile.raw.x);
    const uint32_t r1 = rotate_fp_pair_left1(tile.raw.y);
    const uint32_t r2 = rotate_fp_pair_left1(tile.raw.z);
    const uint32_t r3 = rotate_fp_pair_left1(tile.raw.w);

    *reinterpret_cast<uint2 *>(mantissas_base + tile_idx * TILE_SYMBOLS + SYMBOLS_PER_LANE * idx_in_warp) =
      make_uint2(fp16_pack_mantissas_from_rotated_pairs(r0, r1), fp16_pack_mantissas_from_rotated_pairs(r2, r3));

    auto enc_rot = [&](uint32_t r) {
      se.template encode_symbol<false, Detect>(fp16_get_symbol_from_rotated<1>(r));
      se.template encode_symbol<false, Detect>(fp16_get_symbol_from_rotated<0>(r));
    };
    enc_rot(r3);
    enc_rot(r2);
    enc_rot(r1);
    enc_rot(r0);
  }

  // Same pair map as x2 (lane t owns window_base + 2t). One state, so the pair is two
  // encode_symbol calls (high then low). Partial is the last (short) window.
  template <bool Detect, bool Partial>
  static __device__ void remainder_iter(
    Encoder &se,
    uint8_t *mantissas_base,
    const uint8_t *raw_base,
    int idx_in_warp,
    int window_base,
    int window_n
  )
  {
    const uint16_t *fp16_base = reinterpret_cast<const uint16_t *>(raw_base);
    const int pair_ix = 2 * idx_in_warp;
    const int ix = window_base + pair_ix;
    const bool has_a = !Partial || pair_ix < window_n;
    const bool has_b = !Partial || (pair_ix + 1) < window_n;
    const uint16_t va = has_a ? fp16_base[ix] : 0;
    const uint16_t vb = has_b ? fp16_base[ix + 1] : va;
    const uint32_t r = has_a ? rotate_fp_pair_left1(va, vb) : 0u;
    se.template encode_symbol<Partial, Detect>(fp16_get_symbol_from_rotated<1>(r), has_b);
    if (has_b)
    {
      mantissas_base[ix + 1] = fp16_get_mantissa_from_rotated<1>(r);
    }
    se.template encode_symbol<Partial, Detect>(fp16_get_symbol_from_rotated<0>(r), has_a);
    if (has_a)
    {
      mantissas_base[ix] = fp16_get_mantissa_from_rotated<0>(r);
    }
  }
};

// Two-state fp8 (E4M3), the default. Even symbol indices ride state 0 and odd ones state
// 1, so one adjacent symbol pair is exactly one encode_pair and both states advance
// together - the mirror of symbol_decoder_dual_state::decode_pair. Two rows per tile so a
// tile is four coalesced LDG.64 per lane.
struct FP8X2EncodeImpl
{
  using Encoder = symbol_encoder_dual_state<ans_tablelog(AnsStreamType::Fp8)>;
  static constexpr uint32_t STATES_PER_LANE = 2;
  static constexpr int ROWS_PER_TILE = 2;

  static __device__ __forceinline__ void encode_first_pair(Encoder &se, uint32_t sym_odd, uint32_t sym_even)
  {
    se.encode_first_pair(sym_odd, sym_even);
  }

  static __device__ __forceinline__ void encode_pair(Encoder &se, uint32_t sym_odd, uint32_t sym_even)
  {
    se.encode_pair(sym_odd, sym_even);
  }

  static __device__ __forceinline__ void
  encode_pair_partial(Encoder &se, uint32_t sym_odd, uint32_t sym_even, bool has_odd, bool has_even)
  {
    se.encode_pair<false /*Detect*/, true /*Partial*/>(sym_odd, sym_even, has_odd, has_even);
  }
};

// Single-state fp8 (E4M3). One rANS chain per lane, so a symbol pair is two sequential
// encode_symbol calls in rANS reverse order (odd index first). One row per tile.
struct FP8X1EncodeImpl
{
  using Encoder = symbol_encoder<ans_tablelog(AnsStreamType::Fp8)>;
  static constexpr uint32_t STATES_PER_LANE = 1;
  static constexpr int ROWS_PER_TILE = 1;

  static __device__ __forceinline__ void encode_first_pair(Encoder &se, uint32_t sym_odd, uint32_t sym_even)
  {
    se.encode_first_symbol(sym_odd);
    se.encode_symbol(sym_even);
  }

  static __device__ __forceinline__ void encode_pair(Encoder &se, uint32_t sym_odd, uint32_t sym_even)
  {
    se.encode_symbol(sym_odd);
    se.encode_symbol(sym_even);
  }

  static __device__ __forceinline__ void
  encode_pair_partial(Encoder &se, uint32_t sym_odd, uint32_t sym_even, bool has_odd, bool has_even)
  {
    se.encode_symbol<true /*partial*/>(sym_odd, has_odd);
    se.encode_symbol<true /*partial*/>(sym_even, has_even);
  }
};

// Per-warp CHAR encode. rANS is LIFO, so the leftover 1..TILE_SYMBOLS-1 symbols go first
// as 64-symbol windows (short tail first, then full windows descending), then the full
// tiles high -> low, TILES_PER_ITER at a time -- the exact mirror of
// CharDecodePolicy::decode_body. The encoder ctor seeds state_ at the rANS lower bound, so
// the tile map needs no seeding peel.
template <typename Impl>
template <bool Detect>
inline __device__ void CharEncodePolicy<Impl>::encode_body(Encoder &se, EncodeChunkSmem &enc)
{
  static_assert(not Detect, "CharEncodePolicy does not support detect mode");

  constexpr int TILE_SYMBOLS = Impl::TILE_SYMBOLS;

  const int idx_in_warp = get_lane_id();
  const volatile WarpEncodeOff &off = enc.warp_enc[threadIdx.x / WARP_SIZE];
  const IndexT in_start_idx = off.in_start_idx;
  const int sub_chunk_size = off.sub_chunk_size;

  const uint8_t *raw_base = enc.uncomp_at(in_start_idx);

  const int full_tiles = sub_chunk_size / TILE_SYMBOLS;
  const int rem = sub_chunk_size - full_tiles * TILE_SYMBOLS;

  if (rem != 0)
  {
    constexpr int STEP = WARP_SIZE * 2;
    const int rem_base = full_tiles * TILE_SYMBOLS;
    const int nfull = rem / STEP;
    const int tail = rem - nfull * STEP;
    if (tail != 0)
    {
      Impl::template remainder_iter<Detect, true>(se, raw_base, rem_base + nfull * STEP, tail, idx_in_warp);
    }
    for (int j = nfull - 1; j >= 0; --j)
    {
      Impl::template remainder_iter<Detect, false>(se, raw_base, rem_base + j * STEP, STEP, idx_in_warp);
    }
  }

  // Tiles are visited high to low. A tile's symbol-to-lane map does not depend on any
  // other tile, so encoding UNROLL of them per iteration emits the exact same symbol
  // order -- and therefore the same bitstream -- as one per iteration. What it buys is
  // one bounds test and one loop-carried index per UNROLL tiles instead of per tile, and
  // UNROLL loads in flight over the first encode, replacing the depth-1 prefetch (whose
  // in-loop `tile_idx - 1 >= 0` test and tile register copy both cost more than they hid).
  constexpr int UNROLL = Impl::TILES_PER_ITER;
  int tile_idx = full_tiles - 1;
  for (; tile_idx >= UNROLL - 1; tile_idx -= UNROLL)
  {
    typename Impl::Tile tiles[UNROLL];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u)
    {
      Impl::load_tile(raw_base, tile_idx - u, idx_in_warp, tiles[u]);
    }
#pragma unroll
    for (int u = 0; u < UNROLL; ++u)
    {
      Impl::template encode_tile<Detect>(se, tiles[u]);
    }
  }
  // Odd leftover tile when UNROLL does not divide full_tiles.
  for (; tile_idx >= 0; --tile_idx)
  {
    typename Impl::Tile tile;
    Impl::load_tile(raw_base, tile_idx, idx_in_warp, tile);
    Impl::template encode_tile<Detect>(se, tile);
  }
}

// FP8 (E4M3) sub-chunk encoder
template <typename Impl>
template <bool Detect>
inline __device__ void FP8EncodePolicy<Impl>::encode_body(typename Impl::Encoder &se, EncodeChunkSmem &enc)
{
  static_assert(
    not Detect,
    "FP8EncodePolicy floors the whole packed_exp alphabet, so a sampled model cannot miss a symbol"
  );

  const int idx_in_warp = get_lane_id();
  const volatile WarpEncodeOff &off = enc.warp_enc[threadIdx.x / WARP_SIZE];
  const IndexT in_start_idx = off.in_start_idx;
  const int sub_chunk_size = off.sub_chunk_size;
  const uint32_t mantissas_offset = off.mantissas_offset;
  // Raw bytes for this sub-chunk's first symbol (2 raw bytes per pair).
  const uint32_t raw_off = INPUT_BYTES_PER_SYMBOL * in_start_idx;

  const int full_tiles = sub_chunk_size / TILE_SYMBOLS;
  const int rem = sub_chunk_size - full_tiles * TILE_SYMBOLS;

  if (rem != 0)
  {
    // During each remainder iteration, each lane encodes two symbols at a time.
    constexpr int STEP = WARP_SIZE * 2;
    const int rem_base = full_tiles * TILE_SYMBOLS;
    const int nfull = rem / STEP;
    const int tail = rem - nfull * STEP;
    if (tail != 0)
    {
      remainder_iter<true>(
        se,
        enc.comp_at(mantissas_offset),
        enc.uncomp_at(raw_off),
        idx_in_warp,
        rem_base + nfull * STEP,
        tail
      );
    }
    for (int j = nfull - 1; j >= 0; --j)
    {
      remainder_iter<false>(
        se,
        enc.comp_at(mantissas_offset),
        enc.uncomp_at(raw_off),
        idx_in_warp,
        rem_base + j * STEP,
        STEP
      );
    }
  }

  int tile_idx = full_tiles - 1;
  if (rem == 0 && full_tiles > 0)
  {
    Tile cur;
    load_tile(enc.uncomp_at(raw_off), tile_idx, idx_in_warp, cur);
    encode_tile<true /*Seed*/>(se, enc.comp_at(mantissas_offset), tile_idx, idx_in_warp, cur);
    --tile_idx;
  }
  for (; tile_idx >= 0; --tile_idx)
  {
    Tile cur;
    load_tile(enc.uncomp_at(raw_off), tile_idx, idx_in_warp, cur);
    encode_tile<false /*Seed*/>(se, enc.comp_at(mantissas_offset), tile_idx, idx_in_warp, cur);
  }
}

// rANS reverse: leftover 1..TILE_SYMBOLS-1 as 64-symbol remainder iters (last short
// window first), then full tiles high -> low. Each tile is loaded just before encode
// so the next-tile register window does not stay live across encode_pair (occupancy).
template <typename Impl>
template <bool Detect>
inline __device__ void FP16EncodePolicy<Impl>::encode_body(typename Impl::Encoder &se, EncodeChunkSmem &enc)
{
  constexpr int TILE_SYMBOLS = Impl::TILE_SYMBOLS;

  const int idx_in_warp = get_lane_id();
  const volatile WarpEncodeOff &off = enc.warp_enc[threadIdx.x / WARP_SIZE];
  const IndexT in_start_idx = off.in_start_idx;
  const int sub_chunk_size = off.sub_chunk_size;
  const uint32_t mantissas_offset = off.mantissas_offset;
  const uint32_t raw_off = 2u * in_start_idx;

  const int full_tiles = sub_chunk_size / TILE_SYMBOLS;
  const int rem = sub_chunk_size - full_tiles * TILE_SYMBOLS;

  // Sampled-histogram miss detection, checked after the remainder peels and after every
  // tile. An uncovered symbol forces the whole chunk to be re-encoded from an exact
  // histogram; this caps wasted work at one window. Warp-uniform, so all lanes leave
  // together; the abandoned partial stream is overwritten by the retry, and
  // encode_sub_chunk's closing uncovered_any() still reports the miss.
  auto detect_bail = [&]() {
    if constexpr (Detect)
    {
      return se.uncovered_any();
    }
    else
    {
      return false;
    }
  };

  if (rem != 0)
  {
    constexpr int STEP = WARP_SIZE * 2;
    const int rem_base = full_tiles * TILE_SYMBOLS;
    const int nfull = rem / STEP;
    const int tail = rem - nfull * STEP;
    if (tail != 0)
    {
      Impl::template remainder_iter<Detect, true>(
        se,
        enc.comp_at(mantissas_offset),
        enc.uncomp_at(raw_off),
        idx_in_warp,
        rem_base + nfull * STEP,
        tail
      );
    }
    for (int j = nfull - 1; j >= 0; --j)
    {
      Impl::template remainder_iter<Detect, false>(
        se,
        enc.comp_at(mantissas_offset),
        enc.uncomp_at(raw_off),
        idx_in_warp,
        rem_base + j * STEP,
        STEP
      );
    }

    if (detect_bail())
    {
      return;
    }
  }

  for (int tile_idx = full_tiles - 1; tile_idx >= 0; --tile_idx)
  {
    typename Impl::Tile cur;
    Impl::load_tile(enc.uncomp_at(raw_off), tile_idx, idx_in_warp, cur);
    Impl::template encode_tile<Detect>(se, enc.comp_at(mantissas_offset), tile_idx, idx_in_warp, cur);
    if (detect_bail())
    {
      return;
    }
  }
}

// Pack the low three bytes of four rotated FP32 values into one lane's three-word
// mantissa record.
inline __device__ uint3
pack_fp32_mantissa_record(uint32_t rotated0, uint32_t rotated1, uint32_t rotated2, uint32_t rotated3)
{
  return make_uint3(
    __byte_perm(rotated0, rotated1, 0x4210u),
    __byte_perm(rotated1, rotated2, 0x5421u),
    __byte_perm(rotated2, rotated3, 0x6542u)
  );
}

// FP32 encode mirrors FP32DecodePolicy::decode_body. The mapped partial tile is
// emitted first, followed by full tiles from high to low because rANS is LIFO.
template <typename Impl>
template <bool Detect>
inline __device__ void FP32EncodePolicy<Impl>::encode_body(Encoder &se, EncodeChunkSmem &enc)
{
  constexpr int LANE_SYMBOLS_PER_GROUP = FP32Tile::LANE_SYMBOLS_PER_GROUP;
  constexpr int GROUPS_PER_TILE = FP32Tile::GROUPS_PER_TILE;
  constexpr int TILE_SYMBOLS = FP32Tile::TILE_SYMBOLS;

  const int lane_id = get_lane_id();
  const volatile WarpEncodeOff &off = enc.warp_enc[threadIdx.x / WARP_SIZE];
  const IndexT in_start_idx = off.in_start_idx;
  const int sub_chunk_size = off.sub_chunk_size;
  const uint32_t mantissas_offset = off.mantissas_offset;
  const uint32_t raw_offset = static_cast<uint32_t>(INPUT_BYTES_PER_SYMBOL) * in_start_idx;

  const int full_tiles = sub_chunk_size / TILE_SYMBOLS;
  const int remainder_base = full_tiles * TILE_SYMBOLS;
  const int remainder = sub_chunk_size - remainder_base;

  auto detect_bail = [&]() {
    if constexpr (Detect)
    {
      return se.uncovered_any();
    }
    else
    {
      return false;
    }
  };

  if (remainder != 0)
  {
    const uint32_t *const input = reinterpret_cast<const uint32_t *>(enc.uncomp_at(raw_offset)) + remainder_base;
    uint8_t *const mantissas = enc.comp_at(mantissas_offset) + MANTISSA_BYTES_PER_SYMBOL * remainder_base;

    auto rotate_active = [&](int value_offset, bool active) {
      if (!active)
      {
        return 0u;
      }
      const uint32_t rotated = rotate_fp32_left1(input[value_offset]);
      uint8_t *const destination = mantissas + MANTISSA_BYTES_PER_SYMBOL * value_offset;
      destination[0] = fp32_get_mantissa_from_rotated<0>(rotated);
      destination[1] = fp32_get_mantissa_from_rotated<1>(rotated);
      destination[2] = fp32_get_mantissa_from_rotated<2>(rotated);
      return rotated;
    };

#pragma unroll
    for (int pair = FP32Tile::LANE_PAIRS_PER_TILE - 1; pair >= 0; --pair)
    {
      const int even_offset = FP32Tile::lane_pair_value_offset(pair, lane_id);
      const int odd_offset = even_offset + 1;
      const bool even_active = even_offset < remainder;
      const bool odd_active = odd_offset < remainder;

      const uint32_t even_rotated = rotate_active(even_offset, even_active);
      const uint32_t odd_rotated = rotate_active(odd_offset, odd_active);
      Impl::template encode_remainder_pair<Detect>(se, odd_rotated, even_rotated, odd_active, even_active);
    }

    if (detect_bail())
    {
      return;
    }
  }

  if (full_tiles == 0)
  {
    return;
  }

  struct Tile
  {
    uint4 groups[GROUPS_PER_TILE];
  };

  const uint4 *const tile_input = reinterpret_cast<const uint4 *>(enc.uncomp_at(raw_offset));
  uint8_t *const mantissas = enc.comp_at(mantissas_offset);

  auto load_tile = [&](int tile, Tile &destination) {
    const int lane_word = tile * (TILE_SYMBOLS / LANE_SYMBOLS_PER_GROUP) + lane_id;
#pragma unroll
    for (int group = 0; group < GROUPS_PER_TILE; ++group)
    {
      destination.groups[group] = tile_input[lane_word + group * (FP32Tile::GROUP_SYMBOLS / LANE_SYMBOLS_PER_GROUP)];
    }
  };

  auto encode_tile = [&](int tile, const Tile &current) {
    uint32_t rotated[GROUPS_PER_TILE][LANE_SYMBOLS_PER_GROUP];
    uint3 records[GROUPS_PER_TILE];
#pragma unroll
    for (int group = 0; group < GROUPS_PER_TILE; ++group)
    {
      rotated[group][0] = rotate_fp32_left1(current.groups[group].x);
      rotated[group][1] = rotate_fp32_left1(current.groups[group].y);
      rotated[group][2] = rotate_fp32_left1(current.groups[group].z);
      rotated[group][3] = rotate_fp32_left1(current.groups[group].w);
      records[group] =
        pack_fp32_mantissa_record(rotated[group][0], rotated[group][1], rotated[group][2], rotated[group][3]);
    }

#pragma unroll
    for (int segment = 0; segment < FP32Tile::LANE_SEGMENTS_PER_GROUP; ++segment)
    {
      FP32Tile::Segment output;
#pragma unroll
      for (int group = 0; group < GROUPS_PER_TILE; ++group)
      {
        output.values[group] = segment == 0 ? records[group].x : segment == 1 ? records[group].y : records[group].z;
      }
      *reinterpret_cast<FP32Tile::Segment *>(mantissas + FP32Tile::lane_segment_offset(tile, segment, lane_id)) =
        output;
    }

#pragma unroll
    for (int group = GROUPS_PER_TILE - 1; group >= 0; --group)
    {
      Impl::template encode_group<Detect>(se, rotated[group][0], rotated[group][1], rotated[group][2], rotated[group][3]);
    }
  };

  int tile = full_tiles - 1;
  Tile current;
  Tile next;
  load_tile(tile, current);
  for (;;)
  {
    if (tile > 0)
    {
      load_tile(tile - 1, next);
    }
    encode_tile(tile, current);
    if (detect_bail())
    {
      return;
    }
    if (tile == 0)
    {
      break;
    }
    current = next;
    --tile;
  }
}
} // namespace detail
} // namespace ans_gpu_lib
