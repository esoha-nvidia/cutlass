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

#include "common_utils.hpp" // nvcomp::bytesUntilAlignmentBoundary

#include <ans/ans_utils.cuh>
#include <ans/symbol_decoder.cuh>
#include <ans/symbol_encoder.cuh>

namespace ans_gpu_lib
{
namespace detail
{

// Per-sub-chunk encode context (defined in compress_kernels.cuh).
struct SubChunkEncodeCtx;

// Compress-side policies (char / fp16 / fp8): type-specific histogram + encode
// hooks for compute_histogram / histogram_warp_impl / encode_sub_chunk.
// Shared defaults live in EncodePolicyBase; each policy overrides what differs.
template <typename Derived>
struct EncodePolicyBase
{
  // uint2 (LDG.64) loads; pipeline depth 8 keeps ~64 bytes/lane in flight.
  using LoadT = uint2;
  static constexpr int LOADS_PER_ITER = 8;

  static __device__ IndexT num_symbols(IndexT bytes) { return bytes / Derived::INPUT_BYTES_PER_SYMBOL; }
  static __device__ IndexT input_byte_offset(IndexT in_start) { return in_start * Derived::INPUT_BYTES_PER_SYMBOL; }

  // Uncompressed size recorded in the sub-chunk header. fp8 overrides (raw byte count).
  static __device__ IndexT header_uncomp_size(IndexT bytes) { return Derived::num_symbols(bytes); }

  // Side-band base for this warp's first symbol (unused for char).
  static __device__ uint8_t *histogram_sideband_warp_base(uint8_t *comp_chunk, IndexT in_start)
  {
    return comp_chunk + sizeof(size_t) + in_start * Derived::SIDEBAND_BYTES_PER_SYMBOL;
  }

  // Size prefix + 8-byte-aligned ANS stream start. fp8 overrides (flag + odd byte).
  static __device__ uint8_t *histogram_write_chunk_prefix(uint8_t *comp_chunk, IndexT bytes)
  {
    const IndexT sideband_bytes = Derived::num_symbols(bytes) * Derived::SIDEBAND_BYTES_PER_SYMBOL;
    // The prefix itself is a format-fixed 64-bit field, so widen for the store.
    const size_t info_size = nvcomp::roundUpTo(sideband_bytes + sizeof(size_t), sizeof(size_t));
    *reinterpret_cast<size_t *>(comp_chunk) = info_size;
    return comp_chunk + info_size;
  }

  // Trailing chunk bytes not covered by the ANS stream (tid==0). Only fp8 needs one.
  static __device__ void write_chunk_tail(uint8_t * /*comp*/, const void * /*uncomp*/, IndexT /*bytes*/) {}
};

// Char: no side-band; size prefix is sizeof(size_t).
struct CharEncodePolicy : EncodePolicyBase<CharEncodePolicy>
{
  static constexpr int PARTITION_ALIGN = 16; // symbols
  static constexpr bool HAS_PEEL = true;
  static constexpr int INPUT_BYTES_PER_SYMBOL = 1;
  static constexpr int SIDEBAND_BYTES_PER_SYMBOL = 0;

  // Encoder async-copy staging window, in symbols (== bytes; char is 1 byte/symbol).
  // Single-buffered: each outer iteration cp.async-stages this many bytes into the
  // per-warp buffer, then encodes them. Tunable; must be a multiple of
  // WARP_SIZE * sizeof(uchar4) so the coalesced uchar4 loads and the WARP_SIZE-wide row
  // readback divide evenly.
  static constexpr int ENC_WINDOW_SYMBOLS = 512; // matches the FP16 encoder's 512-symbol window
  static_assert(
    ENC_WINDOW_SYMBOLS % (WARP_SIZE * static_cast<int>(sizeof(uchar4))) == 0,
    "CharEncodePolicy::ENC_WINDOW_SYMBOLS must be a multiple of WARP_SIZE * sizeof(uchar4)"
  );

  // Single-buffered, so the per-warp staging allocation is exactly one window.
  static constexpr int ENC_BUF_BYTES = ENC_WINDOW_SYMBOLS;

  // Peel to the next uint2 boundary; returns peeled byte count.
  static __device__ int
  histogram_peel_head_warp(int idx_in_warp, uint32_t *counts, const uint8_t *ip, int size, uint8_t * /*sideband*/)
  {
    const int alignment_rem = static_cast<int>(nvcomp::bytesUntilAlignmentBoundary(ip, sizeof(uint2)));
    if (idx_in_warp < min(alignment_rem, size))
    {
      atomicAdd(&counts[ip[idx_in_warp]], 1);
    }
    return alignment_rem;
  }

  // One uint2 = 8 symbols. Union avoids PRMT swizzles from a direct uchar4 load.
  template <bool /*FIX_ALIGNMENT*/>
  static __device__ void histogram_process_word(uint2 reg, uint32_t *counts, uint8_t * /*sideband_dst*/)
  {
    union reg_val
    {
      uint2 reg;
      uchar4 bytes[2];
    } rv;
    rv.reg = reg;
#pragma unroll
    for (int ix = 0; ix < 2; ++ix)
    {
      atomicAdd(&counts[rv.bytes[ix].x], 1);
      atomicAdd(&counts[rv.bytes[ix].y], 1);
      atomicAdd(&counts[rv.bytes[ix].z], 1);
      atomicAdd(&counts[rv.bytes[ix].w], 1);
    }
  }

  static __device__ void histogram_process_scalar_tail_warp(
    uint32_t idx_in_warp,
    uint32_t *counts,
    const uint8_t *tail_in,
    uint8_t * /*tail_sideband*/,
    int tail_size
  )
  {
    if (static_cast<int>(idx_in_warp) < tail_size)
    {
      atomicAdd(&counts[tail_in[idx_in_warp]], 1);
    }
  }

  // ---- encode phase ----

  // Minimal input reader for the char encoder: a raw byte pointer the shared rANS
  // prologue can seek/retreat and read per lane. fp16 uses Fp16EncodePolicy::InputReader,
  // which exposes the same seek_forward / retreat / sym_at_lane surface.
  struct ByteReader
  {
    const uint8_t *ip_;
    int tid_;
    __device__ void seek_forward(int symbols) { ip_ += symbols; }
    __device__ void retreat(int symbols) { ip_ -= symbols; }
    __device__ uint8_t sym_at_lane() const { return ip_[tid_]; }
  };

  // Staging policy driving staged_encode_pipeline. VecT-wide cp.async (VecT = uint4 for
  // 16-byte transfers, or uchar4 for the 4-byte fallback) and byte-identity readback.
  // Defined out-of-line in compress_kernels.cuh.
  template <typename VecT>
  struct StagePolicy;

  // Stream nloop full ENC_WINDOW_SYMBOLS windows through the staged cp.async pipeline,
  // head -> low. Picks the uint4 / uchar4 / scalar path from window_base's alignment.
  // Defined out-of-line in compress_kernels.cuh.
  static __device__ void encode_symbols_async_pipeline(
    const uint8_t *input_buf,
    uint8_t *tmp_input_buf,
    const uint8_t *window_base,
    int nloop,
    symbol_encoder &se
  );

  static __device__ void
  encode_body(symbol_encoder &se, const SubChunkEncodeCtx &ctx, const void *uncomp_chunk, uint8_t *input_buf);
};

// FP16: exponent histogrammed, mantissa in raw side-band.
struct Fp16EncodePolicy : EncodePolicyBase<Fp16EncodePolicy>
{
  static constexpr int PARTITION_ALIGN = 8; // symbols
  // Input is always >= 8-byte aligned; no peel; mantissa store is STG.32/word.
  static constexpr bool HAS_PEEL = false;
  static constexpr int INPUT_BYTES_PER_SYMBOL = 2;
  static constexpr int SIDEBAND_BYTES_PER_SYMBOL = 1;

  // Encoder async-copy staging window (bytes/warp per outer iter). Tunable; must be a
  // multiple of WARP_SIZE * sizeof(uint4) so the uint4 and uint2 staging loads and the
  // WARP_SIZE-wide row readback divide evenly.
  static constexpr int ENC_WINDOW_BYTES = 1024;
  static_assert(
    ENC_WINDOW_BYTES % (WARP_SIZE * static_cast<int>(sizeof(uint4))) == 0,
    "Fp16EncodePolicy::ENC_WINDOW_BYTES must be a multiple of WARP_SIZE * sizeof(uint4)"
  );
  static constexpr int ENC_WINDOW_SYMBOLS = ENC_WINDOW_BYTES / static_cast<int>(sizeof(uint16_t));
  static constexpr int ENC_ROWS = ENC_WINDOW_SYMBOLS / WARP_SIZE; // warp rows per staged window

  // Per-warp staging allocation. Larger than the window: the aligned-down over-read
  // reads from src-8 when the source is 8-aligned-but-not-16-aligned, so extra 16-byte
  // headroom is appended to keep [src, src + ENC_WINDOW_BYTES) fully covered.
  static constexpr int ENC_BUF_BYTES = ENC_WINDOW_BYTES + 32;

  static __host__ __device__ uint8_t get_exp(uint16_t fp16_val) { return static_cast<uint8_t>(fp16_val >> 8); }

  // Split one fp16 into the histogrammed exponent byte and the raw mantissa side-band byte.
  static __host__ __device__ void split_float(uint16_t float_input, uint8_t &exponent, uint8_t &mantissa)
  {
    exponent = get_exp(float_input);
    mantissa = static_cast<uint8_t>(float_input);
  }

  // One uint2 = 4 fp16. Histograms the 4 exponents and streams the 4 mantissa bytes as
  // one STG.32. sideband_dst is always 4-byte aligned (per-warp base is 8-byte aligned,
  // per-word offset a multiple of 4), so no dst-alignment split is needed.
  template <bool /*FIX_ALIGNMENT*/>
  static __device__ void histogram_process_word(uint2 reg, uint32_t *counts, uint8_t *sideband_dst)
  {
    // reg.x bytes: [0]=m0, [1]=e0, [2]=m1, [3]=e1
    // reg.y bytes: [4]=m2, [5]=e2, [6]=m3, [7]=e3
    uint32_t packed_exp = __byte_perm(reg.x, reg.y, 0x7531);

    uint32_t e0 = packed_exp & 0xFF;
    uint32_t e1 = __byte_perm(packed_exp, 0, 0x4441);
    uint32_t e2 = __byte_perm(packed_exp, 0, 0x4442);
    uint32_t e3 = packed_exp >> 24;

    atomicAdd(&counts[e0], 1);
    atomicAdd(&counts[e1], 1);
    atomicAdd(&counts[e2], 1);
    atomicAdd(&counts[e3], 1);

    // Stream the 4 mantissa bytes (low byte of each fp16): m0 | m1<<8 | m2<<16 | m3<<24.
    uint32_t packed_man = __byte_perm(reg.x, reg.y, 0x6420);
    *reinterpret_cast<uint32_t *>(sideband_dst) = packed_man;
  }

  static __device__ void histogram_process_scalar_tail_warp(
    uint32_t idx_in_warp,
    uint32_t *counts,
    const uint8_t *tail_in,
    uint8_t *tail_sideband,
    int tail_size
  )
  {
    if (static_cast<int>(idx_in_warp) < tail_size)
    {
      uint8_t exponent, mantissa;
      split_float(reinterpret_cast<const uint16_t *>(tail_in)[idx_in_warp], exponent, mantissa);
      atomicAdd(&counts[exponent], 1);
      tail_sideband[idx_in_warp] = mantissa;
    }
  }

  // ---- encode phase (defined out-of-line in compress_kernels.cuh) ----

  // Staging policy driving staged_encode_pipeline: interior windows use the uint4
  // over-read, the lowest window uses exact uint2 when 8-off.
  template <int SHIFT_BYTES>
  struct StagePolicy;

  // Owns the per-warp input read pointer and staging buffer for one sub-chunk encode.
  class InputReader;

  // Read one staged window back as this lane's ENC_ROWS exponent bytes. shift_sym
  // compensates the aligned-down over-read: symbol k is staged at buf16[k + shift_sym].
  static __device__ void
  extract_exp_batch(const uint16_t *__restrict__ buf16, int tid, int shift_sym, uint8_t syms[ENC_ROWS]);

  // Stage one ENC_WINDOW_BYTES window into smem with 16-byte (uint4) cp.async,
  // rounding the source down to a 16-byte boundary (src - SHIFT_BYTES) so the copies are
  // aligned even when src is only 8-aligned. SHIFT_BYTES is 0 or 8.
  //  - SHIFT_BYTES == 0: exact uint4
  //  - SHIFT_BYTES == 8: aligned-down over-read with one extra read
  template <int SHIFT_BYTES>
  static __device__ void stage_window_overread(uint8_t *tmp_input_buf, const uint8_t *src, int tid);

  // 8-byte staging: reads exactly [src, src + ENC_WINDOW_BYTES); symbol k lands at buf16[k].
  static __device__ void stage_window_u2(uint8_t *tmp_input_buf, const uint8_t *src, int tid);

  static __device__ void
  encode_body(symbol_encoder &se, const SubChunkEncodeCtx &ctx, const void *uncomp_chunk, uint8_t *input_buf);
};

// FP8: packed_exp histogrammed; packed_signs_mantissas side-band; overrides prefix + tail handling.
struct Fp8EncodePolicy : EncodePolicyBase<Fp8EncodePolicy>
{
  static constexpr int PARTITION_ALIGN = 8; // pairs
  static constexpr bool HAS_PEEL = false; // input is always uint2-aligned
  static constexpr int INPUT_BYTES_PER_SYMBOL = 2; // raw bytes per pair
  static constexpr int SIDEBAND_BYTES_PER_SYMBOL = 1;

  // Encode-tile geometry. The block-exp load requires each lane to own a contiguous
  // run of symbols within a tile, which fixes the tile size; these are format-critical,
  // not tunable. The per-warp staging buffer is a depth-2 ping-pong of one raw tile
  // (tile T-1's cp.async overlaps tile T's encode), so it is fully derived from them.
  static constexpr int ENC_SYMBOLS_PER_LANE = 8; // contiguous symbols per lane per tile
  static constexpr int ENC_TILE_SYMBOLS = WARP_SIZE * ENC_SYMBOLS_PER_LANE; // 256
  static constexpr int ENC_TILE_RAW_BYTES = INPUT_BYTES_PER_SYMBOL * ENC_TILE_SYMBOLS; // 512
  static constexpr int ENC_BUF_BYTES = 2 * ENC_TILE_RAW_BYTES; // depth-2 ping-pong

  // fp8 stores the raw byte count N; decode derives P = N/2 and the odd-ness.
  static __device__ IndexT header_uncomp_size(IndexT bytes) { return bytes; }

  // E4M3 layout per byte: [sign:1 | exp:4 | mantissa:3].
  static __host__ __device__ uint8_t get_exp(uint8_t b) { return (b >> 3) & 0x0F; }
  static __host__ __device__ uint8_t get_signs_mantissas(uint8_t b) { return ((b & 0x80) >> 4) | (b & 0x07); }

  // Pack an adjacent FP8 pair into one packed_exp byte and one packed_signs_mantissas
  // byte (low nibble from b0, high nibble from b1). Scalar form, used by tails.
  static __host__ __device__ void
  split_pair(uint8_t b0, uint8_t b1, uint8_t &packed_exp, uint8_t &packed_signs_mantissas)
  {
    packed_exp = static_cast<uint8_t>(get_exp(b0) | (get_exp(b1) << 4));
    packed_signs_mantissas = static_cast<uint8_t>(get_signs_mantissas(b0) | (get_signs_mantissas(b1) << 4));
  }

  // Split one uint32 = 4 raw FP8 bytes (b0,b1,b2,b3) = 2 adjacent pairs.
  // Returns the 2 packed_exp bytes in packed_exp_lo16 (byte0=pair0, byte1=pair1)
  // and the 2 packed_signs_mantissas bytes in packed_signs_mantissas_lo16. The 4
  // bytes are processed one per byte lane with whole-word masks/shifts (the ALU does
  // 4 lanes' work per instruction instead of ~24 byte-granularity ops), then the two
  // result bytes are gathered with a single __byte_perm (PRMT). This removes the
  // per-byte ALU pressure that dominates the FP8 histogram (math-pipe throttle).
  // Bit layout mirrors Fp8EncodePolicy::get_exp / get_signs_mantissas.
  static __device__ void
  split_fp8_quad_e4m3(uint32_t w, uint32_t &packed_exp_lo16, uint32_t &packed_signs_mantissas_lo16)
  {
    // Per-byte exponent nibble: (b >> 3) & 0xF, vectorized across the 4 bytes.
    const uint32_t exp4 = (w >> 3) & 0x0F0F0F0Fu;
    // Per-byte sign+mantissa nibble: ((b & 0x80) >> 4) | (b & 0x07).
    const uint32_t signs_mantissas4 = ((w & 0x80808080u) >> 4) | (w & 0x07070707u);

    // Pack adjacent nibbles: byte0 = lo(byteK) | (hi from byteK+1). exp4|exp4>>4
    // leaves pair0's packed byte in byte0 and pair1's in byte2; gather both into
    // the low 16 bits with one PRMT (select source bytes 0 and 2).
    const uint32_t exp_packed = exp4 | (exp4 >> 4);
    const uint32_t signs_mantissas_packed = signs_mantissas4 | (signs_mantissas4 >> 4);
    packed_exp_lo16 = __byte_perm(exp_packed, 0u, 0x4420u); // bytes {0,2} -> {0,1}
    packed_signs_mantissas_lo16 = __byte_perm(signs_mantissas_packed, 0u, 0x4420u);
  }

  // Exp-only variant of split_fp8_quad_e4m3 for the fp8 encoder, which recomputes
  // packed_exp from the raw input but does NOT need the sign/mantissa side-band (that
  // was already written by the histogram). Returns the 2 packed_exp bytes of the 2
  // pairs in the low 16 bits (byte0 = pair0, byte1 = pair1), skipping the
  // signs_mantissas work.
  static __device__ uint32_t fp8_quad_packed_exp_e4m3(uint32_t w)
  {
    const uint32_t exp4 = (w >> 3) & 0x0F0F0F0Fu;
    const uint32_t exp_packed = exp4 | (exp4 >> 4);
    return __byte_perm(exp_packed, 0u, 0x4420u); // bytes {0,2} -> {0,1}
  }

  // Prefix includes FP8 flag; accounts for optional odd trailing byte.
  static __device__ uint8_t *histogram_write_chunk_prefix(uint8_t *comp_chunk, IndexT bytes)
  {
    const IndexT num_pairs = bytes / 2;
    const IndexT tail_byte = bytes & 1;
    // ANS_MANTISSAS_SIZE_FP8_FLAG lives in bit 63, so the prefix must stay 64-bit.
    const size_t info_size = nvcomp::roundUpTo(num_pairs + tail_byte + sizeof(size_t), sizeof(size_t));
    *reinterpret_cast<size_t *>(comp_chunk) = info_size | ANS_MANTISSAS_SIZE_FP8_FLAG;
    return comp_chunk + info_size;
  }

  // One uint2 = 8 raw FP8 bytes = 4 pairs: histogram the 4 packed_exp symbols and
  // emit ONE packed_signs_mantissas word (4 bytes).
  //
  // The packed_exp bytes are NOT stored: the encoder re-reads the raw FP8 input and
  // re-splits on the fly (mirroring the fp16 encoder), so Phase 1 only needs the
  // histogram counts and the raw side-band. Only the signs_mantissas side-band is
  // persisted, and lane L owns side-band word L so the warp's 32 STG.32 are fully
  // coalesced (one 128 B sector per store). sideband_dst is 4 B aligned.
  template <bool /*FIX_ALIGNMENT*/>
  static __device__ void histogram_process_word(uint2 reg, uint32_t *counts, uint8_t *sideband_dst)
  {
    // Split each of the 2 uint32 words (2 pairs each) into 2 packed_exp and 2
    // packed_signs_mantissas bytes (low 16 bits), then assemble the side-band word.
    uint32_t exp_lo16[2];
    uint32_t signs_mantissas_lo16[2];
    split_fp8_quad_e4m3(reg.x, exp_lo16[0], signs_mantissas_lo16[0]);
    split_fp8_quad_e4m3(reg.y, exp_lo16[1], signs_mantissas_lo16[1]);

    // word.x -> bytes {0,1}, word.y -> bytes {2,3}.
    uint32_t signs_mantissas_word = signs_mantissas_lo16[0] | (signs_mantissas_lo16[1] << 16);

    // Histogram the 4 packed_exp symbols (low + high byte of each word's lo16).
    atomicAdd(&counts[exp_lo16[0] & 0xFFu], 1);
    atomicAdd(&counts[(exp_lo16[0] >> 8) & 0xFFu], 1);
    atomicAdd(&counts[exp_lo16[1] & 0xFFu], 1);
    atomicAdd(&counts[(exp_lo16[1] >> 8) & 0xFFu], 1);

    *reinterpret_cast<uint32_t *>(sideband_dst) = signs_mantissas_word;
  }

  static __device__ void histogram_process_scalar_tail_warp(
    uint32_t idx_in_warp,
    uint32_t *counts,
    const uint8_t *tail_in,
    uint8_t *tail_sideband,
    int tail_size
  )
  {
    if (static_cast<int>(idx_in_warp) < tail_size)
    {
      const uint8_t b0 = tail_in[2 * idx_in_warp + 0];
      const uint8_t b1 = tail_in[2 * idx_in_warp + 1];
      uint8_t packed_exp, packed_signs_mantissas;
      split_pair(b0, b1, packed_exp, packed_signs_mantissas);
      tail_sideband[idx_in_warp] = packed_signs_mantissas;
      atomicAdd(&counts[packed_exp], 1);
    }
  }

  // Odd-N trailing raw byte after the packed_signs_mantissas side-band (tid==0).
  static __device__ void write_chunk_tail(uint8_t *comp_chunk, const void *uncomp_chunk, IndexT bytes)
  {
    const IndexT num_pairs = bytes / 2;
    const IndexT tail_byte = bytes & 1;
    if (tail_byte)
    {
      (comp_chunk + sizeof(size_t))[num_pairs] = reinterpret_cast<const uint8_t *>(uncomp_chunk)[bytes - 1];
    }
  }

  // ---- encode phase (defined out-of-line in compress_kernels.cuh) ----

  // Async-copy one tile's RAW FP8 input from gmem into a per-warp staging slot.
  static __device__ void prefetch_tile_raw(uint8_t *stage, const uint8_t *src, int lane_id);

  // packed_exp of one symbol, read straight from the raw FP8 input. Symbol s (a pair) is
  // the two raw bytes raw_base[2*s], raw_base[2*s+1]; their exponent nibbles are packed
  // into one packed_exp byte (exp-only; side-band was already written by the histogram).
  // Used by the (rare) remainder.
  static __device__ uint8_t packed_exp_at(const uint8_t *raw_base, int sym_ix);

  static __device__ void
  encode_body(symbol_encoder &se, const SubChunkEncodeCtx &ctx, const void *uncomp_chunk, uint8_t *input_buf);
};

// ===========================================================================
// Decode policies, one per output data type. Each owns its compile-time knobs and
// the type-specific decode logic; decode_sub_chunk is templated on the policy.

// 8-bit symbols: the decoded symbol is the output byte.
// Stateless

struct CharDecodePolicy
{
  using out_t = uint8_t;
  static constexpr int NUM_SYMBOLS_PER_THREAD_META_ITER = NUM_CHAR_SYMBOLS_PER_THREAD_META_ITER;
  static constexpr int NUM_SYMBOLS_PER_WARP_META_ITER = NUM_CHAR_SYMBOLS_PER_WARP_META_ITER;
  static constexpr int OUT_TYPE_SIZE = sizeof(out_t);

  template <bool BOUNDS_CHECK>
  __device__ void store_one(symbol_decoder<BOUNDS_CHECK> &sd, out_t *out, int ix)
  {
    out[ix] = sd.decode_symbol_full();
  }

  // Store an already-decoded symbol (used by the guarded final partial row).
  __device__ void store_symbol(out_t *out, int ix, uint8_t symbol) { out[ix] = symbol; }

  template <bool BOUNDS_CHECK>
  __device__ void prefetch_initial(symbol_decoder<BOUNDS_CHECK> &sd, uint16_t *warp_renorm_buf, int /*main_iters*/)
  {
    sd.prefetch_renorm_buffer(warp_renorm_buf, true);
  }

  template <bool BOUNDS_CHECK>
  __device__ void decode_main_iter(
    symbol_decoder<BOUNDS_CHECK> &sd,
    out_t *out,
    uint16_t *warp_renorm_buf,
    int /*ix_main_iter*/,
    int /*main_iters*/,
    int ix_decode
  )
  {
    sd.refill_if_needed(warp_renorm_buf, true);

#pragma unroll
    for (int k = 0; k < NUM_SYMBOLS_PER_THREAD_META_ITER; ++k)
    {
      out[ix_decode + k * WARP_SIZE] = sd.decode_symbol_full();
    }
  }
};

// 16-bit fp: output uint16 = (mantissa byte | symbol << 8). Mantissa bytes are
// staged from global via a depth-2 double buffer so iter i+1's LDGSTS overlaps
// iter i's consume; the main-loop iter runs refill -> wait -> prefetch next stage ->
// consume current stage (LDS mantissa + decode + byte_perm). Holds the per-warp
// staging pointers and lane id so callers don't thread them through every method.
struct Fp16DecodePolicy
{
  using out_t = uint16_t;
  static constexpr int NUM_SYMBOLS_PER_THREAD_META_ITER = NUM_FP16_SYMBOLS_PER_THREAD_META_ITER;
  static constexpr int NUM_SYMBOLS_PER_WARP_META_ITER = NUM_FP16_SYMBOLS_PER_WARP_META_ITER;
  static constexpr int OUT_TYPE_SIZE = sizeof(out_t);

  uint8_t *warp_mantissa0_;
  const uint8_t *mantissas_;
  int lane_id_;

  __device__ Fp16DecodePolicy(uint8_t *warp_mantissa0, const uint8_t *mantissas, int lane_id)
      : warp_mantissa0_{warp_mantissa0}
      , mantissas_{mantissas}
      , lane_id_{lane_id}
  {}

  template <bool BOUNDS_CHECK>
  __device__ void store_one(symbol_decoder<BOUNDS_CHECK> &sd, out_t *out, int ix)
  {
    uint8_t mantissa = mantissas_[ix];
    uint8_t symbol = sd.decode_symbol_full();
    out[ix] = mantissa | (static_cast<uint16_t>(symbol) << 8);
  }

  // Store an already-decoded symbol (used by the guarded final partial row).
  __device__ void store_symbol(out_t *out, int ix, uint8_t symbol)
  {
    uint8_t mantissa = mantissas_[ix];
    out[ix] = mantissa | (static_cast<uint16_t>(symbol) << 8);
  }

  template <bool BOUNDS_CHECK>
  __device__ void prefetch_initial(symbol_decoder<BOUNDS_CHECK> &sd, uint16_t *warp_renorm_buf, int main_iters)
  {
    // When there's a main loop, prefetch iter-0's mantissa into stage 0 and
    // prefetch the renorm buffer unsynced so both ride one commit drained by the
    // i==0 wait at the loop top. With no main loop, sync here so the tail can consume.
    if (main_iters > 0)
    {
      prefetch_async_mantissa_stage(warp_mantissa0_, mantissas_);
    }
    sd.prefetch_renorm_buffer(warp_renorm_buf, main_iters == 0 /*sync*/);
  }

  template <bool BOUNDS_CHECK>
  __device__ void decode_main_iter(
    symbol_decoder<BOUNDS_CHECK> &sd,
    out_t *out,
    uint16_t *warp_renorm_buf,
    int ix_main_iter,
    int main_iters,
    int ix_decode
  )
  {
    // sync=false leaves the renorm LDGSTS in flight, drained by the wait below.
    sd.refill_if_needed(warp_renorm_buf, false);

    // Drain the pending commit so the current stage's mantissa (prefetched in
    // the previous iter / pre-loop, possibly with a renorm refill) has landed.
    __pipeline_wait_prior(0);
    __syncwarp();

    // Prefetch the next iter's mantissa (depth-2) so its LDGSTS is in flight
    // while we consume the current stage. Left pending for the next iter's top wait.
    if (ix_main_iter + 1 < main_iters)
    {
      const uint32_t next_iter = static_cast<uint32_t>(ix_main_iter + 1);
      prefetch_async_mantissa_stage(
        warp_mantissa0_ + (next_iter % 2) * MANTISSA_WARP_STRIDE,
        mantissas_ + next_iter * MANTISSA_STAGE_SIZE
      );
      __pipeline_commit();
    }

    const int t = lane_id_;
    const uint8_t *cur_stage = warp_mantissa0_ + static_cast<uint32_t>(ix_main_iter % 2) * MANTISSA_WARP_STRIDE;

    // NUM_SYMBOLS_PER_INNER_ITER decodes per inner step: LDS the mantissa byte,
    // decode a symbol, byte_perm into an fp16, store U16. Built in local arrays
    // to convince the compiler of the best instruction order.
#pragma unroll
    for (int k = 0; k < NUM_SYMBOLS_PER_THREAD_META_ITER; k += NUM_SYMBOLS_PER_INNER_ITER)
    {
      uint32_t mantissa_bytes[NUM_SYMBOLS_PER_INNER_ITER];
      for (int j = 0; j < NUM_SYMBOLS_PER_INNER_ITER; ++j)
      {
        mantissa_bytes[j] =
          static_cast<uint32_t>(*reinterpret_cast<const uint8_t *>(&cur_stage[(k + j) * WARP_SIZE + t]));
      }
      uint8_t syms[NUM_SYMBOLS_PER_INNER_ITER];
      for (int j = 0; j < NUM_SYMBOLS_PER_INNER_ITER; ++j)
      {
        syms[j] = sd.decode_symbol_full();
      }
      uint32_t fp16s[NUM_SYMBOLS_PER_INNER_ITER];
      for (int j = 0; j < NUM_SYMBOLS_PER_INNER_ITER; ++j)
      {
        fp16s[j] = __byte_perm(mantissa_bytes[j], static_cast<uint32_t>(syms[j]), 0x1140);
      }
      for (int j = 0; j < NUM_SYMBOLS_PER_INNER_ITER; ++j)
      {
        out[ix_decode + (k + j) * WARP_SIZE] = static_cast<uint16_t>(fp16s[j]);
      }
    }
  }

private:
  // Async-copy one warp's mantissa stage (MANTISSA_STAGE_SIZE bytes) as uint2
  // LDGSTS. Caller commits/waits.
  __device__ void prefetch_async_mantissa_stage(uint8_t *stage, const uint8_t *src_for_iter)
  {
    constexpr int MANTISSA_UINT2_PER_LANE = static_cast<int>(MANTISSA_STAGE_SIZE / (WARP_SIZE * sizeof(uint2)));
    static_assert(
      MANTISSA_UINT2_PER_LANE * WARP_SIZE * sizeof(uint2) == MANTISSA_STAGE_SIZE,
      "MANTISSA_STAGE_SIZE must be a multiple of WARP_SIZE * sizeof(uint2)"
    );
    const uint2 *m_src = reinterpret_cast<const uint2 *>(src_for_iter);
    uint2 *m_dst = reinterpret_cast<uint2 *>(stage);
#pragma unroll
    for (int j = 0; j < MANTISSA_UINT2_PER_LANE; ++j)
    {
      __pipeline_memcpy_async(m_dst + j * WARP_SIZE + lane_id_, m_src + j * WARP_SIZE + lane_id_, sizeof(uint2));
    }
  }
};

// Per-chunk, per-CTA decoding table, built in-kernel by construct_decoding_table and read
// on the decode side. 4 bytes per slot:
//   bits  0..7  : sym
//   bits  8..19 : pdf
//   bits 20..31 : smcdf
// Single LDS.U32 + smart unpack on the decode side (sym = byte 0 consumed by
// __byte_perm; pdf = (entry >> 8) & 0xFFF; smcdf = entry >> 20 with no mask).
// Live across both the build and decode phases, so it stays out of the
// build/decode union below.
struct DecodeTable
{
  uint32_t slots[1 << DEFAULT_TABLELOG];
};

template <int BLOCK_DIM_X>
using TableBuildScan = nvcomp::cub::BlockScan<uint32_t, BLOCK_DIM_X>;

// Per-CTA scratch used only during the table-build phase. Overlaps with the
// decode buffers via the SetupAndDecodeSmem union (the two phases are never live
// at the same time).
template <int BLOCK_DIM_X>
struct TableBuildScratch
{
  uint2 pdfs_and_cdfs[NV_SYMBOL_COUNT];
  typename TableBuildScan<BLOCK_DIM_X>::TempStorage scan_smem;
};

// Per-CTA decode-phase buffers (char / fp16).
//
// renorm_buf: per-warp renorm prefetch window (RENORM_PREFETCH_BUF_SIZE_U16
// uint16 per warp).
//
// mantissa_staging: per-warp double-buffered staging for the FP16 main loop's
// mantissa bytes (1 byte per FP16 decode). Two stages per warp let the LDGSTS
// for iter i+1 be in flight while iter i consumes; stage s for warp w lives at
// offset (2*w + s) * MANTISSA_WARP_STRIDE. The per-iter LDGSTS uses uint2 (8 B)
// form sourced directly from sub_chunk_mantissas (8 B-aligned); the buffer is
// 16 B-aligned so each stage base is 16 B-aligned.
struct DecodeBuffers
{
  uint8_t mantissa_staging[NUM_DECOMP_WARPS_PER_CTA * 2u * MANTISSA_WARP_STRIDE];
  // Add a +1 here to allow threads to read at RENORM_PREFETCH_BUF_SIZE_U16 index without OOB access
  // This will be uninitialized, but unused. The unused access is free due to broadcast.
  uint16_t renorm_buf[NUM_DECOMP_WARPS_PER_CTA * RENORM_PREFETCH_BUF_SIZE_U16 + 1];
};

// Per-CTA decode-phase buffers (fp8). fp8 loads the packed_signs_mantissas side-band with a
// plain LDG (no cp.async staging), so there is no mantissa_staging; those bytes
// are folded into a deeper per-warp renorm window (RENORM_PREFETCH_BUF_SIZE_U16_FP8),
// which keeps refills out of the hot loop at no extra shared memory vs the
// char/fp16 layout.
struct DecodeBuffersFp8
{
  // +1 for the same OOB-safe broadcast read at the buffer end as DecodeBuffers.
  uint16_t renorm_buf[NUM_DECOMP_WARPS_PER_CTA * RENORM_PREFETCH_BUF_SIZE_U16_FP8 + 1];
};

// The table-build scratch and the decode buffers are never live at the same
// time: construct_decoding_table finishes (writing only the separate `table`)
// and a __syncthreads separates it from the decode phase, which is the first
// thing to touch renorm_buf / mantissa_staging. Overlap them in a union so the
// build scratch costs no shared memory on top of the (larger) decode buffers.
// DecodeBuffersT selects the char/fp16 vs fp8 layout (defaulting to char/fp16).
template <int BLOCK_DIM_X, typename DecodeBuffersT = DecodeBuffers>
union SetupAndDecodeSmem
{
  TableBuildScratch<BLOCK_DIM_X> build;
  DecodeBuffersT decode;
};

} // namespace detail
} // namespace ans_gpu_lib
