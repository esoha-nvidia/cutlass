/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include <type_traits>

#include "common.h"

#include <ans/ans_utils.cuh>
#include <ans/symbol_decoder.cuh>
#include <ans/types.cuh>
#include <nvcomp/utils.hpp>

namespace ans_gpu_lib
{
namespace detail
{

// Builds the per-chunk decoding table directly into the caller-provided shared
// `table` (1 << tablelog slots). Designed to run at BLOCK_DIM_X threads where
// 2 * BLOCK_DIM_X == NV_SYMBOL_COUNT (i.e. 128 threads cover 256
// symbols, 2 symbols per thread). `scratch` is caller-provided shared memory
// (see TableBuildScratch) so it can be overlapped with the decode buffers.
template <bool BOUNDS_CHECK, int BLOCK_DIM_X>
__device__ void construct_decoding_table(
  uint32_t *table,
  TableBuildScratch<BLOCK_DIM_X> &scratch,
  const void *comp_chunk,
  const size_t comp_chunk_size
)
{
  // Each thread owns SYMBOLS_PER_THREAD symbols; the j-th symbol of thread t is
  // t + j * BLOCK_DIM_X. With BLOCK_DIM_X == 128 this covers [0, 256).
  constexpr int SYMBOLS_PER_THREAD = NV_SYMBOL_COUNT / BLOCK_DIM_X;
  static_assert(
    SYMBOLS_PER_THREAD * BLOCK_DIM_X == NV_SYMBOL_COUNT,
    "construct_decoding_table assumes BLOCK_DIM_X evenly divides the symbol count"
  );

  uint32_t lane_id = get_lane_id();
  uint32_t wid = threadIdx.x / WARP_SIZE_U;
  uint32_t num_warps = blockDim.x / WARP_SIZE_U;

  const uint8_t *chunk_start = static_cast<const uint8_t *>(comp_chunk);
  const uint8_t *chunk_end = chunk_start + comp_chunk_size;

  size_t mantissas_size = ans_mantissas_size_value(*reinterpret_cast<const size_t *>(chunk_start));
  const uint8_t *ans_comp_chunk_start =
    static_cast<const uint8_t *>(round_up_align_address(chunk_start + mantissas_size, 8));

  auto sc_header = (ANS_sub_chunk_header *)ans_comp_chunk_start;
  const int16_t *norm_counts = sc_header->get_norm_counts();
  uint32_t max_symbol_value = safe_guard_generic<BOUNDS_CHECK>(
    safe_guard_generic<BOUNDS_CHECK>(&sc_header->max_symbol_value_, chunk_start, chunk_end),
    NV_SYMBOL_COUNT
  );
  [[maybe_unused]] uint32_t tablelog = safe_guard_generic<BOUNDS_CHECK>(&sc_header->tablelog_, chunk_start, chunk_end);
  assert(tablelog == DEFAULT_TABLELOG);

  uint2 *pdfs_and_cdfs = scratch.pdfs_and_cdfs;

  // Populate pdfs with unconditional LDGs + stores, so the norm_counts loads are
  // in flight while the dependent (higher-latency to resolve) max_symbol_value
  // load is still outstanding. Guarding the store on (symbol <= max_symbol_value)
  // instead would let the compiler sink/predicate the load behind that
  // comparison and serialize the global loads -- the stall we're avoiding here.
  //
  // Each thread covers SYMBOLS_PER_THREAD symbols (t, t + BLOCK_DIM_X, ...), so
  // 2 * BLOCK_DIM_X == NV_SYMBOL_COUNT slots are populated with no index
  // guard. Slots past max_symbol_value get pdf 0 (selected on max_symbol_value):
  // they then contribute 0 to the exclusive prefix sum, so in-range cdfs are
  // unaffected, and they're independently masked by `active` in the fill below.
  uint32_t pdf_items[SYMBOLS_PER_THREAD];
#pragma unroll
  for (int ix_symbol = 0; ix_symbol < SYMBOLS_PER_THREAD; ++ix_symbol)
  {
    const int symbol = threadIdx.x + ix_symbol * BLOCK_DIM_X;
    const int16_t raw = safe_guard_generic<BOUNDS_CHECK>(&norm_counts[symbol], chunk_start, chunk_end);
    pdf_items[ix_symbol] = (symbol <= max_symbol_value) ? static_cast<uint32_t>(abs(raw)) : 0u;
  }

// 256-element exclusive prefix sum over BLOCK_DIM_X threads (SYMBOLS_PER_THREAD
// items/thread).
#pragma unroll
  for (int j = 0; j < SYMBOLS_PER_THREAD; ++j)
  {
    pdfs_and_cdfs[threadIdx.x + j * BLOCK_DIM_X].x = pdf_items[j];
  }
  __syncthreads();

  using Scan = TableBuildScan<BLOCK_DIM_X>;

  uint32_t scan_in[SYMBOLS_PER_THREAD];
#pragma unroll
  for (int j = 0; j < SYMBOLS_PER_THREAD; ++j)
  {
    scan_in[j] = pdfs_and_cdfs[threadIdx.x * SYMBOLS_PER_THREAD + j].x;
  }

  uint32_t scan_out[SYMBOLS_PER_THREAD];
  Scan(scratch.scan_smem).ExclusiveSum(scan_in, scan_out);

#pragma unroll
  for (int j = 0; j < SYMBOLS_PER_THREAD; ++j)
  {
    pdfs_and_cdfs[threadIdx.x * SYMBOLS_PER_THREAD + j].y = scan_out[j];
  }
  __syncthreads();

// The whole symbol range is covered in SYMBOLS_PER_THREAD warp-uniform passes:
// each pass handles num_warps * WARP_SIZE = BLOCK_DIM_X symbols, so
// SYMBOLS_PER_THREAD passes cover all NV_SYMBOL_COUNT symbols.
//
// Most symbols have pdf 0 or 1: a pdf-1 symbol is written directly by its
// owning lane (pdf 0 writes nothing). The rare "heavy" symbols (pdf > 1) are
// balloted out and processed one at a time, with all 32 lanes cooperating to
// fill that symbol's slots.
//
// Symbols are interleaved across warps (warp w owns w, w+num_warps,
// w+2*num_warps, ...) rather than handed out in contiguous blocks. Heavy
// symbols tend to cluster (e.g. a run of large pdfs), so a contiguous block
// assignment would dump the whole cluster onto one warp while the others
// idle; the round-robin spread keeps the heavy work balanced across warps.
#pragma unroll
  for (int ix_symbol = 0; ix_symbol < SYMBOLS_PER_THREAD; ++ix_symbol)
  {
    const int symbol = (lane_id * num_warps + wid) + ix_symbol * BLOCK_DIM_X;
    const bool active = symbol <= max_symbol_value;
    const uint32_t pdf = active ? pdfs_and_cdfs[symbol].x : 0;
    const uint32_t cdf = active ? pdfs_and_cdfs[symbol].y : 0;

    // Common case: a single-slot symbol is written by its owning lane.
    if (pdf == 1)
    {
      uint32_t slot = safe_guard_generic<BOUNDS_CHECK>(cdf, 1 << DEFAULT_TABLELOG);
      table[slot] = (static_cast<uint32_t>(pdf) << 8) | static_cast<uint32_t>(symbol);
    }

    // Heavy symbols (pdf > 1): cooperatively fill, one heavy lane at a time. The
    // ballot/shuffle are warp-uniform since every lane reaches them together.
    uint32_t heavy = __ballot_sync(WARP_ALL, pdf > 1);
    while (heavy != 0)
    {
      const int lane = __ffs(heavy) - 1;
      heavy &= heavy - 1;

      const uint32_t h_symbol = __shfl_sync(WARP_ALL, static_cast<uint32_t>(symbol), lane);
      const uint32_t h_pdf = __shfl_sync(WARP_ALL, pdf, lane);
      const uint32_t h_cdf = __shfl_sync(WARP_ALL, cdf, lane);

      for (uint32_t ix_pdf = lane_id; ix_pdf < h_pdf; ix_pdf += WARP_SIZE)
      {
        uint32_t slot = safe_guard_generic<BOUNDS_CHECK>(h_cdf + ix_pdf, 1 << DEFAULT_TABLELOG);
        table[slot] = (ix_pdf << 20) | (h_pdf << 8) | h_symbol;
      }
    }
  }
}

// Aggregate the per-lane decode error across the warp and write the chunk status
// (lane 0 only). Shared by all data types.
__device__ void write_decode_status(nvcompStatus_t err, int lane_id, nvcompStatus_t *status)
{
  bool found_err = __ballot_sync(WARP_ALL, err != nvcompSuccess); // aggregate errors across all threads in chunk
  if (status && lane_id == 0)
  {
    *status = found_err ? nvcompErrorCannotDecompress : nvcompSuccess;
  }
}

// Per-warp decode of one sub-chunk: a single main loop over full meta-iters and a
// single remainder/tail (scalar rows + final partial row). All type-specific
// work (initial prefetch, per-iter body, store) is delegated to the Decoder
// policy (CharDecodePolicy / Fp16DecodePolicy), so this scaffold has no per-type branching.
template <typename DecoderPolicy, typename SymbolDecoderT>
__device__ void decode_sub_chunk(
  DecoderPolicy decoder,
  SymbolDecoderT &sd,
  void *uncomp_sub_chunk,
  uint16_t *warp_renorm_buf,
  int num_decodes,
  int lane_id,
  size_t uncomp_chunk_size,
  size_t *actual_uncomp_chunk_size,
  nvcompStatus_t *status
)
{
  using out_t = typename DecoderPolicy::out_t;
  constexpr int NUM_SYMBOLS_PER_THREAD_META_ITER = DecoderPolicy::NUM_SYMBOLS_PER_THREAD_META_ITER;
  constexpr int NUM_SYMBOLS_PER_WARP_META_ITER = DecoderPolicy::NUM_SYMBOLS_PER_WARP_META_ITER;

  out_t *out = reinterpret_cast<out_t *>(uncomp_sub_chunk);

  // Multi-byte output (fp16) is written as out_t, so the output buffer must be at
  // least out_t-aligned or the stores would fault; report a clean error instead.
  // Compiles away for char (OUT_TYPE_SIZE == 1). out is warp-uniform and its
  // alignment relative to the chunk base is grid-uniform, so the warp returns as a
  // whole and there is no post-decode barrier to skip.
  if constexpr (DecoderPolicy::OUT_TYPE_SIZE > 1)
  {
    if ((reinterpret_cast<uintptr_t>(out) % DecoderPolicy::OUT_TYPE_SIZE) != 0)
    {
      if (status && lane_id == 0)
      {
        *status = nvcompErrorOutputBufferAlignmentTooSmall;
      }
      return;
    }
  }

  // full_rows counts rows where every lane has a symbol; the final (possibly
  // partial) row is handled separately in the tail.
  const int full_rows = num_decodes / WARP_SIZE;
  const int main_iters = full_rows / NUM_SYMBOLS_PER_THREAD_META_ITER;
  int ix_decode = lane_id;

  // Pre-loop: initial full renorm fill (the renorm buffer is a continuous cursor
  // across iters, only re-anchored + refetched on refill_if_needed, so most iters
  // issue no renorm LDGSTS). The policy also stages iter-0 inputs as needed.
  decoder.prefetch_initial(sd, warp_renorm_buf, main_iters);

  // ---------- single main loop ----------
  for (int i = 0; i < main_iters; ++i, ix_decode += NUM_SYMBOLS_PER_WARP_META_ITER)
  {
    decoder.decode_main_iter(sd, out, warp_renorm_buf, i, main_iters, ix_decode);
  }

  // ---------- single remainder/tail ----------
  // Skipped only when num_decodes is an exact multiple of a full warp meta-iter
  // (the main loop covered everything).
  if (num_decodes % NUM_SYMBOLS_PER_WARP_META_ITER > 0)
  {
    sd.refill_if_needed(warp_renorm_buf, true);

    for (int i = 0; i < full_rows % NUM_SYMBOLS_PER_THREAD_META_ITER; ++i, ix_decode += WARP_SIZE)
    {
      decoder.store_one(sd, out, ix_decode);
    }

    // Final partial row: the state already holds each active lane's last symbol;
    // lanes past the end read junk, so all lanes decode but only active ones store.
    if (num_decodes % WARP_SIZE > 0)
    {
      uint8_t symbol = sd.decode_symbol_full();
      if (ix_decode < num_decodes)
      {
        decoder.store_symbol(out, ix_decode, symbol);
      }
    }
  }

  nvcompStatus_t err = nvcompSuccess;
  if constexpr (SymbolDecoderT::bounds_check)
  {
    err = sd.getError();
  }
  write_decode_status(err, lane_id, status);

  if (actual_uncomp_chunk_size && lane_id == 0)
  {
    *actual_uncomp_chunk_size = DecoderPolicy::OUT_TYPE_SIZE * uncomp_chunk_size;
  }
}

// FP8 (E4M3) reconstruct helpers. Each ANS symbol is a packed_exp byte; the
// matching side-band byte is packed_signs_mantissas. One symbol reconstructs TWO
// FP8 output bytes (a uint16). For each of the two FP8 values in the pair, the
// packed_exp byte holds a 4-bit exponent nibble and the packed_signs_mantissas
// byte holds a nibble = (sign << 3) | mantissa(3 bits).

// Reconstruct one FP8 (E4M3) output byte from its exponent nibble (4 bits) and its
// sign+mantissa nibble (sign in bit 3, mantissa in bits 0..2):
//   output = sign(bit 7) | exponent(bits 6..3) | mantissa(bits 2..0)
inline __device__ uint32_t fp8_reconstruct_byte(uint32_t exp_nibble, uint32_t signs_mantissas_nibble)
{
  const uint32_t sign = (signs_mantissas_nibble & 0x8) << 4; // nibble bit 3 -> byte bit 7
  const uint32_t exponent = exp_nibble << 3; // -> byte bits 6..3
  const uint32_t mantissa = signs_mantissas_nibble & 0x7; // -> byte bits 2..0
  return sign | exponent | mantissa;
}

// Reconstruct the two FP8 bytes of one symbol (a packed pair) and pack them into
// a uint16 (first FP8 byte in the low byte, second in the high byte). Scalar form,
// used by the row-map remainder tail.
inline __device__ uint16_t fp8_reconstruct_pair(uint8_t packed_exp, uint8_t packed_signs_mantissas)
{
  const uint32_t byte_lo = fp8_reconstruct_byte(packed_exp & 0xF, packed_signs_mantissas & 0xF);
  const uint32_t byte_hi = fp8_reconstruct_byte((packed_exp >> 4) & 0xF, (packed_signs_mantissas >> 4) & 0xF);
  return static_cast<uint16_t>(byte_lo | (byte_hi << 8));
}

// Reconstruct 8 FP8 output bytes from 4 packed symbols (4 packed pairs) at once.
// packed_exp / packed_signs_mantissas hold the 4 packed_exp / packed_signs_mantissas
// bytes, one per byte lane. Returns the 8 output bytes as a uint2 (symbol k's
// uint16 in bytes {2k, 2k+1}).
//
// The inputs are packed one byte per lane of a uint32 so the per-nibble math from
// fp8_reconstruct_byte runs on all 4 bytes at once via whole-word masks/shifts --
// the ALU does 4 lanes' work per instruction instead of ~48 byte-granularity ops.
// This is safe because the per-lane shifts never carry across byte lanes (max
// output byte 0xFB < 0x100).
inline __device__ uint2 fp8_reconstruct_eight(uint32_t packed_exp, uint32_t packed_signs_mantissas)
{
  // Split each of the 4 packed bytes into its low nibble (the first FP8 byte of
  // the pair) and high nibble (the second), one nibble per byte lane.
  const uint32_t exp_lo = packed_exp & 0x0F0F0F0Fu;
  const uint32_t exp_hi = (packed_exp >> 4) & 0x0F0F0F0Fu;
  const uint32_t signs_mantissas_lo = packed_signs_mantissas & 0x0F0F0F0Fu;
  const uint32_t signs_mantissas_hi = (packed_signs_mantissas >> 4) & 0x0F0F0F0Fu;

  // Reconstruct, across all 4 lanes: out_byte = sign<<7 | exp<<3 | mantissa.
  // out_lo = the 4 first FP8 bytes; out_hi = the 4 second FP8 bytes.
  const uint32_t out_lo = ((signs_mantissas_lo & 0x08080808u) << 4) | (exp_lo << 3) |
                          (signs_mantissas_lo & 0x07070707u);
  const uint32_t out_hi = ((signs_mantissas_hi & 0x08080808u) << 4) | (exp_hi << 3) |
                          (signs_mantissas_hi & 0x07070707u);

  // Interleave the per-lane bytes back into symbol order {lo0,hi0,lo1,hi1 | ...}.
  // out_lo bytes are operands 0..3, out_hi bytes are 4..7 in __byte_perm's index space.
  uint2 result;
  result.x = __byte_perm(out_lo, out_hi, 0x5140u); // {lo0, hi0, lo1, hi1}
  result.y = __byte_perm(out_lo, out_hi, 0x7362u); // {lo2, hi2, lo3, hi3}
  return result;
}

// Decode the next 4 packed_exp symbols and pack them into one uint32, one byte per
// lane (symbol j in byte j), the layout fp8_reconstruct_eight expects.
template <typename SymbolDecoderT>
inline __device__ uint32_t decode_four_symbols_and_pack_register(SymbolDecoderT &sd)
{
  uint32_t packed = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j)
  {
    packed |= static_cast<uint32_t>(sd.decode_symbol_full()) << (8 * j);
  }
  return packed;
}

// Store 8 reconstructed FP8 bytes (a uint2) to dst using the widest vector store
// the output buffer's alignment permits. OUT_ALIGN is the byte alignment the host
// guarantees for the FP8 output buffer (the tile store address is dst + a multiple
// of 16, so its alignment equals the buffer's). One STG.64 at 8 B; otherwise split
// into narrower stores so a 4/2/1 B-aligned output buffer is still legal.
template <int OUT_ALIGN>
inline __device__ void fp8_store_eight(uint8_t *dst, uint2 v)
{
  static_assert(OUT_ALIGN == 8 || OUT_ALIGN == 4 || OUT_ALIGN == 2 || OUT_ALIGN == 1, "unsupported OUT_ALIGN");
  if constexpr (OUT_ALIGN >= 8)
  {
    *reinterpret_cast<uint2 *>(dst) = v;
  }
  else if constexpr (OUT_ALIGN == 4)
  {
    reinterpret_cast<uint32_t *>(dst)[0] = v.x;
    reinterpret_cast<uint32_t *>(dst)[1] = v.y;
  }
  else if constexpr (OUT_ALIGN == 2)
  {
    uint16_t *d = reinterpret_cast<uint16_t *>(dst);
    d[0] = static_cast<uint16_t>(v.x);
    d[1] = static_cast<uint16_t>(v.x >> 16);
    d[2] = static_cast<uint16_t>(v.y);
    d[3] = static_cast<uint16_t>(v.y >> 16);
  }
  else
  {
#pragma unroll
    for (int b = 0; b < 4; ++b)
    {
      dst[b] = static_cast<uint8_t>(v.x >> (8 * b));
      dst[4 + b] = static_cast<uint8_t>(v.y >> (8 * b));
    }
  }
}

// Store one reconstructed FP8 pair (a uint16) to out16[ix]. For a 1 B-aligned
// output buffer the uint16 store would be misaligned, so split it into 2 bytes.
template <int OUT_ALIGN>
inline __device__ void fp8_store_pair(uint16_t *out16, int ix, uint16_t v)
{
  if constexpr (OUT_ALIGN >= 2)
  {
    out16[ix] = v;
  }
  else
  {
    uint8_t *dst = reinterpret_cast<uint8_t *>(out16) + 2 * ix;
    dst[0] = static_cast<uint8_t>(v);
    dst[1] = static_cast<uint8_t>(v >> 8);
  }
}

// FP8 (E4M3) decode. Mirror of Fp8EncodePolicy::encode_body: full 256-symbol tiles in
// BLOCK order (tile 0..full-1, k=0..B-1, lane t -> output symbol tile*256 + B*t + k)
// followed by a row-map remainder (rem = num_decodes % 256; rem == 0 is the common
// case). The encoder produces the exact mirror.
//
// Per-tile machinery:
//  - plain wide LDG (uint2 = 8 B) of the lane's contiguous packed_signs_mantissas side-band
//    (no cp.async stage: the side-band is tiny and off the decode critical path,
//    so the prefetch/commit/wait/syncwarp bookkeeping cost more than it saved).
//  - fp8_reconstruct_eight: reconstruct 4 symbols (8 FP8 values) per ALU batch.
//  - two 8 B output writes per lane per tile (block addressing), each emitted at
//    OUT_ALIGN granularity via fp8_store_eight: a single STG.64 when the output
//    buffer is 8 B-aligned, otherwise split into narrower stores so a 4/2/1 B
//    aligned buffer is still legal. (Block addressing makes consecutive lanes 16 B
//    apart, so the warp store isn't fully coalesced, but fp8 decode is
//    instruction-issue bound, not store-bound.)
// The remainder uses per-symbol scalar reconstruct (fp8_reconstruct_pair).
//
// OUT_ALIGN is the byte alignment the caller guarantees for the FP8 output buffer
// (8/4/2/1); the host dispatches the matching specialization from the runtime
// pointer alignment. num_decodes is the sub-chunk's symbol (pair) count;
// uncomp_chunk_size is the chunk's true FP8 byte count N (for
// actual_uncomp_chunk_size reporting).
template <int OUT_ALIGN, typename SymbolDecoderT>
__device__ void decode_sub_chunk_fp8(
  SymbolDecoderT &sd,
  void *uncomp_sub_chunk,
  const uint8_t *sub_chunk_mantissas, // packed_signs_mantissas side-band (1 byte/symbol, symbol order)
  uint16_t *warp_renorm_buf,
  int num_decodes,
  int lane_id,
  size_t uncomp_chunk_size,
  size_t *actual_uncomp_chunk_size,
  nvcompStatus_t *status
)
{
  constexpr int B = 8; // contiguous symbols per lane per tile
  constexpr int TILE = WARP_SIZE * B; // 256 symbols per tile
  constexpr int TILE_BYTES = TILE; // 1 side-band byte per symbol

  uint16_t *out16 = reinterpret_cast<uint16_t *>(uncomp_sub_chunk);
  const int full_tiles = num_decodes / TILE;

  sd.prefetch_renorm_buffer(warp_renorm_buf, true /*sync*/);

  for (int T = 0; T < full_tiles; ++T)
  {
    const int out_base = T * TILE + B * lane_id; // lane's contiguous output run
    // Lane's B (=8) packed_signs_mantissas side-band bytes, contiguous at
    // sub_chunk_mantissas[T*256 + 8*lane .. +7]. One wide LDG (uint2 = 8 B, base is
    // 8 B-aligned): .x = bytes 0..3, .y = bytes 4..7.
    const uint2 packed_signs_mantissas =
      *reinterpret_cast<const uint2 *>(sub_chunk_mantissas + T * TILE_BYTES + B * lane_id);
    sd.refill_if_needed(warp_renorm_buf, true);

    // Decode the lane's 8 symbols in two groups of 4, reconstructing 4 outputs
    // (a uint2) per group, then store each group as one 8 B write (at OUT_ALIGN
    // granularity).
    uint32_t packed_exp_four = decode_four_symbols_and_pack_register(sd);
    const uint2 out_first4 = fp8_reconstruct_eight(packed_exp_four, packed_signs_mantissas.x);
    packed_exp_four = decode_four_symbols_and_pack_register(sd);
    const uint2 out_next4 = fp8_reconstruct_eight(packed_exp_four, packed_signs_mantissas.y);

    uint8_t *out = reinterpret_cast<uint8_t *>(out16 + out_base);
    fp8_store_eight<OUT_ALIGN>(out, out_first4);
    fp8_store_eight<OUT_ALIGN>(out + sizeof(uint2), out_next4);
  }

  // Row-map remainder (rem = num_decodes % 256), the exact mirror of the encoder's
  // groups R (full rows) + P (partial row): symbol index s = rem_base + r*32 + lane
  // is owned by lane (s % 32). rANS forward order is rows low -> high, then the
  // partial row, so the decode order continues seamlessly after the block tiles.
  const int rem_base = full_tiles * TILE;
  const int rem = num_decodes - rem_base; // 0..255
  if (rem > 0)
  {
    const int rem_full_rows = rem / WARP_SIZE;
    const int rpart = rem - rem_full_rows * WARP_SIZE; // 0..31
    // Full remainder rows, low -> high.
    for (int r = 0; r < rem_full_rows; ++r)
    {
      sd.refill_if_needed(warp_renorm_buf, true);
      const int s_ix = rem_base + r * WARP_SIZE + lane_id;
      const uint8_t packed_signs_mantissas = sub_chunk_mantissas[s_ix];
      const uint8_t packed_exp = sd.decode_symbol_full();
      fp8_store_pair<OUT_ALIGN>(out16, s_ix, fp8_reconstruct_pair(packed_exp, packed_signs_mantissas));
    }
    // Partial row: all lanes decode (lockstep ballot), only the first rpart store.
    if (rpart > 0)
    {
      sd.refill_if_needed(warp_renorm_buf, true);
      const int s_ix = rem_base + rem_full_rows * WARP_SIZE + lane_id;
      const uint8_t packed_exp = sd.decode_symbol_full();
      if (lane_id < rpart)
      {
        fp8_store_pair<OUT_ALIGN>(out16, s_ix, fp8_reconstruct_pair(packed_exp, sub_chunk_mantissas[s_ix]));
      }
    }
  }

  nvcompStatus_t err = nvcompSuccess;
  if constexpr (SymbolDecoderT::bounds_check)
  {
    err = sd.getError();
  }
  write_decode_status(err, lane_id, status);

  if (actual_uncomp_chunk_size && lane_id == 0)
  {
    *actual_uncomp_chunk_size = uncomp_chunk_size; // = N (bytes)
  }
}

// Pick the FP8 store granularity from the output buffer's runtime alignment and
// dispatch the matching decode_sub_chunk_fp8 specialization. The host advertises
// only 1 B output alignment (it cannot inspect the device output pointer), so the
// store width is chosen here, in-kernel, from the actual uncomp_sub_chunk address:
// 8 B-aligned buffers get the single-STG.64 fast path; otherwise narrower stores.
// uncomp_sub_chunk is the warp's contiguous output run, so its alignment governs
// every store this warp makes.
template <typename SymbolDecoderT>
__device__ void decode_sub_chunk_fp8_dispatch(
  SymbolDecoderT &sd,
  void *uncomp_sub_chunk,
  const uint8_t *sub_chunk_mantissas,
  uint16_t *warp_renorm_buf,
  int num_decodes,
  int lane_id,
  size_t uncomp_chunk_size,
  size_t *actual_uncomp_chunk_size,
  nvcompStatus_t *status
)
{
  const uintptr_t addr = reinterpret_cast<uintptr_t>(uncomp_sub_chunk);
  auto run = [&](auto out_align_tag) {
    constexpr int OUT_ALIGN = decltype(out_align_tag)::value;
    decode_sub_chunk_fp8<OUT_ALIGN>(
      sd,
      uncomp_sub_chunk,
      sub_chunk_mantissas,
      warp_renorm_buf,
      num_decodes,
      lane_id,
      uncomp_chunk_size,
      actual_uncomp_chunk_size,
      status
    );
  };
  if ((addr & 7u) == 0)
  {
    run(std::integral_constant<int, 8>{});
  }
  else if ((addr & 3u) == 0)
  {
    run(std::integral_constant<int, 4>{});
  }
  else if ((addr & 1u) == 0)
  {
    run(std::integral_constant<int, 2>{});
  }
  else
  {
    run(std::integral_constant<int, 1>{});
  }
}

// Compile-time decode mode. Generic branches at runtime on the bitstream (used
// when the host does not know the data type, e.g. char/default). Fp16/Fp8 fix the
// type at compile time so that specialization contains ONLY that decode path and
// is register-allocated for it alone -- a type the host specified via the
// decompress opts does not pay the other types' register footprint (which would
// otherwise raise the shared kernel's ceiling and spill to local memory / LDL).
enum class DecodeMode
{
  Generic,
  Fp16,
  Fp8
};

template <DecodeMode MODE, bool BOUNDS_CHECK, int BLOCK_DIM_X>
__device__ void decompress_chunk(
  const void *comp_chunk,
  const size_t comp_chunk_size,
  void *const *uncomp_chunks,
  size_t uncomp_chunk_buf_size,
  size_t *actual_uncomp_chunk_size,
  nvcompStatus_t *status,
  uint8_t max_sub_chunk_count
)
{
  // Per-CTA shared decoding table. Live across
  // both the build and decode phases, so it stays out of the build/decode union.
  __shared__ __align__(sizeof(uint4)) DecodeTable table_smem;
  uint32_t *table = table_smem.slots;

  // Setup-phase strategy (fused table build):
  //   1. Drive the mantissas_size -> ans_comp_chunk_start -> header-reads
  //      dependency chain.
  //   2. Build the decoding table directly
  //      into shared memory (construct_decoding_table, which syncs internally).
  //      No global decoding-table load: every CTA constructs its own table
  //      in shared memory. This is work inefficient when multiple CTAs are processing
  //      the same chunk, but table construction is inexpensive.
  //   3. Issue the per-warp sub-chunk header LDGs *before* the table build's
  //      trailing __syncthreads so they overlap with it. Guarded on a
  //      warp_active flag so the trailing warps in a partial CTA don't OOB-read
  //      the offsets table.

  const int tid = blockDim.x * blockIdx.y + threadIdx.x;
  const int lane_id = get_lane_id();
  // ix_warp is the in-CTA warp id (0..NUM_DECOMP_WARPS_PER_CTA-1), used to index
  // the per-CTA shared renorm/mantissa buffers. sub_chunk_idx is the global
  // sub-chunk index (includes blockIdx.y), used for input/output addressing.
  const int ix_warp = threadIdx.x / WARP_SIZE;
  const int sub_chunk_idx = tid / WARP_SIZE;

  // ---------- (1) Pointer setup + chunk-metadata dependency chain ----------
  void *uncomp_chunk = uncomp_chunks[blockIdx.x];

  const uint8_t *comp_chunk_start = reinterpret_cast<const uint8_t *>(comp_chunk);
  const uint8_t *mantissas_start = comp_chunk_start + sizeof(size_t);
  const uint8_t *comp_chunk_end = comp_chunk_start + comp_chunk_size;
  const size_t raw_mantissas_size = *reinterpret_cast<const size_t *>(comp_chunk_start);
  // FP8 is self-describing via the MSB flag in the mantissas_size prefix (it
  // carries a side-band like fp16, so the size alone cannot distinguish them).
  // In Fp16/Fp8 mode the type is fixed at compile time (the unused branches and
  // their registers are then eliminated from the specialization).
  constexpr bool compiletime_is_fp8 = (MODE == DecodeMode::Fp8);
  const bool runtime_is_fp8 = (MODE == DecodeMode::Generic) ? ans_mantissas_size_is_fp8(raw_mantissas_size)
                                                            : compiletime_is_fp8;
  const size_t mantissas_size = ans_mantissas_size_value(raw_mantissas_size);
  const uint8_t *ans_comp_chunk_start =
    static_cast<const uint8_t *>(round_up_align_address(comp_chunk_start + mantissas_size, 8));
  uint8_t *op = reinterpret_cast<uint8_t *>(uncomp_chunk);

  const ANS_sub_chunk_header *sub_chunk_header = reinterpret_cast<const ANS_sub_chunk_header *>(ans_comp_chunk_start);

  nvcompStatus_t err = nvcompSuccess;
  size_t uncomp_chunk_size =
    safe_guard_generic<BOUNDS_CHECK>(&sub_chunk_header->uncomp_chunk_size_, comp_chunk_start, comp_chunk_end, &err);
  const size_t max_sub_chunk_size =
    safe_guard_generic<BOUNDS_CHECK>(&sub_chunk_header->max_sub_chunk_size_, ans_comp_chunk_start, comp_chunk_end, &err);

  // For fp8 the header's uncomp_chunk_size_ stores the true FP8 byte count N; the
  // ANS symbol count is chunk_symbols = N / 2 (full pairs). char/fp16 store the
  // symbol count directly. chunk_symbols is in ANS-symbol (pair) units, used for
  // sub-chunk partitioning and addressing (max_sub_chunk_size is also in symbols).
  const size_t chunk_symbols = runtime_is_fp8 ? uncomp_chunk_size / 2 : uncomp_chunk_size;

  // ---------- CTA-uniform early returns ----------
  if (uncomp_chunk_size == 0)
  {
    if (status && tid == 0)
    {
      *status = err;
    }

    if (actual_uncomp_chunk_size && tid == 0)
    {
      *actual_uncomp_chunk_size = 0;
    }

    return;
  }

  // The Fp8/Fp16 specializations force the decode type from MODE (overriding the
  // self-describing stream) for lean registers, so a mismatched stream would
  // mis-decode or read OOB. Reject it: classify the stream's actual type from the
  // header (fp8 = MSB flag; char = no side-band, mantissas_size == sizeof(size_t);
  // fp16 = otherwise) and fail the chunk if it disagrees with MODE. CTA-uniform
  // (depends only on the chunk header), so the whole CTA returns together. Checked
  // after the empty-chunk return because an empty fp16 chunk also has
  // mantissas_size == sizeof(size_t) and would otherwise look like char. Generic
  // handles any type, so it is exempt.
  if constexpr (MODE != DecodeMode::Generic)
  {
    const bool stream_is_fp8 = ans_mantissas_size_is_fp8(raw_mantissas_size);
    const bool stream_is_char = !stream_is_fp8 && (mantissas_size == sizeof(size_t));
    const bool stream_matches_mode = compiletime_is_fp8 ? stream_is_fp8 : (!stream_is_fp8 && !stream_is_char);
    if (!stream_matches_mode)
    {
      if (status && tid == 0)
      {
        *status = nvcompErrorCannotDecompress;
      }
      return;
    }
  }

  // Debug backstop: when a max sub chunk count was requested, the chunk must not
  // decompose into more sub chunks than that.
  const int actual_sub_chunk_count =
    max(nvcomp::narrow_cast<int>(nvcomp::roundUpDiv(chunk_symbols, max_sub_chunk_size)), 1);
  if ((actual_sub_chunk_count > max_sub_chunk_count) && max_sub_chunk_count != 0)
  {
    if (status && tid == 0)
    {
      *status = nvcompErrorSubChunkCountTooSmall;
    }
    assert(false); // Capture subchunkCountTooSmall error for debug
    return;
  }

  // prevent decoding more symbols than we can read
  // Always perform this check, since we guarantee it in the documentation of DecompressAsync.
  // Done before the no-active-sub-chunk return below so it also covers chunks with
  // no full pairs (e.g. fp8 N == 1, chunk_symbols == 0).
  if (uncomp_chunk_size > uncomp_chunk_buf_size)
  {
    if (status && tid == 0)
    {
      *status = nvcompErrorOutputBufferTooSmall;
    }
    return;
  }

  // FP8 odd N: the unpaired trailing byte was stored raw at side-band[chunk_symbols]
  // (right after the P packed_signs_mantissas bytes) and is not a decoded ANS symbol. Emit it
  // here, before the per-warp early returns, so it is written even when the
  // chunk has no full pairs (e.g. N == 1) or this CTA owns no active sub-chunk.
  // One writer per chunk: block-row 0, thread 0. Also set the actual output size
  // here so it is reported even when no warp runs decode_sub_chunk_fp8 (N == 1).
  if (runtime_is_fp8 && blockIdx.y == 0 && threadIdx.x == 0)
  {
    if (uncomp_chunk_size & 1)
    {
      op[uncomp_chunk_size - 1] = mantissas_start[chunk_symbols];
    }
    if (actual_uncomp_chunk_size)
    {
      *actual_uncomp_chunk_size = uncomp_chunk_size; // = N (bytes)
    }
  }

  // If no warp in the parent CTA is processing a sub chunk, we can return
  // (If >= 1 warp in the parent CTA has a sub chunk, we can't return until we
  // help with copying the decoding table to shmem)
  if ((sub_chunk_idx / NUM_DECOMP_WARPS_PER_CTA) * NUM_DECOMP_WARPS_PER_CTA * max_sub_chunk_size >= chunk_symbols)
  {
    return;
  }

  // mantissas_size == sizeof(size_t) marks uint8 data; anything else (with no fp8
  // flag) is fp16. The fp16 output-alignment check lives in decode_sub_chunk (it is
  // out_t-driven); fp8 has no minimum output alignment (decode_sub_chunk_fp8_dispatch
  // picks the store width from the runtime pointer alignment).
  const bool fp16_data = !runtime_is_fp8 && (mantissas_size != sizeof(size_t));

  constexpr uint32_t RENORM_DEPTH_U16 = compiletime_is_fp8 ? RENORM_PREFETCH_BUF_SIZE_U16_FP8
                                                           : RENORM_PREFETCH_BUF_SIZE_U16;
  using DecodeBuffersForMode = std::conditional_t<compiletime_is_fp8, DecodeBuffersFp8, DecodeBuffers>;
  using SubChunkDecoder = symbol_decoder<BOUNDS_CHECK, RENORM_DEPTH_U16, RENORM_REFILL_THRESHOLD_U16>;

  // Table-build scratch and decode buffers overlap in a union; never live at the
  // same time.
  __shared__ __align__(sizeof(uint4)) SetupAndDecodeSmem<BLOCK_DIM_X, DecodeBuffersForMode> smem;
  uint16_t *renorm_buf = smem.decode.renorm_buf;

  // ---------- (2) Build the decoding table into shared memory ----------
  construct_decoding_table<BOUNDS_CHECK, BLOCK_DIM_X>(table, smem.build, comp_chunk, comp_chunk_size);

  // ---------- Per-warp setup ----------
  const bool is_uint8_data = !fp16_data && !runtime_is_fp8;

  // Output addressing. char: 1 byte/symbol. fp16: 2 bytes/symbol (uint16). fp8: 2
  // bytes/symbol (a decoded pair). The per-warp symbol count is in pairs for fp8.
  uint8_t *uncomp_sub_chunk;
  size_t uncomp_sub_chunk_size; // symbol (pair) count for this sub-chunk
  if (is_uint8_data)
  {
    uncomp_sub_chunk = op + max_sub_chunk_size * sub_chunk_idx;
    uncomp_sub_chunk_size = min(max_sub_chunk_size, op + uncomp_chunk_size - uncomp_sub_chunk);
  }
  else
  {
    // fp16 and fp8 both write 2 output bytes per symbol.
    uncomp_sub_chunk = op + 2 * max_sub_chunk_size * sub_chunk_idx;
    const size_t sub_chunk_start_symbol = static_cast<size_t>(sub_chunk_idx) * max_sub_chunk_size;
    uncomp_sub_chunk_size = min(max_sub_chunk_size, chunk_symbols - sub_chunk_start_symbol);
  }

  const uint8_t *sub_chunk_mantissas = mantissas_start + sub_chunk_idx * max_sub_chunk_size;

  // Per-warp sub-chunk header LDGs issued now so they overlap with the
  // table-build __syncthreads barrier below. Guarded on warp_active so the
  // trailing warps in a partial CTA never read past the offsets table.
  const bool warp_active = sub_chunk_idx * max_sub_chunk_size < chunk_symbols;
  const uint8_t *comp_sub_chunk = nullptr;
  size_t comp_sub_chunk_size = 0;
  if (warp_active)
  {
    comp_sub_chunk = ans_comp_chunk_start + safe_guard_generic<BOUNDS_CHECK>(
                                              &sub_chunk_header->get_sub_chunk_offsets()[sub_chunk_idx],
                                              ans_comp_chunk_start,
                                              comp_chunk_end
                                            );
    comp_sub_chunk_size = safe_guard_generic<BOUNDS_CHECK>(
      &sub_chunk_header->get_sub_chunk_sizes()[sub_chunk_idx],
      ans_comp_chunk_start,
      comp_chunk_end
    );
  }

  // ---------- CTA-wide barrier: make the freshly-built shared table visible ----------
  __syncthreads();

  if (!warp_active)
  {
    return;
  }

  uint16_t *warp_renorm_buf = renorm_buf + ix_warp * RENORM_DEPTH_U16;
  SubChunkDecoder sd(table, comp_sub_chunk, comp_sub_chunk_size, warp_renorm_buf);

  // Mantissa staging is only present (and only used) for the fp16 main loop;
  // fp8 reads its side-band with a plain LDG and has no mantissa_staging member.
  // The if constexpr discards the access for the fp8 specialization (where this
  // pointer is unused).
  [[maybe_unused]] uint8_t *warp_mantissa0 = nullptr;
  if constexpr (!compiletime_is_fp8)
  {
    warp_mantissa0 = smem.decode.mantissa_staging + (2 * ix_warp) * MANTISSA_WARP_STRIDE;
  }

  // signed for the decode index math
  const int num_decodes = static_cast<int>(uncomp_sub_chunk_size);

  // Decode dispatch. if constexpr on MODE so each specialization compiles ONLY its
  // decode path: Fp8 -> only fp8 (block tiles + row-map remainder); Fp16 -> only
  // fp16; Generic -> runtime char/fp16/fp8. This is what keeps each specialization's
  // register allocation lean (no other type's footprint).
  if constexpr (MODE == DecodeMode::Fp8)
  {
    // One fp8 decode for any sub-chunk size: full 256-symbol tiles (block map) +
    // a row-map remainder, the exact mirror of the encoder. num_decodes % 256 == 0
    // is just the empty-remainder case. Store granularity is picked from the output
    // pointer alignment (8 B fast path down to 1 B).
    decode_sub_chunk_fp8_dispatch(
      sd,
      uncomp_sub_chunk,
      sub_chunk_mantissas,
      warp_renorm_buf,
      num_decodes,
      lane_id,
      uncomp_chunk_size, // N (true byte count)
      actual_uncomp_chunk_size,
      status
    );
  }
  else if constexpr (MODE == DecodeMode::Fp16)
  {
    decode_sub_chunk(
      Fp16DecodePolicy{warp_mantissa0, sub_chunk_mantissas, lane_id},
      sd,
      uncomp_sub_chunk,
      warp_renorm_buf,
      num_decodes,
      lane_id,
      uncomp_chunk_size,
      actual_uncomp_chunk_size,
      status
    );
  }
  else // DecodeMode::Generic: runtime branch on the bitstream-derived predicates.
  {
    if (is_uint8_data)
    {
      decode_sub_chunk(
        CharDecodePolicy{},
        sd,
        uncomp_sub_chunk,
        warp_renorm_buf,
        num_decodes,
        lane_id,
        uncomp_chunk_size,
        actual_uncomp_chunk_size,
        status
      );
    }
    else if (runtime_is_fp8)
    {
      // One fp8 decode for any size: block tiles + row-map remainder (mirror of
      // the encoder). Note: Generic uses the default renorm depth, not the deeper
      // fp8 window, but that only affects refill frequency, not correctness.
      decode_sub_chunk_fp8_dispatch(
        sd,
        uncomp_sub_chunk,
        sub_chunk_mantissas,
        warp_renorm_buf,
        num_decodes,
        lane_id,
        uncomp_chunk_size,
        actual_uncomp_chunk_size,
        status
      );
    }
    else
    {
      decode_sub_chunk(
        Fp16DecodePolicy{warp_mantissa0, sub_chunk_mantissas, lane_id},
        sd,
        uncomp_sub_chunk,
        warp_renorm_buf,
        num_decodes,
        lane_id,
        uncomp_chunk_size,
        actual_uncomp_chunk_size,
        status
      );
    }
  }
}

__global__ void decompress_get_sizes_kernel(
  const void *const *comp_chunks,
  const size_t *comp_chunk_sizes,
  // need the other arg here for the comp block sizes to ensure safety
  size_t *actual_uncomp_chunk_sizes,
  const size_t num_uncomp_chunks
)
{
  auto chunk_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (chunk_id < num_uncomp_chunks)
  {
    actual_uncomp_chunk_sizes[chunk_id] =
      read_metadata_uncomp_block_size(comp_chunks[chunk_id], comp_chunk_sizes[chunk_id]);
  }
}

} // namespace detail
} //namespace ans_gpu_lib
