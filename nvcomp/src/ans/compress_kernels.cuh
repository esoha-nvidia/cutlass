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

#include <cuda_pipeline.h>

#include <ciso646>

#include <ans/ans_utils.cuh>
#include <ans/symbol_encoder.cuh>
#include <ans/types.cuh> // Char/Fp16/Fp8EncodePolicy (their encode_body is defined below)
#include <stdio.h>

namespace ans_gpu_lib
{
namespace detail
{

// Shared 2-step async prefetch: the "wait" half. Drain the most recent committed
// cp.async batch, then converge the warp before the staged bytes are read back. Used
// by every staged encode path (char / fp16 / fp8).
inline __device__ void async_prefetch_wait()
{
  __pipeline_wait_prior(0);
  __syncwarp();
}

// Single-stage (depth-1) single-buffered async-copy encode engine.
template <typename StagePolicy>
inline __device__ void staged_encode_pipeline(StagePolicy pol, int nloop, symbol_encoder &se)
{
  constexpr int ROWS = StagePolicy::ROWS;
  if (nloop <= 0)
  {
    return;
  }

  pol.stage(0, nloop); // stage window 0

  for (int i = 0; i < nloop - 1; i++)
  {
    uint8_t syms[ROWS];
    pol.wait_extract(i, nloop, syms); // wait + syncwarp + extract window i into registers

    // WAR: every lane must have grabbed its symbols before the single buffer is refilled.
    __syncwarp();
    pol.stage(i + 1, nloop); // refill with window i+1, in flight during the encode below

#pragma unroll
    for (int r = 0; r < ROWS; ++r)
    {
      se.encode_symbol(syms[r]);
    }
  }

  // Last window: wait + extract + encode, no refill.
  uint8_t syms[ROWS];
  pol.wait_extract(nloop - 1, nloop, syms);
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
  {
    se.encode_symbol(syms[r]);
  }
}

// =====================================================================
// Shared encode scaffold (used by all modes)
// =====================================================================

// Requires uncomp_chunk_size > 0 (empty chunks are handled at the fused kernel entry).
struct SubChunkEncodeCtx
{
  ANS_sub_chunk_header *header;
  uint8_t *tmp_op; // this sub-chunk's compressed-output slot
  IndexT in_start_idx; // first symbol index of this sub-chunk
  int sub_chunk_size; // symbols in this sub-chunk
  int idx_in_warp;
};

inline __device__ void begin_sub_chunk_encode(
  int sub_chunk_idx,
  IndexT uncomp_chunk_size, // symbols
  void *comp_chunk, // sub-chunk header base (ans_comp_chunk)
  int max_sub_chunk_size,
  SubChunkEncodeCtx &ctx
)
{
  ctx.idx_in_warp = get_lane_id();
  // ans_comp_chunk is already 8-byte aligned: output buffer is 8-aligned and
  // histogram_write_chunk_prefix advances by a multiple of sizeof(size_t).
  assert(reinterpret_cast<uintptr_t>(comp_chunk) % alignof(ANS_sub_chunk_header) == 0);
  ctx.header = reinterpret_cast<ANS_sub_chunk_header *>(comp_chunk);
  ctx.in_start_idx = static_cast<IndexT>(sub_chunk_idx) * max_sub_chunk_size;
  ctx.sub_chunk_size =
    static_cast<int>(min(uncomp_chunk_size - ctx.in_start_idx, static_cast<IndexT>(max_sub_chunk_size)));

  const IndexT max_comp_sub_chunk_size = static_cast<IndexT>(get_max_comp_sub_chunk_size(max_sub_chunk_size));
  uint8_t *const sub_chunk_0_start = ctx.header->get_sub_chunk_0_start();
  ctx.tmp_op = get_comp_sub_chunk_output_ptr(sub_chunk_idx, sub_chunk_0_start, max_comp_sub_chunk_size);

  // Requires uncomp_chunk_size > 0.
  assert(uncomp_chunk_size > 0);
  assert(ctx.in_start_idx < uncomp_chunk_size);
}

// Flush the rANS state and record this sub-chunk's compressed size in the header. Paired with
// begin_sub_chunk_encode; used at every encoder's finalize site(s).
inline __device__ void finish_sub_chunk_encode(const SubChunkEncodeCtx &ctx, int sub_chunk_idx, symbol_encoder &se)
{
  se.encode_final_state();
  if (ctx.idx_in_warp == 0)
  {
    ctx.header->get_sub_chunk_sizes()[sub_chunk_idx] = se.get_comp_stream_size(ctx.tmp_op);
  }
}

template <typename EncodePolicy>
inline __device__ void encode_sub_chunk(
  int sub_chunk_idx,
  const void *uncomp_chunk,
  IndexT uncomp_chunk_size, // symbols
  void *comp_chunk, // sub-chunk header base (ans_comp_chunk)
  int max_sub_chunk_size,
  const uint2 *shared_table,
  uint8_t *input_buf
)
{
  SubChunkEncodeCtx ctx;
  begin_sub_chunk_encode(sub_chunk_idx, uncomp_chunk_size, comp_chunk, max_sub_chunk_size, ctx);
  symbol_encoder se(ctx.idx_in_warp, shared_table, ctx.tmp_op);
  EncodePolicy::encode_body(se, ctx, uncomp_chunk, input_buf);
  finish_sub_chunk_encode(ctx, sub_chunk_idx, se);
}

// Shared rANS seeding prologue for char/fp16 (fp8 has its own geometry). Seeds the
// sub-warp remainder lanes and the first full warp row in rANS reverse order. Returns
// true when the whole sub-chunk fit in the < WARP_SIZE remainder, so the caller stops
// (encode_sub_chunk flushes).
template <typename Reader>
inline __device__ bool seed_rans_prologue(symbol_encoder &se, Reader &r, int sub_chunk_size, int idx_in_warp)
{
  const int rem = sub_chunk_size % WARP_SIZE; // symbols handled by a subset of the warp only
  const bool in_remainder = idx_in_warp < rem;
  r.seek_forward(sub_chunk_size - rem); // go to the head (high address)

  if (in_remainder)
  {
    se.encode_first_symbol(r.sym_at_lane());
  }
  r.retreat(WARP_SIZE);

  if (sub_chunk_size < WARP_SIZE)
  {
    return true; // whole sub-chunk was the remainder; caller returns, scaffold flushes
  }

  if (not in_remainder)
  {
    se.encode_first_symbol(r.sym_at_lane());
  }

  se.encode_symbol<true>(r.sym_at_lane(), in_remainder);
  return false;
}

// =====================================================================
// Char encode path
// =====================================================================

template <typename VecT>
struct CharEncodePolicy::StagePolicy
{
  static constexpr int W = ENC_WINDOW_SYMBOLS; // bytes per window (== symbols)
  static constexpr int ROWS = W / WARP_SIZE; // symbols/lane per window
  static constexpr int VECS_PER_LANE = W / (WARP_SIZE * static_cast<int>(sizeof(VecT)));
  static constexpr int VEC_WORDS = VECS_PER_LANE * WARP_SIZE; // VecT per window

  const uint8_t *input_buf_; // per-warp buffer, read back per byte
  VecT *smem_; // same buffer, written via VecT cp.async
  const VecT *ip_vec_; // gmem read pointer (VecT view), walks high -> low
  int tid_;

  __device__ void stage(int /*batch_idx*/, int /*nloop*/)
  {
#pragma unroll
    for (int k = 0; k < VECS_PER_LANE; ++k)
    {
      const auto idx = k * WARP_SIZE_U + tid_;
      __pipeline_memcpy_async(smem_ + idx, ip_vec_ + idx, sizeof(VecT));
    }
    __pipeline_commit();
    ip_vec_ -= VEC_WORDS;
  }

  __device__ void wait_extract(int /*batch_idx*/, int /*nloop*/, uint8_t syms[ROWS])
  {
    async_prefetch_wait();
#pragma unroll
    for (int r = 0; r < ROWS; ++r)
    {
      syms[r] = input_buf_[(ROWS - 1 - r) * WARP_SIZE_U + tid_];
    }
  }
};

inline __device__ void CharEncodePolicy::encode_symbols_async_pipeline(
  const uint8_t *input_buf,
  uint8_t *tmp_input_buf,
  const uint8_t *window_base,
  const int nloop,
  symbol_encoder &se
)
{
  int tid = get_lane_id();

  if (nloop <= 0)
  {
    return;
  }

  const bool window_align16 = (reinterpret_cast<uintptr_t>(window_base) % sizeof(uint4)) == 0;

  // 16-byte (uint4) staging when the window is a whole number of uint4 per lane and the
  // source window is 16-aligned (smem staging is always 16-aligned).
  if constexpr (ENC_WINDOW_SYMBOLS % (WARP_SIZE * static_cast<int>(sizeof(uint4))) == 0)
  {
    if (window_align16)
    {
      staged_encode_pipeline(
        StagePolicy<uint4>{
          input_buf,
          reinterpret_cast<uint4 *>(tmp_input_buf),
          reinterpret_cast<const uint4 *>(window_base),
          tid
        },
        nloop,
        se
      );
      return;
    }
  }

  // 4-byte (uchar4) cp.async path. CHAR input is required to be 4-byte aligned, and
  // every window_base retreat is a multiple of WARP_SIZE, so this is always reachable
  // when the uint4 path above was not taken.
  assert((reinterpret_cast<uintptr_t>(window_base) % sizeof(uchar4)) == 0);
  staged_encode_pipeline(
    StagePolicy<uchar4>{
      input_buf,
      reinterpret_cast<uchar4 *>(tmp_input_buf),
      reinterpret_cast<const uchar4 *>(window_base),
      tid
    },
    nloop,
    se
  );
}

inline __device__ void CharEncodePolicy::encode_body(
  symbol_encoder &se,
  const SubChunkEncodeCtx &ctx,
  const void *uncomp_chunk,
  uint8_t *input_buf
)
{
  const int idx_in_warp = ctx.idx_in_warp;
  const IndexT in_start_idx = ctx.in_start_idx;
  const int sub_chunk_size = ctx.sub_chunk_size;

  // Shared rANS seeding prologue. On a false return, `r.ip_` sits one full warp row
  // below the head, where the scalar remainder/window pipeline continues.
  ByteReader r{static_cast<const uint8_t *>(uncomp_chunk) + in_start_idx, idx_in_warp};
  if (seed_rans_prologue(se, r, sub_chunk_size, idx_in_warp))
  {
    return; // sub-chunk fit in the < WARP_SIZE remainder; encode_sub_chunk flushes
  }
  const uint8_t *ip = r.ip_;
  const int rem = sub_chunk_size % WARP_SIZE;

  // Remaining symbols: a WARP_SIZE-row scalar loop for the sub-window remainder, then
  // the full ENC_WINDOW_SYMBOLS windows through the async pipeline.
  const int remainder = sub_chunk_size - rem - WARP_SIZE;
  int nloop = (remainder % ENC_WINDOW_SYMBOLS) / WARP_SIZE;
  for (int i = 0; i < nloop; i++)
  {
    ip -= WARP_SIZE;
    se.encode_symbol(ip[idx_in_warp]);
  }

  nloop = remainder / ENC_WINDOW_SYMBOLS;

  ip -= ENC_WINDOW_SYMBOLS;
  encode_symbols_async_pipeline(input_buf, input_buf, ip, nloop, se);
}

// =====================================================================
// FP16 encode path
// =====================================================================

inline __device__ void
Fp16EncodePolicy::extract_exp_batch(const uint16_t *__restrict__ buf16, int tid, int shift_sym, uint8_t syms[ENC_ROWS])
{
  uint16_t fp16_vals[ENC_ROWS];

  for (int j = 0; j < ENC_ROWS; j++)
  {
    fp16_vals[j] = buf16[j * WARP_SIZE_U + tid + shift_sym];
  }

  for (int j = ENC_ROWS - 1; j >= 0; j--)
  {
    int row = ENC_ROWS - 1 - j;
    syms[row] = get_exp(fp16_vals[j]);
  }
}

template <int SHIFT_BYTES>
inline __device__ void Fp16EncodePolicy::stage_window_overread(uint8_t *tmp_input_buf, const uint8_t *src, int tid)
{
  // uint4 loads per lane to cover the window.
  constexpr int NUM_UINT4 = ENC_WINDOW_BYTES / (WARP_SIZE * static_cast<int>(sizeof(uint4)));
  uint4 *smem_uint4 = reinterpret_cast<uint4 *>(tmp_input_buf);
  const uint4 *src4 = reinterpret_cast<const uint4 *>(src - SHIFT_BYTES);
#pragma unroll
  for (int k = 0; k < NUM_UINT4; k++)
  {
    const int idx = k * WARP_SIZE_U + tid;
    __pipeline_memcpy_async(smem_uint4 + idx, src4 + idx, sizeof(uint4));
  }
  if constexpr (SHIFT_BYTES != 0)
  {
    if (tid == 0)
    {
      constexpr int idx = NUM_UINT4 * WARP_SIZE_U;
      __pipeline_memcpy_async(smem_uint4 + idx, src4 + idx, sizeof(uint4));
    }
  }
  __pipeline_commit();
}

inline __device__ void Fp16EncodePolicy::stage_window_u2(uint8_t *tmp_input_buf, const uint8_t *src, int tid)
{
  // uint2 loads per lane to cover the window.
  constexpr int NUM_UINT2 = ENC_WINDOW_BYTES / (WARP_SIZE * static_cast<int>(sizeof(uint2)));
  uint2 *smem_uint2 = reinterpret_cast<uint2 *>(tmp_input_buf);
  const uint2 *src2 = reinterpret_cast<const uint2 *>(src);
#pragma unroll
  for (int k = 0; k < NUM_UINT2; k++)
  {
    const int idx = k * WARP_SIZE_U + tid;
    __pipeline_memcpy_async(smem_uint2 + idx, src2 + idx, sizeof(uint2));
  }
  __pipeline_commit();
}

template <int SHIFT_BYTES>
struct Fp16EncodePolicy::StagePolicy
{
  static constexpr int ROWS = Fp16EncodePolicy::ENC_ROWS;
  static constexpr int SHIFT_SYM = SHIFT_BYTES >> 1;

  uint8_t *smem_; // per-warp staging buffer
  const uint16_t *ip_; // gmem read pointer (fp16 elements), walks high -> low
  int tid_;

  const __device__ uint8_t *src_bytes() const { return reinterpret_cast<const uint8_t *>(ip_); }
  const __device__ uint16_t *buf16() const { return reinterpret_cast<const uint16_t *>(smem_); }

  __device__ void stage(int batch_idx, int nloop)
  {
    if constexpr (SHIFT_BYTES == 0)
    {
      // 16-aligned: uint4 over-read (SHIFT 0 = exact) for every window.
      Fp16EncodePolicy::stage_window_overread<0>(smem_, src_bytes(), tid_);
    }
    else
    {
      // 8-off: interior windows use the uint4 over-read; the lowest window is the
      // plain uint2 (no over-read, no OOB below the sub-chunk start).
      if (batch_idx == nloop - 1)
      {
        Fp16EncodePolicy::stage_window_u2(smem_, src_bytes(), tid_);
      }
      else
      {
        Fp16EncodePolicy::stage_window_overread<SHIFT_BYTES>(smem_, src_bytes(), tid_);
      }
    }
    ip_ -= Fp16EncodePolicy::ENC_WINDOW_SYMBOLS;
  }

  __device__ void wait_extract(int batch_idx, int nloop, uint8_t syms[ROWS])
  {
    async_prefetch_wait();
    // The uint2-staged lowest window lands with no shift; uint4 over-read needs SHIFT_SYM.
    const int shift_sym = (SHIFT_BYTES == 0) ? 0 : ((batch_idx == nloop - 1) ? 0 : SHIFT_SYM);
    Fp16EncodePolicy::extract_exp_batch(buf16(), tid_, shift_sym, syms);
  }
};

// Encapsulates the per-warp input read pointer (tmp_ip) and the shared staging buffer
// for one sub-chunk encode. The read pointer
// walks head -> tail (high address -> low); shift_bytes (0 or 8) is the
// sub-chunk alignment. All cp.async staging / buffer state lives here.
class Fp16EncodePolicy::InputReader
{
  const uint16_t *ip_; // input read pointer (fp16 elements), moves head -> tail
  uint8_t *smem_; // per-warp staging buffer
  int tid_; // lane id
  int shift_bytes_; // 0 (16-aligned) or 8 (8-off)

  const __device__ uint16_t *buf16() const { return reinterpret_cast<const uint16_t *>(smem_); }

public:
  __device__ InputReader(const uint16_t *ip, uint8_t *smem, int tid, int shift_bytes)
      : ip_(ip)
      , smem_(smem)
      , tid_(tid)
      , shift_bytes_(shift_bytes)
  {}

  // ---- read pointer ops (prologue / rANS init, and the scalar head_partial path) ----
  __device__ void seek_forward(int symbols) { ip_ += symbols; }
  __device__ void retreat(int symbols) { ip_ -= symbols; }
  // The fp16 "symbol" is the exponent byte.
  __device__ uint8_t sym_at_lane() const { return Fp16EncodePolicy::get_exp(ip_[tid_]); }

  // ---- head_partial (the < 512 batch just below the prologue) ----
  // Stage from read pointer - 512: exact uint4 when 16-aligned, uint2 when 8-off; both
  // read exactly the window (no over-read, no readback shift).
  __device__ void prefetch_head_partial()
  {
    const uint8_t *src = reinterpret_cast<const uint8_t *>(ip_ - Fp16EncodePolicy::ENC_WINDOW_SYMBOLS);
    if (shift_bytes_ == 0)
    {
      Fp16EncodePolicy::stage_window_overread<0>(smem_, src, tid_);
    }
    else
    {
      Fp16EncodePolicy::stage_window_u2(smem_, src, tid_);
    }
  }
  // Exponent byte of a staged head_partial row (caller has already waited).
  __device__ uint8_t staged_row_exp(int row_in_buf) const
  {
    return Fp16EncodePolicy::get_exp(buf16()[row_in_buf * WARP_SIZE_U + tid_]);
  }

  // ---- steady-state pipeline ----
  template <int SHIFT_BYTES>
  __device__ void run_pipeline(int nloop, symbol_encoder &se)
  {
    if (nloop <= 0)
    {
      return;
    }

    // WAR guard: lagging lanes may still be reading smem_ from the head_partial
    // batch (staged_row_exp) when the first stage() overwrites it via cp.async.
    __syncwarp();

    staged_encode_pipeline(StagePolicy<SHIFT_BYTES>{smem_, ip_, tid_}, nloop, se);
  }
};

inline __device__ void Fp16EncodePolicy::encode_body(
  symbol_encoder &se,
  const SubChunkEncodeCtx &ctx,
  const void *uncomp_chunk,
  uint8_t *input_buf
)
{
  const int idx_in_warp = ctx.idx_in_warp;
  const IndexT in_start_idx = ctx.in_start_idx;
  const int sub_chunk_size = ctx.sub_chunk_size;

  const uint16_t *ip = reinterpret_cast<const uint16_t *>(uncomp_chunk) + in_start_idx; // raw fp16
  // Alignment of this sub-chunk, either 0 or 8.
  assert((reinterpret_cast<uintptr_t>(ip) % sizeof(uint2)) == 0);
  const int shift_bytes = static_cast<int>(reinterpret_cast<uintptr_t>(ip) % sizeof(uint4));

  // The reader (fp16 buffering class) owns the input read pointer and staging buffer.
  InputReader reader(ip, input_buf, idx_in_warp, shift_bytes);

  // Shared rANS seeding prologue (same as char, reading exponent bytes via the reader).
  if (seed_rans_prologue(se, reader, sub_chunk_size, idx_in_warp))
  {
    return; // encode_sub_chunk flushes
  }
  const int rem = sub_chunk_size % WARP_SIZE;

  int remainder = sub_chunk_size - rem - WARP_SIZE;
  int head_partial_nloop = (remainder % ENC_WINDOW_SYMBOLS) / WARP_SIZE;

  const bool head_partial_prefetch_safe = head_partial_nloop > 0 &&
                                          (in_start_idx + remainder) >= static_cast<IndexT>(ENC_WINDOW_SYMBOLS);

  if (head_partial_prefetch_safe)
  {
    // Prefetch the head_partial batch (no over-read, no readback shift; never
    // reads below the sub-chunk start), then encode its rows from the buffer.
    reader.prefetch_head_partial();
    async_prefetch_wait();

    for (int i = 0; i < head_partial_nloop; i++)
    {
      remainder -= WARP_SIZE;
      int row_in_buf = (ENC_ROWS - 1) - i;
      se.encode_symbol(reader.staged_row_exp(row_in_buf));
    }
    reader.retreat(WARP_SIZE * head_partial_nloop);
  }
  else
  {
    for (int i = 0; i < head_partial_nloop; i++)
    {
      remainder -= WARP_SIZE;
      reader.retreat(WARP_SIZE);
      se.encode_symbol(reader.sym_at_lane());
    }
  }

  // Steady-state pipeline: 512-symbol outer iters (wide cp.async prefetch).
  int pipe_nloop = remainder / ENC_WINDOW_SYMBOLS;
  if (pipe_nloop > 0)
  {
    reader.retreat(ENC_WINDOW_SYMBOLS);
    if (shift_bytes == 0)
    {
      reader.run_pipeline<0>(pipe_nloop, se);
    }
    else
    {
      reader.run_pipeline<8>(pipe_nloop, se);
    }
  }
}

// =====================================================================
// FP8 encode path
// =====================================================================

inline __device__ void Fp8EncodePolicy::prefetch_tile_raw(uint8_t *stage, const uint8_t *src, int lane_id)
{
  uint2 *stage2 = reinterpret_cast<uint2 *>(stage);
  const uint2 *src2 = reinterpret_cast<const uint2 *>(src);
#pragma unroll
  for (int h = 0; h < 2; ++h)
  {
    const int w = h * WARP_SIZE + lane_id;
    __pipeline_memcpy_async(stage2 + w, src2 + w, sizeof(uint2));
  }
}

inline __device__ uint8_t Fp8EncodePolicy::packed_exp_at(const uint8_t *raw_base, int sym_ix)
{
  const uint8_t b0 = raw_base[2 * sym_ix + 0];
  const uint8_t b1 = raw_base[2 * sym_ix + 1];
  return static_cast<uint8_t>(get_exp(b0) | (get_exp(b1) << 4));
}

// FP8 (E4M3) sub-chunk encoder. Recomputes packed_exp on the fly by re-reading the
// raw FP8 input and re-splitting each pair (mirroring the fp16 encoder, which
// re-reads raw fp16 and extracts the exponent) -- there is no packed_exp gmem
// scratch. It never touches the (separately evolving) fp16 encoder machinery.
//
// One mapping for any sub-chunk size: the sub-chunk is full 256-symbol tiles in
// BLOCK order (lane t owns the contiguous run [tile*256 + 8*t .. +7]) followed by
// a row-map remainder rem = sub_chunk_size % 256 (lane t owns {rem_base + r*32 + t}).
// rem == 0 (the common case, sub_chunk_size a multiple of 256) is just the empty
// remainder. The decoder (decode_sub_chunk_fp8) consumes the exact mirror: tiles
// 0..full forward, then the remainder rows, then the partial row.
//
// Per tile, lane t's 8 symbols are the 16 contiguous raw bytes at
// raw_base + 2*(tile*256 + 8*t); these are staged through shared via a depth-2
// cp.async pipeline (tile T-1's load overlaps tile T's encode) and re-split into 8
// packed_exp bytes on readback, so the global loads are off the critical path. The
// (rare) remainder uses per-symbol raw reads. stage_buf is a per-warp staging
// buffer of Fp8EncodePolicy::ENC_BUF_BYTES (two raw-tile ping-pong slots).
inline __device__ void Fp8EncodePolicy::encode_body(
  symbol_encoder &se,
  const SubChunkEncodeCtx &ctx,
  const void *uncomp_chunk, // raw FP8 input base (full chunk)
  uint8_t *stage_buf // per-warp staging (depth-2 ping-pong; ENC_BUF_BYTES)
)
{
  // Tile geometry is defined by the policy (see Fp8EncodePolicy in types.cuh); the
  // block exp load requires each lane to own B contiguous symbols per tile.
  constexpr int B = ENC_SYMBOLS_PER_LANE;
  constexpr int TILE = ENC_TILE_SYMBOLS;

  const int idx_in_warp = ctx.idx_in_warp;
  const IndexT in_start_idx = ctx.in_start_idx;
  const int sub_chunk_size = ctx.sub_chunk_size;

  // Raw bytes for this sub-chunk's first symbol (2 raw bytes per pair).
  const uint8_t *raw_base = static_cast<const uint8_t *>(uncomp_chunk) + 2 * in_start_idx;

  // Full 256-symbol tiles (block map) + a row-map remainder rem (0..255).
  const int full_tiles = sub_chunk_size / TILE;
  const int rem = sub_chunk_size - full_tiles * TILE; // 0..255
  const int rem_base = full_tiles * TILE;
  const int rem_full_rows = rem / WARP_SIZE; // full 32-rows in the remainder
  const int rpart = rem - rem_full_rows * WARP_SIZE; // 0..31 partial-row lanes
  const bool in_partial = idx_in_warp < rpart;

  // Seeding generalized across the remainder->tiles boundary. Three groups of
  // symbols in rANS reverse order:
  //   (P) the partial row     : rpart lanes,  symbol rem_base + rem_full_rows*32 + t
  //   (R) full remainder rows : rem_full_rows rows, row map
  //   (T) full tiles          : full_tiles tiles, block map
  // The "first full group" is the globally-last 32-wide group AFTER the partial
  // row: the last full remainder row if rem_full_rows>0, else the last tile's
  // last symbol (k=B-1). At that group: partial lanes already seeded -> encode
  // with active gating; the other lanes seed it (encode_first_symbol). After
  // that, every symbol is a plain warp-wide encode_symbol.

  // (P) Seed the partial-row lanes (ballot-free subset).
  if (in_partial)
  {
    const int sym_ix = rem_base + rem_full_rows * WARP_SIZE + idx_in_warp;
    se.encode_first_symbol(packed_exp_at(raw_base, sym_ix));
  }

  // (R) Full remainder rows, high -> low.
  for (int r = rem_full_rows - 1; r >= 0; --r)
  {
    const uint8_t s = packed_exp_at(raw_base, rem_base + r * WARP_SIZE + idx_in_warp);
    if (r == rem_full_rows - 1)
    {
      // First full group: non-partial lanes seed, partial lanes encode (active).
      if (!in_partial)
      {
        se.encode_first_symbol(s);
      }
      se.encode_symbol<true>(s, in_partial);
    }
    else
    {
      se.encode_symbol(s);
    }
  }

  {
    constexpr int TILE_RAW_BYTES = ENC_TILE_RAW_BYTES; // 512 (16 raw bytes/lane)
    constexpr int STAGE_STRIDE = TILE_RAW_BYTES; // ping-pong slot stride
    static_assert(ENC_BUF_BYTES == 2 * STAGE_STRIDE, "stage_buf must hold exactly two ping-pong slots");
    const bool first_full_is_tile = (rem_full_rows == 0);

    if (full_tiles > 0)
    {
      // Pre-stage the first encoded tile (highest T) into its ping-pong slot.
      // Racecheck flag: without this, one thread in the warp can be reading the last tile of a sub-chunk
      // while another thread is prefetching the first tile of the next sub-chunk.
      __syncwarp();
      const int T0 = full_tiles - 1;
      prefetch_tile_raw(stage_buf + (T0 & 1) * STAGE_STRIDE, raw_base + T0 * TILE_RAW_BYTES, idx_in_warp);
      __pipeline_commit();
    }

    for (int T = full_tiles - 1; T >= 0; --T)
    {
      async_prefetch_wait();

      // Prefetch the next tile to encode (T-1) into the other slot, overlapping
      // this tile's encode.
      if (T - 1 >= 0)
      {
        prefetch_tile_raw(stage_buf + ((T - 1) & 1) * STAGE_STRIDE, raw_base + (T - 1) * TILE_RAW_BYTES, idx_in_warp);
        __pipeline_commit();
      }

      // Lane's 16 contiguous raw bytes for this tile from the staged buffer (two
      // uint2 LDS; cur_stage + 16*lane is 8 B-aligned), re-split into the 8
      // packed_exp bytes pe[0..7] (fp8_quad_packed_exp_e4m3: 4 raw bytes -> 2
      // packed_exp; exp-only, the side-band was already written by the histogram).
      const uint8_t *cur_stage = stage_buf + (T & 1) * STAGE_STRIDE;
      static_assert(B == 8, "the two-uint2 load and the packed_exp unroll below are hand-written for 8 symbols/lane");
      const uint2 *lane_raw = reinterpret_cast<const uint2 *>(cur_stage + 2 * B * idx_in_warp);
      uint2 raw0 = lane_raw[0]; // pairs 0..3
      uint2 raw1 = lane_raw[1]; // pairs 4..7
      __align__(4) uint8_t pe[B];
      {
        uint32_t e_lo16;
        e_lo16 = fp8_quad_packed_exp_e4m3(raw0.x);
        pe[0] = static_cast<uint8_t>(e_lo16 & 0xFFu);
        pe[1] = static_cast<uint8_t>((e_lo16 >> 8) & 0xFFu);
        e_lo16 = fp8_quad_packed_exp_e4m3(raw0.y);
        pe[2] = static_cast<uint8_t>(e_lo16 & 0xFFu);
        pe[3] = static_cast<uint8_t>((e_lo16 >> 8) & 0xFFu);
        e_lo16 = fp8_quad_packed_exp_e4m3(raw1.x);
        pe[4] = static_cast<uint8_t>(e_lo16 & 0xFFu);
        pe[5] = static_cast<uint8_t>((e_lo16 >> 8) & 0xFFu);
        e_lo16 = fp8_quad_packed_exp_e4m3(raw1.y);
        pe[6] = static_cast<uint8_t>(e_lo16 & 0xFFu);
        pe[7] = static_cast<uint8_t>((e_lo16 >> 8) & 0xFFu);
      }
#pragma unroll
      for (int k = B - 1; k >= 0; --k)
      {
        const uint8_t s = pe[k];
        const bool is_first_full_group = first_full_is_tile && (T == full_tiles - 1) && (k == B - 1);
        if (is_first_full_group)
        {
          // Mirrors (R)'s first-full-group seed. If rpart>0: non-partial lanes seed,
          // partial lanes encode (active). If rpart==0 (no partial row at all): all
          // lanes seed here uniformly, no encode.
          if (rpart > 0)
          {
            if (!in_partial)
            {
              se.encode_first_symbol(s);
            }
            se.encode_symbol<true>(s, in_partial);
          }
          else
          {
            se.encode_first_symbol(s);
          }
        }
        else
        {
          se.encode_symbol(s);
        }
      }
    }
  }
}

} // end namespace detail
} // end namespace ans_gpu_lib
