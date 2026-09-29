/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include <cuda_pipeline.h>

#include "nvcomp/shared_types.h"
#include "nvcomp/utils.hpp"

#include <ans/ans_arch_profile.cuh>
#include <ans/ans_utils.cuh>

namespace ans_gpu_lib
{
namespace detail
{

template <bool BOUNDS_CHECK>
class bounds_check_state;

template <>
class bounds_check_state<false>
{
protected:
  __device__ void init_bounds(const uint16_t *, const uint16_t *) {}

  template <typename T>
  __device__ T guard_load(const T *address) const
  {
    return *address;
  }

public:
  __device__ nvcompStatus_t getError() const { return nvcompSuccess; }
};

template <>
class bounds_check_state<true>
{
  const uint16_t *bit_stream_start_;
  const uint16_t *bit_stream_end_;
  // mutable so guard_load (const) can record a bounds-check failure.
  mutable nvcompStatus_t error_ = nvcompSuccess;

protected:
  __device__ void init_bounds(const uint16_t *start, const uint16_t *end)
  {
    bit_stream_start_ = start;
    bit_stream_end_ = end;
  }

  template <typename T>
  __device__ T guard_load(const T *address) const
  {
    return safe_guard_generic<true, const T, const T>(address, bit_stream_start_, bit_stream_end_, &error_);
  }

public:
  __device__ nvcompStatus_t getError() const { return error_; }
};

// Shared core for both decoders (host `symbol_decoder` and nvcompDx
// `symbol_decoder_dx`): the rANS `state_`, the renorm threshold, the
// bounds-check plumbing (`safe_guard` + a couple of protected bounds
// accessors), and the constructor that loads the initial per-lane state from
// the tail of the bit stream. Each decoder owns its own table format, renorm
// strategy, and decode loop; none of that lives here.
template <bool BOUNDS_CHECK>
class symbol_decoder_base : public bounds_check_state<BOUNDS_CHECK>
{
public:
  const uint16_t *bit_stream_;

protected:
  uint32_t state_;

  static constexpr uint32_t state_renorm_thresh_ = 1 << 16;

  template <typename T>
  __device__ T safe_guard(const T *address)
  {
    return this->guard_load(address);
  }

public:
  __device__ symbol_decoder_base(const uint8_t *comp_sub_chunk, size_t cs_size)
  {
    bit_stream_ = reinterpret_cast<const uint16_t *>(comp_sub_chunk);
    const uint16_t *bit_stream_end = bit_stream_ + cs_size / 2;

    this->init_bounds(bit_stream_, bit_stream_end);
    int lid = get_lane_id();
    uint32_t lo = safe_guard(&bit_stream_end[-2 * lid - 2]);
    uint32_t hi = safe_guard(&bit_stream_end[-2 * lid - 1]);

    state_ = (hi << 16) | lo;
  }
};

// nvcompDx decoder. Renormalizes from a sliding per-warp decoding buffer in
// shared memory (refilled by `refill_buffer`) and reads the rANS tables via a
// separate `symbol_table_` + `cdf_table_`. Used only by
// libnvcompdx/src/ans/decompress_device.cu.
template <bool BOUNDS_CHECK>
class symbol_decoder_dx : public symbol_decoder_base<BOUNDS_CHECK>
{
  const uint8_t *symbol_table_;
  const uint16_t *cdf_table_;
  uint16_t *decoding_buffer_;
  uint16_t decoding_buf_size_;
  uint32_t bit_stream_offset_;

  using symbol_decoder_base<BOUNDS_CHECK>::bit_stream_;
  using symbol_decoder_base<BOUNDS_CHECK>::state_;
  using symbol_decoder_base<BOUNDS_CHECK>::state_renorm_thresh_;

  // Renormalize from the sliding decoding buffer. `read_buf` points just past
  // the last valid uint16 in the buffer; renormalizing lanes pull the next
  // word at `read_buf - offset`, bounds-checked via the base `safe_guard`.
  __device__ uint32_t
  transition_state_and_renormalize(uint32_t pdf, uint32_t smcdf, const uint16_t *read_buf, bool active)
  {
    state_ = pdf * (state_ >> DEFAULT_TABLELOG) + smcdf;

    bool need_to_renormalize = (state_ < state_renorm_thresh_) & active;
    uint32_t reading = __ballot_sync(WARP_ALL, need_to_renormalize);

    if (need_to_renormalize)
    {
      short offset = __popc(reading & get_lane_mask());
      state_ = (state_ << 16) | this->safe_guard(read_buf - offset);
    }

    return __popc(reading);
  }

  __device__ uint32_t transition_state_and_renormalize(uint32_t pdf, uint32_t smcdf, const uint16_t *read_buf)
  {
    state_ = pdf * (state_ >> DEFAULT_TABLELOG) + smcdf;

    bool need_to_renormalize = state_ < state_renorm_thresh_;
    uint32_t reading = __ballot_sync(WARP_ALL, need_to_renormalize);

    if (need_to_renormalize)
    {
      short offset = __popc(reading & get_lane_mask());
      state_ = (state_ << 16) | this->safe_guard(read_buf - offset);
    }

    return __popc(reading);
  }

public:
  __device__ symbol_decoder_dx(
    const uint8_t *symbol_table,
    const uint16_t *cdf_table,
    uint8_t *decoding_buffer,
    const uint8_t *comp_sub_chunk,
    size_t cs_size
  )
      : symbol_decoder_base<BOUNDS_CHECK>(comp_sub_chunk, cs_size)
      , symbol_table_{symbol_table}
      , cdf_table_{cdf_table}
  {

    bit_stream_offset_ = cs_size / 2 - 64;
    decoding_buffer_ = reinterpret_cast<uint16_t *>(decoding_buffer);

    int16_t num_16b_to_load = min(128, bit_stream_offset_);
    bit_stream_offset_ -= num_16b_to_load;

    int lid = get_lane_id();
    for (int i = lid; i < num_16b_to_load; i += WARP_SIZE)
    {
      decoding_buffer_[i] = bit_stream_[bit_stream_offset_ + i];
    }
    __syncwarp();

    decoding_buf_size_ = num_16b_to_load;
  }

  __device__ uint8_t get_symbol()
  {
    // FIXME: Why didn't this matter when it was wrong?
    return symbol_table_[state_ & ((1 << DEFAULT_TABLELOG) - 1)];
  }

  __device__ void refill_buffer()
  {
    int lid = get_lane_id();
    if (bit_stream_offset_ < 64)
    {
      for (int i = 32 + lid; i >= 0; i -= WARP_SIZE)
      {
        uint16_t tmp = decoding_buffer_[i];
        __syncwarp();
        decoding_buffer_[bit_stream_offset_ + i] = tmp;
      }
      __syncwarp();

      for (int i = lid; i < bit_stream_offset_; i += WARP_SIZE)
      {
        decoding_buffer_[i] = bit_stream_[i];
      }
      __syncwarp();

      decoding_buf_size_ += bit_stream_offset_;
      bit_stream_offset_ = 0;
    }
    else
    {
      bit_stream_offset_ -= 64;

      // Note: transition_state_and_renormalize might still be reading
      //       the buffer we are about to refill
      __syncwarp();

      for (int i = lid; i < 64; i += WARP_SIZE)
      {
        decoding_buffer_[i + 64] = decoding_buffer_[i];
        __syncwarp();
        decoding_buffer_[i] = bit_stream_[bit_stream_offset_ + i];
      }

      decoding_buf_size_ += 64;
      __syncwarp();
    }
  }

  __device__ void read_from_tables(uint32_t &sym, uint32_t &pdf, uint32_t &smcdf)
  {
    uint32_t idx = state_ & ((1U << DEFAULT_TABLELOG) - 1U);
    sym = symbol_table_[idx];
    uint32_t cdf0 = cdf_table_[sym];
    uint32_t cdf1 = cdf_table_[sym + 1];
    pdf = cdf1 - cdf0;
    smcdf = idx - cdf0;
  }

  __device__ uint8_t decode_symbol(bool active)
  {
    uint32_t sym, pdf, smcdf;
    read_from_tables(sym, pdf, smcdf);

    transition_state_and_renormalize(pdf, smcdf, decoding_buffer_ + decoding_buf_size_, active);

    return sym;
  }

  // Different API to decode_symbol to reduce register usage
  __device__ void decode_symbol_full(uint8_t *res)
  {
    uint32_t sym, pdf, smcdf;
    read_from_tables(sym, pdf, smcdf);

    *res = sym;
    uint16_t num_reading = transition_state_and_renormalize(pdf, smcdf, decoding_buffer_ + decoding_buf_size_);

    decoding_buf_size_ -= num_reading;
    if (decoding_buf_size_ < 64)
    {
      // Note: refill_buffer() internally performs a syncwarp() before
      //       commencing the update of the decoding buffer.
      refill_buffer();
    }
  }
};

// RENORM_BUF_U16 is the per-warp renorm buffer depth (uint16) this decoder reads
// from / refills; REFILL_THRESH_U16 is the cursor low-water mark that triggers a
// refill. They default to the global fp16/char tuning so existing instantiations
// are unchanged; fp8 instantiates with a deeper buffer (it reclaims the
// mantissa-staging shared memory it no longer needs).
template <
  bool BOUNDS_CHECK,
  uint32_t RENORM_BUF_U16 = RENORM_PREFETCH_BUF_SIZE_U16,
  uint32_t REFILL_THRESH_U16 = RENORM_REFILL_THRESHOLD_U16>
class symbol_decoder : public symbol_decoder_base<BOUNDS_CHECK>
{
public:
  // Exposes the bounds-check flag so callers that take the decoder type directly
  // (rather than its template args) can pull it back off the type.
  static constexpr bool bounds_check = BOUNDS_CHECK;

private:
  // 32-bit shared-mem offset of the per-CTA decoding table. Set once in the
  // ctor via __cvta_generic_to_shared.
  uint32_t table_off_;

  // Per-warp renorm buffer in shared memory, holding RENORM_PREFETCH_BUF_SIZE_U16
  // uint16 (RENORM_PREFETCH_BUF_SIZE bytes). The buffer holds multiple outer
  // iters' worth of renorm data; buf_pos_ is a continuous uint16 cursor that
  // decreases as symbols are consumed and is only re-anchored (via
  // advance_bitstream_pos) + refilled (via prefetch_renorm_buffer) when it drops
  // below RENORM_REFILL_THRESHOLD_U16. LDGSTS goes through L1 so the redundant
  // lines on a refill are cheap.
  uint16_t *renorm_buf_;

  // buf_pos_ tracks the cursor into the renorm
  // buffer (set via the ctor or advance_bitstream_pos, decremented by
  // __popc(reading) per call), so the read offset is always in
  // [0, RENORM_PREFETCH_BUF_SIZE_U16) and no wrap mask is needed.
  uint32_t buf_pos_;

  // Absolute pointer into the global bit stream pointing at the end of the
  // window we'll reload at the next outer iter. Initialized in the ctor to
  // the rounded-up renorm_end and advanced by full 16 B units per outer iter
  // via advance_bitstream_pos.
  const uint16_t *bit_stream_pos_;

  using symbol_decoder_base<BOUNDS_CHECK>::bit_stream_;
  using symbol_decoder_base<BOUNDS_CHECK>::state_;

  // Branch-free renormalization. We always issue the shared-memory load and
  // compute the renormalized state, then pick the result with a select. This
  // keeps the warp converged across the unrolled decode loop so the compiler
  // can schedule ILP across iterations instead of emitting a divergent @!P
  // BRA + reconverge per symbol.
  __device__ uint32_t transition_state_and_renormalize(uint32_t pdf, uint32_t smcdf)
  {
    state_ = pdf * (state_ >> DEFAULT_TABLELOG) + smcdf;

    bool need_to_renormalize = state_ < symbol_decoder_base<BOUNDS_CHECK>::state_renorm_thresh_;
    uint32_t reading = __ballot_sync(WARP_ALL, need_to_renormalize);

    uint32_t offset = __popc(reading & get_lane_mask());

    // Unconditionally read the U16, but don't use the result if we don't need to renormalize
    // This can lead to a load outside of the RENORM_PREFETCH_BUF_SIZE_U16 range. We handle
    // this by ensuring the renorm buf has one extra U16 at the end.
    // Racecheck flags this, since the load can access another warp's buffer, but it is safe.
    uint32_t word = renorm_buf_[buf_pos_ - offset];
    uint32_t renorm_state = (state_ << 16) | word;
    state_ = need_to_renormalize ? renorm_state : state_;

    return __popc(reading);
  }

  // Single-LDS table lookup. The address calc uses `mad.lo.u32 addr, idx, 4,
  // table_off` so the *4 byte-scaling for uint32 indexing and the addition
  // of the runtime base happen in one IMAD.
  __device__ uint32_t lookup_decoding_table(uint32_t state) const
  {
    uint32_t entry;
    asm volatile("{\n\t"
                 " .reg .u32 addr;\n\t"
                 " and.b32 addr, %2, %3;\n\t" // idx = state & 511
                 " mad.lo.u32 addr, addr, %4, %1;\n\t" // addr = idx*4 + table_off
                 " ld.shared.u32 %0, [addr];\n\t"
                 "}"
                 : "=r"(entry)
                 : "r"(table_off_),
                   "r"(state),
                   "n"((1u << DEFAULT_TABLELOG) - 1u),
                   "n"(static_cast<uint32_t>(sizeof(uint32_t))));
    return entry;
  }

  __device__ void read_from_tables(uint32_t &sym, uint32_t &pdf, uint32_t &smcdf)
  {
    // Entry layout: bits 0..7 = sym, 8..19 = pdf (12 bits), 20..31 = smcdf.
    // Smart unpack:
    //   - sym is byte 0 of entry; we pass the whole register and rely on the
    //     uint8_t cast at decode_symbol_full's return / the FP16 output
    //     path's byte_perm to drop the high bits.
    //   - pdf needs explicit mask after the shift (bits above 12 are smcdf).
    //   - smcdf needs only a shift; bits 12+ are zero after `>> 20` on
    //     uint32 (no mask op).
    uint32_t entry = lookup_decoding_table(state_);
    sym = entry;
    pdf = (entry >> 8) & 0xFFFu;
    smcdf = entry >> 20;
  }

  // Advance bit_stream_pos_ by as many full 16 B units (= 8 uint16) as we've
  // fully consumed since the last refill; the leftover (T mod 8) uint16 of a
  // partially-consumed unit stays valid and is reloaded into the top of the
  // refilled buffer.
  //
  // Sets buf_pos_ so the next read (into the refilled buffer) still targets the
  // next-to-be-consumed uint16: buf_pos_new = RENORM_PREFETCH_BUF_SIZE_U16 -
  // (T mod 8), where T = RENORM_PREFETCH_BUF_SIZE_U16 - buf_pos_ is the uint16
  // count consumed since the last refill.
  __device__ void advance_bitstream_pos()
  {
    uint32_t consumed = RENORM_BUF_U16 - buf_pos_;
    uint32_t whole_units = consumed & ~7u;
    uint32_t leftover = consumed - whole_units;
    bit_stream_pos_ -= whole_units;
    buf_pos_ = RENORM_BUF_U16 - leftover;
  }

public:
  __device__ symbol_decoder(const uint32_t *table, const uint8_t *comp_sub_chunk, size_t cs_size, uint16_t *renorm_buf)
      : symbol_decoder_base<BOUNDS_CHECK>(comp_sub_chunk, cs_size)
      , table_off_{static_cast<uint32_t>(__cvta_generic_to_shared(table))}
      , renorm_buf_{renorm_buf}
  {
    if (cs_size == 0)
    {
      buf_pos_ = 0;
      bit_stream_pos_ = this->bit_stream_;
      __syncwarp();
      return;
    }

    const uint16_t *bs = symbol_decoder_base<BOUNDS_CHECK>::bit_stream_;

    // renorm_end is the first uint16 *past* the renorm region. The 64-uint16
    // tail starting at renorm_end is the leading state of each lane and is
    // loaded separately by the caller. The window we'll load is the
    // RENORM_PREFETCH_BUF_SIZE_U16 uint16 immediately preceding bit_stream_pos_.
    //
    // Round bit_stream_pos_ UP to a 16-byte boundary so prefetch_renorm_buffer's
    // uint4 LDGSTS source (= bit_stream_pos_ - RENORM_PREFETCH_BUF_SIZE_U16) is 16-byte
    // aligned. The up-to-7 uint16 between the original renorm_end and the
    // rounded-up boundary belong to the buffer of initial states for this subchunk (no OOB); they get loaded
    // into the top of renorm_buf but are never read, because buf_pos_ starts
    // at (RENORM_PREFETCH_BUF_SIZE_U16 - extra_uint16) so the first read targets
    // renorm_end - 1 at renorm_buf[RENORM_PREFETCH_BUF_SIZE_U16 - 1 - extra_uint16].
    // advance_bitstream_pos preserves the alignment by only advancing in
    // multiples of 8 uint16 (= 16 B).
    const uint16_t *renorm_end = bs + cs_size / 2 - 64;
    uintptr_t renorm_end_addr = reinterpret_cast<uintptr_t>(renorm_end);
    uintptr_t aligned_addr = reinterpret_cast<uintptr_t>(nvcomp::roundUpToAlignment<uint4>(renorm_end));
    uint32_t extra_uint16 = static_cast<uint32_t>((aligned_addr - renorm_end_addr) / sizeof(uint16_t));
    bit_stream_pos_ = reinterpret_cast<const uint16_t *>(aligned_addr);
    buf_pos_ = RENORM_BUF_U16 - extra_uint16;
  }

  // Re-anchor (advance_bitstream_pos) and prefetch the full
  // RENORM_PREFETCH_BUF_SIZE_U16 uint16 renorm buffer from the bit stream into
  // the warp's renorm shared buffer: each lane issues
  // RENORM_PREFETCH_BUF_SIZE_U16 / (WARP_SIZE * 8) uint4 LDGSTS (32 bytes per
  // lane per LDGSTS). The reload overwrites the whole buffer to avoid complexity
  // from moving already-fetched data within the caches.
  //
  // advance_bitstream_pos is a no-op on the initial fill (nothing consumed yet,
  // buf_pos_ at its ctor value) and re-anchors bit_stream_pos_ by the fully
  // consumed 16 B units on a refill, so it is folded in here unconditionally.
  //
  // When `sync`, commit + wait(0) + __syncwarp so renorm_buf_ is fully
  // populated on return. When !sync, only commit the LDGSTS (leaving it in
  // flight, batched with any async copies the caller prefetched beforehand); the
  // caller is then responsible for the wait/__syncwarp before reading renorm_buf_.
  __device__ void prefetch_renorm_buffer(uint16_t *renorm_buf_for_warp, bool sync)
  {
    // Note: transition_state_and_renormalize might still be reading
    //       the buffer we are about to refill
    __syncwarp();
    advance_bitstream_pos();

    int lid = get_lane_id();
    uint4 *dst = reinterpret_cast<uint4 *>(renorm_buf_for_warp);
    const uint4 *src = reinterpret_cast<const uint4 *>(bit_stream_pos_ - RENORM_BUF_U16);
    constexpr uint32_t UINT4_PER_LANE = RENORM_BUF_U16 / (WARP_SIZE * (sizeof(uint4) / sizeof(uint16_t)));
    static_assert(
      UINT4_PER_LANE * WARP_SIZE * (sizeof(uint4) / sizeof(uint16_t)) == RENORM_BUF_U16,
      "RENORM_BUF_U16 must be a multiple of WARP_SIZE * 8"
    );

// Clamp each read to the start of the bitstream, aligned to 16B:
#ifdef IMPLICIT_BUFFER_ALIGNMENT
    const uint4 *src_lo =
      reinterpret_cast<const uint4 *>(reinterpret_cast<uintptr_t>(this->bit_stream_) & ~uintptr_t(15));

#pragma unroll
    for (uint32_t i = 0; i < UINT4_PER_LANE; ++i)
    {
      const uint4 *s = src + i * WARP_SIZE + lid;
      s = s < src_lo ? src_lo : s;
      uint4 *d = dst + i * WARP_SIZE + lid;
      // This can never OOB if the buffer is valid, since the upper bound is guarded as explained in the constructor, and
      // the lower bound takes advantage of the implicit buffer alignment
      __pipeline_memcpy_async(d, reinterpret_cast<const void *>(s), sizeof(uint4));
    }
#else
#error "Not implemented"
#endif // IMPLICIT_BUFFER_ALIGNMENT

    __pipeline_commit();
    if (sync)
    {
      __pipeline_wait_prior(0);
      __syncwarp();
    }
  }

  // True when the renorm buffer can no longer satisfy two worst-case outer
  // iters and must be re-anchored + refilled before the next consume.
  __device__ bool needs_refill() const { return buf_pos_ < REFILL_THRESH_U16; }

  // Re-anchor + refill only when the buffer can no longer satisfy a worst-case
  // meta-iter. No-op otherwise (the continuous cursor still covers the next
  // iter). For the initial fill call prefetch_renorm_buffer directly.
  __device__ void refill_if_needed(uint16_t *renorm_buf_for_warp, bool sync)
  {
    if (needs_refill())
    {
      prefetch_renorm_buffer(renorm_buf_for_warp, sync);
    }
  }

  __device__ uint8_t decode_symbol_full()
  {
    uint32_t sym, pdf, smcdf;
    read_from_tables(sym, pdf, smcdf);

    buf_pos_ -= transition_state_and_renormalize(pdf, smcdf);
    return sym;
  }
};

} // namespace detail
} // namespace ans_gpu_lib
