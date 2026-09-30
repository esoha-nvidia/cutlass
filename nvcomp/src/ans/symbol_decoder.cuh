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

#include <cuda/std/tuple>
#include <cuda_pipeline.h>

#include "nvcomp/shared_types.h"
#include "nvcomp/utils.hpp"

#include <ans/ans_arch_profile.cuh>
#include <ans/ans_utils.cuh>

namespace ans_gpu_lib
{
namespace detail
{

// PRMT 0x4065: dest = {pdf1[11:0], s0, s1}. pdf1 is then packed & 0xFFF, not (entry >> 8) & 0xFFF.
__device__ __forceinline__ uint32_t pack_pair_entries(uint32_t entry0, uint32_t entry1)
{
  return __byte_perm(entry0, entry1, 0x4065u);
}

// After pack_pair_entries, `entry` is {pdf1[11:0], s0, s1}. FUSE inserts those symbols into
// the mantissa word and undoes the encoder's 1-bit pair rotate; otherwise SECOND
// gathers two pairs into {s0,s1,s2,s3}.
template <bool FUSE_MANTISSA, bool SECOND>
__device__ __forceinline__ void
fp16_commit_decoded_pair(uint32_t &out, uint32_t entry, [[maybe_unused]] uint32_t mantissas)
{
  if constexpr (FUSE_MANTISSA)
  {
    constexpr uint32_t INSERT = SECOND ? 0x7362u : 0x7160u;
    out = __byte_perm(mantissas, entry, INSERT);
    out = __funnelshift_r(out, out, 1);
  }
  else
  {
    out = SECOND ? __byte_perm(out, entry, 0x7632u) : entry;
  }
}

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

// nvcompDx decoder. Renormalizes from a sliding per-warp decoding buffer in
// shared memory (refilled by `refill_buffer`) and reads the rANS tables via a
// separate `symbol_table_` + `cdf_table_`. Used only by
// libnvcompdx/src/ans/decompress_device.cu. Single rANS state per lane.
template <bool BOUNDS_CHECK>
class symbol_decoder_dx : public bounds_check_state<BOUNDS_CHECK>
{
  const uint16_t *bit_stream_;
  uint32_t state_;
  static constexpr uint32_t state_renorm_thresh_ = 1 << 16;

  const uint8_t *symbol_table_;
  const uint16_t *cdf_table_;
  uint16_t *decoding_buffer_;
  uint16_t decoding_buf_size_;
  uint32_t bit_stream_offset_;

  // Renormalize from the sliding decoding buffer. `read_buf` points just past
  // the last valid uint16 in the buffer; renormalizing lanes pull the next
  // word at `read_buf - offset`, bounds-checked via `guard_load`.
  __device__ uint32_t
  transition_state_and_renormalize(uint32_t pdf, uint32_t smcdf, const uint16_t *read_buf, bool active)
  {
    state_ = pdf * (state_ >> DX_DEFAULT_TABLELOG) + smcdf;

    bool need_to_renormalize = (state_ < state_renorm_thresh_) & active;
    uint32_t reading = __ballot_sync(WARP_ALL, need_to_renormalize);

    if (need_to_renormalize)
    {
      short offset = __popc(reading & get_lane_mask());
      state_ = (state_ << 16) | this->guard_load(read_buf - offset);
    }

    return __popc(reading);
  }

  __device__ uint32_t transition_state_and_renormalize(uint32_t pdf, uint32_t smcdf, const uint16_t *read_buf)
  {
    state_ = pdf * (state_ >> DX_DEFAULT_TABLELOG) + smcdf;

    bool need_to_renormalize = state_ < state_renorm_thresh_;
    uint32_t reading = __ballot_sync(WARP_ALL, need_to_renormalize);

    if (need_to_renormalize)
    {
      short offset = __popc(reading & get_lane_mask());
      state_ = (state_ << 16) | this->guard_load(read_buf - offset);
    }

    return __popc(reading);
  }

public:
  __device__ symbol_decoder_dx(
    const uint8_t *symbol_table,
    const uint16_t *cdf_table,
    uint8_t *decoding_buffer,
    const uint8_t *comp_sub_chunk,
    uint32_t cs_size
  )
      : bit_stream_{reinterpret_cast<const uint16_t *>(comp_sub_chunk)}
      , state_{0}
      , symbol_table_{symbol_table}
      , cdf_table_{cdf_table}
  {
    const uint16_t *bit_stream_end = bit_stream_ + cs_size / sizeof(uint16_t);
    this->init_bounds(bit_stream_, bit_stream_end);
    {
      int lid = get_lane_id();
      uint32_t lo = this->guard_load(&bit_stream_end[-2 * lid - 2]);
      uint32_t hi = this->guard_load(&bit_stream_end[-2 * lid - 1]);
      state_ = (hi << 16) | lo;
    }

    bit_stream_offset_ = cs_size / sizeof(uint16_t) - 64;
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
    return symbol_table_[state_ & ((1 << DX_DEFAULT_TABLELOG) - 1)];
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
    uint32_t idx = state_ & ((1U << DX_DEFAULT_TABLELOG) - 1U);
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

template <bool BOUNDS_CHECK, uint32_t RENORM_BUF_U16, uint32_t REFILL_THRESH_U16, uint32_t TAIL_U16, uint32_t TABLELOG>
class symbol_decoder_base : public bounds_check_state<BOUNDS_CHECK>
{
protected:
  static constexpr uint32_t STATE_RENORM_THRESH = 1u << 16;

  uint32_t table_off_; // 32-bit shared byte address
  // Warp renorm buffer base as a 32-bit shared byte address. The generic pointer
  // is reconstructed only when a refill actually fires, so the decode loop does
  // not park a 64-bit smem pointer across every row.
  uint32_t renorm_base_off_;

  static constexpr uint32_t RENORM_BUF_BYTES = RENORM_BUF_U16 * sizeof(uint16_t);
  static constexpr uint32_t REFILL_THRESH_BYTES = REFILL_THRESH_U16 * sizeof(uint16_t);

  // Renorm cursor as a 32-bit shared-memory byte offset.
  uint32_t read_off_;

  __device__ uint16_t *renorm_buf_generic() const
  {
    return reinterpret_cast<uint16_t *>(__cvta_shared_to_generic(static_cast<size_t>(renorm_base_off_)));
  }

  // This warp's WarpRefillState sits directly after its renorm window.
  __device__ WarpRefillState *refill_state() const
  {
    return reinterpret_cast<WarpRefillState *>(
      __cvta_shared_to_generic(static_cast<size_t>(renorm_base_off_ + RENORM_BUF_BYTES))
    );
  }

  const __device__ uint16_t *bit_stream() const { return refill_state()->bit_stream; }

  // Pack two uint16 from the bitstream tail into a 32-bit rANS state (lo, then hi).
  __device__ uint32_t init_state(const uint16_t *state_arr) const
  {
    return (this->guard_load(&state_arr[1]) << 16) | this->guard_load(&state_arr[0]);
  }

  __device__ uint32_t lookup_renorm_word(uint32_t offset) const
  {
    const uint32_t addr = read_off_ - 2u * offset;
    uint32_t word;
    asm("ld.shared.u16 %0, [%1];" : "=r"(word) : "r"(addr));
    return word;
  }

  __device__ cuda::std::tuple<uint32_t, uint32_t> lookup_renorm_word_pair(uint32_t offset) const
  {
    const uint32_t addr = read_off_ - 2u * offset;
    uint32_t word_at, word_before;
    asm("ld.shared.u16 %0, [%2];\n\t"
        "ld.shared.u16 %1, [%2+-2];"
        : "=r"(word_at), "=r"(word_before)
        : "r"(addr));
    return {word_at, word_before};
  }

  // Packed table word: [0:7]=sym, [8:19]=pdf, [20:31]=smcdf. Returns {raw, pdf, smcdf}.
  __device__ cuda::std::tuple<uint32_t, uint32_t, uint32_t> lookup_decoding_table(uint32_t state) const
  {
    uint32_t idx;
    // AND in asm so ptxas cannot reassociate (state & mask)*4 + base into
    // (state*4) & 0xffc + base. The *4 then folds into IMAD.SHL with table_off_.
    asm("and.b32 %0, %1, %2;" : "=r"(idx) : "r"(state), "n"((1u << TABLELOG) - 1u));
    uint32_t table_entry;
    asm("ld.shared.u32 %0, [%1];" : "=r"(table_entry) : "r"(table_off_ + idx * 4u));
    return {table_entry, (table_entry >> 8) & 0xFFFu, table_entry >> 20};
  }

  // Consume only 16-byte units so the next refill window stays uint4-aligned.
  // Leftover bytes stay live at the high end of the refilled buffer. Returns the advanced
  // window position.
  __device__ uint32_t advance_bitstream_pos()
  {
    const uint32_t consumed_bytes = RENORM_BUF_BYTES - (read_off_ - renorm_base_off_);
    const uint32_t consumed_aligned = nvcomp::roundDownTo(consumed_bytes, sizeof(uint4));
    const uint32_t leftover_bytes = consumed_bytes - consumed_aligned;
    const uint32_t consumed_u16 = consumed_aligned / sizeof(uint16_t);
    WarpRefillState *const rs = refill_state();
    const uint32_t prev = rs->pos_offset_u16;
    const uint32_t pos = (prev > consumed_u16) ? (prev - consumed_u16) : 0u;

    // Separates the read above from the write-back below.
    __syncwarp();
    if (get_lane_id() == 0)
    {
      rs->pos_offset_u16 = pos;
    }
    read_off_ = renorm_base_off_ + (RENORM_BUF_BYTES - leftover_bytes);
    return pos;
  }

public:
  __device__
  symbol_decoder_base(const uint32_t *table, const uint8_t *comp_sub_chunk, uint32_t cs_size, uint16_t *renorm_buf)
      : table_off_{static_cast<uint32_t>(__cvta_generic_to_shared(table))}
      , renorm_base_off_{static_cast<uint32_t>(__cvta_generic_to_shared(renorm_buf))}
  {
    const uint16_t *const bit_stream_base = reinterpret_cast<const uint16_t *>(comp_sub_chunk);
    const uint16_t *bit_stream_end = bit_stream_base + cs_size / sizeof(uint16_t);
    this->init_bounds(bit_stream_base, bit_stream_end);

    if (cs_size == 0)
    {
      read_off_ = renorm_base_off_;
      if (get_lane_id() == 0)
      {
        WarpRefillState *const rs = refill_state();
        rs->bit_stream = bit_stream_base;
        rs->pos_offset_u16 = 0;
      }
      __syncwarp();
      return;
    }

    // Tail after renorm_end is loaded separately. Round the refill window up to 16 B
    // so the uint4 LDGSTS source is aligned. The extra uint16 (into the tail, no
    // OOB) land unread at the high end of the buffer: read_off_ starts so the
    // first load is still renorm_end - 1.
    const uint16_t *renorm_end = bit_stream_end - TAIL_U16;
    uintptr_t renorm_end_addr = reinterpret_cast<uintptr_t>(renorm_end);
    uintptr_t aligned_addr = reinterpret_cast<uintptr_t>(nvcomp::roundUpToAlignment<uint4>(renorm_end));
    uint32_t extra_uint16 = static_cast<uint32_t>((aligned_addr - renorm_end_addr) / sizeof(uint16_t));
    read_off_ = renorm_base_off_ + (RENORM_BUF_BYTES - sizeof(uint16_t) * extra_uint16);

    // The __syncwarp opening prefetch_renorm_buffer publishes this before the first refill.
    if (get_lane_id() == 0)
    {
      WarpRefillState *const rs = refill_state();
      rs->bit_stream = bit_stream_base;
      rs->pos_offset_u16 = static_cast<uint32_t>(reinterpret_cast<const uint16_t *>(aligned_addr) - bit_stream_base);
    }

    // Async so LDGSTS overlaps the caller's tail-state loads.
    prefetch_renorm_buffer(false /*async: caller waits*/);
  }

  __device__ void wait_initial_prefetch()
  {
    __pipeline_wait_prior(0);
    __syncwarp();
  }

  __device__ void prefetch_renorm_buffer(bool sync)
  {
    __syncwarp();
    const uint32_t pos_offset_u16 = advance_bitstream_pos();

    int lid = get_lane_id();
    uint4 *dst = reinterpret_cast<uint4 *>(renorm_buf_generic());
    const uint16_t *const bit_stream_base = bit_stream();
    const uint16_t *window_end = bit_stream_base + pos_offset_u16;
    const uint4 *src_lo =
      reinterpret_cast<const uint4 *>(nvcomp::roundDownTo(reinterpret_cast<uintptr_t>(bit_stream_base), sizeof(uint4)));

    const uint4 *src = reinterpret_cast<const uint4 *>(window_end - RENORM_BUF_U16);
    constexpr uint32_t UINT4_PER_LANE = RENORM_BUF_U16 / (WARP_SIZE * (sizeof(uint4) / sizeof(uint16_t)));
    static_assert(
      UINT4_PER_LANE * WARP_SIZE * (sizeof(uint4) / sizeof(uint16_t)) == RENORM_BUF_U16,
      "RENORM_BUF_U16 must be a multiple of WARP_SIZE * 8"
    );

#pragma unroll
    for (uint32_t i = 0; i < UINT4_PER_LANE; ++i)
    {
      const uint4 *s = src + i * WARP_SIZE + lid;
      s = s < src_lo ? src_lo : s;
      uint4 *d = dst + i * WARP_SIZE + lid;
      __pipeline_memcpy_async(d, reinterpret_cast<const void *>(s), sizeof(uint4));
    }

    __pipeline_commit();
    if (sync)
    {
      __pipeline_wait_prior(0);
      __syncwarp();
    }
  }

  __device__ void refill_if_needed(bool sync)
  {
    if (read_off_ < renorm_base_off_ + REFILL_THRESH_BYTES)
    {
      prefetch_renorm_buffer(sync);
    }
  }

  template <bool Partial>
  static __device__ bool need_to_renormalize(uint32_t state, uint32_t thresh, [[maybe_unused]] bool active = true)
  {
    const bool need = state < thresh;
    if constexpr (Partial)
    {
      return active && need;
    }
    return need;
  }

  static __device__ void renormalize_state(uint32_t &state, bool need, uint32_t word)
  {
    const uint32_t renorm_state = (state << 16) | word;
    state = need ? renorm_state : state;
  }
};

// One rANS state per lane (CHAR, FP8, FLOAT16 x1).
template <bool BOUNDS_CHECK, uint32_t RENORM_BUF_U16, uint32_t REFILL_THRESH_U16, uint32_t TABLELOG>
class symbol_decoder
    : public symbol_decoder_base<BOUNDS_CHECK, RENORM_BUF_U16, REFILL_THRESH_U16, TAIL_U16_PER_STATE * WARP_SIZE_U, TABLELOG>
{
  using BaseSymbolDecoder =
    symbol_decoder_base<BOUNDS_CHECK, RENORM_BUF_U16, REFILL_THRESH_U16, TAIL_U16_PER_STATE * WARP_SIZE_U, TABLELOG>;

  using BaseSymbolDecoder::lookup_decoding_table;
  using BaseSymbolDecoder::lookup_renorm_word;
  using BaseSymbolDecoder::need_to_renormalize;
  using BaseSymbolDecoder::read_off_;
  using BaseSymbolDecoder::renormalize_state;
  using BaseSymbolDecoder::STATE_RENORM_THRESH;

  uint32_t state_;

public:
  __device__ symbol_decoder(const uint32_t *table, const uint8_t *comp_sub_chunk, uint32_t cs_size, uint16_t *renorm_buf)
      : BaseSymbolDecoder(table, comp_sub_chunk, cs_size, renorm_buf)
      , state_{0}
  {
    if (cs_size == 0)
    {
      return;
    }

    const uint16_t *const bit_stream_base = reinterpret_cast<const uint16_t *>(comp_sub_chunk);
    const uint16_t *tail = bit_stream_base + cs_size / sizeof(uint16_t) - 2 * get_lane_id() - 2;
    state_ = this->init_state(&tail[0]);
  }

  template <bool Partial = false>
  __device__ uint8_t decode_symbol([[maybe_unused]] bool active = true)
  {
    uint32_t sym, pdf, smcdf;
    cuda::std::tie(sym, pdf, smcdf) = lookup_decoding_table(state_);
    state_ = pdf * (state_ >> TABLELOG) + smcdf;

    const bool need = need_to_renormalize<Partial>(state_, STATE_RENORM_THRESH, active);

    const uint32_t reading = __ballot_sync(WARP_ALL, need);
    const uint32_t word = lookup_renorm_word(__popc(reading & get_lane_mask()));

    renormalize_state(state_, need, word);
    read_off_ -= sizeof(uint16_t) * __popc(reading);
    return static_cast<uint8_t>(sym);
  }

  // One rANS state, two sequential symbols. SECOND=false writes {s0, s1, ...};
  // SECOND=true gathers {s0,s1,s2,s3}. FUSE_MANTISSA PRMTs those symbols into
  // `mantissas` and undoes the 1-bit pair rotate. Partial skips packing and
  // writes the two raw table words (symbol in byte 0) to `out` and `*out1`.
  template <bool FUSE_MANTISSA, bool SECOND, bool Partial = false>
  __forceinline__ __device__ void decode_pair(
    uint32_t &out,
    [[maybe_unused]] uint32_t mantissas = 0,
    [[maybe_unused]] bool active0 = true,
    [[maybe_unused]] bool active1 = true,
    [[maybe_unused]] uint32_t *out1 = nullptr
  )
  {
    uint32_t entry0, pdf0, smcdf0;
    cuda::std::tie(entry0, pdf0, smcdf0) = lookup_decoding_table(state_);
    state_ = pdf0 * (state_ >> TABLELOG) + smcdf0;
    const bool need0 = need_to_renormalize<Partial>(state_, STATE_RENORM_THRESH, active0);
    const uint32_t reading0 = __ballot_sync(WARP_ALL, need0);
    renormalize_state(state_, need0, lookup_renorm_word(__popc(reading0 & get_lane_mask())));
    read_off_ -= sizeof(uint16_t) * __popc(reading0);

    if constexpr (Partial)
    {
      out = entry0;
      uint32_t pdf1, smcdf1;
      cuda::std::tie(*out1, pdf1, smcdf1) = lookup_decoding_table(state_);
      state_ = pdf1 * (state_ >> TABLELOG) + smcdf1;
    }
    else
    {
      uint32_t entry1 = cuda::std::get<0>(lookup_decoding_table(state_));
      const uint32_t smcdf1 = entry1 >> 20;
      entry0 = pack_pair_entries(entry0, entry1);
      const uint32_t pdf1 = entry0 & 0xFFFu;
      state_ = pdf1 * (state_ >> TABLELOG) + smcdf1;
    }

    const bool need1 = need_to_renormalize<Partial>(state_, STATE_RENORM_THRESH, active1);
    const uint32_t reading1 = __ballot_sync(WARP_ALL, need1);
    renormalize_state(state_, need1, lookup_renorm_word(__popc(reading1 & get_lane_mask())));
    read_off_ -= sizeof(uint16_t) * __popc(reading1);

    if constexpr (!Partial)
    {
      fp16_commit_decoded_pair<FUSE_MANTISSA, SECOND>(out, entry0, mantissas);
    }
  }
};

// Two rANS states per lane (default). Mirror of symbol_encoder_dual_state.
template <bool BOUNDS_CHECK, uint32_t RENORM_BUF_U16, uint32_t REFILL_THRESH_U16, uint32_t TABLELOG>
class symbol_decoder_dual_state
    : public symbol_decoder_base<
        BOUNDS_CHECK,
        RENORM_BUF_U16,
        REFILL_THRESH_U16,
        TAIL_U16_PER_STATE * WARP_SIZE_U * 2,
        TABLELOG>
{
  using Base =
    symbol_decoder_base<BOUNDS_CHECK, RENORM_BUF_U16, REFILL_THRESH_U16, TAIL_U16_PER_STATE * WARP_SIZE_U * 2, TABLELOG>;

  using Base::lookup_decoding_table;
  using Base::lookup_renorm_word_pair;
  using Base::need_to_renormalize;
  using Base::read_off_;
  using Base::renormalize_state;
  using Base::STATE_RENORM_THRESH;

  uint32_t state0_;
  uint32_t state1_;

public:
  __device__
  symbol_decoder_dual_state(const uint32_t *table, const uint8_t *comp_sub_chunk, uint32_t cs_size, uint16_t *renorm_buf)
      : Base(table, comp_sub_chunk, cs_size, renorm_buf)
      , state0_{0}
      , state1_{0}
  {
    if (cs_size == 0)
    {
      return;
    }

    // Two-state tail: 4 uint16 per lane with both of this lane's states adjacent --
    // [state0 lo, state0 hi, state1 lo, state1 hi] at bit_stream_end[-4t-4 .. -4t-1].
    const uint16_t *const bit_stream_base = reinterpret_cast<const uint16_t *>(comp_sub_chunk);
    const uint16_t *tail = bit_stream_base + cs_size / sizeof(uint16_t) - 4 * get_lane_id() - 4;
    state0_ = this->init_state(&tail[0]);
    state1_ = this->init_state(&tail[2]);
  }

  // Dual-state pair. SECOND=false writes {s0, s1, ...}; SECOND=true gathers {s0,s1,s2,s3}.
  // FUSE_MANTISSA PRMTs those symbols into `mantissas` and undoes the 1-bit pair rotate.
  // Partial skips packing and writes the two raw table words (symbol in byte 0) to `out`
  // and `*out1`.
  template <bool FUSE_MANTISSA, bool SECOND, bool Partial = false>
  __forceinline__ __device__ void decode_pair(
    uint32_t &out,
    [[maybe_unused]] uint32_t mantissas = 0,
    [[maybe_unused]] bool active0 = true,
    [[maybe_unused]] bool active1 = true,
    [[maybe_unused]] uint32_t *out1 = nullptr
  )
  {
    uint32_t entry0, pdf0, smcdf0;
    cuda::std::tie(entry0, pdf0, smcdf0) = lookup_decoding_table(state0_);
    state0_ = pdf0 * (state0_ >> TABLELOG) + smcdf0;

    if constexpr (Partial)
    {
      uint32_t pdf1, smcdf1;
      cuda::std::tie(*out1, pdf1, smcdf1) = lookup_decoding_table(state1_);
      state1_ = pdf1 * (state1_ >> TABLELOG) + smcdf1;
      out = entry0;
    }
    else
    {
      uint32_t entry1 = cuda::std::get<0>(lookup_decoding_table(state1_));
      const uint32_t smcdf1 = entry1 >> 20;
      entry0 = pack_pair_entries(entry0, entry1);
      const uint32_t pdf1 = entry0 & 0xFFFu;
      state1_ = pdf1 * (state1_ >> TABLELOG) + smcdf1;
      fp16_commit_decoded_pair<FUSE_MANTISSA, SECOND>(out, entry0, mantissas);
    }

    const bool need0 = need_to_renormalize<Partial>(state0_, STATE_RENORM_THRESH, active0);
    const bool need1 = need_to_renormalize<Partial>(state1_, STATE_RENORM_THRESH, active1);
    const uint32_t reading0 = __ballot_sync(WARP_ALL, need0);
    const uint32_t reading1 = __ballot_sync(WARP_ALL, need1);

    const uint32_t offset = __popc(reading1 & get_lane_mask_lt()) + __popc(reading0 & get_lane_mask());
    auto [word0, word1] = lookup_renorm_word_pair(offset);
    renormalize_state(state0_, need0, word0);
    renormalize_state(state1_, need1, word1);
    read_off_ -= sizeof(uint16_t) * (__popc(reading0) + __popc(reading1));
  }
};

} // namespace detail
} // namespace ans_gpu_lib
