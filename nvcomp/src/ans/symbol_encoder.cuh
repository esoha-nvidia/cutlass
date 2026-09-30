/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this 
 * software and associated documentation files (the "Software"), to deal in the Software without 
 * restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, 
 * sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is 
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies 
 * or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY 
 * OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT 
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include <ciso646>

#include "nvcomp/shared_types.h"

#include <ans/ans_utils.cuh>

namespace ans_gpu_lib
{
namespace detail
{

template <typename ETT>
class read_table_entry
{
public:
  static __device__ uint32_t get_pdf(ETT symbol);
  static __device__ uint32_t get_cdf(ETT symbol);
  static __device__ uint32_t get_magic(ETT symbol);
  static __device__ uint32_t get_shift(ETT symbol);
};

template <>
class read_table_entry<ETT_DEVICE>
{
public:
  static __device__ uint32_t get_pdf(ETT_DEVICE entry) { return entry.x >> 16; }
  static __device__ uint32_t get_cdf(ETT_DEVICE entry) { return entry.x & 0xffff; }
  static __device__ uint32_t get_magic(ETT_DEVICE entry) { return entry.y; }
  static __device__ uint32_t get_shift(ETT_DEVICE entry) { return entry.z; }
};

// compilation of class in different translation units
template <typename ETT>
class symbol_encoder_dx
{
  ETT *encoding_table_;
  uint32_t state_;
  int lane_id_;
  uint16_t *op_;
#ifndef NDEBUG
  bool initialized = false;
#endif

public:
  __device__ symbol_encoder_dx(int lane_id, ETT *encoding_table, uint8_t *op)
      : encoding_table_{encoding_table}
      , state_(0)
      , // default value for coverity
      lane_id_(lane_id)
      , op_(reinterpret_cast<uint16_t *>(op))
  {}

  __device__ void encode_first_symbol(uint8_t symbol)
  {
    // get the lowest state corresponding to the symbol to save the cost of
    // encoding the symbol
    state_ = (1 << 16) + read_table_entry<ETT>::get_cdf(encoding_table_[symbol]);
#ifndef NDEBUG
    initialized = true;
#endif
  }

  __device__ void encode_symbol(uint8_t symbol)
  {
    assert(initialized);

    ETT table_entry = encoding_table_[symbol];

    uint32_t pdf = read_table_entry<ETT>::get_pdf(table_entry);
    uint32_t cdf = read_table_entry<ETT>::get_cdf(table_entry);
    uint32_t magic = read_table_entry<ETT>::get_magic(table_entry);
    uint32_t shift = read_table_entry<ETT>::get_shift(table_entry);

    // avoid doing a 64-bit multiplication
    uint32_t need_to_renormalize = state_ > (((pdf - 1) << (32 - DX_DEFAULT_TABLELOG)) |
                                             ((1 << (32 - DX_DEFAULT_TABLELOG)) - 1));

    uint32_t writing = __ballot_sync(WARP_ALL, need_to_renormalize);

    uint32_t offset = __popc(writing >> (lane_id_ + 1));
    if (need_to_renormalize)
    {
      op_[offset] = state_;
      state_ >>= 16;
    }

    op_ += __popc(writing);

    uint32_t div = (state_ + __umulhi(state_, magic)) >> shift;
    uint32_t mod = state_ - div * pdf;

    state_ = (div << DX_DEFAULT_TABLELOG) + cdf + mod;
  }

  __device__ void encode_symbol(uint8_t symbol, bool active)
  {
    assert(initialized);
    ETT table_entry = encoding_table_[symbol];

    uint32_t pdf = read_table_entry<ETT>::get_pdf(table_entry);
    uint32_t cdf = read_table_entry<ETT>::get_cdf(table_entry);
    uint32_t magic = read_table_entry<ETT>::get_magic(table_entry);
    uint32_t shift = read_table_entry<ETT>::get_shift(table_entry);

    // avoid doing a 64-bit multiplication
    uint32_t need_to_renormalize =
      active && (state_ > (((pdf - 1) << (32 - DX_DEFAULT_TABLELOG)) | ((1 << (32 - DX_DEFAULT_TABLELOG)) - 1)));

    uint32_t writing = __ballot_sync(WARP_ALL, need_to_renormalize);

    if (not active)
    {
      op_ += __popc(writing);
      return;
    }

    uint32_t offset = __popc(writing >> (lane_id_ + 1));
    if (need_to_renormalize)
    {
      op_[offset] = state_;
      state_ >>= 16;
    }

    op_ += __popc(writing);

    // Equivalent to `state_ / pdf` (see the `magic` derivation where the encoding
    // table is built, in normalize_and_build_table Phase 3).
    uint32_t div = (state_ + __umulhi(state_, magic)) >> shift;
    uint32_t mod = state_ - div * pdf;

    state_ = (div << DX_DEFAULT_TABLELOG) + cdf + mod;
  }

  __device__ void encode_final_state()
  {
    op_ += 64;
    op_[-2 * lane_id_ - 2] = state_;
    op_[-2 * lane_id_ - 1] = state_ >> 16;
  }

  __device__ IndexT get_comp_stream_size(uint8_t *orig_op)
  {
    return static_cast<IndexT>(reinterpret_cast<uint8_t *>(op_) - orig_op);
  }
};

// Packed 64-bit table-entry field layout: entry.x holds {cdf, div_scale, shift};
// entry.y holds magic.
constexpr uint32_t PACKED_TABLE_CDF_BITS = 16;
constexpr uint32_t PACKED_TABLE_SHIFT_BITS = 5;
constexpr uint32_t PACKED_TABLE_SHIFT_MASK = (1u << PACKED_TABLE_SHIFT_BITS) - 1;
constexpr uint32_t PACKED_TABLE_CDF_MASK = (1u << PACKED_TABLE_CDF_BITS) - 1u;
constexpr uint32_t PACKED_TABLE_VALUE_OFFSET = PACKED_TABLE_CDF_BITS;

template <uint32_t TABLELOG>
inline __host__ __device__ uint32_t pack_cdf_div_scale_shift(uint32_t pdf, uint32_t cdf, uint32_t shift)
{
  constexpr uint32_t VALUE_MASK = (1u << TABLELOG) - 1u;
  constexpr uint32_t SHIFT_OFFSET = PACKED_TABLE_CDF_BITS + TABLELOG;
  static_assert(
    SHIFT_OFFSET + PACKED_TABLE_SHIFT_BITS <= 32,
    "Packed ANS table stores cdf, a tablelog-bit value, and shift in entry.x"
  );
  const uint32_t div_scale = (1u << TABLELOG) - pdf;
  return ((shift & PACKED_TABLE_SHIFT_MASK) << SHIFT_OFFSET) | ((div_scale & VALUE_MASK) << PACKED_TABLE_VALUE_OFFSET) |
         (cdf & PACKED_TABLE_CDF_MASK);
}

inline __device__ uint32_t packed_cdf(uint32_t packed_cdf_divscale_shift)
{
  return packed_cdf_divscale_shift & PACKED_TABLE_CDF_MASK;
}

template <uint32_t TABLELOG>
inline __device__ uint32_t packed_div_scale(uint32_t packed_cdf_divscale_shift)
{
  constexpr uint32_t VALUE_MASK = (1u << TABLELOG) - 1u;
  return (packed_cdf_divscale_shift >> PACKED_TABLE_VALUE_OFFSET) & VALUE_MASK;
}

template <uint32_t TABLELOG>
inline __device__ uint32_t packed_shift(uint32_t packed_cdf_divscale_shift)
{
  constexpr uint32_t SHIFT_OFFSET = PACKED_TABLE_CDF_BITS + TABLELOG;
  return packed_cdf_divscale_shift >> SHIFT_OFFSET;
}

// Packed encode-table load: {x = {cdf,div_scale,shift}, y = magic}. AND the
// symbol in asm so ptxas cannot reassociate (sym & 0xff)*8 + base into
// (sym*8) & 0x7f8 + base; the *8 then folds into IMAD.SHL with table_off.
inline __device__ uint2 lds_packed_table_entry(uint32_t table_off, uint32_t sym)
{
  uint32_t idx;
  asm("and.b32 %0, %1, 255;" : "=r"(idx) : "r"(sym));
  uint2 e;
  asm("ld.shared.v2.u32 {%0, %1}, [%2];"
      : "=r"(e.x), "=r"(e.y)
      : "r"(table_off + idx * static_cast<uint32_t>(sizeof(uint2))));
  return e;
}

// =====================================================================
// nvcomp compress encoder.
// entry.x = {cdf, div_scale, shift}, entry.y = magic.
// (The templated symbol_encoder_dx<ETT> above is retained for libnvcompdx.)
// =====================================================================
template <uint32_t TABLELOG>
class symbol_encoder
{
  // Encoding table (uint2/symbol: {x = packed {cdf,div_scale,shift}, y = magic}) in shared
  // memory. table_off_ is its 32-bit shared address (from __cvta_generic_to_shared,
  // computed once in the ctor) so each read is a single ld.shared.v2.u32 -- no per-load
  // generic->shared LOP3. Index math is AND-in-asm + IMAD.SHL (see lds_packed_table_entry).
  uint32_t table_off_;
  uint32_t state_;
  // The densify mask is NOT parked here: get_lane_mask_gt() rematerializes it via S2R at
  // each renorm, which frees a register across the hot encode loop.
  uint16_t *op_;
  // Sampled-histogram fallback: a PER-THREAD running MIN of every table magic this lane
  // touched (only on the Detect=true instantiation -- compile-time gated, and dead code
  // otherwise). A covered symbol's magic is >= 1 and an uncovered one is 0, so min_magic_
  // == 0 at the end iff this lane hit a symbol the model does not cover.
  uint32_t min_magic_;

public:
  // lane_id is kept for the uniform Encoder ctor; densify rematerializes its mask instead.
  __device__ symbol_encoder(int /*lane_id*/, const uint2 *table, uint8_t *op)
      : table_off_{static_cast<uint32_t>(__cvta_generic_to_shared(table))}
      , state_(1u << 16)
      , op_(reinterpret_cast<uint16_t *>(op))
      , min_magic_(0xFFFFFFFFu) // min identity: no magic seen yet
  {}

  // Warp-combined: true iff ANY lane in the warp saw an uncovered symbol. Call once after
  // the encode loop (a single ballot, not a per-symbol shared write).
  __device__ bool uncovered_any() const { return __ballot_sync(WARP_ALL, min_magic_ == 0u) != 0u; }

private:
  __device__ uint2 read_table(uint32_t sym) const { return lds_packed_table_entry(table_off_, sym); }

private:
  __device__ void renorm_store(uint16_t *addr, uint32_t need_to_renormalize)
  {
    // Predicated store of state_ to the renorm buffer. A tight control on the predicate
    // has led to better performance due to better ILP / instruction ordering.
    asm(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.u32 p, %3, 0;\n\t" // bool p = need_to_renormalize;
      "@p st.global.b16 [%1], %2;\n\t" // if (p) *(uint16_t*)%1 = %2;
      "@p shr.b32 %0, %0, 16;\n\t" // if (p) state_ >>= 16;
      "}"
      : "+r"(state_) // %0: state_      (in/out)
      : "l"(addr), // %1: addr        (in)
        "h"(static_cast<uint16_t>(state_)), // %2: state_lo    (in)
        "r"(need_to_renormalize) // %3: predicate   (in)
    );
  }

public:
  __forceinline__ __device__ void encode_first_symbol(uint32_t symbol)
  {
    // Only cdf needed for the first symbol (low 16 bits of entry.x).
    // Unused in detect path, so we don't need to check for a missing symbol.
    uint32_t cdf = packed_cdf(read_table(symbol).x);
    state_ = (1 << 16) + cdf;
  }

  template <bool partial = false, bool Detect = false>
  __forceinline__ __device__ void encode_symbol(uint32_t symbol, [[maybe_unused]] bool active = true)
  {
    const uint2 e = read_table(symbol); // one ld.shared.v2.u32: {x = packed, y = magic}
    uint32_t magic = e.y;
    uint32_t cdf = packed_cdf(e.x);
    uint32_t div_scale = packed_div_scale<TABLELOG>(e.x);
    uint32_t shift = packed_shift<TABLELOG>(e.x);

    // Uncovered-symbol detection, compile-time gated. An inactive partial lane feeds the
    // min identity so its meaningless magic cannot trip the check.
    if constexpr (Detect)
    {
      const uint32_t m = (partial && !active) ? 0xFFFFFFFFu : magic;
      min_magic_ = min(min_magic_, m);
    }

    uint32_t need_to_renormalize = ((state_ >> (32 - TABLELOG)) + div_scale) >= (1u << TABLELOG);
    if constexpr (partial)
    {
      need_to_renormalize = active && need_to_renormalize;
    }

    uint32_t writing = __ballot_sync(WARP_ALL, need_to_renormalize);

    if constexpr (partial)
    {
      if (not active)
      {
        op_ += __popc(writing);
        return;
      }
    }

    uint32_t offset = __popc(writing & get_lane_mask_gt());
    renorm_store(&op_[offset], need_to_renormalize);

    op_ += __popc(writing);

    uint32_t div = (state_ + __umulhi(state_, magic)) >> shift;
    state_ = state_ + cdf + div * div_scale;
  }

  __device__ void encode_final_state()
  {
    int lane_id = get_lane_id();
    op_ += 64;
    op_[-2 * lane_id - 2] = state_;
    op_[-2 * lane_id - 1] = state_ >> 16;
  }

  __device__ IndexT get_comp_stream_size(uint8_t *orig_op)
  {
    return static_cast<IndexT>(reinterpret_cast<uint8_t *>(op_) - orig_op);
  }
};

template <uint32_t TABLELOG>
class symbol_encoder_dual_state
{
  uint32_t table_off_;
  uint32_t state0_;
  uint32_t state1_;
  uint16_t *op_;
  // Sampled-histogram fallback: a PER-THREAD register running MIN of every table magic this
  // lane touched. A covered symbol's magic is >= 1, an uncovered one is 0, so min_magic_
  // == 0 at the end iff this lane hit an uncovered symbol.
  uint32_t min_magic_;

public:
  __device__ symbol_encoder_dual_state(int /*lane_id*/, const uint2 *table, uint8_t *op)
      : table_off_{static_cast<uint32_t>(__cvta_generic_to_shared(table))}
      , state0_(1u << 16)
      , state1_(1u << 16)
      , op_(reinterpret_cast<uint16_t *>(op))
      , min_magic_(0xFFFFFFFFu)
  {}

  __device__ bool uncovered_any() const { return __ballot_sync(WARP_ALL, min_magic_ == 0u) != 0u; }

private:
  __device__ uint2 read_table(uint32_t sym) const { return lds_packed_table_entry(table_off_, sym); }

private:
  template <int ByteOff = 0>
  __device__ void renorm_store(uint32_t &state, uint16_t *addr, uint32_t need_to_renormalize)
  {
    // Predicated store of state to the renorm buffer. A tight control on the predicate
    // has led to better ILP / instruction ordering. ByteOff enables immediate addressing
    // for both stores in a pair with one base address.
    asm("{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.u32 p, %3, 0;\n\t"
        "@p st.global.b16 [%1+%4], %2;\n\t"
        "@p shr.b32 %0, %0, 16;\n\t"
        "}"
        : "+r"(state)
        : "l"(addr), "h"(static_cast<uint16_t>(state)), "r"(need_to_renormalize), "n"(ByteOff));
  }

public:
  // Seed both states from the highest-index symbol pair, which saves encoding it: the
  // decoder reaches that pair holding exactly the state this leaves behind, so it needs no
  // matching change and an encoder is free to seed or not per sub-chunk. Only valid for
  // symbols the model covers (an uncovered symbol has cdf 0), so callers must either
  // histogram exactly or cover the whole alphabet. Mirrors
  // symbol_encoder::encode_first_symbol.
  __forceinline__ __device__ void encode_first_pair(uint32_t s_g1, uint32_t s_g0)
  {
    state1_ = (1u << 16) + packed_cdf(read_table(s_g1).x);
    state0_ = (1u << 16) + packed_cdf(read_table(s_g0).x);
  }

  template <bool Detect = false, bool Partial = false>
  __forceinline__ __device__ void encode_pair(
    uint32_t s_g1,
    uint32_t s_g0,
    [[maybe_unused]] bool active_g1 = true,
    [[maybe_unused]] bool active_g0 = true
  )
  {
    const uint2 e1 = read_table(s_g1);
    const uint2 e0 = read_table(s_g0);
    const uint32_t x1 = e1.x, mg1 = e1.y;
    const uint32_t x0 = e0.x, mg0 = e0.y;
    const uint32_t cdf1 = packed_cdf(x1), ds1 = packed_div_scale<TABLELOG>(x1), sh1 = packed_shift<TABLELOG>(x1);
    const uint32_t cdf0 = packed_cdf(x0), ds0 = packed_div_scale<TABLELOG>(x0), sh0 = packed_shift<TABLELOG>(x0);

    if constexpr (Detect)
    {
      if constexpr (Partial)
      {
        min_magic_ = min(min_magic_, min(active_g1 ? mg1 : 0xFFFFFFFFu, active_g0 ? mg0 : 0xFFFFFFFFu));
      }
      else
      {
        min_magic_ = min(min_magic_, min(mg1, mg0));
      }
    }

    uint32_t need1 = ((state1_ >> (32 - TABLELOG)) + ds1) >= (1u << TABLELOG);
    uint32_t need0 = ((state0_ >> (32 - TABLELOG)) + ds0) >= (1u << TABLELOG);
    if constexpr (Partial)
    {
      need1 = active_g1 && need1;
      need0 = active_g0 && need0;
    }
    const uint32_t writing1 = __ballot_sync(WARP_ALL, need1);
    const uint32_t writing0 = __ballot_sync(WARP_ALL, need0);

    uint16_t *const slot_b = op_ + (__popc(writing1 & get_lane_mask_ge()) + __popc(writing0 & get_lane_mask_gt()));
    renorm_store<-2>(state1_, slot_b, need1);
    renorm_store<0>(state0_, slot_b, need0);

    const uint32_t div1 = (state1_ + __umulhi(state1_, mg1)) >> sh1;
    const uint32_t div0 = (state0_ + __umulhi(state0_, mg0)) >> sh0;
    if constexpr (Partial)
    {
      if (active_g1)
      {
        state1_ = state1_ + cdf1 + div1 * ds1;
      }
      if (active_g0)
      {
        state0_ = state0_ + cdf0 + div0 * ds0;
      }
    }
    else
    {
      state1_ = state1_ + cdf1 + div1 * ds1;
      state0_ = state0_ + cdf0 + div0 * ds0;
    }

    op_ += __popc(writing1) + __popc(writing0);
  }

  __device__ void encode_final_state()
  {
    int lane_id = get_lane_id();
    op_ += 128;
    uint16_t *const tail = op_ - 4 * lane_id - 4;
    tail[0] = state0_;
    tail[1] = state0_ >> 16;
    tail[2] = state1_;
    tail[3] = state1_ >> 16;
  }

  __device__ IndexT get_comp_stream_size(uint8_t *orig_op)
  {
    return static_cast<IndexT>(reinterpret_cast<uint8_t *>(op_) - orig_op);
  }
};

} // end namespace detail
} // end namespace ans_gpu_lib
