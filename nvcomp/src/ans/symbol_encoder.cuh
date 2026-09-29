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
    uint32_t need_to_renormalize = state_ >
                                   (((pdf - 1) << (32 - DEFAULT_TABLELOG)) | ((1 << (32 - DEFAULT_TABLELOG)) - 1));

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

    state_ = (div << DEFAULT_TABLELOG) + cdf + mod;
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
      active && (state_ > (((pdf - 1) << (32 - DEFAULT_TABLELOG)) | ((1 << (32 - DEFAULT_TABLELOG)) - 1)));

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

    state_ = (div << DEFAULT_TABLELOG) + cdf + mod;
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
constexpr uint32_t PACKED_TABLE_VALUE_MASK = (1u << DEFAULT_TABLELOG) - 1;
constexpr uint32_t PACKED_TABLE_CDF_MASK = (1u << PACKED_TABLE_CDF_BITS) - 1;
constexpr uint32_t PACKED_TABLE_VALUE_OFFSET = PACKED_TABLE_CDF_BITS;
constexpr uint32_t PACKED_TABLE_SHIFT_OFFSET = PACKED_TABLE_CDF_BITS + DEFAULT_TABLELOG;
static_assert(
  PACKED_TABLE_SHIFT_OFFSET + PACKED_TABLE_SHIFT_BITS <= 32,
  "Packed ANS table stores cdf, a tablelog-bit value, and shift in entry.x"
);

inline __host__ __device__ uint32_t pack_cdf_div_scale_shift(uint32_t pdf, uint32_t cdf, uint32_t shift)
{
  uint32_t div_scale = (1u << DEFAULT_TABLELOG) - pdf;
  return ((shift & PACKED_TABLE_SHIFT_MASK) << PACKED_TABLE_SHIFT_OFFSET) |
         ((div_scale & PACKED_TABLE_VALUE_MASK) << PACKED_TABLE_VALUE_OFFSET) | (cdf & PACKED_TABLE_CDF_MASK);
}

inline __device__ uint32_t packed_cdf(uint32_t packed_cdf_divscale_shift)
{
  return packed_cdf_divscale_shift & PACKED_TABLE_CDF_MASK;
}

inline __device__ uint32_t packed_div_scale(uint32_t packed_cdf_divscale_shift)
{
  return (packed_cdf_divscale_shift >> PACKED_TABLE_VALUE_OFFSET) & PACKED_TABLE_VALUE_MASK;
}

inline __device__ uint32_t packed_shift(uint32_t packed_cdf_divscale_shift)
{
  return packed_cdf_divscale_shift >> PACKED_TABLE_SHIFT_OFFSET;
}

// =====================================================================
// nvcomp compress encoder.
// entry.x = {cdf, div_scale, shift}, entry.y = magic.
// (The templated symbol_encoder_dx<ETT> above is retained for libnvcompdx.)
// =====================================================================
class symbol_encoder
{
  const uint2 *table_;
  uint32_t state_;
  uint32_t lane_mask_gt_;
  uint16_t *op_;
#ifndef NDEBUG
  bool initialized = false;
#endif

public:
  __device__ symbol_encoder(int lane_id, const uint2 *table, uint8_t *op)
      : table_{table}
      , state_(0)
      , lane_mask_gt_(lane_id == WARP_SIZE_U - 1 ? 0u : (WARP_ALL << (lane_id + 1)))
      , op_(reinterpret_cast<uint16_t *>(op))
  {}

private:
  // Predicated renorm emit: if need_to_renormalize, store the low 16 bits of
  // state_ to addr and shift state_ down by 16. Kept as PTX so only the store +
  // shift stay predicated (see the per-instruction C++ equivalents in the
  // comments), avoiding a divergent branch + reconverge per symbol.
  __device__ void renorm_store(uint16_t *addr, uint32_t need_to_renormalize)
  {
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
  __device__ void encode_first_symbol(uint8_t symbol)
  {
    // Only cdf needed for the first symbol (low 16 bits of entry.x).
    uint32_t cdf = packed_cdf(table_[symbol].x);
    state_ = (1 << 16) + cdf;
#ifndef NDEBUG
    initialized = true;
#endif
  }

  template <bool partial = false>
  __device__ void encode_symbol(uint8_t symbol, [[maybe_unused]] bool active = true)
  {
    assert(initialized);

    uint2 e = table_[symbol];
    uint32_t magic = e.y;
    uint32_t cdf = packed_cdf(e.x);
    uint32_t div_scale = packed_div_scale(e.x);
    uint32_t shift = packed_shift(e.x);

    uint32_t need_to_renormalize = ((state_ >> (32 - DEFAULT_TABLELOG)) + div_scale) >= (1u << DEFAULT_TABLELOG);
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

    uint32_t offset = __popc(writing & lane_mask_gt_);
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

} // end namespace detail
} // end namespace ans_gpu_lib
