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

#include <cub/cub.cuh>

#include <cassert>

#include "ans/constants.hpp"
#include "CudaConstants.h"
#include "nvcomp/shared_types.h"

#include <nvcomp/utils.hpp>

namespace ans_gpu_lib
{
namespace
{

// Encoding table type by nvcompDx. nvcomp uses the packed uint2 table read by symbol_encoder.
typedef uint3 ETT_DEVICE;

// alignment must be a power of 2. Host ANS layout rounds offsets, not addresses;
// nvcompDx still uses these to place header arrays.
__device__ __host__ void *round_up_align_address(void *addr, uint32_t alignment)
{
  uintptr_t uint_addr = reinterpret_cast<uintptr_t>(addr);
  return uint_addr & (alignment - 1) ? reinterpret_cast<void *>(uint_addr + alignment - (uint_addr & (alignment - 1)))
                                     : addr;
}

const __device__ __host__ void *round_up_align_address(const void *addr, uint32_t alignment)
{
  uintptr_t uint_addr = reinterpret_cast<uintptr_t>(addr);
  return uint_addr & (alignment - 1) ? reinterpret_cast<void *>(uint_addr + alignment - (uint_addr & (alignment - 1)))
                                     : addr;
}

constexpr uint32_t ANS_SUB_CHUNK_SLOT_ALIGN = sizeof(uint4);

// Self-describing stream type in ANS_fixed_header::type_.
enum class AnsStreamType : uint8_t
{
  Char = 0,
  Fp16 = 1,
  Fp8 = 2,
  Fp32 = 3
};

inline AnsStreamType ans_stream_type_from_data_type(nvcompType_t data_type)
{
  switch (data_type)
  {
    case NVCOMP_TYPE_CHAR:
      return AnsStreamType::Char;
    case NVCOMP_TYPE_UCHAR:
      return AnsStreamType::Char;
    case NVCOMP_TYPE_FLOAT16:
      return AnsStreamType::Fp16;
    case NVCOMP_TYPE_FLOAT8_E4M3:
      return AnsStreamType::Fp8;
    case NVCOMP_TYPE_FLOAT32:
      return AnsStreamType::Fp32;
    default:
      assert(false);
      return AnsStreamType::Char;
  }
}

inline constexpr __host__ __device__ uint32_t ans_tablelog(AnsStreamType type)
{
  switch (type)
  {
    case AnsStreamType::Char:
      return CHAR_TABLELOG;
    case AnsStreamType::Fp16:
      return FP16_TABLELOG;
    case AnsStreamType::Fp8:
      return FP8_TABLELOG;
    case AnsStreamType::Fp32:
      return FP32_TABLELOG;
    default:
      assert(false);
      return 0u;
  }
}

// Default selected by a zero states_per_lane compression option.
inline constexpr uint8_t ans_default_states_per_lane(AnsStreamType type)
{
  switch (type)
  {
    case AnsStreamType::Char:
    case AnsStreamType::Fp16:
    case AnsStreamType::Fp8:
    case AnsStreamType::Fp32:
      return 2;
    default:
      assert(false);
      return 0;
  }
}

inline __host__ __device__ bool ans_known_stream_type(AnsStreamType type)
{
  switch (type)
  {
    case AnsStreamType::Char:
    case AnsStreamType::Fp16:
    case AnsStreamType::Fp8:
    case AnsStreamType::Fp32:
      return true;
    default:
      return false;
  }
}

// Sub-chunk count from the two header fields. max_sub_chunk_size_ is in bytes.
inline __host__ __device__ uint32_t ans_derived_num_sub_chunks(uint32_t uncomp_bytes, uint32_t max_sub_chunk_size)
{
  const uint32_t n = nvcomp::roundUpDiv(uncomp_bytes, max_sub_chunk_size);
  return n == 0 ? 1 : n;
}

// CHAR is 1 B/symbol; FLOAT16 and FLOAT8 are 2 B/symbol (a decoded pair for FP8);
// FLOAT32 is 4 B/symbol.
inline constexpr __host__ __device__ uint32_t ans_bytes_per_symbol(AnsStreamType type)
{
  switch (type)
  {
    case AnsStreamType::Char:
      return 1u;
    case AnsStreamType::Fp16:
      return 2u;
    case AnsStreamType::Fp8:
      return 2u;
    case AnsStreamType::Fp32:
      return 4u;
    default:
      assert(false);
      return 0u;
  }
}

// Mantissa bytes one ANS symbol contributes. CHAR has none; FLOAT16 and FLOAT8 keep one
// byte per symbol; FLOAT32 keeps its value's sign and 23-bit mantissa, so three.
inline constexpr __host__ __device__ uint32_t ans_mantissa_bytes_per_symbol(AnsStreamType type)
{
  switch (type)
  {
    case AnsStreamType::Char:
      return 0u;
    case AnsStreamType::Fp16:
      return 1u;
    case AnsStreamType::Fp8:
      return 1u;
    case AnsStreamType::Fp32:
      return 3u;
    default:
      assert(false);
      return 0u;
  }
}

// ANS symbols from uncompressed bytes. An odd FP8 leftover byte is not a symbol.
inline constexpr __host__ __device__ uint32_t ans_num_symbols(AnsStreamType type, uint32_t uncomp_bytes)
{
  return uncomp_bytes / ans_bytes_per_symbol(type);
}

// Mantissa length from type + uncompressed byte count. CHAR has none; FLOAT16 is 1 B per
// symbol; FLOAT8 is 1 B per pair plus a raw leftover byte when N is odd; FLOAT32 is 3 B
// per value.
inline __host__ __device__ uint32_t ans_mantissas_bytes(AnsStreamType type, uint32_t uncomp_bytes)
{
  switch (type)
  {
    case AnsStreamType::Fp16:
      return uncomp_bytes / ans_bytes_per_symbol(type);
    case AnsStreamType::Fp8:
      return nvcomp::roundUpDiv(uncomp_bytes, ans_bytes_per_symbol(type));
    case AnsStreamType::Fp32:
      return ans_mantissa_bytes_per_symbol(type) * ans_num_symbols(type, uncomp_bytes);
    case AnsStreamType::Char:
      return 0u;
    default:
      assert(false);
      return 0u;
  }
}

// Number of int16_t norm_counts entries the window occupies. min <= max always holds (an
// empty chunk stores the degenerate [0, 0]), so this is never zero.
inline __host__ __device__ uint32_t ans_symbol_range_entries(uint8_t min_symbol, uint8_t max_symbol)
{
  return max_symbol - min_symbol + 1;
}

// Forward declare these helpers so that ANS_fixed_header can use them.
// These rely on the ANS_fixed_header layout, so they must be defined after it.
inline __host__ __device__ uint32_t ans_sub_chunk_sizes_offset();
inline __host__ __device__ uint32_t ans_mantissas_offset(AnsStreamType type, uint32_t num_sub_chunks);
inline __host__ __device__ uint32_t
ans_norm_counts_offset(AnsStreamType type, uint32_t uncomp_bytes, uint32_t num_sub_chunks);
inline __host__ __device__ uint32_t ans_sub_chunk_0_offset(
  AnsStreamType type,
  uint32_t uncomp_bytes,
  uint32_t num_sub_chunks,
  uint8_t min_symbol,
  uint8_t max_symbol
);

// 16-byte header at offset 0. uncomp_bytes_ is always uncompressed bytes (every type).
// type_ is AnsStreamType; states_per_lane_ is the count (1, 2, or reserved 4).
// Mantissa length and num_sub_chunks are derived, not stored.
// num_sub_chunks = max(1, ceil_div(uncomp_bytes_, max_sub_chunk_size_)); sizes sit
// immediately after this header (already uint32-aligned). Every later position is an
// offset, never an address, so a chunk is valid at any 16 B-aligned address.
struct __align__(16) ANS_fixed_header
{
  uint32_t uncomp_bytes_;
  uint32_t max_sub_chunk_size_; // bytes
  uint8_t min_symbol_;
  uint8_t max_symbol_;
  uint8_t tablelog_; // configured host-mode tablelog, 10..11
  uint8_t type_; // AnsStreamType
  uint8_t states_per_lane_;
  uint8_t reserved_[3];

  __host__ __device__ AnsStreamType type() const { return static_cast<AnsStreamType>(type_); }

  __host__ __device__ uint8_t states_per_lane() const { return states_per_lane_; }

  __host__ __device__ uint32_t num_sub_chunks() const
  {
    return ans_derived_num_sub_chunks(uncomp_bytes_, max_sub_chunk_size_);
  }

  __host__ __device__ uint32_t mantissas_bytes() const { return ans_mantissas_bytes(type(), uncomp_bytes_); }

  __host__ __device__ uint32_t mantissas_offset() const { return ans_mantissas_offset(type(), num_sub_chunks()); }

  __host__ __device__ uint32_t norm_counts_offset() const
  {
    return ans_norm_counts_offset(type(), uncomp_bytes_, num_sub_chunks());
  }

  __host__ __device__ uint32_t sub_chunk_0_offset() const
  {
    return ans_sub_chunk_0_offset(type(), uncomp_bytes_, num_sub_chunks(), min_symbol_, max_symbol_);
  }

  __host__ __device__ uint32_t *get_sub_chunk_sizes()
  {
    return reinterpret_cast<uint32_t *>(reinterpret_cast<uint8_t *>(this) + ans_sub_chunk_sizes_offset());
  }

  const __host__ __device__ uint32_t *get_sub_chunk_sizes() const
  {
    return reinterpret_cast<const uint32_t *>(reinterpret_cast<const uint8_t *>(this) + ans_sub_chunk_sizes_offset());
  }

  __host__ __device__ int16_t *get_norm_counts()
  {
    return reinterpret_cast<int16_t *>(reinterpret_cast<uint8_t *>(this) + norm_counts_offset());
  }

  const __host__ __device__ int16_t *get_norm_counts() const
  {
    return reinterpret_cast<const int16_t *>(reinterpret_cast<const uint8_t *>(this) + norm_counts_offset());
  }

  __host__ __device__ uint8_t *get_sub_chunk_0_start()
  {
    return reinterpret_cast<uint8_t *>(this) + sub_chunk_0_offset();
  }

  const __host__ __device__ uint8_t *get_sub_chunk_0_start() const
  {
    return reinterpret_cast<const uint8_t *>(this) + sub_chunk_0_offset();
  }

  __host__ __device__ void init(
    uint32_t uncomp_bytes,
    uint32_t max_sub_chunk_size,
    uint8_t min_symbol,
    uint8_t max_symbol,
    uint8_t tablelog,
    AnsStreamType type,
    uint8_t states_per_lane
  )
  {
    uncomp_bytes_ = uncomp_bytes;
    max_sub_chunk_size_ = max_sub_chunk_size;
    min_symbol_ = min_symbol;
    max_symbol_ = max_symbol;
    tablelog_ = tablelog;
    type_ = static_cast<uint8_t>(type);
    states_per_lane_ = states_per_lane;
    reserved_[0] = 0;
    reserved_[1] = 0;
    reserved_[2] = 0;
  }
};

static_assert(sizeof(ANS_fixed_header) == 16, "ANS_fixed_header size is part of the chunk format");
static_assert(
  alignof(ANS_fixed_header) == ANS_SUB_CHUNK_SLOT_ALIGN,
  "ANS_fixed_header alignment must match 16-byte chunk / uint4 slot alignment"
);

inline __host__ __device__ uint32_t ans_sub_chunk_sizes_offset()
{
  return static_cast<uint32_t>(sizeof(ANS_fixed_header));
}

inline __host__ __device__ uint32_t ans_mantissas_offset(AnsStreamType type, uint32_t num_sub_chunks)
{
  const uint32_t after_sizes = ans_sub_chunk_sizes_offset() + static_cast<uint32_t>(sizeof(uint32_t)) * num_sub_chunks;
  if (ans_mantissa_bytes_per_symbol(type) != 0u)
  {
    // uint2 mantissa accesses need an 8-aligned region.
    return nvcomp::roundUpTo(after_sizes, static_cast<uint32_t>(sizeof(uint2)));
  }
  return after_sizes;
}

inline __host__ __device__ uint32_t
ans_norm_counts_offset(AnsStreamType type, uint32_t uncomp_bytes, uint32_t num_sub_chunks)
{
  return nvcomp::roundUpTo(
    ans_mantissas_offset(type, num_sub_chunks) + ans_mantissas_bytes(type, uncomp_bytes),
    static_cast<uint32_t>(sizeof(int16_t))
  );
}

inline __host__ __device__ uint32_t ans_sub_chunk_0_offset(
  AnsStreamType type,
  uint32_t uncomp_bytes,
  uint32_t num_sub_chunks,
  uint8_t min_symbol,
  uint8_t max_symbol
)
{
  return nvcomp::roundUpTo(
    ans_norm_counts_offset(type, uncomp_bytes, num_sub_chunks) +
      static_cast<uint32_t>(sizeof(int16_t)) * ans_symbol_range_entries(min_symbol, max_symbol),
    ANS_SUB_CHUNK_SLOT_ALIGN
  );
}

// Worst-case final-state tail written by encode_final_state, in bytes: one 32-bit
// state (TAIL_U16_PER_STATE uint16) per state, times WARP_SIZE lanes.
inline constexpr __host__ __device__ uint32_t get_final_state_tail_bytes(uint32_t states_per_lane)
{
  return states_per_lane * WARP_SIZE * static_cast<uint32_t>(TAIL_U16_PER_STATE) * sizeof(uint16_t);
}

// Given an input to compress of size `uncomp_chunk_size` (symbols), the encoder's
// states per lane, and its tablelog, return the worst-case compressed size of one
// sub-chunk slot.
inline __host__ __device__ uint32_t
get_max_comp_sub_chunk_size(uint32_t uncomp_chunk_size, uint32_t states_per_lane, uint32_t tablelog)
{
  // The worst case is that each input symbol takes `tablelog` bits. Round that
  // bit count up to the uint16 renormalization-word granularity.
  constexpr uint32_t bits_per_u16 = static_cast<uint32_t>(sizeof(uint16_t) * BITS_PER_BYTE);
  const uint32_t normalization_words = nvcomp::roundUpDiv(tablelog * uncomp_chunk_size, bits_per_u16);
  const uint32_t normalization_bytes = normalization_words * static_cast<uint32_t>(sizeof(uint16_t));
  const uint32_t total_bytes = normalization_bytes + get_final_state_tail_bytes(states_per_lane);

  // Each sub-chunk slot starts on an ANS_SUB_CHUNK_SLOT_ALIGN (16 B) boundary so
  // consecutive slots stay uint4-aligned for compacting defrag.
  return nvcomp::roundUpTo(total_bytes, ANS_SUB_CHUNK_SLOT_ALIGN);
}

// Slot bases need no rounding: get_max_comp_sub_chunk_size() returns a multiple of
// ANS_SUB_CHUNK_SLOT_ALIGN, and slot 0 starts at a multiple of it, so every slot inherits
// that alignment from the chunk.
inline __device__ uint8_t *
get_comp_sub_chunk_output_ptr(uint32_t sub_chunk_idx, uint8_t *sub_chunk_0_start, uint32_t comp_sub_chunk_slot_bytes)
{
  return sub_chunk_0_start + sub_chunk_idx * comp_sub_chunk_slot_bytes;
}

template <bool BOUNDS_CHECK, typename T1, typename T2>
__device__ T1 safe_guard_generic(T1 *address, T2 *lb, T2 *ub, nvcompStatus_t *err = nullptr)
{
  if constexpr (BOUNDS_CHECK)
  {
    if ((reinterpret_cast<intptr_t>(lb) > reinterpret_cast<intptr_t>(address)) ||
        (reinterpret_cast<intptr_t>(address) + sizeof(T1) > reinterpret_cast<intptr_t>(ub)))
    {
      if (err)
      {
        *err = nvcompErrorCannotDecompress;
      }
      return 0;
    }
  }
  return *address;
}

template <bool BOUNDS_CHECK, typename T, typename S>
__device__ T safe_guard_generic(T index, S max_size)
{
  if (BOUNDS_CHECK && index >= max_size)
  {
    return 0;
  }
  return index;
}

} // namespace

namespace detail
{
namespace
{

__device__ int get_lane_id()
{
  int id;
  asm("mov.u32 %0, %%laneid;" : "=r"(id));
  return id;
}

inline __device__ uint32_t get_lane_mask()
{
  uint32_t mask;
  asm("mov.u32 %0, %%lanemask_le;" : "=r"(mask));
  return mask;
}

// lanemask_lt (get_lane_mask() without this lane's own bit). The two-state decode_pair
// uses it to mirror symbol_encoder_dual_state::encode_pair's lanemask_ge addressing.
inline __device__ uint32_t get_lane_mask_lt()
{
  uint32_t mask;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(mask));
  return mask;
}

// High-lane-first densify mask (lanes > self). Rematerialized via S2R at each use so the
// encoder does not park a constant in a live register across the encode loop.
inline __device__ uint32_t get_lane_mask_gt()
{
  uint32_t mask;
  asm("mov.u32 %0, %%lanemask_gt;" : "=r"(mask));
  return mask;
}

// lanemask_ge (= gt | own bit). encode_pair addresses group 1 with this so
// popc(w1 & ge) == popc(w1 & gt) + need1 (see symbol_encoder_dual_state::encode_pair).
inline __device__ uint32_t get_lane_mask_ge()
{
  uint32_t mask;
  asm("mov.u32 %0, %%lanemask_ge;" : "=r"(mask));
  return mask;
}

// Bitwise select: MASK bits come from `on_bits`, the rest from `off_bits`. Written as PTX
// because the equivalent C++ (`(a & MASK) | (b & ~MASK)`, in any phrasing) does not survive
// the front end: whenever a caller discards part of the result, demanded-bit simplification
// narrows ~MASK to only the live bits, the two masks stop being complementary, and the merge
// costs two LOP3 instead of one. immLut 0xE4 is the mux truth table over (a, b, MASK).
template <uint32_t MASK>
inline __device__ uint32_t select_bits(uint32_t on_bits, uint32_t off_bits)
{
  uint32_t selected;
  asm("lop3.b32 %0, %1, %2, %3, 0xE4;" : "=r"(selected) : "r"(on_bits), "r"(off_bits), "n"(MASK));
  return selected;
}

__device__ uint32_t read_metadata_uncomp_block_size(const void *comp_chunk, const uint32_t comp_chunk_size)
{
  if (comp_chunk_size < sizeof(ANS_fixed_header))
  {
    return 0;
  }

  const ANS_fixed_header *fixed_header = reinterpret_cast<const ANS_fixed_header *>(comp_chunk);
  return fixed_header->uncomp_bytes_;
}

// One-value-per-thread exclusive prefix sum, for the callers that do not need the block
// total back. Each warp scans itself and the per-warp totals are combined with a single
// reduction, which is cheaper than a full BlockScan and needs only WARPS words of shared
// memory instead of the scan's temp storage.
template <int BLOCK_DIM_X>
__device__ uint32_t block_excl_prefix_sum(uint32_t val)
{
  static_assert(BLOCK_DIM_X % WARP_SIZE == 0, "block_excl_prefix_sum needs whole warps");
  constexpr int WARPS = BLOCK_DIM_X / WARP_SIZE;
  const int lane = static_cast<int>(threadIdx.x % WARP_SIZE_U);
  const int warp = static_cast<int>(threadIdx.x / WARP_SIZE_U);

  using WarpScanT = nvcomp::cub::WarpScan<uint32_t>;
  __shared__ typename WarpScanT::TempStorage warp_scan_tmp[WARPS];
  uint32_t excl;
  uint32_t warp_sum;
  WarpScanT(warp_scan_tmp[warp]).ExclusiveSum(val, excl, warp_sum);

  __shared__ uint32_t warp_totals[WARPS];
  if (lane == 0)
  {
    warp_totals[warp] = warp_sum;
  }
  __syncthreads();

  // Exclusive sum of the totals of all warps below this one.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const uint32_t warp_excl = __reduce_add_sync(WARP_ALL, (lane < warp) ? warp_totals[lane] : 0u);
#else
  uint32_t warp_excl = 0;
#pragma unroll
  for (int w = 0; w < WARPS; ++w)
  {
    warp_excl += (w < warp) ? warp_totals[w] : 0u;
  }
#endif

  return excl + warp_excl;
}

// Inclusive prefix of one value per thread. Warp scan when num_items fits in a warp,
// otherwise a block scan. Threads with tid >= num_items must pass val = 0. All
// THREADS_PER_BLOCK threads must participate. Returns this thread's inclusive prefix;
// if tid < num_items, also writes it to offsets[tid].
template <int THREADS_PER_BLOCK>
__device__ uint32_t cta_inclusive_prefix_sum(uint32_t *offsets, uint32_t val, uint32_t num_items)
{
  static_assert(MAX_SUB_CHUNKS_PER_CHUNK <= THREADS_PER_BLOCK, "BlockScan must cover every sub-chunk");

  using WarpScan = nvcomp::cub::WarpScan<uint32_t>;
  using BlockScan = nvcomp::cub::BlockScan<uint32_t, THREADS_PER_BLOCK>;
  union ScanTmp
  {
    typename WarpScan::TempStorage warp;
    typename BlockScan::TempStorage block;
  };
  __shared__ ScanTmp scan_tmp;

  const uint32_t tid = threadIdx.x;
  uint32_t inclusive = 0;
  if (num_items <= WARP_SIZE)
  {
    if (tid < WARP_SIZE)
    {
      WarpScan(scan_tmp.warp).InclusiveSum(val, inclusive);
    }
  }
  else
  {
    BlockScan(scan_tmp.block).InclusiveSum(val, inclusive);
  }

  if (tid < num_items)
  {
    offsets[tid] = inclusive;
  }
  return inclusive;
}

template <typename T>
__device__ T warp_incl_prefix_sum(int lane_id, T val)
{
#pragma unroll
  for (uint32_t i = 1; i < WARP_SIZE_U; i *= 2)
  {
    uint32_t n = __shfl_up_sync(WARP_ALL, val, i, WARP_SIZE);
    if (lane_id >= i)
    {
      val += n;
    }
  }
  return val;
}

template <typename T, typename CG>
__device__ void block_excl_prefix_sum(T lane_value, T &lane_result, T *shared_buffer, T &total, CG &group)
{
  constexpr int warp_count = CG::size() / WARP_SIZE;
  const int lane_id = group.thread_rank() % WARP_SIZE;
  const int warp_id = group.thread_rank() / WARP_SIZE;
  // Each warp does a warp-level inclusive prefix sum
  T lane_prefix = warp_incl_prefix_sum(lane_id, lane_value);
  // Top lanes write to shared memory
  if (lane_id == (WARP_SIZE - 1))
  {
    shared_buffer[warp_id] = lane_prefix;
  }
  group.sync();

  if (warp_id == 0)
  {
    // Now the very first warp does a reduction again
    T warp_value = lane_id < warp_count ? shared_buffer[lane_id] : 0;
    T warp_prefix = warp_incl_prefix_sum(lane_id, warp_value);
    if (lane_id < warp_count)
    {
      // Note:
      // We are not writing to [line_id], but instead we use the bin
      // [warp_count + lane_id], i.e., shifted by the number of warps
      // collaborating. This way, we can avoid one thread block-wide
      // synchronization, as a fast warp can proceed in the for loop
      // without affecting a slow warp still reading out the previous
      // iteration's result.
      shared_buffer[warp_count + lane_id] = warp_prefix;
    }
  }
  group.sync();

  // Write out lane result
  // Note:
  // The idea from above is brought down here, instead of
  // reading from [warp_id - 1], we are reading from the [warp_count + warp_id - 1]
  // and similarly, instead of reading from [warp_count - 1] we are reading from [warp_count*2 - 1].
  lane_result = lane_prefix - lane_value + (warp_id > 0 ? shared_buffer[warp_count + warp_id - 1] : 0);
  total = shared_buffer[warp_count * 2 - 1];
}

template <typename T>
__device__ T warp_excl_prefix_sum(int lane_id, T val, T &total)
{
  T init_val = val;
#pragma unroll
  for (uint32_t i = 1; i < WARP_SIZE_U; i *= 2)
  {
    uint32_t n = __shfl_up_sync(WARP_ALL, val, i, WARP_SIZE);
    if (lane_id >= i)
    {
      val += n;
    }
  }
  total = __shfl_sync(WARP_ALL, val, WARP_SIZE - 1);
  return val - init_val;
}

// Warp-wide sum, broadcast to every lane.
inline __device__ uint32_t warp_reduce_add(uint32_t value)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return __reduce_add_sync(WARP_ALL, value);
#else
  for (uint32_t i = WARP_SIZE_U / 2; i > 0; i >>= 1)
  {
    value += __shfl_xor_sync(WARP_ALL, value, i, WARP_SIZE);
  }
  return value;
#endif
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
template <typename T>
__device__ T reduce_max(T symbol)
{
  return __reduce_max_sync(WARP_ALL, symbol);
}

#else
template <typename T>
__device__ T reduce_max(T symbol)
{
  for (uint32_t i = WARP_SIZE_U / 2; i > 0; i >>= 1)
  {
    uint8_t next_symbol = __shfl_down_sync(WARP_ALL, symbol, i, WARP_SIZE);
    symbol = umax(symbol, next_symbol);
  }
  return __shfl_sync(WARP_ALL, symbol, 0);
}

#endif

} // namespace
} // namespace detail
} // namespace ans_gpu_lib
