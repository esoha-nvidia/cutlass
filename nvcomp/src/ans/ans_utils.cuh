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

// Can tune this to get a compromise between compression ratio and
// mitigating tail effects. More waves per SM -> less tail effects from unequal block execution time.
// Fewer waves per SM -> fewer sub chunks -> less compression ratio overhead
constexpr int NUM_WAVES_PER_SM = 1;

// alignment must be a power of 2
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

// The first size_t of a compressed chunk is the "mantissas_size" prefix (offset
// from the chunk start to the 8-byte-aligned ANS sub-chunk header; equivalently
// the size of the side-band region). The actual value is bounded by the chunk
// size (<= 2^25), so the top bits are free. We stash a 1-bit data-type marker in
// the MSB to make FP8 self-describing: it carries a side-band like FP16
// (mantissas_size > sizeof(size_t)), so the size alone cannot distinguish them.
//   bit 63 set  -> FP8 (E4M3); uncomp_chunk_size_ holds the true byte count N
//   bit 63 clear -> char (mantissas_size == sizeof(size_t)) or FP16 (> sizeof)
// Readers must mask the flag off before using the value as a length/offset.
constexpr size_t ANS_MANTISSAS_SIZE_FP8_FLAG = (static_cast<size_t>(1) << 63);

inline __host__ __device__ bool ans_mantissas_size_is_fp8(size_t raw_mantissas_size)
{
  return (raw_mantissas_size & ANS_MANTISSAS_SIZE_FP8_FLAG) != 0;
}

inline __host__ __device__ size_t ans_mantissas_size_value(size_t raw_mantissas_size)
{
  return raw_mantissas_size & ~ANS_MANTISSAS_SIZE_FP8_FLAG;
}

struct __align__(8) ANS_sub_chunk_header
{
  size_t num_sub_chunks_;
  size_t uncomp_chunk_size_;
  size_t max_sub_chunk_size_;
  uint32_t max_symbol_value_;
  uint32_t tablelog_;
  // The next three fields are implied: They exist off
  // the end of the struct where they would be if they were
  // actual struct members.
  // int16_t norm_counts[max_symbol_value+1]
  // size_t sub_chunk_offsets[num_sub_chunks]
  // size_t sub_chunk_sizes[num_sub_chunks]

  __device__ void init(
    size_t num_sub_chunks,
    size_t uncomp_chunk_size,
    uint32_t max_sub_chunk_size,
    uint32_t max_symbol_value,
    uint32_t tablelog
  )
  {

    num_sub_chunks_ = num_sub_chunks;
    uncomp_chunk_size_ = uncomp_chunk_size;
    max_sub_chunk_size_ = max_sub_chunk_size;
    max_symbol_value_ = max_symbol_value;
    tablelog_ = tablelog;
  }

  __device__ int16_t *get_norm_counts() { return reinterpret_cast<int16_t *>(&(this->tablelog_) + 1); }

  const __device__ int16_t *get_norm_counts() const
  {
    return reinterpret_cast<const int16_t *>(&(this->tablelog_) + 1);
  }
  // TODO these should probably be ints
  __device__ size_t *get_sub_chunk_offsets()
  {
    return reinterpret_cast<size_t *>(round_up_align_address(get_norm_counts() + max_symbol_value_ + 1, 8));
  }

  const __device__ size_t *get_sub_chunk_offsets() const
  {
    return reinterpret_cast<const size_t *>(round_up_align_address(get_norm_counts() + max_symbol_value_ + 1, 8));
  }

  __device__ size_t *get_sub_chunk_sizes() { return get_sub_chunk_offsets() + num_sub_chunks_; }

  const __device__ size_t *get_sub_chunk_sizes() const { return get_sub_chunk_offsets() + num_sub_chunks_; }

  __device__ size_t get_header_size() const
  {
    return reinterpret_cast<uintptr_t>(get_sub_chunk_sizes() + num_sub_chunks_) - reinterpret_cast<uintptr_t>(this);
  }

  // uchar4-aligned start of sub-chunk 0 ANS output, immediately after this header.
  __device__ uint8_t *get_sub_chunk_0_start()
  {
    return static_cast<uint8_t *>(
      round_up_align_address(reinterpret_cast<uint8_t *>(this) + get_header_size(), sizeof(uchar4))
    );
  }

  const __device__ uint8_t *get_sub_chunk_0_start() const
  {
    return static_cast<const uint8_t *>(
      round_up_align_address(reinterpret_cast<const uint8_t *>(this) + get_header_size(), sizeof(uchar4))
    );
  }

  __device__ size_t get_sub_chunk_0_offset() const
  {
    return reinterpret_cast<uintptr_t>(get_sub_chunk_0_start()) - reinterpret_cast<uintptr_t>(this);
  }

  // comp_chunk must be 8-byte aligned before calling.
  static inline __host__ __device__ ANS_sub_chunk_header *from_aligned_comp_chunk(void *comp_chunk)
  {
    const size_t mantissas_size = ans_mantissas_size_value(*reinterpret_cast<size_t *>(comp_chunk));
    uint8_t *const ans_comp_chunk =
      static_cast<uint8_t *>(round_up_align_address(reinterpret_cast<uint8_t *>(comp_chunk) + mantissas_size, 8));
    return reinterpret_cast<ANS_sub_chunk_header *>(ans_comp_chunk);
  }

  // comp_chunk must be 8-byte aligned before calling.
  static inline const __host__ __device__ ANS_sub_chunk_header *from_aligned_comp_chunk(const void *comp_chunk)
  {
    const size_t mantissas_size = ans_mantissas_size_value(*reinterpret_cast<const size_t *>(comp_chunk));
    const uint8_t *const ans_comp_chunk = static_cast<const uint8_t *>(
      round_up_align_address(reinterpret_cast<const uint8_t *>(comp_chunk) + mantissas_size, 8)
    );
    return reinterpret_cast<const ANS_sub_chunk_header *>(ans_comp_chunk);
  }
};

// In addition to the ANS normalizations, what is the maximum overhead for one subchunk?
inline constexpr __host__ __device__ size_t get_sub_chunk_overhead() noexcept
{
  // The symbol_encoder's state is a uint16_t.
  using state_type = uint16_t;
  // Each of the 32 threads must write the remainder of the 16-bit final state.
  return sizeof(state_type) * WARP_SIZE;
}

// Given an input to compress of size `uncomp_chunk_size`, return the worst-case compressed size of one subchunk.
inline __host__ __device__ size_t get_max_comp_sub_chunk_size(size_t uncomp_chunk_size)
{
  // The worst case for compression is that each input byte takes up the DEFAULT_TABLELOG bits.  This number of bits needs to be rounded up to multiple of the normalization_size.
  const auto normalization_words = (DEFAULT_TABLELOG * uncomp_chunk_size + (sizeof(uint16_t) * BITS_PER_BYTE - 1)) /
                                   (sizeof(uint16_t) * BITS_PER_BYTE);
  const auto normalization_bytes = normalization_words * sizeof(uint16_t);
  constexpr auto all_overhead_bytes = get_sub_chunk_overhead();

  const auto total_bytes = normalization_bytes + all_overhead_bytes;

  // Each subchunk starts on a 32-bit boundary so if the total size of this one is not 32-bit aligned, we need to add padding.

  return nvcomp::roundUpTo(total_bytes, sizeof(uint32_t));
}

inline __device__ uint8_t *
get_comp_sub_chunk_output_ptr(size_t sub_chunk_idx, uint8_t *sub_chunk_0_start, size_t comp_sub_chunk_slot_bytes)
{
  return static_cast<uint8_t *>(
    round_up_align_address(sub_chunk_0_start + sub_chunk_idx * comp_sub_chunk_slot_bytes, sizeof(uint32_t))
  );
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

__device__ size_t read_metadata_uncomp_block_size(const void *comp_chunk, const size_t comp_chunk_size)
{

  // sub_chunk header cannot be well formed, don't try to retrieve uncomp_chunk_size_
  if (comp_chunk_size < 2 * sizeof(size_t))
  {
    return 0;
  }

  const size_t raw_mantissas_size = *reinterpret_cast<const size_t *>(comp_chunk);
  const bool runtime_is_fp8 = ans_mantissas_size_is_fp8(raw_mantissas_size);
  const size_t mantissas_size = ans_mantissas_size_value(raw_mantissas_size);
  if (mantissas_size >= comp_chunk_size)
  {
    return 0;
  }

  const uint8_t *chunk = static_cast<const uint8_t *>(
    round_up_align_address(reinterpret_cast<const uint8_t *>(comp_chunk) + mantissas_size, 8)
  );
  const ANS_sub_chunk_header *sub_chunk_header = reinterpret_cast<const ANS_sub_chunk_header *>(chunk);

  size_t uncomp_chunk_size = sub_chunk_header->uncomp_chunk_size_;

  if (mantissas_size > sizeof(size_t) && !runtime_is_fp8)
  {
    // If mantissas_size is larger than sizeof(size_t) (and not flagged as fp8), this
    // is fp16 data and we need to 2x because the mantissa bytes aren't included
    // in the sub chunk header size.
    uncomp_chunk_size *= 2;
  }
  return uncomp_chunk_size;
}

template <int BLOCK_DIM_X, typename T>
__device__ T block_excl_prefix_sum(T val, T &acc)
{
  using Scan = nvcomp::cub::BlockScan<T, BLOCK_DIM_X>;
  __shared__ typename Scan::TempStorage smem;

  T prefix;
  Scan(smem).ExclusiveSum(val, prefix, acc);

  return prefix;
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
