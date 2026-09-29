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

#include <cuda/atomic>

#include <cstdio>
#include <type_traits>

#include "CudaConstants.h"
#include "include/types.h"
#include "nvcomp_device_common.cuh"

#include <assert.h>
#include <cooperative_groups.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

// Like assert but it prints and waits a little, so that the printing
// will surely complete before the failure.
#if !!ASSERTIONS == true
#define expect_op(expression, value, op)                                                                               \
  do                                                                                                                   \
  {                                                                                                                    \
    int64_t __x = (expression);                                                                                        \
    int64_t __y = (value);                                                                                             \
    if (!(__x op __y))                                                                                                 \
    {                                                                                                                  \
      printf(                                                                                                          \
        "Assertion failed: expected %s %s %s"                                                                          \
        " but actually !(%lld%s%lld)"                                                                                  \
        ", file " __FILE__ ", line " TOSTRING(__LINE__) "\n",                                                          \
        #expression,                                                                                                   \
        #op,                                                                                                           \
        #value,                                                                                                        \
        __x,                                                                                                           \
        #op,                                                                                                           \
        __y                                                                                                            \
      );                                                                                                               \
      assert(__x == __y);                                                                                              \
    }                                                                                                                  \
  } while (false)
#else
#define expect_op(expression, value, op)                                                                               \
  do                                                                                                                   \
  {                                                                                                                    \
  } while (false)
#endif

#define expect_eq(expression, value) expect_op(expression, value, ==)
#define expect_lt(expression, value) expect_op(expression, value, <)
#define expect_le(expression, value) expect_op(expression, value, <=)

using uint = unsigned int;

static constexpr __host__ __device__ uint const_ffs(uint x) { return (x == 1) ? 0 : 1 + const_ffs(x >> 1); }

static constexpr uint const_min(uint x, uint y) { return x < y ? x : y; }

static constexpr __device__ uint const_max(uint x, uint y) { return x > y ? x : y; }

// Return power of two same as x or bigger.
static constexpr __device__ uint upper_power_of_two(uint x) { return (1 << const_ffs(x - 1)) << 1; }

// Extra bits for the code length codes
const __device__ uint8_t CLC_EXTRA_BITS[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 7};

inline __device__ uint8_t get_clc_extra_bits(uint symbol)
{
  // Speculative threads may decode from invalid offsets producing garbage symbols.
  // Returning 0 is safe, as these paths are discarded once real stream boundaries resolve.
  return symbol < sizeof(CLC_EXTRA_BITS) ? CLC_EXTRA_BITS[symbol] : 0;
}

const __device__ uint8_t CLC_UNCOMPRESSED_SIZE_TABLE[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 11};

inline __device__ uint8_t get_clc_uncompressed_size(uint symbol)
{
  // Speculative threads may decode from invalid offsets producing garbage symbols.
  // Returning 0 is safe, as these paths are discarded once real stream boundaries resolve.
  return symbol < sizeof(CLC_UNCOMPRESSED_SIZE_TABLE) ? CLC_UNCOMPRESSED_SIZE_TABLE[symbol] : 0;
}

constexpr uint LITLEN_COUNT = 286;
constexpr uint DIST_COUNT = 30;

const __device__ uint8_t EXTRA_BITS[2][const_max(LITLEN_COUNT, DIST_COUNT)] = {
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0},
  {0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13}
};
inline __device__ uint8_t get_extra_bits(uint table, uint symbol) { return EXTRA_BITS[table][symbol]; }

const __device__ uint16_t DISTANCE_BASE_TABLE[] = {1,    2,    3,    4,    5,    7,    9,    13,    17,    25,
                                                   33,   49,   65,   97,   129,  193,  257,  385,   513,   769,
                                                   1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
inline __device__ uint16_t get_distance_base(uint symbol) { return DISTANCE_BASE_TABLE[symbol]; }

// This tells the uncompressed size of the current symbol being
// decoded.  Page 11 RFC1951.
const __device__ uint16_t UNCOMPRESSED_SIZE_TABLE[2][const_max(LITLEN_COUNT, DIST_COUNT)] = {
  {1,   1,   1,   1,   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1, 1, 1, 1, 1, 1, 1,
   1,   1,   1,   1,   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1, 1, 1, 1, 1, 1, 1,
   1,   1,   1,   1,   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1, 1, 1, 1, 1, 1, 1,
   1,   1,   1,   1,   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1, 1, 1, 1, 1, 1, 1,
   1,   1,   1,   1,   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1, 1, 1, 1, 1, 1, 1,
   1,   1,   1,   1,   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1, 1, 1, 1, 1, 1, 1,
   1,   1,   1,   1,   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1, 1, 1, 1, 1, 1, 1,
   1,   1,   1,   1,   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 3, 4, 5, 6, 7, 8, 9, 10, // 0 extra bits.
   11,  13,  15,  17, // 1 extra bit.
   19,  23,  27,  31, // 2 extra bits.
   35,  43,  51,  59, // 3 extra bits.
   67,  83,  99,  115, // 4 extra bits.
   131, 163, 195, 227, // 5 extra bits.
   258},
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
};
inline __device__ uint16_t get_uncompressed_size(uint table, uint symbol)
{
  return UNCOMPRESSED_SIZE_TABLE[table][symbol];
}

constexpr uint BLOCK_TYPE_MASK = 0x6; // bits[2:1] are the block type.
// The block types below are already shifted to be aligned with the
// block type mask.
constexpr uint BLOCK_TYPE_NO_COMPRESSION = 0x0;
constexpr uint BLOCK_TYPE_FIXED = 0x2;
constexpr uint BLOCK_TYPE_DYNAMIC = 0x4;

constexpr uint BLOCK_FINAL_MASK = 0x1; // bits[0] is the final bit.

constexpr uint MAX_LOOKBACK = 32768;
constexpr uint MAX_SYMBOL_BITS = 15;

// There are two trees, litlen and distance.
constexpr uint LITLEN_HUFFMAN_INDEX = 0;
constexpr uint DISTANCE_HUFFMAN_INDEX = 1;
constexpr uint HUFFMAN_TREES_COUNT = 2;

template <uint THREADS_PER_BLOCK, size_t SIZE, typename S, typename D>
inline __device__ void copy_with_runt(D &dst, const size_t dst_index, const S &src, const size_t src_index)
{
  for (size_t offset = 0; SIZE > THREADS_PER_BLOCK && offset < SIZE / THREADS_PER_BLOCK; offset++)
  {
    size_t index = offset * THREADS_PER_BLOCK + threadIdx.x;
    dst[dst_index + index] = src[src_index + index];
  }
  if (threadIdx.x < SIZE % THREADS_PER_BLOCK)
  {
    size_t index = SIZE / THREADS_PER_BLOCK * THREADS_PER_BLOCK + threadIdx.x;
    dst[dst_index + index] = src[src_index + index];
  }
}

// Copy `size` bytes from src to dst using all the threads in the
// block.  dst can be global memory if only one block is used or it
// can be shared memory if it's for multiple blocks.
template <uint THREADS_PER_BLOCK, size_t SIZE>
inline __device__ void shmemcpy(volatile void *dst, volatile const void *src)
{
  // So that we don't have to deal with any rounding.
  using copy_type = uint32_t;
  static_assert(
    SIZE % sizeof(copy_type) == 0,
    "The shared memory being copied must have a size that is "
    "a multiple of the copy_type size."
  );
  constexpr uint elements_to_copy = SIZE / sizeof(copy_type);

  for (size_t offset = 0; elements_to_copy > THREADS_PER_BLOCK && offset < elements_to_copy / THREADS_PER_BLOCK;
       offset++)
  {
    size_t index = offset * THREADS_PER_BLOCK + threadIdx.x;
    (reinterpret_cast<volatile copy_type *>(dst))[index] = (reinterpret_cast<volatile const copy_type *>(src))[index];
  }
  size_t index = elements_to_copy / THREADS_PER_BLOCK * THREADS_PER_BLOCK + threadIdx.x;
  if (index < elements_to_copy)
  {
    (reinterpret_cast<volatile copy_type *>(dst))[index] = (reinterpret_cast<volatile const copy_type *>(src))[index];
  }
}

inline __device__ size_t atomicExch(volatile size_t *address, size_t value)
{
  static_assert(
    sizeof(size_t) == sizeof(unsigned long long int),
    "This architecture is expected to have size_t that is the same "
    "size as unsigned long long, which atomicExch supports."
  );
  return atomicExch((unsigned long long int *)(address), (unsigned long long int)(value));
}

// Byte-safe read of 32 bits at an arbitrary bit offset.
// Handles partial words at the boundary via loadUpTo4Bytes.
inline __device__ uint32_t
get_aligned_32_bits_safe(const uint8_t *data, size_t bit_offset, size_t data_offset, ptrdiff_t data_size_bytes)
{
  ptrdiff_t lo_byte = static_cast<ptrdiff_t>(bit_offset / 32 + data_offset) * 4;
  uint32_t lo = nvcomp::loadUpTo4Bytes(data + lo_byte, data_size_bytes - lo_byte);
  uint32_t hi = nvcomp::loadUpTo4Bytes(data + lo_byte + 4, data_size_bytes - (lo_byte + 4));
  return __funnelshift_r(lo, hi, bit_offset);
}

// Return the 32 bits at 32*data_offset+bit_offset into the data.
template <bool BOUNDS_CHECK = false, typename T>
inline __device__ uint32_t
get_aligned_32_bits(const T &data, size_t bit_offset, size_t data_offset = 0, size_t data_size_bytes = 0)
{
  static_assert(
    alignof(std::remove_reference_t<decltype(data[0])>) == alignof(uint32_t),
    "The data should be 32-bit aligned."
  );
  if constexpr (BOUNDS_CHECK)
  {
    return get_aligned_32_bits_safe(reinterpret_cast<const uint8_t *>(data), bit_offset, data_offset, data_size_bytes);
  }
  else
  {
    uint lo_index = bit_offset / (sizeof(data[0]) * 8) + data_offset;
    uint hi_index = lo_index + 1;
    return __funnelshift_r(data[lo_index], data[hi_index], bit_offset);
  }
}

// This behaves like a cooperative_group::grid but potentially for a
// subset of the blocks instead of all of them.
class PartialGrid
{
public:
  __host__ PartialGrid(uint block_count)
      : generation(0)
      , count(block_count)
      , total(block_count)
  {}

  __device__ void init(uint block_count)
  {
    generation.store(0, cuda::std::memory_order_relaxed);
    count.store(block_count, cuda::std::memory_order_relaxed);
    total = block_count;
  }

  __device__ uint get_total() { return total; }

  __device__ void sync()
  {
    // This block-wide synchronization is required as all threads need to finish
    // writing into global memory before the leader arrives at the barrier.
    // This is because the __syncthreads at the end of this function guarantees
    // a memory barrier only within CTA, not grid-wide.
    __syncthreads();

    if (threadIdx.x == 0)
    {
      int gen = generation.load(cuda::std::memory_order_relaxed);
      if (count.fetch_sub(1, cuda::std::memory_order_acq_rel) == 1)
      {
        count.store(total, cuda::std::memory_order_relaxed);
        generation.store(gen + 1, cuda::std::memory_order_release);
      }
      else
      {
        // spinlock
        while (generation.load(cuda::std::memory_order_acquire) == gen)
        {
          __nanosleep(10);
        }
      }
    }

    __syncthreads();
  }

private:
  cuda::atomic<uint, cuda::thread_scope_device> generation;
  cuda::atomic<uint, cuda::thread_scope_device> count;
  uint total;
};
