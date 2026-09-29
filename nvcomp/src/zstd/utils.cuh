/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <cub/cub.cuh>

#include <cuda/atomic>
#include <cuda_runtime.h>

#include <cassert>
#include <ciso646>
#include <cstdint>

#include "constants.cuh"
#include "exception.hpp"

namespace zstd
{

// TODO: unify these with rest of nvcomp
template <typename T>
inline __host__ __device__ T ceilDiv(const T num, const T den)
{
  return (num + den - 1) / den;
}

template <typename T>
inline __host__ __device__ T roundUpTo(const T in, const T quot)
{
  return quot * ceilDiv(in, quot);
}

inline __device__ int thread_warp_ix() { return threadIdx.x % 32; }
inline __device__ int ix_warp() { return threadIdx.x / 32; }

template <typename... Ts>
inline __device__ void print0(const char *valstr, Ts &&...ts)
{
  if (thread_warp_ix() == 0)
  {
    printf(valstr, ts...);
  }
}

template <int n_threads, typename T>
inline __device__ T custom_exclusive_scan(T input, unsigned mask, int ix_thread)
{
  T output = 0;

#pragma unroll
  for (int ix_shuffle = 1; ix_shuffle < n_threads; ix_shuffle <<= 1)
  {
    T inc = __shfl_up_sync(mask, output + input, ix_shuffle);
    if (ix_thread >= ix_shuffle)
    {
      output += inc;
    }
  }
  return output;
}

template <int n_threads, typename T>
inline __device__ T custom_inclusive_scan_registers(T input, unsigned mask, int ix_thread)
{
  T output = input;

#pragma unroll
  for (int ix_shuffle = 1; ix_shuffle < n_threads; ix_shuffle <<= 1)
  {
    T inc = __shfl_up_sync(mask, output, ix_shuffle);
    if (ix_thread >= ix_shuffle)
    {
      output += inc;
    }
  }
  return output;
}

// Calculate an inclusive sum scan from the bytes of reg1 and reg2
// The total sum is going to be reg1[0:7] + reg1[8:15] + reg1[16:23] + reg1[24:31] + reg2[0:7] + reg2[8:15]
// The individual threads will receive the intermediate results:
// ix 0: reg1[0:7]
// ix 1: reg1[0:7] + reg1[8:15]
// ix 2: reg1[0:7] + reg1[8:15] + reg1[16:23]
// ix 3: reg1[0:7] + reg1[8:15] + reg1[16:23] + reg1[24:31]
// ix 4: reg1[0:7] + reg1[8:15] + reg1[16:23] + reg1[24:31] + reg2[0:7]
// ix 5: reg1[0:7] + reg1[8:15] + reg1[16:23] + reg1[24:31] + reg2[0:7] + reg2[8:15]
inline __device__ uint8_t
custom_scan_num_bits(uint8_t &total_num_bits, unsigned reg1, unsigned reg2, unsigned ix_in_block)
{
  uint8_t res;
#pragma unroll
  for (int ix = 0; ix < 4; ++ix)
  {
    total_num_bits += (reg1 >> (ix * 8)) & 0xff;
    if (ix == ix_in_block)
    {
      res = total_num_bits;
    }
  }

#pragma unroll
  for (int ix = 0; ix < 2; ++ix)
  {
    total_num_bits += (reg2 >> (ix * 8)) & 0xff;
    if (ix + 4 == ix_in_block)
    {
      res = total_num_bits;
    }
  }
  return res;
}

template <
  typename T,
  typename = std::enable_if_t<not std::is_signed<T>::value and sizeof(T) == 8 and sizeof(unsigned long long int) == 8>>
__device__ __forceinline__ uint64_t atomicAddUint64Wrapper(T *input, T add)
{
  return atomicAdd(reinterpret_cast<unsigned long long int *>(input), static_cast<unsigned long long int>(add));
}

inline __device__ int lower_bound_descending_11(
  uint16_t search_val,
  const unsigned bc2,
  const unsigned bc3,
  const unsigned bc5,
  const unsigned bc6,
  const unsigned bc8,
  const unsigned bc9,
  const unsigned bc11,
  const unsigned of0, // contains offset 11, 5, 8, 2 (first iteration results)
  const unsigned of25, // contains offsets 1, 0, 4, 3 (Half of iteration results from next iter)
  const unsigned of811, // contains offsets 7, 6, 10, 9 (Half of iteration results from next iter)
  uint8_t &offset,
  uint16_t &basecode
)
{

  /*
  This does the same as below. Since the search is predictable, we can preload what we need into registers
  if (search_array[5] <= search_val) ix_result = 5;
  if (search_array[ix_result - 3] <= search_val) ix_result -= 3;
  if (search_array[ix_result - 1] <= search_val) ix_result -= 1;
  if (search_array[ix_result - 1] <= search_val) ix_result -= 1;
  */

  const unsigned short_mask = 0xffff;
  int ix_result = 11;
  offset = of0 & 0xff;

  unsigned lookup_val = bc11;
  unsigned next_reg = bc9;
  unsigned next_off = of811;
  if ((bc5 & short_mask) <= search_val)
  {
    ix_result = 5;
    lookup_val = bc5;
    next_reg = bc3;
    offset = (of0 >> 8) & 0xff;
    next_off = of25;
  }
  basecode = lookup_val & 0xffff; // Currently either bc5 or bc11

  const unsigned new_lookup_val = ix_result == 5 ? bc2 : bc8;
  if ((new_lookup_val & short_mask) <= search_val)
  {
    ix_result -= 3;
    lookup_val = new_lookup_val;
    next_reg = ix_result == 2 ? 4096 : bc6; // bc0 is always maximum, can't select it
    int offset_shift_bits = ix_result == 2 ? 24 : 16;
    offset = (of0 >> offset_shift_bits) & 0xff;
    next_off >>= 16;
  }

  basecode = lookup_val & 0xffff;

  lookup_val >>= 16;

  if (lookup_val <= search_val)
  {
    basecode = lookup_val;
    ix_result -= 1;
    offset = next_off & 0xff;
  }

  if (next_reg <= search_val)
  {
    ix_result -= 1;
    basecode = next_reg;
    offset = (next_off >> 8) & 0xff;
  }

  return ix_result;
}

template <int N>
inline __device__ void lower_bound(int &ix_result, int search_val, int search_reg)
{
  ix_result = 0;
#pragma unroll
  for (int ix = N - 1; ix >= 0; --ix)
  {
    int this_search_val = __shfl_sync(WARP_ALL, search_reg, ix_result + (1 << ix));

    if (this_search_val <= search_val)
    {
      ix_result += (1 << ix);
    }
  }
}

template <int N, typename data_t>
inline __device__ void
flexible_lower_bound(int &ix_result, data_t search_val, const data_t *search_array, int array_size)
{
  // Adapted from binary_search_leftmost algorithm from here:
  // https://en.wikipedia.org/wiki/Binary_search_algorithm
  int &left = ix_result;
  left = 0;
  int right = array_size;

#pragma unroll
  for (int iter = 0; iter < N; ++iter)
  {
    int ix_mid = (left + right) / 2;

    if (search_array[ix_mid] <= search_val)
    {
      left = ix_mid;
    }
    else
    {
      right = ix_mid;
    }
  }
}

inline __device__ int do_per_thread_input_aligned_copy(const uint8_t *src, uint8_t *dst, int copy_length)
{
  if (copy_length > 0)
  {
    // Do a 4-byte copy
    // Some of the values here might be uninitialized, which flags initcheck
    char4 input = *reinterpret_cast<const char4 *>(src);
    dst[0] = input.x;

    if (copy_length > 1)
    {
      dst[1] = input.y;
    }
    if (copy_length > 2)
    {
      dst[2] = input.z;
    }
    if (copy_length > 3)
    {
      dst[3] = input.w;
    }
    return min(copy_length, 4);
  }
  else
  {
    return 0;
  }
}

inline __device__ void warp_input_4byte_aligned_copy(const uint8_t *src, uint8_t *dst, int copy_length)
{
  // Then we can finish the copy, doing reading 4 bytes at a time
  for (int ix_base = 0; ix_base < copy_length; ix_base += WARP_SIZE * 4)
  {
    const int thread_ix = ix_base + 4 * thread_warp_ix();
    int bytes_remaining = copy_length - thread_ix;
    do_per_thread_input_aligned_copy(&src[thread_ix], &dst[thread_ix], bytes_remaining);
  }
}

inline __device__ void warp_input_nonaligned_copy(const uint8_t *src, uint8_t *dst, int copy_length)
{
  for (int i = thread_warp_ix(); i < copy_length; i += WARP_SIZE)
  {
    dst[i] = src[i];
  }
}

struct MemoryManager
{
  uint8_t *ptr;
  int offset;
  uintptr_t share_val;
  int max_offset;

  __device__ MemoryManager(uint8_t *ptr_in, int max_offset)
      : ptr(ptr_in + sizeof(size_t))
      , offset(0)
      , share_val(uintptr_t(ptr_in))
      , max_offset(max_offset - sizeof(size_t)) // we steal the first 8 bytes for a shared value
  {}

  MemoryManager() = default;

  template <typename T = uint8_t, int align_req = alignof(T)>
  inline __device__ T *allocate(int num_values)
  {
    static_assert(align_req <= 8, "This is only valid for <= 8 byte alignment");
    int new_loc = 0; // Default value for Coverity. Real value will be computed by 0th thread.
    if (thread_warp_ix() == 0)
    {
      int req_vals = sizeof(T) * num_values + align_req - 1;
      new_loc = atomicAdd(&offset, req_vals);
      if (align_req != 1)
      {
        // Check whether alignment is required
        uint8_t align_val = (uintptr_t)(ptr + new_loc) % align_req;
        if (align_val > 0)
        {
          new_loc += align_req - align_val;
        }
        assert((uintptr_t)(ptr + new_loc) % align_req == 0);
      }
      assert(offset <= max_offset);
    }

    __syncwarp(WARP_ALL); // This is a suppression for now, will remove since it's redundant with the below
    new_loc = __shfl_sync(WARP_ALL, new_loc, 0);
    return reinterpret_cast<T *>(ptr + new_loc);
  }

  template <typename T>
  void __device__ free(T *)
  {} // no op
};

// Returns the index (0-indexed) of the highest set bit
// If no bit is set in `val`, it returns 255
inline __device__ uint8_t highest_set_bit(unsigned val)
{
  // Get the largest power of 2 <= val
  return 31 - __clz(val);
}

inline __device__ size_t parse_frame_header(const uint8_t *&frame_loc)
{
  assert(frame_loc[0] == 0x28 && frame_loc[1] == 0xB5 && frame_loc[2] == 0x2F && frame_loc[3] == 0xFD);
  frame_loc += 4; // Proceed past magic number size

  // Need to go through and read some stuff about the frame.
  uint8_t frame_header = *(frame_loc++);
  uint8_t single_segment_flag = frame_header & (1 << 5);
  uint8_t fcs_flag = frame_header >> 6;
  uint8_t fcs_size = 1 << fcs_flag;
  if (not single_segment_flag and not fcs_flag)
  {
    fcs_size = 0;
  }
  assert(fcs_size <= 8);

#ifndef NDEBUG
  uint8_t dictionary_flag = frame_header & 0x3;
  assert(dictionary_flag == 0);

  uint8_t reserved = frame_header & (1 << 3);
  assert(reserved == 0);
#endif

  bool read_window_descriptor = not single_segment_flag;
  if (read_window_descriptor)
  {
    frame_loc++;
  }

  size_t uncomp_frame_size = 0;
  if (fcs_size == 1)
  {
    uncomp_frame_size = *(frame_loc++);
  }
  else if (fcs_size == 2)
  {
    uncomp_frame_size = 256;
    uncomp_frame_size += *(frame_loc++);
    uncomp_frame_size += (*(frame_loc++)) << 8;
  }
  else
  {
    for (int ix = 0; ix < fcs_size; ++ix)
    {
      uncomp_frame_size += static_cast<size_t>(*(frame_loc++)) << (8 * ix);
    }
  }
  return uncomp_frame_size;
}

struct TmpBufferManager
{
  uint64_t *tmp_buffer_loc; // Pointer to an 8-byte integer, indicating the currently used bytes out of tmp_buffer_size
  uint8_t *tmp_buffer; // Buffer's base location
  size_t tmp_buffer_size; // Total size of the buffer usable as scratch pad [bytes]

  __device__ TmpBufferManager(uint64_t *tmp_buffer_loc, uint8_t *tmp_buffer, size_t tmp_buffer_size)
      : tmp_buffer_loc(tmp_buffer_loc)
      , tmp_buffer(tmp_buffer)
      , tmp_buffer_size(tmp_buffer_size)
  {}

  __device__ uint8_t *allocate(size_t alloc_size)
  {
    size_t this_loc = atomicAddUint64Wrapper(tmp_buffer_loc, alloc_size);
    size_t this_loc_end = this_loc + alloc_size;
    if (this_loc_end > tmp_buffer_size)
    {
#ifdef IS_ZSTD_DECOMP
      printf("nvcomp ZSTD scratch buffer allocation exceeded. please contact NVIDIA.\n");
#endif // IS_ZSTD_DECOMP
#ifndef NDEBUG
      printf("this_loc %llu tmp loc %llu alloc %llu\n", this_loc, this_loc_end, alloc_size);
#endif // NDEBUG
      // Note: executing `trap;` generates an "unspecified launch failure" CUDA error
      // TODO(bnagy): replace with device_status = nvcompErrorCannotCompress, and investigate the performance impact
      asm("trap;");
    }
    assert(this_loc_end <= tmp_buffer_size);
    return tmp_buffer + this_loc;
  }

  template <typename T = uint8_t, size_t align_req = alignof(T)>
  __device__ T *allocate(size_t alloc_size)
  {
    size_t alloc_total_size = alloc_size * sizeof(T) + align_req - 1;
    size_t this_loc = atomicAddUint64Wrapper(tmp_buffer_loc, alloc_total_size);
    size_t this_loc_end = this_loc + alloc_total_size;
    if (this_loc_end > tmp_buffer_size)
    {
#ifdef IS_ZSTD_DECOMP
      printf("nvcomp ZSTD scratch buffer allocation exceeded. please contact NVIDIA.\n");
#endif // IS_ZSTD_DECOMP
#ifndef NDEBUG
      printf("this_loc %llu tmp loc %llu alloc %llu\n", this_loc, this_loc_end, alloc_total_size);
#endif // NDEBUG
      // Note: executing `trap;` generates an "unspecified launch failure" CUDA error
      // TODO(bnagy): replace with device_status = nvcompErrorCannotCompress, and investigate the performance impact
      asm("trap;");
    }

    uint8_t align_val = (uintptr_t)(tmp_buffer + this_loc) % align_req;
    assert(this_loc_end <= tmp_buffer_size);
    uint8_t align_incr = align_val > 0 ? align_req - align_val : 0;
    return reinterpret_cast<T *>(tmp_buffer + this_loc + align_incr);
  }
};

// TODO: add usage of intrinsics here (reduce_max_sync/reduce_add_sync)
inline __device__ int warpReduceMax(unsigned int val, unsigned int am)
{
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
  {
    val = max(val, __shfl_down_sync(am, val, offset));
  }
  val = __shfl_sync(am, val, 0);
  return val;
}

template <typename T>
inline __device__ T warpReduceSum(T val, unsigned int am)
{
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
  {
    val += __shfl_down_sync(am, val, offset);
  }
  val = __shfl_sync(am, val, 0);
  return val;
}

// warpMatchAdd adds 1 to counts at the "val" index
// for every val in the warp. If, for example, val 0 appears 5 times in the warp,
// the global count for 0 increments by 5 in a single operation.
template <typename Match_t, typename Incr_t>
inline __device__ void warpMatchAdd(const Match_t val, const unsigned mask, Incr_t *const counts)
{
  int match_mask = __match_any_sync(mask, val);
  __syncwarp(mask); // This is redundant but silences racecheck
  if (__ffs(match_mask) - 1 == thread_warp_ix())
  {
    counts[val] += __popc(match_mask);
  }
}

template <typename T>
__device__ T reduce_min(T symbol)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return __reduce_min_sync(WARP_ALL, symbol);
#else
  for (uint32_t i = WARP_SIZE_U / 2; i > 0; i >>= 1)
  {
    T next_symbol = __shfl_down_sync(WARP_ALL, symbol, i, WARP_SIZE);
    symbol = min(symbol, next_symbol);
  }

  return __shfl_sync(WARP_ALL, symbol, 0);
#endif
}

inline __device__ void minValWithIndex(int &loc, int &min_val, const int val)
{
  min_val = reduce_min(val);
  // Find the first thread with this minimum value
  loc = __ffs(__ballot_sync(WARP_ALL, val == min_val)) - 1;
}

inline __device__ void minANSBlockIndex(int &loc, int &min_val, const int val)
{
  minValWithIndex(loc, min_val, val);
  // Find the first block with minimum value.
  loc /= NUM_THREADS_PER_ANS_BLOCK;
}

inline int get_cuda_arch(int device_id = 0)
{
  int major, minor;
  CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id));
  CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id));
  return major * 100 + minor * 10;
}

inline int get_arch_id(int device_id = 0)
{
  int cuda_arch = get_cuda_arch(device_id);
  for (int i = 0; i < NUM_ARCH_IDS; i++)
  {
    if (cuda_arch >= CONFIG_ARCH[i])
    {
      return i;
    }
  }
  return NUM_ARCH_IDS - 1;
}

inline __host__ __device__ size_t compute_max_sequence_count(const size_t max_chunk_size)
{
  return max_chunk_size / ZSTD_MIN_MATCH_LENGTH;
}

// Custom wait to avoid atomic::wait() exponential backoff.
// This reduces the load associated with many warps performing this operation
template <typename atomic_t, typename T>
inline __device__ void wait_for_atomic(atomic_t &atomic_val, const T desired_val, T &cached_val, const int sleep_ns)
{
  if (cached_val < desired_val)
  {
    if (thread_warp_ix() == 0)
    {
      cached_val = atomic_val.load(cuda::std::memory_order_relaxed);

      while (cached_val < desired_val)
      {
        __nanosleep(sleep_ns);
        cached_val = atomic_val.load(cuda::std::memory_order_relaxed);
      }

      cuda::atomic_thread_fence(cuda::std::memory_order_acquire, cuda::thread_scope_device);
    }
    __syncwarp(); // Needed to commute fence across threads
    cached_val = __shfl_sync(WARP_ALL, cached_val, 0);
  }
}

} // namespace zstd
