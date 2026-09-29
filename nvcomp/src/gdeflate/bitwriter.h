/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include <cuda.h>
#include <cuda/atomic>

#include <cassert>
#include <cstdint>

#include "common.h"
#include "gdeflate_constants.h"

namespace gdeflate
{

template <typename T>
inline __device__ void atomicFetchOr(T *ptr, T val)
{
  cuda::atomic_ref<T, cuda::thread_scope_device> atomic_val{ptr[0]};
  atomic_val.fetch_or(val, cuda::std::memory_order_relaxed);
}

// Class to cooperatively write out the standard bitstream
// This differs from warp_bitwriter in the write functionality.
// warp_bitwriter writes a swizzled huffman bitstream which
// makes it decoder parallelizable.
template <
  typename T = uint32_t, // Type of the input gdeflate stream
  typename Tb = uint64_t, // Type of the internal bitstream state in registers
  unsigned int N = WARP_SIZE_U> // Number of SIMD lanes
class warp_bitwriter_standard
{

  T *output_ptr_next;
  T *output_ptr;
  T *output_hwm;
  Tb buf;
  unsigned int cnt;

public:
  static constexpr unsigned int width = sizeof(T) * 8;

  // output_pos resides in shared memory, hance
  // cuda::thread_scope_block for the corresponding atomic variable.
  __device__ warp_bitwriter_standard(T *base)
      : output_ptr(base)
      , output_ptr_next(nullptr)
      , output_hwm(base + N)
      , buf{0}
      , cnt{0}
  {}

  __device__ T *get_output_ptr() { return output_ptr; }

  //one thread write header
  inline __device__ void standard_write_header(
    T bits,
    unsigned int n,
    cuda::atomic<size_t, cuda::thread_scope_block> &output_pos,
    bool active = true
  )
  {

    assert(n <= width);
    //Calcute the des position considering of Little-endian
    //Use memory_order_relaxed to load the output_pos because
    //order does not matter here, output_pos is atomically
    //updated by offset_by_bits + n, other threads should just
    //see the updated value.
    size_t offset_by_bits = output_pos.load(cuda::std::memory_order_relaxed);
    uint32_t offset = offset_by_bits / width;
    uint32_t end = (offset_by_bits + n) / width;
    uint8_t cur_cnt = offset_by_bits % width;
    // Note: In practice this is always true as the function is called with n = 3 or n = 14, and being the header, output_pos is small.
    uint8_t next_cnt = end == offset ? width : (width - cur_cnt);

    T cur_buf = 0;
    T next_buf = 0;

    //feed
    if (active)
    {
      cur_buf |= (T)(bits & mask<T>(n)) << cur_cnt;

      // Note, always false, since next_cnt == width
      if (next_cnt < width)
      {
        next_buf |= ((T)(bits & mask<T>(n)) >> next_cnt);
      }
    }

    //flush
    __syncwarp();
    if (active)
    {
      *(output_ptr + offset) |= cur_buf;
      *(output_ptr + offset + 1) |= next_buf;
      output_pos.store(offset_by_bits + n, cuda::memory_order_relaxed);
    }
    __syncwarp();
  }

  //TODO: CUB BlockScan limits this implementation to 1 warp per CTA
  inline __device__ void standard_write(
    T bits,
    unsigned int n,
    cuda::atomic<size_t, cuda::thread_scope_block> &output_pos,
    bool active = true,
    bool check = false
  )
  {
    assert(n <= width);
    //prefix sum to get the output stream offset index for every thread.
    typedef nvcomp::cub::WarpScan<size_t> WarpScan;
    __shared__ typename WarpScan::TempStorage temp_storage;
    size_t offset_by_bits = active ? n : 0;
    WarpScan(temp_storage).ExclusiveSum(offset_by_bits, offset_by_bits);

    //Calcute the des position considering of Little-endian
    offset_by_bits += output_pos.load(cuda::std::memory_order_relaxed);
    uint32_t offset = offset_by_bits / width;
    uint32_t end = (offset_by_bits + n) / width;
    uint8_t cur_cnt = offset_by_bits % width;
    uint8_t next_cnt = end == offset ? width : (width - cur_cnt);

    T cur_buf = 0;
    T next_buf = 0;

    //feed
    if (active)
    {
      cur_buf = (T)(bits & mask<T>(n)) << cur_cnt;
      if (next_cnt < width)
      {
        next_buf = (T)(bits & mask<T>(n)) >> next_cnt;
      }
    }

    //flush
    __syncwarp();
    if (active)
    {
      atomicFetchOr(output_ptr + offset, cur_buf);
      if (next_buf != 0)
      {
        atomicFetchOr(output_ptr + offset + 1, next_buf);
      }
    }
    if (threadIdx.x == WARP_SIZE_U - 1u)
    {
      output_pos.store(active ? offset_by_bits + n : offset_by_bits, cuda::std::memory_order_relaxed);
    }
    __syncwarp();
  }

  //TODO: CUB BlockScan limits this implementation to 1 warp per CTA
  inline __device__ void standard_write(
    Tb bits,
    unsigned int n,
    cuda::atomic<size_t, cuda::thread_scope_block> &output_pos,
    bool active = true,
    bool check = false
  )
  {
    uint8_t Tb_width = sizeof(uint64_t) * 8;
    assert(n <= Tb_width);

    //prefix sum to get the output stream offset index for every thread.
    typedef nvcomp::cub::WarpScan<size_t> WarpScan;
    __shared__ typename WarpScan::TempStorage temp_storage;
    size_t offset_by_bits = active ? n : 0;
    WarpScan(temp_storage).ExclusiveSum(offset_by_bits, offset_by_bits);

    //Calcute the des position considering of Little-endian
    //Use memory_order_relaxed to load and store updated
    //output_pos since we only need to preserve atomicity
    //Morever output_pos only gets updated by offset_bits + n
    //which is not changed by other threads.
    offset_by_bits += output_pos.load(cuda::std::memory_order_relaxed);
    uint32_t offset = offset_by_bits / Tb_width;
    uint32_t end = (offset_by_bits + n) / Tb_width;
    uint8_t cur_cnt = offset_by_bits % Tb_width;
    uint8_t next_cnt = end == offset ? Tb_width : (Tb_width - cur_cnt);

    Tb cur_buf = 0;
    Tb next_buf = 0;

    //feed
    if (active)
    {
      cur_buf = (Tb)(bits & mask<Tb>(n)) << cur_cnt;
      if (next_cnt < Tb_width)
      {
        next_buf = (Tb)(bits & mask<Tb>(n)) >> next_cnt;
      }
    }

    //flush
    __syncwarp();
    if (active)
    {
      atomicFetchOr(reinterpret_cast<unsigned long long *>(output_ptr) + offset, (unsigned long long)(cur_buf));
      if (next_buf != 0)
      {
        atomicFetchOr(reinterpret_cast<unsigned long long *>(output_ptr) + offset + 1, (unsigned long long)(next_buf));
      }
    }
    if (threadIdx.x == WARP_SIZE_U - 1u)
    {
      output_pos.store(active ? offset_by_bits + n : offset_by_bits, cuda::std::memory_order_relaxed);
    }
    // syncwarp is needed here because otherwise some threads which are not active or not indexed 31,
    // might read an old value of output_pos when standard_write is called again.
    __syncwarp();
  }
};

// Class to cooperatively write out the swizzled bitstream
template <
  typename T = uint32_t, // Type of the input gdeflate stream
  typename Tb = uint64_t, // Type of the internal bitstream state in registers
  unsigned int N = WARP_SIZE_U> // Number of SIMD lanes
class warp_bitwriter
{

  T *output_ptr_next;
  T *output_ptr;
  T *output_hwm;
  Tb buf;
  unsigned int cnt;

  inline __device__ void flush(bool active = true)
  {
    __syncwarp();
    active = active && (cnt >= width);
    if (active)
    {
      *output_ptr = (T)(buf);
      buf >>= width;
      cnt -= width;
      // Advance output_ptr
      output_ptr = output_ptr_next;
      output_ptr_next = nullptr;
    }
    __syncwarp();
    // Reserve next slot if you have used at least 1 bit in current slot
    bool reserve = (output_ptr_next == nullptr) && (cnt > 0);
    unsigned int ballot = __ballot_sync(WARP_ALL, reserve);
    unsigned int offset = __popc(ballot & ltMask()) - 1;
    if (reserve)
    {
      output_ptr_next = output_hwm + offset;
    }
    // Advance the output_hwm pointer for all threads
    output_hwm += __popc(ballot);
  }

public:
  static constexpr unsigned int width = sizeof(T) * 8;

  __device__ warp_bitwriter(T *base)
      : output_ptr(base)
      , output_ptr_next(nullptr)
      , output_hwm(base + N)
      , buf{0}
      , cnt{0}
  {
    output_ptr = base + threadIdx.x;
  }

  __device__ T *get_output_ptr() { return output_ptr; }

  inline __device__ void write(T bits, unsigned int n, bool active = true)
  {
    assert(n <= width);
    feed(bits, n, active);
  }

  inline __device__ void feed(T bits, unsigned int n, bool active = true)
  {
    assert(n <= width);
    if (active)
    {
      buf |= (Tb)(bits & mask<T>(n)) << cnt;
      cnt += n;
    }
    flush(active);
  }

  inline __device__ T *finalize()
  {
    // cnt should be < width here, so set to width to pad with zeros and then flush
    cnt = cnt > 0 ? width : 0;
    flush();
    assert(cnt == 0);

    // Return pointer to the end of stream
    // Note: instead of returning the minimum `output_ptr` in the warp, we return the precalculated
    //       `output_hwm` as it is aligned to a 128-byte boundary.
    return output_hwm;
  }
};

} // namespace gdeflate
