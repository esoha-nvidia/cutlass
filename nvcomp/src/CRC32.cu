/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * This code is based on that written by Mauro Bisson <maurob@nvidia.com>
 *
 * These functions were either written by or lightly 
 * modified from those of Mauro Bisson's cuCRC32.
 *
 * gf2Poly32Multiply_d
 * crc32Shift_d
 * processChunk1ByteST_BL_d
 * processChunk_BL_d
 * crc32_Batched_BL_k
 * processChunk1ByteST_WP_d
 * processChunk_WP_d
 * crc32_Batched_WP_k_
 *
 * All other functions were written by Nico Iskos <niskos@nvidia.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "common.h"
#include "CRC32.hpp"
#include "CRC32Table.h"
#include "exception.hpp"
#include "nvcomp.hpp"
#include "nvcomp/shared_types.h"

#define WSIZE 32

#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

#define DIV_UP(a, b) (((a) + ((b) - 1)) / (b))

namespace nvcomp
{

constexpr int block_x = 32;
constexpr int block_y = 2;
constexpr int rbytes = 32;

template <int BYTES>
union __align__(BYTES) __byte_t
{
  unsigned int u[BYTES / sizeof(unsigned int)];
};

template <int BDIM_X, typename T>
__device__ __forceinline__ T __block_xor(T v)
{
  __shared__ T sh[BDIM_X / WSIZE];

  const int lid = threadIdx.x % WSIZE;
  const int wid = threadIdx.x / WSIZE;

#pragma unroll
  for (int i = WSIZE / 2; i; i >>= 1)
  {
    v ^= __shfl_down_sync(WARP_ALL, v, i);
  }
  if (lid == 0)
  {
    sh[wid] = v;
  }

  __syncthreads();
  if (wid == 0)
  {
    v = (lid < (BDIM_X / WSIZE)) ? sh[lid] : 0;

#pragma unroll
    for (int i = (BDIM_X / WSIZE) / 2; i; i >>= 1)
    {
      v ^= __shfl_down_sync(WARP_ALL, v, i);
    }
  }
  __syncthreads();
  return v;
}

template <int BDIM_X, typename T>
__device__ __forceinline__ T __block_xor(T v, T *sh)
{
  const int lid = threadIdx.x % WSIZE;
  const int wid = threadIdx.x / WSIZE;

#pragma unroll
  for (int i = WSIZE / 2; i; i >>= 1)
  {
    v ^= __shfl_down_sync(WARP_ALL, v, i);
  }
  if (lid == 0)
  {
    sh[wid] = v;
  }

  __syncthreads();
  if (wid == 0)
  {
    v = (lid < (BDIM_X / WSIZE)) ? sh[lid] : 0;

#pragma unroll
    for (int i = (BDIM_X / WSIZE) / 2; i; i >>= 1)
    {
      v ^= __shfl_down_sync(WARP_ALL, v, i);
    }
  }
  __syncthreads();
  return v;
}

__device__ unsigned int gf2Poly32Multiply_d(unsigned int x, unsigned int y, unsigned int mod)
{
  unsigned int prod = 0;

#pragma unroll
  for (int i = 0; i < 32; i++)
  {
    prod ^= (y & 1) ? x : 0;
    x = (x << 1) ^ (x & 0x80000000 ? mod : 0);
    y >>= 1;
  }
  return prod;
}

template <typename LEN_T>
__device__ unsigned int crc32Shift_d(unsigned int crc, LEN_T len)
{
  unsigned int power = POLY;

  crc = __brev(crc);

  for (LEN_T i = 0; i < 8 * (len & 3); i++)
  {
    crc = (crc << 1) ^ (crc & 0x80000000 ? POLY : 0);
  }

  len >>= 2;
  if (!len)
  {
    return __brev(crc);
  }

  while (1)
  {
    if (len & 1)
    {
      crc = gf2Poly32Multiply_d(crc, power, POLY);
    }
    len >>= 1;
    if (!len)
    {
      break;
    }
    power = gf2Poly32Multiply_d(power, power, POLY);
  }

  crc = __brev(crc);

  return crc;
}

__device__ unsigned int process_1_byte_single_checksum(
  size_t n,
  const size_t n0,
  unsigned int crc,
  const unsigned char *__restrict__ __ptrLDG,
  const unsigned int *__restrict__ __shCRC_t0,
  const uint8_t *restricted_crc_address
)
{
  crc = __brev(crc) ^ 0xFFFFFFFF;

  for (int i = 0; i < n; i++)
  {
    uint32_t target_word = __ptrLDG[i];
    if ((restricted_crc_address <= &__ptrLDG[i]) && (&__ptrLDG[i] < restricted_crc_address + sizeof(uint32_t)))
    {
      target_word = 0;
    }

    crc ^= __brev(target_word);
    crc = (crc << 8) ^ __shCRC_t0[crc >> 24];
  }
  crc = __brev(~crc);

  return n0 ? crc32Shift_d(crc, n0) : crc;
}

template <int BDIM_X, int NBYTE>
__device__ unsigned int process_bytes_single_checksum_permuted(
  size_t n,
  unsigned int crc,
  const __byte_t<NBYTE> *__restrict__ __ptrLDG,
  const unsigned int *__restrict__ __shCRC_t0,
  const unsigned int *__restrict__ __shCRC_t1,
  const unsigned int *__restrict__ __shCRC_t2,
  const unsigned int *__restrict__ __shCRC_t3,
  const uint32_t *restricted_crc_address
)
{
  const int tid = blockIdx.x * BDIM_X + threadIdx.x;
  const size_t NUM_THREADS = BDIM_X * gridDim.x;

  const size_t nread = n / NBYTE;

  for (size_t i = 0; i < nread; i += BDIM_X * gridDim.x)
  {
    if (i + tid < nread)
    {

      crc = ~__brev(crc);

#pragma unroll
      for (int j = 0; j < NBYTE / 4; j++)
      {
        const uint32_t *target_word_ptr = reinterpret_cast<const uint32_t *>(&__ptrLDG[i + tid]) + j;
        uint32_t target_word = *target_word_ptr;
        if (target_word_ptr == restricted_crc_address)
        {
          target_word = 0;
        }

        crc ^= __brev(target_word);
        crc = (__shCRC_t3[(crc >> 24) & 0xFF]) ^ (__shCRC_t2[(crc >> 16) & 0xFF]) ^ (__shCRC_t1[(crc >> 8) & 0xFF]) ^
              (__shCRC_t0[(crc) & 0xFF]);
      }
      crc = ~__brev(crc);
    }
  }

  // after each thread has computed the running checksum of its contiguous chunks,
  // do shift to combine checksums of permuted interleaved streams
  const size_t n0 = (NUM_THREADS - 1 - tid) * (n / NUM_THREADS);

  crc = crc32Shift_d(crc, n0);

  return crc;
}

template <typename LEN_T>
__device__ unsigned int process_1_byte_chunk_checksums(
  const LEN_T n,
  unsigned int crc,
  const unsigned char *__ptrLDG,
  const unsigned int *__shCRC_t0
)
{
  crc = __brev(crc) ^ 0xFFFFFFFF;

  for (int i = 0; i < n; i++)
  {
    crc ^= __brev(__ptrLDG[i]);
    crc = (crc << 8) ^ __shCRC_t0[crc >> 24];
  }

  crc = __brev(~crc);
  return crc;
}

template <int BDIM_X, int NBYTE, typename LEN_T>
__device__ unsigned int process_bytes_chunk_checksums(
  const LEN_T n,
  unsigned int crc,
  const __byte_t<NBYTE> *__ptrLDG,
  const unsigned int *__shCRC_t0,
  const unsigned int *__shCRC_t1,
  const unsigned int *__shCRC_t2,
  const unsigned int *__shCRC_t3
)
{
  const int tidx = threadIdx.x;

  const int n0BytesFull = (BDIM_X - 1 - tidx) * sizeof(__byte_t<NBYTE>);
  const int n0BytesPart = ((n % BDIM_X) - 1 - tidx) * sizeof(__byte_t<NBYTE>);

  for (LEN_T i = 0; i < n; i += BDIM_X)
  {
    if (i + tidx < n)
    {

      const __byte_t<NBYTE> __tmp = __ptrLDG[i + tidx];

      crc = ~__brev(crc);

#pragma unroll
      for (int j = 0; j < NBYTE / 4; j++)
      {
        crc ^= __brev(__tmp.u[j]);
        crc = (__shCRC_t3[(crc >> 24) & 0xFF]) ^ (__shCRC_t2[(crc >> 16) & 0xFF]) ^ (__shCRC_t1[(crc >> 8) & 0xFF]) ^
              (__shCRC_t0[(crc) & 0xFF]);
      }
      crc = ~__brev(crc);

      const int n0 = (i + BDIM_X <= n) ? n0BytesFull : n0BytesPart;
      crc = crc32Shift_d(crc, n0);
    }

    if constexpr (BDIM_X == WSIZE)
    {
#pragma unroll
      for (int l = WSIZE / 2; l; l >>= 1)
      {
        crc ^= __shfl_down_sync(WARP_ALL, crc, l, WSIZE);
      }
    }
    else
    {
      crc = __block_xor<BDIM_X>(crc);
    }
    if (tidx)
    {
      crc = 0;
    }
  }

  return crc;
}

template <int BDIM_X, int NBYTE, typename LEN_T>
__device__ unsigned int process_bytes_chunk_checksums_permuted(
  const LEN_T n,
  unsigned int crc,
  const __byte_t<NBYTE> *__ptrLDG,
  const unsigned int *__shCRC_t0,
  const unsigned int *__shCRC_t1,
  const unsigned int *__shCRC_t2,
  const unsigned int *__shCRC_t3
)
{

  const int tidx = threadIdx.x;

  for (LEN_T i = 0; i < n; i += BDIM_X)
  {
    if (i + tidx < n)
    {
      const __byte_t<NBYTE> __tmp = __ptrLDG[i + tidx];

      crc = ~__brev(crc);

#pragma unroll
      for (int j = 0; j < NBYTE / 4; j++)
      {
        crc ^= __brev(__tmp.u[j]);
        crc = (__shCRC_t3[(crc >> 24) & 0xFF]) ^ (__shCRC_t2[(crc >> 16) & 0xFF]) ^ (__shCRC_t1[(crc >> 8) & 0xFF]) ^
              (__shCRC_t0[(crc) & 0xFF]);
      }
      crc = ~__brev(crc);
    }
  }

  // after each thread has computed the running checksum of its contiguous chunks,
  // do shift to combine checksums of permuted interleaved streams
  const int n0 = (BDIM_X - 1 - tidx) * NBYTE * (n / BDIM_X);
  crc = crc32Shift_d(crc, n0);

  if constexpr (BDIM_X == WSIZE)
  {
#pragma unroll
    for (int l = WSIZE / 2; l; l >>= 1)
    {
      crc ^= __shfl_down_sync(WARP_ALL, crc, l, WSIZE);
    }
  }
  else
  {
    crc = __block_xor<BDIM_X>(crc);
  }

  return crc;
}

// (BDIM_X == 32 && BDIM_Y == ANY) ||
// (BDIM_X  > 32 && BDIM_Y ==   1)
template <int BDIM_X, int BDIM_Y, int RDSIZE>
__device__ void non_interleaved_store_chunk_checksums(
  unsigned int *crcDst,
  size_t n,
  const unsigned char *m,
  nvcompStatus_t *status,
  bool active
)
{

  __shared__ unsigned int __shCRC_t0[256];
  __shared__ unsigned int __shCRC_t1[256];
  __shared__ unsigned int __shCRC_t2[256];
  __shared__ unsigned int __shCRC_t3[256];

  const int tidx = threadIdx.x;
  const int tid = threadIdx.y * BDIM_X + threadIdx.x;

#pragma unroll
  for (int i = 0; i < 256; i += BDIM_X * BDIM_Y)
  {
    if (i + tid < 256)
    {
      __shCRC_t0[i + tid] = crcTable0_d[i + tid];
      __shCRC_t1[i + tid] = crcTable1_d[i + tid];
      __shCRC_t2[i + tid] = crcTable2_d[i + tid];
      __shCRC_t3[i + tid] = crcTable3_d[i + tid];
    }
  }
  __syncthreads();

  if (!active)
  {
    return;
  }

  unsigned int crc = 0;

  const int misalign = reinterpret_cast<unsigned long long>(m) % RDSIZE;
  if (misalign)
  {
    // buffer can be misaligned and with a length less
    // than that necessary to hit the next aligned address
    const size_t nread = MIN(n, RDSIZE - misalign);

    if (!tidx)
    {
      crc = process_1_byte_chunk_checksums(nread, crc, m, __shCRC_t0);
    }

    n -= nread;
    m += nread;
  }

  if (n >= RDSIZE)
  {
    const size_t nread = n / RDSIZE;
    crc = process_bytes_chunk_checksums<BDIM_X, RDSIZE>(
      nread,
      crc,
      reinterpret_cast<const __byte_t<RDSIZE> *>(m),
      __shCRC_t0,
      __shCRC_t1,
      __shCRC_t2,
      __shCRC_t3
    );
    n -= nread * RDSIZE;
    m += nread * RDSIZE;
  }
  if (n)
  {
    if (!tidx)
    {
      crc = process_1_byte_chunk_checksums(n, crc, m, __shCRC_t0);
    }
  }
  if (!tidx && status)
  {
    if (*crcDst ^ crc)
    {
      *status = nvcompErrorBadChecksum;
    }
  }
  else if (!tidx)
  {
    *crcDst = crc;
  }
}

// (BDIM_X == 32 && BDIM_Y == ANY) ||
// (BDIM_X  > 32 && BDIM_Y ==   1)
template <int BDIM_X, int BDIM_Y, int RDSIZE>
__device__ uint32_t chunk_checksums(const unsigned int *crcDst, size_t n, const unsigned char *m, bool active)
{

  __shared__ unsigned int __shCRC_t0[256];
  __shared__ unsigned int __shCRC_t1[256];
  __shared__ unsigned int __shCRC_t2[256];
  __shared__ unsigned int __shCRC_t3[256];

  const int tidx = threadIdx.x;
  const int tid = threadIdx.y * BDIM_X + threadIdx.x;

#pragma unroll
  for (int i = 0; i < 256; i += BDIM_X * BDIM_Y)
  {
    if (i + tid < 256)
    {
      __shCRC_t0[i + tid] = crcTable0_d[i + tid];
      __shCRC_t1[i + tid] = crcTable1_d[i + tid];
      __shCRC_t2[i + tid] = crcTable2_d[i + tid];
      __shCRC_t3[i + tid] = crcTable3_d[i + tid];
    }
  }
  __syncthreads();

  if (!active)
  {
    return 0;
  }

  unsigned int crc = 0;

  const int misalign = reinterpret_cast<unsigned long long>(m) % RDSIZE;
  if (misalign)
  {
    // buffer can be misaligned and with a length less
    // than that necessary to hit the next aligned address
    const size_t nread = MIN(n, RDSIZE - misalign);

    if (!tidx)
    {
      crc = process_1_byte_chunk_checksums(nread, crc, m, __shCRC_t0);
    }

    n -= nread;
    m += nread;
  }
  if (n >= RDSIZE)
  {
    const size_t nread = n / RDSIZE;
    crc = process_bytes_chunk_checksums_permuted<BDIM_X, RDSIZE>(
      nread,
      crc,
      reinterpret_cast<const __byte_t<RDSIZE> *>(m),
      __shCRC_t0,
      __shCRC_t1,
      __shCRC_t2,
      __shCRC_t3
    );
    n -= nread * RDSIZE;
    m += nread * RDSIZE;
  }
  if (n)
  {
    if (!tidx)
    {
      crc = process_1_byte_chunk_checksums(n, crc, m, __shCRC_t0);
    }
  }

  return crc;
}

// (BDIM_X == 32 && BDIM_Y == ANY) ||
// (BDIM_X  > 32 && BDIM_Y ==   1)
template <int BDIM_X, int BDIM_Y, int RDSIZE>
__device__ void store_chunk_checksums(unsigned int *crcDst, size_t n, const unsigned char *m, bool active)
{
  uint32_t crc = chunk_checksums<BDIM_X, BDIM_Y, RDSIZE>(crcDst, n, m, active);

  if (!threadIdx.x && active)
  {
    *crcDst = crc;
  }

  return;
}

// (BDIM_X == 32 && BDIM_Y == ANY) ||
// (BDIM_X  > 32 && BDIM_Y ==   1)
template <int BDIM_X, int BDIM_Y, int RDSIZE>
__device__ void
verify_chunk_checksums(const unsigned int *crcDst, size_t n, const unsigned char *m, nvcompStatus_t *status, bool active)
{
  uint32_t crc = chunk_checksums<BDIM_X, BDIM_Y, RDSIZE>(crcDst, n, m, active);

  if (!threadIdx.x && active)
  {
    // verify chunk checksum
    if (*crcDst != crc)
    {
      *status = nvcompErrorBadChecksum;
    }
  }

  return;
}

template <int BDIM_X, int BDIM_Y, int RDSIZE>
__global__ void store_comp_chunk_checksums_kernel(
  const size_t num_chunks,
  const size_t *msgLen_d,
  const size_t *offsets,
  const uint8_t *comp_buffer,
  uint32_t *crcDst
)
{

  size_t i = blockIdx.x * size_t(BDIM_Y) + threadIdx.y;
  bool active = true;
  if (i >= num_chunks)
  {
    i = 0;
    active = false;
  }

  store_chunk_checksums<BDIM_X, BDIM_Y, RDSIZE>(crcDst + i, msgLen_d[i], comp_buffer + offsets[i], active);
}

template <int BDIM_X, int BDIM_Y, int RDSIZE>
__global__ void non_interleaved_decomp_chunk_checksums_kernel(
  const size_t num_chunks,
  const void *const *device_uncompressed_ptrs,
  const size_t *device_uncompressed_bytes,
  uint32_t *device_CRC32_ptrs
)
{

  size_t i = blockIdx.x * size_t(BDIM_Y) + threadIdx.y;
  bool active = true;
  if (i >= num_chunks)
  {
    i = 0;
    active = false;
  }

  non_interleaved_store_chunk_checksums<BDIM_X, BDIM_Y, RDSIZE>(
    device_CRC32_ptrs + i,
    device_uncompressed_bytes[i],
    reinterpret_cast<const unsigned char *>(device_uncompressed_ptrs[i]),
    nullptr,
    active
  );
}

template <int BDIM_X, int BDIM_Y, int RDSIZE>
__global__ void store_decomp_chunk_checksums_kernel(
  const size_t num_chunks,
  const size_t uncomp_block_size,
  const size_t last_uncomp_block_size,
  const uint8_t *decomp_buffer,
  uint32_t *crcDst
)
{
  size_t i = blockIdx.x * size_t(BDIM_Y) + threadIdx.y;
  bool active = true;
  if (i >= num_chunks)
  {
    i = 0;
    active = false;
  }

  store_chunk_checksums<BDIM_X, BDIM_Y, RDSIZE>(
    crcDst + i,
    i == num_chunks - 1 ? last_uncomp_block_size : uncomp_block_size,
    decomp_buffer + uncomp_block_size * i,
    active
  );
}

template <int BDIM_X, int BDIM_Y, int RDSIZE>
__global__ void verify_comp_chunk_checksums_kernel(
  const size_t num_chunks,
  const size_t *msgLen_d,
  const size_t *offsets,
  const uint8_t *comp_buffer,
  const uint32_t *crcDst,
  nvcompStatus_t *status
)
{

  size_t i = blockIdx.x * size_t(BDIM_Y) + threadIdx.y;
  bool active = true;
  if (i >= num_chunks)
  {
    i = 0;
    active = false;
  }

  verify_chunk_checksums<BDIM_X, BDIM_Y, RDSIZE>(crcDst + i, msgLen_d[i], comp_buffer + offsets[i], status, active);
}

template <int BDIM_X, int BDIM_Y, int RDSIZE>
__global__ void verify_decomp_chunk_checksums_kernel(
  const size_t num_chunks,
  const size_t uncomp_block_size,
  const size_t last_uncomp_block_size,
  const uint8_t *decomp_buffer,
  const uint32_t *crcDst,
  nvcompStatus_t *status
)
{
  size_t i = blockIdx.x * size_t(BDIM_Y) + threadIdx.y;
  bool active = true;
  if (i >= num_chunks)
  {
    i = 0;
    active = false;
  }

  verify_chunk_checksums<BDIM_X, BDIM_Y, RDSIZE>(
    crcDst + i,
    i == num_chunks - 1 ? last_uncomp_block_size : uncomp_block_size,
    decomp_buffer + uncomp_block_size * i,
    status,
    active
  );
}

template <int BDIM_X, int RDSIZE>
__global__ void store_or_verify_single_checksum_kernel(
  uint32_t *crc_dst,
  const uint32_t *restricted_crc_address,
  size_t n_bytes,
  const size_t *n_bytes_ptr,
  const unsigned char *m,
  uint32_t *atomic_ctr,
  nvcompStatus_t *status
)
{

  __shared__ unsigned int __shCRC_t0[256];
  __shared__ unsigned int __shCRC_t1[256];
  __shared__ unsigned int __shCRC_t2[256];
  __shared__ unsigned int __shCRC_t3[256];

  const int tidx = threadIdx.x;
  const int tid = blockIdx.x * BDIM_X + threadIdx.x;

  // size may be passed as value or pointer
  size_t n = n_bytes_ptr ? *n_bytes_ptr : n_bytes;

  if (!tid)
  {
    *crc_dst = 0;
  }

#pragma unroll
  for (int i = 0; i < 256; i += BDIM_X)
  {
    if (i + tidx < 256)
    {
      __shCRC_t0[i + tidx] = crcTable0_d[i + tidx];
      __shCRC_t1[i + tidx] = crcTable1_d[i + tidx];
      __shCRC_t2[i + tidx] = crcTable2_d[i + tidx];
      __shCRC_t3[i + tidx] = crcTable3_d[i + tidx];
    }
  }
  __syncthreads();

  unsigned int in_crc = 0;
  unsigned int out_crc = 0x0;

  const int misalign = reinterpret_cast<unsigned long long>(m) % RDSIZE;
  if (misalign)
  {
    // buffer can be misaligned and with a length less
    // than that necessary to hit the next aligned address
    const size_t nread = MIN(n, RDSIZE - misalign);

    if (!tid)
    {
      out_crc = process_1_byte_single_checksum(
        nread,
        n - nread,
        in_crc,
        m,
        __shCRC_t0,
        reinterpret_cast<const uint8_t *>(restricted_crc_address)
      );
    }
    n -= nread;
    m += nread;
  }
  if (n >= RDSIZE)
  {
    out_crc ^= process_bytes_single_checksum_permuted<BDIM_X, RDSIZE>(
      n,
      misalign ? 0 : in_crc,
      reinterpret_cast<const __byte_t<RDSIZE> *>(m),
      __shCRC_t0,
      __shCRC_t1,
      __shCRC_t2,
      __shCRC_t3,
      restricted_crc_address
    );
    const size_t nread = (n / RDSIZE) * RDSIZE;
    n -= nread;
    m += nread;
  }
  if (n && !tid)
  {
    out_crc ^= process_1_byte_single_checksum(
      n,
      size_t(0),
      0,
      m,
      __shCRC_t0,
      reinterpret_cast<const uint8_t *>(restricted_crc_address)
    );
  }

#pragma unroll
  for (int l = WSIZE / 2; l; l >>= 1)
  {
    out_crc ^= __shfl_down_sync(WARP_ALL, out_crc, l, WSIZE);
  }
  if ((tidx % WSIZE) == 0)
  {
    atomicXor(crc_dst, out_crc);
    if (status && (atomicAdd(atomic_ctr, 1) + 1 == gridDim.x))
    {
      if (*crc_dst != *restricted_crc_address)
      {
        *status = nvcompErrorBadChecksum;
      }
    }
  }
  return;
}

void compute_uncomp_chunk_checksums(
  size_t batch_size,
  const void *const *device_uncompressed_ptrs,
  const size_t *device_uncompressed_bytes,
  uint32_t *device_CRC32_ptrs,
  cudaStream_t stream
)
{

  const dim3 block(block_x, block_y);
  const dim3 grid(nvcomp::cuda_dim_cast(DIV_UP(batch_size, block_y)));

  non_interleaved_decomp_chunk_checksums_kernel<block_x, block_y, rbytes>
    <<<grid, block, 0, stream>>>(batch_size, device_uncompressed_ptrs, device_uncompressed_bytes, device_CRC32_ptrs);
  CUDA_CHECK(cudaGetLastError());
}

std::vector<Checksum_t>
compute_uncomp_chunk_checksums(const std::vector<std::vector<uint8_t>> &uncompressed_chunks, cudaStream_t stream)
{

  const size_t num_chunks = uncompressed_chunks.size();
  std::vector<void *> h_uncomp_chunks(num_chunks);
  for (size_t i = 0; i < num_chunks; i++)
  {
    void *d_uncomp_chunk;
    CUDA_CHECK(cudaMallocAsync(&d_uncomp_chunk, uncompressed_chunks[i].size(), stream));
    h_uncomp_chunks[i] = d_uncomp_chunk;

    CUDA_CHECK(cudaMemcpyAsync(
      d_uncomp_chunk,
      uncompressed_chunks[i].data(),
      uncompressed_chunks[i].size(),
      cudaMemcpyHostToDevice,
      stream
    ));
  }

  void **d_uncomp_chunks;
  CUDA_CHECK(cudaMallocAsync(&d_uncomp_chunks, num_chunks * sizeof(void *), stream));
  CUDA_CHECK(
    cudaMemcpyAsync(d_uncomp_chunks, h_uncomp_chunks.data(), num_chunks * sizeof(void *), cudaMemcpyHostToDevice, stream)
  );

  size_t *d_uncomp_sizes;
  CUDA_CHECK(cudaMallocAsync(&d_uncomp_sizes, num_chunks * sizeof(size_t), stream));
  std::vector<size_t> h_uncomp_sizes(num_chunks);
  for (size_t i = 0; i < num_chunks; i++)
  {
    h_uncomp_sizes[i] = uncompressed_chunks[i].size();
  }
  CUDA_CHECK(
    cudaMemcpyAsync(d_uncomp_sizes, h_uncomp_sizes.data(), num_chunks * sizeof(size_t), cudaMemcpyHostToDevice, stream)
  );

  Checksum_t *d_checksums;
  CUDA_CHECK(cudaMallocAsync(&d_checksums, num_chunks * sizeof(Checksum_t), stream));
  compute_uncomp_chunk_checksums(
    num_chunks,
    reinterpret_cast<const void *const *>(d_uncomp_chunks),
    reinterpret_cast<const size_t *>(d_uncomp_sizes),
    d_checksums,
    stream
  );

  // Copy back to host
  std::vector<Checksum_t> h_checksums(num_chunks);
  CUDA_CHECK(
    cudaMemcpyAsync(h_checksums.data(), d_checksums, num_chunks * sizeof(Checksum_t), cudaMemcpyDeviceToHost, stream)
  );
  CUDA_CHECK(cudaFreeAsync(d_uncomp_chunks, stream));
  CUDA_CHECK(cudaFreeAsync(d_uncomp_sizes, stream));
  CUDA_CHECK(cudaFreeAsync(d_checksums, stream));
  for (size_t i = 0; i < num_chunks; i++)
  {
    CUDA_CHECK(cudaFreeAsync(h_uncomp_chunks[i], stream));
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  return h_checksums;
}

void store_decomp_chunk_checksums(
  size_t num_chunks,
  size_t decomp_chunk_size,
  size_t decomp_buffer_size,
  const uint8_t *decomp_buffer,
  uint32_t *decomp_chunk_checksums,
  CommonHeader *common_header,
  cudaStream_t stream
)
{
  const dim3 block(block_x, block_y);
  const dim3 grid(nvcomp::cuda_dim_cast(DIV_UP(num_chunks, block_y)));
  store_decomp_chunk_checksums_kernel<block_x, block_y, rbytes><<<grid, block, 0, stream>>>(
    num_chunks,
    decomp_chunk_size,
    decomp_buffer_size % decomp_chunk_size,
    decomp_buffer,
    decomp_chunk_checksums
  );
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemsetAsync(&common_header->include_per_chunk_decomp_buffer_checksums, 1, sizeof(bool), stream));
}

void store_comp_chunk_checksums(
  size_t num_chunks,
  const size_t *comp_chunk_sizes,
  const size_t *comp_chunk_offsets,
  const uint8_t *comp_buffer,
  uint32_t *comp_chunk_checksums,
  CommonHeader *common_header,
  cudaStream_t stream
)
{
  const dim3 block(block_x, block_y);
  const dim3 grid(nvcomp::cuda_dim_cast(DIV_UP(num_chunks, block_y)));
  store_comp_chunk_checksums_kernel<block_x, block_y, rbytes>
    <<<grid, block, 0, stream>>>(num_chunks, comp_chunk_sizes, comp_chunk_offsets, comp_buffer, comp_chunk_checksums);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemsetAsync(&common_header->include_per_chunk_comp_buffer_checksums, 1, sizeof(bool), stream));
}

void verify_decomp_chunk_checksums(
  size_t num_chunks,
  size_t decomp_chunk_size,
  size_t decomp_buffer_size,
  const uint8_t *decomp_buffer,
  const uint32_t *decomp_chunk_checksums,
  nvcompStatus_t *status,
  cudaStream_t stream
)
{
  const dim3 block(block_x, block_y);
  const dim3 grid(nvcomp::cuda_dim_cast(DIV_UP(num_chunks, block_y)));
  verify_decomp_chunk_checksums_kernel<block_x, block_y, rbytes><<<grid, block, 0, stream>>>(
    num_chunks,
    decomp_chunk_size,
    decomp_buffer_size % decomp_chunk_size,
    decomp_buffer,
    decomp_chunk_checksums,
    status
  );
  CUDA_CHECK(cudaGetLastError());
}

void verify_comp_chunk_checksums(
  size_t num_chunks,
  const size_t *comp_chunk_sizes,
  const size_t *comp_chunk_offsets,
  const uint8_t *comp_buffer,
  const uint32_t *comp_chunk_checksums,
  nvcompStatus_t *status,
  cudaStream_t stream
)
{
  const dim3 block(block_x, block_y);
  const dim3 grid(nvcomp::cuda_dim_cast(DIV_UP(num_chunks, block_y)));
  verify_comp_chunk_checksums_kernel<block_x, block_y, rbytes><<<grid, block, 0, stream>>>(
    num_chunks,
    comp_chunk_sizes,
    comp_chunk_offsets,
    comp_buffer,
    comp_chunk_checksums,
    status
  );
  CUDA_CHECK(cudaGetLastError());
}

/*
 * store_single_checksum may be used in several ways:
 *
 * 1. Computing and storing a checksum for a single decompressed file chunk
 * 2. Computing and storing a checksum for a single compressed file chunk
 * 3. Computing and storing a checksum for the compressed file header + chunk checksums
 *
 * In case 1, file_size is the size of the decompressed chunk and file_size_ptr is unused.
 *
 * In case 2, file_size is the size of the decompressed chunk and file_size_ptr points to the 
 * size of the compressed chunk. The size of the decompressed chunk is needed to estimate the grid 
 * size for optimal performance.
 *
 * In case 3, file_size is the size of the compressed file header + chunk checksuns and 
 * file_size_ptr is unused.
 */
void store_single_checksum(
  const uint8_t *file,
  size_t file_size,
  const size_t *file_size_ptr,
  CommonHeader *common_header,
  uint32_t *crc_dst,
  bool is_chunk_level_checksum,
  cudaStream_t stream
)
{

  constexpr int bytes_per_warp_chunk = 1 << 18;
  constexpr int bytes_per_warp_file_level = 1 << 10;
  const size_t opt_perf_grid_x = is_chunk_level_checksum
                                   ? (file_size + bytes_per_warp_chunk - 1) / bytes_per_warp_chunk
                                   : (file_size + bytes_per_warp_file_level - 1) / bytes_per_warp_file_level;
  const size_t grid_x = (opt_perf_grid_x > 0) ? opt_perf_grid_x : 1;

  const dim3 block(block_x);
  const dim3 grid(nvcomp::cuda_dim_cast(grid_x));
  uint32_t *dummy_atomic_ctr = nullptr;
  nvcompStatus_t *dummy_status = nullptr;

  store_or_verify_single_checksum_kernel<block_x, rbytes>
    <<<grid, block, 0, stream>>>(crc_dst, crc_dst, file_size, file_size_ptr, file, dummy_atomic_ctr, dummy_status);
  CUDA_CHECK(cudaGetLastError());

  if (file_size_ptr && is_chunk_level_checksum)
  {
    // must be a single compressed chunk if file size was given as a pointer
    CUDA_CHECK(cudaMemsetAsync(&common_header->include_per_chunk_comp_buffer_checksums, 1, sizeof(bool), stream));
  }
  else if (is_chunk_level_checksum)
  {
    CUDA_CHECK(cudaMemsetAsync(&common_header->include_per_chunk_decomp_buffer_checksums, 1, sizeof(bool), stream));
  }
}

/*
 * store_single_checksum may be used in several ways:
 *
 * 1. Verifying a checksum for a single decompressed file chunk
 * 2. Verifying a checksum for a single compressed file chunk
 * 3. Verifying a checksum for the compressed file header + chunk checksums
 *
 * In case 1, file_size is the size of the decompressed chunk and file_size_ptr is unused.
 *
 * In case 2, file_size is the size of the decompressed chunk and file_size_ptr points to the 
 * size of the compressed chunk. The size of the decompressed chunk is needed to estimate the grid 
 * size for optimal performance.
 *
 * In case 3, file_size is the size of the compressed file header + chunk checksuns and 
 * file_size_ptr is unused.
 */
void verify_single_checksum(
  const uint8_t *file,
  size_t file_size,
  const size_t *file_size_ptr,
  uint8_t *scratch_buffer,
  nvcompStatus_t *status,
  const uint32_t *restricted_crc_address,
  bool is_chunk_level_checksum,
  cudaStream_t stream
)
{

  // calculate and verify checksum. scratch_buffer may be under-aligned (a custom
  // scratch allocator can hand back any base), so align it to uint32_t before
  // using it as a CRC temporary to avoid a misaligned 4-byte device store.
  uint32_t *crc_dst = roundUpToAlignment<uint32_t>(scratch_buffer);
  uint32_t *atomic_ctr = crc_dst + 1;
  CUDA_CHECK(cudaMemsetAsync(atomic_ctr, 0, sizeof(uint32_t), stream));

  constexpr int bytes_per_warp_chunk = 1 << 18;
  constexpr int bytes_per_warp_file_level = 1 << 10;
  const size_t opt_perf_grid_x = is_chunk_level_checksum
                                   ? (file_size + bytes_per_warp_chunk - 1) / bytes_per_warp_chunk
                                   : (file_size + bytes_per_warp_file_level - 1) / bytes_per_warp_file_level;
  const size_t grid_x = (opt_perf_grid_x > 0) ? opt_perf_grid_x : 1;

  const dim3 block(block_x);
  const dim3 grid(nvcomp::cuda_dim_cast(grid_x));
  store_or_verify_single_checksum_kernel<block_x, rbytes>
    <<<grid, block, 0, stream>>>(crc_dst, restricted_crc_address, file_size, file_size_ptr, file, atomic_ctr, status);
  CUDA_CHECK(cudaGetLastError());
}

void verify_all_checksums(
  const size_t *comp_chunk_offsets,
  const size_t *comp_chunk_sizes,
  const uint8_t *comp_data_buffer,
  const uint8_t *decomp_buffer,
  size_t uncomp_chunk_size,
  const uint32_t *comp_chunk_checksums,
  const uint32_t *decomp_chunk_checksums,
  uint8_t *scratch_buffer,
  const CommonHeader *common_header,
  const DecompressionConfig &config,
  nvcompStatus_t *status,
  cudaStream_t stream
)
{

  size_t *dummy_file_size_ptr = nullptr;

  // Verify chunk checksums
  if (comp_chunk_offsets)
  { // for compressors using BatchManager
    verify_comp_chunk_checksums(
      config.num_chunks,
      comp_chunk_sizes,
      comp_chunk_offsets,
      comp_data_buffer,
      comp_chunk_checksums,
      status,
      stream
    );

    verify_decomp_chunk_checksums(
      config.num_chunks,
      uncomp_chunk_size,
      config.decomp_data_size,
      decomp_buffer,
      decomp_chunk_checksums,
      status,
      stream
    );
  }
  else
  { // for bitcomp
    bool is_chunk_level_checksum = true;
    verify_single_checksum(
      comp_data_buffer,
      config.decomp_data_size,
      &common_header->comp_data_size,
      scratch_buffer,
      status,
      comp_chunk_checksums,
      is_chunk_level_checksum,
      stream
    );

    verify_single_checksum(
      decomp_buffer,
      config.decomp_data_size,
      dummy_file_size_ptr,
      scratch_buffer,
      status,
      decomp_chunk_checksums,
      is_chunk_level_checksum,
      stream
    );
  }

  // Verify file-level checksum
  const uint8_t *full_checksum_buf = reinterpret_cast<const uint8_t *>(common_header);
  size_t full_file_checksum_size = reinterpret_cast<const uint8_t *>(decomp_chunk_checksums + config.num_chunks) -
                                   full_checksum_buf;

  bool is_chunk_level_checksum = false;
  verify_single_checksum(
    full_checksum_buf,
    full_file_checksum_size,
    dummy_file_size_ptr,
    scratch_buffer,
    status,
    &common_header->decomp_buffer_checksum,
    is_chunk_level_checksum,
    stream
  );
}

void store_all_checksums(
  const size_t *comp_chunk_offsets,
  const size_t *comp_chunk_sizes,
  const uint8_t *comp_data_buffer,
  const uint8_t *decomp_buffer,
  size_t uncomp_chunk_size,
  uint32_t *comp_chunk_checksums,
  uint32_t *decomp_chunk_checksums,
  uint8_t *scratch_buffer,
  CommonHeader *common_header,
  const CompressionConfig &comp_config,
  cudaStream_t stream
)
{

  size_t *dummy_file_size_ptr = nullptr;

  // Verify chunk checksums
  if (comp_chunk_offsets)
  { // for compressors using BatchManager
    store_decomp_chunk_checksums(
      comp_config.num_chunks,
      uncomp_chunk_size,
      comp_config.uncompressed_buffer_size,
      decomp_buffer,
      decomp_chunk_checksums,
      common_header,
      stream
    );

    store_comp_chunk_checksums(
      comp_config.num_chunks,
      comp_chunk_sizes,
      comp_chunk_offsets,
      comp_data_buffer,
      comp_chunk_checksums,
      common_header,
      stream
    );
  }
  else
  { // for bitcomp
    bool is_chunk_level_checksum = true;
    store_single_checksum(
      decomp_buffer,
      comp_config.uncompressed_buffer_size,
      dummy_file_size_ptr,
      common_header,
      decomp_chunk_checksums,
      is_chunk_level_checksum,
      stream
    );

    store_single_checksum(
      comp_data_buffer,
      comp_config.uncompressed_buffer_size,
      &common_header->comp_data_size,
      common_header,
      comp_chunk_checksums,
      is_chunk_level_checksum,
      stream
    );
  }

  const uint8_t *full_checksum_buf = reinterpret_cast<const uint8_t *>(common_header);
  size_t full_file_checksum_size = reinterpret_cast<const uint8_t *>(decomp_chunk_checksums + comp_config.num_chunks) -
                                   full_checksum_buf;

  bool is_chunk_level_checksum = false;
  store_single_checksum(
    full_checksum_buf,
    full_file_checksum_size,
    dummy_file_size_ptr,
    common_header,
    &common_header->decomp_buffer_checksum,
    is_chunk_level_checksum,
    stream
  );
}

void cuCRC32_permuted_test(unsigned int num_chunks, uint32_t *crc_dst, uint32_t chunk_size, uint8_t *buf)
{

  dim3 block(block_x, block_y);
  dim3 grid(DIV_UP(num_chunks, block_y));
  store_decomp_chunk_checksums_kernel<block_x, block_y, rbytes>
    <<<grid, block, 0, 0>>>(num_chunks, chunk_size, chunk_size, buf, crc_dst);
  CUDA_CHECK(cudaGetLastError());
}
} // namespace nvcomp
