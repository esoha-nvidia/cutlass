/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * Written by Mauro Bisson <maurob@nvidia.com>
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

// TODO: Unify whitespace handling with the rest of the codebase. Notably:
// * Spaces instead of tabs for indentation.
// * Spaces around binary operators.

// TODO: Most template parameters are only ever instantiated with a single
// value, get rid of them.

// TODO: Directly passing a structure containing kernel kind, bytes per read,
// and blocks per message rather than that data encoded in an integer would
// allow removing lots of conversion-related code.

#include <algorithm>
#include <map>
#include <optional>
#include <string>

#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef _MSC_VER
#include <intrin.h>
#endif

#include <cuda_runtime.h>
#include <CudaConstants.h>

#include "cuCRC32.h"
#include "CudaUtils.h"
#include "device_guard.h"
#include "exception.hpp"

#include <nvcomp.hpp>
#include <nvcomp/utils.hpp>

namespace
{

using Ull = unsigned long long;
using nvcomp::DeviceGuard;

constexpr int WarpKernel = 0;
constexpr int BlockKernel = 1;

constexpr int ThreadsPerBlock = 64;

constexpr int ThreadsPerBlock_max_k = 1024;

constexpr int ConfBlocksPerMsgBits = 16;
constexpr int ConfBytesPerReadBits = 8;
constexpr int ConfKernelBits = 8;

constexpr unsigned int ConfBytesPerReadMask = (1u << ConfBytesPerReadBits) - 1;
constexpr unsigned int ConfKernelMask = (1u << ConfKernelBits) - 1;

int ceilLog2(unsigned int mask)
{
#ifdef _MSC_VER
  unsigned long index = 0;
  bool isZero = (_BitScanReverse(&index, mask - 1) == 0);
  return isZero ? 0 : index + 1;
#else // _MSC_VER
  return 8 * sizeof(unsigned int) - __builtin_clz(mask - 1);
#endif // _MSC_VER
}

unsigned int nextPow2(unsigned int x) { return 1u << ceilLog2(x); }

template <int Bytes>
union __align__(Bytes) __byte_t
{
  unsigned int u[Bytes / sizeof(unsigned int)];
};

template <int BlockDimX, typename T>
__device__ __forceinline__ T __block_max(T v)
{
  __shared__ T sh[BlockDimX / WARP_SIZE];

  const int lid = threadIdx.x % WARP_SIZE;
  const int wid = threadIdx.x / WARP_SIZE;

#pragma unroll
  for (int i = WARP_SIZE / 2; i; i >>= 1)
  {
    const T __t = __shfl_down_sync(WARP_ALL, v, i);
    v = max(v, __t);
  }
  if (lid == 0)
  {
    sh[wid] = v;
  }

  __syncthreads();
  if (wid == 0)
  {
    if constexpr (BlockDimX / WARP_SIZE > WARP_SIZE)
    {
      v = (lid < (BlockDimX / WARP_SIZE)) ? sh[lid] : 0;
    }
    else
    {
      v = sh[lid];
    }

#pragma unroll
    for (int i = (BlockDimX / WARP_SIZE) / 2; i; i >>= 1)
    {
      const T __t = __shfl_down_sync(WARP_ALL, v, i);
      v = max(v, __t);
    }
  }
  __syncthreads();
  return v;
}

template <int BlockDimX, typename LenT, typename ValT>
__global__ void max_k(const LenT n, const ValT *__restrict__ v, ValT *__restrict__ maxv)
{
  assert(gridDim.x == 1);

  const LenT tid = threadIdx.x;

  ValT mymax = 0;
  for (LenT i = 0; i < n; i += BlockDimX)
  {
    if (i + tid >= n)
    {
      break;
    }
    mymax = max(mymax, v[i + tid]);
  }

  mymax = __block_max<BlockDimX>(mymax);
  if (!tid)
  {
    *maxv = mymax;
  }
}

template <int BlockDimX, int BlockDimY>
__device__ void genTables_d(
  unsigned int poly,
  unsigned int *__restrict__ table0,
  unsigned int *__restrict__ table1,
  unsigned int *__restrict__ table2,
  unsigned int *__restrict__ table3
)
{
  const int tid = threadIdx.y * BlockDimX + threadIdx.x;

#pragma unroll
  for (int i = 0; i < 256; i += BlockDimX * BlockDimY)
  {

    unsigned int tbl_idx = i + tid;

    if (tbl_idx >= 256)
    {
      break;
    }

    unsigned int MSB = tbl_idx << 24;

    auto generateTableEntry = [&](unsigned int *table) {
      for (int j = 0; j < 8; j++)
      {
        MSB = (MSB << 1) ^ ((MSB & 0x80000000) ? poly : 0);
      }
      table[tbl_idx] = MSB;
    };

    generateTableEntry(table0);
    generateTableEntry(table1);
    generateTableEntry(table2);
    generateTableEntry(table3);
  }
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

template <typename LenT>
__device__ unsigned int crc32Shift_d(crcSpec_t spec, unsigned int crc, LenT len)
{
  unsigned int power = spec.poly;

  if (len == 0)
  {
    return crc;
  }

  crc ^= spec.init;

  for (LenT i = 0; i < 8 * (len & 3); i++)
  {
    crc = (crc << 1) ^ (crc & 0x80000000 ? spec.poly : 0);
  }

  len >>= 2;
  if (len == 0)
  {
    return crc;
  }

  while (true)
  {
    if (len & 1)
    {
      crc = gf2Poly32Multiply_d(crc, power, spec.poly);
    }
    len >>= 1;
    if (len == 0)
    {
      break;
    }
    power = gf2Poly32Multiply_d(power, power, spec.poly);
  }
  return crc;
}

template <typename LenT>
__device__ unsigned int processChunk1ByteST_WP_d(
  const crcSpec_t spec,
  const LenT n,
  unsigned int crc,
  const unsigned char *__restrict__ __ptrLDG,
  const unsigned int *__restrict__ __shCRC_t0
)
{
  if (spec.ref_in)
  {
    for (int i = 0; i < n; i++)
    {
      crc ^= __brev(__ptrLDG[i]);
      crc = (crc << 8) ^ __shCRC_t0[crc >> 24];
    }
  }
  else
  {
    for (int i = 0; i < n; i++)
    {
      crc ^= __ptrLDG[i] << 24;
      crc = (crc << 8) ^ __shCRC_t0[crc >> 24];
    }
  }
  return crc;
}

template <int BlockDimX, int NumBytes, bool RefIn, typename LenT>
__device__ unsigned int processChunk_WP_d(
  const crcSpec_t spec,
  const LenT n,
  unsigned int crc,
  const __byte_t<NumBytes> *__restrict__ __ptrLDG,
  const unsigned int *__restrict__ __shCRC_t0,
  const unsigned int *__restrict__ __shCRC_t1,
  const unsigned int *__restrict__ __shCRC_t2,
  const unsigned int *__restrict__ __shCRC_t3
)
{
  const int tidx = threadIdx.x;

  const int n0BytesFull = (BlockDimX - 1 - tidx) * sizeof(__byte_t<NumBytes>);
  const int n0BytesPart = ((n % BlockDimX) - 1 - tidx) * sizeof(__byte_t<NumBytes>);

  for (LenT i = 0; i < n; i += BlockDimX)
  {
    if (i + tidx < n)
    {
      const __byte_t<NumBytes> __tmp = __ptrLDG[i + tidx];

#pragma unroll
      for (int j = 0; j < NumBytes / 4; j++)
      {
        if constexpr (RefIn)
        {
          crc ^= __brev(__tmp.u[j]);
        }
        else
        {
          crc ^= __byte_perm(__tmp.u[j], 0, 0x0123);
        }
        crc = (__shCRC_t3[(crc >> 24) & 0xFF]) ^ (__shCRC_t2[(crc >> 16) & 0xFF]) ^ (__shCRC_t1[(crc >> 8) & 0xFF]) ^
              (__shCRC_t0[(crc) & 0xFF]);
      }

      const int n0 = (i + BlockDimX <= n) ? n0BytesFull : n0BytesPart;
      crc = crc32Shift_d(spec, crc, n0);
    }
    else
    {
      crc = 0;
    }

#pragma unroll
    for (int l = WARP_SIZE / 2; l; l >>= 1)
    {
      crc ^= __shfl_down_sync(WARP_ALL, crc, l, WARP_SIZE);
    }

    if (tidx)
    {
      crc = spec.init;
    }
  }
  return crc;
}

template <int BlockDimX, int BlockDimY, int BytesPerRead, typename NumT, typename LenT>
static __global__ __launch_bounds__(BlockDimX *BlockDimY) void crc32_Batched_WP_k(
  const __grid_constant__ crcSpec_t spec,
  const NumT nMsg,
  unsigned int *__restrict__ crcBuf,
  const LenT *__restrict__ msgLen_d,
  const unsigned char *const __restrict__ *__restrict__ msg
)
{
  __shared__ unsigned int __shCRC_t0[256];
  __shared__ unsigned int __shCRC_t1[256];
  __shared__ unsigned int __shCRC_t2[256];
  __shared__ unsigned int __shCRC_t3[256];

  const int tidx = threadIdx.x;

  genTables_d<BlockDimX, BlockDimY>(spec.poly, __shCRC_t0, __shCRC_t1, __shCRC_t2, __shCRC_t3);
  __syncthreads();

  const int mid = blockIdx.x * BlockDimY + threadIdx.y;
  if (mid >= nMsg)
  {
    return;
  }

  LenT n = msgLen_d[mid];
  if (n == 0)
  {
    return;
  }

  const unsigned char *__restrict__ m = msg[mid];

  unsigned int crc = spec.init;

  const int misalign = reinterpret_cast<Ull>(m) % BytesPerRead;
  if (misalign)
  {
    // buffer can be misaligned and with a length less
    // than that necessary to hit the next aligned address
    const LenT nread = min(n, LenT(BytesPerRead - misalign));

    if (!tidx)
    {
      crc = processChunk1ByteST_WP_d(spec, nread, crc, m, __shCRC_t0);
    }

    n -= nread;
    m += nread;
  }
  if (n >= BytesPerRead)
  {
    const LenT nread = n / BytesPerRead;
    if (spec.ref_in)
    {
      crc = processChunk_WP_d<BlockDimX, BytesPerRead, 1>(
        spec,
        nread,
        crc,
        reinterpret_cast<const __byte_t<BytesPerRead> *>(m),
        __shCRC_t0,
        __shCRC_t1,
        __shCRC_t2,
        __shCRC_t3
      );
    }
    else
    {
      crc = processChunk_WP_d<BlockDimX, BytesPerRead, 0>(
        spec,
        nread,
        crc,
        reinterpret_cast<const __byte_t<BytesPerRead> *>(m),
        __shCRC_t0,
        __shCRC_t1,
        __shCRC_t2,
        __shCRC_t3
      );
    }
    n -= nread * BytesPerRead;
    m += nread * BytesPerRead;
  }
  if (n && !tidx)
  {
    crc = processChunk1ByteST_WP_d(spec, n, crc, m, __shCRC_t0);
  }

  if (!tidx)
  {
    crcBuf[mid] ^= crc;
  }
}

template <typename LenT>
__device__ unsigned int processChunk1ByteST_BL_d(
  crcSpec_t spec,
  const LenT n,
  const LenT n0,
  const unsigned char *__restrict__ __ptrLDG,
  const unsigned int *__restrict__ __shCRC_t0
)
{
  unsigned int crc = spec.init;
  if (spec.ref_in)
  {
    for (int i = 0; i < n; i++)
    {
      crc ^= __brev(__ptrLDG[i]);
      crc = (crc << 8) ^ __shCRC_t0[crc >> 24];
    }
  }
  else
  {
    for (int i = 0; i < n; i++)
    {
      crc ^= __ptrLDG[i] << 24;
      crc = (crc << 8) ^ __shCRC_t0[crc >> 24];
    }
  }
  crc = n0 ? crc32Shift_d(spec, crc, n0) : crc;

  return crc;
}

template <int NumBytes, bool RefIn, typename LenT>
__device__ unsigned int processChunk_BL_d(
  crcSpec_t spec,
  const LenT n,
  const __byte_t<NumBytes> *__restrict__ __ptrLDG,
  const unsigned int *__restrict__ __shCRC_t0,
  const unsigned int *__restrict__ __shCRC_t1,
  const unsigned int *__restrict__ __shCRC_t2,
  const unsigned int *__restrict__ __shCRC_t3
)
{
  const int tid = blockIdx.y * blockDim.x + threadIdx.x;

  LenT n0Bytes = n - (tid + 1) * sizeof(__byte_t<NumBytes>);

  const LenT nread = n / NumBytes;

  unsigned int ret = 0;

  for (LenT i = 0; i < nread; i += blockDim.x * gridDim.y)
  {
    if (i + tid < nread)
    {
      unsigned int crc = spec.init;
      const __byte_t<NumBytes> __tmp = __ptrLDG[i + tid];

#pragma unroll
      for (int j = 0; j < NumBytes / 4; j++)
      {
        if constexpr (RefIn)
        {
          crc ^= __brev(__tmp.u[j]);
        }
        else
        {
          crc ^= __byte_perm(__tmp.u[j], 0, 0x0123);
        }
        crc = (__shCRC_t3[(crc >> 24) & 0xFF]) ^ (__shCRC_t2[(crc >> 16) & 0xFF]) ^ (__shCRC_t1[(crc >> 8) & 0xFF]) ^
              (__shCRC_t0[(crc) & 0xFF]);
      }

      ret ^= crc32Shift_d(spec, crc, n0Bytes);

      n0Bytes -= sizeof(__byte_t<NumBytes>) * blockDim.x * gridDim.y;
    }
  }
  return ret;
}

template <int BlockDimX, int BytesPerRead, typename NumT, typename LenT>
__global__ __launch_bounds__(BlockDimX) void crc32_Batched_BL_k(
  const crcSpec_t spec,
  const NumT nMsg,
  unsigned int *__restrict__ crcBuf,
  const LenT *__restrict__ msgLen_d,
  const unsigned char *const __restrict__ *__restrict__ msg
)
{
  __shared__ unsigned int __shCRC_t0[256];
  __shared__ unsigned int __shCRC_t1[256];
  __shared__ unsigned int __shCRC_t2[256];
  __shared__ unsigned int __shCRC_t3[256];

  const int tidx = threadIdx.x;
  const int tid = blockIdx.y * BlockDimX + threadIdx.x;

  const int mid = blockIdx.x;
  if (mid >= nMsg)
  {
    return;
  }

  LenT n = msgLen_d[mid];
  if (n == 0)
  {
    return;
  }

  genTables_d<BlockDimX, 1>(spec.poly, __shCRC_t0, __shCRC_t1, __shCRC_t2, __shCRC_t3);
  __syncthreads();

  const unsigned char *__restrict__ m = msg[mid];

  unsigned int out_crc = 0x0;

  const int misalign = reinterpret_cast<Ull>(m) % BytesPerRead;

  // the crc srcCrc[mid] (it can either be spec.init for the
  // first call or the partial crc produced by last call)
  // must be used as the state for only the first parallel
  // chunk (processed only by thread 0); thus it must be
  // used as the initial state by the first processChunk*()
  // call execute by thread 0

  if (misalign)
  {
    // buffer can be misaligned and with a length less
    // than that necessary to hit the next aligned address
    const LenT nread = min(n, LenT(BytesPerRead - misalign));

    if (!tid)
    {
      out_crc = processChunk1ByteST_BL_d(spec, nread, n - nread, m, __shCRC_t0);
    }
    n -= nread;
    m += nread;
  }
  if (n >= BytesPerRead)
  {
    // for all tid > 0 in_crc == spec.init;
    // for tid == 0:
    //        in_crc == crcSrc[mid] if buffer was not misaligned
    //        in_crc == spec.init otherwise
    if (spec.ref_in)
    {
      out_crc ^= processChunk_BL_d<BytesPerRead, 1>(
        spec,
        n,
        reinterpret_cast<const __byte_t<BytesPerRead> *>(m),
        __shCRC_t0,
        __shCRC_t1,
        __shCRC_t2,
        __shCRC_t3
      );
    }
    else
    {
      out_crc ^= processChunk_BL_d<BytesPerRead, 0>(
        spec,
        n,
        reinterpret_cast<const __byte_t<BytesPerRead> *>(m),
        __shCRC_t0,
        __shCRC_t1,
        __shCRC_t2,
        __shCRC_t3
      );
    }
    const LenT nread = (n / BytesPerRead) * BytesPerRead;
    n -= nread;
    m += nread;
  }
  if (n && !tid)
  {
    out_crc ^= processChunk1ByteST_BL_d(spec, n, LenT(0), m, __shCRC_t0);
  }
#pragma unroll
  for (int l = WARP_SIZE / 2; l; l >>= 1)
  {
    out_crc ^= __shfl_down_sync(WARP_ALL, out_crc, l, WARP_SIZE);
  }
  if ((tidx % WARP_SIZE) == 0)
  {
    atomicXor(crcBuf + mid, out_crc);
  }
}

template <typename NumT, typename LenT>
__global__ void
pre_shift_k(const crcSpec_t spec, const NumT n, const LenT *__restrict__ msgLen_d, unsigned int *__restrict__ crc)
{
  const NumT tid = NumT(blockIdx.x) * blockDim.x + threadIdx.x;

  if (tid >= n)
  {
    return;
  }

  if (crc[tid] == spec.init && msgLen_d[tid] != 0)
  {
    // Shortcut, always encountered on the first call to cuCRC32Add.
    crc[tid] = 0;
  }
  else
  {
    crc[tid] = crc32Shift_d(spec, crc[tid], msgLen_d[tid]);
  }
}

// TODO: Deduplicate w.r.t. (nearly) analogous functions in CudaUtils and python/.

// Check if the given device (and current driver) can use asynchronous memory
// allocations and deallocations
bool can_use_async_mem_ops(int device_id)
{
  int attribute_res_val = 0;
  cudaError_t result = cudaDeviceGetAttribute(&attribute_res_val, cudaDevAttrMemoryPoolsSupported, device_id);
  // Note: Certain attributes (e.g. cudaDevAttrMemoryPoolsSupported) might have been
  //       introduced in later CTK versions that are unknown by the current runtime
  //       environment, even if it compiles fine locally. In such cases, an invalid argument
  //       error is expected.
  if (result == cudaErrorInvalidValue)
  {
    cudaGetLastError(); // reset the cuda error (if any)
  }
  else
  {
    CUDA_CHECK(result);
  }
  if (result == cudaSuccess && attribute_res_val == 1)
  {
    return true;
  }
  return false;
}

// Construct map from device to whether it supports async memory operations.
std::map<int, bool> getAsyncAllocSupportMap()
{
  int deviceCount;
  CUDA_CHECK(::cudaGetDeviceCount(&deviceCount));

  std::map<int, bool> retMap{};
  for (int deviceIdx = 0; deviceIdx < deviceCount; ++deviceIdx)
  {
    retMap.insert({deviceIdx, can_use_async_mem_ops(deviceIdx)});
  }

  return retMap;
}

bool isAsyncAllocSupported(int device)
{
  static std::map<int, bool> asyncAllocSupportMap = getAsyncAllocSupportMap();
  return asyncAllocSupportMap.at(device);
}

// TODO: Introduce a RAII cudaMalloc wrapper, use instead of functional wrapper
// logic below.

cudaError_t cudaMallocWrapperImpl(void **ptr, size_t size, cudaStream_t stream)
{
  DeviceGuard device_guard(stream);
  int device = -1;
  CUDA_CHECK(cudaGetDevice(&device));

  if (isAsyncAllocSupported(device))
  {
    return cudaMallocAsync(ptr, size, stream);
  }
  else
  {
    return cudaMalloc(ptr, size);
  }
}

template <typename T>
cudaError_t cudaMallocWrapper(T **ptr, size_t size, cudaStream_t stream)
{
  return cudaMallocWrapperImpl(reinterpret_cast<void **>(ptr), size, stream);
}

cudaError_t cudaFreeWrapper(void *ptr, cudaStream_t stream)
{
  DeviceGuard device_guard(stream);
  int device = -1;
  CUDA_CHECK(cudaGetDevice(&device));

  if (isAsyncAllocSupported(device))
  {
    return cudaFreeAsync(ptr, stream);
  }
  else
  {
    return cudaFree(ptr);
  }
}

#define NVCOMP_CRC32_FOR_BYTES_PER_READ(MACRO, ...)                                                                    \
  MACRO(4, __VA_ARGS__)                                                                                                \
  MACRO(8, __VA_ARGS__)                                                                                                \
  MACRO(16, __VA_ARGS__)                                                                                               \
  MACRO(32, __VA_ARGS__)                                                                                               \
  MACRO(64, __VA_ARGS__)                                                                                               \
  MACRO(128, __VA_ARGS__)                                                                                              \
  MACRO(256, __VA_ARGS__)                                                                                              \
  MACRO(512, __VA_ARGS__)                                                                                              \
  MACRO(1024, __VA_ARGS__)                                                                                             \
  MACRO(2048, __VA_ARGS__)

} // namespace

int cuCRC32Add(
  const crcCtx_t *ctx,
  unsigned int conf,
  unsigned int nMsg,
  const Ull *msgLen_d,
  const unsigned char *const *msg_d,
  cudaStream_t stream
)
{
  // TODO: Remove error checks after unifying with nvcompBatchedCRC32Async.
  if (!ctx || !conf || !msgLen_d || !msg_d || nMsg < 1)
  {
    return CUCRC32_ERROR_INVALID_PARAMS;
  }

  // get context data
  unsigned int *crcBuf_d = ctx->crcBuf;

  int blocksPerMsg = 0;
  int bytesPerRead = 0;
  int kernel = 0;

  int rv = cuCRC32ConfToParam(conf, &kernel, &bytesPerRead, &blocksPerMsg);
  if (rv != CUCRC32_SUCCESS)
  {
    return rv;
  }
  if (kernel != WarpKernel && kernel != BlockKernel)
  {
    fprintf(stderr, "%s:%d: error, unknown kernel specified in conf (%d)\n", __func__, __LINE__, kernel);
    return CUCRC32_ERROR_INVALID_PARAMS;
  }

  pre_shift_k<<<nvcomp::roundUpDiv(nMsg, ThreadsPerBlock), ThreadsPerBlock, 0, stream>>>(
    ctx->spec,
    nMsg,
    msgLen_d,
    crcBuf_d
  );
  CUDA_CHECK(cudaGetLastError());

  if (kernel == WarpKernel)
  {
    dim3 block(WARP_SIZE, ThreadsPerBlock / WARP_SIZE);
    dim3 grid(nvcomp::roundUpDiv(nMsg, ThreadsPerBlock / WARP_SIZE));
#define NVCOMP_CRC32_INTERNAL_SWITCH_CASE_1(BPR, ...)                                                                  \
  case BPR:                                                                                                            \
    crc32_Batched_WP_k<WARP_SIZE, ThreadsPerBlock / WARP_SIZE, BPR>                                                    \
      <<<grid, block, 0, stream>>>(ctx->spec, nMsg, crcBuf_d, msgLen_d, msg_d);                                        \
    break;
    switch (bytesPerRead)
    {
      NVCOMP_CRC32_FOR_BYTES_PER_READ(NVCOMP_CRC32_INTERNAL_SWITCH_CASE_1);
      default:
        throw ::nvcomp::NVCompException(
          nvcompErrorInvalidValue,
          "read bytes for kernel 0 must be a power of 2 between 4 and 2048 (included), but is " +
            std::to_string(bytesPerRead)
        );
    }
    CUDA_CHECK(cudaGetLastError());
#undef NVCOMP_CRC32_INTERNAL_SWITCH_CASE_1
  }
  else
  {
    dim3 block(ThreadsPerBlock);
    dim3 grid(nMsg, blocksPerMsg);
#define NVCOMP_CRC32_INTERNAL_SWITCH_CASE_2(BPR, ...)                                                                  \
  case BPR:                                                                                                            \
    crc32_Batched_BL_k<ThreadsPerBlock, BPR><<<grid, block, 0, stream>>>(ctx->spec, nMsg, crcBuf_d, msgLen_d, msg_d);  \
    break;
    switch (bytesPerRead)
    {
      NVCOMP_CRC32_FOR_BYTES_PER_READ(NVCOMP_CRC32_INTERNAL_SWITCH_CASE_2);
      default:
        throw ::nvcomp::NVCompException(
          nvcompErrorInvalidValue,
          "read bytes for kernel 1 must be a power of 2 between 4 and 2048 (included), but is " +
            std::to_string(bytesPerRead)
        );
    }
    CUDA_CHECK(cudaGetLastError());
#undef NVCOMP_CRC32_INTERNAL_SWITCH_CASE_2
  }

  return CUCRC32_SUCCESS;
}

namespace
{

static int cuCRC32ffs(int mask)
{
#ifdef _MSC_VER
  unsigned long index = 0;
  bool isZero = (_BitScanForward(&index, mask) == 0);
  return isZero ? 0 : index + 1;
#else
  return __builtin_ffs(mask);
#endif
}

static int cuCRC32popcount(unsigned int mask)
{
#ifdef _MSC_VER
  return __popcnt(mask);
#else
  return __builtin_popcount(mask);
#endif
}

} // namespace

int cuCRC32ParamToConf(int kernel, int readBytes, int blocksPerMsg, unsigned int *conf)
{
  if (kernel != WarpKernel && kernel != BlockKernel)
  {
    return CUCRC32_ERROR_INVALID_KERNEL;
  }
  if (readBytes < 4 || readBytes > 2048 || cuCRC32popcount(readBytes) != 1)
  {
    return CUCRC32_ERROR_INVALID_READBYTES;
  }
  if (kernel == BlockKernel && (blocksPerMsg < 1 || blocksPerMsg >= (1 << ConfBlocksPerMsgBits)))
  {
    return CUCRC32_ERROR_INVALID_BLKXMSG;
  }

  if (kernel == WarpKernel)
  {
    blocksPerMsg = 0;
  }

  *conf = (blocksPerMsg << (ConfBytesPerReadBits + ConfKernelBits)) | ((cuCRC32ffs(readBytes) - 1) << ConfKernelBits) |
          kernel;

  return CUCRC32_SUCCESS;
}

int cuCRC32ConfToParam(unsigned int conf, int *kernel, int *readBytes, int *blocksPerMsg)
{
  *kernel = conf & ConfKernelMask;
  *readBytes = 1 << ((conf >> ConfKernelBits) & ConfBytesPerReadMask);
  *blocksPerMsg = conf >> (ConfBytesPerReadBits + ConfKernelBits);

  return CUCRC32_SUCCESS;
}

namespace
{

Ull getMaxLen(unsigned int n, const Ull *v_d, cudaStream_t stream)
{
  Ull max_h = 0;

  Ull *max_d = nullptr;
  CUDA_CHECK(cudaMallocWrapper(&max_d, sizeof(*max_d), stream));

  max_k<ThreadsPerBlock_max_k><<<1, ThreadsPerBlock_max_k, 0, stream>>>(n, v_d, max_d);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpyAsync(&max_h, max_d, sizeof(max_h), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaFreeWrapper(max_d, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return max_h;
}

static void flushL2(int device, cudaStream_t stream = 0)
{
  // flush L2
  int l2_size = 0;
  int *l2_flush_buf = nullptr;

  CUDA_CHECK(cudaDeviceGetAttribute(&l2_size, cudaDevAttrL2CacheSize, device));
  CUDA_CHECK(cudaMallocWrapper(&l2_flush_buf, l2_size, stream));
  CUDA_CHECK(cudaMemsetAsync(l2_flush_buf, 0, l2_size, stream));
  CUDA_CHECK(cudaFreeWrapper(l2_flush_buf, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

} // namespace

int cuCRC32ConfSearch(
  const crcCtx_t *ctx,
  unsigned int nMsg,
  const Ull *msgLen_d,
  const unsigned char *const *msg_d,
  unsigned int *conf,
  cudaStream_t stream
)
{
  constexpr float EarlyExitPercentThreshold = 2.f;

  if (!msgLen_d || !msg_d || nMsg < 1)
  {
    return CUCRC32_ERROR_INVALID_PARAMS;
  }

  unsigned int *crcBuf_d = ctx->crcBuf;

  // Zero CRC buffer to prevent uninitialized access warnings.
  CUDA_CHECK(cudaMemsetAsync(crcBuf_d, 0, nMsg * sizeof(unsigned int), stream));

  cudaEvent_t start{}, stop{};
  float et = 0;

  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  Ull maxMsgLen = getMaxLen(nMsg, msgLen_d, stream);

  float tmin = FLT_MAX;
  unsigned int bestKernel = WarpKernel;
  unsigned int bestBytesPerRead = 8; // for warp kernel
  unsigned int bestBlocksPerMsg = 0; // only for block kernel

  dim3 block(WARP_SIZE, ThreadsPerBlock / WARP_SIZE);
  dim3 grid(nvcomp::roundUpDiv(nMsg, ThreadsPerBlock / WARP_SIZE));

  float last_et = FLT_MAX;

  DeviceGuard device_guard(stream);

  const int n_sm = nvcomp::CudaUtils::get_sm_count(stream);

  int dev_id{-1};
  CUDA_CHECK(cudaGetDevice(&dev_id));

  int max_blocksxsm{-1};
  CUDA_CHECK(cudaDeviceGetAttribute(&max_blocksxsm, cudaDevAttrMaxBlocksPerMultiprocessor, dev_id));

  // read sizes are tested from the largest to the smallest
  // because the runtimes through the range follow a convex
  // parabola and (i) for large messages the runtimes with
  // large read sizes are much smaller than with small read
  // sizes; and (ii) with small messages the absolute runtimes
  // are very small regardless of the read size.
  // So starting from the largest read sizes and exiting as
  // soon as the there is a significant increases in runtime
  // is an effective way to make sure the minimum time has been
  // recorded.
  // To be conservative, we exit when the runtime with the current
  // conf is 2x the proevious one.

  // if the number of messages is not enough to guarantee at least
  // half maximum theoretical occupancy do not even try warp kernels
  const int thresBlocksXSM = std::max(1, max_blocksxsm / 2);
  if (nMsg >= static_cast<unsigned int>(thresBlocksXSM) * (ThreadsPerBlock / WARP_SIZE) * n_sm)
  {
    // test kernel crc32_Batched_WP_k()
    for (unsigned int bytesPerRead = 2048; bytesPerRead >= 4; bytesPerRead /= 2)
    {

      Ull bytesPerMsg = (Ull)WARP_SIZE * bytesPerRead;
      if (bytesPerMsg > maxMsgLen)
      {
        continue;
      }

#define NVCOMP_CRC32_INTERNAL_SWITCH_CASE_3(BPR, ...)                                                                  \
  case BPR:                                                                                                            \
    crc32_Batched_WP_k<WARP_SIZE, ThreadsPerBlock / WARP_SIZE, BPR>                                                    \
      <<<grid, block, 0, stream>>>(ctx->spec, nMsg, crcBuf_d, msgLen_d, msg_d);                                        \
    break;

      CUDA_CHECK(cudaEventRecord(start, stream));
      switch (bytesPerRead)
      {
        NVCOMP_CRC32_FOR_BYTES_PER_READ(NVCOMP_CRC32_INTERNAL_SWITCH_CASE_3)
      }
      CUDA_CHECK(cudaEventRecord(stop, stream));

#undef NVCOMP_CRC32_INTERNAL_SWITCH_CASE_3

      CUDA_CHECK(cudaEventSynchronize(stop));
      CUDA_CHECK(cudaEventElapsedTime(&et, start, stop));
      if (et < tmin)
      {
        tmin = et;
        bestBytesPerRead = bytesPerRead;
      }
      //printf("Wrp_k<%d><<<(%d, %d), (%d, %d)>>> time: %E ms\n", bytesPerRead, grid.x, grid.y, block.x, block.y, et);
      flushL2(dev_id, stream);

      // if with current bytesPerRead kernel time starts to increase
      // then there's no need to continue with larger values
      if (et > last_et * EarlyExitPercentThreshold)
      {
        break;
      }
      last_et = et;
    }
  }

  // test kernel crc32_Batched_BL_k()
  block = dim3(ThreadsPerBlock, 1, 1);
  grid = dim3(nMsg, 1, 1);

  unsigned int blocksPerMsg = 1;

  int max_grid_dim_y;
  CUDA_CHECK(cudaDeviceGetAttribute(&max_grid_dim_y, cudaDevAttrMaxGridDimY, dev_id));
  // TODO (uhofmann): check if the std::max has any meaning below and if there is a case where
  // we would end up with n_sm <= 0.
  const int max_blocks_mult_sm = (max_grid_dim_y / std::max(1, n_sm)) * n_sm;

  while (true)
  {
    grid.y = blocksPerMsg;

    last_et = FLT_MAX;

    for (unsigned int bytesPerRead = 2048; bytesPerRead >= 4; bytesPerRead /= 2)
    {

      Ull bytesPerMsg = (Ull)grid.y * ThreadsPerBlock * bytesPerRead;
      if (bytesPerMsg > maxMsgLen)
      {
        continue;
      }

#define NVCOMP_CRC32_INTERNAL_SWITCH_CASE_4(BPR, ...)                                                                  \
  case BPR:                                                                                                            \
    crc32_Batched_BL_k<ThreadsPerBlock, BPR><<<grid, block, 0, stream>>>(ctx->spec, nMsg, crcBuf_d, msgLen_d, msg_d);  \
    break;

      CUDA_CHECK(cudaEventRecord(start, stream));
      switch (bytesPerRead)
      {
        NVCOMP_CRC32_FOR_BYTES_PER_READ(NVCOMP_CRC32_INTERNAL_SWITCH_CASE_4)
      }
      CUDA_CHECK(cudaEventRecord(stop, stream));

#undef NVCOMP_CRC32_INTERNAL_SWITCH_CASE_4

      CUDA_CHECK(cudaEventSynchronize(stop));
      CUDA_CHECK(cudaEventElapsedTime(&et, start, stop));
      if (et < tmin)
      {
        bestKernel = BlockKernel;
        tmin = et;
        bestBytesPerRead = bytesPerRead;
        bestBlocksPerMsg = blocksPerMsg;
      }
      //printf("Blk_k<%d, %d><<<(%d, %d), %d>>> time: %E ms\n", ThreadsPerBlock, bytesPerRead, grid.x, grid.y, block.x, et);
      flushL2(dev_id, stream);

      // if with current bytesPerRead kernel time starts to increase
      // then there's no need to continue with larger values
      if (et > last_et * EarlyExitPercentThreshold)
      {
        break;
      }
      last_et = et;
    }
    blocksPerMsg *= 2;

    if (blocksPerMsg / 2 < static_cast<unsigned int>(n_sm) && blocksPerMsg > static_cast<unsigned int>(n_sm))
    {
      blocksPerMsg = static_cast<unsigned int>(n_sm);
    }
    if (blocksPerMsg > static_cast<unsigned int>(max_blocks_mult_sm))
    {
      break;
    }
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  *conf = 0;
  cuCRC32ParamToConf(bestKernel, bestBytesPerRead, bestBlocksPerMsg, conf);

  return CUCRC32_SUCCESS;
}

namespace
{

template <typename NumT>
static __global__ void initState_k(crcSpec_t spec, NumT n, unsigned int *__restrict__ crcBuf)
{
  const NumT tid = NumT(blockIdx.x) * blockDim.x + threadIdx.x;

  if (tid < n)
  {
    crcBuf[tid] = spec.init;
  }
}

} // namespace

int cuCRC32Beg(const crcCtx_t *ctx, unsigned int nMsg, cudaStream_t stream)
{
  if (nMsg < 1)
  {
    return CUCRC32_ERROR_INVALID_PARAMS;
  }

  // get context data
  unsigned int *crcBuf_d = ctx->crcBuf;

  initState_k<<<nvcomp::roundUpDiv(nMsg, ThreadsPerBlock), ThreadsPerBlock, 0, stream>>>(ctx->spec, nMsg, crcBuf_d);
  CUDA_CHECK(cudaGetLastError());

  return CUCRC32_SUCCESS;
}

namespace
{

template <int BlockDimX, typename NumT>
static __global__ void finalizeState_k(crcSpec_t spec, NumT n, unsigned int *__restrict__ state)
{
  const NumT tid = NumT(blockIdx.x) * blockDim.x + threadIdx.x;

  if (tid < n)
  {
    unsigned int crc = state[tid];

    if (spec.ref_out)
    {
      crc = __brev(crc);
    }
    crc ^= spec.xorout;

    state[tid] = crc;
  }
}

} // namespace

int cuCRC32End(const crcCtx_t *ctx, unsigned int nMsg, cudaStream_t stream)
{
  if (nMsg < 1)
  {
    return CUCRC32_ERROR_INVALID_PARAMS;
  }

  if (ctx->spec.ref_out || ctx->spec.xorout)
  {

    // get context data
    unsigned int *crcDst_d = ctx->crcBuf;

    finalizeState_k<ThreadsPerBlock>
      <<<nvcomp::roundUpDiv(nMsg, ThreadsPerBlock), ThreadsPerBlock, 0, stream>>>(ctx->spec, nMsg, crcDst_d);
    CUDA_CHECK(cudaGetLastError());
  }
  return CUCRC32_SUCCESS;
}

namespace
{

std::optional<int> parameterizedHeuristic_Warp(
  Ull nSM,
  unsigned int nMsg,
  Ull maxMsgLen,
  unsigned int *conf,
  int maxBytesPerRead,
  bool requireSingleWarpIteration
)
{
  // Check warp kernel applicability.

  int bytesPerRead = int(std::max<Ull>(4, std::min<Ull>(maxBytesPerRead, maxMsgLen / WARP_SIZE)));
  bytesPerRead = nextPow2(bytesPerRead);
  int blocksPerSM = 0;

#define NVCOMP_CRC32_INTERNAL_SWITCH_CASE_5(BPR, ...)                                                                  \
  case BPR:                                                                                                            \
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(                                                          \
      &blocksPerSM,                                                                                                    \
      crc32_Batched_WP_k<WARP_SIZE, ThreadsPerBlock / WARP_SIZE, BPR, unsigned int, Ull>,                              \
      ThreadsPerBlock,                                                                                                 \
      0                                                                                                                \
    ));                                                                                                                \
    break;

  switch (bytesPerRead)
  {
    NVCOMP_CRC32_FOR_BYTES_PER_READ(NVCOMP_CRC32_INTERNAL_SWITCH_CASE_5)
  }

#undef NVCOMP_CRC32_INTERNAL_SWITCH_CASE_5

  // Use warp kernels if the number of messages is large enough to guarantee at
  // least half maximum occupancy and loop-iteration requirements are met.
  const int threshBlocksPerSM = std::max(1, blocksPerSM / 2);

  bool useWarpKernel = Ull(nMsg) >= Ull(threshBlocksPerSM) * (ThreadsPerBlock / WARP_SIZE) * nSM;
  if (requireSingleWarpIteration)
  {
    useWarpKernel = useWarpKernel && maxMsgLen <= WARP_SIZE * maxBytesPerRead;
  }

  if (useWarpKernel)
  {
    return cuCRC32ParamToConf(WarpKernel, bytesPerRead, 0, conf);
  }
  else
  {
    return std::nullopt;
  }
}

int parameterizedHeuristic_Block(
  Ull nSM,
  Ull maxGridDimY,
  unsigned int nMsg,
  Ull maxMsgLen,
  unsigned int *conf,
  int minBytesPerRead,
  int maxBytesPerRead,
  int maxBlocksPerSM
)
{
  using NumT = unsigned int;
  using LenT = Ull;

  LenT blocksPerMsg = 0;

  LenT maxBlocks = 0;
  LenT maxBlocksPerMsg = 0;

  // Note: The control flow in this function may seem overly convoluted
  // because it attempts to exactly reproduce the behavior of the original
  // cuCRC32 implementation. Further investigation is needed before
  // simplifying.

  if (maxBlocksPerSM > 0)
  {
    maxBlocks = std::min(nSM * maxBlocksPerSM, maxGridDimY * nMsg);
    maxBlocksPerMsg = std::max<LenT>(1, maxBlocks / nMsg);
  }
  else
  {
    maxBlocksPerMsg = (maxGridDimY / std::max<LenT>(1, nSM)) * nSM;
    maxBlocks = maxBlocksPerMsg * nMsg;
  }

  LenT bytesPerRead = maxBytesPerRead;
  while (true)
  {
    LenT bytesPerBlock = ThreadsPerBlock * bytesPerRead;
    blocksPerMsg = nvcomp::roundUpDiv(maxMsgLen, bytesPerBlock);
    if (maxBlocksPerSM > 0)
    {
      // TODO: This has been copied from the cuCRC32 code, but
      // is difficult to interpret.
      // maxBlocksPerMsg would seem to make more sense than
      // just maxBlocks. Investigate.
      blocksPerMsg = std::min(blocksPerMsg, maxBlocks);
    }

    if (bytesPerRead == minBytesPerRead || (bytesPerBlock <= maxMsgLen && blocksPerMsg * nMsg >= nSM))
    {
      blocksPerMsg = std::min(blocksPerMsg, maxBlocksPerMsg);
      break;
    }

    bytesPerRead /= 2;
  }

  return cuCRC32ParamToConf(BlockKernel, int(bytesPerRead), int(blocksPerMsg), conf);
}

int parameterizedHeuristic(
  const int n_sm,
  const int max_grid_dim_y,
  unsigned int nMsg,
  Ull maxMsgLen,
  unsigned int *conf,
  int maxBytesPerRead_Warp,
  bool requireSingleWarpIteration,
  int minBytesPerRead_Block,
  int maxBytesPerRead_Block,
  int maxBlocksPerSM
)
{
  using NumT = unsigned int;
  using LenT = Ull;

  const LenT nSM = LenT(n_sm);

  std::optional<int> warpKernelStatus =
    parameterizedHeuristic_Warp(nSM, nMsg, maxMsgLen, conf, maxBytesPerRead_Warp, requireSingleWarpIteration);
  if (warpKernelStatus.has_value())
  {
    return *warpKernelStatus;
  }

  return parameterizedHeuristic_Block(
    nSM,
    LenT(max_grid_dim_y),
    nMsg,
    maxMsgLen,
    conf,
    minBytesPerRead_Block,
    maxBytesPerRead_Block,
    maxBlocksPerSM
  );
}

int heur_sm80(const int n_sm, const int max_grid_dim_y, unsigned int nMsg, Ull maxMsgLen, unsigned int *conf)
{
  return parameterizedHeuristic(
    n_sm,
    max_grid_dim_y,
    nMsg,
    maxMsgLen,
    conf,
    /*maxBytesPerRead_Warp=*/1024,
    /*requireSingleWarpIteration=*/true,
    /*minBytesPerRead_Block=*/128,
    /*maxBytesPerRead_Block=*/512,
    /*maxBlocksPerSM=*/128
  );
}

int heur_sm89(const int n_sm, const int max_grid_dim_y, unsigned int nMsg, Ull maxMsgLen, unsigned int *conf)
{
  return parameterizedHeuristic(
    n_sm,
    max_grid_dim_y,
    nMsg,
    maxMsgLen,
    conf,
    /*maxBytesPerRead_Warp=*/128,
    /*requireSingleWarpIteration=*/false,
    /*minBytesPerRead_Block=*/128,
    /*maxBytesPerRead_Block=*/512,
    /*maxBlocksPerSM=*/0
  );
}

int heur_sm100(const int n_sm, const int max_grid_dim_y, unsigned int nMsg, Ull maxMsgLen, unsigned int *conf)
{
  return parameterizedHeuristic(
    n_sm,
    max_grid_dim_y,
    nMsg,
    maxMsgLen,
    conf,
    /*maxBytesPerRead_Warp=*/2048,
    /*requireSingleWarpIteration=*/false,
    /*minBytesPerRead_Block=*/128,
    /*maxBytesPerRead_Block=*/2048,
    /*maxBlocksPerSM=*/0
  );
}

} // namespace

int cuCRC32DispatchHeur(
  const int arch,
  const int n_sm,
  const int max_grid_dim_y,
  unsigned int nMsg,
  unsigned long long maxMsgLen,
  unsigned int *conf
)
{
  // For the time being, use the heuristic for SM 80 for any SM number
  // less or equal to 80, and the heuristic for SM 100 for any SM number
  // strictly greater than 89.
  if (arch <= 80)
  {
    return heur_sm80(n_sm, max_grid_dim_y, nMsg, maxMsgLen, conf);
  }
  else if (arch <= 89)
  {
    return heur_sm89(n_sm, max_grid_dim_y, nMsg, maxMsgLen, conf);
  }
  else
  {
    return heur_sm100(n_sm, max_grid_dim_y, nMsg, maxMsgLen, conf);
  }
}

int cuCRC32ConfHeur(
  const crcCtx_t *ctx,
  unsigned int nMsg,
  const Ull *msgLen_d,
  unsigned int *conf,
  Ull maxMsgLen,
  cudaStream_t stream
)
{
  if (!ctx || (maxMsgLen == 0 && !msgLen_d) || !conf || nMsg < 1)
  {
    return CUCRC32_ERROR_INVALID_PARAMS;
  }

  if (maxMsgLen == 0)
  {
    maxMsgLen = getMaxLen(nMsg, msgLen_d, stream);
  }

  DeviceGuard device_guard(stream);

  int currDev = -1;
  CUDA_CHECK(cudaGetDevice(&currDev));

  // use the stream to derive the current number of available SMs
  // falls back to SMs on device for prior green context (13.10)
  // driver or compiler
  const int n_sm = nvcomp::CudaUtils::get_sm_count(stream);

  int max_grid_dim_y = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&max_grid_dim_y, cudaDevAttrMaxGridDimY, currDev));

  int major, minor;
  CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, currDev));
  CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, currDev));
  int merged_arch = major * 10 + minor;

  return cuCRC32DispatchHeur(merged_arch, n_sm, max_grid_dim_y, nMsg, maxMsgLen, conf);
}
