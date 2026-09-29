/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
*/

#include "bitcomp_private.h"
#include "bitmask.h"
#include "header.h"
#include "loadStore.h"
#include "transpose.h"

#include <stdio.h>

#pragma once

namespace bitcomp
{

namespace zbmap
{

// *************************************************************************************************
// Prefix sum inside a warp using shuffles

template <typename T>
inline __device__ void warpPrefixSum(T &val, uint &laneid)
{
  T tmp = __shfl_up_sync(0xffffffff, val, 1);
  if (laneid >= 1)
  {
    val += tmp;
  }
  tmp = __shfl_up_sync(0xffffffff, val, 2);
  if (laneid >= 2)
  {
    val += tmp;
  }
  tmp = __shfl_up_sync(0xffffffff, val, 4);
  if (laneid >= 4)
  {
    val += tmp;
  }
  tmp = __shfl_up_sync(0xffffffff, val, 8);
  if (laneid >= 8)
  {
    val += tmp;
  }
  tmp = __shfl_up_sync(0xffffffff, val, 16);
  if (laneid >= 16)
  {
    val += tmp;
  }
}

// *************************************************************************************************

inline __device__ uint mask2Value(const uint4 *sm)
{
  // Each thread loads 32 contiguous bytes as 2 x uint4
  uint4 v0, v1;
  v0 = sm[2 * threadIdx.x];
  v1 = sm[2 * threadIdx.x + 1];

  // Process 8 input bytes at once = 8 bits of output mask
  uint output = bitmask::nzbmask(v0.x, v0.y) + (bitmask::nzbmask(v0.z, v0.w) << 8) +
                (bitmask::nzbmask(v1.x, v1.y) << 16) + (bitmask::nzbmask(v1.z, v1.w) << 24);
  return output;
}

// *************************************************************************************************

template <typename T, bitcompDataType_t typeId, bitcompMode_t compMode, bitcompIntFormat_t ifmt, bool useAtomic>
inline __device__ void
encoder(const char *in, uint *out, uint64 *counter, uint64 &blockOffset, uint64 nbytes, uint blockIndex, T delta = 0.0)
{
  __shared__ uint4 sm[512];
  __shared__ uint mask2[256];
  __shared__ uint mask1[32];
  __shared__ uint mask0;
  unsigned int *smi = (unsigned int *)sm;
  unsigned char *smb = (unsigned char *)sm;
  unsigned char *mask2b = (unsigned char *)mask2;
  __shared__ uint warp_count_data[8];
  __shared__ int overflow, oversize, shm_lcompw, shm_lbmap;
  __shared__ uint64 shm_offset;
  uint warpid = threadIdx.x / 32;
  uint laneid = threadIdx.x & 31;

  // First thread of the first block updates global header
  if (blockIndex == 0 && threadIdx.x == 0)
  {
    header::setFlags<typeId, compMode, BITCOMP_SPARSE_ALGO, ifmt>(out);
    header::setUncompressedSize(out, nbytes);
    header::setScalingDelta(out, delta);
  }
  uint64 hdrlenw = header::computeHeaderLengthInWords(nbytes);

  // Initialize the shared memory
  if (threadIdx.x == 0)
  {
    overflow = 0;
    oversize = 0;
    mask0 = 0;
  }
  __syncthreads();

  uint64 inputOffset = blockIndex * 8192ULL;
  int blockBytes = min(8192ULL, nbytes - inputOffset);

  // Check if we can use 16-byte memory accesses on the input
  in += inputOffset;
  bool aligned = (((uintptr_t)in & 0xf) | (blockBytes & 0xf)) == 0;

  // Load the data from global memory to shared, optional: integer quantization
  if (loadStore::loadInputToShared<T, compMode, ifmt>(in, sm, blockBytes, aligned, delta) != 0)
  {
    overflow = 1;
  }

  __syncthreads();

  // If no overflow, continue compression
  if (overflow == 0)
  {
    uint vi[8];

    // Bit-plan transposition in shared memory
    transpose::bitPlanTranspose<sizeof(T), false>(sm);

    __syncthreads();

    // mask2 = bitmap of zeroes for each byte of 8KB buffer
    // Each thread computes the mask for 32 bytes -> 1024 bytes per warp
    mask2[threadIdx.x] = mask2Value(sm);

    // Each Warp will access the mask2 data it just wrote
    __syncwarp(0xffffffff);

    // Build mask1 from mask2 (manually enrolled 4x, the compiler doesn't want to do it)
    uint smoffset = warpid * 128 + laneid;
    {
      unsigned char b0 = mask2b[smoffset];
      unsigned char b1 = mask2b[smoffset + 32];
      unsigned char b2 = mask2b[smoffset + 64];
      unsigned char b3 = mask2b[smoffset + 96];
      uint tmp0 = __ballot_sync(0xffffffff, b0);
      uint tmp1 = __ballot_sync(0xffffffff, b1);
      uint tmp2 = __ballot_sync(0xffffffff, b2);
      uint tmp3 = __ballot_sync(0xffffffff, b3);
      if (laneid == 0)
      {
        mask1[warpid * 4] = tmp0;
        mask1[warpid * 4 + 1] = tmp1;
        mask1[warpid * 4 + 2] = tmp2;
        mask1[warpid * 4 + 3] = tmp3;
      }
    }

    // Count the number of non-zero data bytes for this warp
    uint nzd;
    nzd = __popc(mask2[threadIdx.x]);
    warpPrefixSum(nzd, laneid);
    if (laneid == 31)
    {
      warp_count_data[warpid] = nzd;
    }

    __syncthreads();

    // Warp 0 builds mask0 and consolidates the sums from all warps
    uint nz1, nz2;
    if (warpid == 0)
    {
      uint m1 = mask1[threadIdx.x];
      uint m0 = __ballot_sync(0xffffffff, m1);
      if (laneid == 0)
      {
        mask0 = m0;
      }
      // Count the number of bytes used for mask1 and mask2
      nz2 = __popc(m1);
      nz1 = __popc(m0);
      warpPrefixSum(nz2, laneid); // Thread 31 has the full count of mask2 bytes
      // Prefix sum the per-warp non-zero data count
      if (laneid < 8)
      {
        uint sum, tmp;
        sum = warp_count_data[threadIdx.x];
        tmp = __shfl_up_sync(0x000000ff, sum, 1);
        if (threadIdx.x > 0)
        {
          sum += tmp;
        }
        tmp = __shfl_up_sync(0x000000ff, sum, 2);
        if (threadIdx.x > 1)
        {
          sum += tmp;
        }
        tmp = __shfl_up_sync(0x000000ff, sum, 4);
        if (threadIdx.x > 3)
        {
          sum += tmp;
        }
        warp_count_data[threadIdx.x] = sum;
      }
      __syncwarp(0xffffffff);
      // Warp 0 thread 31 computes the compressed size, gets an output offset, updates header
      // WARNING: For batch mode (!useAtomic), only thread 31 has the right value in blockOffset!
      // Which means that the oversize case must also be handled by thread 31.
      if (laneid == 31)
      {
        // Full size of compressed data for this block, in words
        uint lcompw = 1 + nz1 + (nz2 + warp_count_data[7] + 3) / 4;
        shm_lcompw = lcompw;
        if (lcompw < 2048)
        {
          if (useAtomic)
          {
            blockOffset = atomicAdd(counter, (uint64)lcompw);
          }
          shm_offset = blockOffset + hdrlenw;
          // Write the block info in the header
          header::setBlockInfo(out, blockIndex, blockOffset, lcompw);
          if (!useAtomic)
          {
            blockOffset += lcompw;
          }
          // Share the length of the bitmaps in bytes with all threads
          shm_lbmap = 4 + nz1 * 4 + nz2;
        }
        else
        {
          // Block is oversize, we'll store the original data
          oversize = 1;
        }
      }
    }

    __syncthreads();

    if (oversize == 0)
    {

      // Load 8 consecutive 32-bit values from shared memory (32 bytes per thread)
      // before we overwrite the shared memory with compressed data (2-way conflict)
      uint4 v4[2];
      v4[0] = sm[threadIdx.x * 2];
      v4[1] = sm[threadIdx.x * 2 + 1];
      memcpy(vi, v4, 2 * sizeof(uint4));

      __syncthreads();

      // Store the compressed data in shared memory first

      // First warp writes the bitmaps
      if (warpid == 0)
      {
        uint m0 = mask0;
        uint m1 = mask1[threadIdx.x];
        // Thread 0 writes the L0 bitmap
        if (laneid == 0)
        {
          smi[0] = m0;
        }
        // First warp writes the L1 bitmap
        if (m1 != 0)
        {
          uint tmp = __popc(m0 & ((1 << threadIdx.x) - 1));
          smi[1 + tmp] = m1;
        }
        // First warp writes the L2 bitmap (32 bytes per thread, one byte at a time)
        uint off = 4 + nz1 * 4;
        uint tmp = __shfl_up_sync(0xffffffff, nz2, 1);
        if (laneid > 0)
        {
          off += tmp;
        }
        for (int i = 0; i < 32; i++)
        {
          unsigned char b = mask2b[laneid * 32 + i];
          if (b)
          {
            smb[off++] = b;
          }
        }
      }

      // All warps write the data
      uint off = shm_lbmap;
      // Skip the data from the previous warps
      if (warpid > 0)
      {
        off += warp_count_data[warpid - 1];
      }
      // Skip the data from the other threads in the warp
      uint tmp = __shfl_up_sync(0xffffffff, nzd, 1);
      if (laneid > 0)
      {
        off += tmp;
      }
      if (mask2[threadIdx.x])
      {
        // Write up to 32 bytes per thread
        for (int i = 0; i < 8; i++)
        {
          for (int j = 0; j < 4; j++)
          {
            if (vi[i] & 0xff)
            {
              smb[off++] = vi[i];
            }
            vi[i] >>= 8;
          }
        }
      }

      __syncthreads();

      // Now write the 32-bit compressed data to global memory
      out += shm_offset;
      for (int i = threadIdx.x; i < shm_lcompw; i += 256)
      {
        out[i] = smi[i];
      }

    } // end if oversize

  } // end if overflow

  if (overflow || oversize)
  {
    // Use thread 31, as it's the only thread that has the right blockOffset in batch mode
    // (in which case we're not using atomics)
    if (threadIdx.x == 31)
    {
      if (useAtomic)
      {
        blockOffset = atomicAdd(counter, 2048ULL);
      }
      shm_offset = blockOffset + hdrlenw;
      // Write the block info in the header, with the incompressible flag
      header::setBlockInfoOverflow(out, blockIndex, blockOffset, 2048);
      if (!useAtomic)
      {
        blockOffset += 2048;
      }
    }
    __syncthreads();

    // Copy the original data as is
    loadStore::saveIncompressibleBlock<T>(in, out + shm_offset, blockBytes, aligned);
  }
  // In batch mode, the main encoder kernel expects thread 0 to have the proper value
  // for blockOffset.
  if (!useAtomic)
  {
    blockOffset = __shfl_sync(0xffffffff, blockOffset, 31);
  }
}

// *************************************************************************************************

template <typename T, bitcompMode_t compMode, bitcompIntFormat_t ifmt>
__device__ void decoder(const uint *in, char *out, int blockStart, int blockEnd, bool aligned, T delta)
{
  __shared__ uint4 sm[512];
  __shared__ uint warp_bits2[8];

  uint warpid = threadIdx.x / 32;
  uint laneid = threadIdx.x & 31;

  // Bitmask decompression strategy:
  // All the warps read mask0 and rebuild mask1 in parallel (duplicated work)
  // Each thread gets the right mask1 value using shuffles, and builds a mask2 value
  // Then, a global prefix sum computes the offset for each thread to read its data bytes.

  uint mask0 = in[0];
  if (mask0 == 0)
  {
    // Set the output block to zero and return
    loadStore::zeroOutputBlock<T>(out, blockStart, blockEnd, aligned);
    return;
  }

  // Initialize the data in shared memory to zero
  for (int i = threadIdx.x; i < 512; i += 256)
  {
    sm[i] = make_uint4(0, 0, 0, 0);
  }

  uint vi[8];
  uint offset, nz1, nz2, mybit, tmp;
  uint mask1 = 0;
  uint mask2 = 0;

  // Each warp rebuilds all 32 mask1 values (one value per lane)
  nz1 = __popc(mask0);
  mybit = 1 << laneid;
  offset = 1 + __popc(mask0 & (mybit - 1));
  if (mask0 & mybit)
  {
    mask1 = in[offset];
  }

  // Prefix sum of how many bits are set in mask1 values at each lane
  uint bits1 = __popc(mask1);
  warpPrefixSum(bits1, laneid);

  // Lane 31 has the number of non-zero bytes in mask2
  nz2 = __shfl_sync(0xffffffff, bits1, 31);

  // All the warps decompress the mask2 bitmap, each thread builds 32 bits = 4 bytes of mask2.
  // 8 threads will use the same mask1 value (4 bits per thread)
  int i1 = threadIdx.x / 8;
  int sublane = laneid & 7;
  // Get the mask1 value from the right thread in the warp
  mask1 = __shfl_sync(0xffffffff, mask1, i1);
  mybit = 1 << (sublane * 4);
  // Offset for mask2 is now in bytes
  offset = 4 + nz1 * 4;
  // Skip the bytes from lower mask1 values
  tmp = __shfl_sync(0xffffffff, bits1, i1 - 1);
  if (i1 > 0)
  {
    offset += tmp;
  }
  // Skip the bytes from the lower threads using the same mask1 value
  offset += __popc(mask1 & (mybit - 1));
  // Build the mask2 with 4 consecutive bytes
  for (int i = 0; i < 4; i++)
  {
    if (mask1 & (mybit << i))
    {
      mask2 |= ((unsigned char *)in)[offset++] << (8 * i);
    }
  }

  // Compute the prefix sum of mask2 bits accross all the threads
  uint bits2 = __popc(mask2);
  warpPrefixSum(bits2, laneid);
  if (laneid == 31)
  {
    warp_bits2[warpid] = bits2;
  }
  __syncthreads();
  if (threadIdx.x < 8)
  {
    uint sum;
    sum = warp_bits2[threadIdx.x];
    tmp = __shfl_up_sync(0x000000ff, sum, 1);
    if (threadIdx.x > 0)
    {
      sum += tmp;
    }
    tmp = __shfl_up_sync(0x000000ff, sum, 2);
    if (threadIdx.x > 1)
    {
      sum += tmp;
    }
    tmp = __shfl_up_sync(0x000000ff, sum, 4);
    if (threadIdx.x > 3)
    {
      sum += tmp;
    }
    warp_bits2[threadIdx.x] = sum;
  }
  __syncthreads();

  // Offset for the data bytes
  offset = 4 + nz1 * 4 + nz2;
  // Skip the bytes from the lower warps
  if (warpid > 0)
  {
    offset += warp_bits2[warpid - 1];
  }
  // Skip the bytes from the lower threads in the warp
  tmp = __shfl_sync(0xffffffff, bits2, laneid - 1);
  if (laneid > 0)
  {
    offset += tmp;
  }
  // Read up to 32 bytes, to rebuild 8 32-bit values, store in shared memory
  mybit = 1;
  for (int i = 0; i < 8; i++)
  {
    vi[i] = 0;
    for (int j = 0; j < 4; j++)
    {
      if (mask2 & mybit)
      {
        vi[i] |= ((unsigned char *)in)[offset++] << (j * 8);
      }
      mybit <<= 1;
    }
  }

  // Store 32 contiguous bytes in shared memory
  uint4 v4[2];
  memcpy(v4, vi, 2 * sizeof(uint4));
  sm[threadIdx.x * 2] = v4[0];
  sm[threadIdx.x * 2 + 1] = v4[1];

  __syncthreads();

  // Backward bit-plan transpose in shared memory
  transpose::bitPlanTransposeBack<sizeof(T)>(sm);

  __syncthreads();

  // Backward quantize+format on the fly as we save into global memory
  if constexpr (compMode == BITCOMP_LOSSLESS)
  {
    loadStore::storeSharedToOutput<T, compMode, ifmt, true>(out, sm, blockStart, blockEnd, aligned, delta);
  }
  else
  {
    if (utilities::zeroMantissaBits(delta) == delta)
    {
      loadStore::storeSharedToOutput<T, compMode, ifmt, true>(out, sm, blockStart, blockEnd, aligned, delta);
    }
    else
    {
      loadStore::storeSharedToOutput<T, compMode, ifmt, false>(out, sm, blockStart, blockEnd, aligned, delta);
    }
  }
}

} // namespace zbmap
} // namespace bitcomp
