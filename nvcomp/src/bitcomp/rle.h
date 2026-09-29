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
#include "rle_common.h"
#include "transpose.h"

#include <stdio.h>

#pragma once

namespace bitcomp
{

namespace rle
{

// *************************************************************************************************
// Make a bitmap for bytes that have the same direct higher neighbor, for 8 x 32-bit values

inline __device__ void getH1Bitmap32(uint v[8], uint h_nb, uint &maskh1)
{
  uint v0h, v1h, v2h, v3h, v4h, v5h, v6h, v7h;
  maskh1 = 0;

  // Shift the data 1 byte down + XOR
  v0h = v[0] ^ __funnelshift_rc(v[0], v[1], 8);
  v1h = v[1] ^ __funnelshift_rc(v[1], v[2], 8);
  v2h = v[2] ^ __funnelshift_rc(v[2], v[3], 8);
  v3h = v[3] ^ __funnelshift_rc(v[3], v[4], 8);
  v4h = v[4] ^ __funnelshift_rc(v[4], v[5], 8);
  v5h = v[5] ^ __funnelshift_rc(v[5], v[6], 8);
  v6h = v[6] ^ __funnelshift_rc(v[6], v[7], 8);
  v7h = v[7] ^ __funnelshift_rc(v[7], h_nb, 8);

  maskh1 = bitmask::zbmask(v0h, v1h) + (bitmask::zbmask(v2h, v3h) << 8) + (bitmask::zbmask(v4h, v5h) << 16) +
           (bitmask::zbmask(v6h, v7h) << 24);
}

// *************************************************************************************************
// Make two bitmaps for bytes that are zero of FF, for 8 x 32-bit values

inline __device__ void getZeroFFBitmap32(uint v[8], uint &mask00, uint &maskff)
{
  mask00 = bitmask::zbmask(v[0], v[1]) + (bitmask::zbmask(v[2], v[3]) << 8) + (bitmask::zbmask(v[4], v[5]) << 16) +
           (bitmask::zbmask(v[6], v[7]) << 24);
  maskff = bitmask::ffbmask(v[0], v[1]) + (bitmask::ffbmask(v[2], v[3]) << 8) + (bitmask::ffbmask(v[4], v[5]) << 16) +
           (bitmask::ffbmask(v[6], v[7]) << 24);
}

// *************************************************************************************************
// Prefix sum of the 256 x control-lengths and the 256 x data-lengths in shared memory,
// using 2 warps.
// Threads 0..31 are summing the control lengths (first 256 elements)
// Threads 32..63 are summing the data lengths (next 256 elements)
// Using uint4 to minimize shared memory bank conflicts.

inline __device__ void warpSumControlDataBytes(uint4 *sm)
{
  uint4 l0, l1, t0, t1;
  int lane = threadIdx.x & 31;

  l0 = sm[threadIdx.x * 2];
  l1 = sm[threadIdx.x * 2 + 1];

  // Stride 1
  t1.w = __shfl_up_sync(0xffffffff, l1.w, 1);
  l1.w += l1.z;
  l1.z += l1.y;
  l1.y += l1.x;
  l1.x += l0.w;
  l0.w += l0.z;
  l0.z += l0.y;
  l0.y += l0.x;
  if (lane > 0)
  {
    l0.x += t1.w;
  }

  // Stride 2
  t1.w = __shfl_up_sync(0xffffffff, l1.w, 1);
  t1.z = __shfl_up_sync(0xffffffff, l1.z, 1);
  l1.w += l1.y;
  l1.z += l1.x;
  l1.y += l0.w;
  l1.x += l0.z;
  l0.w += l0.y;
  l0.z += l0.x;
  if (lane > 0)
  {
    l0.y += t1.w;
    l0.x += t1.z;
  }

  // Stride 4
  t1.w = __shfl_up_sync(0xffffffff, l1.w, 1);
  t1.z = __shfl_up_sync(0xffffffff, l1.z, 1);
  t1.y = __shfl_up_sync(0xffffffff, l1.y, 1);
  t1.x = __shfl_up_sync(0xffffffff, l1.x, 1);
  l1.w += l0.w;
  l1.z += l0.z;
  l1.y += l0.y;
  l1.x += l0.x;
  if (lane > 0)
  {
    l0.w += t1.w;
    l0.z += t1.z;
    l0.y += t1.y;
    l0.x += t1.x;
  }

  // Stride 8
  t0.x = __shfl_up_sync(0xffffffff, l0.x, 1);
  t0.y = __shfl_up_sync(0xffffffff, l0.y, 1);
  t0.z = __shfl_up_sync(0xffffffff, l0.z, 1);
  t0.w = __shfl_up_sync(0xffffffff, l0.w, 1);
  t1.x = __shfl_up_sync(0xffffffff, l1.x, 1);
  t1.y = __shfl_up_sync(0xffffffff, l1.y, 1);
  t1.z = __shfl_up_sync(0xffffffff, l1.z, 1);
  t1.w = __shfl_up_sync(0xffffffff, l1.w, 1);
  if (lane > 0)
  {
    l0.x += t0.x;
    l0.y += t0.y;
    l0.z += t0.z;
    l0.w += t0.w;
    l1.x += t1.x;
    l1.y += t1.y;
    l1.z += t1.z;
    l1.w += t1.w;
  }

  // Stride 16
  t0.x = __shfl_up_sync(0xffffffff, l0.x, 2);
  t0.y = __shfl_up_sync(0xffffffff, l0.y, 2);
  t0.z = __shfl_up_sync(0xffffffff, l0.z, 2);
  t0.w = __shfl_up_sync(0xffffffff, l0.w, 2);
  t1.x = __shfl_up_sync(0xffffffff, l1.x, 2);
  t1.y = __shfl_up_sync(0xffffffff, l1.y, 2);
  t1.z = __shfl_up_sync(0xffffffff, l1.z, 2);
  t1.w = __shfl_up_sync(0xffffffff, l1.w, 2);
  if (lane > 1)
  {
    l0.x += t0.x;
    l0.y += t0.y;
    l0.z += t0.z;
    l0.w += t0.w;
    l1.x += t1.x;
    l1.y += t1.y;
    l1.z += t1.z;
    l1.w += t1.w;
  }

  // Stride 32
  t0.x = __shfl_up_sync(0xffffffff, l0.x, 4);
  t0.y = __shfl_up_sync(0xffffffff, l0.y, 4);
  t0.z = __shfl_up_sync(0xffffffff, l0.z, 4);
  t0.w = __shfl_up_sync(0xffffffff, l0.w, 4);
  t1.x = __shfl_up_sync(0xffffffff, l1.x, 4);
  t1.y = __shfl_up_sync(0xffffffff, l1.y, 4);
  t1.z = __shfl_up_sync(0xffffffff, l1.z, 4);
  t1.w = __shfl_up_sync(0xffffffff, l1.w, 4);
  if (lane > 3)
  {
    l0.x += t0.x;
    l0.y += t0.y;
    l0.z += t0.z;
    l0.w += t0.w;
    l1.x += t1.x;
    l1.y += t1.y;
    l1.z += t1.z;
    l1.w += t1.w;
  }

  // Stride 64
  t0.x = __shfl_up_sync(0xffffffff, l0.x, 8);
  t0.y = __shfl_up_sync(0xffffffff, l0.y, 8);
  t0.z = __shfl_up_sync(0xffffffff, l0.z, 8);
  t0.w = __shfl_up_sync(0xffffffff, l0.w, 8);
  t1.x = __shfl_up_sync(0xffffffff, l1.x, 8);
  t1.y = __shfl_up_sync(0xffffffff, l1.y, 8);
  t1.z = __shfl_up_sync(0xffffffff, l1.z, 8);
  t1.w = __shfl_up_sync(0xffffffff, l1.w, 8);
  if (lane > 7)
  {
    l0.x += t0.x;
    l0.y += t0.y;
    l0.z += t0.z;
    l0.w += t0.w;
    l1.x += t1.x;
    l1.y += t1.y;
    l1.z += t1.z;
    l1.w += t1.w;
  }

  // Stride 128
  t0.x = __shfl_up_sync(0xffffffff, l0.x, 16);
  t0.y = __shfl_up_sync(0xffffffff, l0.y, 16);
  t0.z = __shfl_up_sync(0xffffffff, l0.z, 16);
  t0.w = __shfl_up_sync(0xffffffff, l0.w, 16);
  t1.x = __shfl_up_sync(0xffffffff, l1.x, 16);
  t1.y = __shfl_up_sync(0xffffffff, l1.y, 16);
  t1.z = __shfl_up_sync(0xffffffff, l1.z, 16);
  t1.w = __shfl_up_sync(0xffffffff, l1.w, 16);
  if (lane > 15)
  {
    l0.x += t0.x;
    l0.y += t0.y;
    l0.z += t0.z;
    l0.w += t0.w;
    l1.x += t1.x;
    l1.y += t1.y;
    l1.z += t1.z;
    l1.w += t1.w;
  }

  sm[threadIdx.x * 2] = l0;
  sm[threadIdx.x * 2 + 1] = l1;
}

// *************************************************************************************************
// Inclusive prefix sum at warp or sub-warp level.
// All threads must participate

template <int width, typename T>
inline __device__ void prefixsum(T &value, T &total, int laneId)
{
  static_assert(width == 2 || width == 4 || width == 8 || width == 16 || width == 32, "Incompatible width");
#pragma unroll
  for (int ioff = 1; ioff < width; ioff += ioff)
  {
    T tmp = __shfl_up_sync(~0, value, ioff);
    if (laneId >= ioff)
    {
      value += tmp;
    }
  }
  total = __shfl_sync(~0, value, width - 1);
}

// *************************************************************************************************
// Inclusive prefix sum on 256 threads. All threads must participate
// The user is responsible for protecting the shared memory between calls
template <typename T>
inline __device__ void prefixsum_256(T &value, T &total, T sm[8], int warpId, int laneId)
{
  // Prefix sum inside the warp
  T unused;
  prefixsum<32>(value, unused, laneId);
  if (laneId == 31)
  {
    sm[warpId] = value;
  }
  __syncthreads();

  // Prefix sum the 8 warp sums
  // Default value for coverity.
  // Real value is set in the if statement below
  // Only values from first 8 threads are used to compute prefsums.
  T warpsum = 0;
  if (laneId < 8)
  {
    warpsum = sm[laneId];
  }
  prefixsum<8>(warpsum, total, laneId);
  // Add the sum of lower warps
  warpsum = __shfl_sync(~0, warpsum, warpId - 1);
  if (warpId > 0)
  {
    value += warpsum;
  }
}

// *************************************************************************************************
// RLE encoder, with 256 threads working on 8KB blocks

template <typename T, bitcompDataType_t typeId, bitcompMode_t compMode, bitcompIntFormat_t ifmt, bool useAtomic>
inline __device__ void encoder(
  const char *__restrict__ in,
  uint *__restrict__ out,
  uint64 *__restrict__ counter,
  uint64 &blockOffset,
  uint64 nbytes,
  uint blockIndex,
  T delta = 0.0
)
{

  // Shared memory to exchange data between threads, and to store compressed data
  // Includes padding for 256 x 9 x 32-bit values (padded after bit-plan transpose)
  // The padding is OK for the worst case compressed length : repetition of [3 bytes duplicate + 33 bytes non-duplicates]
  // = 2 bytes for duplicates + 35 bytes for non-duplicates = 37 bytes of
  // compressed data for 36 bytes of data -> 8436 bytes for a block of NOMINAL_BLOCK_SIZE bytes
  __shared__ uint4 sm[576];
  uint *smi = reinterpret_cast<uint *>(sm);
  unsigned char *smb = reinterpret_cast<unsigned char *>(sm);
  __shared__ int overflow;
  __shared__ uint64 shm_offset;
  __shared__ unsigned int smmask32[9]; // Need warps + 1 elements

  int laneId = threadIdx.x & 31;
  int warpId = threadIdx.x / 32;

  // First thread of the first block updates global header
  if (blockIndex == 0 && threadIdx.x == 0)
  {
    header::setFlags<typeId, compMode, BITCOMP_DEFAULT_ALGO, ifmt>(out);
    header::setUncompressedSize(out, nbytes);
    header::setScalingDelta(out, delta);
  }
  uint64 hdrlenw = header::computeHeaderLengthInWords(nbytes);

  // Initialize the shared memory
  if (threadIdx.x == 0)
  {
    overflow = 0;
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

    // Each thread processes 32 bytes = 8 uints + 2 neighbors = 40 bytes
    uint v[8], l_neighbor, h_neighbor;
    uint dup, first;
    int length;

    // Bit-plan transposition in shared memory, with padding
    transpose::bitPlanTranspose<sizeof(T), true>(sm);

    __syncthreads();

    // Load 8 contiguous values from shared memory + 2 neighbors
    uint smindex = threadIdx.x * 8 + threadIdx.x;
    for (int i = 0; i < 8; i++)
    {
      v[i] = smi[smindex + i];
    }
    // Load the lower and higher neighbors, if they exist. If they don't exist,
    // make sure they don't match the last byte (false repetition).
    if (threadIdx.x > 0)
    {
      l_neighbor = smi[smindex - 2]; // -2 because we skip the padding
    }
    else
    {
      l_neighbor = ~v[0] << 24;
    }
    if (threadIdx.x < 255)
    {
      h_neighbor = smi[smindex + 9]; // +9 because we skip the SMem padding
    }
    else
    {
      h_neighbor = ~v[7] >> 24;
    }

    __syncthreads();

    // Build bitmaps to identify :
    //  - Bytes that are duplicated bytes (strings of at least 3 duplicates)
    //  - Bytes that are the first byte of a duplicate or non-duplicate string
    // Each thread processes 32 bytes -> 32-bit masks, the masks are set to 1 when :
    //  H1 = Higher neighbor is the same
    //  H2 = Our higher neighbor has a same higher neighbor
    //  L1 = Our lower neighbor is the same
    //  L2 = Our lower neighbor has a same lower neighbor
    // A byte is part of at least 3 duplicates :
    //  - If the 2 lower neighbors have higher duplicates (L1 and L2), or
    //  - If it has a higher duplicate (H1), and either a lower duplicate(L1), or a second higer duplicate(H2)
    // A byte is the first of a string if :
    //  - It is a duplicate and the lower neighbor is not a duplicate, or vice versa.
    //  - It is a duplicate, its lower neighbor is a duplicate too but for different values (L1=0)
    {
      uint h1, h2, l1, l2;
      getH1Bitmap32(v, h_neighbor, h1);
      // Set the bytes of l_neighbor and h_neighbor to zero if they're the same as their higher neighbor
      l_neighbor ^= __funnelshift_rc(l_neighbor, v[0], 8);
      h_neighbor = (h_neighbor ^ (h_neighbor >> 8)) & 0xff;
      // Compute H2, L1, L2 bitmaps from H1, v0, v9
      l1 = h1 << 1;
      if ((l_neighbor & 0xff000000) == 0)
      {
        l1++;
      }
      h2 = h1 >> 1;
      if (h_neighbor == 0)
      {
        h2 |= 1U << 31;
      }
      l2 = l1 << 1;
      if ((l_neighbor & 0x00ff0000) == 0)
      {
        l2++;
      }
      // Compute the duplicate bitmap
      dup = (h1 & (l1 | h2)) | (l1 & l2);

      // Next thread will need to know if the last value is a duplicate
      smi[threadIdx.x] = dup >> 31;

      __syncthreads();

      // Find out if a value is the first of a duplicated string, or the first of a non-duplicated string
      first = dup << 1;
      if (threadIdx.x > 0)
      {
        first |= smi[threadIdx.x - 1];
      }
      else
      {
        first |= 1;
      }
      __syncthreads();
      first = (dup ^ first) | (dup & first & ~l1);
    }

    // A string of duplicates or non-duplicates can span across several threads.
    // We need to do some kind of prefix sum and keep adding next thread's lengths if the string continues past them
    // Count how many bytes can contribute to the threads before this one.
    length = __clz(__brev(first));
    smi[threadIdx.x] = length;

    // Build a bitmap of the threads which have a length of 32, so we can skip them quickly
    // and find the first higher thread which has a length which is not 32.
    // This is where the string starting at this thread will end.
    uint mask32 = __ballot_sync(0xffffffff, length == 32);

    if (laneId == 0)
    {
      smmask32[warpId] = mask32;
    }
    if (threadIdx.x == 0)
    {
      smmask32[8] = 0; // Stopping condition for loop below
    }
    __syncthreads();

    // Find the position of the first higher bit which is not set in mask32.
    uint maskbelow = 0xffffffff << (laneId + 1); // Mask out threads below, including this one.
    mask32 = ~mask32 & maskbelow;
    uint ibit = warpId * 32;
    int iwarp = warpId + 1;
    while (mask32 == 0)
    {
      ibit += 32;
      mask32 = ~smmask32[iwarp++];
    }
    ibit += __ffs(mask32) - 1;

    // All the threads between this thread and thread "ibit" have a length of 32.
    length = (ibit - threadIdx.x - 1) * 32;
    if (ibit < 256) // If ibit=256, all the higher threads have a length of 32.
    {
      length += smi[ibit];
    }

    // length is the contributing length of the threads after this one.
    // A double control byte happens for non-duplicate strings of length >= 32,
    //   or for duplicate strings of length >= 35
    // Thus, each thread, with 32 bytes per thread, can only have one double control byte.
    // We need to count the number of control bytes and data bytes of each thread.
    {
      int dataBytes, extBytes, msb, control;
      uint keep, mask, maskzero, maskff, blockBytes;

      // Build a mask for the bytes that must be kept.
      // We keep all non-duplicate bytes, and the first of duplicate bytes unless they're 0x00 or 0xFF.
      getZeroFFBitmap32(v, maskzero, maskff);
      keep = ~dup | (dup & first & ~(maskzero | maskff));

      msb = __clz(first);

      // Compute the last control byte to know if it's an extended control byte (long length)
      // Length stored = length-1 for non-duplicates, or length-3 for duplicates
      // Local length of the current thread's last string
      control = 0;
      mask = 0x80000000 >> msb; // if first is zero, mask becomes 0 too.
      if (dup & mask)
      {
        length -= 2;
        // While we're in there, start to compute some of the last control byte
        control = CTRL_DUP;
        if (maskzero & mask)
        {
          control = CTRL_DUP00;
        }
        if (maskff & mask)
        {
          control = CTRL_DUPFF;
        }
      }
      if (first)
      {
        length += msb;
      }
      else
      {
        length = 0;
      }

      // Number of control and data bytes in this thread (not counting double control byte)
      dataBytes = __popc(keep);
      int controlBytes = __popc(first);

      // If the length is 32 or more, we have an extended control byte (max 1 per thread)
      int warpExtMask = __ballot_sync(0xffffffff, length > 31);

      // Perform a prefix sum of both controlBytes and dataBytes at the same time,
      // so that each thread can know where to start writing its own compressed data
      // Using sm[256..767] to avoid a synchtreads after loading warpSumLength (sm[0..255])
      smi[threadIdx.x + 256] = controlBytes;
      smi[threadIdx.x + 512] = dataBytes;
      if (laneId == 0)
      {
        smi[warpId + 768] = __popc(warpExtMask);
      }
      __syncthreads();
      // Warp 0 sums controlBytes in sm[256:511]
      // Warp 1 sums dataBytes    in sm[512:767]
      if (threadIdx.x < 64)
      {
        warpSumControlDataBytes(reinterpret_cast<uint4 *>(smi + 256));
      }
      // Warp 2 sums warpExtBytes in sm[768:775] (one value per warp)
      if (warpId == 2)
      {
        uint value = smi[768 + laneId];
        uint unused;
        prefixsum<8>(value, unused, laneId);
        smi[768 + laneId] = value;
      }
      __syncthreads();

      // Now we know how many bytes each thread must write, (control bytes and data bytes).
      //
      // Compressed data layout:
      //
      // The control bytes are written in order in memory, even though each thread write its
      // control bytes backward, because we need to deal with extended control bytes first,
      // and only the last contol byte can be a double control byte for any given thread.
      //
      // The control and extension bytes are written backwards in memory (thread 0 at the end of the block)
      // but also in reverse order (starting from the last control code, towards the first one).
      // The first 2 bytes of the compressed output contain the number of control codes.
      // The data bytes are stored after that, in order.
      // So the compressed data layout is:
      // [ ncodehi ncodelo data0 data1 ... dataN ExtK ... Ext0 CtrlJ ... Ctrl0 ]
      // This layout allows a faster decompression.

      // Total number of bytes for this compressed block, multiple of 4.
      blockBytes = 2 + smi[511] + smi[767] + smi[775];
      blockBytes = (blockBytes + 3) & 0xfffffffc;

      // Continue if the block can be compressed (smaller than original size)
      if (blockBytes < NOMINAL_BLOCK_SIZE)
      {
        // Thread 0 updates the header and gets an output offset to share with all the threads
        if (threadIdx.x == 0)
        {
          uint64 lcompw = blockBytes >> 2;
          if (useAtomic)
          {
            blockOffset = atomicAdd(counter, lcompw);
          }
          shm_offset = blockOffset + hdrlenw;
          header::setBlockInfo(out, blockIndex, blockOffset, lcompw);
          if (!useAtomic)
          {
            blockOffset += lcompw;
          }
        }

        // Offset where this thread starts writing its data bytes
        dataBytes = 2;
        if (threadIdx.x > 0)
        {
          dataBytes += smi[512 + threadIdx.x - 1];
        }

        // Offset where this thread starts writing its last control byte
        controlBytes = blockBytes - smi[256 + threadIdx.x];
        int sumControlBytes = smi[511];

        // Offset where this thread writes its control byte extension (if any)
        // From the end of the block, backward, skip control bytes, and lower warps & threads.
        uint maskBelow = (1 << laneId) - 1; // Mask for lower lanes
        extBytes = blockBytes - smi[511] - __popc(warpExtMask & maskBelow) - 1;
        if (warpId > 0)
        {
          extBytes -= smi[768 + warpId - 1];
        }

        __syncthreads();

        // From now on, the shared memory will contain compressed data, accessed as as bytes (smb)
        // ---------------------------------------------------------------------------------------

        // Thread 0 writes the number of control codes in the first 2 bytes
        if (threadIdx.x == 0)
        {
          smb[0] = sumControlBytes >> 8;
          smb[1] = sumControlBytes & 0xff;
        }

        // Write control bytes, if any
        if (first)
        {
          // Clear the MSB bit in "first"
          first = first & ~mask;

          // Write the last control code, which can be an extended one
          if (length > 31)
          {
            smb[extBytes] = length;
            length = length >> 8;
            control |= CTRL_LONG;
          }
          control |= length;
          smb[controlBytes++] = control;

          // Encode all the other control bytes while there are bits set in "first"
          while (first)
          {
            // Find a new MSB and compute a new length
            int oldmsb = msb;
            msb = __clz(first);
            length = msb - oldmsb - 1;

            control = 0;
            // If the string is a duplicate, use length - 2
            // (the actual length is length + 1, or length + 3 for duplicates)
            // Clear the FSB bit in "first" at the same time.
            mask = 0x80000000 >> msb;
            first = first & ~mask;
            if (dup & mask)
            {
              length -= 2;
              control = CTRL_DUP;
              if (maskzero & mask)
              {
                control = CTRL_DUP00;
              }
              if (maskff & mask)
              {
                control = CTRL_DUPFF;
              }
            }
            // Write the new control byte
            control |= length;
            smb[controlBytes++] = control;
          }
        }

        // Now write all the data bytes, one byte at a time
        if (keep)
        {
          for (int i = 0; i < 8; i++)
          {
            if (keep & 0x1)
            {
              smb[dataBytes++] = (v[i]);
            }
            if (keep & 0x2)
            {
              smb[dataBytes++] = (v[i] >> 8);
            }
            if (keep & 0x4)
            {
              smb[dataBytes++] = (v[i] >> 16);
            }
            if (keep & 0x8)
            {
              smb[dataBytes++] = (v[i] >> 24);
            }
            keep >>= 4;
          }
        }

        __syncthreads();

        // Write the compressed data to global memory, in 32-bit words
        out += shm_offset;
        for (int i = threadIdx.x; i < (blockBytes >> 2); i += 256)
        {
          out[i] = smi[i];
        }

        // Compression successful
        return;

      } // End if (blockBytes < NOMINAL_BLOCK_SIZE)
    }

  } // End if overflow

  // If we got here, there was either overflow during the integer quantization,
  // or the block does not compress. Store the original data instead, with overflow flag.

  if (threadIdx.x == 0)
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

// *************************************************************************************************
// RLE decoder, with 8 warps, one warp prepares the data for the others, asynchronously

// Decompression strategy:
// 1) - All the threads copy the compressed data in shared memory
//    - All the threads initialize the output shared memory to zero.
// 2) Loop on all the control codes, 64 codes at once (2 codes x 32 threads)
//    - One producer warp decodes the next 64 control codes (code, read offset, write offset / length).
//      The codes for zero repeats are skiped, so it might produce less than 64 codes.
//      The average code length is also estimated.
//    - If the average length is < 8, use all 8 warps as 64 sub-warps of 4 threads each.
//      Each sub-warp gets at most one code (and hopefully the producer gets none)
//    - If the average length is >= 8, use only the non-producer warps to loop through the codes.
// 3) Process the shared memory (bit-plane transpose, quantization)

using ushort = unsigned short;

template <typename T, bitcompMode_t compMode, bitcompIntFormat_t ifmt>
__device__ __forceinline__ void decoder(
  const uint *__restrict__ in,
  char *__restrict__ out,
  int blockStart,
  int blockEnd,
  bool aligned,
  T delta,
  uint lcompw
)
{

  __shared__ uint4 smOut[512]; // 8KB
  __shared__ uint smComp[2049]; // 8KB + 1
  __shared__ int sm_extls[8];
  __shared__ uint sm_sums[8];

  unsigned char *smbComp = (unsigned char *)smComp;
  unsigned char *smbOut = (unsigned char *)smOut;

  int warpId = threadIdx.x / 32;
  int laneId = threadIdx.x & 31;

  // Read all the compressed data to shared memory
  for (int i = threadIdx.x; i < lcompw; i += 256)
  {
    smComp[i] = in[i];
  }
  if (threadIdx.x == 0)
  {
    smComp[2048] = 0xffffffff; // To rebuild 0xff strings
  }

  // Initialize the output to zero
  for (int i = threadIdx.x; i < 512; i += 256)
  {
    smOut[i] = make_uint4(0, 0, 0, 0);
  }

  __syncthreads();

  // Load the number of control codes, at the beginning of the data
  int nCodes = (smbComp[0] << 8) + smbComp[1];

  // Offset counters, only tracked on producer warp
  uint idxControl = lcompw * 4 - 1;
  uint idxExt = idxControl - nCodes;
  uint idxData = 2; // First 2 bytes for the number of codes
  uint idxOut = 0;

  // Loop on all the codes, 256 at a time
  for (int iCode = 0; iCode < nCodes; iCode += 256)
  {
    // ******* Prepare 256 new codes ********

    // Read one control code per thread, backward from the end of the array
    bool validCode = (iCode + threadIdx.x < nCodes);
    uint bcode = 0; // Default length for invalid codes = 0
    if (validCode)
    {
      bcode = smbComp[idxControl - threadIdx.x];
    }
    uint length = bcode & 0x1f;
    bool longCode = bcode & CTRL_LONG;
    bool dupCode = bcode & CTRL_DUP;
    bcode &= 0xc0;

    idxControl -= 256;

    // Find out which threads need to read an extension (long code length)
    uint maskBelow = (1 << laneId) - 1;
    uint extendedMask = __ballot_sync(~0, longCode);
    uint offBelow = __popc(extendedMask & maskBelow);
    // Write the number of extensions per warp in shared memory
    uint warpExtensions = __popc(extendedMask);
    if (laneId == 0)
    {
      sm_extls[warpId] = warpExtensions;
    }
    __syncthreads();

    // Prefix sum the counts per warp

    // Default value for coverity.
    // Real value is set in the if statement below
    // Only values from first 8 threads are used to compute prefsums.
    int prefsum = 0, total;
    if (laneId < 8)
    {
      prefsum = sm_extls[laneId];
    }
    prefixsum<8>(prefsum, total, laneId);
    prefsum = __shfl_sync(~0, prefsum, warpId - 1);
    if (warpId > 0)
    {
      offBelow += prefsum;
    }

    // Threads with a long code read the length extension
    if (longCode)
    {
      length = (length << 8) + smbComp[idxExt - offBelow];
    }

    // Adjust the extensions pointer based on the total number of extensions read
    idxExt -= total;

    // Adjust the length (+1 for non-duplicates, +3 for duplicates)
    // Invalid codes must keep a length of 0.
    if (validCode)
    {
      length += dupCode ? 3 : 1;
    }

    // Number of bytes read when processing this code
    uint lread = 0;
    if (!dupCode)
    {
      lread = length;
    }
    if (bcode == CTRL_DUP) // Duplicate but not DUP00 nor DUPFF.
    {
      lread = 1;
    }

    // Prefix sums of the number of bytes read and written (2 x 16bit at once)
    uint sums = lread * 65536 + length;
    uint sums_total;
    prefixsum_256(sums, sums_total, sm_sums, warpId, laneId);
    uint sumw = sums & 0xffff;
    uint sumr = sums >> 16;

    // Disabling DUP_00 codes (output already set to 0)
    if (bcode == CTRL_DUP00)
    {
      length = 0;
    }

    int offW = idxOut + sumw - length;
    // Increment is 1 only for non-duplicates
    int inc = (dupCode) ? 0 : 1;
    // Read offset is set to NOMINAL_BLOCK_SIZE for 0xff
    int offR = (bcode == CTRL_DUPFF) ? NOMINAL_BLOCK_SIZE : idxData + sumr - lread;

    // Update the offset counters
    idxData += sums_total >> 16;
    idxOut += sums_total & 0xffff;

    // ******** Process the 256 codes ********

    // Work collectively on the codes with long lengths
    bool longdata = (length >= 16);
    uint longmask = __ballot_sync(~0, longdata);
    uint pack1 = length * 65536 + inc;
    uint pack2 = offW * 65536 + offR;
#pragma unroll 1
    while (longmask)
    {
      // CLZ is cheaper than FFS -> higher threads first
      int i = __clz(longmask);
      longmask ^= 0x80000000 >> i;
      uint unpack1 = __shfl_sync(~0, pack1, 31 - i);
      uint unpack2 = __shfl_sync(~0, pack2, 31 - i);
      uint length = unpack1 >> 16;
      uint inc = unpack1 & 1;
      uint offW = unpack2 >> 16;
      uint offR = unpack2 & 0xffff;

#pragma unroll 1
      for (int j = 0; j < length; j += 32)
      {
        if (j + laneId < length)
        {
          smbOut[offW + j + laneId] = smbComp[offR + (j + laneId) * inc];
        }
      }
    }
    // Disable the long codes already processed
    if (longdata)
    {
      length = 0;
    }

    // For the short codes, each thread processes its own code
#pragma unroll 1
    for (int j = 0; j < length; j++)
    {
      smbOut[offW + j] = smbComp[offR + j * inc];
    }

  } // End loop icode

  __syncthreads();

  // Backward bit-plan transpose in shared memory
  transpose::bitPlanTransposeBack<sizeof(T)>(smOut);

  __syncthreads();

  // Backward quantize+format on the fly as we save into global memory
  if constexpr (compMode == BITCOMP_LOSSLESS)
  {
    loadStore::storeSharedToOutput<T, compMode, ifmt, false>(out, smOut, blockStart, blockEnd, aligned, delta);
  }
  else
  {
    if (utilities::zeroMantissaBits(delta) == delta)
    {
      loadStore::storeSharedToOutput<T, compMode, ifmt, true>(out, smOut, blockStart, blockEnd, aligned, delta);
    }
    else
    {
      loadStore::storeSharedToOutput<T, compMode, ifmt, false>(out, smOut, blockStart, blockEnd, aligned, delta);
    }
  }
}

} // namespace rle

} // namespace bitcomp
