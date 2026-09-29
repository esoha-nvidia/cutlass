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

#pragma once

namespace bitcomp
{

namespace transpose
{

// *************************************************************************************************
// msb_first = true implies lsb_first = false
template <bool msb_first = true>
inline __device__ void shiftBits(uint4 &v0, uint4 &v1)
{
  if constexpr (msb_first)
  {
    v0.x <<= 1;
    v0.y <<= 1;
    v0.z <<= 1;
    v0.w <<= 1;
    v1.x <<= 1;
    v1.y <<= 1;
    v1.z <<= 1;
    v1.w <<= 1;
  }
  else
  {
    v0.x >>= 1;
    v0.y >>= 1;
    v0.z >>= 1;
    v0.w >>= 1;
    v1.x >>= 1;
    v1.y >>= 1;
    v1.z >>= 1;
    v1.w >>= 1;
  }
}

/**
     * @brief Extract bits from 8 uint4 values (32 bytes) into a single uint.
     * @param x0, x1, x2, x3, x4, x5, x6, x7 The 8 uint4 values to extract bits from.
     * @param msb_first Whether to extract bits from the most significant bit first and 
     *                  the least significant bit last.
     * @return The extracted bits.
     */
template <bool msb_first = true>
inline __device__ uint extractBits(uint x0, uint x1, uint x2, uint x3, uint x4, uint x5, uint x6, uint x7)
{
  uint result = 0;
  if constexpr (msb_first)
  {
    constexpr uint msb_mask = 0x80808080;
    result = ((x0 & msb_mask) >> 7) | ((x1 & msb_mask) >> 6) | ((x2 & msb_mask) >> 5) | ((x3 & msb_mask) >> 4) |
             (x4 & msb_mask) >> 3 | (x5 & msb_mask) >> 2 | (x6 & msb_mask) >> 1 | (x7 & msb_mask);
  }
  else
  {
    constexpr uint lsb_mask = 0x01010101;
    result = (x0 & lsb_mask) | ((x1 & lsb_mask) << 1) | ((x2 & lsb_mask) << 2) | ((x3 & lsb_mask) << 3) |
             (x4 & lsb_mask) << 4 | ((x5 & lsb_mask) << 5) | ((x6 & lsb_mask) << 6) | ((x7 & lsb_mask) << 7);
  }
  return result;
}

/**
     * @brief Set bits from a single uint into 8 uint4 values (32 bytes).
     * @param x0, x1, x2, x3, x4, x5, x6, x7 The 8 uint4 values to set bits into.
     * @param vb The uint value to set bits from.
     * @param bitPlane The bit plane to set bits from.
     * @param msb_first Whether to set bits from the most significant bit first and 
     *                  the least significant bit last.
     */
template <bool msb_first = true>
inline __device__ void
setBitsFromBitPlane(uint &x0, uint &x1, uint &x2, uint &x3, uint &x4, uint &x5, uint &x6, uint &x7, uint vb, int bitPlane)
{
  if constexpr (msb_first)
  {
    x0 |= ((vb & 0x01010101) << 7) >> bitPlane;
    x1 |= ((vb & 0x02020202) << 6) >> bitPlane;
    x2 |= ((vb & 0x04040404) << 5) >> bitPlane;
    x3 |= ((vb & 0x08080808) << 4) >> bitPlane;
    x4 |= ((vb & 0x10101010) << 3) >> bitPlane;
    x5 |= ((vb & 0x20202020) << 2) >> bitPlane;
    x6 |= ((vb & 0x40404040) << 1) >> bitPlane;
    x7 |= ((vb & 0x80808080) << 0) >> bitPlane;
  }
  else
  {
    x0 |= ((vb & 0x01010101) >> 0) << bitPlane;
    x1 |= ((vb & 0x02020202) >> 1) << bitPlane;
    x2 |= ((vb & 0x04040404) >> 2) << bitPlane;
    x3 |= ((vb & 0x08080808) >> 3) << bitPlane;
    x4 |= ((vb & 0x10101010) >> 4) << bitPlane;
    x5 |= ((vb & 0x20202020) >> 5) << bitPlane;
    x6 |= ((vb & 0x40404040) >> 6) << bitPlane;
    x7 |= ((vb & 0x80808080) >> 7) << bitPlane;
  }
}

// *************************************************************************************************
// Bit-transpose the data for 1,2,4,8 byte data-types only
// Other data sizes should be done as a combination of byte transpose first.
// Optional: output can be padded with 4 bytes every 32 bytes,
//   to allow subsequent load of 32 consecutive bytes without bank conflicts.

// Update (04/2020): The bits of the very first value end up in bit 0 of the byte stream,
// instead of the bit 7 previously. This is a more natural order and will be faster on CPU.

// msb_first = true implies lsb_first = false
template <int elemBytes, bool pad32_4, bool msb_first = true, bool use_variable_worker_threads = false>
__device__ void bitPlanTranspose(uint4 *sm, int worker_threads = 256)
{
  // Load 256 bits of contigous data (not padded)
  uint4 v0 = sm[2 * threadIdx.x];
  uint4 v1 = sm[2 * threadIdx.x + 1];

  __syncthreads();

  if constexpr (elemBytes == 1)
  {
    // Reorganize the data to process 4 bits ((7,7,7,7), then (6,6,6,6), ..., (0,0,0,0)) at a time, for 8 values
    uint tmp[8];
    tmp[0] = __byte_perm(v0.x, v1.x, 0x0145);
    tmp[1] = __byte_perm(v0.x, v1.x, 0x2367);
    tmp[2] = __byte_perm(v0.y, v1.y, 0x0145);
    tmp[3] = __byte_perm(v0.y, v1.y, 0x2367);
    tmp[4] = __byte_perm(v0.z, v1.z, 0x0145);
    tmp[5] = __byte_perm(v0.z, v1.z, 0x2367);
    tmp[6] = __byte_perm(v0.w, v1.w, 0x0145);
    tmp[7] = __byte_perm(v0.w, v1.w, 0x2367);
    v0.x = __byte_perm(tmp[0], tmp[4], 0x5173);
    v0.y = __byte_perm(tmp[0], tmp[4], 0x4062);
    v0.z = __byte_perm(tmp[1], tmp[5], 0x5173);
    v0.w = __byte_perm(tmp[1], tmp[5], 0x4062);
    v1.x = __byte_perm(tmp[2], tmp[6], 0x5173);
    v1.y = __byte_perm(tmp[2], tmp[6], 0x4062);
    v1.z = __byte_perm(tmp[3], tmp[7], 0x5173);
    v1.w = __byte_perm(tmp[3], tmp[7], 0x4062);
    uint index = threadIdx.x;
    uint *smi = (uint *)sm;
    if constexpr (pad32_4)
    {
      index += threadIdx.x / 8;
    }
    for (int i = 0; i < 8; i++)
    {
      // vb contains the bit i of 32 bytes.
      uint vb = extractBits<msb_first>(v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w);
      // Shift all the input values to process the next bit
      shiftBits<msb_first>(v0, v1);
      // Write the bit i of 32 bytes to shared memory
      // threadIdx.x = index will write to smi[index].
      // Overall worker_threads will need worker_threads*32 bits
      // for bit i of the input data = "worker_threads" uint
      // elements of smi.
      if constexpr (use_variable_worker_threads)
      {
        if (threadIdx.x < worker_threads)
        {
          smi[index] = vb;
        }
      }
      else
      {
        smi[index] = vb;
      }

      if constexpr (pad32_4)
      {
        index += worker_threads + 32;
      }
      else
      {
        // The addition of worker_threads corresponds to the number of threads in the CTA
        // which are actively processing the data.
        // Each thread is processing 32 bytes of data.
        index += worker_threads;
      }
    }
  }

  if constexpr (elemBytes == 2 || elemBytes == 4)
  {

    int offset[3] = {0, 0, 0};
    if constexpr (pad32_4)
    {
      offset[0] = (worker_threads + 32) * 8;
      offset[1] = (worker_threads + 32) * 16;
      offset[2] = (worker_threads + 32) * 24;
    }
    else
    {
      offset[0] = worker_threads * 8;
      offset[1] = worker_threads * 16;
      offset[2] = worker_threads * 24;
    }
    int index_update = worker_threads;
    if constexpr (pad32_4)
    {
      index_update = worker_threads + 32;
    }

    if constexpr (elemBytes == 2)
    {
      uint4 w0 = v0;
      // Reorganize the data to process 4 bits ((15,15,7,7), then (14,14,6,6), ..., (8,8,0,0)) at a time, for 8 values
      v0.x = __byte_perm(w0.x, v1.x, 0x5140);
      v0.y = __byte_perm(w0.y, v1.y, 0x5140);
      v0.z = __byte_perm(w0.z, v1.z, 0x5140);
      v0.w = __byte_perm(w0.w, v1.w, 0x5140);
      v1.x = __byte_perm(w0.x, v1.x, 0x7362);
      v1.y = __byte_perm(w0.y, v1.y, 0x7362);
      v1.z = __byte_perm(w0.z, v1.z, 0x7362);
      v1.w = __byte_perm(w0.w, v1.w, 0x7362);
      uint index = threadIdx.x;
      if constexpr (pad32_4)
      {
        index += (threadIdx.x / 16) * 2;
      }
      for (int i = 0; i < 8; i++)
      {
        uint vb = extractBits<msb_first>(v0.x, v1.x, v0.y, v1.y, v0.z, v1.z, v0.w, v1.w);
        // Shift all the input values to process the next bit
        shiftBits<msb_first>(v0, v1);
        unsigned short *smh = (unsigned short *)sm;
        if constexpr (use_variable_worker_threads)
        {
          if (threadIdx.x < worker_threads)
          {
            if constexpr (msb_first)
            {
              smh[index] = vb >> 16;
              smh[index + offset[0]] = vb;
            }
            else
            {
              smh[index] = vb;
              smh[index + offset[0]] = vb >> 16;
            }
          }
        }
        else
        {
          if constexpr (msb_first)
          {
            smh[index] = vb >> 16;
            smh[index + offset[0]] = vb;
          }
          else
          {
            smh[index] = vb;
            smh[index + offset[0]] = vb >> 16;
          }
        } // else use_variable_worker_threads = false
        index += index_update;
      }
    }

    if constexpr (elemBytes == 4)
    {
      uint index = threadIdx.x;
      if constexpr (pad32_4)
      {
        index += (threadIdx.x / 32) * 4;
      }
      for (int i = 0; i < 8; i++)
      {
        uint vb = extractBits<msb_first>(v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w);
        // Process 4 bits ((31,23,15,7), then (30,22,14,6), ..., (24,16,8,0)) at a time, for 8 values
        // Shift all the input values to process the next bit
        shiftBits<msb_first>(v0, v1);
        // Store the 4 bytes to shared memory
        unsigned char *smb = (unsigned char *)sm;
        if constexpr (use_variable_worker_threads)
        {
          if (threadIdx.x < worker_threads)
          {
            if constexpr (msb_first)
            {
              smb[index] = vb >> 24;
              smb[index + offset[0]] = vb >> 16;
              smb[index + offset[1]] = vb >> 8;
              smb[index + offset[2]] = vb;
            }
            else
            {
              smb[index] = vb;
              smb[index + offset[0]] = vb >> 8;
              smb[index + offset[1]] = vb >> 16;
              smb[index + offset[2]] = vb >> 24;
            }
          }
        }
        else
        {
          if constexpr (msb_first)
          {
            smb[index] = vb >> 24;
            smb[index + offset[0]] = vb >> 16;
            smb[index + offset[1]] = vb >> 8;
            smb[index + offset[2]] = vb;
          }
          else
          {
            smb[index] = vb;
            smb[index + offset[0]] = vb >> 8;
            smb[index + offset[1]] = vb >> 16;
            smb[index + offset[2]] = vb >> 24;
          }
        } // else use_variable_worker_threads = false
        index += index_update;
      }
    }
  } // if constexpr (elemBytes == 2 || elemBytes == 4)

  // This part is untouched from its original version since LZ4 does not support 8 byte elements
  if constexpr (elemBytes == 8)
  {
    uint mask_lh = (threadIdx.x & 1) ? 0xf0f0f0f0 : 0x0f0f0f0f;
    uint index = threadIdx.x / 2 + (threadIdx.x & 1) * 4096;
    if constexpr (pad32_4)
    {
      index += (index / 32) * 4;
    }
    for (int i = 0; i < 8; i++)
    {
      // Process 8 bits (63,55,47,39,31,23,15,7, ..., (56,48,40,32,24,16,8,0)) at once for 4 values
      uint vb = extractBits<true>(v0.y, v0.w, v1.y, v1.w, v0.x, v0.z, v1.x, v1.z);
      // Shift all the input values to process the next bit
      shiftBits<true>(v0, v1);
      // Exchange 4 bits with neighbor to write full bytes
      uint vbn = __shfl_xor_sync(0xffffffff, vb, 1);
      vbn &= mask_lh;
      if (threadIdx.x & 1)
      {
        vbn >>= 4;
      }
      else
      {
        vbn <<= 4;
      }
      vb = (vb & mask_lh) | vbn;
      // Store the 4 bytes to shared memory (2-way bank conflict)
      unsigned char *smb = (unsigned char *)sm;
      if constexpr (pad32_4)
      {
        smb[index] = vb >> 24;
        smb[index + 1152] = vb >> 16;
        smb[index + 2304] = vb >> 8;
        smb[index + 3456] = vb;
        index += 144;
      }
      else
      {
        smb[index] = vb >> 24;
        smb[index + 1024] = vb >> 16;
        smb[index + 2048] = vb >> 8;
        smb[index + 3072] = vb;
        index += 128;
      }
    }
  }
}

// *************************************************************************************************
// Backward bit-transpose the data for 8, 16, 32, 64 bits only.

// msb_first = true implies lsb_first = false
template <int elemBytes, bool msb_first = true>
__device__ void bitPlanTransposeBack(uint4 *sm, int worker_threads = 256)
{
  uint4 v0, v1;

  if constexpr (elemBytes == 1)
  {
    v0 = make_uint4(0, 0, 0, 0);
    v1 = make_uint4(0, 0, 0, 0);
    uint *smi = (uint *)sm;
    for (int i = 0; i < 8; i++)
    {
      uint vb = smi[threadIdx.x + i * worker_threads];
      setBitsFromBitPlane<msb_first>(v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w, vb, i);
    }

    uint tmp[8];
    tmp[0] = __byte_perm(v0.x, v0.y, 0x0426);
    tmp[1] = __byte_perm(v0.z, v0.w, 0x0426);
    tmp[2] = __byte_perm(v1.x, v1.y, 0x0426);
    tmp[3] = __byte_perm(v1.z, v1.w, 0x0426);
    tmp[4] = __byte_perm(v0.x, v0.y, 0x1537);
    tmp[5] = __byte_perm(v0.z, v0.w, 0x1537);
    tmp[6] = __byte_perm(v1.x, v1.y, 0x1537);
    tmp[7] = __byte_perm(v1.z, v1.w, 0x1537);

    v0.x = (__byte_perm(tmp[0], tmp[1], 0x6723));
    v0.y = (__byte_perm(tmp[2], tmp[3], 0x6723));
    v0.z = (__byte_perm(tmp[4], tmp[5], 0x6723));
    v0.w = (__byte_perm(tmp[6], tmp[7], 0x6723));
    v1.x = (__byte_perm(tmp[0], tmp[1], 0x4501));
    v1.y = (__byte_perm(tmp[2], tmp[3], 0x4501));
    v1.z = (__byte_perm(tmp[4], tmp[5], 0x4501));
    v1.w = (__byte_perm(tmp[6], tmp[7], 0x4501));
  }

  if constexpr (elemBytes == 2)
  {
    uint4 w0 = make_uint4(0, 0, 0, 0);
    v1 = make_uint4(0, 0, 0, 0);
    unsigned short *smh = (unsigned short *)sm;
    const int worker_thread_times_8 = worker_threads * 8;
    for (int i = 0; i < 8; i++)
    {
      uint vb = 0;
      if constexpr (msb_first)
      {
        vb = (smh[threadIdx.x + i * worker_threads] << 16) |
             (smh[threadIdx.x + i * worker_threads + worker_thread_times_8]);
      }
      else
      {
        vb = (smh[threadIdx.x + i * worker_threads + worker_thread_times_8] << 16) |
             (smh[threadIdx.x + i * worker_threads]);
      }
      setBitsFromBitPlane<msb_first>(w0.x, v1.x, w0.y, v1.y, w0.z, v1.z, w0.w, v1.w, vb, i);
    }

    v0.x = __byte_perm(w0.x, v1.x, 0x6420);
    v0.y = __byte_perm(w0.y, v1.y, 0x6420);
    v0.z = __byte_perm(w0.z, v1.z, 0x6420);
    v0.w = __byte_perm(w0.w, v1.w, 0x6420);
    v1.x = __byte_perm(w0.x, v1.x, 0x7531);
    v1.y = __byte_perm(w0.y, v1.y, 0x7531);
    v1.z = __byte_perm(w0.z, v1.z, 0x7531);
    v1.w = __byte_perm(w0.w, v1.w, 0x7531);
  }

  if constexpr (elemBytes == 4)
  {
    v0 = make_uint4(0, 0, 0, 0);
    v1 = make_uint4(0, 0, 0, 0);
    const int worker_thread_times_8 = worker_threads * 8;
    const int worker_thread_times_16 = worker_threads * 16;
    const int worker_thread_times_24 = worker_threads * 24;
    unsigned char *smb = (unsigned char *)sm;
    for (int i = 0; i < 8; i++)
    {
      uint vb = (smb[threadIdx.x + i * worker_threads] << 24) |
                (smb[threadIdx.x + i * worker_threads + worker_thread_times_8] << 16) |
                (smb[threadIdx.x + i * worker_threads + worker_thread_times_16] << 8) |
                smb[threadIdx.x + i * worker_threads + worker_thread_times_24];
      setBitsFromBitPlane<true>(v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w, vb, i);
    }
    if constexpr (!msb_first)
    {
      v0.x = __brev(v0.x);
      v0.y = __brev(v0.y);
      v0.z = __brev(v0.z);
      v0.w = __brev(v0.w);
      v1.x = __brev(v1.x);
      v1.y = __brev(v1.y);
      v1.z = __brev(v1.z);
      v1.w = __brev(v1.w);
    }
  }

  // This has not been tested to work with msb_first = false
  if constexpr (elemBytes == 8)
  {
    v0 = make_uint4(0, 0, 0, 0);
    v1 = make_uint4(0, 0, 0, 0);
    uint mask_lh = (threadIdx.x & 1) ? 0xf0f0f0f0 : 0x0f0f0f0f;
    unsigned char *smb = ((unsigned char *)sm) + (threadIdx.x / 2) + (threadIdx.x & 1) * 4096;
    for (int i = 0; i < 8; i++)
    {
      uint vb = (smb[i * 128] << 24) | (smb[i * 128 + 1024] << 16) | (smb[i * 128 + 2048] << 8) | smb[i * 128 + 3072];
      // Exchange 4 bits with neighbor
      uint vbn = __shfl_xor_sync(0xffffffff, vb, 1) & mask_lh;
      if (threadIdx.x & 1)
      {
        vbn >>= 4;
      }
      else
      {
        vbn <<= 4;
      }
      vb = (vb & mask_lh) | vbn;
      setBitsFromBitPlane<true>(v0.y, v0.w, v1.y, v1.w, v0.x, v0.z, v1.x, v1.z, vb, i);
    }
  }

  __syncthreads();

  // Store 256 bits of contigous data
  if (threadIdx.x < worker_threads)
  {
    sm[2 * threadIdx.x] = v0;
    sm[2 * threadIdx.x + 1] = v1;
  }
}

// *************************************************************************************************
// Transpose bytes of data for which the structure size is not 1, 2, 4 or 8 bytes.

/*
      __device__ inline void bitPlanTransposeGeneric (uint *sm, int bytes) {

      }
    */

} // namespace transpose
} // namespace bitcomp
