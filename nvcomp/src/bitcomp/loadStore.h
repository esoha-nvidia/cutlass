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

#include <cuda_fp16.h>

#include "bitcomp_private.h"
#include "quantize.h"

#pragma once

namespace bitcomp
{

namespace loadStore
{

// *************************************************************************************************
// Load the input values to shared memory, pad with zeroes
// Optional: do an integer quantization for FP inputs (only 32-bit and 64-bit supported)
// Use 128-bit load/stores when properly alignned, otherwise use type native
// Signed integers can be stored in a custom format instead of 2s complement

template <typename T, bitcompMode_t compMode, bitcompIntFormat_t ifmt>
inline __device__ int
loadInputToShared(const void *__restrict__ input, uint4 *__restrict__ sm, uint blockBytes, bool aligned, T delta = 1.0)
{
  int overflow = 0;

  // 32-Byte union to view the same data as different formats
  union vector_32
  {
    uint4 v4[2]; // to generate 128-bit ld/st
    uint2 v2[4]; // to generate 64-bit ld/st
    half2 h2[8];
    short2 s2[8];
    ushort2 us2[8];
    float f[8];
    int i[8];
    uint ui[8];
    double d[4];
    long long l[4];
    unsigned long long ul[4];
    unsigned short us[16];
    short s[16];
    unsigned char uc[32];
    signed char c[32];
  } vec32;

  // *******************************************************************
  // Load coalesced into registers, using 128-bit instructions if possible
  if (aligned)
  {
    // Load 2 x 16 bytes per thread, default is zero for padding
    vec32.v4[0] = make_uint4(0, 0, 0, 0);
    vec32.v4[1] = make_uint4(0, 0, 0, 0);
    const uint4 *input_16 = reinterpret_cast<const uint4 *>(input);
    if (threadIdx.x < blockBytes / 16)
    {
      vec32.v4[0] = input_16[threadIdx.x];
    }
    if (threadIdx.x + 256 < blockBytes / 16)
    {
      vec32.v4[1] = input_16[threadIdx.x + 256];
    }
  }
  else
  {
    if (sizeof(T) == 8)
    {
      // Load 4 x 8 bytes per thread, default is zero for padding
      const uint2 *input_8 = reinterpret_cast<const uint2 *>(input);
      for (int i = 0; i < 4; i++)
      {
        vec32.v2[i] = make_uint2(0, 0);
        if (threadIdx.x + i * 256 < blockBytes / 8)
        {
          vec32.v2[i] = input_8[threadIdx.x + i * 256];
        }
      }
    }
    else if (sizeof(T) == 4)
    {
      // Load 8 x 4 bytes per thread, default is zero for padding
      const uint *input_4 = reinterpret_cast<const uint *>(input);
      for (int i = 0; i < 8; i++)
      {
        vec32.ui[i] = 0;
        if (threadIdx.x + i * 256 < blockBytes / 4)
        {
          vec32.ui[i] = input_4[threadIdx.x + i * 256];
        }
      }
    }
    else if (sizeof(T) == 2)
    {
      // Load 16 x 2 bytes per thread, default is zero for padding
      const short *input_2 = reinterpret_cast<const short *>(input);
      for (int i = 0; i < 16; i++)
      {
        vec32.s[i] = 0;
        if (threadIdx.x + i * 256 < blockBytes / 2)
        {
          vec32.s[i] = input_2[threadIdx.x + i * 256];
        }
      }
    }
    else if (sizeof(T) == 1)
    {
      const unsigned char *input_1 = reinterpret_cast<const unsigned char *>(input);
      // Load 32 x 1 bytes per thread, default is zero for padding
      for (int i = 0; i < 32; i++)
      {
        vec32.uc[i] = 0;
        if (threadIdx.x + i * 256 < blockBytes)
        {
          vec32.uc[i] = input_1[threadIdx.x + i * 256];
        }
      }
    }
  }

  // *******************************************************************
  // Optional: quantization from FP to integral types.
  // Caution: overflow tests must handle NaN properly.
  // To improve precision during quantizaton:
  //   For FP16, use FP32 to have enough bits.
  //   For FP32 and FP64, compute remainder which will be != 0 for large input values
  if (compMode == BITCOMP_LOSSY_FP_TO_SIGNED)
  {
    if (sizeof(T) == 2)
    {
      for (int i = 0; i < 8; i++)
      {
        vec32.s2[i] = quantize<short2>(vec32.h2[i], delta, overflow);
      }
    }
    if (sizeof(T) == 4)
    {
      for (int i = 0; i < 8; i++)
      {
        vec32.i[i] = quantize<int32_t>(vec32.f[i], delta, overflow);
      }
    }
    if (sizeof(T) == 8)
    {
      for (int i = 0; i < 4; i++)
      {
        vec32.l[i] = quantize<int64_t>(vec32.d[i], delta, overflow);
      }
    }
  }
  // For unsigned, negative values can't be accepted
  if (compMode == BITCOMP_LOSSY_FP_TO_UNSIGNED)
  {
    if (sizeof(T) == 2)
    {
      for (int i = 0; i < 8; i++)
      {
        vec32.us2[i] = quantize<ushort2>(vec32.h2[i], delta, overflow);
      }
    }
    if (sizeof(T) == 4)
    {
      for (int i = 0; i < 8; i++)
      {
        vec32.ui[i] = quantize<uint32_t>(vec32.f[i], delta, overflow);
      }
    }
    if (sizeof(T) == 8)
    {
      for (int i = 0; i < 4; i++)
      {
        vec32.ul[i] = quantize<uint64_t>(vec32.d[i], delta, overflow);
      }
    }
  }

  // *******************************************************************
  // Optional: Custom integer format, instead of 2s complement
  if (ifmt == BITCOMP_CUSTOM_INTEGER)
  {
    // Custom integer format: Positive numbers are unchanged, negative have a sign bit + absolute value - 1
    // E.g. with 8-bit data: 0x80 = -1, 0x81 = -2, 0xff = -128
    if (sizeof(T) == 1)
    {
      for (int i = 0; i < 32; i++)
      {
        if (vec32.c[i] < 0)
        {
          vec32.uc[i] = -(vec32.c[i] + 1) | 0x80;
        }
      }
    }
    if (sizeof(T) == 2)
    {
      for (int i = 0; i < 16; i++)
      {
        if (vec32.s[i] < 0)
        {
          vec32.us[i] = -(vec32.s[i] + 1) | 0x8000;
        }
      }
    }
    if (sizeof(T) == 4)
    {
      for (int i = 0; i < 8; i++)
      {
        if (vec32.i[i] < 0)
        {
          vec32.ui[i] = -(vec32.i[i] + 1) | (1 << 31);
        }
      }
    }
    if (sizeof(T) == 8)
    {
      for (int i = 0; i < 4; i++)
      {
        if (vec32.l[i] < 0)
        {
          vec32.ul[i] = -(vec32.l[i] + 1) | (1ULL << 63);
        }
      }
    }
  }

  // *******************************************************************
  // Store to shared memory, in the same order as it was in global memory
  if (aligned)
  {
    sm[threadIdx.x] = vec32.v4[0];
    sm[threadIdx.x + 256] = vec32.v4[1];
  }
  else
  {
    if (sizeof(T) == 8)
    {
      uint2 *smv = reinterpret_cast<uint2 *>(sm);
      for (int i = 0; i < 4; i++)
      {
        smv[threadIdx.x + i * 256] = vec32.v2[i];
      }
    }
    else if (sizeof(T) == 4)
    {
      uint *smv = reinterpret_cast<uint *>(sm);
      for (int i = 0; i < 8; i++)
      {
        smv[threadIdx.x + i * 256] = vec32.ui[i];
      }
    }
    else if (sizeof(T) == 2)
    {
      unsigned short *smv = reinterpret_cast<unsigned short *>(sm);
      for (int i = 0; i < 16; i++)
      {
        smv[threadIdx.x + i * 256] = vec32.us[i];
      }
    }
    else if (sizeof(T) == 1)
    {
      unsigned char *smv = reinterpret_cast<unsigned char *>(sm);
      for (int i = 0; i < 32; i++)
      {
        smv[threadIdx.x + i * 256] = vec32.uc[i];
      }
    }
  }

  return overflow;
}

// *************************************************************************************************

template <typename T, bitcompMode_t compMode, bitcompIntFormat_t ifmt, bool PowTwoDelta>
inline __device__ void storeSharedToOutput(
  void *__restrict__ output,
  const uint4 *__restrict__ sm,
  int blockStart,
  int blockEnd,
  bool aligned,
  T delta = 1.0
)
{

  // 32-Byte union to view the same data as different formats
  union vector_32
  {
    uint4 v4[2]; // to generate 128-bit ld/st
    uint2 v2[4]; // to generate 64-bit ld/st
    float f[8];
    half2 h2[8];
    short2 s2[8];
    ushort2 us2[8];
    int i[8];
    uint ui[8];
    double d[4];
    long long l[4];
    unsigned long long ul[4];
    unsigned short us[16];
    short s[16];
    unsigned char uc[32];
    signed char c[32];

#if defined(__CUDACC_VER_MAJOR__) && __CUDACC_VER_MAJOR__ < 11 && defined(_WINDOWS)
    // This is to work around that __half2 has a non-trivial default constructor.
    __device__ vector_32() {}
#endif
  } vec32;

  // *******************************************************************
  // Load from shared memory, in the same order as we'll write coalesced in global memory
  if (aligned)
  {
    vec32.v4[0] = sm[threadIdx.x];
    vec32.v4[1] = sm[threadIdx.x + 256];
  }
  else
  {
    if (sizeof(T) == 8)
    {
      const uint2 *smv = reinterpret_cast<const uint2 *>(sm);
      for (int i = 0; i < 4; i++)
      {
        vec32.v2[i] = smv[threadIdx.x + i * 256];
      }
    }
    else if (sizeof(T) == 4)
    {
      const uint *smv = reinterpret_cast<const uint *>(sm);
      for (int i = 0; i < 8; i++)
      {
        vec32.ui[i] = smv[threadIdx.x + i * 256];
      }
    }
    else if (sizeof(T) == 2)
    {
      const unsigned short *smv = reinterpret_cast<const unsigned short *>(sm);
      for (int i = 0; i < 16; i++)
      {
        vec32.us[i] = smv[threadIdx.x + i * 256];
      }
    }
    else if (sizeof(T) == 1)
    {
      const unsigned char *smv = reinterpret_cast<const unsigned char *>(sm);
      for (int i = 0; i < 32; i++)
      {
        vec32.uc[i] = smv[threadIdx.x + i * 256];
      }
    }
  }

  // *******************************************************************
  // Optional: Custom integer format, instead of 2s complement
  if (ifmt == BITCOMP_CUSTOM_INTEGER)
  {
    // Custom integer format: Positive numbers are unchanged, negative have a sign bit + absolute value - 1
    // E.g. with 8-bit data: 0x80 = -1, 0x81 = -2, 0xff = -128
    if (sizeof(T) == 1)
    {
      for (int i = 0; i < 32; i++)
      {
        if (vec32.uc[i] & 0x80)
        {
          vec32.uc[i] = 0x7f - vec32.uc[i];
        }
      }
    }
    if (sizeof(T) == 2)
    {
      for (int i = 0; i < 16; i++)
      {
        if (vec32.us[i] & 0x8000)
        {
          vec32.us[i] = 0x7fff - vec32.us[i];
        }
      }
    }
    if (sizeof(T) == 4)
    {
      for (int i = 0; i < 8; i++)
      {
        if (vec32.ui[i] & (1U << 31))
        {
          vec32.ui[i] = 0x7fffffff - vec32.ui[i];
        }
      }
    }
    if (sizeof(T) == 8)
    {
      for (int i = 0; i < 4; i++)
      {
        if (vec32.ul[i] & (1ULL << 63))
        {
          vec32.ul[i] = ~(1ULL << 63) - vec32.ul[i];
        }
      }
    }
  }

  // *******************************************************************
  // Optional: revert quantization from FP to integral types.
  if constexpr (compMode == BITCOMP_LOSSY_FP_TO_SIGNED)
  {
    if constexpr (sizeof(T) == 2)
    {
      for (int i = 0; i < 8; i++)
      {
        vec32.h2[i] = dequantize<short2>(vec32.s2[i], delta);
      }
    }
    if constexpr (sizeof(T) == 4)
    {
      for (int i = 0; i < 8; i++)
      {
        vec32.f[i] = dequantize<int32_t, PowTwoDelta>(vec32.i[i], delta);
      }
    }
    if constexpr (sizeof(T) == 8)
    {
      for (int i = 0; i < 4; i++)
      {
        vec32.d[i] = dequantize<int64_t, PowTwoDelta>(vec32.l[i], delta);
      }
    }
  }
  if constexpr (compMode == BITCOMP_LOSSY_FP_TO_UNSIGNED)
  {
    if constexpr (sizeof(T) == 2)
    {
      for (int i = 0; i < 8; i++)
      {
        vec32.h2[i] = dequantize<ushort2>(vec32.us2[i], delta);
      }
    }
    if constexpr (sizeof(T) == 4)
    {
      for (int i = 0; i < 8; i++)
      {
        vec32.f[i] = dequantize<uint32_t, PowTwoDelta>(vec32.ui[i], delta);
      }
    }
    if constexpr (sizeof(T) == 8)
    {
      for (int i = 0; i < 4; i++)
      {
        vec32.d[i] = dequantize<uint64_t, PowTwoDelta>(vec32.ul[i], delta);
      }
    }
  }

  // *******************************************************************
  // Store coalesced to global memory, using 128-bit instructions if possible
  if (aligned)
  {
    // Store 2 x 16 bytes per thread
    uint4 *output_16 = reinterpret_cast<uint4 *>(output);
    for (int i = 0; i < 2; i++)
    {
      int off = threadIdx.x + i * 256;
      if (16 * off >= blockStart && 16 * off < blockEnd)
      {
        output_16[off] = vec32.v4[i];
      }
    }
  }
  else
  {
    if (sizeof(T) == 8)
    {
      // Store 4 x 8 bytes per thread
      uint2 *output_8 = reinterpret_cast<uint2 *>(output);
      for (int i = 0; i < 4; i++)
      {
        int off = threadIdx.x + i * 256;
        if (8 * off >= blockStart && 8 * off < blockEnd)
        {
          output_8[off] = vec32.v2[i];
        }
      }
    }
    else if (sizeof(T) == 4)
    {
      // Store 8 x 4 bytes per thread
      uint *output_4 = reinterpret_cast<uint *>(output);
      for (int i = 0; i < 8; i++)
      {
        int off = threadIdx.x + i * 256;
        if (4 * off >= blockStart && 4 * off < blockEnd)
        {
          output_4[off] = vec32.ui[i];
        }
      }
    }
    else if (sizeof(T) == 2)
    {
      // Store 16 x 2 bytes per thread
      short *output_2 = reinterpret_cast<short *>(output);
      for (int i = 0; i < 16; i++)
      {
        int off = threadIdx.x + i * 256;
        if (2 * off >= blockStart && 2 * off < blockEnd)
        {
          output_2[off] = vec32.s[i];
        }
      }
    }
    else if (sizeof(T) == 1)
    {
      unsigned char *output_1 = reinterpret_cast<unsigned char *>(output);
      // Store 32 x 1 bytes per thread
      for (int i = 0; i < 32; i++)
      {
        int off = threadIdx.x + i * 256;
        if (off >= blockStart && off < blockEnd)
        {
          output_1[threadIdx.x + i * 256] = vec32.uc[i];
        }
      }
    }
  }
}

// *************************************************************************************************
// Save incompressible data (up to 8KB) to the output. The output is 32-bit aligned,
// can't make any assumptions on larger alignment

template <typename T>
inline __device__ void
saveIncompressibleBlock(const void *__restrict__ input, uint *__restrict__ output, int nbytes, bool aligned)
{
  if (aligned || sizeof(T) >= 4)
  {
    {
      const uint *in = reinterpret_cast<const uint *>(input);
      for (int i = 0; i < 8; i++)
      {
        if (threadIdx.x + i * 256 < nbytes / 4)
        {
          output[threadIdx.x + i * 256] = in[threadIdx.x + i * 256];
        }
      }
    }
  }
  else
  {
    if (sizeof(T) == 2)
    {
      const short *in = reinterpret_cast<const short *>(input);
      short *out = reinterpret_cast<short *>(output);
      for (int i = 0; i < 16; i++)
      {
        if (threadIdx.x + i * 256 < nbytes / 2)
        {
          out[threadIdx.x + i * 256] = in[threadIdx.x + i * 256];
        }
      }
    }
    else if (sizeof(T) == 1)
    {
      const char *in = reinterpret_cast<const char *>(input);
      char *out = reinterpret_cast<char *>(output);
      for (int i = 0; i < 32; i++)
      {
        if (threadIdx.x + i * 256 < nbytes)
        {
          out[threadIdx.x + i * 256] = in[threadIdx.x + i * 256];
        }
      }
    }
  }
}

// *************************************************************************************************
// Restore incompressible data (up to 8KB) to the output. The input is 32-bit aligned.
// Supports partial decompression. Output is mapped to the uncompressed data from start to end.

template <typename T>
inline __device__ void restoreIncompressibleBlock(
  const uint *__restrict__ input,
  void *__restrict__ output,
  int blockStart,
  int blockEnd,
  bool aligned
)
{
  if (aligned || sizeof(T) >= 4)
  {
    uint *out = reinterpret_cast<uint *>(output);
    for (int i = 0; i < 8; i++)
    {
      int off = threadIdx.x + i * 256;
      if (4 * off >= blockStart && 4 * off < blockEnd)
      {
        out[off] = input[off];
      }
    }
  }
  else
  {
    if (sizeof(T) == 2)
    {
      const short *in = reinterpret_cast<const short *>(input);
      short *out = reinterpret_cast<short *>(output);
      for (int i = 0; i < 16; i++)
      {
        int off = threadIdx.x + i * 256;
        if (2 * off >= blockStart && 2 * off < blockEnd)
        {
          out[off] = in[off];
        }
      }
    }
    else if (sizeof(T) == 1)
    {
      const char *in = reinterpret_cast<const char *>(input);
      char *out = reinterpret_cast<char *>(output);
      for (int i = 0; i < 32; i++)
      {
        int off = threadIdx.x + i * 256;
        if (off >= blockStart && off < blockEnd)
        {
          out[off] = in[off];
        }
      }
    }
  }
}

// *************************************************************************************************
// Set an uncompressed block to zero

template <typename T>
inline __device__ void zeroOutputBlock(void *output, int blockStart, int blockEnd, bool aligned)
{
  if (aligned)
  {
    int4 *out = reinterpret_cast<int4 *>(output);
    for (int i = 0; i < 2; i++)
    {
      int off = threadIdx.x + i * 256;
      if (16 * off >= blockStart && 16 * off < blockEnd)
      {
        out[off] = make_int4(0, 0, 0, 0);
      }
    }
  }
  else
  {
    if (sizeof(T) == 8)
    {
      uint64 *out = reinterpret_cast<uint64 *>(output);
      for (int i = 0; i < 4; i++)
      {
        int off = threadIdx.x + i * 256;
        if (8 * off >= blockStart && 8 * off < blockEnd)
        {
          out[off] = 0;
        }
      }
    }
    else if (sizeof(T) == 4)
    {
      uint *out = reinterpret_cast<uint *>(output);
      for (int i = 0; i < 8; i++)
      {
        int off = threadIdx.x + i * 256;
        if (4 * off >= blockStart && 4 * off < blockEnd)
        {
          out[off] = 0;
        }
      }
    }
    else if (sizeof(T) == 2)
    {
      short *out = reinterpret_cast<short *>(output);
      for (int i = 0; i < 16; i++)
      {
        int off = threadIdx.x + i * 256;
        if (2 * off >= blockStart && 2 * off < blockEnd)
        {
          out[off] = 0;
        }
      }
    }
    else if (sizeof(T) == 1)
    {
      char *out = reinterpret_cast<char *>(output);
      for (int i = 0; i < 32; i++)
      {
        int off = threadIdx.x + i * 256;
        if (off >= blockStart && off < blockEnd)
        {
          out[off] = 0;
        }
      }
    }
  }
}

// *************************************************************************************************
} // namespace loadStore
} // namespace bitcomp
