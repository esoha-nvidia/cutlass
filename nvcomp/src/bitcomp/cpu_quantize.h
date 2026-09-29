/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026 NVIDIA CORPORATION & AFFILIATES.
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

#include <cuda_fp16.h>

#include "bitcomp_private.h"
#include "quantize.h"

#include <math.h>

namespace bitcomp
{

// *************************************************************************************************
// FP->INT quantization, with detection of overflow, 8KB buffer
// Caution: overflow tests must handle NaN or Inf properly.
// To improve precision during quantizaton:
// For FP32 and FP64, compute remainder which will be != 0 for large input values

template <bitcompMode_t compMode, typename T>
inline int cpu_quantize(unsigned char *inout, T delta)
{
  int overflow = 0;
  using InType = InputSelector<T>;
  using OutType = OutputSelector<T, compMode>;

  InType *input = reinterpret_cast<InType *>(inout);
  OutType *output = reinterpret_cast<OutType *>(inout);

  static_assert(
    NOMINAL_BLOCK_SIZE % sizeof(InType) == 0,
    "NOMINAL_BLOCK_SIZE must be a multiple of the input type size"
  );
  constexpr int N = NOMINAL_BLOCK_SIZE / sizeof(InType);
  for (int i = 0; i < N; i++)
  {
    output[i] = quantize<OutType>(input[i], delta, overflow);
  }
  return overflow;
}

// *************************************************************************************************
// INT->FP backward quantization

template <typename T, bitcompMode_t compMode, bool PowTwoDelta>
inline void quantizeBwd(unsigned char *inout, T delta)
{
  using OrigT = InputSelector<T>;
  using QuantT = OutputSelector<T, compMode>;
  OrigT *orig = reinterpret_cast<OrigT *>(inout);
  QuantT *quantized = reinterpret_cast<QuantT *>(inout);
  static_assert(
    NOMINAL_BLOCK_SIZE % sizeof(OrigT) == 0,
    "NOMINAL_BLOCK_SIZE must be a multiple of the output type size"
  );
  constexpr int N = NOMINAL_BLOCK_SIZE / sizeof(OrigT);
  for (int i = 0; i < N; i++)
  {
    orig[i] = dequantize<QuantT, PowTwoDelta>(quantized[i], delta);
  }
}

// *************************************************************************************************
// Custom integer format, 8KB buffer

template <int elemBytes, bitcompIntFormat_t ifmt>
inline void customizeInteger(unsigned char *inout)
{
  // Custom integer format, instead of 2s complement
  if (ifmt == BITCOMP_CUSTOM_INTEGER)
  {
    // Custom integer format: Positive numbers are unchanged, negative have a sign bit + absolute value - 1
    // E.g. with 8-bit data: 0x80 = -1, 0x81 = -2, 0xff = -128
    if (elemBytes == 1)
    {
      signed char *buf = (signed char *)inout;
      for (int i = 0; i < NOMINAL_BLOCK_SIZE; i++)
      {
        if (buf[i] < 0)
        {
          buf[i] = -(buf[i] + 1) | 0x80;
        }
      }
    }
    if (elemBytes == 2)
    {
      short *buf = (short *)inout;
      for (int i = 0; i < 4096; i++)
      {
        if (buf[i] < 0)
        {
          buf[i] = -(buf[i] + 1) | 0x8000;
        }
      }
    }
    if (elemBytes == 4)
    {
      int *buf = (int *)inout;
      for (int i = 0; i < 2048; i++)
      {
        if (buf[i] < 0)
        {
          buf[i] = -(buf[i] + 1) | (1 << 31);
        }
      }
    }
    if (elemBytes == 8)
    {
      long long *buf = (long long *)inout;
      for (int i = 0; i < 1024; i++)
      {
        if (buf[i] < 0)
        {
          buf[i] = -(buf[i] + 1) | (1ULL << 63);
        }
      }
    }
  }
}

// *************************************************************************************************
// Restore 2s-complement integer format
template <int elemBytes, bitcompIntFormat_t ifmt>
inline void customizeIntegerBwd(unsigned char *inout)
{
  // Custom integer format, instead of 2s complement
  if (ifmt == BITCOMP_CUSTOM_INTEGER)
  {
    // Custom integer format: Positive numbers are unchanged, negative have a sign bit + absolute value - 1
    // E.g. with 8-bit data: 0x80 = -1, 0x81 = -2, 0xff = -128
    if (elemBytes == 1)
    {
      unsigned char *buf = (unsigned char *)inout;
      for (int i = 0; i < NOMINAL_BLOCK_SIZE; i++)
      {
        if (buf[i] & 0x80)
        {
          buf[i] = 0x7f - buf[i];
        }
      }
    }
    if (elemBytes == 2)
    {
      unsigned short *buf = (unsigned short *)inout;
      for (int i = 0; i < 4096; i++)
      {
        if (buf[i] & 0x8000)
        {
          buf[i] = 0x7fff - buf[i];
        }
      }
    }
    if (elemBytes == 4)
    {
      uint *buf = (uint *)inout;
      for (int i = 0; i < 2048; i++)
      {
        if (buf[i] & (1U << 31))
        {
          buf[i] = 0x7fffffff - buf[i];
        }
      }
    }
    if (elemBytes == 8)
    {
      unsigned long long *buf = (unsigned long long *)inout;
      for (int i = 0; i < 1024; i++)
      {
        if (buf[i] & (1ULL << 63))
        {
          buf[i] = ~(1ULL << 63) - buf[i];
        }
      }
    }
  }
}

} // namespace bitcomp