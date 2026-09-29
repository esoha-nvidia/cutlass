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

#include "bitcomp_private.h"
#include "cpu_transpose.h"
#include "header.h"

#include <stdio.h>

#ifdef __x86_64
#include <immintrin.h>
#endif
#ifdef _MSC_VER
#include <intrin.h>
#endif

#pragma once

namespace bitcomp
{

namespace zbm
{

// CPU encoder working on an 8KB buffer
#ifdef __x86_64
__attribute__((target("avx2")))
#endif
inline int
encoder(const unsigned char *bufin, unsigned char *bufout)
{
  uint mask0;

#ifdef __x86_64
  alignas(32) unsigned char mask2[1024];
  alignas(32) uint mask1[32];

  __m256i vzero = _mm256_setzero_si256();
  __m256i vff = _mm256_set1_epi32(0xffffffff);
  // Build mask2 which is a bitmap of the non-zero input bytes
  for (int i = 0; i < NOMINAL_BLOCK_SIZE; i += 32)
  {
    __m256i v = _mm256_load_si256((__m256i *)(bufin + i));
    __m256i z = _mm256_xor_si256(_mm256_cmpeq_epi8(v, vzero), vff);
    ((uint *)mask2)[i / 32] = _mm256_movemask_epi8(z);
  }
  // Build mask1 which is a bitmap of the non-zero bytes in mask2
  for (int i = 0; i < 1024; i += 32)
  {
    __m256i v = _mm256_load_si256((__m256i *)(mask2 + i));
    __m256i z = _mm256_xor_si256(_mm256_cmpeq_epi8(v, vzero), vff);
    mask1[i / 32] = _mm256_movemask_epi8(z);
  }
#else
  alignas(16) unsigned char mask2[1024];
  alignas(16) uint mask1[32];

  // Build mask2 which is a bitmap of the non-zero input bytes
  for (int i = 0; i < 1024; i++)
  {
    unsigned char mask = 0;
    for (int j = 0; j < 8; j++)
    {
      if (bufin[i * 8 + j])
      {
        mask += 1 << j;
      }
    }
    mask2[i] = mask;
  }
  // Build mask1 which is a bitmap of the non-zero bytes in mask2
  for (int i = 0; i < 32; i++)
  {
    unsigned int mask = 0;
    for (int j = 0; j < 32; j++)
    {
      if (mask2[i * 32 + j])
      {
        mask += 1 << j;
      }
    }
    mask1[i] = mask;
  }
#endif
  // Build mask0 which is a bitmap of the non-zero words in mask1
  mask0 = 0;
  for (int i = 0; i < 32; i++)
  {
    if (mask1[i])
    {
      mask0 += 1 << i;
    }
  }

  // Store all the non-zero values, starting with mask0
  uint *out32 = (uint *)bufout;
  out32[0] = mask0;
  int lcomp = 1;
  if (mask0)
  {
    // Try to skip the first large chunks of zeroes
#ifndef _MSC_VER
    int start_mask1 = __builtin_ffs(mask0) - 1;
#else
    unsigned long bitIndex;
    _BitScanForward(&bitIndex, mask0);
    int start_mask1 = int(bitIndex);
#endif
    int start_mask2 = start_mask1 * 32;
    int start_data = start_mask1 * 256;
    for (int i = start_mask1; i < 32; i++)
    {
      if (mask1[i])
      {
        out32[lcomp++] = mask1[i];
      }
    }
    // Continue with mask1 and the data, in bytes
    lcomp *= 4;
    for (int i = start_mask2; i < 1024; i++)
    {
      if (mask2[i])
      {
        bufout[lcomp++] = mask2[i];
      }
    }
    for (int i = start_data; i < NOMINAL_BLOCK_SIZE; i++)
    {
      if (bufin[i])
      {
        bufout[lcomp++] = bufin[i];
      }
    }
    // Pad with zeroes until the size is a multiple of 4 bytes
    while (lcomp & 3)
    {
      bufout[lcomp++] = 0;
    }
    // Switch back to words
    lcomp /= 4;
  }

  return lcomp;
}

#ifdef __x86_64
__attribute__((target("avx2")))
#endif
inline void
decoder(const unsigned char *bufin, unsigned char *bufout)
{
#ifdef __x86_64
  alignas(32) unsigned char mask2[1024];
  alignas(32) uint mask1[32];
#else
  alignas(16) unsigned char mask2[1024];
  alignas(16) uint mask1[32];
#endif
  uint mask0;
  uint *buf32 = (uint *)bufin;

  // Start zeroing out the output buffer
  memset(bufout, 0, NOMINAL_BLOCK_SIZE);

  // Read mask0
  mask0 = buf32[0];
  if (mask0 == 0)
  {
    return;
  }

  // memset (mask1, 0, 32 * sizeof (uint));
  memset(mask2, 0, 1024);

  int lcomp = 1;
#ifndef _MSC_VER
  int start_mask1 = __builtin_ffs(mask0) - 1;
#else
  unsigned long bitIndex;
  _BitScanForward(&bitIndex, mask0);
  int start_mask1 = int(bitIndex);
#endif
  int start_mask2 = start_mask1 * 32;

  // Initial portion of mask1 and mask2 might be just zeroes
  for (int i = 0; i < start_mask1; i++)
  {
    mask1[i] = 0;
  }
  for (int i = 0; i < start_mask2; i++)
  {
    mask2[i] = 0;
  }

  // Rebuild mask1
  for (int i = start_mask1; i < 32; i++)
  {
    if (mask0 & (1 << i))
    {
      mask1[i] = buf32[lcomp++];
    }
    else
    {
      mask1[i] = 0;
    }
  }

  // Switch to bytes for mask1 and the data
  lcomp *= 4;

  // Rebuild mask2
  for (int i = start_mask1; i < 32; i++)
  {
    for (int j = 0; j < 32; j++)
    {
      if (mask1[i] & (1 << j))
      {
        mask2[i * 32 + j] = bufin[lcomp++];
      }
      else
      {
        mask2[i * 32 + j] = 0;
      }
    }
  }

  // Get the non-zero data bytes
  for (int i = start_mask2; i < 1024; i++)
  {
    for (int j = 0; j < 8; j++)
    {
      if (mask2[i] & (1 << j))
      {
        bufout[i * 8 + j] = bufin[lcomp++];
      }
    }
  }
}

} // namespace zbm

} // namespace bitcomp
