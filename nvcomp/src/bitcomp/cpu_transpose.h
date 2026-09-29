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

#include "bitcomp_private.h"

#ifdef __x86_64

// Optimized bit-transpose for the X86 architectures

#include <immintrin.h>

// x86 SSE implementations of forward bit transpose
template <int BYTES>
__attribute__((target("avx2"))) inline void
bitTransposeFwd(unsigned char *__restrict__ in, unsigned char *__restrict__ out)
{
  // 8-bit AVX2
  if (BYTES == 1)
  {
    __m256i *vin = (__m256i *)in;
    int *i_out = (int *)out;
    // Loop on 8KB = NOMINAL_BLOCK_SIZE values = 256 AVX vectors
    for (int i = 0; i < 256; i++)
    {
      int off = i;
      // Load 32 bytes in an AVX vector
      __m256i v = _mm256_load_si256(vin + i);
      // Extract the 32 MSBs at once
      for (int ibit = 0; ibit < 8; ibit++)
      {
        i_out[off] = _mm256_movemask_epi8(v);
        v = _mm256_slli_epi32(v, 1);
        off += 256;
      }
    }
  }

  // 16-bit AVX2
  if (BYTES == 2)
  {
    __m256i *vin = (__m256i *)in;
    int *iout = (int *)out;
    __m256i vshufl = _mm256_set_epi8(
      15,
      13,
      11,
      9,
      7,
      5,
      3,
      1,
      14,
      12,
      10,
      8,
      6,
      4,
      2,
      0,
      15,
      13,
      11,
      9,
      7,
      5,
      3,
      1,
      14,
      12,
      10,
      8,
      6,
      4,
      2,
      0
    );
    __m256i vperm = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);
    // Loop on 8KB = 4096 values = 256 AVX vectors, 2 vectors at a time
    for (int i = 0; i < 256; i += 2)
    {
      __m256i v[2], tmp[2];
      for (int j = 0; j < 2; j++)
      {
        tmp[j] = _mm256_loadu_si256(vin + i + j);
        // Group same bytes together inside 128-bit lane
        tmp[j] = _mm256_shuffle_epi8(tmp[j], vshufl);
        // Group same bytes together across 128-bit lanes
        tmp[j] = _mm256_permutevar8x32_epi32(tmp[j], vperm);
      }
      // Group same bytes across 2 vectors
      v[0] = _mm256_permute2f128_si256(tmp[0], tmp[1], 0x20); // Bytes 0
      v[1] = _mm256_permute2f128_si256(tmp[0], tmp[1], 0x31); // Bytes 1
      for (int j = 0; j < 8; j++)
      {
        iout[j * 128 + i / 2] = _mm256_movemask_epi8(v[1]);
        iout[j * 128 + i / 2 + 1024] = _mm256_movemask_epi8(v[0]);
        v[0] = _mm256_slli_epi64(v[0], 1);
        v[1] = _mm256_slli_epi64(v[1], 1);
      }
    }
  }

  // 32-bit AVX2
  if (BYTES == 4)
  {
    __m256i *vin = (__m256i *)in;
    int *iout = (int *)out;
    __m256i vshufl = _mm256_set_epi8(
      15,
      11,
      7,
      3,
      14,
      10,
      6,
      2,
      13,
      9,
      5,
      1,
      12,
      8,
      4,
      0,
      15,
      11,
      7,
      3,
      14,
      10,
      6,
      2,
      13,
      9,
      5,
      1,
      12,
      8,
      4,
      0
    );
    __m256i vperm = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
    // Loop on 8KB = 2048 values = 256 AVX vectors, 4 vectors at a time
    for (int i = 0; i < 256; i += 4)
    {
      __m256i v[4];
      __m256i tmp[4];
      for (int j = 0; j < 4; j++)
      {
        v[j] = _mm256_loadu_si256(vin + i + j);
        // Group same bytes together inside 128-bit lane
        v[j] = _mm256_shuffle_epi8(v[j], vshufl);
        // Group same bytes together across 128-bit lanes
        v[j] = _mm256_permutevar8x32_epi32(v[j], vperm);
      }
      // Group same bytes across 4 vectors
      tmp[0] = _mm256_unpacklo_epi64(v[0], v[1]); // Bytes0, bytes2
      tmp[1] = _mm256_unpacklo_epi64(v[2], v[3]); // Bytes0, bytes2
      tmp[2] = _mm256_unpackhi_epi64(v[0], v[1]); // Bytes1, bytes3
      tmp[3] = _mm256_unpackhi_epi64(v[2], v[3]); // Bytes1, bytes3
      v[0] = _mm256_permute2f128_si256(tmp[0], tmp[1], 0x20); // Bytes 0
      v[2] = _mm256_permute2f128_si256(tmp[0], tmp[1], 0x31); // Bytes 2
      v[1] = _mm256_permute2f128_si256(tmp[2], tmp[3], 0x20); // Bytes 1
      v[3] = _mm256_permute2f128_si256(tmp[2], tmp[3], 0x31); // Bytes 3
      for (int j = 0; j < 8; j++)
      {
        iout[j * 64 + i / 4] = _mm256_movemask_epi8(v[3]);
        iout[j * 64 + i / 4 + 512] = _mm256_movemask_epi8(v[2]);
        iout[j * 64 + i / 4 + 1024] = _mm256_movemask_epi8(v[1]);
        iout[j * 64 + i / 4 + 1536] = _mm256_movemask_epi8(v[0]);
        v[0] = _mm256_slli_epi64(v[0], 1);
        v[1] = _mm256_slli_epi64(v[1], 1);
        v[2] = _mm256_slli_epi64(v[2], 1);
        v[3] = _mm256_slli_epi64(v[3], 1);
      }
    }
  }

  // 64-bit SSE
  if (BYTES == 8)
  {
    __m128i *vin = (__m128i *)in;
    short *sout = (short *)out;
    // Loop on 8KB = 1024 values = 512 SSE vectors, with 8 vectors per iteration
    for (int i = 0; i < 512; i += 8)
    {
      __m128i v[8];
      __m128i vtmp[8];
      // Load 16 x 64-bit values in 8 vectors
      for (int j = 0; j < 8; j++)
      {
        v[j] = _mm_loadu_si128(vin + i + j);
      }

      // Reorder the bytes so we can extract the bits that belong together easily
      vtmp[0] = _mm_unpacklo_epi8(v[0], v[4]);
      vtmp[1] = _mm_unpackhi_epi8(v[0], v[4]);
      vtmp[2] = _mm_unpacklo_epi8(v[1], v[5]);
      vtmp[3] = _mm_unpackhi_epi8(v[1], v[5]);
      vtmp[4] = _mm_unpacklo_epi8(v[2], v[6]);
      vtmp[5] = _mm_unpackhi_epi8(v[2], v[6]);
      vtmp[6] = _mm_unpacklo_epi8(v[3], v[7]);
      vtmp[7] = _mm_unpackhi_epi8(v[3], v[7]);

      __m128i vmask = _mm_set1_epi32(0x80808080);
      int off = i / 8;
      // Extract 4 bits at a time
      for (int ibit = 0; ibit < 8; ibit++)
      {
        // Extract the bits (63,55,47,39,31,23,15,7), then shift left
        //  to extract bits (62,54,46,38,30,22,14,6) with same mask
        v[ibit] = _mm_and_si128(vtmp[7], vmask);
        vtmp[7] = _mm_slli_epi32(vtmp[7], 1);
        // The extracted bits must be shifted right by j bits before being added to v[ibit].
        for (int j = 1; j < 8; j++)
        {
          v[ibit] = _mm_or_si128(v[ibit], _mm_srli_epi32(_mm_and_si128(vtmp[7 - j], vmask), j));
          vtmp[7 - j] = _mm_slli_epi32(vtmp[7 - j], 1);
        }
        // Store each half-word in vector containing 8 different bits.
        sout[off] = _mm_extract_epi16(v[ibit], 7);
        sout[off + 512] = _mm_extract_epi16(v[ibit], 6);
        sout[off + 1024] = _mm_extract_epi16(v[ibit], 5);
        sout[off + 1536] = _mm_extract_epi16(v[ibit], 4);
        sout[off + 2048] = _mm_extract_epi16(v[ibit], 3);
        sout[off + 2560] = _mm_extract_epi16(v[ibit], 2);
        sout[off + 3072] = _mm_extract_epi16(v[ibit], 1);
        sout[off + 3584] = _mm_extract_epi16(v[ibit], 0);
        off += 64;
      }
    }
  }
}

// x86 SSE implementations of backward bit transpose
template <int BYTES>
__attribute__((target("avx2"))) inline void
bitTransposeBwd(unsigned char *__restrict__ in, unsigned char *__restrict__ out)
{
  // 8-bit AVX2
  if (BYTES == 1)
  {
    int *i_in = (int *)in;
    __m256i *vout = (__m256i *)out;
    __m256i vbit = _mm256_set_epi8(
      '\x80',
      '\x40',
      '\x20',
      '\x10',
      '\x08',
      '\x04',
      '\x02',
      '\x01',
      '\x80',
      '\x40',
      '\x20',
      '\x10',
      '\x08',
      '\x04',
      '\x02',
      '\x01',
      '\x80',
      '\x40',
      '\x20',
      '\x10',
      '\x08',
      '\x04',
      '\x02',
      '\x01',
      '\x80',
      '\x40',
      '\x20',
      '\x10',
      '\x08',
      '\x04',
      '\x02',
      '\x01'
    );
    __m256i vperm =
      _mm256_set_epi8(3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0);
    // __m256i vperm = _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
    // 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    __m256i vmask0 = _mm256_set1_epi8('\x80');
    // Loop on 8KB = NOMINAL_BLOCK_SIZE values = 256 AVX vectors
    for (int i = 0; i < 256; i++)
    {
      // Bit 7
      __m256i vin = _mm256_set1_epi32(i_in[i]);
      vin = _mm256_shuffle_epi8(vin, vperm);
      __m256i vmask = vmask0;
      vin = _mm256_and_si256(vin, vbit);
      vin = _mm256_cmpeq_epi8(vin, vbit);
      __m256i v = _mm256_and_si256(vmask, vin);
      // Other 7 bits
      for (int j = 1; j < 8; j++)
      {
        vin = _mm256_set1_epi32(i_in[i + j * 256]);
        vin = _mm256_shuffle_epi8(vin, vperm);
        vmask = _mm256_srli_epi32(vmask, 1);
        vin = _mm256_and_si256(vin, vbit);
        vin = _mm256_cmpeq_epi8(vin, vbit);
        v = _mm256_or_si256(v, _mm256_and_si256(vmask, vin));
      }
      _mm256_storeu_si256(vout + i, v);
    }
  }
  // 16-bit AVX2
  if (BYTES == 2)
  {
    __m256i *vout = (__m256i *)out;
    int *i_in = (int *)in;
    __m256i vmask0 = _mm256_set1_epi32(0x80808080);
    __m256i vbits =
      _mm256_set_epi32(0x80804040, 0x20201010, 0x08080404, 0x02020101, 0x80804040, 0x20201010, 0x08080404, 0x02020101);
    __m256i vshufl =
      _mm256_set_epi8(7, 6, 7, 6, 3, 2, 3, 2, 7, 6, 7, 6, 3, 2, 3, 2, 5, 4, 5, 4, 1, 0, 1, 0, 5, 4, 5, 4, 1, 0, 1, 0);
    // Loop on 8KB = 2048 values = 256 AVX vectors, 2 vectors at a time
    for (int i = 0; i < 256; i += 2)
    {
      // Load 64 bits of bits 15,7
      __m256i vtmp, vin[2], v[2];
      vtmp = _mm256_unpacklo_epi8(_mm256_set1_epi32(i_in[1024 + i / 2]), _mm256_set1_epi32(i_in[i / 2]));
      vtmp = _mm256_shuffle_epi8(vtmp, vshufl);
      vin[0] = _mm256_shuffle_epi32(vtmp, 0x00);
      vin[1] = _mm256_shuffle_epi32(vtmp, 0x55);

      __m256i vmask = vmask0;
      for (int k = 0; k < 2; k++)
      {
        vin[k] = _mm256_and_si256(vin[k], vbits);
        vin[k] = _mm256_cmpeq_epi8(vin[k], vbits);
        v[k] = _mm256_and_si256(vin[k], vmask);
      }
      // Other 7 bits
      for (int j = 1; j < 8; j++)
      {
        vtmp = _mm256_unpacklo_epi8(
          _mm256_set1_epi32(i_in[1024 + j * 128 + i / 2]),
          _mm256_set1_epi32(i_in[j * 128 + i / 2])
        );
        vtmp = _mm256_shuffle_epi8(vtmp, vshufl);

        vin[0] = _mm256_shuffle_epi32(vtmp, 0x00);
        vin[1] = _mm256_shuffle_epi32(vtmp, 0x55);
        vmask = _mm256_srli_epi32(vmask, 1);
        for (int k = 0; k < 2; k++)
        {
          vin[k] = _mm256_and_si256(vin[k], vbits);
          vin[k] = _mm256_cmpeq_epi8(vin[k], vbits);
          v[k] = _mm256_or_si256(v[k], _mm256_and_si256(vin[k], vmask));
        }
      }
      for (int k = 0; k < 2; k++)
      {
        _mm256_storeu_si256(vout + i + k, v[k]);
      }
    }
  }

  // 32-bit AVX2
  if (BYTES == 4)
  {
    __m256i *vout = (__m256i *)out;
    int *i_in = (int *)in;
    __m256i vmask0 = _mm256_set1_epi32(0x80808080);
    __m256i vbits =
      _mm256_set_epi32(0x80808080, 0x40404040, 0x20202020, 0x10101010, 0x08080808, 0x04040404, 0x02020202, 0x01010101);
    // Loop on 8KB = 2048 values = 256 AVX vectors, 4 vectors at a time
    for (int i = 0; i < 256; i += 4)
    {
      // Bits Load 32-bits of bits 31,23,15,7
      __m256i vtmp, vin[4], v[4];
      vtmp = _mm256_unpacklo_epi16(
        _mm256_unpacklo_epi8(_mm256_set1_epi32(i_in[1536 + i / 4]), _mm256_set1_epi32(i_in[1024 + i / 4])),
        _mm256_unpacklo_epi8(_mm256_set1_epi32(i_in[512 + i / 4]), _mm256_set1_epi32(i_in[i / 4]))
      );
      vin[0] = _mm256_shuffle_epi32(vtmp, 0x00);
      vin[1] = _mm256_shuffle_epi32(vtmp, 0x55);
      vin[2] = _mm256_shuffle_epi32(vtmp, 0xaa);
      vin[3] = _mm256_shuffle_epi32(vtmp, 0xff);
      __m256i vmask = vmask0;
      for (int k = 0; k < 4; k++)
      {
        vin[k] = _mm256_and_si256(vin[k], vbits);
        vin[k] = _mm256_cmpeq_epi8(vin[k], vbits);
        v[k] = _mm256_and_si256(vin[k], vmask);
      }
      // Other 7 bits
      for (int j = 1; j < 8; j++)
      {
        vtmp = _mm256_unpacklo_epi16(
          _mm256_unpacklo_epi8(
            _mm256_set1_epi32(i_in[1536 + j * 64 + i / 4]),
            _mm256_set1_epi32(i_in[1024 + j * 64 + i / 4])
          ),
          _mm256_unpacklo_epi8(_mm256_set1_epi32(i_in[512 + j * 64 + i / 4]), _mm256_set1_epi32(i_in[j * 64 + i / 4]))
        );
        vin[0] = _mm256_shuffle_epi32(vtmp, 0x00);
        vin[1] = _mm256_shuffle_epi32(vtmp, 0x55);
        vin[2] = _mm256_shuffle_epi32(vtmp, 0xaa);
        vin[3] = _mm256_shuffle_epi32(vtmp, 0xff);
        vmask = _mm256_srli_epi32(vmask, 1);
        for (int k = 0; k < 4; k++)
        {
          vin[k] = _mm256_and_si256(vin[k], vbits);
          vin[k] = _mm256_cmpeq_epi8(vin[k], vbits);
          v[k] = _mm256_or_si256(v[k], _mm256_and_si256(vin[k], vmask));
        }
      }
      for (int k = 0; k < 4; k++)
      {
        _mm256_storeu_si256(vout + i + k, v[k]);
      }
    }
  }
  // 64-bit AVX2
  if (BYTES == 8)
  {
    __m256i *vout = (__m256i *)out;
    int *i_in = (int *)in;
    __m256i vb0, vb1, tmp, v[8];
    __m256i vmask0 = _mm256_set1_epi32(0x80808080);
    __m256i vbit0 =
      _mm256_set_epi32(0x08080808, 0x08080808, 0x04040404, 0x04040404, 0x02020202, 0x02020202, 0x01010101, 0x01010101);
    __m256i vbit1 =
      _mm256_set_epi32(0x80808080, 0x80808080, 0x40404040, 0x40404040, 0x20202020, 0x20202020, 0x10101010, 0x10101010);
    // Loop on 8KB = 1024 values = 256 AVX vectors, with 8 vectors per iteration
    for (int i = 0; i < 256; i += 8)
    {
      // Load 8 x 32-bit values (bits 63,55,47,39,31,23,15,7) into 2 vectors
      // Not fully interleaved, instead each lane is duplicated
      tmp = _mm256_unpacklo_epi16(
        _mm256_unpacklo_epi8(_mm256_set1_epi32(i_in[1792 + i / 8]), _mm256_set1_epi32(i_in[1536 + i / 8])),
        _mm256_unpacklo_epi8(_mm256_set1_epi32(i_in[1280 + i / 8]), _mm256_set1_epi32(i_in[1024 + i / 8]))
      );
      vb1 = _mm256_unpacklo_epi16(
        _mm256_unpacklo_epi8(_mm256_set1_epi32(i_in[768 + i / 8]), _mm256_set1_epi32(i_in[512 + i / 8])),
        _mm256_unpacklo_epi8(_mm256_set1_epi32(i_in[256 + i / 8]), _mm256_set1_epi32(i_in[i / 8]))
      );
      vb0 = _mm256_unpacklo_epi32(tmp, vb1);
      vb1 = _mm256_unpackhi_epi32(tmp, vb1);
      __m256i vmask = vmask0;

      tmp = _mm256_unpacklo_epi64(vb0, vb0);
      v[0] = _mm256_and_si256(vmask, _mm256_cmpeq_epi8(vbit0, _mm256_and_si256(vbit0, tmp)));
      v[1] = _mm256_and_si256(vmask, _mm256_cmpeq_epi8(vbit1, _mm256_and_si256(vbit1, tmp)));
      vb0 = _mm256_unpackhi_epi64(vb0, vb0);
      v[2] = _mm256_and_si256(vmask, _mm256_cmpeq_epi8(vbit0, _mm256_and_si256(vbit0, vb0)));
      v[3] = _mm256_and_si256(vmask, _mm256_cmpeq_epi8(vbit1, _mm256_and_si256(vbit1, vb0)));
      tmp = _mm256_unpacklo_epi64(vb1, vb1);
      v[4] = _mm256_and_si256(vmask, _mm256_cmpeq_epi8(vbit0, _mm256_and_si256(vbit0, tmp)));
      v[5] = _mm256_and_si256(vmask, _mm256_cmpeq_epi8(vbit1, _mm256_and_si256(vbit1, tmp)));
      vb1 = _mm256_unpackhi_epi64(vb1, vb1);
      v[6] = _mm256_and_si256(vmask, _mm256_cmpeq_epi8(vbit0, _mm256_and_si256(vbit0, vb1)));
      v[7] = _mm256_and_si256(vmask, _mm256_cmpeq_epi8(vbit1, _mm256_and_si256(vbit1, vb1)));

      // Other bits
      for (int j = 1; j < 8; j++)
      {
        tmp = _mm256_unpacklo_epi16(
          _mm256_unpacklo_epi8(
            _mm256_set1_epi32(i_in[1792 + j * 32 + i / 8]),
            _mm256_set1_epi32(i_in[1536 + j * 32 + i / 8])
          ),
          _mm256_unpacklo_epi8(
            _mm256_set1_epi32(i_in[1280 + j * 32 + i / 8]),
            _mm256_set1_epi32(i_in[1024 + j * 32 + i / 8])
          )
        );
        vb1 = _mm256_unpacklo_epi16(
          _mm256_unpacklo_epi8(
            _mm256_set1_epi32(i_in[768 + j * 32 + i / 8]),
            _mm256_set1_epi32(i_in[512 + j * 32 + i / 8])
          ),
          _mm256_unpacklo_epi8(_mm256_set1_epi32(i_in[256 + j * 32 + i / 8]), _mm256_set1_epi32(i_in[j * 32 + i / 8]))
        );
        vb0 = _mm256_unpacklo_epi32(tmp, vb1);
        vb1 = _mm256_unpackhi_epi32(tmp, vb1);
        vmask = _mm256_srli_epi32(vmask, 1);

        tmp = _mm256_unpacklo_epi64(vb0, vb0);
        v[0] = _mm256_or_si256(v[0], _mm256_and_si256(vmask, _mm256_cmpeq_epi8(vbit0, _mm256_and_si256(vbit0, tmp))));
        v[1] = _mm256_or_si256(v[1], _mm256_and_si256(vmask, _mm256_cmpeq_epi8(vbit1, _mm256_and_si256(vbit1, tmp))));
        vb0 = _mm256_unpackhi_epi64(vb0, vb0);
        v[2] = _mm256_or_si256(v[2], _mm256_and_si256(vmask, _mm256_cmpeq_epi8(vbit0, _mm256_and_si256(vbit0, vb0))));
        v[3] = _mm256_or_si256(v[3], _mm256_and_si256(vmask, _mm256_cmpeq_epi8(vbit1, _mm256_and_si256(vbit1, vb0))));
        tmp = _mm256_unpacklo_epi64(vb1, vb1);
        v[4] = _mm256_or_si256(v[4], _mm256_and_si256(vmask, _mm256_cmpeq_epi8(vbit0, _mm256_and_si256(vbit0, tmp))));
        v[5] = _mm256_or_si256(v[5], _mm256_and_si256(vmask, _mm256_cmpeq_epi8(vbit1, _mm256_and_si256(vbit1, tmp))));
        vb1 = _mm256_unpackhi_epi64(vb1, vb1);
        v[6] = _mm256_or_si256(v[6], _mm256_and_si256(vmask, _mm256_cmpeq_epi8(vbit0, _mm256_and_si256(vbit0, vb1))));
        v[7] = _mm256_or_si256(v[7], _mm256_and_si256(vmask, _mm256_cmpeq_epi8(vbit1, _mm256_and_si256(vbit1, vb1))));
      }
      for (int j = 0; j < 8; j++)
      {
        _mm256_storeu_si256(vout + i + j, v[j]);
      }
    }
  }
}
#else

// General implementation for any little-endian architecture

// Naive general purpose implementation of forward bit transpose
template <int BYTES>
inline void bitTransposeFwd(unsigned char *__restrict__ in, unsigned char *__restrict__ out)
{
  // Loop on all the bytes per element
  for (int ib = 0; ib < BYTES; ib++)
  {
    // Little endian -> backward from BYTES-1 to start with most significant byte
    unsigned char *pin = in + BYTES - 1 - ib;
    unsigned char *pout = out + ib * (bitcomp::NOMINAL_BLOCK_SIZE / BYTES);
    // Loop on the buffer for a given byte number, load 8 bytes at a time
    for (int j = 0; j < bitcomp::NOMINAL_BLOCK_SIZE / BYTES; j += 8)
    {
      unsigned char b[8];
      for (int i = 0; i < 8; i++)
      {
        b[i] = *pin;
        pin += BYTES;
      }
      // Extract each bit
      unsigned char bits = 0;
      for (int i = 0; i < 8; i++)
      {
        bits = ((b[0] & 0x80) >> 7) | ((b[1] & 0x80) >> 6) | ((b[2] & 0x80) >> 5) | ((b[3] & 0x80) >> 4) |
               ((b[4] & 0x80) >> 3) | ((b[5] & 0x80) >> 2) | ((b[6] & 0x80) >> 1) | (b[7] & 0x80);
        for (int k = 0; k < 8; k++)
        {
          b[k] <<= 1;
        }
        pout[i * (1024 / BYTES)] = bits;
      }
      pout++;
    }
  }
}

// Naive general purpose implementation of backward bit transpose
template <int BYTES>
inline void bitTransposeBwd(unsigned char *__restrict__ in, unsigned char *__restrict__ out)
{
  // Loop on all the bytes per element
  for (int ib = 0; ib < BYTES; ib++)
  {
    // Little endian -> backward from BYTES-1 to start with most significant byte
    unsigned char *pout = out + BYTES - 1 - ib;
    unsigned char *pin = in + ib * (bitcomp::NOMINAL_BLOCK_SIZE / BYTES);
    // Loop on the buffer for a given byte number, 8 bytes at a time
    for (int j = 0; j < bitcomp::NOMINAL_BLOCK_SIZE / BYTES; j += 8)
    {
      unsigned char b[8];
      // First bit
      unsigned char bits = *pin;
      for (int k = 0; k < 8; k++)
      {
        b[k] = bits & 0x80;
        bits <<= 1;
      }
      // Other 7 bits
      for (int i = 1; i < 8; i++)
      {
        bits = pin[i * (1024 / BYTES)];
        for (int k = 0; k < 8; k++)
        {
          b[k] |= (bits & 0x80) >> i;
          bits <<= 1;
        }
      }
      pin++;
      // Store the result (b[7] has bit 0 = first value)
      for (int i = 0; i < 8; i++)
      {
        *pout = b[7 - i];
        pout += BYTES;
      }
    }
  }
}

#endif