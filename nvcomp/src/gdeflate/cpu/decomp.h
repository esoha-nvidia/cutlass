/*
 * Copyright (c) 2021-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include <cassert>
#include <cstdint>

#include "deflate.h"
#include "gdeflate/gdeflate_constants.h"

namespace gdeflate
{

// Decompressor (should also work for serial decompression, when N = 1)
// Works as a callback in parse from deflate.h
template <unsigned int N>
class decomp
{
  uint8_t *cpydst[N];
  unsigned int length[N];

public:
  uint8_t *dst;

  decomp(uint8_t *out)
      : dst(out)
  {}

  void start_lencodes() {}
  void start_codelens() {}
  void start_symbols() {}

  void header(unsigned int /*bits*/, unsigned int /*len*/) {}
  void uncomp_size(unsigned int /*bits*/, unsigned int /*len*/) {}
  void lencode(unsigned int /*len*/) {}
  void codelen(unsigned int /*bits*/, unsigned int /*len*/, unsigned int /*xlen*/, unsigned int /*sym*/) {}

  void litlen(unsigned int lane, unsigned int bits, unsigned int len, unsigned int xlen, unsigned int sym)
  {
    if (sym == 256)
    {
      return;
    }

    if (sym < 256)
    {
      *dst++ = (uint8_t)sym;
    }
    else
    {
      // lenbase64 covers the Deflate64 case where symbol 285 maps to base
      // length 3 + 16 extra bits (standard DEFLATE had a fixed length of 258).
      length[lane] = lenbase64[sym - gdeflate_literal_symbols] + ((bits >> len) & mask<unsigned int>(xlen));
      cpydst[lane] = dst;
      dst += length[lane];
    }
  }

  void dist(unsigned int lane, unsigned int bits, unsigned int len, unsigned int xlen, unsigned int sym)
  {
    // distanceTable64 includes the Deflate64 distance symbols 30-31 (each with
    // 14 extra bits), covering distances up to 64K.
    unsigned int offset = distanceTable64[sym] + ((bits >> len) & mask<unsigned int>(xlen));
    uint8_t *dest = cpydst[lane];
    for (unsigned int i = 0; i < length[lane]; i++)
    {
      *dest = *(dest - offset);
      ++dest;
    }
  }

  void uncomp(unsigned int bits, unsigned int len)
  {
    (void)len;
    assert(len == 8);
    *dst++ = (uint8_t)(bits & 0xff);
  }
};

} // namespace gdeflate
