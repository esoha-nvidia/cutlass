/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES.
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

#include <array>
#include <cstring>

#include "gdeflate/common.h"
#include "gdeflate/gdeflate_constants.h"
#include "nvcomp/utils.hpp"

namespace gdeflate
{

inline uint32_t brev(uint32_t x)
{
  x = (((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1));
  x = (((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2));
  x = (((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4));
  x = (((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8));
  return ((x >> 16) | (x << 16));
}

// Reads bits in the input using T as the access grain (up to uint32_t) w/ SIMD width N
template <unsigned int N, typename T = uint32_t, typename Tb = uint64_t>
class bitreader
{
  const T *input; // Pointer to the next word to read
  Tb buf[N]; // Bit buffers
  unsigned int cnt[N]; // Number of bits currently in the buffers

  // Refill i'th lane's bit buffer
  void refill(unsigned int i = 0)
  {
    assert(i < N);
    if (cnt[i] < width)
    {
      buf[i] |= (Tb)(*input++) << cnt[i];
      cnt[i] += width;
    }
  }

  static_assert((N & (N - 1)) == 0, "Bitreader SIMD width must be a power of 2");
  static_assert(sizeof(Tb) == 2 * sizeof(T), "Bit buffer size must be twice the access grain");

public:
  static constexpr unsigned int width = sizeof(T) * 8;
  typedef T type;

  // Pre-initialize bitbuffers (use 64 bit loads?)
  bitreader(const T *base)
      : input(base)
  {
    for (unsigned int i = 0; i < N; i++)
    {
      buf[i] = (Tb)(*input++);
      cnt[i] = width;
    }
  }

  // Consume n bits from the i'th bit buffer and refill if needed
  void eat(unsigned int n, unsigned int i = 0)
  {
    assert(n <= width);
    assert(i < N);
    buf[i] >>= n;
    cnt[i] -= n;
    refill(i);
  }

  // Peek bits from stream
  T peek(unsigned int n, unsigned int i = 0)
  {
    assert(n <= width);
    return (T)buf[i] & mask<T>(n);
  }

  // Read count fixed-size units of n bits invoking a callback function
  template <typename Tfunc>
  void read(unsigned int count, unsigned int n, Tfunc f)
  {
    assert(n <= width);
    for (unsigned int i = 0; i < count; i++)
    {
      f(i, (T)buf[i % N] & mask<T>(n));
      eat(n, i % N);
    }
  }

  // Read variably-sized bit fields invoking a callback function which returns the bit field size, stop if the returned number is negative or 0
  template <typename Tfunc>
  void read(Tfunc f, unsigned int lane = 0)
  {
    // bool done = false;
    int n;
    do
    {
      n = f(lane, (T)buf[lane]);
      if (n != 0)
      {
        eat(abs(n), lane);
      }
      lane = (lane + 1) % N;
    } while (n > 0);
  }

  // Read exactly count variably-sized bit fields invoking a callback function which returns the bit field size
  template <typename Tfunc>
  void read(unsigned int count, Tfunc f, unsigned int lane = 0)
  {
    // bool done = false;
    for (unsigned int i = 0; i < count; i++)
    {
      unsigned int n = f(lane, (T)buf[lane]);
      if (n != 0)
      {
        eat(n, lane);
      }
      lane = (lane + 1) % N;
    };
  }

  // Read n bits at the next byte-aligned boundary from stream 0
  T read_aligned(unsigned int n)
  {
    assert(n + (cnt[0] & 7) <= width);
    buf[0] >>= cnt[0] & 7;
    T res = buf[0] & mask<T>(n);
    cnt[0] &= ~7;
    eat(n, 0);
    return res;
  }
};

// Huffman decoder
template <unsigned int N, unsigned int L, typename T = uint32_t>
class decoder
{
  unsigned int symbols[N];
  unsigned int offset[L + 1];
  T basecode[L + 2]; // +2 to account for the sentinel entry at the end

public:
  static constexpr unsigned int width = sizeof(T) * 8;

  decoder(unsigned int n, const unsigned int *codelen)
      : symbols{0}
      , offset{0}
  {

    assert(n <= N);

    // Calculate histogram of code lengths and find max code length
    unsigned int maxcodelen = 0;
    unsigned int counts[L + 1] = {0};
    for (unsigned int i = 0; i < n; i++)
    {
      ++counts[codelen[i]];
      if (codelen[i] > maxcodelen)
      {
        maxcodelen = codelen[i];
      }
    }

    assert(maxcodelen <= L);

    // Scatter symbols to appropriate locations in the symbol table
    counts[0] = 0;
    for (unsigned int i = 1; i <= L; i++)
    {
      offset[i] = offset[i - 1] + counts[i - 1];
    }
    for (unsigned int i = 0; i < n; i++)
    {
      if (codelen[i] != 0)
      {
        symbols[offset[codelen[i]]++] = i;
      }
    }

    // Generate base codes for each code length
    basecode[0] = 0;
    for (unsigned int i = 1; i <= L; i++)
    {
      basecode[i] = (basecode[i - 1] + counts[i - 1]) << 1;
    }
    for (unsigned int i = 1; i <= L; i++)
    {
      T tmp = basecode[i] << (width - i);
      basecode[i] = tmp < basecode[i] ? (T)-1 : tmp;
    }
    basecode[L + 1] = (T)-1; // Place a sentinel value
  }

  // Map code to length
  unsigned int len4code(T code) const
  { // code is assumed to be left-aligned (MSB on the left)
    unsigned int codelen = 1;
    while (code >= basecode[codelen])
    {
      ++codelen;
    }
    return codelen - 1;
  }

  // Map code to symbol
  unsigned int sym4code(T code, unsigned int len) const
  { // code is assumed to be left-aligned (MSB on the left)
    return symbols[offset[len - 1] + ((code - basecode[len]) >> (width - len))];
  }

  template <typename Treader>
  unsigned int decode(Treader &br)
  {
    unsigned int bits = brev(br.peek(L));
    unsigned int len = len4code(bits);
    unsigned int sym = sym4code(bits, len);
    br.eat(len);

    return sym;
  }
};

template <unsigned int N, typename T, typename Tb, typename Tcallbacks>
void read_lencodes(
  unsigned int (&lencode)[gdeflate_alphabet_size],
  bitreader<N, T, Tb> &br,
  Tcallbacks &callbacks,
  unsigned int hclen
)
{
  assert(hclen > 0 && hclen <= gdeflate_alphabet_size);

  callbacks.start_lencodes();

  br.read(hclen, 3, [&](unsigned int i, unsigned int len) {
    callbacks.lencode(len);
    lencode[map[i]] = len;
  });
}

template <unsigned int N, typename T, typename Tb, typename Tcallbacks>
void unpack_codelens(
  unsigned int *codelen,
  unsigned int count,
  bitreader<N, T, Tb> &br,
  Tcallbacks &callbacks,
  const unsigned int (&lencode)[gdeflate_alphabet_size]
)
{
  decoder<gdeflate_alphabet_size, gdeflate_codelen_max_codelen> dec(
    gdeflate_alphabet_size,
    (const unsigned int *)&lencode
  );

  callbacks.start_codelens();

  br.read([&](unsigned int /*i*/, uint32_t bits) -> int {
    bits &= mask<uint32_t>(7 + 7);
    unsigned int code = brev(bits);
    unsigned int len = dec.len4code(code);
    assert(len <= gdeflate_codelen_max_codelen);
    unsigned int sym = dec.sym4code(code, len);
    assert(sym < gdeflate_alphabet_size);

    unsigned int n, prev, extra = 0;
    unsigned int xlen = 0;

    switch (sym)
    {
      case 16:
        xlen = 2;
        extra = (bits >> len) & 3;
        n = 3 + extra;
        assert(n <= count);
        prev = codelen[-1];
        count -= n;
        while (n--)
        {
          *codelen++ = prev;
        }
        break;
      case 17:
        xlen = 3;
        extra = (bits >> len) & 7;
        n = 3 + extra;
        assert(n <= count);
        memset(codelen, 0, sizeof(unsigned int) * n);
        codelen += n;
        count -= n;
        break;
      case 18:
        xlen = 7;
        extra = (bits >> len) & 127;
        n = 11 + extra;
        assert(n <= count);
        memset(codelen, 0, sizeof(unsigned int) * n);
        codelen += n;
        count -= n;
        break;
      default:
        *codelen++ = sym;
        --count;
        break;
    }

    callbacks.codelen(bits, len, xlen, sym);
    len += xlen;

    // TODO: Figure out how to use nvcomp::narrow_cast here
    const int slen = static_cast<int>(len);
    return (count == 0) ? -slen : slen;
  });
}

template <typename Tcallbacks>
unsigned int parse_litlen(
  const decoder<gdeflate_litlen_symbols, gdeflate_max_codelen> &dec,
  unsigned int i,
  unsigned int code,
  unsigned int bits,
  unsigned int &sym,
  Tcallbacks &callbacks
)
{
  // Decode literal code
  unsigned int len = dec.len4code(code);
  assert(len <= gdeflate_max_codelen);
  sym = dec.sym4code(code, len);
  assert(sym < gdeflate_valid_litlen_symbols);

  unsigned int xlen = sym >= gdeflate_literal_symbols ? xlenbits64[sym - gdeflate_literal_symbols] : 0;

  callbacks.litlen(i, bits, len, xlen, sym);

  return len + xlen;
}

template <typename Tcallbacks>
unsigned int parse_dist(
  const decoder<gdeflate_distance_symbols, gdeflate_max_codelen> &dec,
  unsigned int i,
  unsigned int code,
  unsigned int bits,
  unsigned int &sym,
  Tcallbacks &callbacks
)
{
  unsigned int len = dec.len4code(code);
  assert(len <= gdeflate_max_codelen);
  sym = dec.sym4code(code, len);
  assert(sym < gdeflate_distance_symbols);

  callbacks.dist(i, bits, len, xdistbits64[sym], sym);

  return len + xdistbits64[sym];
}

template <unsigned int N, typename T, typename Tb, typename Tcb>
void parse_symbols(bitreader<N, T, Tb> &br, Tcb &cb, unsigned int hlit, unsigned int hdist, const unsigned int *codelen)
{
  decoder<gdeflate_litlen_symbols, gdeflate_max_codelen> ldec(hlit, codelen);
  decoder<gdeflate_distance_symbols, gdeflate_max_codelen> ddec(hdist, codelen + hlit);

  cb.start_symbols();

  bool iscopy[N] = {0};
  unsigned int lane = 0;

  br.read([&](unsigned int i, uint32_t bits) -> int {
    lane = i;

    // Max Huffman code length + max extra bits (Deflate64 symbol 285)
    bits &= mask<uint32_t>(gdeflate_max_codelen + gdeflate_max_extra_bits);
    unsigned int code = brev(bits);

    unsigned int sym;
    // TODO: Figure out how to use nvcomp::narrow_cast here
    const int len = static_cast<int>(
      iscopy[i] ? parse_dist(ddec, i, code, bits, sym, cb) : parse_litlen(ldec, i, code, bits, sym, cb)
    );

    iscopy[i] = (sym >= gdeflate_literal_symbols);

    return (sym == 256) ? -len : len;
  });

  // Run another round of SIMD processing to handle still outstanding copies
  br.read(
    N,
    [&](unsigned i, uint32_t bits) {
      // Max Huffman code length + max extra bits (Deflate64 symbol 285)
      bits &= mask<uint32_t>(gdeflate_max_codelen + gdeflate_max_extra_bits);
      unsigned int code = brev(bits);
      unsigned int sym;
      return iscopy[i] ? parse_dist(ddec, i, code, bits, sym, cb) : 0;
    },
    (lane + 1) % N
  );
}

void fixed_codelens(unsigned int *codelen)
{
  for (unsigned int i = 0; i < gdeflate_litlen_symbols; i++)
  {
    if (i < 144)
    {
      codelen[i] = 8;
    }
    else if (i < 256)
    {
      codelen[i] = 9;
    }
    else if (i < 280)
    {
      codelen[i] = 7;
    }
    else
    {
      codelen[i] = 8;
    }
  }

  for (unsigned int i = 0; i < gdeflate_distance_symbols; i++)
  {
    codelen[i + gdeflate_litlen_symbols] = 5;
  }
}

template <unsigned int N = 1, typename Tcallbacks>
void parse(const void *input, Tcallbacks &callbacks)
{
  unsigned int header, bfinal, btype;

  bitreader<N, uint32_t, uint64_t> br((const uint32_t *)input);

  do
  {
    header = br.peek(3 + 14);
    bfinal = header & 1;
    btype = (header >> 1) & 3;
    br.eat(3);

    callbacks.header(header, 3);

    unsigned int codelen[gdeflate_total_symbols];

    assert(btype != 3);

    switch (btype)
    {
      case 0: { // Uncompressed block
        unsigned int size = br.read_aligned(16);
        unsigned int nsize = br.read_aligned(16);
        assert(size == (~nsize & 0xffff));
        callbacks.uncomp_size(size, 16);
        callbacks.uncomp_size(nsize, 16);
        br.read(size, 8, [&](unsigned int /*i*/, unsigned int byte) { callbacks.uncomp(byte, 8); });
        break;
      }
      case 1: // Fixed huffman block
      {
        fixed_codelens(codelen);
        parse_symbols(br, callbacks, gdeflate_litlen_symbols, gdeflate_distance_symbols, codelen);
        break;
      }
      case 2: // Dynamic huffman block
      {
        br.eat(14);
        callbacks.header(header >> 3, 14);
        unsigned int hlit = ((header >> 3) & 31) + gdeflate_literal_symbols;
        unsigned int hdist = ((header >> 8) & 31) + 1;
        unsigned int lencode[gdeflate_alphabet_size] = {0};
        read_lencodes(lencode, br, callbacks, ((header >> 13) & 15) + 4);
        unpack_codelens(codelen, hlit + hdist, br, callbacks, lencode);
        parse_symbols(br, callbacks, hlit, hdist, codelen);
        break;
      }
    }

  } while (bfinal == 0);
}

} // namespace gdeflate
