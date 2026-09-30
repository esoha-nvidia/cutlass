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

#include <cstring>
#include <vector>

#include "deflate.h"
#include "gdeflate/common.h"
#include "gdeflate/gdeflate_constants.h"

namespace nvcomp
{
namespace gdeflate
{

// Swizzles a deflate stream into the libdeflate-compatible gdeflate format.
// Works as a callback for parse() from deflate.h.
//
// The output interleaves N bit-streams (lanes). Each symbol is assigned to a
// lane following the same round-robin pattern used by the libdeflate gdeflate
// compressor / decompressor:
//   - Block header bits stay on lane 0 (no advance).
//   - Code-length code lengths, code lengths, literals and EOB each occupy one
//     lane and then advance.
//   - Match lengths occupy one lane, but the corresponding distance code is
//     *deferred*: it is written to the same lane when that lane comes back
//     around in the next round of 32.
//   - BT0 (uncompressed) blocks write LEN (16 bits, no alignment, no NLEN)
//     on lane 0, then each data byte on successive lanes with advance.
template <typename T, unsigned int N = 32>
class swizzler
{
  static constexpr unsigned int PACKET_BITS = sizeof(T) * 8;

  uint64_t bitbuf[N];
  int bitcount[N];
  int input_bitcount[N];
  unsigned int write_slot[N];
  unsigned int next_slot[N];
  bool has_next[N];
  std::vector<T> words;
  unsigned int idx;

  struct deferred_copy
  {
    unsigned int round;
    unsigned int bits;
    unsigned int nbits;
  };
  deferred_copy copies[N];
  unsigned int round;
  bool skip_nlen;

  void flush_packet()
  {
    words[write_slot[idx]] = static_cast<T>(bitbuf[idx]);
    bitbuf[idx] >>= PACKET_BITS;
    bitcount[idx] -= PACKET_BITS;
    write_slot[idx] = next_slot[idx];
    has_next[idx] = false;
  }

  void add_bits(unsigned int bits, unsigned int nbits)
  {
    bitbuf[idx] |= static_cast<uint64_t>(bits & ::gdeflate::mask<unsigned int>(nbits)) << bitcount[idx];
    bitcount[idx] += nbits;
    input_bitcount[idx] -= nbits;
    if (bitcount[idx] >= static_cast<int>(PACKET_BITS))
    {
      flush_packet();
    }
    if (input_bitcount[idx] < static_cast<int>(PACKET_BITS) && !has_next[idx])
    {
      next_slot[idx] = static_cast<unsigned int>(words.size());
      words.push_back(0);
      has_next[idx] = true;
      input_bitcount[idx] += PACKET_BITS;
    }
  }

  void advance()
  {
    idx = (idx + 1) % N;
    if (idx == 0)
    {
      round++;
    }
  }

  void write_prev_round_copies()
  {
    while (round > 1 && copies[idx].round == round - 1)
    {
      copies[idx].round = 0;
      add_bits(copies[idx].bits, copies[idx].nbits);
      advance();
    }
  }

  void write_tail_copies()
  {
    unsigned int split = idx % N;
    for (unsigned int n = split; n < N; n++)
    {
      if (round > 1 && copies[n].round == round - 1)
      {
        idx = n;
        add_bits(copies[n].bits, copies[n].nbits);
      }
    }
    for (unsigned int n = 0; n < split; n++)
    {
      if (copies[n].round == round)
      {
        idx = n;
        add_bits(copies[n].bits, copies[n].nbits);
      }
    }
  }

public:
  swizzler()
      : bitbuf{}
      , bitcount{}
      , input_bitcount{}
      , write_slot{}
      , next_slot{}
      , has_next{}
      , words(N, 0)
      , idx(0)
      , copies{}
      , round(1)
      , skip_nlen(false)
  {
    for (unsigned int n = 0; n < N; n++)
    {
      write_slot[n] = n;
      input_bitcount[n] = PACKET_BITS;
    }
  }

  void header(unsigned int bits, unsigned int len)
  {
    if (len == 3)
    {
      idx = 0;
      round = 1;
      for (auto &c : copies)
      {
        c = {};
      }
      skip_nlen = false;
    }
    add_bits(bits, len);
  }

  void uncomp_size(unsigned int bits, unsigned int len)
  {
    if (skip_nlen)
    {
      skip_nlen = false;
      return;
    }
    add_bits(bits, len);
    skip_nlen = true;
  }

  void uncomp(unsigned int bits, unsigned int len)
  {
    add_bits(bits, len);
    advance();
  }

  void start_lencodes() { idx = 0; }

  void lencode(unsigned int len)
  {
    add_bits(len, 3);
    advance();
  }

  void start_codelens() { idx = 0; }

  void codelen(unsigned int bits, unsigned int len, unsigned int xlen, unsigned int /*sym*/)
  {
    add_bits(bits, len + xlen);
    advance();
  }

  void start_symbols()
  {
    idx = 0;
    round = 1;
    for (auto &c : copies)
    {
      c = {};
    }
  }

  void litlen(unsigned int /*i*/, unsigned int bits, unsigned int len, unsigned int xlen, unsigned int sym)
  {
    write_prev_round_copies();
    add_bits(bits, len + xlen);
    if (sym >= ::gdeflate::gdeflate_literal_symbols)
    {
      copies[idx].round = round;
    }
    advance();
    if (sym == 256)
    {
      write_tail_copies();
    }
  }

  void dist(unsigned int /*i*/, unsigned int bits, unsigned int len, unsigned int xlen, unsigned int /*sym*/)
  {
    unsigned int lane = (idx + N - 1) % N;
    copies[lane].bits = bits & ::gdeflate::mask<unsigned int>(len + xlen);
    copies[lane].nbits = len + xlen;
  }

  unsigned int estimate_size() const { return static_cast<unsigned int>(words.size()) + N; }

  unsigned int serialize(T *dst) const
  {
    std::vector<T> out(words);
    for (unsigned int n = 0; n < N; n++)
    {
      if (bitcount[n] > 0)
      {
        out[write_slot[n]] = static_cast<T>(bitbuf[n]);
      }
    }
    std::memcpy(dst, out.data(), out.size() * sizeof(T));
    return static_cast<unsigned int>(out.size());
  }
};

} // namespace gdeflate
} // namespace nvcomp
