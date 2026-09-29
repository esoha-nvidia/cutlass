/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef NVCOMP_STRINGUTILS_H
#define NVCOMP_STRINGUTILS_H

namespace nvcomp
{

template <int Len>
struct CharArray
{
  static_assert(Len >= 1);

  inline constexpr CharArray()
      : data{}
  {}

  inline constexpr CharArray(const char (&str)[Len])
      : data{}
  {
    for (int i = 0; i < Len; ++i)
    {
      data[i] = str[i];
    }
  }

  template <int Len2>
  inline constexpr auto operator+(const CharArray<Len2> &rhs) const
  {
    CharArray<Len + Len2 - 1> result;
    for (int i = 0; i < Len - 1; ++i)
    {
      result.data[i] = data[i];
    }
    for (int i = 0; i < Len2; ++i)
    {
      result.data[Len - 1 + i] = rhs.data[i];
    }
    return result;
  }

  char data[Len];
};

// Deduction guide for CharArray
template <int Len>
CharArray(const char (&)[Len]) -> CharArray<Len>;

} // namespace nvcomp

#endif
