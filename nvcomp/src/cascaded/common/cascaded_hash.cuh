/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <climits>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace nvcomp::cascaded
{

// TODO: this hash is used to calculate a set where my_set=my_hash%num_sets. Can we make this hash better with num_set awareness?
// TODO: we may often use hash where the "min" and "max" values are known in advance. Can we make the hash better with range awareness?

// MurmurHash3 (the 32-bit variant, MurmurHash3_x86_32), designed by Austin Appleby
// MurmurHash3 reference impl found here: https://en.wikipedia.org/wiki/MurmurHash
//
// I am using the lowbias32 finalizer instead of the original Murmur finalizer.
// lowbias32 reference: https://github.com/skeeto/hash-prospector
template <typename data_t>
inline __device__ uint32_t hash32(const data_t &value)
{
  // These constants taken directly from reference impl
  const uint32_t c1 = 0xcc9e2d51;
  const uint32_t c2 = 0x1b873593;
  const uint32_t r1 = 15;
  const uint32_t r2 = 13;
  const uint32_t m = 5;
  const uint32_t n = 0xe6546b64;

  uint32_t hash = 0;
  const uint8_t *data = reinterpret_cast<const uint8_t *>(&value);
  constexpr uint32_t len = sizeof(data_t);

  const uint32_t bits_per_hash = 32;

// Process 4 bytes at a time
#pragma unroll
  for (uint32_t i = 0; i + 3 < len; i += 4)
  {
    uint32_t k;
    // Note: use std::memcpy rather than the GCC/Clang builtin so the code
    // compiles under MSVC on Windows. nvcc folds this into a single 4-byte
    // load on all targets.
    std::memcpy(&k, data + i, 4);
    k *= c1;
    k = (k << r1) | (k >> (bits_per_hash - r1));
    k *= c2;
    hash ^= k;
    hash = ((hash << r2) | (hash >> (bits_per_hash - r2))) * m + n;
  }

  // Handle remaining bytes
  uint32_t k = 0;
#pragma unroll
  for (uint32_t i = (len / 4) * 4; i < len; i++)
  {
    k ^= uint32_t{data[i]} << (CHAR_BIT * (i % 4));
  }
  if (len & 3)
  {
    k *= c1;
    k = (k << r1) | (k >> (bits_per_hash - r1));
    k *= c2;
    hash ^= k;
  }

  // Finalization (lowbias32 version, not the original)
  hash ^= (hash >> 16);
  hash *= 0x7feb352d;
  hash ^= (hash >> 15);
  hash *= 0x846ca68b;
  hash ^= (hash >> 16);

  return hash;
}

} // namespace nvcomp::cascaded
