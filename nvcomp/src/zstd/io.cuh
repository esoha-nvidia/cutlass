/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

#include "utils.cuh"

namespace zstd
{

inline __device__ void write_bits_le(uint32_t *dst, int &offset, uint32_t write_val, uint8_t num_bits)
{
  int bit_offset = offset % 32;
  int word_pos = offset / 32;
  write_val &= static_cast<unsigned>(uint64_t{1ul << num_bits} - 1);
  uint64_t write_bits = uint64_t{write_val} << bit_offset;
  uint32_t lo = write_bits;
  uint32_t hi = write_bits >> 32;

  atomicOr(&dst[word_pos], lo);
  if (hi)
  {
    atomicOr(&dst[word_pos + 1], hi);
  }

  offset += num_bits;
}

inline __device__ void
write_bytes_le_unaligned(uint8_t *const dst, int &ix_offset, uint32_t write_val, const int num_bytes)
{
  for (int ix = 0; ix < num_bytes; ++ix)
  {
    dst[ix_offset++] = write_val & 0xff;
    write_val >>= 8;
  }
}

inline __device__ void write_bits_le_2way(uint32_t *dst, int &offset, uint32_t write_val, uint8_t num_bits)
{
  int shfl_bits = __shfl_sync(0x3, num_bits, 1);
  if (thread_warp_ix() == 0)
  {
    offset += shfl_bits;
  }
  if (num_bits > 0)
  {
    write_bits_le(dst, offset, write_val, num_bits);
  }
}

inline __device__ uint32_t read_bits_single_thread_bytewise(const uint8_t *src, int num_bits, const size_t offset)
{
  assert(num_bits <= 32);
  assert(num_bits >= 0);

  uint32_t res = 0;
  // Skip over bytes that aren't in range
  src += offset / 8;
  uint32_t bit_offset = offset % 8;

  uint32_t mask = 0xff;
  int shift = 0;
  while (num_bits > 0)
  {
    if (num_bits < 8)
    {
      mask = (1 << num_bits) - 1;
    }
    res += ((static_cast<uint32_t>(*src++) >> bit_offset) & mask) << shift;
    shift += 8 - bit_offset;
    num_bits -= 8 - bit_offset;
    bit_offset = 0;
  }
  return res;
}

inline __device__ uint32_t read_bits_single_thread(const uint8_t *src, const int num_bits, const size_t offset)
{
  assert(num_bits <= 32);
  assert(num_bits >= 0);

  // Skip over bytes that aren't in range
  src += offset / 8;
  uint32_t bit_offset = offset % 8;

  // Get 4-byte aligned pointer. Just need to read two of these.
  uint8_t align_val = (uintptr_t)(src) % 4;
  src -= align_val;
  const uint32_t *align_src = reinterpret_cast<const uint32_t *>(src);
  uint32_t read_val = *align_src++;
  int skip_bits = bit_offset + align_val * 8;
  assert(skip_bits < 32);
  int read_bits = 32 - skip_bits;
  int rem_bits = num_bits - read_bits;
  uint32_t read_val2 = rem_bits > 0 ? *align_src : 0;
  uint32_t mask = (1 << num_bits) - 1;
  return mask & __funnelshift_r(read_val, read_val2, skip_bits);
}

inline __device__ uint32_t
do_read_stream_bits(const uint8_t *const src, const int bits, int32_t &offset, const int buffer_size)
{
  int32_t actual_offset = offset;
  int32_t actual_bits = bits;

  if (offset < 0)
  {
    actual_bits += offset;
    actual_offset = 0;
  }

  actual_bits = max(actual_bits, 0);
  uint32_t res = 0;
  if (actual_bits > 0)
  {
    const int last_offset_bytes = (actual_offset + actual_bits + 7) / 8;
    // Use bytewise reads if too close to the end to protect against
    // reading off the end of the buffer.
    // TODO: make a cache for this that is initialized properly at the end of the buffer.
    if (buffer_size - last_offset_bytes > 3) [[likely]]
    {
      res = read_bits_single_thread(src, actual_bits, actual_offset);
    }
    else
    {
      res = read_bits_single_thread_bytewise(src, actual_bits, actual_offset);
      // assert(res == read_bits_single_thread(src, actual_bits, actual_offset));
    }
  }

  if (offset < 0)
  {
    // Fill in the bottom "overflowed" bits with 0's
    // Note: in testing, this is always false
    res = -offset >= 32 ? 0 : (res << -offset);
  }
  return res;
}

template <int n_threads>
inline __device__ uint32_t read_stream_bits_shared_shuffles(
  const uint8_t *const src,
  const int bits,
  int32_t &offset,
  int ix_thread,
  const unsigned mask,
  const int buffer_size
)
{
  assert(ix_thread < n_threads);
  int inc_scan_offset = custom_inclusive_scan_registers<n_threads>(bits, mask, ix_thread);
  offset -= inc_scan_offset;

  return do_read_stream_bits(src, bits, offset, buffer_size);
}

struct BitReader
{
  // private:
  const uint8_t *input_buffer;
  int buffer_size;
  int bit_offset;

public:
  __device__ BitReader(const uint8_t *base, const int buffer_size)
      : input_buffer(base)
      , buffer_size(buffer_size)
      , bit_offset(0)
  {}

  __device__ BitReader make_subreader(int n_bytes)
  {
    assert(bit_offset % 8 == 0);
    assert(n_bytes <= rem_bytes());
    BitReader new_bitreader{input_buffer + (bit_offset + 7) / 8, n_bytes};

    bit_offset += n_bytes * 8;
    return new_bitreader;
  }

  inline __device__ int rem_bytes() { return buffer_size - bit_offset / 8; }

  inline __device__ void rewind_bits(uint8_t n_bits) { bit_offset -= n_bits; }

  inline const __device__ uint8_t *get_read_pointer(uint32_t n_bytes)
  {
    assert(bit_offset % 8 == 0);
    const uint8_t *res = input_buffer + bit_offset / 8;
    bit_offset += n_bytes * 8;
    return res;
  }

  inline __device__ uint32_t read_bits(uint32_t num_bits)
  {
    assert(num_bits <= 32);
    uint32_t res = read_bits_single_thread_bytewise(input_buffer, num_bits, bit_offset);

    bit_offset += num_bits;
    // Then update the stuff.
    return res;
  }

  inline __device__ void align_stream()
  {
    if (bit_offset % 8 != 0)
    {
      bit_offset += (8 - bit_offset % 8);
    }
  }

  inline __device__ bool is_stream_aligned() { return bit_offset % 8 == 0; }
};

} // namespace zstd
