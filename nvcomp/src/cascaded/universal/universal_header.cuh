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

#include <cassert>
#include <cstdint>

#include "nvcomp.h"

namespace universal_header
{

constexpr uint32_t header_size_bytes = 8;
constexpr uint32_t header_size_words = header_size_bytes / sizeof(uint32_t);

// Layout (8 bytes total):
//   byte 0: legacy num_rle / new-format algo flags byte 0. num_RLE always <= 7
//   byte 1: legacy num_delta / new-format algo flags byte 1. num_Delta always <= 7
//   byte 2: legacy use_bitpack (0/1) / new-format algo flags byte 2
//   byte 3: bits 0-6 = data_type (nvcompType_t, <= 0x7F)
//           bit 7   = NEW_FORMAT_BIT (0 = legacy, 1 = new-format)
//   bytes 4..7: uncompressed_size (little-endian uint32)

// Dummy headers are encoded as legacy-compatible headers:
//   bytes 0..2 == 0 and NEW_FORMAT_BIT is clear. Both the legacy and next writers
//   produce identical bytes for dummy buffers.

constexpr uint8_t NEW_FORMAT_BIT = 0x80;
constexpr uint8_t DATA_TYPE_MASK = 0x7F;

// ================================================================================ Predicates

inline __device__ bool is_cascaded_next_compressed(const uint8_t *buffer) { return (buffer[3] & NEW_FORMAT_BIT) != 0; }

inline __device__ bool is_dummy_compressed(const uint8_t *buffer)
{
  // A dummy compressed file can be decompressed using legacy or next kernels
  return buffer[0] == 0 && buffer[1] == 0 && buffer[2] == 0;
}

inline __device__ bool is_legacy_compressed(const uint8_t *buffer)
{
  // Classification is based solely on NEW_FORMAT_BIT to avoid predicate ordering pitfalls.
  return !is_cascaded_next_compressed(buffer);
}

// ================================================================================ Writers

inline __device__ void write_legacy_header(
  uint8_t *buffer,
  nvcompType_t data_type,
  uint32_t uncompressed_size,
  uint32_t num_rles,
  uint32_t num_deltas,
  uint32_t use_bitpack
)
{
  assert(num_rles <= 0xFF);
  assert(num_deltas <= 0xFF);
  assert(use_bitpack <= 1);
  assert(static_cast<uint32_t>(data_type) <= DATA_TYPE_MASK);

  buffer[0] = static_cast<uint8_t>(num_rles);
  buffer[1] = static_cast<uint8_t>(num_deltas);
  buffer[2] = static_cast<uint8_t>(use_bitpack);
  buffer[3] = static_cast<uint8_t>(data_type) & DATA_TYPE_MASK; // legacy: bit 7 clear

  memcpy(buffer + 4, &uncompressed_size, sizeof(uncompressed_size));
}

inline __device__ void write_dummy_header(uint8_t *buffer, nvcompType_t data_type, uint32_t uncompressed_size)
{
  assert(static_cast<uint32_t>(data_type) <= DATA_TYPE_MASK);
  write_legacy_header(buffer, data_type, uncompressed_size, 0, 0, 0);
}

inline __device__ void write_cascaded_next_header(
  uint8_t *buffer,
  nvcompType_t data_type,
  uint32_t uncompressed_size,
  uint32_t algorithm_flags // only low 24 bits are stored (bytes 0..2)
)
{
  assert((algorithm_flags & 0xFF000000u) == 0);
  assert(static_cast<uint32_t>(data_type) <= DATA_TYPE_MASK);
  // Disallow the dummy-looking encoding so is_dummy stays unambiguous.
  assert((algorithm_flags & 0x00FFFFFFu) != 0);

  buffer[0] = static_cast<uint8_t>(algorithm_flags & 0xFF);
  buffer[1] = static_cast<uint8_t>((algorithm_flags >> 8) & 0xFF);
  buffer[2] = static_cast<uint8_t>((algorithm_flags >> 16) & 0xFF);
  buffer[3] = (static_cast<uint8_t>(data_type) & DATA_TYPE_MASK) | NEW_FORMAT_BIT;

  memcpy(buffer + 4, &uncompressed_size, sizeof(uncompressed_size));
}

// ================================================================================ Readers

inline __device__ uint32_t get_num_rles(const uint8_t *buffer)
{
  assert(is_legacy_compressed(buffer));
  return buffer[0];
}
inline __device__ uint32_t get_num_deltas(const uint8_t *buffer)
{
  assert(is_legacy_compressed(buffer));
  return buffer[1];
}
inline __device__ uint32_t get_use_bitpack(const uint8_t *buffer)
{
  assert(is_legacy_compressed(buffer));
  return buffer[2];
}

inline __device__ nvcompType_t get_data_type(const uint8_t *buffer)
{
  return static_cast<nvcompType_t>(buffer[3] & DATA_TYPE_MASK);
}

inline __device__ uint32_t get_algo_flags(const uint8_t *buffer)
{
  // Returns the 24-bit algorithm-flags field for new-format headers.
  assert(is_cascaded_next_compressed(buffer));

  return static_cast<uint32_t>(buffer[0]) | (static_cast<uint32_t>(buffer[1]) << 8) |
         (static_cast<uint32_t>(buffer[2]) << 16);
}

inline __device__ uint32_t get_uncompressed_size(const uint8_t *buffer)
{
  uint32_t v;
  memcpy(&v, buffer + 4, sizeof(v));
  return v;
}

} // namespace universal_header
