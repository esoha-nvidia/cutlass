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
#include <cstring>

#include "nvcomp/shared_types.h"

namespace nvcomp::cascaded::universal_header
{

constexpr uint32_t HEADER_SIZE_BYTES = 8;
constexpr uint32_t HEADER_SIZE_WORDS = HEADER_SIZE_BYTES / sizeof(uint32_t);

// Layout (8 bytes total):
//   byte 0: reserved; must be zero
//   byte 1: format version
//   byte 2: bits 0-3 = compression mode
//           bits 4-7 = terminal codec
//   byte 3: bits 0-6 = data_type (nvcompType_t, <= 0x7F)
//           bit 7   = NEW_FORMAT_BIT (0 = legacy, 1 = new-format)
//   bytes 4..7: uncompressed_size (little-endian uint32)

constexpr uint8_t NEW_FORMAT_BIT = 0x80;
constexpr uint8_t DATA_TYPE_MASK = 0x7F;
constexpr uint8_t FORMAT_VERSION = 0;
constexpr uint8_t RESERVED_BYTE = 0;
constexpr uint8_t MODE_MASK = 0x0F;
constexpr uint8_t TERMINAL_CODEC_SHIFT = 4;
constexpr uint8_t TERMINAL_CODEC_MASK = 0x0F;

enum class CompressionMode : uint8_t
{
  Asymmetric = 0,
  Symmetric = 1
};

enum class TerminalCodec : uint8_t
{
  CascadedBitpack = 0
};

inline constexpr __host__ __device__ uint8_t pack_route(const CompressionMode mode, const TerminalCodec terminal_codec)
{
  return static_cast<uint8_t>(
    static_cast<uint8_t>(mode) | (static_cast<uint8_t>(terminal_codec) << TERMINAL_CODEC_SHIFT)
  );
}

// ================================================================================ Predicates

inline __device__ bool is_cascaded_next_compressed(const uint8_t *buffer) { return (buffer[3] & NEW_FORMAT_BIT) != 0; }

inline __device__ bool is_legacy_compressed(const uint8_t *buffer) { return !is_cascaded_next_compressed(buffer); }

// ================================================================================ Writers

inline __device__ void write_cascaded_next_header(
  uint8_t *buffer,
  nvcompType_t data_type,
  uint32_t uncompressed_size,
  CompressionMode mode,
  TerminalCodec terminal_codec
)
{
  assert(static_cast<uint8_t>(mode) <= MODE_MASK);
  assert(static_cast<uint8_t>(terminal_codec) <= TERMINAL_CODEC_MASK);
  assert(static_cast<uint32_t>(data_type) <= DATA_TYPE_MASK);

  buffer[0] = RESERVED_BYTE;
  buffer[1] = FORMAT_VERSION;
  buffer[2] = pack_route(mode, terminal_codec);
  buffer[3] = (static_cast<uint8_t>(data_type) & DATA_TYPE_MASK) | NEW_FORMAT_BIT;

  memcpy(buffer + 4, &uncompressed_size, sizeof(uncompressed_size));
}

inline __device__ void
write_dummy_header(uint8_t *buffer, nvcompType_t data_type, uint32_t uncompressed_size, CompressionMode mode)
{
  write_cascaded_next_header(buffer, data_type, uncompressed_size, mode, TerminalCodec::CascadedBitpack);
}

// ================================================================================ Readers

inline __device__ nvcompType_t get_data_type(const uint8_t *buffer)
{
  return static_cast<nvcompType_t>(buffer[3] & DATA_TYPE_MASK);
}

inline __device__ uint8_t get_format_version(const uint8_t *buffer)
{
  assert(is_cascaded_next_compressed(buffer));
  return buffer[1];
}

inline __device__ CompressionMode get_mode(const uint8_t *buffer)
{
  assert(is_cascaded_next_compressed(buffer));
  return static_cast<CompressionMode>(buffer[2] & MODE_MASK);
}

inline __device__ TerminalCodec get_terminal_codec(const uint8_t *buffer)
{
  assert(is_cascaded_next_compressed(buffer));
  return static_cast<TerminalCodec>((buffer[2] >> TERMINAL_CODEC_SHIFT) & TERMINAL_CODEC_MASK);
}

inline __device__ bool has_supported_preamble(const uint8_t *buffer)
{
  return buffer[0] == RESERVED_BYTE && get_format_version(buffer) == FORMAT_VERSION &&
         get_terminal_codec(buffer) == TerminalCodec::CascadedBitpack;
}

inline __device__ uint32_t get_uncompressed_size(const uint8_t *buffer)
{
  uint32_t v;
  memcpy(&v, buffer + 4, sizeof(v));
  return v;
}

} // namespace nvcomp::cascaded::universal_header
