/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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

#include "crc/cuCRC32.h"
#include "nvcomp/utils.hpp"

namespace nvcomp
{
// Internally chunked deflate block size
// Larger (48KiB or 64KiB) has worse performance, currently best we measured is for 32KiB chunks
constexpr int DEFLATE_BLOCK_SIZE = (1 << 15);

// gzip member header written at the front of every chunk's output:
// magic(2) + CM(1) + FLG(1) + MTIME(4) + XFL(1) + OS(1) = 10 bytes.
static constexpr size_t GZIP_HEADER_BYTES = 10;

// Minimal gzip header: magic, CM=deflate, FLG=0, MTIME=0, XFL=0, OS=0xff.
// MTIME=0 is explicitly "no timestamp available" so this is fully spec-compliant.
// If at any point we need to add more flags to gzip header we will make this not static.
// For now the minimal header is fully sufficient.
static constexpr uint8_t MINIMAL_GZIP_HEADER[GZIP_HEADER_BYTES] = {0x1f, 0x8b, 0x08, 0x00, 0, 0, 0, 0, 0x00, 0xff};

// gzip member footer written at the end of every chunk's output:
// CRC32(4) + ISIZE(4)
static constexpr size_t GZIP_FOOTER_BYTES = 8;

enum class gzipOperatingMode
{
  ONESHOT,
  STREAMING
};
using gzipOperatingMode_t = gzipOperatingMode;

static constexpr __host__ __device__ void assemble_gzip_footer(uint8_t *footer, uint32_t crc, uint32_t isize)
{
  footer[0] = static_cast<uint8_t>(crc & 0xff);
  footer[1] = static_cast<uint8_t>((crc >> 8) & 0xff);
  footer[2] = static_cast<uint8_t>((crc >> 16) & 0xff);
  footer[3] = static_cast<uint8_t>((crc >> 24) & 0xff);
  footer[4] = static_cast<uint8_t>(isize & 0xff);
  footer[5] = static_cast<uint8_t>((isize >> 8) & 0xff);
  footer[6] = static_cast<uint8_t>((isize >> 16) & 0xff);
  footer[7] = static_cast<uint8_t>((isize >> 24) & 0xff);
}

// Max size of inserted BTYPE0 padding deflate block (1 spilled header byte + 4-byte LEN/NLEN = 5).
static constexpr size_t BTYPE0_EMPTY_MAX_BLOCK_SIZE = 5;

// Per-sub-block output slot stride.
// gdeflate's bitwriter writes with an 8-byte (uint64) atomicOr, so every slot base must be 8-byte aligned.
// The stride therefore rounds the worst-case sub-block
// Unified to avoid drift in getMaxOutputChunkSize and fill_descriptors kernel.
constexpr __host__ __device__ size_t gzip_deflate_slot_stride(size_t max_comp_deflate_size)
{
  return roundUpTo(max_comp_deflate_size + BTYPE0_EMPTY_MAX_BLOCK_SIZE, sizeof(uint64_t));
}

// Byte offset where the defragmented deflate stream begins in the output buffer.
// ONESHOT reserves the 10-byte gzip header in front of it; STREAMING
// flushes the header to the output stream separately, so the packed stream starts at 0.
template <gzipOperatingMode_t GZIP_MODE>
constexpr __host__ __device__ size_t gzip_deflate_packed_base()
{
  static_assert(
    GZIP_MODE == gzipOperatingMode::ONESHOT || GZIP_MODE == gzipOperatingMode::STREAMING,
    "gzip_deflate_packed_base: unhandled gzipOperatingMode"
  );
  if constexpr (GZIP_MODE == gzipOperatingMode::ONESHOT)
  {
    return GZIP_HEADER_BYTES; // packed deflate immediately follows the in-buffer header
  }
  else
  {
    return 0; // STREAMING: header is flushed to the output stream separately
  }
}

// Byte offset of the first deflate block source slot, where gdeflate compresses into. Must be 8-byte
// aligned because gdeflate's bitwriter uses an 8-byte (uint64) atomicOr -- hence rounded up from the
// packed base.
template <gzipOperatingMode_t GZIP_MODE>
constexpr __host__ __device__ size_t gzip_deflate_slot_base()
{
  return roundUpTo(gzip_deflate_packed_base<GZIP_MODE>(), sizeof(uint64_t));
}

// Tile size used in defragmentation kernel.
static constexpr int DEFRAG_TILE_SIZE = (1 << 14);

// Standard gzip CRC32 spec (IEEE 802.3 / PKZIP): poly=0x04C11DB7, init=~0,
static constexpr crcSpec_t GZIP_CRC32_SPEC = {0x04C11DB7u, 0xFFFFFFFFu, 1, 1, 0xFFFFFFFFu};

// STREAMING constants

static constexpr size_t GZIP_STREAMING_WINDOW_SIZE = 64 * 1024 * 1024; // 64 MiB

// Windows in the streaming compress pipeline. Three stages (read/compress/write) need >= 3; a couple extra let read/write run ahead of the GPU.
// One CPU thread cycles through NUM_WINDOWS serially; each stage operates simultaneously, but the total ordering of emitted/read windows is guaranteed.
static constexpr int STREAMING_NUM_WINDOWS = 4;

} // namespace nvcomp
