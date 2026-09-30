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

#include <cstdint>

#include "ans/constants.hpp"

namespace ans_gpu_lib
{
namespace detail
{

// CTA-uniform chunk bases plus per-warp 32-bit offsets from comp_chunk. Pointers are
// rematerialized at use (comp_at / uncomp_at) so they do not occupy registers across
// the encode loop. Offsets fit in 32 bits: a chunk is at most 1 << 24 bytes.
struct WarpEncodeOff
{
  uint32_t subchunk_output; // bitstream slot
  uint32_t mantissas_offset; // 0 if this policy does not write mantissas
  uint32_t in_start_idx; // first symbol of this sub-chunk
  int sub_chunk_size; // symbols
};
static_assert(sizeof(WarpEncodeOff) == 16, "WarpEncodeOff is four 32-bit fields; ans_compress_smem_bytes assumes 16");

struct EncodeChunkSmem
{
  // Volatile so ptxas reloads from smem at each rematerialize rather than parking
  // values in the register file across encode (and, on the sampled path, across redo).
  const void *volatile uncomp_chunk;
  void *volatile comp_chunk;
  volatile uint32_t bytes;
  volatile uint32_t symbols;
  volatile uint32_t num_sub_chunks;
  volatile uint32_t sample_shift;
  volatile int max_sub_chunk_size;
  volatile uint32_t mantissas_chunk_offset;
  volatile WarpEncodeOff warp_enc[NUM_COMP_WARPS_PER_CTA];

  __device__ uint8_t *comp_at(uint32_t off) const { return static_cast<uint8_t *>(comp_chunk) + off; }
  const __device__ uint8_t *uncomp_at(uint32_t off) const { return static_cast<const uint8_t *>(uncomp_chunk) + off; }
};

// Decoder state read only by the renorm refill path, parked per warp in shared memory so it
// does not occupy registers across the decode loop.
struct __align__(sizeof(uint4)) WarpRefillState
{
  const uint16_t *bit_stream;
  uint32_t pos_offset_u16;
};
static_assert(sizeof(WarpRefillState) == 16, "ans_decompress_smem_bytes assumes 16");

} // namespace detail
} // namespace ans_gpu_lib
