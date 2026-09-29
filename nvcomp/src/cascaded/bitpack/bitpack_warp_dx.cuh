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

#include <assert.h>

/*

This a special bitpack implementation designed for GPU performance. The
most critical goal is: the number of bits written per thread is ALWAYS
a multiple of 8 and thus writes are byte aligned. This is implemented
so as to have no compression ratio penalty.

This implementation treats all data types as 4B unsigned integers. I am 
intentionally not templating these functions on data_type for the following
reasons:

    For 1B or 2B types - Again, templating on the type results in more
        expensive reductions with no ratio benefit. Instead of templating on
        data_type, the caller should try to reduce their metadata (changing 
        byte mask and/or base value) because the metadata should include
        repeated bytes/shorts if the data is truly the smaller type.

    For 8B types - An 8B AND/OR reduction would require 2x the shuffles and
        2x the registers per word, and would produce a 64-bit change_mask.
        Constant high bytes in the 64-bit mask cost no payload bits but
        also yield no ratio benefit over splitting the data into low/high
        4B halves. Callers with 8B data should "word plane" into two 4B
        streams and bitpack each independently.

The compression API exposes both a fused entry point and a two-phase split:

    warp_bitpack_compress (fused) — does init + apply in one call. Use this
        when you don't need to know the compressed size before writing
        (e.g. fixed-size output buffers).

    warp_bitpack_compress_init  — does only the AND/OR reduction. Returns
        (change_mask, base_value, payload_size). No writes. Use this when
        you need to reserve output space ahead of writing (e.g. variable-
        length appenders / bump allocators).

    warp_bitpack_compress_apply — writes the bit-plane payload into a
        pre-reserved output region using a precomputed change_mask. No
        reduction.

The fused warp_bitpack_compress is implemented as init + apply, so there
is exactly one source of truth for both the reduction and the write.

*/

namespace bitpack
{

// TODO: move this to cascaded_constants.cuh
constexpr uint32_t WARP_SIZE = 32;
constexpr uint32_t bitpack_batch_size_bytes = 1024;
constexpr uint32_t bitpack_batch_words_per_thread = bitpack_batch_size_bytes / sizeof(uint32_t) / WARP_SIZE;

static_assert(
  bitpack_batch_words_per_thread == 8,
  "This implementation processes 8 words per thread (writes uint8_t packed type to output buffer)"
);

// Warp-wide AND reduction // TODO: move this to cascaded_utils.cuh
inline __device__ uint32_t warp_reduce_and(uint32_t v)
{
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
  {
    v &= __shfl_xor_sync(0xFFFFFFFF, v, offset);
  }
  return v;
}

// Warp-wide OR reduction // TODO: move this to cascasded_utils.cuh
inline __device__ uint32_t warp_reduce_or(uint32_t v)
{
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
  {
    v |= __shfl_xor_sync(0xFFFFFFFF, v, offset);
  }
  return v;
}

/**
 *  @return The bitpacked size (excluding metadata) of the batch corresponding to this change mask
 */
inline __device__ uint32_t warp_bitpack_get_compressed_size(uint32_t change_mask)
{
  return __popc(change_mask) * WARP_SIZE;
}

/**
 *  Phase 1 of bitpacking: reduce the warp's words to compute the change_mask,
 *  base_value, and payload_size for an upcoming warp_bitpack_compress_apply call.
 *
 *  After calling this, the caller knows the exact compressed payload size and
 *  can reserve space in an output buffer (e.g. via a bump allocator) before
 *  calling warp_bitpack_compress_apply to perform the actual write.
 *
 *  @param my_words     A thread's private set of 8 elements to be bitpacked together in this batch.
 *  @param change_mask  Output: 4B mask of bits that change across the batch. Must be stored by the
 *                      caller to recover this batch.
 *  @param base_value   Output: 4B value of bits common to all elements in the batch. Must be stored
 *                      by the caller to recover this batch.
 *  @param payload_size Output: number of bytes that warp_bitpack_compress_apply will write to its
 *                      output buffer for this batch.
 */
inline __device__ void
warp_bitpack_compress_init(const uint32_t *my_words, uint32_t &change_mask, uint32_t &base_value, uint32_t &payload_size)
{
  // AND/OR reduction across the warp.
  uint32_t local_and = ~0u, local_or = 0u;
#pragma unroll
  for (int i = 0; i < bitpack_batch_words_per_thread; ++i)
  {
    local_and &= my_words[i];
    local_or |= my_words[i];
  }
  const uint32_t mask_ones = warp_reduce_and(local_and);
  const uint32_t mask_zeros = ~warp_reduce_or(local_or);
  change_mask = ~(mask_ones | mask_zeros);
  base_value = mask_ones;
  payload_size = warp_bitpack_get_compressed_size(change_mask);
}

/**
 *  Phase 2 of bitpacking: write the bit-plane payload into a pre-reserved
 *  output region. The caller must have already obtained `change_mask` from
 *  warp_bitpack_compress_init and reserved at least `payload_size` bytes at
 *  `output`.
 *
 *  @param my_words    A thread's private set of 8 elements to be bitpacked together in this batch.
 *  @param output      Destination buffer; must have at least payload_size bytes available.
 *  @param change_mask Mask of bits that change across the batch, as returned by
 *                     warp_bitpack_compress_init.
 */
inline __device__ void warp_bitpack_compress_apply(const uint32_t *my_words, uint8_t *output, uint32_t change_mask)
{
  if ((threadIdx.x & (WARP_SIZE - 1)) == 0)
  {
    assert(blockDim.y == 1 && blockDim.z == 1);
  }

  const uint32_t lane_id = threadIdx.x & (WARP_SIZE - 1);

  uint32_t mask_copy = change_mask;
  uint32_t output_off = 0;
  while (mask_copy)
  {
    const int bit_pos = __ffs(mask_copy) - 1;
    mask_copy &= mask_copy - 1;

    uint8_t packed = 0;
#pragma unroll
    for (int i = 0; i < bitpack_batch_words_per_thread; ++i)
    {
      packed |= ((my_words[i] >> bit_pos) & 1u) << i;
    }
    output[output_off + lane_id] = packed;
    output_off += WARP_SIZE;
  }
}

/**
 *  Bitpacks 1KB of data treated as unsigned 32 bit integers (fused init + apply).
 *
 *  Use this entry point when you don't need to reserve output space ahead of
 *  writing. For variable-length appenders that need the payload size before
 *  reserving, call warp_bitpack_compress_init followed by
 *  warp_bitpack_compress_apply instead.
 *
 *  @param my_words A thread's private set of 8 elements to be bitpacked together in this batch.
 *  @param output The buffer where the bitpacked values are written.
 *  @param change_mask Output: 4B mask showing which bits change between elements. Must be stored by
 *                     the caller to recover this batch.
 *  @param base_value Output: 4B value showing the bit values all batch elements have in common.
 *                    Must be stored by the caller to recover this batch.
 *  @param payload_size Output: number of bytes written to `output`.
 */
inline __device__ void warp_bitpack_compress(
  const uint32_t *my_words,
  uint8_t *output,
  uint32_t &change_mask,
  uint32_t &base_value,
  uint32_t &payload_size
)
{
  warp_bitpack_compress_init(my_words, change_mask, base_value, payload_size);
  // sync not necessary
  warp_bitpack_compress_apply(my_words, output, change_mask);
}

/**
 *  @brief Recovers a 1KB batch that was bitpacked using the given metadata.
 *
 *  @param input A pointer to the start of the compressed buffer
 *  @param my_words An 8-element array where this thread's elements will be recovered to
 *  @param change_mask The caller must provide the change mask showing which bits are changing between uncompressed elements.
 *  @param base_value The caller must provide this value showing the common bit values shared by all elements in the batch.
 */
inline __device__ void
warp_bitpack_decompress(const uint8_t *input, uint32_t *my_words, uint32_t change_mask, uint32_t base_value)
{
  if ((threadIdx.x & (WARP_SIZE - 1)) == 0)
  {
    assert(blockDim.y == 1 && blockDim.z == 1);
  }
  const uint32_t lane_id = threadIdx.x & (WARP_SIZE - 1);

// Initialize all words to base_value (constant bits already in place,
// changing bits are zero and will be filled in below).
#pragma unroll
  for (int i = 0; i < bitpack_batch_words_per_thread; ++i)
  {
    my_words[i] = base_value;
  }

  // For each set bit in change_mask, every lane reads one byte and unpacks it.
  uint32_t mask_copy = change_mask;
  uint32_t input_off = 0;

  while (mask_copy)
  {
    const int bit_pos = __ffs(mask_copy) - 1;
    mask_copy &= mask_copy - 1;

    const uint8_t packed = input[input_off + lane_id];
    input_off += WARP_SIZE;

#pragma unroll
    for (int i = 0; i < bitpack_batch_words_per_thread; ++i)
    {
      my_words[i] |= ((uint32_t)((packed >> i) & 1u)) << bit_pos;
    }
  }
}

} // namespace bitpack
