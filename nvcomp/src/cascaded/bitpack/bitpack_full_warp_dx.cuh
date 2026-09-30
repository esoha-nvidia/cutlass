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

#include "cascaded/bitpack/bitpack_common.cuh"
#include "cascaded/bitpack/bitpack_constants.cuh"
#include "cascaded/common/cascaded_warp_reductions.cuh"
#include "CudaConstants.h"

#include <assert.h>

/*

GPU bitpack. Each selected bit produces one byte per active lane, so all writes
are byte aligned.

UNIFIED num_active PARAMETER:
  This codec is parameterized on `num_active` = the number of participating
  lanes. Each active lane owns 8 words. Inactive lanes (>= num_active) contribute
  identity to the reductions and do not write.

    num_active == WARP_SIZE (32): the full 256-word (1KB) input.

    num_active < WARP_SIZE: a variable-length input of N uint32_t words, where
      num_active = ceil(N / 8). Unused slots in the final active lane must be
      initialized. Duplicate-last padding is recommended because other padding
      values may introduce additional changing bits.

CALLER CONTRACT:
  - Every lane in a converged warp must call each operation.
  - num_active must be warp-uniform and in [0, WARP_SIZE].
  - Active lanes are contiguous, beginning at lane 0.
  - Payload pointers and metadata values must be warp-uniform.
  - Each lane supplies its own private eight-word input or output array.

The payload layout (per byte position) is unchanged in structure; the only
difference vs. the original is that the per-plane output stride and the number
of writing lanes are `num_active` instead of WARP_SIZE.

PAYLOAD LAYOUT (per byte position b in [0,4), LSB-first):
  mask_byte == 0xFF  -> full byte: store bytes directly in word order (8
                        word-rows, each of num_active bytes).
  mask_byte != 0xFF  -> partial: bit-plane the set bits (one byte per active
                        lane per set bit).
  Total = popc(change_mask) * num_active bytes.
*/

namespace nvcomp::cascaded::bitpack
{

static_assert(
  BITPACK_WORDS_PER_THREAD == 8u,
  "This implementation processes 8 words per thread (writes uint8_t packed type to output buffer)"
);

struct WarpBitpackCompressionPlan
{
  uint32_t change_mask;
  uint32_t base_value;
  uint32_t payload_size;
};

// Payload size given a change mask and the number of active lanes.
inline __device__ uint32_t warp_bitpack_get_compressed_size(const uint32_t change_mask, const uint32_t num_active)
{
  return __popc(change_mask) * num_active;
}

/**
 *  Phase 1: reduce the active lanes' words to compute change_mask, base_value,
 *  payload_size. Inactive lanes (>= num_active) contribute identity, so the
 *  warp reductions are correct over the full warp.
 *
 *  @param my_words Eight uint32_t words owned by the calling lane
 *  @param num_active Number of active lanes beginning at lane 0
 *  @return The change mask, base value, and exact payload size
 */
inline __device__ WarpBitpackCompressionPlan
warp_bitpack_compress_init(const uint32_t *my_words, const uint32_t num_active = WARP_SIZE)
{
  assert(blockDim.x % WARP_SIZE == 0u && blockDim.y == 1u && blockDim.z == 1u);
  assert(num_active <= WARP_SIZE);

  const uint32_t lane_id = threadIdx.x % WARP_SIZE;
  const bool active = (lane_id < num_active);

  uint32_t local_or = 0u;
  uint32_t local_and = ~0u;

#pragma unroll
  for (uint32_t i = 0u; i < BITPACK_WORDS_PER_THREAD; ++i)
  {
    const uint32_t v = active ? my_words[i] : 0u; // OR identity for inactive lanes
    local_or |= v;
    local_and &= active ? my_words[i] : ~0u; // AND identity for inactive lanes
  }

  const uint32_t or_result = warp_reduce<WARP_SIZE, warp_reduction_op::bitwise_or>(local_or);
  const uint32_t mask_ones = warp_reduce<WARP_SIZE, warp_reduction_op::bitwise_and>(local_and);
  const uint32_t mask_zeros = ~or_result;
  const uint32_t change_mask = num_active == 0u ? 0u : ~(mask_ones | mask_zeros);
  const uint32_t base_value = num_active == 0u ? 0u : mask_ones;
  return {change_mask, base_value, warp_bitpack_get_compressed_size(change_mask, num_active)};
}

/**
 *  Phase 2: write the payload. Active lanes (< num_active) write; the per-plane
 *  / per-word-row output stride is num_active. Full mask bytes stored as-is;
 *  partial bytes bit-planed. output_off is warp-uniform (all lanes advance it
 *  identically); only active lanes write.
 *
 *  Fast path when change_mask == 0xFFFFFFFF: every bit varies so bitpack gains
 *  nothing. Words are stored in word-major / lane-minor layout (word i for all
 *  active lanes occupies bytes [i*num_active*4, (i+1)*num_active*4)), giving
 *  eight coalesced 128-byte rows for a full warp instead of 32 byte-extract
 *  iterations.
 *  Payload size is identical: popc(0xFFFFFFFF)*num_active = 32*num_active bytes.
 */
inline __device__ void warp_bitpack_compress_apply(
  const uint32_t *my_words,
  uint8_t *output,
  const uint32_t change_mask,
  const uint32_t num_active = WARP_SIZE
)
{
  assert(blockDim.x % WARP_SIZE == 0u && blockDim.y == 1u && blockDim.z == 1u);
  assert(num_active <= WARP_SIZE);

  const uint32_t lane_id = threadIdx.x % WARP_SIZE;
  detail::group_bitpack_compress_apply(my_words, output, change_mask, lane_id, num_active);
}

/**
 *  Fused init + apply. num_active defaults to WARP_SIZE.
 */
inline __device__ WarpBitpackCompressionPlan
warp_bitpack_compress(const uint32_t *my_words, uint8_t *output, const uint32_t num_active = WARP_SIZE)
{
  const WarpBitpackCompressionPlan plan = warp_bitpack_compress_init(my_words, num_active);
  warp_bitpack_compress_apply(my_words, output, plan.change_mask, num_active);
  return plan;
}

/**
 *  Inverse. Active lanes (< num_active) recover their 8 words; inactive lanes
 *  leave my_words at base_value (caller ignores them). Per-plane / per-word-row
 *  input stride is num_active. Mirrors compress_apply exactly.
 *
 *  Fast path when change_mask == 0xFFFFFFFF: mirrors the compress fast path.
 *  Words are read from word-major / lane-minor layout with 8 coalesced reads.
 *  base_value is always 0 in this case (no constant bits) but respected anyway.
 */
inline __device__ void warp_bitpack_decompress(
  const uint8_t *input,
  uint32_t *my_words,
  const uint32_t change_mask,
  const uint32_t base_value,
  const uint32_t num_active = WARP_SIZE
)
{
  assert(blockDim.x % WARP_SIZE == 0u && blockDim.y == 1u && blockDim.z == 1u);
  assert(num_active <= WARP_SIZE);

  const uint32_t lane_id = threadIdx.x % WARP_SIZE;
  detail::group_bitpack_decompress(input, my_words, change_mask, base_value, lane_id, num_active);
}

} // namespace nvcomp::cascaded::bitpack
