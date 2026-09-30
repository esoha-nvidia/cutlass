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
#include "nvcomp_device_common.cuh"

#include <assert.h>

/*

Split-warp GPU bitpack. Lanes [0, 15] and [16, 31] encode independent inputs of
up to 512 bytes with separate change masks and base values. The two payloads
are concatenated:

  half 0: [0, popc(change_mask_0) * num_active_0)
  half 1: [half_0_size,
           half_0_size + popc(change_mask_1) * num_active_1)

Each active lane still owns eight uint32_t words and writes one byte per
selected bit plane. num_active_0 and num_active_1 are independent counts in
[0, 16], which supports both split uint32_t inputs and word-planed uint64_t
inputs.

Within each half, selected bytes use the same layout as full-warp bitpack:
complete changing bytes are stored directly in word order, while partially
changing bytes are bit-planed in increasing bit position. A change mask of
0xFFFFFFFF uses the word-major, lane-minor fast path.

CALLER CONTRACT:
  - Every lane in a converged warp must call each operation.
  - num_active_0 and num_active_1 must be warp-uniform and in [0, 16].
  - Active lanes are contiguous from the start of each half warp.
  - Payload pointers and metadata values must be warp-uniform.
  - Each lane supplies its own private eight-word input or output array.

*/

namespace nvcomp::cascaded::bitpack
{

constexpr uint32_t SPLIT_WARP_LANES = HALF_WARP_SIZE_U;

struct SplitWarpLane
{
  uint32_t half;
  uint32_t half_lane;
  uint32_t change_mask;
  uint32_t num_active;
  uint32_t payload_offset;
};

inline __device__ SplitWarpLane this_split_warp_lane(
  const uint32_t change_mask_0,
  const uint32_t change_mask_1,
  const uint32_t num_active_0,
  const uint32_t num_active_1
)
{
  const uint32_t lane = lane_id();
  const uint32_t half = lane / SPLIT_WARP_LANES;
  return {
    half,
    lane & (SPLIT_WARP_LANES - 1u),
    half == 0u ? change_mask_0 : change_mask_1,
    half == 0u ? num_active_0 : num_active_1,
    half == 0u ? 0u : __popc(change_mask_0) * num_active_0
  };
}

static_assert(
  BITPACK_WORDS_PER_THREAD == 8u,
  "This implementation processes 8 words per thread (writes uint8_t packed type to output buffer)"
);

struct SplitWarpBitpackCompressionPlan
{
  // do not change the order of the arguments since the warp joined write to shared
  // memory depends on it
  uint32_t change_mask_0;
  uint32_t change_mask_1;
  uint32_t base_value_0;
  uint32_t base_value_1;
  uint32_t payload_size;
};

// Returns the combined payload size in bytes for the two half-warps.
inline __device__ uint32_t warp_bitpack_split_get_compressed_size(
  const uint32_t change_mask_0,
  const uint32_t change_mask_1,
  const uint32_t num_active_0 = SPLIT_WARP_LANES,
  const uint32_t num_active_1 = SPLIT_WARP_LANES
)
{
  return __popc(change_mask_0) * num_active_0 + __popc(change_mask_1) * num_active_1;
}

/**
 * Compute independent change masks and base values for the two half warps.
 * Every lane receives both halves' results.
 *
 * @param my_words Eight uint32_t words owned by the calling lane
 * @param num_active_0 Number of active lanes beginning at lane 0
 * @param num_active_1 Number of active lanes beginning at lane 16
 * @return Both halves' change masks and base values, plus the combined payload size
 */
inline __device__ SplitWarpBitpackCompressionPlan warp_bitpack_split_compress_init(
  const uint32_t *my_words,
  const uint32_t num_active_0 = SPLIT_WARP_LANES,
  const uint32_t num_active_1 = SPLIT_WARP_LANES
)
{
  assert(blockDim.x % WARP_SIZE == 0u && blockDim.y == 1u && blockDim.z == 1u);
  assert(num_active_0 <= SPLIT_WARP_LANES && num_active_1 <= SPLIT_WARP_LANES);

  const uint32_t lane = lane_id();
  const uint32_t half = lane / SPLIT_WARP_LANES;
  const uint32_t half_lane = lane & (SPLIT_WARP_LANES - 1u);
  const uint32_t num_active = half == 0u ? num_active_0 : num_active_1;
  const bool active = half_lane < num_active;

  uint32_t local_or = 0u;
  uint32_t local_and = ~0u;
#pragma unroll
  for (uint32_t i = 0u; i < BITPACK_WORDS_PER_THREAD; ++i)
  {
    local_or |= active ? my_words[i] : 0u;
    local_and &= active ? my_words[i] : ~0u;
  }
  const uint32_t half_or = warp_reduce<SPLIT_WARP_LANES, warp_reduction_op::bitwise_or>(local_or);
  const uint32_t or_0 = __shfl_sync(WARP_ALL, half_or, 0);
  const uint32_t or_1 = __shfl_sync(WARP_ALL, half_or, SPLIT_WARP_LANES);

  const uint32_t half_and = warp_reduce<SPLIT_WARP_LANES, warp_reduction_op::bitwise_and>(local_and);
  const uint32_t and_0 = __shfl_sync(WARP_ALL, half_and, 0);
  const uint32_t and_1 = __shfl_sync(WARP_ALL, half_and, SPLIT_WARP_LANES);

  const uint32_t base_value_0 = num_active_0 == 0u ? 0u : and_0;
  const uint32_t base_value_1 = num_active_1 == 0u ? 0u : and_1;
  const uint32_t change_mask_0 = num_active_0 == 0u ? 0u : ~(and_0 | ~or_0);
  const uint32_t change_mask_1 = num_active_1 == 0u ? 0u : ~(and_1 | ~or_1);
  return {
    change_mask_0,
    change_mask_1,
    base_value_0,
    base_value_1,
    warp_bitpack_split_get_compressed_size(change_mask_0, change_mask_1, num_active_0, num_active_1)
  };
}

/**
 * Writes the two concatenated half-warp payloads.
 *
 * `output` must provide at least the number of bytes returned by
 * warp_bitpack_split_get_compressed_size for the supplied masks and active
 * lane counts.
 *
 * @param my_words Eight uint32_t words owned by the calling lane
 * @param output Destination for the concatenated payloads
 * @param change_mask_0 Changing-bit mask for lanes [0, 15]
 * @param change_mask_1 Changing-bit mask for lanes [16, 31]
 * @param num_active_0 Number of active lanes beginning at lane 0
 * @param num_active_1 Number of active lanes beginning at lane 16
 */
inline __device__ void warp_bitpack_split_compress_apply(
  const uint32_t *my_words,
  uint8_t *output,
  const uint32_t change_mask_0,
  const uint32_t change_mask_1,
  const uint32_t num_active_0 = SPLIT_WARP_LANES,
  const uint32_t num_active_1 = SPLIT_WARP_LANES
)
{
  assert(blockDim.x % WARP_SIZE == 0u && blockDim.y == 1u && blockDim.z == 1u);
  assert(num_active_0 <= SPLIT_WARP_LANES && num_active_1 <= SPLIT_WARP_LANES);

  const SplitWarpLane split = this_split_warp_lane(change_mask_0, change_mask_1, num_active_0, num_active_1);
  detail::group_bitpack_compress_apply(
    my_words,
    output + split.payload_offset,
    split.change_mask,
    split.half_lane,
    split.num_active
  );
}

/** Fused split-warp initialization and payload write. */
inline __device__ SplitWarpBitpackCompressionPlan warp_bitpack_split_compress(
  const uint32_t *my_words,
  uint8_t *output,
  const uint32_t num_active_0 = SPLIT_WARP_LANES,
  const uint32_t num_active_1 = SPLIT_WARP_LANES
)
{
  const SplitWarpBitpackCompressionPlan plan = warp_bitpack_split_compress_init(my_words, num_active_0, num_active_1);
  warp_bitpack_split_compress_apply(my_words, output, plan.change_mask_0, plan.change_mask_1, num_active_0, num_active_1);
  return plan;
}

/**
 * Reconstructs the two independent half-warp inputs.
 *
 * Each active lane receives eight uint32_t words. Values written for inactive
 * lanes are padding and must be ignored by the caller.
 *
 * @param input Source containing the two concatenated payloads
 * @param my_words Destination for the calling lane's eight uint32_t words
 * @param change_mask_0 Changing-bit mask for lanes [0, 15]
 * @param change_mask_1 Changing-bit mask for lanes [16, 31]
 * @param base_value_0 Common-bit base for lanes [0, 15]
 * @param base_value_1 Common-bit base for lanes [16, 31]
 * @param num_active_0 Number of active lanes beginning at lane 0
 * @param num_active_1 Number of active lanes beginning at lane 16
 */
inline __device__ void warp_bitpack_split_decompress(
  const uint8_t *input,
  uint32_t *my_words,
  const uint32_t change_mask_0,
  const uint32_t change_mask_1,
  const uint32_t base_value_0,
  const uint32_t base_value_1,
  const uint32_t num_active_0 = SPLIT_WARP_LANES,
  const uint32_t num_active_1 = SPLIT_WARP_LANES
)
{
  assert(blockDim.x % WARP_SIZE == 0u && blockDim.y == 1u && blockDim.z == 1u);
  assert(num_active_0 <= SPLIT_WARP_LANES && num_active_1 <= SPLIT_WARP_LANES);

  const SplitWarpLane split = this_split_warp_lane(change_mask_0, change_mask_1, num_active_0, num_active_1);
  const uint32_t base_value = split.half == 0u ? base_value_0 : base_value_1;
  detail::group_bitpack_decompress(
    input + split.payload_offset,
    my_words,
    split.change_mask,
    base_value,
    split.half_lane,
    split.num_active
  );
}

} // namespace nvcomp::cascaded::bitpack
