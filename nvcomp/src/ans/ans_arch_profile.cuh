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

#include <cuda_runtime.h>

#include <cassert>
#include <cstdint>

#include <ans/ans_utils.cuh>
#include <ans/simple_types.cuh>

namespace ans_gpu_lib
{
namespace detail
{

// fp16 tiling vocabulary, shared by EncodePolicy/DecodePolicy:
//   row  = the symbols one contiguous inner loop en/decodes (32 lanes x 8 = 256),
//   tile = one encode_tile (x2: 2 rows, x1: 1 row); format-visible.
//   meta = CHAR decoder-only main-loop grouping (CHAR_DECODE_ROWS_PER_META rows unrolled
//          together). Not used for FP16.
constexpr int NUM_ANS_ARCH_IDS = 3;

constexpr size_t CONFIG_ANS_SMEM_PER_SM[NUM_ANS_ARCH_IDS] = {100 * 1024, 228 * 1024, 64 * 1024};

// Rows the char decoder unrolls per main-loop iteration. Char has no side-band mantissa
// loads to batch, so this is purely an unroll factor: it widens the window the scheduler
// has to overlap neighbouring rows' rANS dependency chains.
constexpr int CONFIG_ANS_CHAR_DECODE_ROWS_PER_META[NUM_ANS_ARCH_IDS] = {8, 8, 4};
constexpr uint32_t CONFIG_ANS_RENORM_BUF_UINT16[NUM_ANS_ARCH_IDS] = {1280u, 1536u, 512u};
constexpr int CONFIG_ANS_DECOMP_MIN_BLOCKS_PER_SM[NUM_ANS_ARCH_IDS] = {4, 6, 4};
constexpr int CONFIG_ANS_COMP_MIN_BLOCKS_PER_SM[NUM_ANS_ARCH_IDS] = {4, 6, 4};

// Per-warp cp.async staging budget; the per-policy depth (rows in flight) is derived from
// it and the policy's load width. Deeper is not automatically better: the pipeline only
// reaches steady state once a warp slice is >= 2 * depth rows, so 4 KB/warp (8 uint4 rows
// for fp16) needs twice the slice length of 2 KB before the prefetch pays for itself, and
// 4 KB/warp on sm_120 would also cap occupancy at 2 blocks.
constexpr int CONFIG_ANS_HIST_STAGE_BYTES_PER_WARP[NUM_ANS_ARCH_IDS] = {2048, 2048, 1024};

constexpr int ans_arch_id_from_cuda_arch(int cuda_arch)
{
  return cuda_arch >= 1200 ? 0 : cuda_arch >= 1000 ? 1 : NUM_ANS_ARCH_IDS - 1;
}

#ifdef __CUDA_ARCH__
constexpr int ANS_ARCH_ID = ans_arch_id_from_cuda_arch(__CUDA_ARCH__);
#else
constexpr int ANS_ARCH_ID = 0;
#endif

constexpr int CHAR_DECODE_ROWS_PER_META = CONFIG_ANS_CHAR_DECODE_ROWS_PER_META[ANS_ARCH_ID];
static_assert(CHAR_DECODE_ROWS_PER_META > 0, "the CHAR meta-iter must make progress");

constexpr uint32_t ANS_RENORM_BUF_UINT16 = CONFIG_ANS_RENORM_BUF_UINT16[ANS_ARCH_ID];
static_assert(
  ANS_RENORM_BUF_UINT16 % (WARP_SIZE_U * (sizeof(uint4) / sizeof(uint16_t))) == 0,
  "renorm depth must be a multiple of WARP_SIZE*8 (uint4 LDGSTS granularity)"
);
constexpr int ANS_COMP_MIN_BLOCKS_PER_SM = CONFIG_ANS_COMP_MIN_BLOCKS_PER_SM[ANS_ARCH_ID];
constexpr int ANS_HIST_STAGE_BYTES_PER_WARP = CONFIG_ANS_HIST_STAGE_BYTES_PER_WARP[ANS_ARCH_ID];

namespace
{
// TODO: improve the smem calculation by having one top level struct that contains all the smem used by comp/decomp
// Compress excess over the terms below (CUB temporaries, defrag offsets), taken as cuobjdump
// -res-usage on compress_kernel minus that sum, not measured per component:
// 4016 B before sm_90, 5040 B from sm_90 on; the max keeps this an upper bound.
constexpr size_t ANS_COMP_SCRATCH_SMEM_BYTES = 5040;

constexpr size_t ans_compress_smem_bytes(int i)
{
  return static_cast<size_t>(NUM_COMP_WARPS_PER_CTA) * static_cast<size_t>(CONFIG_ANS_HIST_STAGE_BYTES_PER_WARP[i]) +
         NV_SYMBOL_COUNT * sizeof(int) + 16u + // min/max/uncovered + padding
         NV_SYMBOL_COUNT * sizeof(uint2) + sizeof(uint32_t) +
         sizeof(uint32_t) + // packed_chunk_size_bytes + chunk_idx_handoff
         sizeof(EncodeChunkSmem) + ANS_COMP_SCRATCH_SMEM_BYTES;
}

// CUB scan temporaries, allocated on top of the terms below. Measured with cuobjdump
// -res-usage on a kernel whose only __shared__ is cta_inclusive_prefix_sum's scan:
// 1184 B before sm_90, 2208 B from sm_90 on; the max keeps the estimates upper bounds.
constexpr size_t ANS_SCAN_SMEM_BYTES = 2208;

// Decode footprint for a given tablelog: decoding table, one renorm buffer per decode warp,
// sub-chunk offsets, the handoff word (16 B once aligned), and scan scratch.
constexpr size_t ans_decompress_smem_bytes(int i, uint32_t tablelog)
{
  return (static_cast<size_t>(1u) << tablelog) * sizeof(uint32_t) +
         static_cast<size_t>(NUM_DECOMP_WARPS_PER_CTA) *
           (static_cast<size_t>(CONFIG_ANS_RENORM_BUF_UINT16[i]) * sizeof(uint16_t) + sizeof(WarpRefillState)) +
         MAX_SUB_CHUNKS_PER_CHUNK * sizeof(uint32_t) + sizeof(uint4) + ANS_SCAN_SMEM_BYTES;
}

// Keep the configured occupancy target when shared memory permits it; otherwise
// lower the launch-bounds target to the number of blocks the table can fit.
constexpr int ans_decomp_min_blocks_per_sm(int i, uint32_t tablelog)
{
  const int smem_limit = static_cast<int>(CONFIG_ANS_SMEM_PER_SM[i] / ans_decompress_smem_bytes(i, tablelog));
  return CONFIG_ANS_DECOMP_MIN_BLOCKS_PER_SM[i] < smem_limit ? CONFIG_ANS_DECOMP_MIN_BLOCKS_PER_SM[i] : smem_limit;
}

inline constexpr int ans_decomp_min_blocks_per_sm(uint32_t tablelog)
{
  return ans_decomp_min_blocks_per_sm(ANS_ARCH_ID, tablelog);
}

constexpr bool ans_decompress_smem_fits(int i, uint32_t tablelog)
{
  const int blocks = ans_decomp_min_blocks_per_sm(i, tablelog);
  return blocks > 0 &&
         ans_decompress_smem_bytes(i, tablelog) * static_cast<size_t>(blocks) <= CONFIG_ANS_SMEM_PER_SM[i];
}

// Only one block is proven to fit: tier 2 spans sm_75's 64 KiB and sm_90's 228 KiB, so
// pairing the tier's floor budget with its ceiling footprint describes no real device.
constexpr bool ans_compress_smem_fits(int i)
{
  return CONFIG_ANS_COMP_MIN_BLOCKS_PER_SM[i] > 0 && ans_compress_smem_bytes(i) <= CONFIG_ANS_SMEM_PER_SM[i];
}

#ifndef NDEBUG
// The launch-bounds targets are derived from the footprints above, so those have to bound
// what ptxas allocated. Callers pass the MAX_TABLELOG ceiling, not the mode's own tablelog.
template <typename KernelT, typename EstimateForArchT>
inline void ans_assert_smem_within_estimate(KernelT kernel, EstimateForArchT estimate_for_arch)
{
  cudaFuncAttributes attr{};
  if (cudaFuncGetAttributes(&attr, kernel) == cudaSuccess)
  {
    // ptxVersion is compute capability major * 10 + minor, whereas __CUDA_ARCH__
    // uses major * 100 + minor * 10.
    const int arch_id = ans_arch_id_from_cuda_arch(attr.ptxVersion * 10);
    const size_t estimate = estimate_for_arch(arch_id);
    assert(attr.sharedSizeBytes <= estimate && "ANS smem estimate is below the compiled kernel footprint");
  }
}
#endif

static_assert(ans_decompress_smem_fits(0, MAX_TABLELOG), "ANS tier 0 (sm_120): maximum tablelog decode does not fit");
static_assert(ans_compress_smem_fits(0), "ANS tier 0 (sm_120): compress smem exceeds its occupancy budget");

static_assert(ans_decompress_smem_fits(1, MAX_TABLELOG), "ANS tier 1 (sm_100): maximum tablelog decode does not fit");
static_assert(ans_compress_smem_fits(1), "ANS tier 1 (sm_100): compress smem exceeds its occupancy budget");

static_assert(ans_decompress_smem_fits(2, MAX_TABLELOG), "ANS tier 2 (default): maximum tablelog decode does not fit");
static_assert(ans_compress_smem_fits(2), "ANS tier 2 (default): compress smem exceeds its occupancy budget");

static_assert(NUM_ANS_ARCH_IDS == 3, "update the per-tier static_asserts when adding ANS arch tiers");
} // namespace

} // namespace detail
} // namespace ans_gpu_lib
