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

#include <cstdint>

#include <ans/ans_utils.cuh>

namespace ans_gpu_lib
{
namespace detail
{

// Per-architecture tuning for the ANS decompress kernel, following the same
// CONFIG_*[ARCH_ID] table pattern used by zstd (src/zstd/constants.cuh). These
// knobs trade off register/shared-memory pressure, occupancy, and per-iteration
// work; the best values differ by GPU. The active values are selected purely at
// compile time on the device via ANS_ARCH_ID (derived from __CUDA_ARCH__); the
// host launch reads no per-arch knob (block width is the fixed
// NUM_DECOMP_WARPS_PER_CTA), so there is nothing to keep in sync at runtime.
//
// Only two tiers exist for now -- B200 (the current tuned values) and a
// conservative default for every other GPU (minimum rolls + minimum prefetch).
// Adding a tuned tier later is a localized edit: insert a value into each
// CONFIG_ANS_* array and add the matching __CUDA_ARCH__ branch below.

// Arch thresholds (major*100 + minor*10), descending. The CONFIG_ANS_ARCH
// thresholds and the device #if tiers below must stay in sync.
constexpr int NUM_ANS_ARCH_IDS = 2;
constexpr int CONFIG_ANS_ARCH[NUM_ANS_ARCH_IDS] = {1000, 0}; // B200 (sm_100), then default

// FP16 decodes per lane per outer iter (multiple of 8 and of inner unrolls).
constexpr int CONFIG_ANS_FP16_NUM_ROLLS[NUM_ANS_ARCH_IDS] = {16, 8};
// Inner unroll of the FP16 main loop (FP16_NUM_ROLLS must be a multiple of it).
constexpr int CONFIG_ANS_INNER_UNROLLS[NUM_ANS_ARCH_IDS] = {4, 4};
// uint8 decodes per lane per outer iter (>= 1).
constexpr int CONFIG_ANS_UINT8_NUM_ROLLS[NUM_ANS_ARCH_IDS] = {12, 6};
// Per-warp renorm buffer depth in uint16 (multiple of WARP_SIZE*8, >= worst-case iter).
constexpr uint32_t CONFIG_ANS_RENORM_BUF_UINT16[NUM_ANS_ARCH_IDS] = {1024u, 512u};
// Decompress __launch_bounds__ minimum blocks per SM (B200 = 12; default 8).
constexpr int CONFIG_ANS_DECOMP_MIN_BLOCKS_PER_SM[NUM_ANS_ARCH_IDS] = {12, 8};
// Compress (FP16 fused kernel) __launch_bounds__ minimum blocks per SM. Block is
// NUM_COMP_WARPS_PER_CTA * WARP_SIZE = 128 threads, so minBlocks * 128 must not
// exceed the arch's max threads/SM (sm_75 = 1024 -> max 8). The default tier
// covers sm_75, so it must stay <= 8.
constexpr int CONFIG_ANS_COMP_MIN_BLOCKS_PER_SM[NUM_ANS_ARCH_IDS] = {10, 8};

// Warps per CTA is NOT arch dependent: the fused table build needs CTA width
// 128 (= NUM_DECOMP_WARPS_PER_CTA * WARP_SIZE), so it stays fixed in
// constants.hpp and the host/device launch use it directly.

// Translate __CUDA_ARCH__ into an index into the CONFIG_ANS_* arrays. MUST match
// the CONFIG_ANS_ARCH thresholds. Host TUs (no __CUDA_ARCH__) default to index
// 0; they don't read any tier-varying knob.
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 1000
constexpr int ANS_ARCH_ID = 0;
#else
constexpr int ANS_ARCH_ID = (NUM_ANS_ARCH_IDS - 1);
#endif
#else
constexpr int ANS_ARCH_ID = 0;
#endif

// Active (device-compiled) tuning knobs, selected by ANS_ARCH_ID. Only
// meaningful inside the kernel.
constexpr int NUM_FP16_SYMBOLS_PER_THREAD_META_ITER = CONFIG_ANS_FP16_NUM_ROLLS[ANS_ARCH_ID];
constexpr int NUM_SYMBOLS_PER_INNER_ITER = CONFIG_ANS_INNER_UNROLLS[ANS_ARCH_ID];
constexpr int NUM_CHAR_SYMBOLS_PER_THREAD_META_ITER = CONFIG_ANS_UINT8_NUM_ROLLS[ANS_ARCH_ID];
constexpr uint32_t RENORM_PREFETCH_BUF_SIZE = CONFIG_ANS_RENORM_BUF_UINT16[ANS_ARCH_ID] * sizeof(uint16_t);
constexpr uint32_t RENORM_PREFETCH_BUF_SIZE_U16 = RENORM_PREFETCH_BUF_SIZE / sizeof(uint16_t);
constexpr int ANS_DECOMP_MIN_BLOCKS_PER_SM = CONFIG_ANS_DECOMP_MIN_BLOCKS_PER_SM[ANS_ARCH_ID];
constexpr int ANS_COMP_MIN_BLOCKS_PER_SM = CONFIG_ANS_COMP_MIN_BLOCKS_PER_SM[ANS_ARCH_ID];

// Warp-wide symbols decoded per meta-iter (one per lane per roll across the warp).
constexpr int NUM_FP16_SYMBOLS_PER_WARP_META_ITER = NUM_FP16_SYMBOLS_PER_THREAD_META_ITER * WARP_SIZE;
constexpr int NUM_CHAR_SYMBOLS_PER_WARP_META_ITER = NUM_CHAR_SYMBOLS_PER_THREAD_META_ITER * WARP_SIZE;

// Renorm-buffer refill threshold (uint16). Refill when buf_pos_ can no longer
// satisfy one worst-case meta-iter. Each lane decodes
// NUM_SYMBOLS_PER_THREAD_META_ITER symbols, each consuming up to
// DEFAULT_TABLELOG bits of state, and the bit stream is read in 16-bit chunks, so
// a single lane reads at most roundUpDiv(DEFAULT_TABLELOG * rolls, 16) uint16.
// Per-iter consumption is the SUM of per-lane reads, so we round the PER-LANE
// bits->uint16 conversion UP and THEN scale by WARP_SIZE.
constexpr int MAX_NUM_SYMBOLS_PER_THREAD_META_ITER =
  std::max(NUM_FP16_SYMBOLS_PER_THREAD_META_ITER, NUM_CHAR_SYMBOLS_PER_THREAD_META_ITER);
constexpr uint32_t RENORM_REFILL_THRESHOLD_U16 =
  static_cast<uint32_t>(WARP_SIZE) *
  nvcomp::roundUpDiv(DEFAULT_TABLELOG * static_cast<uint32_t>(MAX_NUM_SYMBOLS_PER_THREAD_META_ITER), 16u);

// NUM_CHAR_SYMBOLS_PER_THREAD_META_ITER (CHAR decodes/lane/iter),
// NUM_FP16_SYMBOLS_PER_THREAD_META_ITER (FP16 decodes/lane/iter),
// RENORM_PREFETCH_BUF_SIZE (per-warp renorm buffer size in bytes), and
// RENORM_REFILL_THRESHOLD_U16 (refill trigger) are per-arch tunable and defined
// in ans_arch_profile.cuh. The renorm buffer is decoupled from per-iter
// consumption: it holds multiple iters' worth of renorm data and is only
// refilled when buf_pos_ drops below RENORM_REFILL_THRESHOLD_U16, so most outer
// iters issue *only* the mantissa LDGSTS (no renorm LDGSTS, no commit/wait, no
// bookkeeping) under typical data.

// Per-iter mantissa staging payload (bytes): one mantissa byte per FP16
// decode = NUM_FP16_SYMBOLS_PER_THREAD_META_ITER * WARP_SIZE.
constexpr uint32_t MANTISSA_STAGE_SIZE = static_cast<uint32_t>(NUM_FP16_SYMBOLS_PER_THREAD_META_ITER) * WARP_SIZE;

// Per-warp per-stage stride (bytes) of the mantissa staging buffer. The
// per-lane mantissa LDGSTS uses uint2 (8 B) form, sourced directly from
// sub_chunk_mantissas (naturally 8 B-aligned) with no front-pad and no
// leading-slot carry: each stage holds exactly the MANTISSA_STAGE_SIZE payload
// for one outer iter, where stage[m] == sub_chunk_mantissas[iter *
// MANTISSA_STAGE_SIZE + m]. 8 B alignment is sufficient for the uint2 LDGSTS
// dst; we keep the stride 16 B-aligned so each warp/stage slice base is
// 16-byte aligned (no functional requirement, just clean addressing).
constexpr uint32_t MANTISSA_WARP_STRIDE = ((MANTISSA_STAGE_SIZE + 15u) / 16u) * 16u;

// FP8 per-warp renorm buffer depth (uint16). The fp8 decode paths load the
// packed_signs_mantissas side-band with a plain LDG (no cp.async staging), so the fp8 decode
// shared-memory layout drops mantissa_staging entirely and folds those bytes
// into a deeper renorm buffer. fp8 compresses poorly (small renorm region per
// symbol), so a deeper window pushes refills (and their per-iter bookkeeping)
// out of the hot loop at no extra shared memory vs the fp16/char layout.
// Reclaimed u16 = per-warp mantissa bytes / 2 = (2 * MANTISSA_WARP_STRIDE) / 2 =
// MANTISSA_WARP_STRIDE. Must stay a multiple of WARP_SIZE*8 for the uint4 LDGSTS.
constexpr uint32_t RENORM_PREFETCH_BUF_SIZE_U16_FP8 = RENORM_PREFETCH_BUF_SIZE_U16 + MANTISSA_WARP_STRIDE;
static_assert(
  RENORM_PREFETCH_BUF_SIZE_U16_FP8 % (static_cast<uint32_t>(WARP_SIZE) * 8u) == 0,
  "fp8 renorm depth must be a multiple of WARP_SIZE*8 (uint4 LDGSTS granularity)"
);

// Coupling constraints, validated for every tier so a bad edit fails to compile
// regardless of which GPU is being built for.
namespace
{
constexpr bool ans_config_is_valid(int i)
{
  return CONFIG_ANS_FP16_NUM_ROLLS[i] % 8 == 0 && // mantissa uint2 LDGSTS granularity
         CONFIG_ANS_INNER_UNROLLS[i] > 0 && CONFIG_ANS_FP16_NUM_ROLLS[i] % CONFIG_ANS_INNER_UNROLLS[i] == 0 &&
         CONFIG_ANS_UINT8_NUM_ROLLS[i] > 0 &&
         CONFIG_ANS_RENORM_BUF_UINT16[i] % (static_cast<uint32_t>(WARP_SIZE) * 8u) == 0 && // uint4 LDGSTS granularity
         // renorm buffer must cover one worst-case FP16 (or char, whichever is larger) iter of consumption: the
         // per-lane worst case rounded UP to whole uint16, summed across the warp.
         CONFIG_ANS_RENORM_BUF_UINT16[i] >=
           static_cast<uint32_t>(WARP_SIZE) *
             nvcomp::roundUpDiv(DEFAULT_TABLELOG * static_cast<uint32_t>(CONFIG_ANS_FP16_NUM_ROLLS[i]), 16u) &&
         CONFIG_ANS_RENORM_BUF_UINT16[i] >=
           static_cast<uint32_t>(WARP_SIZE) *
             nvcomp::roundUpDiv(DEFAULT_TABLELOG * static_cast<uint32_t>(CONFIG_ANS_UINT8_NUM_ROLLS[i]), 16u) &&
         CONFIG_ANS_DECOMP_MIN_BLOCKS_PER_SM[i] > 0 && CONFIG_ANS_COMP_MIN_BLOCKS_PER_SM[i] > 0;
}
static_assert(ans_config_is_valid(0), "ANS arch config tier 0 violates a decode coupling constraint");
static_assert(ans_config_is_valid(1), "ANS arch config tier 1 violates a decode coupling constraint");
static_assert(NUM_ANS_ARCH_IDS == 2, "update the per-tier static_asserts when adding ANS arch tiers");
} // namespace

} // namespace detail
} // namespace ans_gpu_lib
