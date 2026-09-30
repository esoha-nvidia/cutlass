/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or its
 * affiliates is strictly prohibited.
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <type_traits>

#include "cascaded/next/terminal_codecs/terminal_cascaded/terminal_cascaded_constants.cuh"
#include "cascaded/universal/universal_header.cuh"

namespace nvcomp::cascaded::next::terminal_cascaded
{

template <typename T>
constexpr bool USE_UINT64_WORD_PLANES = std::is_same_v<T, uint64_t>;

// Pipeline-0 style packed u32 metadata sizes (selector | mask | primary [| secondary]).
inline constexpr uint32_t U32_PACKED_METADATA_BYTES = 9u;
inline constexpr uint32_t U32_PACKED_DELTA_FOR_METADATA_BYTES = 13u;
inline constexpr uint32_t U64_PACKED_METADATA_BYTES = 17u;
inline constexpr uint32_t U64_PACKED_DELTA_FOR_METADATA_BYTES = 25u;

template <typename T>
inline __device__ T load_unaligned(const void *src)
{
  T value;
  memcpy(&value, src, sizeof(value));
  return value;
}

template <typename T>
inline __device__ void store_unaligned(void *dst, const T value)
{
  memcpy(dst, &value, sizeof(value));
}

inline __host__ __device__ uint32_t total_words(const uint32_t uncompressed_bytes)
{
  return (uncompressed_bytes + 3u) / 4u;
}

inline __host__ __device__ uint32_t num_kc_batches_for(const uint32_t uncompressed_bytes)
{
  return (uncompressed_bytes + KC_SIZE_BYTES - 1u) / KC_SIZE_BYTES;
}

inline __host__ __device__ uint32_t num_sub_units_for(const uint32_t num_kc_batches)
{
  return (num_kc_batches + WARPS_PER_CTA - 1u) / WARPS_PER_CTA;
}

inline __host__ __device__ uint32_t kc_words(const uint32_t total_wds, const uint32_t kc_idx)
{
  const uint32_t base = kc_idx * WORDS_PER_KC;
  const uint32_t rem = total_wds - base;
  return rem < WORDS_PER_KC ? rem : WORDS_PER_KC;
}

inline __host__ __device__ uint32_t num_active_for(const uint32_t kc_wds)
{
  return (kc_wds + WORDS_PER_THREAD - 1u) / WORDS_PER_THREAD;
}

inline __host__ __device__ uint32_t preamble_size(const uint32_t num_sub_units)
{
  return universal_header::HEADER_SIZE_BYTES + num_sub_units * 2u;
}

inline __host__ __device__ uint32_t max_compressed_chunk_bytes(const uint32_t uncompressed_bytes)
{
  const uint32_t num_kc_batches = num_kc_batches_for(uncompressed_bytes);
  const uint32_t num_sub_units = num_sub_units_for(num_kc_batches);
  return preamble_size(num_sub_units) + num_kc_batches * COMPRESS_SCRATCH_BYTES_PER_WARP;
}

inline __host__ __device__ uint32_t kc_size_bytes(const uint32_t uncompressed_bytes, const uint32_t kc_idx)
{
  const uint32_t kc_start = kc_idx * KC_SIZE_BYTES;
  const uint32_t remaining = uncompressed_bytes - kc_start;
  return remaining < KC_SIZE_BYTES ? remaining : KC_SIZE_BYTES;
}

} // namespace nvcomp::cascaded::next::terminal_cascaded
