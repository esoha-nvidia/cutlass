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

#include "ans/ans_utils.cuh"
#include "ans/constants.hpp"
#include "nvcomp/utils.hpp"

namespace ans_gpu_lib
{
namespace detail
{

// Sub-chunk owning this destination copy. Each thread's copies are increasing, so the
// cursor only advances forward, avoiding a binary search's per-copy log2(num_sub_chunks)
// dependent shared reads.
inline __device__ uint32_t
defrag_sub_chunk_for_copy(const uint32_t *subchunk_copy_offsets, uint32_t copy, uint32_t &hint)
{
  while (subchunk_copy_offsets[hint] <= copy)
  {
    ++hint;
  }
  return hint;
}

template <int THREADS_PER_BLOCK>
__device__ void simple_defrag_chunk_cta(ANS_fixed_header *header, uint32_t subchunk_comp_buffer_size, uint32_t &size_out)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t num_sub_chunks = header->num_sub_chunks();
  const uint32_t *const sub_chunk_sizes = header->get_sub_chunk_sizes();
  uint4 *const compressed_payload = reinterpret_cast<uint4 *>(header->get_sub_chunk_0_start());
  const uint32_t sub_chunk_0_offset = header->sub_chunk_0_offset();
  const uint32_t subchunk_stride = subchunk_comp_buffer_size / sizeof(uint4);

  __shared__ uint32_t subchunk_copy_offsets[MAX_SUB_CHUNKS_PER_CHUNK];

  const uint32_t sub_chunk_bytes = (tid < num_sub_chunks) ? sub_chunk_sizes[tid] : 0u;
  const uint32_t aligned_copies = (tid < num_sub_chunks) ? nvcomp::roundUpDiv(sub_chunk_bytes, sizeof(uint4)) : 0u;
  const uint32_t packed_end_copies =
    cta_inclusive_prefix_sum<THREADS_PER_BLOCK>(subchunk_copy_offsets, aligned_copies, num_sub_chunks);
  if (tid == num_sub_chunks - 1)
  {
    size_out = sub_chunk_0_offset + (packed_end_copies - aligned_copies) * sizeof(uint4) + sub_chunk_bytes;
  }
  __syncthreads();

  // Copy from higher addresses to lower addresses. Sync after load
  // to avoid reading in-progress writes. Don't need to sync after write
  // because we'll never read what was written in the previous iteration.
  //
  // Each thread carries VECS_PER_THREAD copies per barrier rather than one. A copy's
  // source index is never below its destination index, so a batch's stores can only
  // land below the next batch's loads: widening the batch keeps the same one-barrier
  // phasing and the same guarantee. What it buys is VECS_PER_THREAD independent loads
  // in flight across the barrier instead of one, so the wait overlaps global-load
  // latency rather than exposing it, and one barrier per BATCH_COPIES copies instead
  // of per THREADS_PER_BLOCK.
  constexpr uint32_t VECS_PER_THREAD = ANS_DEFRAG_UINT4S_PER_THREAD;
  constexpr uint32_t BATCH_COPIES = static_cast<uint32_t>(THREADS_PER_BLOCK) * VECS_PER_THREAD;

  const uint32_t first_copy = subchunk_copy_offsets[0];
  const uint32_t num_copies = subchunk_copy_offsets[num_sub_chunks - 1];

  // Cursor for the forward resolver above. Every index this thread resolves, in this
  // loop and the partial batch below, is strictly increasing, so it stays a lower bound.
  uint32_t hint = 1;

  // Full batches only, so the hot path carries no per-copy bounds test. Both bounds are
  // CTA-uniform, so the barrier inside is reached by every thread the same number of times.
  const uint32_t full_end = first_copy + nvcomp::roundDownTo(num_copies - first_copy, BATCH_COPIES);
  uint32_t copy_base = first_copy;

  // Both batches index and gather identically; only the bounds test differs, so it stays
  // at the call sites and the full-batch loop keeps its untested copies.
  const auto copy_index = [&](uint32_t v) { return copy_base + v * static_cast<uint32_t>(THREADS_PER_BLOCK) + tid; };
  const auto gather_copy = [&](uint32_t copy) {
    const uint32_t ix_subchunk = defrag_sub_chunk_for_copy(subchunk_copy_offsets, copy, hint);
    return compressed_payload[ix_subchunk * subchunk_stride + (copy - subchunk_copy_offsets[ix_subchunk - 1])];
  };

  for (; copy_base < full_end; copy_base += BATCH_COPIES)
  {
    uint4 r[VECS_PER_THREAD];
#pragma unroll
    for (uint32_t v = 0; v < VECS_PER_THREAD; ++v)
    {
      r[v] = gather_copy(copy_index(v));
    }
    __syncthreads();
#pragma unroll
    for (uint32_t v = 0; v < VECS_PER_THREAD; ++v)
    {
      compressed_payload[copy_index(v)] = r[v];
    }
  }

  // Leftover partial batch: bounds-tested, but only here, out of the hot loop. Copies
  // grow with v, so once one is out of range the rest are too and the cursor stays valid.
  if (copy_base < num_copies)
  {
    uint4 r[VECS_PER_THREAD];
#pragma unroll
    for (uint32_t v = 0; v < VECS_PER_THREAD; ++v)
    {
      const uint32_t copy = copy_index(v);
      if (copy < num_copies)
      {
        r[v] = gather_copy(copy);
      }
    }
    __syncthreads();
#pragma unroll
    for (uint32_t v = 0; v < VECS_PER_THREAD; ++v)
    {
      const uint32_t copy = copy_index(v);
      if (copy < num_copies)
      {
        compressed_payload[copy] = r[v];
      }
    }
  }
}

} // namespace detail
} // namespace ans_gpu_lib
