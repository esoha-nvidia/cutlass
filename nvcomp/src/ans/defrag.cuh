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

namespace ans_gpu_lib
{
namespace detail
{
// Caller-owned scratch for defrag_chunk_cta. Kept as two separate fields (not a
// union): keeping s_meta[] and the BlockScan temp distinct saves a syncthreads
// between the scan and the s_meta store. Small enough (~1 KB at THREADS=128) to
// overlay on a fused compressor's histogram-count/encoding-table storage without
// changing its shared-memory footprint.
template <int THREADS_PER_BLOCK>
struct DefragSmem
{
  // Per-sub-chunk metadata, indexed by sub-chunk: .x = packed dest word offset
  // (exclusive-prefix of word-rounded lengths; entry N holds the total word
  // count sentinel), .y = source slot WORD offset (sc * slot_words) from the
  // uint32 sub-chunk-0 base.
  uint2 s_meta[MAX_SUB_CHUNKS_PER_CHUNK + 1];
  typename nvcomp::cub::BlockScan<uint32_t, THREADS_PER_BLOCK>::TempStorage s_scan_tmp;
};

// CTA-wide defrag of a single compressed chunk. Compacts sub-chunks 1..N-1
// back-to-back right after sub-chunk 0, writes the updated per-sub-chunk offsets
// into the header, and writes the chunk's final compressed size through size_out.
template <int THREADS_PER_BLOCK, int VECS_PER_THREAD>
__device__ void
defrag_chunk_cta(void *comp_chunk, uint32_t slot_words, size_t &size_out, DefragSmem<THREADS_PER_BLOCK> &smem)
{
  static_assert(
    THREADS_PER_BLOCK >= MAX_SUB_CHUNKS_PER_CHUNK,
    "defrag_chunk_cta needs at least as many threads as the max sub-chunks per chunk"
  );

  static_assert(VECS_PER_THREAD > 0, "defrag_chunk_cta needs at least one uint4 register per thread");
  constexpr int BATCH_VECS = THREADS_PER_BLOCK * VECS_PER_THREAD; // dest uint4 vectors per batch
  constexpr int WORDS_PER_VEC = sizeof(uint4) / sizeof(uint32_t);

  const int tid = threadIdx.x;

  uint2 *const s_meta = smem.s_meta;

  using Scan = nvcomp::cub::BlockScan<uint32_t, THREADS_PER_BLOCK>;

  // comp_chunk may be unaligned; align before reading the mantissa size.
  void *const aligned_comp_chunk = round_up_align_address(comp_chunk, 8);
  ANS_sub_chunk_header *const header = ANS_sub_chunk_header::from_aligned_comp_chunk(aligned_comp_chunk);

  const uint32_t num_sub_chunks = static_cast<uint32_t>(header->num_sub_chunks_);
  const size_t *const sub_chunk_sizes = header->get_sub_chunk_sizes();

  // Base (uint32-aligned) source pointer of sub-chunk sc: the block-uniform
  // sub-chunk-0 base (held in a register) plus the sub-chunk's source-slot word
  // offset (s_meta[sc].y). Avoids the old 8-byte shared pointer load (which was
  // 2-way bank-conflicting) and the per-vec4 64-bit multiply, while keeping the
  // 64-bit slot invariants out of the body's live register set.
  const uint32_t *const sub_chunk_0_base =
    reinterpret_cast<const uint32_t *>(round_up_align_address(header->get_sub_chunk_0_start(), sizeof(uint32_t)));

  // Destination starts right after sub-chunk 0, which stays where it is.
  const size_t sub_chunk_0_offset = header->get_sub_chunk_0_offset();
  const size_t packed_offset = nvcomp::roundUpTo(sub_chunk_0_offset + sub_chunk_sizes[0], 4u);
  uint32_t *const destination = reinterpret_cast<uint32_t *>(reinterpret_cast<uint8_t *>(header) + packed_offset);

  // Block exclusive scan of the word-rounded sub-chunk lengths. Thread t owns
  // sub-chunk t: contributes its word length for sub-chunks 1..N-1, 0 otherwise
  // (sub-chunk 0 stays in place; out-of-range threads contribute 0).
  const bool owns_packed = (tid > 0) && (tid < num_sub_chunks);
  const uint32_t my_len = owns_packed ? nvcomp::roundUpDiv(sub_chunk_sizes[tid], sizeof(uint32_t)) : uint32_t{0};
  uint32_t total_moved_words;
  uint32_t my_prefix;
  Scan(smem.s_scan_tmp).ExclusiveSum(my_len, my_prefix, total_moved_words);

  if (tid < num_sub_chunks)
  {
    // Pack the packed-dest prefix and the source-slot word offset into one uint2
    // (one STS.64). The src_off (sc*slot_words) lets the hot resolver add the
    // block-uniform base in a register instead of dereferencing an 8-byte shared
    // pointer.
    s_meta[tid] = make_uint2(my_prefix, tid * slot_words);

    // Folded update_offsets: thread t owns sub-chunk t, so write its post-defrag header offset
    // straight from the exclusive scan -- sub-chunk 0 stays in place (keeps its offset), and each
    // moved sub-chunk lands at packed_offset + my_prefix words. The thread owning the LAST sub-chunk
    // also writes the chunk's final compressed size (last offset + last un-rounded size + the
    // mantissa prefix).
    const size_t off = (tid == 0) ? sub_chunk_0_offset
                                  : packed_offset + static_cast<size_t>(my_prefix) * sizeof(uint32_t);
    header->get_sub_chunk_offsets()[tid] = off;
    if (tid == num_sub_chunks - 1)
    {
      // Mask off the FP8 self-describing MSB flag so the size is the real prefix.
      const size_t mantissas_size = ans_mantissas_size_value(*reinterpret_cast<size_t *>(aligned_comp_chunk));
      size_out = mantissas_size + off + sub_chunk_sizes[tid];
    }
  }
  if (tid == 0)
  {
    // Sentinel: end-of-stream prefix. .y is never read (src_base is only called
    // with a valid sub-chunk < num_sub_chunks) but is set for a defined value.
    s_meta[num_sub_chunks] = make_uint2(total_moved_words, num_sub_chunks * slot_words);
  }
  __syncthreads();

  // Whole block agrees on total_moved_words (block-wide scan), so this branch never
  // splits the block; skipping is deadlock-free. A single-sub-chunk chunk has
  // nothing to move (total_moved_words == 0) -- returning here also keeps the
  // body_words computation below from underflowing.
  if (total_moved_words == 0)
  {
    return;
  }

  assert(total_moved_words * sizeof(uint32_t) >= ans_gpu_lib::get_sub_chunk_overhead());

  const auto src_base = [sub_chunk_0_base, s_meta](int sc) -> const uint32_t * {
    return sub_chunk_0_base + s_meta[sc].y;
  };

  // Min sub-chunk size is 64B, so always safe
  const auto load_head_word = [&](uint32_t id) -> uint32_t {
    assert(id < WORDS_PER_VEC && id < ans_gpu_lib::get_sub_chunk_overhead() / sizeof(uint32_t));
    return src_base(1)[id]; // head words always live in the first moved sub-chunk
  };

  // Tail words will be at most 3, and since every sub-chunk is at
  // least get_sub_chunk_overhead() (64B = 16 words), the last sub-chunk alone
  // always covers them.
  const auto load_tail_word = [&](uint32_t global_id) -> uint32_t {
    const uint32_t sub_chunk_id = num_sub_chunks - 1;
    assert(s_meta[sub_chunk_id].x <= global_id);
    return src_base(sub_chunk_id)[global_id - s_meta[sub_chunk_id].x];
  };

  // Resolve the sub-chunk of stream word g via a FORWARD linear scan seeded from
  // a cached lower-bound `hint` (advanced in place). Each thread reads its dest
  // vectors in strictly increasing stream order, so the previous vector's
  // sub-chunk is always a valid lower bound for the next.
  const auto resolve_hint = [&](uint32_t g, int &hint) -> int {
    while (s_meta[hint + 1].x <= g)
    {
      ++hint;
    }
    return hint;
  };

  // Load the 4 source words for dest stream positions [g, g+4) into a uint4 using
  // the hinted resolver. Fast path: all 4 lie in one sub-chunk's interior, read
  // with the widest naturally-aligned vector access the source permits. Slow path
  // (rare straddle): gather each word independently (binary-searched).
  // Performance note: replacing the resolved read with a trivial far-ahead load
  // (no resolve_hint, no s_meta LDS, no address arithmetic) measured ~+4.2%
  // on defrag_chunks_bench, because the kernel is memory-bound and the address
  // math is almost entirely hidden under global-load latency.
  //
  // compute-sanitizer initcheck note: this can load uninitialized memory, but is not an issue.
  const auto load_vec4_hint = [&](uint32_t g, int &hint) -> uint4 {
    const int sc = resolve_hint(g, hint);
    // One LDS.64 fetches both the sub-chunk's packed-dest prefix (.x) and its
    // source-slot word offset (.y); reads are warp-broadcast so conflict-free.
    const uint2 meta = s_meta[sc];
    const uint32_t dst_base = meta.x;
    const uint32_t dst_in_sc = g - dst_base;
    const uint32_t len = s_meta[sc + 1].x - dst_base;
    const uint32_t *const src = sub_chunk_0_base + meta.y + dst_in_sc;
    uint4 out;
    if (dst_in_sc + WORDS_PER_VEC <= len)
    {
      const uintptr_t addr = reinterpret_cast<uintptr_t>(src);
      if ((addr % sizeof(uint4)) == 0)
      {
        out = *reinterpret_cast<const uint4 *>(src);
      }
      else if ((addr % sizeof(uint2)) == 0)
      {
        const uint2 a = reinterpret_cast<const uint2 *>(src)[0];
        const uint2 b = reinterpret_cast<const uint2 *>(src)[1];
        out = make_uint4(a.x, a.y, b.x, b.y);
      }
      else
      {
        out = make_uint4(src[0], src[1], src[2], src[3]);
      }
    }
    else
    {
      // Straddle: the 4 words span a sub-chunk boundary. Resolve each word with
      // a LOCAL forward cursor seeded from sc (already resolved for g), instead
      // of an independent binary search. The boundary is crossed at most a few
      // times within 4 consecutive words, so this is 0-1 comparisons/word and
      // keeps the binary-search machinery (lo/hi/mid) out of the hot loop's
      // register allocation entirely. cur stays a valid lower bound because the
      // word indices g..g+3 are strictly increasing.
      uint32_t w[4];
      int cur = sc;
#pragma unroll
      for (int j = 0; j < 4; ++j)
      {
        const uint32_t gj = g + static_cast<uint32_t>(j);
        while (s_meta[cur + 1].x <= gj)
        {
          ++cur;
        }
        w[j] = src_base(cur)[gj - s_meta[cur].x];
      }
      out = make_uint4(w[0], w[1], w[2], w[3]);
    }
    return out;
  };

  // Align the destination stream to a 16-byte boundary so the bulk copy can use
  // uint4 vector stores. The first `head_words` words (0..3) are copied scalar.
  const uint32_t dest_misalign_words =
    (static_cast<uint32_t>(reinterpret_cast<uintptr_t>(destination)) / sizeof(uint32_t)) %
    WORDS_PER_VEC; // words past a 16B line
  const uint32_t head_words = dest_misalign_words ? (WORDS_PER_VEC - dest_misalign_words) : 0u;
  assert(head_words < total_moved_words);

  const uint32_t body_words = total_moved_words - head_words;

  const uint32_t body_vecs = body_words / WORDS_PER_VEC; // complete uint4s
  const uint32_t tail_words = body_words % WORDS_PER_VEC; // 0..3 leftover words
  const uint32_t tail_start = head_words + (body_vecs * WORDS_PER_VEC);
  uint4 *const dest_vec = reinterpret_cast<uint4 *>(destination + head_words);

  // Per-thread sub-chunk cursor for the hinted forward-scan resolver. Each
  // thread's vectors are read in strictly increasing stream order, so this only
  // ever advances; it amortizes sub-chunk resolution to ~0-1 comparisons/vector
  // in the large-data configs (vs a full binary search each).
  int hint_sc = 1;

  // Race between this read and a possible write in the next for loop is guarded by the __syncthreads() below.
  if (head_words)
  {
    if (tid < WARP_SIZE)
    {
      uint32_t word;
      if (tid < head_words)
      {
        word = load_head_word(tid);
      }
      __syncwarp();
      if (tid < head_words)
      {
        destination[tid] = word;
      }
    }
  }

  // Main loop: only the FULL batches (every one of the BATCH_VECS vectors is in
  // range), so the hot path carries no per-vector bounds test. Strided indexing
  // keeps the accesses coalesced -- at each r, consecutive threads touch
  // consecutive uint4s. The leftover partial batch is folded into the tail below.
  const uint32_t full_vecs = (body_vecs / BATCH_VECS) * BATCH_VECS;
  for (uint32_t batch_base = 0; batch_base < full_vecs; batch_base += BATCH_VECS)
  {
    uint4 reg[VECS_PER_THREAD];
#pragma unroll
    for (int r = 0; r < VECS_PER_THREAD; ++r)
    {
      const uint32_t v = batch_base + static_cast<uint32_t>(r) * THREADS_PER_BLOCK + tid;
      reg[r] = load_vec4_hint(head_words + v * WORDS_PER_VEC, hint_sc);
    }

    // Performance note: removing this + the partial barrier measured ~3% SLOWER on
    // B200 -- the barrier's block-wide wait overlaps global-load latency, so
    // dropping it just re-exposes that latency as long-scoreboard stalls.
    __syncthreads();

#pragma unroll
    for (int r = 0; r < VECS_PER_THREAD; ++r)
    {
      const uint32_t v = batch_base + static_cast<uint32_t>(r) * THREADS_PER_BLOCK + tid;
      dest_vec[v] = reg[r];
    }
  }

  // Tail, part 1: the partial batch left over after the full iterations (fewer than
  // BATCH_VECS vectors). Bounds-tested here -- once, out of the hot loop -- with the
  // same read -> __syncthreads() -> write phasing as the main loop. Each thread's
  // stream positions stay above its main-loop reads, so hint_sc remains monotonic.
  // (When body_vecs < BATCH_VECS the main loop runs zero times and this is the only
  // vector phase, so the whole copy still uses exactly one barrier.)
  const uint32_t partial_vecs = body_vecs - full_vecs;
  if (partial_vecs)
  {
    uint4 reg[VECS_PER_THREAD];
#pragma unroll
    for (int r = 0; r < VECS_PER_THREAD; ++r)
    {
      const uint32_t v = full_vecs + static_cast<uint32_t>(r) * THREADS_PER_BLOCK + tid;
      if (v < body_vecs)
      {
        reg[r] = load_vec4_hint(head_words + v * WORDS_PER_VEC, hint_sc);
      }
    }

    __syncthreads();

#pragma unroll
    for (int r = 0; r < VECS_PER_THREAD; ++r)
    {
      const uint32_t v = full_vecs + static_cast<uint32_t>(r) * THREADS_PER_BLOCK + tid;
      if (v < body_vecs)
      {
        dest_vec[v] = reg[r];
      }
    }
  }

  // Tail, part 2: the <=3 scalar words after the last full vector. Threads
  // 0..tail_words-1 (warp 0), barrier-free for the same reason as the head.
  if (tail_words)
  {
    if (tid < WARP_SIZE)
    {
      uint32_t word;
      if (tid < tail_words)
      {
        word = load_tail_word(tail_start + tid);
      }
      __syncwarp();
      if (tid < tail_words)
      {
        destination[tail_start + tid] = word;
      }
    }
  }
}
} // namespace detail
} // namespace ans_gpu_lib
