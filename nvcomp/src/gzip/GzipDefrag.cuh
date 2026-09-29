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

#include <cuda/atomic>

#include "GzipKernels.cuh"

namespace nvcomp
{

inline __device__ size_t last_index_at_or_before(const uint64_t *offsets, size_t first, size_t last, size_t target)
{
  size_t lo = first; // inclusive
  size_t hi = last + 1; // exclusive
  while (lo < hi)
  {
    const size_t mid = lo + (hi - lo) / 2;
    if (offsets[mid] <= target)
    {
      lo = mid + 1;
    }
    else
    {
      hi = mid;
    }
  }
  // Offset may start just before `target` and still need to be selected to copy its overlapping bytes.
  // lo == upper_bound(target); step back to the last index <= target, clamped to `first`.
  return (lo == first) ? first : lo - 1;
}

__device__ uint32_t pack_bytes_u32(const uint8_t *src)
{
  return static_cast<uint32_t>(src[0]) | (static_cast<uint32_t>(src[1]) << 8) | (static_cast<uint32_t>(src[2]) << 16) |
         (static_cast<uint32_t>(src[3]) << 24);
}
__device__ void copy_bytes(uint8_t *dst, const uint8_t *src, uint32_t bytes, uint32_t tid, uint32_t num_threads)
{
  constexpr uint32_t WORD_BYTES = sizeof(uint32_t);
  constexpr uint32_t WORDS_PER_THREAD = 2; // unroll factor: independent stores in flight per thread

  const uint32_t align_prefix = min(
    bytes,
    (WORD_BYTES - (static_cast<uint32_t>(reinterpret_cast<uintptr_t>(dst)) & (WORD_BYTES - 1))) & (WORD_BYTES - 1)
  );
  const uint8_t *word_src = src + align_prefix;
  uint32_t *word_dst = reinterpret_cast<uint32_t *>(dst + align_prefix);
  const uint32_t num_words = (bytes - align_prefix) / WORD_BYTES;
  const bool word_src_aligned = (reinterpret_cast<uintptr_t>(word_src) & (WORD_BYTES - 1)) == 0;

  for (uint32_t byte = tid; byte < align_prefix; byte += num_threads) // align dst to a word boundary
  {
    dst[byte] = src[byte];
  }
  for (uint32_t base_word = tid; base_word < num_words; base_word += num_threads * WORDS_PER_THREAD) // bulk
  {
#pragma unroll
    for (uint32_t u = 0; u < WORDS_PER_THREAD; ++u)
    {
      const uint32_t word = base_word + u * num_threads;
      if (word < num_words)
      {
        word_dst[word] = word_src_aligned ? reinterpret_cast<const uint32_t *>(word_src)[word]
                                          : pack_bytes_u32(word_src + word * WORD_BYTES);
      }
    }
  }
  for (uint32_t byte = align_prefix + num_words * WORD_BYTES + tid; byte < bytes; byte += num_threads) // tail
  {
    dst[byte] = src[byte];
  }
}

__device__ __forceinline__ void mark_tile_staged(unsigned char *read_done, size_t global_tile)
{
  cuda::atomic_ref<unsigned char, cuda::thread_scope_device> done(read_done[global_tile]);
  done.store(1, cuda::memory_order_release);
}

__device__ __forceinline__ void wait_until_tile_staged(unsigned char *read_done, size_t global_tile)
{
  cuda::atomic_ref<unsigned char, cuda::thread_scope_device> done(read_done[global_tile]);
  while (done.load(cuda::memory_order_acquire) == 0)
  {
    __nanosleep(50);
  }
}

// Before overwriting this tile's destination range, wait until every tile whose *source* bytes live
// inside that range has finished staging them into its own shared buffer (otherwise this write would
// clobber data another tile still needs to read).
template <size_t DEFRAG_TILE_SIZE, gzipOperatingMode_t GZIP_MODE>
__device__ __forceinline__ void wait_for_overwritten_sources(
  unsigned char *read_done,
  const size_t *deflate_block_dst_sizes,
  const uint64_t *deflate_byte_offsets,
  const size_t global_tile,
  const size_t chunk_tile_start,
  const size_t num_deflates,
  const size_t first_ix_deflate,
  const size_t slot_stride,
  const size_t write_offset,
  const size_t tile_fill
)
{
  const size_t slot_base = gzip_deflate_slot_base<GZIP_MODE>();
  const size_t write_end = write_offset + tile_fill;
  if (write_end <= slot_base)
  {
    return; // destination lies entirely in the reserved gzip header; no source slot overlaps it
  }

  const size_t lo = max(write_offset, slot_base);
  size_t first = min((lo - slot_base) / slot_stride, num_deflates - 1);
  const size_t last = min((write_end - 1 - slot_base) / slot_stride, num_deflates - 1);

  // `lo` may land in the padding gap past `first`'s data; the first live byte is then the next slot.
  size_t first_start = slot_base + first * slot_stride;
  if (lo >= first_start + deflate_block_dst_sizes[first_ix_deflate + first])
  {
    first += 1;
    first_start += slot_stride;
  }
  if (first > last)
  {
    return; // only padding-gap bytes are overwritten -- no live source to protect
  }

  // Packed positions of the first and last overwritten live source byte. dst <= src and packing is
  // contiguous, so these bound one contiguous range of reader tiles.
  const size_t last_start = slot_base + last * slot_stride;
  const size_t packed_start = deflate_byte_offsets[first_ix_deflate + first] + (max(lo, first_start) - first_start);
  const size_t packed_end =
    deflate_byte_offsets[first_ix_deflate + last] +
    (min(write_end, last_start + deflate_block_dst_sizes[first_ix_deflate + last]) - 1 - last_start);

  const size_t packed_base = gzip_deflate_packed_base<GZIP_MODE>();
  const size_t first_tile = (packed_start - packed_base) / DEFRAG_TILE_SIZE;
  const size_t last_tile = (packed_end - packed_base) / DEFRAG_TILE_SIZE;
  for (size_t t = first_tile; t <= last_tile; ++t)
  {
    if (chunk_tile_start + t != global_tile)
    {
      wait_until_tile_staged(read_done, chunk_tile_start + t);
    }
  }
}

static constexpr int DEFRAG_BLOCK_THREADS = 256;
template <size_t DEFRAG_TILE_SIZE, gzipOperatingMode_t GZIP_MODE>
__global__ void __launch_bounds__(DEFRAG_BLOCK_THREADS) defrag_deflate_blocks(
  uint8_t *const *output_buffers,
  const size_t *deflate_block_dst_sizes,
  uint8_t *const *deflate_block_dst_ptrs,
  const uint64_t *deflate_byte_offsets,
  const int *chunk_start_offsets,
  const int *num_deflates_per_chunk,
  const uint64_t *chunk_tile_starts,
  const uint64_t *chunk_tile_counts,
  size_t num_chunks,
  size_t slot_stride,
  unsigned char *read_done
)
{
  extern __shared__ uint8_t staging_buffer[];

  const size_t num_ctas = gridDim.x;

  // Flatten every chunk's tiles into one global, grid-strided tile space. This keeps the grid fully
  // occupied whether there is one huge chunk or thousands of tiny ones.
  const size_t total_global_tiles = chunk_tile_starts[num_chunks - 1] + chunk_tile_counts[num_chunks - 1];

  const size_t num_waves = roundUpDiv(total_global_tiles, num_ctas);

  for (size_t wave = 0; wave < num_waves; ++wave)
  {
    const size_t global_tile = wave * num_ctas + blockIdx.x;
    const bool active = global_tile < total_global_tiles;

    size_t ix_chunk = 0;
    size_t local_tile = 0;
    size_t num_deflates = 0;
    size_t first_ix_deflate_of_this_chunk = 0;
    size_t write_offset = 0;
    size_t tile_fill = 0;
    size_t chunk_tile_start = 0;
    uint8_t *out = nullptr;

    if (active)
    {
      // Map the global tile to its chunk, then to the chunk-local tile.
      ix_chunk = last_index_at_or_before(chunk_tile_starts, 0, num_chunks - 1, global_tile);
      local_tile = global_tile - chunk_tile_starts[ix_chunk];
      chunk_tile_start = chunk_tile_starts[ix_chunk];

      num_deflates = num_deflates_per_chunk[ix_chunk];
      first_ix_deflate_of_this_chunk = chunk_start_offsets[ix_chunk];
      const size_t last_ix_deflate_of_this_chunk = first_ix_deflate_of_this_chunk + num_deflates - 1;
      const size_t destination_defragmented_size = deflate_byte_offsets[last_ix_deflate_of_this_chunk] +
                                                   deflate_block_dst_sizes[last_ix_deflate_of_this_chunk];

      write_offset = gzip_deflate_packed_base<GZIP_MODE>() + local_tile * DEFRAG_TILE_SIZE;
      tile_fill = min(DEFRAG_TILE_SIZE, destination_defragmented_size - write_offset);
      out = output_buffers[ix_chunk];

      // Find the first deflate block at or just before current tile offset
      size_t ix_deflate = last_index_at_or_before(
        deflate_byte_offsets,
        first_ix_deflate_of_this_chunk,
        last_ix_deflate_of_this_chunk,
        write_offset
      );
      size_t deflate_offset = deflate_byte_offsets[ix_deflate];

      // Multiple deflate blocks can fit in single tile
      while (deflate_offset < write_offset + DEFRAG_TILE_SIZE && ix_deflate <= last_ix_deflate_of_this_chunk)
      {
        // source_low - this deflate may be starting in a middle of the tile
        const size_t source_low = max(deflate_offset, write_offset);

        // source_high - this deflate should be read either til the end of the tile or til the end of its size
        const size_t source_high =
          min(write_offset + DEFRAG_TILE_SIZE, deflate_offset + deflate_block_dst_sizes[ix_deflate]);

        // These three are all bounded by DEFRAG_TILE_SIZE / one deflate block, so 32-bit keeps the
        // inner-loop register footprint small even though the surrounding offsets must stay 64-bit.
        const uint32_t bytes_from_this_deflate = static_cast<uint32_t>(source_high - source_low);
        const uint32_t start_read_src_offset = static_cast<uint32_t>(source_low - deflate_offset);
        const uint32_t start_write_staging_offset = static_cast<uint32_t>(source_low - write_offset);

        assert(start_write_staging_offset + bytes_from_this_deflate <= DEFRAG_TILE_SIZE);

        const uint8_t *src = deflate_block_dst_ptrs[ix_deflate];
        copy_bytes(
          staging_buffer + start_write_staging_offset,
          src + start_read_src_offset,
          bytes_from_this_deflate,
          threadIdx.x,
          blockDim.x
        );

        ix_deflate += 1;
        if (ix_deflate <= last_ix_deflate_of_this_chunk)
        {
          deflate_offset = deflate_byte_offsets[ix_deflate];
        }
      }
    }

    __syncthreads(); // staging_buffer is populated before thread 0 publishes the source-read flag
    if (threadIdx.x == 0 && active)
    {
      mark_tile_staged(read_done, global_tile);
      wait_for_overwritten_sources<DEFRAG_TILE_SIZE, GZIP_MODE>(
        read_done,
        deflate_block_dst_sizes,
        deflate_byte_offsets,
        global_tile,
        chunk_tile_start,
        num_deflates,
        first_ix_deflate_of_this_chunk,
        slot_stride,
        write_offset,
        tile_fill
      );
    }

    __syncthreads();
    if (active)
    {
      copy_bytes(out + write_offset, staging_buffer, static_cast<uint32_t>(tile_fill), threadIdx.x, blockDim.x);
    }
    __syncthreads(); // writes from shmem complete before the next wave reuses it (per-CTA)
  }
}
} // namespace nvcomp
