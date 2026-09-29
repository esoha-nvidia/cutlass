/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <atomic>
#include <cassert>

#include <cooperative_groups.h>

#define IS_ZSTD_DECOMP 1
// #define PRINT_INIT_FSE_TIMING_INFO 1

#include "ans.cuh"
#include "common.h"
#include "constants.cuh"
#include "CudaUtils.h"
#include "device_guard.h"
#include "EntropyTables.cuh"
#include "exception.hpp"
#include "huffman.cuh"
#include "io.cuh"
#include "lz.cuh"
#include "types.cuh"
#include "utils.cuh"
#include "zstd_kernel_utils.cuh"
#include "zstdKernels.cuh"

namespace cg = cooperative_groups;
using namespace nvcomp;
using nvcomp::DeviceGuard;

namespace zstd
{

DecompressionScratchHandle::DecompressionScratchHandle(
  uint8_t *tmp_buffer,
  size_t frame_count,
  size_t input_tmp_buffer_size
)
{
  offset_tmp_buffer = tmp_buffer;

  tmp_buffer_loc = reinterpret_cast<uint64_t *>(offset_tmp_buffer);
  offset_tmp_buffer += sizeof(uint64_t);

  global_block_count = reinterpret_cast<int *>(offset_tmp_buffer);
  offset_tmp_buffer += sizeof(int);

  global_huff_block_count = reinterpret_cast<int *>(offset_tmp_buffer);
  offset_tmp_buffer += sizeof(int);

  global_ans_block_count = reinterpret_cast<int *>(offset_tmp_buffer);
  offset_tmp_buffer += sizeof(int);

  global_ix_frame_lz = reinterpret_cast<int *>(offset_tmp_buffer);
  offset_tmp_buffer += sizeof(int);

  global_ix_block_lz = reinterpret_cast<int *>(offset_tmp_buffer);
  offset_tmp_buffer += sizeof(int);

  global_ix_block_ans = reinterpret_cast<int *>(offset_tmp_buffer);
  offset_tmp_buffer += sizeof(int);

  global_ix_block_huff = reinterpret_cast<int *>(offset_tmp_buffer);
  offset_tmp_buffer += sizeof(int);

  global_ix_block_ans_init = reinterpret_cast<int *>(offset_tmp_buffer);
  offset_tmp_buffer += sizeof(int);

  global_ix_block_huff_init = reinterpret_cast<int *>(offset_tmp_buffer);
  offset_tmp_buffer += sizeof(int);

  // Note:
  // Each nvCOMP Zstd chunk contains one Zstd frame
  // More: https://github.com/facebook/zstd/blob/dev/doc/zstd_compression_format.md#zstandard-frames
  zstd_frames = roundUpToAlignment<ZstdFrame>(offset_tmp_buffer);
  offset_tmp_buffer = reinterpret_cast<uint8_t *>(zstd_frames + frame_count);

  // Note:
  // Each Zstd frame must have at least one block, but can have arbitrary many.
  // Pre-allocating only `frame_count` number of blocks is by design, we'll dynamically
  // allocate more if needed. This way, we don't need to re-count the number of blocks, then
  // copy their number back to host.
  // More: https://github.com/facebook/zstd/blob/dev/doc/zstd_compression_format.md#blocks
  block_shares = roundUpToAlignment<DeviceBlockShare>(offset_tmp_buffer);
  offset_tmp_buffer = reinterpret_cast<uint8_t *>(block_shares + frame_count);

  assert((offset_tmp_buffer - tmp_buffer) * sizeof(uint8_t) <= input_tmp_buffer_size);
  tmp_buffer_size = input_tmp_buffer_size - (offset_tmp_buffer - tmp_buffer);

  zstd_frame_count = frame_count;
}

__global__ void
init_buffer_vals(const int frame_count, const unsigned int zero_size, uint8_t *zero_loc, int *block_count)
{
  assert(blockDim.x >= zero_size);
  if (threadIdx.x < zero_size)
  {
    zero_loc[threadIdx.x] = 0;
  }

  if (threadIdx.x == 0)
  {
    *block_count = frame_count;
  }
}

void DecompressionScratchHandle::init_values(cudaStream_t stream)
{
  constexpr unsigned int zero_out_size_bytes = static_cast<unsigned int>(sizeof(uint64_t) + 9 * sizeof(int));
  constexpr size_t dim_size = ((zero_out_size_bytes + WARP_SIZE_U - 1) / WARP_SIZE_U) * WARP_SIZE_U;
  init_buffer_vals<<<1, dim_size, 0, stream>>>(
    narrow_cast<int>(zstd_frame_count),
    zero_out_size_bytes,
    reinterpret_cast<uint8_t *>(tmp_buffer_loc),
    global_block_count
  );
  CUDA_CHECK(cudaGetLastError());
}

size_t DecompressionScratchHandle::total_static_size(const size_t frame_count)
{
  // Note: We create a dummy handle so that we can query the actually used size that
  //       is known without parsing device-side buffers.
  //       `std::numeric_limits<size_t>::max()` denotes the fictive temporary scratch buffer
  //       size and it is only used so that no integer overflow happens internally.
  //       It does not affect the returned static size.
  DecompressionScratchHandle tmp(nullptr, frame_count, std::numeric_limits<size_t>::max());
  return std::numeric_limits<size_t>::max() - tmp.tmp_buffer_size;
}

inline __device__ void
do_raw_block_copy(const DeviceBlockShare &block_share, uint8_t *const current_gpu_dst, const bool lit_only)
{
  // Do RLE for the entire block
  if (block_share.is_rle_block or block_share.is_literal_rle_block)
  {
    for (int ix_byte = thread_warp_ix(); ix_byte < block_share.decomp_block_size; ix_byte += WARP_SIZE)
    {
      current_gpu_dst[ix_byte] = block_share.rle_byte;
    }
  }
  else
  {
    // TODO: more efficient copy
    if (not block_share.is_raw_block)
    {
      // Waiting for huffman to finish
      int decode_lit_count = 0;
#ifdef STAGE_LOGGING
      if (block_share.decode_lit_count < block_share.num_literals)
      {
        print0(
          "bid %d ix warp %d raw block copy waiting to finish frame %d block %d num lits %d clock %lu\n",
          blockIdx.x,
          ix_warp(),
          block_share.ix_frame,
          block_share.ix_block,
          block_share.num_literals,
          cuda::std::chrono::system_clock::now()
        );
      }
#endif
      wait_for_atomic(block_share.decode_lit_count, block_share.num_literals, decode_lit_count, ZSTD_SHORT_SLEEP_NS);
    }

    const uint8_t *copy_source = block_share.is_raw_block or not block_share.is_huff_literal_block
                                   ? block_share.copy_buffer_ptr
                                   : block_share.literal_buffer;
    for (int ix_byte = thread_warp_ix(); ix_byte < block_share.decomp_block_size; ix_byte += WARP_SIZE)
    {
      current_gpu_dst[ix_byte] = copy_source[ix_byte];
    }
  }
}

inline __device__ uint8_t *
get_block_start_ptr(const DeviceBlockShare &block_share, ZstdFrame *frames, uint8_t *const *output_frames)
{
  // Loop back to beginning of frame, compute starting location
  const int ix_frame = block_share.ix_frame;
  const int ix_block = block_share.ix_block;
  auto &frame = frames[ix_frame];
  DeviceBlockShare *block_share_ptr = frame.block_share;
  uint8_t *current_gpu_dst = output_frames[ix_frame];
  for (int ix = 0; ix < ix_block; ++ix)
  {
    const auto &iter_block_share = *block_share_ptr;
    if (not iter_block_share.is_fse_block)
    {
      current_gpu_dst += iter_block_share.decomp_block_size;
    }
    else
    {
      int decode_seq_count = 0;
#ifdef STAGE_LOGGING
      if (iter_block_share.decode_seq_count < iter_block_share.num_sequences)
      {
        print0(
          "bid %d warp %d waiting for all prev blocks to finish FSE %d %d frame %d block %d clock %lu\n",
          blockIdx.x,
          ix_warp(),
          iter_block_share.decode_seq_count.load(),
          iter_block_share.num_sequences,
          iter_block_share.ix_frame,
          iter_block_share.ix_block,
          cuda::std::chrono::system_clock::now()
        );
      }
#endif
      wait_for_atomic(
        iter_block_share.decode_seq_count,
        iter_block_share.num_sequences,
        decode_seq_count,
        ZSTD_SHORT_SLEEP_NS
      );
      current_gpu_dst += iter_block_share.num_literals + iter_block_share.total_seq_bytes;
    }
    block_share_ptr = iter_block_share.next_block_share;
  }

  return current_gpu_dst;
}

// returns lz_status
inline __device__ int coordinate_start_lz_block(DeviceBlockShare &block_share, const bool lit_only)
{
  // Real value will be set by first thread in warp.
  // If default value was set (e.g. to 0), then the exeuction is up to 3% slower. TODO: investigate why
  int lz_status;
  if (lit_only)
  {
    if (thread_warp_ix() == 0)
    {
      lz_status = block_share.lz_status.fetch_or(LZ_STATUS_COPY_STARTED_BIT, cuda::std::memory_order_relaxed);
    }
    lz_status = __shfl_sync(WARP_ALL, lz_status, 0);
  }
  else
  {
    if (thread_warp_ix() == 0)
    {
      lz_status = block_share.lz_status.fetch_or(LZ_STATUS_FULL_STARTED_BIT, cuda::std::memory_order_acquire);
    }
    lz_status = __shfl_sync(WARP_ALL, lz_status, 0);
    if (lz_status != 0)
    {
      // We have a helper precomputing some of the work. Wait for it to finish
      wait_for_atomic(block_share.lz_status, LZ_STATUS_COPY_FINISHED_BIT, lz_status, ZSTD_SHORT_SLEEP_NS);
    }
  }

  return lz_status;
}

inline __device__ void warp_lz_copy(
  ZstdFrame *frames,
  DeviceBlockShare *block_shares,
  size_t *frame_decomp_sizes,
  int32_t *repeat_offset,
  int *global_lz_ix,
  int &shared_ix,
  uint8_t *const *output_frames,
  const int frame_count,
  const int block_count
)
{
  bool lit_only = false;
  uint8_t *current_gpu_dst;
  while (true)
  {
    increment_ix(shared_ix, global_lz_ix);

    if (shared_ix >= block_count)
    {
      return;
    }
    else if (shared_ix >= frame_count)
    {
      lit_only = true;
    }
#ifdef PRINT_TIMING
    print0("block %d frame %d global frame %d warp %d\n", blockIdx.x, shared_ix_frame, *global_ix_frame, ix_warp());
#endif

    DeviceBlockShare *block_share_ptr;
    if (lit_only)
    {
      block_share_ptr = &block_shares[shared_ix];
    }
    else
    {
      auto &frame = frames[shared_ix];
      block_share_ptr = frame.block_share;
      frame_decomp_sizes[shared_ix] = 0;
      current_gpu_dst = output_frames[shared_ix];
    }

    if (not lit_only)
    {
      reinit_repeat_codes(repeat_offset); // unused for literal copy
    }
    while (block_share_ptr != nullptr)
    { // We'll immediately break out of this for lit copy
      auto &block_share = *block_share_ptr;
      int lz_status = coordinate_start_lz_block(block_share, lit_only);

      if (lit_only)
      {
        if (lz_status & LZ_STATUS_FULL_STARTED_BIT)
        {
          break;
        }

        current_gpu_dst = get_block_start_ptr(block_share, frames, output_frames);
      }

      if (not block_share.is_fse_block)
      {
        if (lit_only or not(lz_status & LZ_STATUS_COPY_STARTED_BIT))
        {
          do_raw_block_copy(block_share, current_gpu_dst, lit_only);
          if (lit_only)
          {
            block_share.lz_status.fetch_or(LZ_STATUS_COPY_FINISHED_BIT, cuda::std::memory_order_release);
            break; // breaks out of the inner while loop -- very complicated but needed to avoid redundant giant inlined functions
          }
        }

        if (thread_warp_ix() == 0)
        {
          frame_decomp_sizes[block_share.ix_frame] += block_share.decomp_block_size;
        }
        block_share_ptr = block_share.next_block_share;
        current_gpu_dst += block_share.decomp_block_size;
        continue;
      }

#ifdef STAGE_LOGGING
      if (blockIdx.x == 0)
      {
        print0(
          "decompressing sequences for block %d num sequences %u ix frame %d ix warp %d clock %lu\n",
          block_share.ix_block,
          block_share.num_sequences,
          block_share.ix_frame,
          ix_warp(),
          cuda::std::chrono::system_clock::now()
        );
      }
#endif

#ifdef PRINT_TIMING
      print0(
        "decompressing sequences for ix frame %d block %d num sequences %u ix warp %d clock %lu\n",
        block_share.ix_frame,
        block_share.ix_block,
        block_share.num_sequences,
        ix_warp(),
        cuda::std::chrono::system_clock::now()
      );
#endif

      const bool is_rle_literals = block_share.is_literal_rle_block;

      const uint8_t *literal_buffer = block_share.is_huff_literal_block ? block_share.literal_buffer
                                                                        : block_share.copy_buffer_ptr;
      const bool seq_only = lz_status & LZ_STATUS_COPY_STARTED_BIT;

      decompress_block_lz4(
        current_gpu_dst,
        block_share,
        literal_buffer,
        block_share.rle_byte,
        repeat_offset,
        lit_only,
        seq_only,
        is_rle_literals
      );

      if (lit_only)
      {
        block_share.lz_status.fetch_or(LZ_STATUS_COPY_FINISHED_BIT, cuda::std::memory_order_release);
        break;
      }

#ifdef STAGE_LOGGING
      print0(
        "decompressed sequences for ix frame %d block %d num sequences %u ix warp %d clock %lu\n",
        block_share.ix_frame,
        block_share.ix_block,
        block_share.num_sequences,
        ix_warp(),
        cuda::std::chrono::system_clock::now()
      );
#endif

#ifdef ANS_PRECOMPUTE_OFFSET
      // Update the repeat offset for use in subsequent blocks.
      if (thread_warp_ix() < 3)
      {
        if (block_share.ix_block == 0)
        {
          repeat_offset[thread_warp_ix()] = block_share.repeat_offsets[thread_warp_ix()];
        }
        else
        {
          unsigned tiny_mask = (1 << 3) - 1;
          int this_repeat_offset = block_share.repeat_offsets[thread_warp_ix()];
          if (this_repeat_offset < 0)
          {
            this_repeat_offset = repeat_offset[abs(this_repeat_offset) - 1]; // -1 to zero-index
          }
          __syncwarp(tiny_mask);
          repeat_offset[thread_warp_ix()] = this_repeat_offset;
        }
      }
#endif
      int cached_seq_val = 0;
      // decode seq count isn't valid until
      wait_for_atomic(block_share.decode_seq_count, block_share.num_sequences, cached_seq_val, ZSTD_LONG_SLEEP_NS);

      const int block_size = block_share.num_literals + block_share.total_seq_bytes;
      current_gpu_dst += block_size;
      if (thread_warp_ix() == 0)
      {
        frame_decomp_sizes[block_share.ix_frame] += block_size;
      }

      block_share_ptr = block_share.next_block_share;
    }
  }
}

__device__ void do_decompress(
  int *global_lz_ix_frame,
  int *global_lz_ix_block,
  int *global_ans_ix_block,
  int *global_huff_ix_block,
  const int global_block_count,
  ZstdFrame *frames,
  DeviceBlockShare *block_shares,
  size_t *frame_decomp_sizes,
  const size_t frame_count,
  uint8_t *const *output_frames
)
{
  __shared__ MetaHuffmanTable huff_tables[NUM_HUFF_WARPS_PER_CTA];
  __shared__ int completion_count;
  if (threadIdx.x == 0)
  {
    completion_count = 0;
  }
  __shared__ SharedExtraBitTables extra_tables;
  extra_tables.init();
  __syncthreads();

  // Assess the blocks and copy
  int num_fse_tables = 0;
  uint32_t *this_table_buff = nullptr;
  bool do_fse = ix_warp() < NUM_FSE_WARPS_PER_CTA;
  if (ix_warp() < NUM_FSE_WARPS_PER_CTA)
  {
    __shared__ extern uint32_t table_buff[];
    num_fse_tables = MAX_TOTAL_FSE_TABLES / NUM_FSE_WARPS_PER_CTA;
    int ix_table = ix_warp() * num_fse_tables;
    int rem_tables = MAX_TOTAL_FSE_TABLES % NUM_FSE_WARPS_PER_CTA;
    num_fse_tables += ix_warp() < rem_tables ? 1 : 0;
    ix_table += ix_warp() < rem_tables ? ix_warp() : rem_tables;
    this_table_buff = &table_buff[ix_table * max_fse_tables_words];
  }
  else if (ix_warp() < NUM_FSE_WARPS_PER_CTA + NUM_HUFF_WARPS_PER_CTA)
  {
    // Get a literal buffer
    warp_entropy_literal_decoding(
      block_shares,
      huff_tables[ix_warp() - NUM_FSE_WARPS_PER_CTA],
      global_huff_ix_block,
      global_block_count
    );

    if constexpr (MAX_FSE_WARPS_PER_CTA > NUM_FSE_WARPS_PER_CTA)
    {
      int this_completion_count = 0;
      if (thread_warp_ix() == 0)
      {
        this_completion_count = atomicAdd(&completion_count, 1);
      }
      this_completion_count = __shfl_sync(WARP_ALL, this_completion_count, 0);
      if (this_completion_count == NUM_HUFF_WARPS_PER_CTA - 1)
      {
        // Now take the entire Huff table buffer and do as many tables as we can with the last warp to finish huffman
        num_fse_tables = min(MAX_ANS_TABLES, 4);
        this_table_buff = reinterpret_cast<uint32_t *>(&huff_tables);
        do_fse = true;
      }
    }
  }

  if (do_fse)
  {
    warp_entropy_sequence_decoding(
      block_shares,
      this_table_buff,
      extra_tables,
      num_fse_tables,
      global_ans_ix_block,
      global_block_count
    );
  }

  __shared__ int repeat_codes_shared[NUM_WARPS_PER_CTA][3];
  __shared__ int lz_shared_ix_frame[NUM_WARPS_PER_CTA];

  __syncwarp(); // Only shift to LZ when all threads in the warp have finished entropy decoding
// Do LZ on unoptimized frames.
#ifdef STAGE_LOGGING
  if (do_fse)
  {
    print0(
      "bid %d warp %d finished fse decode clock %lu\n",
      blockIdx.x,
      ix_warp(),
      cuda::std::chrono::system_clock::now()
    );
  }
  else if (ix_warp() < NUM_FSE_WARPS_PER_CTA + NUM_HUFF_WARPS_PER_CTA)
  {
    print0(
      "bid %d warp %d finished huff decode clock %lu\n",
      blockIdx.x,
      ix_warp(),
      cuda::std::chrono::system_clock::now()
    );
  }
#endif

  warp_lz_copy(
    frames,
    block_shares,
    frame_decomp_sizes,
    &repeat_codes_shared[ix_warp()][0],
    global_lz_ix_frame,
    lz_shared_ix_frame[ix_warp()],
    output_frames,
    frame_count,
    global_block_count
  );
}

__launch_bounds__(NUM_WARPS_PER_CTA *WARP_SIZE, 1) __global__ void decompression_kernel(
  uint8_t *const *output_frames,
  size_t *frame_decomp_sizes,
  const size_t *input_decomp_sizes,
  size_t frame_count,
  ZstdFrame *frames,
  DeviceBlockShare *block_shares,
  int *global_lz_ix_frame,
  int *global_lz_ix_block,
  int *global_ans_ix_block,
  int *global_huff_ix_block,
  const int *global_block_count_ptr,
  const int *global_huff_block_count_ptr,
  const int *global_ans_block_count_ptr
)
{
  const int global_block_count = *global_block_count_ptr;
  int early_return = 0;

  if (thread_warp_ix() == 0 and *global_lz_ix_frame >= frame_count and *global_ans_ix_block >= global_block_count and
      *global_huff_ix_block >= global_block_count)
  {
    early_return = true;
  }

  if (__shfl_sync(WARP_ALL, early_return, 0))
  {
    return;
  }

#ifdef STAGE_LOGGING
  if (blockIdx.x == 0 and threadIdx.x == 0)
  {
    printf(
      "huff blocks %d ans blocks %d total %d clock %lu\n",
      *global_huff_block_count_ptr,
      *global_ans_block_count_ptr,
      *global_block_count_ptr,
      cuda::std::chrono::system_clock::now()
    );
  }
#endif // STAGE_LOGGING

  do_decompress(
    global_lz_ix_frame,
    global_lz_ix_block,
    global_ans_ix_block,
    global_huff_ix_block,
    global_block_count,
    frames,
    block_shares,
    frame_decomp_sizes,
    frame_count,
    output_frames
  );
}

__device__ __noinline__ void do_process_sequences_for_frame_sizes(
  DeviceBlockShare *block_shares,
  const int num_blocks,
  FSETables *fse_table_array,
  ANSTableConstructionBuffers &ans_buffers,
  FSETables *&prev_tables
)
{
  __shared__ Uint8WarpScan::TempStorage uint8_storage;
  Uint8WarpScan uint8_warp_scan{uint8_storage};

  __shared__ Uint16WarpScan::TempStorage uint16_storage;
  Uint16WarpScan uint16_warp_scan{uint16_storage};

  __shared__ WarpReduceUint16::TempStorage uint16_reduce_storage;
  WarpReduceUint16 uint16_warp_reduce{uint16_reduce_storage};

  __shared__ unsigned shared_words[2 * FRAME_SIZES_MAX_ANS_TABLES];
  for (int ix_block = 0; ix_block < num_blocks; ++ix_block)
  {
    auto &block_share = block_shares[ix_block];
    if (block_share.ix_block == 0)
    {
      prev_tables = nullptr;
    }

    FSETables *this_tables = &fse_table_array[ix_block];

    decode_sequence_tables(
      block_share.compressed_buffer,
      block_share.comp_block_size,
      ans_buffers,
      block_share.block_header,
      prev_tables,
      *this_tables,
      uint8_warp_scan,
      uint16_warp_scan,
      uint16_warp_reduce
    );
    prev_tables = this_tables;
  }

  assert(num_blocks <= FRAME_SIZES_MAX_ANS_TABLES);
  unsigned num_active_threads = num_blocks * NUM_THREADS_PER_ANS_BLOCK;
  unsigned full_fse_mask = (1 << num_active_threads) - 1;
  auto in_warp_idx = thread_warp_ix();
  if (in_warp_idx < num_active_threads)
  {
    int ix_block = in_warp_idx / NUM_THREADS_PER_ANS_BLOCK;
    int ix_in_block = in_warp_idx % NUM_THREADS_PER_ANS_BLOCK;
    ANSCache ans_cache{ix_block};
    unsigned partial_fse_mask = ix_in_block < 3;
    partial_fse_mask = __ballot_sync(full_fse_mask, partial_fse_mask);
    unsigned *this_shared_words = &shared_words[ix_block * 2];
    uint8_t *this_reg = reinterpret_cast<uint8_t *>(this_shared_words) + ix_in_block;

    FSETables &fse_tables = fse_table_array[ix_block];
    // Do FSE on this one
    auto &block_share = block_shares[ix_block];
    auto &block_header = block_share.block_header;
    BitReader bit_reader{block_share.compressed_buffer, block_share.comp_block_size};
    bit_reader.get_read_pointer(block_header.seq_section_start);

    decode_sequences_for_frame_sizes(
      bit_reader,
      block_share.num_sequences,
      fse_tables,
      block_header,
      block_share,
      ix_in_block,
      ix_block,
      block_share.total_seq_bytes,
      ans_cache,
      this_shared_words,
      this_reg
    );
  }
  __syncwarp();
}

// TODO: Check the magic number in the zstd frame and if it is not present or incorrect, return 0 for uncompressed chunk sizes
__launch_bounds__(32, 10) __global__
  void get_frame_sizes(const uint8_t *const *input_frames, size_t *frame_sizes, int frame_count)
{
  assert(blockDim.x == WARP_SIZE_U); // code below assumes there is one warp on x dimension
  __shared__ uint8_t __align__(8)
    share_array[PER_BLOCK_ANS_SHARE_SIZE * FRAME_SIZES_MAX_ANS_TABLES + sizeof(MemoryManager)];
  MemoryManager *shared_memory_manager_ptr = (MemoryManager *)share_array;
  FSETables *prev_fse_tables = nullptr;

  if (threadIdx.x == 0)
  {
    // Allocate sequences
    *shared_memory_manager_ptr =
      MemoryManager{share_array + sizeof(MemoryManager), PER_BLOCK_ANS_SHARE_SIZE * FRAME_SIZES_MAX_ANS_TABLES};
  }
  __syncwarp(); // Need the memory manager available everywhere

  __shared__ ANSTableConstructionBuffers ans_buffers;

  auto &shared_memory_manager = *shared_memory_manager_ptr;
  __shared__ FSETables fse_tables[FRAME_SIZES_MAX_ANS_TABLES];

  allocate_fse_tables(shared_memory_manager, fse_tables, FRAME_SIZES_MAX_ANS_TABLES);

  __syncthreads();

  // For now, have each block take a frame.
  __shared__ int ix_frame;
  if (threadIdx.x == 0)
  {
    ix_frame = blockIdx.x;
  }
  __syncwarp(); // Need ix_frame below

  __shared__ DeviceBlockShare block_shares[FRAME_SIZES_MAX_ANS_TABLES];
  const __shared__ uint8_t *comp_buffer;

  while (ix_frame < frame_count)
  {
    size_t frame_size;
    if (threadIdx.x == 0)
    {
      frame_sizes[ix_frame] = 0;
      comp_buffer = input_frames[ix_frame];
      frame_size = parse_frame_header(comp_buffer);
    }

    if (__shfl_sync(WARP_ALL, frame_size, 0) > 0)
    {
      // If not zero, we're done with this frame
      if (threadIdx.x == 0)
      {
        frame_sizes[ix_frame] = frame_size;
        ix_frame += gridDim.x;
      }
      __syncwarp();
      continue;
    }

    __syncwarp();

    int ix_block_share = 0;
    int ix_frame_block = 0;
    while (true)
    {
      auto &block_share = block_shares[ix_block_share];

      // TODO: prev table values unused but can just be filled in here for now.
      bool last_block = classify_blocks(comp_buffer, block_share);
      __syncwarp(); // Necessary because some threads might return from classify blocks early.
      if (threadIdx.x == 0)
      {
        if (not block_share.is_compressed_block)
        {
          frame_sizes[ix_frame] += block_share.decomp_block_size;
        }
        else if (not block_share.is_fse_block)
        {
          frame_sizes[ix_frame] += block_share.num_literals;
        }
        else
        {
          block_share.input_frame = input_frames[ix_frame];
          block_share.ix_frame = ix_frame;
          block_share.ix_block = ix_frame_block;
#ifdef STAGE_LOGGING
          print0(
            "frame %d block %d num seq %d num lit %d\n",
            ix_frame,
            ix_frame_block,
            block_share.num_sequences,
            block_share.num_literals
          );
#endif
          ++ix_block_share;
        }
      }
      __syncwarp();
      ix_block_share = __shfl_sync(WARP_ALL, ix_block_share, 0);

      ++ix_frame_block;

      if (ix_block_share == FRAME_SIZES_MAX_ANS_TABLES or (last_block and ix_block_share > 0))
      {
        do_process_sequences_for_frame_sizes(block_shares, ix_block_share, fse_tables, ans_buffers, prev_fse_tables);

        __syncwarp(); // Necessary to ensure all blocks have been computed before the next loop
        if (threadIdx.x == 0)
        {
          for (int ix = 0; ix < ix_block_share; ++ix)
          {
            frame_sizes[ix_frame] += block_shares[ix].total_seq_bytes + block_shares[ix].num_literals;
          }
        }
        ix_block_share = 0;
      }

      if (last_block)
      {
        break;
      }

      if (threadIdx.x == 0)
      {
        comp_buffer += block_share.comp_block_size;
      }
      __syncwarp();
    }

    if (threadIdx.x == 0)
    {
      ix_frame += gridDim.x;
    }
    __syncwarp();
  }
}

__global__ void init_fse_tables(
  uint8_t *tmp_buffer,
  uint64_t *tmp_buffer_loc,
  size_t tmp_buffer_size,
  int *global_ix_block_table_init,
  const int *global_block_count,
  DeviceBlockShare *block_shares
)
{
  __shared__ int ix_block_share[INIT_FSE_WARPS_PER_CTA];

  increment_ix(ix_block_share[ix_warp()], global_ix_block_table_init);

  DeviceBlockShare *block_share_ptr = nullptr;
  if (ix_block_share[ix_warp()] >= *global_block_count)
  {
    return;
  }

  block_share_ptr = &block_shares[ix_block_share[ix_warp()]];

  __shared__ ANSTableConstructionBuffers ans_buffers[INIT_FSE_WARPS_PER_CTA];

  TmpBufferManager tmp_buffer_manager{tmp_buffer_loc, tmp_buffer, tmp_buffer_size};

  __shared__ Uint8WarpScan::TempStorage uint8_storage[INIT_FSE_WARPS_PER_CTA];
  Uint8WarpScan uint8_warp_scan{uint8_storage[ix_warp()]};

  __shared__ Uint16WarpScan::TempStorage uint16_storage[INIT_FSE_WARPS_PER_CTA];
  Uint16WarpScan uint16_warp_scan{uint16_storage[ix_warp()]};

  __shared__ WarpReduceUint16::TempStorage uint16_reduce_storage[INIT_FSE_WARPS_PER_CTA];
  WarpReduceUint16 uint16_warp_reduce{uint16_reduce_storage[ix_warp()]};

  // Did we process the previous block as well within the same warp?
  bool previous_block_processed = false;
  __shared__ FSETables *global_fse_tables[INIT_FSE_WARPS_PER_CTA];

  // Note:
  // We want to keep track of the previous compressed FSE block,
  // because, when the compression mode is repeat mode (0x11, 3),
  // then the table in the previous compressed block with Number_of_Sequences > 0
  // will be used, i.e., when `.is_fse_block` == true.
  DeviceBlockShare *prev_fse_block_share_ptr = nullptr;

  while (block_share_ptr != nullptr)
  {
    auto &block_share = *block_share_ptr;

    // Allocate buffers
    // This block repeats the previous compressed block's OR the dictionary's table
    // Note: there's no dictionary support as of now in Zstd GPU
    bool this_block_repeats = true;
    if (block_share.is_fse_block)
    {
      const uint8_t *input = block_share.compressed_buffer;
      const int offset = block_share.block_header.seq_section_start;

      BitReader bit_reader{input + offset, block_share.comp_block_size - offset};
      uint8_t compression_modes = bit_reader.read_bits(8);
      uint8_t literal_length_mode = (compression_modes >> 6) & 3;
      uint8_t offset_mode = (compression_modes >> 4) & 3;
      uint8_t match_length_mode = (compression_modes >> 2) & 3;

      // Note:
      // A value of 3 (0x11) indicates Repeat_Mode,
      // see "Symbol compression modes" in the Zstd. RFC
      // If this is the first block, then the table in the dictionary will be used.
      // TODO(bnagy): revise logic when dictionary support is added to Zstd
      this_block_repeats = (literal_length_mode == 3 or offset_mode == 3 or match_length_mode == 3);

      if (previous_block_processed xor not this_block_repeats)
      {
        // When:
        // - we processed the previous block AND this block repeats (i.e. we have something to copy from for sure)
        // - we did not process the previous block AND this block does not repeat -> we are the eligible warp to process this
        __syncwarp();
        if (thread_warp_ix() == 0)
        {
          block_share.sequence_buffer = tmp_buffer_manager.allocate<sequence>(block_share.num_sequences);

          global_fse_tables[ix_warp()] = tmp_buffer_manager.allocate<FSETables>(1);
          allocate_fse_table(
            global_fse_tables[ix_warp()]->ll_fse_table,
            tmp_buffer_manager,
            LITERAL_LENGTH_MAX_ACCURACY,
            0
          );
          allocate_fse_table(global_fse_tables[ix_warp()]->of_fse_table, tmp_buffer_manager, OFFSET_MAX_ACCURACY, 1);
          allocate_fse_table(
            global_fse_tables[ix_warp()]->ml_fse_table,
            tmp_buffer_manager,
            MATCH_LENGTH_MAX_ACCURACY,
            2
          );
          block_share.global_fse_tables = global_fse_tables[ix_warp()];
        }

        __syncwarp();

        // literal
        decode_seq_table(
          bit_reader,
          ans_buffers[ix_warp()],
          literal_length_mode,
          SequenceType::LiteralLength,
          global_fse_tables[ix_warp()]->ll_fse_table,
          prev_fse_block_share_ptr ? &prev_fse_block_share_ptr->global_fse_tables->ll_fse_table : nullptr,
          uint8_warp_scan,
          uint16_warp_scan,
          uint16_warp_reduce,
          true /*build table*/
        );

        // offset
        decode_seq_table(
          bit_reader,
          ans_buffers[ix_warp()],
          offset_mode,
          SequenceType::Offset,
          global_fse_tables[ix_warp()]->of_fse_table,
          prev_fse_block_share_ptr ? &prev_fse_block_share_ptr->global_fse_tables->of_fse_table : nullptr,
          uint8_warp_scan,
          uint16_warp_scan,
          uint16_warp_reduce,
          true /*build table*/
        );

        // match length
        decode_seq_table(
          bit_reader,
          ans_buffers[ix_warp()],
          match_length_mode,
          SequenceType::MatchLength,
          global_fse_tables[ix_warp()]->ml_fse_table,
          prev_fse_block_share_ptr ? &prev_fse_block_share_ptr->global_fse_tables->ml_fse_table : nullptr,
          uint8_warp_scan,
          uint16_warp_scan,
          uint16_warp_reduce,
          true /*build table*/
        );

        const unsigned final_rem_bytes = bit_reader.rem_bytes();
        block_share.block_header.block_seq_table_size = block_share.comp_block_size - offset - final_rem_bytes;
      }
      prev_fse_block_share_ptr = block_share_ptr;
    }
    block_share_ptr = block_share.next_block_share;
    if (block_share_ptr == nullptr or (this_block_repeats xor previous_block_processed))
    {
      // Start a new block (this might be in a different frame) with the warp
      // When:
      // - there are no more blocks in the current frame
      // - we processed the previous block AND this block does not repeat -> another warp processed this block
      // - we did NOT process the previous block AND this block does repeat -> another warp processed this block
      increment_ix(ix_block_share[ix_warp()], global_ix_block_table_init);
      if (ix_block_share[ix_warp()] < *global_block_count)
      {
        block_share_ptr = &block_shares[ix_block_share[ix_warp()]];
      }
      else
      {
        block_share_ptr = nullptr;
      }
      previous_block_processed = false;
      prev_fse_block_share_ptr = nullptr;
    }
    else
    {
      previous_block_processed = true;
    }
  }
}

__global__ void init_huff_tables(
  uint8_t *tmp_buffer,
  uint64_t *tmp_buffer_loc,
  size_t tmp_buffer_size,
  int *global_ix_block_table_init,
  const int *global_block_count,
  DeviceBlockShare *block_shares
)
{
  assert(blockDim.x == WARP_SIZE_U); // code below assumes there is one warp on x dimension
  TmpBufferManager tmp_buffer_manager{tmp_buffer_loc, tmp_buffer, tmp_buffer_size};

  __shared__ int ix_block_share;

  increment_ix(ix_block_share, global_ix_block_table_init);

  DeviceBlockShare *block_share_ptr = nullptr;
  if (ix_block_share >= *global_block_count)
  {
    return;
  }

  block_share_ptr = &block_shares[ix_block_share];

  __shared__ Uint8WarpScan::TempStorage uint8_storage;
  Uint8WarpScan uint8_warp_scan{uint8_storage};

  __shared__ Uint16WarpScan::TempStorage uint16_storage;
  Uint16WarpScan uint16_warp_scan{uint16_storage};

  __shared__ WarpReduceUint16::TempStorage uint16_reduce_storage;
  WarpReduceUint16 uint16_warp_reduce{uint16_reduce_storage};

  // The below invalidates the above pointers.
  constexpr int SHARE_ARRAY_SIZE_TABLES = 550 + sizeof(size_t);

  __shared__ uint8_t share_array[SHARE_ARRAY_SIZE_TABLES];

  __shared__ MemoryManager memory_manager;

  if (threadIdx.x == 0)
  {
    memory_manager = MemoryManager{share_array, SHARE_ARRAY_SIZE_TABLES};
  }

  __shared__ HuffmanTable huff_table;
  __syncwarp();

  // Do Huffman
  __shared__ HuffmanTableConstructionBuffers huff_buffers;
  allocate_fse_table(huff_buffers.fse_table, memory_manager, HUF_FSE_WEIGHT_MAX_ACCURACY_LOG);

  // Did we process the previous block as well within the same warp?
  bool previous_block_processed = false;

  // The high-level design here is, in the nominal case where no blocks use repeated tables,
  // each warp on each iteration decodes the huffman table for a single ZSTD block before
  // atomically getting another block.
  // However, each warp will always check whether the table repeats before computing the table.
  // The warp skips if it's a repeated table.
  // If it's not a repeated table, compute the table.
  // Then advance to the next block in the frame. If it's a repeated block, we'll fill in what we've already computed.
  // This loop proceeds until it first encounters a block that doesn't repeat.
  // At that point, the warp will atomically increment again.
  while (block_share_ptr != nullptr)
  {
    auto &block_share = *block_share_ptr;

    // When we processed the previous block AND
    // - this block doesn't repeat -> we're done (other warp was eligible for processing it)
    // - this block repeats -> we are going to copy the previous huffman table from shared memory and continue with the next block within the frame (if any)

    // TODO(bnagy): revise logic when dictionary supported is added to Zstd
    bool this_block_repeats = true;
    if (block_share.is_huff_literal_block)
    {

      // TODO(bnagy): revise logic when dictionary support is added
      this_block_repeats = block_share.block_header.literal_header.block_type == LiteralBlockType::Treeless;
      if (not this_block_repeats xor previous_block_processed)
      {
        // When:
        // - we processed the previous block AND this block repeats
        // - we did not process the previous block AND this block does not repeat -> we are the eligible warp to process this
        if (threadIdx.x == 0)
        {
          block_share.literal_buffer = tmp_buffer_manager.allocate(block_share.num_literals);
        }

        if (not this_block_repeats)
        {
          get_huffman_table(
            block_share.compressed_buffer,
            static_cast<unsigned>(block_share.comp_block_size),
            huff_table,
            huff_buffers,
            block_share
          );
        }
        else
        {
          block_share.block_header.literal_header.table_desc_size = 0;
        }

        __syncwarp();
        __shared__ HuffmanTable *global_huff_table;
        if (threadIdx.x == 0)
        {
          global_huff_table = tmp_buffer_manager.allocate<HuffmanTable>(1);
          block_share.global_huff_table = global_huff_table;
        }
        __syncwarp();

        global_huff_table->deepcopy(huff_table);
      }
    }

    block_share_ptr = block_share.next_block_share;
    if (block_share_ptr == nullptr or (previous_block_processed xor this_block_repeats))
    {
      // Start a new block (this might be in a different frame) with the warp
      // When:
      // - there are no more blocks in the current frame
      // - we processed the previous block AND this block does not repeat -> another warp processed this block
      // - we did NOT process the previous block AND this block does repeat -> another warp processed this block
      increment_ix(ix_block_share, global_ix_block_table_init);
      if (ix_block_share < *global_block_count)
      {
        block_share_ptr = &block_shares[ix_block_share];
      }
      else
      {
        block_share_ptr = nullptr;
      }
      previous_block_processed = false;
    }
    else
    {
      previous_block_processed = true;
    }
  }
}

template <size_t block_size>
__launch_bounds__(block_size) __global__ void gather_frame_blocks(
  const size_t frame_count,
  const uint8_t *const *input_frames,
  size_t *block_count,
  size_t *scratch_space,
  nvcompStatus_t *device_statuses
)
{
  int ix_frame = threadIdx.x + blockIdx.x * block_size;

  size_t num_blocks_in_frame = 0;
  size_t scratch_space_for_frame = 0;
  if (ix_frame < frame_count)
  {
    const uint8_t *input_frame = input_frames[ix_frame];
    parse_frame_header(input_frame);

    // mark chunk valid (if applicable)
    if (device_statuses)
    {
      device_statuses += ix_frame;
      *device_statuses = nvcompSuccess;
    }

    // Count blocks per frame and
    // Accumulate the extra scratch space needed
    // Note: every Zstd frame has at least one block
    do
    {
      ++num_blocks_in_frame;
    } while (!gather_block_scratch_requirement(input_frame, scratch_space_for_frame, device_statuses));

    // The static size within DecompressionScratchHandle::total_static_size
    // only includes space for #frames = #blocks, because this is known a priori to
    // decompression. We need to factor in space needed for the extra blocks within
    // the frame.
    scratch_space_for_frame += (num_blocks_in_frame - 1) * sizeof(DeviceBlockShare);
  }

  // Perform CTA-level reduction
  typedef nvcomp::cub::BlockReduce<size_t, block_size> BlockReduceSize_t;
  __shared__ typename BlockReduceSize_t::TempStorage temp_storage_blocks;
  __shared__ typename BlockReduceSize_t::TempStorage temp_storage_scratch;
  // Number of Zstd blocks
  size_t blocks_in_CTA = BlockReduceSize_t(temp_storage_blocks).Sum(num_blocks_in_frame);
  // Total extra scratch space needed
  size_t scratch_space_in_CTA = BlockReduceSize_t(temp_storage_scratch).Sum(scratch_space_for_frame);

  // Perform grid-level reduction
  if (threadIdx.x == 0)
  {
    atomicAddUint64Wrapper(block_count, blocks_in_CTA);
    atomicAddUint64Wrapper(scratch_space, scratch_space_in_CTA);
  }
}

void gather_frame_blocks_api(
  const uint8_t *const *input_frames,
  size_t frame_count,
  size_t *block_count,
  size_t *scratch_space_required,
  nvcompStatus_t *device_statuses,
  cudaStream_t stream
)
{
  // Note:
  // We'll launch one thread per frame (chunk), given this is a really serial
  // task. Each thread will simply walk through one frame, collecting all
  // Zstd blocks within it. Once they finish, they do a CTA sum reduction
  // then an atomic write-out in the grid.

  // TODO(bnagy): When a larger chunk size is used for compression, the assumption that
  //              there aren't many blocks per frame is not true anymore. Change the scheduling
  //              and the kernel such that we launch e.g., a warp per Zstd frame and fast-forward
  //              threads to process individual Zstd blocks.
  constexpr unsigned int block_dim_x = 64;
  dim3 block_size(block_dim_x, 1, 1);
  dim3 grid_size((cuda_dim_cast(frame_count) + block_size.x - 1) / block_size.x, 1, 1);

  // Determine the actual block count & additional scratch space needed:
  // - Each chunk (zstd)frame needs space for one ZstdFrame - this is known a priori
  // - Each block needs space for one DeviceBlockShare - at least one block per ZstdFrame
  // - We need additional scratch space for each block with Huffman trees (is_huff_literal_block)
  // - We need additional scratch space for each block with FSE (is_fse_block)
  gather_frame_blocks<block_dim_x><<<grid_size, block_size, 0, stream>>>(
    frame_count,
    input_frames,
    block_count,
    scratch_space_required,
    device_statuses
  );
  CUDA_CHECK(cudaGetLastError());
}

template <int num_warps_per_block>
__global__ void classify_frames(
  const uint8_t *const *input_frames,
  DeviceBlockShare *block_shares,
  size_t frame_count,
  uint64_t *tmp_buffer_loc,
  size_t tmp_buffer_size,
  int *global_ix_block,
  int *global_huff_block_count,
  int *global_ans_block_count,
  ZstdFrame *zstd_frames,
  const size_t *input_decomp_sizes
)
{
  const __shared__ uint8_t *frame_comp_buffers[num_warps_per_block];
  assert(ix_warp() < num_warps_per_block);
  const uint8_t *&frame_comp_buffer = frame_comp_buffers[ix_warp()];

  // For now, have each warp take a frame.
  int ix_frame = blockIdx.x * num_warps_per_block + ix_warp();
  if (ix_frame >= frame_count)
  {
    return;
  }

  // We can write to zstd frame location. For now, entire warp does this. Allocate one DeviceBlockShare at a time.
  auto &frame = zstd_frames[ix_frame];
  DeviceBlockShare *block_share;

  int ix_block = ix_frame;
  if (thread_warp_ix() == 0)
  {
    frame_comp_buffer = input_frames[ix_frame];
    parse_frame_header(frame_comp_buffer);
  }
  __syncwarp(); // Racecheck FA
  ix_block = __shfl_sync(WARP_ALL, ix_block, 0);
  block_share = &block_shares[ix_block];

  int num_blocks = 0;
  frame.block_share = block_share;

  int total_literals = 0;
  while (true)
  {
    if (thread_warp_ix() == 0)
    {
      block_share->input_frame = input_frames[ix_frame];
      block_share->ix_frame = ix_frame;
      block_share->ix_block = num_blocks++;
    }
    bool last_block = classify_blocks(frame_comp_buffer, *block_share);

    if (thread_warp_ix() == 0)
    {
      if (block_share->is_huff_literal_block)
      {
        total_literals += block_share->num_literals;
        atomicAdd(global_huff_block_count, 1);
      }
      if (block_share->is_fse_block)
      {
        atomicAdd(global_ans_block_count, 1);
      }
    }

    __syncwarp();
    if (last_block)
    {
      block_share->next_block_share = nullptr;
      return;
    }

    // Get the next block share
    if (thread_warp_ix() == 0)
    {
      frame_comp_buffer += block_share->comp_block_size;

      ix_block = static_cast<int>(atomicAdd(global_ix_block, 1));
      atomicAddUint64Wrapper(tmp_buffer_loc, sizeof(DeviceBlockShare));
    }
    __syncwarp(); // RC FA
    ix_block = __shfl_sync(WARP_ALL, ix_block, 0);

    DeviceBlockShare *next_block_share = &block_shares[ix_block];
    block_share->next_block_share = next_block_share;
    block_share = next_block_share;
  }
}

// Force 8-byte alignment on tmp_buffer
void classify_frames_api(
  const uint8_t *const *input_frames,
  size_t frame_count,
  DecompressionScratchHandle &buffers,
  const size_t *input_decomp_sizes,
  cudaStream_t stream
)
{
  // Launch 2 warps per block to achieve max occupancy
  constexpr int num_warps_per_block = 2;
  constexpr int block_dim = num_warps_per_block * WARP_SIZE;

  assert(frame_count < std::numeric_limits<int>::max());
  // We'll do one frame per warp:
  // This guarantees, that the next/previous block neighborhoods
  // are not shuffled between frames.
  const int grid_dim = (static_cast<int>(frame_count) + num_warps_per_block - 1) / num_warps_per_block;

  classify_frames<num_warps_per_block><<<grid_dim, block_dim, 0, stream>>>(
    input_frames,
    buffers.block_shares,
    frame_count,
    buffers.tmp_buffer_loc,
    buffers.tmp_buffer_size,
    buffers.global_block_count,
    buffers.global_huff_block_count,
    buffers.global_ans_block_count,
    buffers.zstd_frames,
    input_decomp_sizes
  );
  CUDA_CHECK(cudaGetLastError());
}

// Force 8-byte alignment on tmp_buffer
void init_tables_api(size_t frame_count, DecompressionScratchHandle &buffers, cudaStream_t &stream)
{
  constexpr int fse_block_dim = WARP_SIZE * INIT_FSE_WARPS_PER_CTA;
  const int fse_grid_dim = max(1, static_cast<int>(frame_count / INIT_FSE_WARPS_PER_CTA));

#ifdef PRINT_INIT_FSE_TIMING_INFO
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
#endif

  init_fse_tables<<<fse_grid_dim, fse_block_dim, 0, stream>>>(
    buffers.offset_tmp_buffer,
    buffers.tmp_buffer_loc,
    buffers.tmp_buffer_size,
    buffers.global_ix_block_ans_init,
    buffers.global_block_count,
    buffers.block_shares
  );
  CUDA_CHECK(cudaGetLastError());

#ifdef PRINT_INIT_FSE_TIMING_INFO
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, start, stop);
  printf("init fse tables elapsed time %f ms\n", elapsed_time);
#endif

  // Launch 2 warps per block to achieve max occupancy
  constexpr int huff_block_dim = WARP_SIZE;
  const int huff_grid_dim = static_cast<int>(frame_count);

  init_huff_tables<<<huff_grid_dim, huff_block_dim, 0, stream>>>(
    buffers.offset_tmp_buffer,
    buffers.tmp_buffer_loc,
    buffers.tmp_buffer_size,
    buffers.global_ix_block_huff_init,
    buffers.global_block_count,
    buffers.block_shares
  );
  CUDA_CHECK(cudaGetLastError());
}

struct ZstdDecompressRuntimeInitilizer
{
  int num_sms;
  int block_size;
  int num_blocks_per_sm;
  int total_dynamic_shmem_size;

  ZstdDecompressRuntimeInitilizer(cudaStream_t &stream)
  {
    DeviceGuard device_guard(stream);

    int device_id;
    CUDA_CHECK(cudaGetDevice(&device_id));

    num_sms = CudaUtils::get_sm_count(stream);

    // ANS tables are stored in dynamic shmem
    // ARCH_ID is used by the host to retrieve architecture-specific constants
    int arch_id = get_arch_id(device_id);

    total_dynamic_shmem_size = CONFIG_MAX_TOTAL_FSE_TABLES[arch_id] * max_fse_tables_words * sizeof(uint32_t);

#ifdef PRINT_SHMEM_USAGE
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, decompression_kernel);

    const int static_shmem = attr.sharedSizeBytes;

    printf("Shared Memory Usage\n");
    printf("Dynamic: %d\n", total_dynamic_shmem_size);
    printf("Static : %d\n", static_shmem);
    printf("Total  : %d\n", static_shmem + total_dynamic_shmem_size);
#endif

    CUDA_CHECK(
      cudaFuncSetAttribute(decompression_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, total_dynamic_shmem_size)
    );
    block_size = CONFIG_NUM_WARPS_PER_CTA[arch_id] * WARP_SIZE;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &num_blocks_per_sm,
      decompression_kernel,
      block_size,
      total_dynamic_shmem_size
    ));
  }
};

void decompress_frames_api(
  uint8_t *const *output_frames,
  size_t *frame_decomp_sizes,
  const size_t *input_decomp_sizes,
  size_t frame_count,
  DecompressionScratchHandle &buffers,
  cudaStream_t &stream
)
{
  ZstdDecompressRuntimeInitilizer runtime_init(stream);

  int grid_dim = std::min(runtime_init.num_sms * runtime_init.num_blocks_per_sm, static_cast<int>(frame_count));

  decompression_kernel<<<grid_dim, runtime_init.block_size, runtime_init.total_dynamic_shmem_size, stream>>>(
    output_frames,
    frame_decomp_sizes,
    input_decomp_sizes,
    frame_count,
    buffers.zstd_frames,
    buffers.block_shares,
    buffers.global_ix_frame_lz,
    buffers.global_ix_block_lz,
    buffers.global_ix_block_ans,
    buffers.global_ix_block_huff,
    buffers.global_block_count,
    buffers.global_huff_block_count,
    buffers.global_ans_block_count
  );
  CUDA_CHECK(cudaGetLastError());
}

void get_frame_sizes_api(const uint8_t *const *input_frames, size_t *frame_sizes, int frame_count, cudaStream_t &stream)
{
  DeviceGuard device_guard(stream);

  const int block_size = 32;

  int num_blocks_per_sm;
  CUDA_CHECK(
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, get_frame_sizes, block_size, 0 /*shmem_size*/)
  );

  const int sm_count = CudaUtils::get_sm_count(stream);

  int grid_dim = min(sm_count * num_blocks_per_sm, frame_count);

  get_frame_sizes<<<grid_dim, block_size, 0, stream>>>(input_frames, frame_sizes, frame_count);
  CUDA_CHECK(cudaGetLastError());
}

size_t get_reqd_tmp_buffer_size(int num_frames, size_t max_uncompressed_total_size)
{
  // literal bytes and sequence bytes together fit into the 3x the size of the decompressed file
  // if we can assume 4 byte minimum match length. Device block shares also fit here.
  // Also need the maximum size of the tables to be allocated per-block

  // The below is an assumption gathered from analysis of ZSTD frames --
  // TODO: we should replace this with a spin buffer / synchronous scratch size query API.
  // But that won't be a quick fix.
  const size_t num_zstd_blocks_estimate = max(static_cast<size_t>(2 * num_frames), max_uncompressed_total_size / 65536);
  size_t scratch_alloc = num_zstd_blocks_estimate *
                           (sizeof(DeviceBlockShare) + max_fse_tables_words * sizeof(unsigned) + sizeof(HuffmanTable)) +
                         static_cast<size_t>(3.0 * static_cast<double>(max_uncompressed_total_size));
  return scratch_alloc;
}

} // end namespace zstd
