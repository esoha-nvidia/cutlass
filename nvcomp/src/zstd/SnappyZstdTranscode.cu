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

#include "ans.cuh"
#include "ANSSequenceDecoder.cuh"
#include "constants.cuh"
#include "CudaUtils.h"
#include "device_guard.h"
#include "EntropyTables.cuh"
#include "exception.hpp"
#include "HuffDecoder.cuh"
#include "huffman.cuh"
#include "io.cuh"
#include "lz.cuh"
#include "nvcomp/snappy.h"
#include "snappy_transcode.cuh"
#include "types.cuh"
#include "utils.cuh"
#include "zstd_kernel_utils.cuh"
#include "zstdKernels.cuh"

#include <cooperative_groups.h>

namespace cg = cooperative_groups;
using namespace nvcomp;
using nvcomp::DeviceGuard;

namespace zstd
{

inline __device__ void do_raw_block_copy_snappy(const DeviceBlockShare &block_share, uint8_t *output_ptr, int &ix_output)
{
  // Do RLE for the entire block
  if (block_share.is_rle_block)
  {
    // In the case of rle, we can encode the rle as 64 byte matches
    int iter_block_size = block_share.decomp_block_size;
    write_literal_tag(output_ptr, ix_output, 1 /*literal length*/, 0 /*num literal tag bytes*/);
    // Add a 1 byte literal that is the rle
    output_ptr[ix_output++] = block_share.rle_byte;

    while (iter_block_size > 0)
    {
      int this_tag_len = min(iter_block_size, 64);
      write_match_tag(output_ptr, ix_output, 2 /*offset len*/, this_tag_len, 1 /*offset*/);
      iter_block_size -= 64;
    }
  }
  else
  {
    // For raw literals, we can just write a single literal tag
    const int num_literals = block_share.decomp_block_size;
    const int num_literal_tag_bytes = compute_literal_tag_byte_count(num_literals);
    write_literal_tag(output_ptr, ix_output, num_literals, num_literal_tag_bytes);

    // Then just write the literals
    if (not block_share.is_raw_block)
    {
      int current_decoded_lit_count = 0;
      wait_for_atomic(
        block_share.decode_lit_count,
        block_share.num_literals,
        current_decoded_lit_count,
        ZSTD_SHORT_SLEEP_NS
      );
    }

    const volatile uint8_t *copy_source = block_share.is_raw_block ? block_share.copy_buffer_ptr
                                                                   : block_share.literal_buffer;
    for (int ix_byte = thread_warp_ix(); ix_byte < num_literals; ix_byte += WARP_SIZE)
    {
      output_ptr[ix_byte + ix_output] = copy_source[ix_byte];
    }

    ix_output += num_literals;
  }
}

inline __device__ void write_snappy_preamble(size_t iter_uncomp_frame_size, uint8_t *output_ptr, int &ix_output)
{
  bool cont = true;
  while (cont)
  {
    // This is a form of LSIC write.
    // Here, the msb (0x80) in the byte is set if
    // more bytes are required to encode the frame size.
    // Then 7 bits (Little Endian) are written for every output
    // byte. Finally the last byte has a cleared msb indicating
    // the size is finished.
    uint8_t this_val = (iter_uncomp_frame_size & 0x7fUL);
    iter_uncomp_frame_size >>= 7;
    if (iter_uncomp_frame_size > 0)
    {
      this_val |= 0x80;
    }
    else
    {
      cont = false;
    }
    output_ptr[ix_output++] = this_val;
  }
}

inline __device__ void warp_snappy_transcode(
  ZstdFrame *frames,
  size_t *frame_transcode_sizes,
  int32_t *repeat_offset,
  int *shared_ix_output,
  int *shared_ix_literal,
  int *global_ix_frame,
  int &shared_ix_frame,
  uint8_t *const *output_frames,
  const int frame_count,
  const size_t *device_uncompressed_bytes
)
{
  while (true)
  {
    increment_ix(shared_ix_frame, global_ix_frame);
    if (shared_ix_frame >= frame_count)
    {
      break;
    }

    int ix_frame_output = 0;
    int ix_block_output = 0;

    auto &frame = frames[shared_ix_frame];
    DeviceBlockShare *block_share_ptr = frame.block_share;

    uint8_t *output_ptr = output_frames[shared_ix_frame];
    write_snappy_preamble(device_uncompressed_bytes[shared_ix_frame], output_ptr, ix_frame_output);

    output_ptr += ix_frame_output;

    reinit_repeat_codes(repeat_offset);

    while (block_share_ptr != nullptr)
    {
      auto &block_share = *block_share_ptr;

      if (not block_share.is_fse_block)
      {
        do_raw_block_copy_snappy(block_share, output_ptr, ix_block_output);

        block_share_ptr = block_share.next_block_share;
        output_ptr += ix_block_output;
        ix_frame_output += ix_block_output;
        ix_block_output = 0;
        continue;
      }

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

      const uint8_t *literal_buffer = nullptr;
      uint8_t rle_byte = 0;
      bool is_rle = false;
      if (block_share.is_literal_rle_block)
      {
        is_rle = true;
        rle_byte = block_share.rle_byte;
      }
      else
      {
        literal_buffer = block_share.is_huff_literal_block ? block_share.literal_buffer : block_share.copy_buffer_ptr;
      }
      ix_block_output += transcode_snappy_lz(
        output_ptr,
        block_share,
        literal_buffer,
        rle_byte,
        is_rle,
        repeat_offset,
        shared_ix_output,
        shared_ix_literal
      );

#ifdef PRINT_TIMING
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
      output_ptr += ix_block_output;
      ix_frame_output += ix_block_output;
      ix_block_output = 0;

      block_share_ptr = block_share.next_block_share;
    }

    frame_transcode_sizes[shared_ix_frame] = ix_frame_output;
  }
}

__device__ void do_snappy_transcode(
  int *global_lz_ix_frame,
  int *global_lz_ix_block,
  int *global_ans_ix_block,
  int *global_huff_ix_block,
  const int global_block_count,
  ZstdFrame *frames,
  DeviceBlockShare *block_shares,
  size_t *frame_decomp_sizes,
  const size_t frame_count,
  uint8_t *const *output_frames,
  TmpBufferManager &tmp_buffer_manager,
  const size_t *device_uncompressed_bytes
)
{
  __shared__ MetaHuffmanTable huff_tables[NUM_HUFF_WARPS_PER_CTA];
  __shared__ int completion_count;
  if (ix_warp() == 0)
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

    int this_completion_count = 0; // init for coverity
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
      do_fse = (MAX_FSE_WARPS_PER_CTA > NUM_FSE_WARPS_PER_CTA);
    }
  }
  else
  {
    // This sleep prevents LZ warps from spinning whilst waiting for entropy warps to produce initial results
    __nanosleep(ZSTD_VERY_LONG_SLEEP_NS);
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

  if (ix_warp() == 0)
  {
    __shared__ int lz_shared_ix_frame[NUM_WARPS_PER_CTA];
    __shared__ int ix_output[WARP_SIZE];
    __shared__ int ix_literal[WARP_SIZE + 1];
    __shared__ int repeat_codes[3];

    warp_snappy_transcode(
      frames,
      frame_decomp_sizes,
      repeat_codes,
      ix_output,
      ix_literal,
      global_lz_ix_frame,
      lz_shared_ix_frame[ix_warp()],
      output_frames,
      frame_count,
      device_uncompressed_bytes
    );
  }
}

__launch_bounds__(NUM_WARPS_PER_CTA *WARP_SIZE, 1) __global__ void snappy_transcode_kernel(
  uint8_t *const *output_frames,
  size_t *frame_decomp_sizes,
  const size_t *device_uncompressed_bytes,
  size_t frame_count,
  uint8_t *tmp_buffer,
  uint64_t *tmp_buffer_loc,
  size_t tmp_buffer_size,
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
  TmpBufferManager tmp_buffer_manager{tmp_buffer_loc, tmp_buffer, tmp_buffer_size};

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

  do_snappy_transcode(
    global_lz_ix_frame,
    global_ans_ix_block,
    global_huff_ix_block,
    global_lz_ix_block,
    global_block_count,
    frames,
    block_shares,
    frame_decomp_sizes,
    frame_count,
    output_frames,
    tmp_buffer_manager,
    device_uncompressed_bytes
  );
}

struct ZstdTranscodeDecompressRuntimeInitilizer
{
  int num_sms;
  int block_size;
  int num_blocks_per_sm;
  int total_dynamic_shmem_size;
  ZstdTranscodeDecompressRuntimeInitilizer(cudaStream_t &stream)
  {
    DeviceGuard guard(stream);

    int device_id;
    CUDA_CHECK(cudaGetDevice(&device_id));

    num_sms = CudaUtils::get_sm_count(stream);

    // ANS tables are stored in dynamic shmem
    // ARCH_ID is used by the host to retrieve architecture-specific constants
    int arch_id = get_arch_id(device_id);

    total_dynamic_shmem_size = CONFIG_MAX_TOTAL_FSE_TABLES[arch_id] * max_fse_tables_words * sizeof(uint32_t);

    // The extra 4 bytes here allow us to read 2 aligned 4 byte words without bounds checking
    total_dynamic_shmem_size += sizeof(uint32_t);

    /*CUDA_CHECK(cudaFuncSetAttribute(
        snappy_transcode_kernel, 
        cudaFuncAttributePreferredSharedMemoryCarveout, 
        100 ));*/

    CUDA_CHECK(cudaFuncSetAttribute(
      snappy_transcode_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      total_dynamic_shmem_size
    ));
    block_size = CONFIG_NUM_WARPS_PER_CTA[arch_id] * WARP_SIZE;

    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &num_blocks_per_sm,
      snappy_transcode_kernel,
      block_size,
      total_dynamic_shmem_size
    ));
  }
};

void snappy_transcode_api(
  uint8_t *const *output_frames,
  size_t *frame_decomp_sizes,
  const size_t *device_uncompressed_bytes,
  size_t frame_count,
  DecompressionScratchHandle &buffers,
  cudaStream_t &stream
)
{
  ZstdTranscodeDecompressRuntimeInitilizer runtime_init(stream);

  int grid_dim = std::min(runtime_init.num_sms * runtime_init.num_blocks_per_sm, static_cast<int>(frame_count));

  snappy_transcode_kernel<<<grid_dim, runtime_init.block_size, runtime_init.total_dynamic_shmem_size, stream>>>(
    output_frames,
    frame_decomp_sizes,
    device_uncompressed_bytes,
    frame_count,
    buffers.offset_tmp_buffer,
    buffers.tmp_buffer_loc,
    buffers.tmp_buffer_size,
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

// Uses the transcode path. Generally the same as zstd, but does transcode then snappy decompress
nvcompStatus_t nvcompBatchedZstdDecompressAsyncTranscode(
  const void *const *device_compressed_ptrs,
  const size_t *, // device_compressed_bytes - unused
  const size_t *device_uncompressed_bytes,
  size_t *device_transcoded_bytes,
  size_t *device_actual_uncompressed_bytes,
  size_t batch_size,
  void *device_temp_ptr,
  const size_t temp_bytes,
  void *const *device_uncompressed_ptr,
  void *const *device_transcoded_ptrs,
  nvcompStatus_t *device_statuses,
  cudaStream_t stream,
  cudaEvent_t fin_transcode_event = NULL
)
{
  // The temporary buffer includes the "global ix frame" and "tmp_buffer_loc" variables. These need to be set asynchronously
  // before the start of the kernel.
  zstd::DecompressionScratchHandle buffers(reinterpret_cast<uint8_t *>(device_temp_ptr), batch_size, temp_bytes);
  buffers.init_values(stream);

  // For now, no error checking.
  CUDA_CHECK(
    cudaMemsetAsync(reinterpret_cast<void **>(device_statuses), 0, sizeof(nvcompStatus_t) * batch_size, stream)
  );

  zstd::classify_frames_api(
    reinterpret_cast<const uint8_t *const *>(device_compressed_ptrs),
    batch_size,
    buffers,
    device_uncompressed_bytes,
    stream
  );

  zstd::init_tables_api(batch_size, buffers, stream);

  zstd::snappy_transcode_api(
    reinterpret_cast<uint8_t *const *>(device_transcoded_ptrs),
    device_transcoded_bytes,
    device_uncompressed_bytes,
    batch_size,
    buffers,
    stream
  );

  if (fin_transcode_event)
  {
    CUDA_CHECK(cudaEventRecord(fin_transcode_event, stream));
  }

  // Then should be able to just call the snappy transcode buffer
  nvcompBatchedSnappyDecompressAsync(
    device_transcoded_ptrs,
    device_transcoded_bytes,
    device_uncompressed_bytes,
    device_actual_uncompressed_bytes,
    batch_size,
    device_temp_ptr,
    temp_bytes,
    device_uncompressed_ptr,
    nvcompBatchedSnappyDecompressDefaultOpts,
    device_statuses,
    stream
  );

  return nvcompSuccess;
}

} // end namespace zstd
