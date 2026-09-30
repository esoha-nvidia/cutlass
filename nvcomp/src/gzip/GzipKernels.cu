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

#include <cub/cub.cuh>

#include <cuda_runtime_api.h>

#include <cassert>
#include <cstddef>
#include <istream>
#include <limits>
#include <ostream>
#include <string>
#include <system_error>
#include <thread>

#include "common.h"
#include "CorrectnessChecks.cuh"
#include "crc/cuCRC32.h"
#include "cub/block/block_reduce.cuh"
#include "cub/block/block_scan.cuh"
#include "CudaUtils.h"
#include "exception.hpp"
#include "gdeflate/block_utils.cuh"
#include "gdeflate/deflate.h"
#include "gdeflate/gdeflate.h"
#include "GzipConstants.cuh"
#include "GzipDefrag.cuh"
#include "GzipKernels.cuh"
#include "GzipStreamingUtils.h"
#include "nvcomp/gzip.h"
#include "nvcomp/shared_types.h"
#include "nvcomp/utils.hpp"
#include "Reduction.cuh"

namespace nvcomp
{

CompressionScratchHandle::CompressionScratchHandle(
  uint8_t *tmp_storage,
  size_t batch_size,
  size_t max_total_deflate_blocks,
  size_t slot_stride
)
{
  uint8_t *p = tmp_storage;

  deflate_block_src_ptrs = reinterpret_cast<const uint8_t **>(p);
  p += roundUpTo(sizeof(const uint8_t *) * max_total_deflate_blocks, MEMBER_ALIGNMENT);

  deflate_block_src_sizes = reinterpret_cast<size_t *>(p);
  p += roundUpTo(sizeof(size_t) * max_total_deflate_blocks, MEMBER_ALIGNMENT);

  deflate_block_dst_ptrs = reinterpret_cast<uint8_t **>(p);
  p += roundUpTo(sizeof(uint8_t *) * max_total_deflate_blocks, MEMBER_ALIGNMENT);

  deflate_block_dst_sizes = reinterpret_cast<size_t *>(p);
  p += roundUpTo(sizeof(size_t) * max_total_deflate_blocks, MEMBER_ALIGNMENT);

  num_deflates_per_chunk = reinterpret_cast<int *>(p);
  p += roundUpTo(sizeof(int) * batch_size, MEMBER_ALIGNMENT);

  chunk_start_offsets = reinterpret_cast<int *>(p);
  p += roundUpTo(sizeof(int) * batch_size, MEMBER_ALIGNMENT);

  total_deflate_blocks = reinterpret_cast<int *>(p);
  p += roundUpTo(sizeof(int), MEMBER_ALIGNMENT);

  deflate_block_pad_bits = p;
  p += roundUpTo(sizeof(uint8_t) * max_total_deflate_blocks, MEMBER_ALIGNMENT);

  crc32_per_chunk = reinterpret_cast<uint32_t *>(p);
  p += roundUpTo(sizeof(uint32_t) * batch_size, MEMBER_ALIGNMENT);

  deflate_block_statuses = reinterpret_cast<nvcompStatus_t *>(p);
  p += roundUpTo(sizeof(nvcompStatus_t) * max_total_deflate_blocks, MEMBER_ALIGNMENT);

  deflate_byte_offsets = reinterpret_cast<uint64_t *>(p);
  p += roundUpTo(sizeof(uint64_t) * max_total_deflate_blocks, MEMBER_ALIGNMENT);

  chunk_tile_counts = reinterpret_cast<uint64_t *>(p);
  p += roundUpTo(sizeof(uint64_t) * batch_size, MEMBER_ALIGNMENT);

  chunk_tile_starts = reinterpret_cast<uint64_t *>(p);
  p += roundUpTo(sizeof(uint64_t) * batch_size, MEMBER_ALIGNMENT);

  read_done_bytes = roundUpDiv(max_total_deflate_blocks * slot_stride, static_cast<size_t>(DEFRAG_TILE_SIZE)) +
                    batch_size;
  read_done = reinterpret_cast<unsigned char *>(p);
  p += roundUpTo(sizeof(unsigned char) * read_done_bytes, MEMBER_ALIGNMENT);

  size_t scan_bytes = 0;
  cub::DeviceScan::ExclusiveSum(
    nullptr,
    scan_bytes,
    static_cast<uint64_t *>(nullptr),
    static_cast<uint64_t *>(nullptr),
    static_cast<int>(batch_size)
  );
  scan_tmp_bytes = roundUpTo(scan_bytes, MEMBER_ALIGNMENT);
  scan_tmp_ptr = p;
  p += scan_tmp_bytes;

  bytes_used = static_cast<size_t>(p - tmp_storage);
}

size_t
CompressionScratchHandle::required_tmp_bytes(size_t batch_size, size_t max_total_deflate_blocks, size_t slot_stride)
{
  return CompressionScratchHandle(nullptr, batch_size, max_total_deflate_blocks, slot_stride).bytes_used;
}

void compute_CRC32_per_chunk(
  const uint8_t *const *input_buffers,
  const size_t *input_decomp_sizes,
  uint32_t *crc32_per_chunk,
  int batch_size,
  size_t max_chunk_size,
  cudaStream_t stream
)

{
  if (max_chunk_size == 0)
  {
    // Every chunk is empty; CRC32 of empty input is 0. Skip cuCRC32 entirely, since its
    // heuristic configuration (cuCRC32ConfHeur) rejects a zero maximum chunk size.
    CUDA_CHECK(cudaMemsetAsync(crc32_per_chunk, 0, static_cast<size_t>(batch_size) * sizeof(uint32_t), stream));
    return;
  }

  // Standard gzip CRC32 spec (IEEE 802.3 / PKZIP): poly=0x04C11DB7, init=~0,
  crcSpec_t cucrc_spec = {0x04C11DB7u, 0xFFFFFFFFu, 1, 1, 0xFFFFFFFFu};
  crcCtx_t curc_ctx = {crc32_per_chunk, cucrc_spec};

  unsigned int cucrc_conf = 0;
  const int conf_status = cuCRC32ConfHeur(
    &curc_ctx,
    static_cast<unsigned int>(batch_size),
    reinterpret_cast<const unsigned long long *>(input_decomp_sizes),
    &cucrc_conf,
    static_cast<unsigned long long>(max_chunk_size),
    stream
  );
  if (conf_status != CUCRC32_SUCCESS)
  {
    throw NVCompException(nvcompErrorInternal, "cuCRC32ConfHeur failed");
  }

  const int beg_status = cuCRC32Beg(&curc_ctx, static_cast<unsigned int>(batch_size), stream);
  if (beg_status != CUCRC32_SUCCESS)
  {
    throw NVCompException(nvcompErrorInternal, "cuCRC32Beg failed");
  }

  const int add_status = cuCRC32Add(
    &curc_ctx,
    cucrc_conf,
    static_cast<unsigned int>(batch_size),
    reinterpret_cast<const unsigned long long *>(input_decomp_sizes),
    reinterpret_cast<const unsigned char *const *>(input_buffers),
    stream
  );
  if (add_status != CUCRC32_SUCCESS)
  {
    throw NVCompException(nvcompErrorInternal, "cuCRC32Add failed");
  }

  const int end_status = cuCRC32End(&curc_ctx, static_cast<unsigned int>(batch_size), stream);
  if (end_status != CUCRC32_SUCCESS)
  {
    throw NVCompException(nvcompErrorInternal, "cuCRC32End failed");
  }
}

// Small kernel required to fill device data before cub scan
__global__ void
count_deflates_per_chunk(const size_t *input_decomp_sizes, const size_t batch_size, int *num_deflates_per_chunk)
{
  const size_t ix_chunk = threadIdx.x + blockIdx.x * blockDim.x;
  if (ix_chunk >= batch_size)
  {
    return;
  }
  num_deflates_per_chunk[ix_chunk] = static_cast<int>(roundUpDiv(input_decomp_sizes[ix_chunk], DEFLATE_BLOCK_SIZE));
}

// After cub scan prepare pointers to be passed into gdeflate::CompressAsync
template <gzipOperatingMode_t GZIP_MODE>
__global__ void fill_deflate_descriptors(
  const uint8_t *const *input_buffers,
  const size_t *input_decomp_sizes,
  const size_t batch_size,
  uint8_t *const *output_buffers,
  size_t max_comp_deflate_size,
  const int *num_deflates_per_chunk,
  const int *chunk_start_offsets,
  const uint8_t **deflate_block_src_ptrs,
  size_t *deflate_block_src_sizes,
  uint8_t **deflate_block_dst_ptrs,
  int *total_deflate_blocks,
  const size_t max_total_deflate_blocks
)
{
  const size_t ix_chunk = threadIdx.x + blockIdx.x * blockDim.x;
  if (ix_chunk >= batch_size)
  {
    return;
  }

  const int num_deflates = num_deflates_per_chunk[ix_chunk];
  const int chunk_start = chunk_start_offsets[ix_chunk];
  const uint8_t *source = input_buffers[ix_chunk];
  const size_t size = input_decomp_sizes[ix_chunk];
  uint8_t *output = output_buffers[ix_chunk];

  size_t accumulative_offset = 0;
  for (int i = 0; i < num_deflates; ++i)
  {
    const int global_idx = chunk_start + i;
    const size_t in_offset = static_cast<size_t>(i) * DEFLATE_BLOCK_SIZE;
    deflate_block_src_ptrs[global_idx] = source + in_offset;
    deflate_block_src_sizes[global_idx] = min(static_cast<size_t>(DEFLATE_BLOCK_SIZE), size - in_offset);

    // gdeflate's bitwriter uses 8-byte (uint64) atomicOr, so every slot base must be 8-aligned.
    const size_t out_offset = gzip_deflate_slot_base<GZIP_MODE>() + accumulative_offset;
    deflate_block_dst_ptrs[global_idx] = output + out_offset;
    accumulative_offset += gzip_deflate_slot_stride(max_comp_deflate_size);
  }

  // Data is allocated for the max_deflates_per_chunk*batch_size so we need to zero-fill the unused tail.
  const int last_chunk_idx = static_cast<int>(batch_size) - 1;
  const int grand_total = chunk_start_offsets[last_chunk_idx] + num_deflates_per_chunk[last_chunk_idx];
  for (size_t i = static_cast<size_t>(grand_total) + ix_chunk; i < max_total_deflate_blocks; i += batch_size)
  {
    deflate_block_src_ptrs[i] = nullptr;
    deflate_block_src_sizes[i] = 0;
    deflate_block_dst_ptrs[i] = nullptr;
  }

  // Last chunk's thread publishes the grand total.
  if (ix_chunk == batch_size - 1)
  {
    *total_deflate_blocks = grand_total;
  }
}

// Reduce per-sub-block gdeflate statuses into one status per user chunk.
template <int NUM_THREADS>
__global__ void aggregate_deflate_statuses(
  const nvcompStatus_t *deflate_block_statuses,
  const int *num_deflates_per_chunk,
  const int *chunk_start_offsets,
  size_t batch_size,
  nvcompStatus_t *chunk_statuses
)
{
  if (chunk_statuses == nullptr)
  {
    return;
  }

  const int ix_chunk = blockIdx.x;
  const int start = chunk_start_offsets[ix_chunk];
  const int num_deflates = num_deflates_per_chunk[ix_chunk];

  int local = nvcompSuccess;
  for (int i = threadIdx.x; i < num_deflates; i += blockDim.x)
  {
    local = max(local, static_cast<int>(deflate_block_statuses[start + i]));
  }

  using BlockReduce = cub::BlockReduce<int, NUM_THREADS>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  const int agg = BlockReduce(temp_storage).Reduce(local, cub_maximum{});

  if (threadIdx.x == 0)
  {
    chunk_statuses[ix_chunk] = static_cast<nvcompStatus_t>(agg);
  }
}

// Kernel that does 4 things required as a prerequisite for defrag:
// - Each DEFLATE block output from gdeflate compressor has BFIN bit = 1 - this kernel will flip it for non-final blocks.
// - Every DEFLATE block except the last is be padded with an extra BT0 empty block
// - Compute deflate_byte_offsets: where each block lands in the final packed gzip stream.
// - Count how many defrag tiles each chunk needs (the input to the cub scan that prefix-sums them).
template <int NUM_THREADS, gzipOperatingMode_t GZIP_MODE>
__global__ void preprocess_deflate_output(
  size_t *deflate_block_dst_sizes,
  uint8_t *const *deflate_block_dst_ptrs,
  const uint8_t *deflate_block_pad_bits,
  const int *num_deflates_per_chunk,
  const int *chunk_start_offsets,
  uint64_t *deflate_byte_offsets,
  uint64_t *chunk_tile_counts,
  size_t *compressed_sizes,
  unsigned char *read_done,
  size_t read_done_bytes
)
{
  // Clear the defrag atomics flags.
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < read_done_bytes; i += gridDim.x * blockDim.x)
  {
    read_done[i] = 0;
  }

  const int ix_chunk = blockIdx.x;
  const int tid = threadIdx.x;
  const int num_deflates = num_deflates_per_chunk[ix_chunk];
  const int chunk_start = chunk_start_offsets[ix_chunk];

  using BlockScan = cub::BlockScan<uint64_t, NUM_THREADS>;
  __shared__ typename BlockScan::TempStorage temp_storage;
  __shared__ uint64_t base_byte_offset;

  if (tid == 0)
  {
    // ONESHOT reserves the gzip header at the front of the packed output; STREAMING packs from 0.
    base_byte_offset = gzip_deflate_packed_base<GZIP_MODE>();
  }
  __syncthreads();

  for (int tile = 0; tile < num_deflates; tile += blockDim.x)
  {
    const int idx = tile + tid;
    uint64_t consumed = 0;
    if (idx < num_deflates)
    {
      const size_t bytes = deflate_block_dst_sizes[chunk_start + idx];
      const uint8_t padding = deflate_block_pad_bits[chunk_start + idx];
      const uint64_t Lbits = static_cast<uint64_t>(8 * bytes - padding);

      // Deflate compressor by default marks all DEFLATE blocks BFINAL to 1.
      // Here we need to flip that bit, unless this is the last block of the chunk.
      // STREAMING treats every block as non-last: windows are concatenated and the whole stream is
      // terminated separately by a single empty BFINAL=1 BTYPE=0 block, so no window block may be final.
      const bool is_last = (GZIP_MODE == gzipOperatingMode::ONESHOT) && (idx == num_deflates - 1);
      uint8_t *deflate_compressed_block = deflate_block_dst_ptrs[chunk_start + idx];
      if (!is_last)
      {
        deflate_compressed_block[0] ^= 0x01;
        // Non-last deflate blocks: explicitly emit the empty BTYPE=00 stored block that
        // byte-aligns the next one.
        // Layout from bit L (= Lbits) onward:
        //   [BFINAL=0][BTYPE=00] (3 bits) | skip-to-byte-boundary | LEN=0x0000 | NLEN=0xFFFF
        const uint32_t rem = Lbits & 7;
        if (rem != 0)
        {
          // DEFLATE is LSB-first - we need to clear the remaining bits of the last byte.
          deflate_compressed_block[bytes - 1] &= ((1u << rem) - 1);
        }

        // padding added after 3 bits of BFIN and BTYPE 0, roundUpDiv gives us byte offset of LEN
        const size_t consumed_data = roundUpDiv(Lbits + 3, 8);

        if (consumed_data > bytes)
        {
          // If rem is 0, 6, or 7 the BTYPE0 header spills past the last data byte into byte[bytes].
          // We then need to pad the whole byte
          deflate_compressed_block[bytes] = 0;
        }

        uint8_t *trailer = deflate_compressed_block + consumed_data;
        trailer[0] = 0x00; // LEN low
        trailer[1] = 0x00; // LEN high (LEN  = 0x0000)
        trailer[2] = 0xff; // NLEN low
        trailer[3] = 0xff; // NLEN high (NLEN = 0xFFFF = ~LEN)
      }

      if (is_last)
      {
        consumed = roundUpDiv(Lbits, 8); // data only, byte-aligned footer after
      }
      else
      {
        consumed = roundUpDiv(Lbits + 3, 8) + 4u; // + stored header(3b) padded to a byte + LEN/NLEN(4B)
      }

      // After this kernel, deflate_block_dst_sizes holds the block's total consumed output bytes
      // (compressed data + appended empty BT0 block)
      deflate_block_dst_sizes[chunk_start + idx] = consumed;
    }

    // Calculate byte offsets
    uint64_t excl = 0;
    uint64_t tile_total = 0;
    BlockScan(temp_storage).ExclusiveSum(consumed, excl, tile_total);

    if (idx < num_deflates)
    {
      deflate_byte_offsets[chunk_start + idx] = base_byte_offset + excl;
    }
    __syncthreads(); // all reads of base_byte_offset done before update

    if (tid == 0)
    {
      base_byte_offset += tile_total;
    }
    __syncthreads();
  }

  // Save this chunk's defrag tile count for the later cub scan, and the packed deflate-payload size.
  // ONESHOT overwrites compressed_sizes later in write_gzip_frame (header + payload + footer);
  // STREAMING keeps this as the final per-window payload size (no per-window gzip frame).
  if (tid == 0)
  {
    const uint64_t packed_payload_bytes = base_byte_offset - gzip_deflate_packed_base<GZIP_MODE>();
    chunk_tile_counts[ix_chunk] = roundUpDiv(packed_payload_bytes, DEFRAG_TILE_SIZE);
    compressed_sizes[ix_chunk] = packed_payload_bytes;
  }
}

__global__ void write_gzip_frame(
  uint8_t *const *output_buffers,
  const size_t *input_decomp_sizes,
  const size_t *deflate_block_dst_sizes,
  const uint8_t *deflate_block_pad_bits,
  const int *num_deflates_per_chunk,
  const int *chunk_start_offsets,
  const uint64_t *deflate_byte_offsets,
  const uint32_t *crc32_per_chunk,
  size_t *compressed_sizes
)
{
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  const int num_deflates = num_deflates_per_chunk[bid];
  const int chunk_start = chunk_start_offsets[bid];
  uint8_t *out_buf = output_buffers[bid];

  // Minimal gzip header: magic, CM=deflate, FLG=0, MTIME=0, XFL=0, OS=0xff.
  // MTIME=0 is explicitly "no timestamp available" so this is fully spec-compliant.
  // If at any point we need to add more flags to gzip header we will do that here.
  // For now the minimal header is fully sufficient.
  if (tid < static_cast<int>(GZIP_HEADER_BYTES))
  {
    static const uint8_t HDR[GZIP_HEADER_BYTES] = {0x1f, 0x8b, 0x08, 0x00, 0, 0, 0, 0, 0x00, 0xff};
    out_buf[tid] = HDR[tid];
  }

  if (tid == 0)
  {
    uint64_t footer_byte;
    if (num_deflates == 0)
    {
      // Empty input chunk: emit a single empty fixed-Huffman block (0x03 0x00) right after the header
      // so the gzip stream is still valid. Required for spec-compliance (gzip of an empty file).
      out_buf[GZIP_HEADER_BYTES] = 0x03;
      out_buf[GZIP_HEADER_BYTES + 1] = 0x00;
      footer_byte = GZIP_HEADER_BYTES + 2; // header + 2-byte empty block
    }
    else
    {
      // Last sub-block keeps BFINAL set and has no trailer; the footer follows
      // its byte-padded data. byte_offset is already byte-aligned.
      const int last = chunk_start + num_deflates - 1;
      const uint32_t last_bits =
        static_cast<uint32_t>(8 * deflate_block_dst_sizes[last] - deflate_block_pad_bits[last]);
      footer_byte = deflate_byte_offsets[last] + roundUpDiv(last_bits, 8);
    }

    uint8_t *footer = out_buf + footer_byte;
    const uint32_t crc = crc32_per_chunk[bid];
    footer[0] = static_cast<uint8_t>(crc & 0xff);
    footer[1] = static_cast<uint8_t>((crc >> 8) & 0xff);
    footer[2] = static_cast<uint8_t>((crc >> 16) & 0xff);
    footer[3] = static_cast<uint8_t>((crc >> 24) & 0xff);
    const uint32_t isize = static_cast<uint32_t>(input_decomp_sizes[bid]);
    footer[4] = static_cast<uint8_t>(isize & 0xff);
    footer[5] = static_cast<uint8_t>((isize >> 8) & 0xff);
    footer[6] = static_cast<uint8_t>((isize >> 16) & 0xff);
    footer[7] = static_cast<uint8_t>((isize >> 24) & 0xff);

    compressed_sizes[bid] = footer_byte + 8;
  }
}

template <gzipOperatingMode_t GZIP_MODE>
dim3 get_defrag_dim(int threads_per_block, int smem, int num_sm)
{
  int blocks_per_sm = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &blocks_per_sm,
    reinterpret_cast<void *>(defrag_deflate_blocks<DEFRAG_TILE_SIZE, GZIP_MODE>),
    threads_per_block,
    smem
  ));
  return dim3(num_sm * std::max(1, blocks_per_sm));
}

// Mode-templated deflate core shared by the batch (ONESHOT) and streaming paths. Subdivides each chunk
// into DEFLATE_BLOCK_SIZE sub-blocks, gdeflate-compresses them, then defragments into one contiguous
// deflate payload starting at gzip_deflate_packed_base<GZIP_MODE>() in each output buffer. Sets
// compressed_sizes to the packed payload size. ONESHOT reserves the gzip header in front; STREAMING
// packs from offset 0. CRC32 and gzip framing are the caller's responsibility (mode-specific).
template <gzipOperatingMode_t GZIP_MODE>
void gzipDeflateCompress(
  const uint8_t *const *input_buffers,
  const size_t *input_decomp_sizes,
  const size_t max_chunk_size,
  const size_t batch_size,
  uint8_t *tmp_buffer,
  size_t tmp_buffer_size,
  uint8_t *const *compressed_buffers,
  size_t *compressed_sizes,
  nvcompStatus_t *device_statuses,
  CompressionScratchHandle &scratch,
  gdeflate::gdeflate_compression_algo deflate_algorithm,
  cudaStream_t stream
)
{
  // Upper bound on total deflate blocks across all chunks. The actual count will live at `scratch.total_deflate_blocks`.
  const size_t max_total_deflate_blocks = gzip_max_total_deflate_blocks(max_chunk_size, batch_size);

  // Worst-case compressed size / output-slot stride for one DEFLATE_BLOCK_SIZE sub-block.
  size_t max_deflate_comp_chunk_size = 0;
  nvcomp_deflate::DeflateCompressGetMaxOutputChunkSize(DEFLATE_BLOCK_SIZE, &max_deflate_comp_chunk_size);
  size_t slot_stride = gzip_deflate_slot_stride(max_deflate_comp_chunk_size);

  // Stage 1: per-chunk deflate block count
  // We need to fill the information about how many deflate blocks there are in each chunk.
  constexpr int SETUP_BLOCK = 256;
  const size_t setup_grid = roundUpDiv(batch_size, static_cast<size_t>(SETUP_BLOCK));
  count_deflates_per_chunk<<<static_cast<unsigned int>(setup_grid), SETUP_BLOCK, 0, stream>>>(
    input_decomp_sizes,
    batch_size,
    scratch.num_deflates_per_chunk
  );
  CUDA_CHECK(cudaGetLastError());

  // Stage 2: chunk_start_offsets scan, needed to later easily determine which deflate block were part of which chunk.
  // Uses the scan scratch carved out by the CompressionScratchHandle (reused again in stage 6).
  // cub takes the size by mutable reference, so pass a local copy of the member.
  size_t scan_tmp_bytes = scratch.scan_tmp_bytes;
  CUDA_CHECK(
    cub::DeviceScan::ExclusiveSum(
      scratch.scan_tmp_ptr,
      scan_tmp_bytes,
      scratch.num_deflates_per_chunk,
      scratch.chunk_start_offsets,
      static_cast<int>(batch_size),
      stream
    )
  );

  // Stage 3: Fill descriptors so all deflate blocks can be passed to gdeflate compressor separately.
  fill_deflate_descriptors<GZIP_MODE><<<static_cast<unsigned int>(setup_grid), SETUP_BLOCK, 0, stream>>>(
    input_buffers,
    input_decomp_sizes,
    batch_size,
    compressed_buffers,
    max_deflate_comp_chunk_size,
    scratch.num_deflates_per_chunk,
    scratch.chunk_start_offsets,
    scratch.deflate_block_src_ptrs,
    scratch.deflate_block_src_sizes,
    scratch.deflate_block_dst_ptrs,
    scratch.total_deflate_blocks,
    max_total_deflate_blocks
  );
  CUDA_CHECK(cudaGetLastError());

  // The scan scratch is part of the scratch handle carve-out, so bytes_used already accounts for it.
  uint8_t *gdeflate_temp_ptr = tmp_buffer + scratch.bytes_used;
  const size_t gdeflate_temp_bytes = tmp_buffer_size - scratch.bytes_used;

  // On empty input we don't call gdeflate codepaths, we will fill the minimal gzip in further kernels.
  if (max_total_deflate_blocks > 0)
  {
    // Stage 4: Compress all sub-blocks in a single gdeflate call.
    gdeflate::compressAsync(
      reinterpret_cast<const void *const *>(scratch.deflate_block_src_ptrs),
      scratch.deflate_block_src_sizes,
      DEFLATE_BLOCK_SIZE,
      max_total_deflate_blocks,
      gdeflate_temp_ptr,
      gdeflate_temp_bytes,
      reinterpret_cast<void *const *>(scratch.deflate_block_dst_ptrs),
      scratch.deflate_block_dst_sizes,
      deflate_algorithm,
      scratch.deflate_block_statuses,
      stream,
      true /* non-swizzled */,
      scratch.deflate_block_pad_bits,
      scratch.total_deflate_blocks
    );

    // Aggregate per-sub-block statuses back to the user's per-chunk device_statuses.
    if (device_statuses != nullptr)
    {
      aggregate_deflate_statuses<SETUP_BLOCK><<<static_cast<unsigned int>(batch_size), SETUP_BLOCK, 0, stream>>>(
        scratch.deflate_block_statuses,
        scratch.num_deflates_per_chunk,
        scratch.chunk_start_offsets,
        batch_size,
        device_statuses
      );
      CUDA_CHECK(cudaGetLastError());
    }
  }
  else
  {
    // Mark compression successful, no chunk was compressed on GPU.
    try_clear_device_statuses(batch_size, device_statuses, stream);
  }

  // Stage 5: precompute per-deflate-block destination byte offsets, fill BT0 padding blocks, record each
  // chunk's defrag tile count, and publish the packed deflate-payload size into compressed_sizes.
  constexpr int PREPROCESS_NUM_THREADS = 256;
  preprocess_deflate_output<PREPROCESS_NUM_THREADS, GZIP_MODE>
    <<<static_cast<unsigned int>(batch_size), PREPROCESS_NUM_THREADS, 0, stream>>>(
      scratch.deflate_block_dst_sizes,
      scratch.deflate_block_dst_ptrs,
      scratch.deflate_block_pad_bits,
      scratch.num_deflates_per_chunk,
      scratch.chunk_start_offsets,
      scratch.deflate_byte_offsets,
      scratch.chunk_tile_counts,
      compressed_sizes,
      scratch.read_done,
      scratch.read_done_bytes
    );
  CUDA_CHECK(cudaGetLastError());

  // Stage 6: defragment scattered deflate blocks into contiguous buffer.
  if (max_total_deflate_blocks > 0)
  {
    // Exclusive prefix sum of the per-chunk tile counts, reusing the stage-2 scan scratch (free now).
    scan_tmp_bytes = scratch.scan_tmp_bytes;
    CUDA_CHECK(
      cub::DeviceScan::ExclusiveSum(
        scratch.scan_tmp_ptr,
        scan_tmp_bytes,
        scratch.chunk_tile_counts,
        scratch.chunk_tile_starts,
        static_cast<int>(batch_size),
        stream
      )
    );

    size_t num_chunks = batch_size;
    void *kernel_args[] = {
      &compressed_buffers,
      &scratch.deflate_block_dst_sizes,
      &scratch.deflate_block_dst_ptrs,
      &scratch.deflate_byte_offsets,
      &scratch.chunk_start_offsets,
      &scratch.num_deflates_per_chunk,
      &scratch.chunk_tile_starts,
      &scratch.chunk_tile_counts,
      &num_chunks,
      &slot_stride,
      &scratch.read_done
    };
    const int defrag_smem = DEFRAG_TILE_SIZE;
    const int num_sm = CudaUtils::get_sm_count(stream);
    dim3 grid = get_defrag_dim<GZIP_MODE>(DEFRAG_BLOCK_THREADS, defrag_smem, num_sm);
    dim3 block(DEFRAG_BLOCK_THREADS);
    CUDA_CHECK(cudaLaunchCooperativeKernel(
      reinterpret_cast<void *>(defrag_deflate_blocks<DEFRAG_TILE_SIZE, GZIP_MODE>),
      grid,
      block,
      kernel_args,
      defrag_smem,
      stream
    ));
    CUDA_CHECK(cudaGetLastError());
  }
}

void gzipBatchCompress(
  const uint8_t *const *input_buffers,
  const size_t *input_decomp_sizes,
  const size_t max_chunk_size,
  const size_t batch_size,
  uint8_t *tmp_buffer,
  size_t tmp_buffer_size,
  uint8_t *const *compressed_buffers,
  size_t *compressed_sizes,
  nvcompStatus_t *device_statuses,
  gdeflate::gdeflate_compression_algo deflate_algorithm,
  cudaStream_t stream
)
{
  if (batch_size == 0)
  {
    return;
  }
  const size_t max_total_deflate_blocks = gzip_max_total_deflate_blocks(max_chunk_size, batch_size);

  size_t max_deflate_comp_chunk_size = 0;
  nvcomp_deflate::DeflateCompressGetMaxOutputChunkSize(DEFLATE_BLOCK_SIZE, &max_deflate_comp_chunk_size);
  const size_t slot_stride = gzip_deflate_slot_stride(max_deflate_comp_chunk_size);

  CompressionScratchHandle scratch(tmp_buffer, batch_size, max_total_deflate_blocks, slot_stride);

  // CRC32 of each chunk's uncompressed contents, consumed by the per-chunk gzip footer below. No data
  // dependency on compression, so it can run first.
  compute_CRC32_per_chunk(
    input_buffers,
    input_decomp_sizes,
    scratch.crc32_per_chunk,
    static_cast<int>(batch_size),
    max_chunk_size,
    stream
  );

  gzipDeflateCompress<gzipOperatingMode::ONESHOT>(
    input_buffers,
    input_decomp_sizes,
    max_chunk_size,
    batch_size,
    tmp_buffer,
    tmp_buffer_size,
    compressed_buffers,
    compressed_sizes,
    device_statuses,
    scratch,
    deflate_algorithm,
    stream
  );

  // Per-chunk gzip framing: header + CRC32/ISIZE footer; overwrites compressed_sizes with the framed size.
  constexpr unsigned int GZIP_FRAME_THREADS = 32;
  write_gzip_frame<<<static_cast<unsigned int>(batch_size), GZIP_FRAME_THREADS, 0, stream>>>(
    compressed_buffers,
    input_decomp_sizes,
    scratch.deflate_block_dst_sizes,
    scratch.deflate_block_pad_bits,
    scratch.num_deflates_per_chunk,
    scratch.chunk_start_offsets,
    scratch.deflate_byte_offsets,
    scratch.crc32_per_chunk,
    compressed_sizes
  );

  CUDA_CHECK(cudaGetLastError());
}

void StreamGzipCompressionWindow::compress(cudaStream_t stream, gdeflate::gdeflate_compression_algo algorithm)
{
  CUDA_CHECK(cudaStreamWaitEvent(stream, h2d_done, 0));

  const size_t max_total_deflate_blocks = gzip_max_total_deflate_blocks(input_buffer_size, 1);

  size_t max_deflate_comp_chunk_size = 0;
  nvcomp_deflate::DeflateCompressGetMaxOutputChunkSize(DEFLATE_BLOCK_SIZE, &max_deflate_comp_chunk_size);
  const size_t slot_stride = gzip_deflate_slot_stride(max_deflate_comp_chunk_size);

  CompressionScratchHandle scratch(d_temp, 1, max_total_deflate_blocks, slot_stride);

  // STREAMING: emit only the contiguous deflate payload for this window (no per-window gzip frame).
  // d_out_size receives the payload byte count; d_status the per-window compression status.
  gzipDeflateCompress<gzipOperatingMode_t::STREAMING>(
    d_in_ptr,
    d_in_size,
    input_buffer_size,
    /*batch_size=*/1,
    d_temp,
    temp_bytes,
    d_out_ptr,
    d_out_size,
    d_status,
    scratch,
    algorithm,
    stream
  );
  CUDA_CHECK(cudaEventRecord(compress_done, stream));
}

namespace
{
// Per-window sizing for one GZIP_STREAMING_WINDOW_SIZE chunk (batch_size == 1): scratch temp_bytes
// and worst-case compressed output max_out. Single source for both the temp-size query and the run.
void compute_streaming_window_sizing(
  nvcompBatchedGzipCompressOpts_t opts,
  size_t &temp_bytes,
  size_t &max_out,
  cudaStream_t stream
)
{
  if (nvcompBatchedGzipCompressGetTempSize(
        1,
        GZIP_STREAMING_WINDOW_SIZE,
        opts,
        &temp_bytes,
        GZIP_STREAMING_WINDOW_SIZE,
        stream
      ) != nvcompSuccess ||
      nvcompBatchedGzipCompressGetMaxOutputChunkSize(GZIP_STREAMING_WINDOW_SIZE, opts, &max_out) != nvcompSuccess)
  {
    throw NVCompException(nvcompErrorInternal, "gzipStreamingCompress: failed to size temp/output buffers");
  }
}

// Total caller-provided device workspace: a running-CRC32 cell + NUM_WINDOWS back-to-back per-window
// slices. Single source of truth shared by gzipStreamingCompressTempSize() and the bounds check.
size_t streaming_total_device_bytes(size_t temp_bytes, size_t max_out)
{
  // One size_t-aligned uint32_t cell for the running CRC32 (kept aligned so the window slices that
  // follow stay aligned), then NUM_WINDOWS back-to-back per-window slices.
  return roundUpTo(sizeof(uint32_t), sizeof(size_t)) +
         static_cast<size_t>(STREAMING_NUM_WINDOWS) * StreamGzipCompressionWindow::device_scratch_bytes_requirement(
                                                        GZIP_STREAMING_WINDOW_SIZE,
                                                        max_out,
                                                        temp_bytes
                                                      );
}
} // namespace

size_t gzipStreamingCompressTempSize(nvcompBatchedGzipCompressOpts_t opts)
{
  size_t temp_bytes = 0;
  size_t max_out = 0;
  compute_streaming_window_sizing(opts, temp_bytes, max_out, nullptr);
  return streaming_total_device_bytes(temp_bytes, max_out);
}

void gzipStreamingCompress(
  std::istream &input,
  std::ostream &output,
  nvcompBatchedGzipCompressOpts_t opts,
  void *device_temp,
  size_t device_temp_bytes,
  cudaStream_t stream
)
{
  const gdeflate::gdeflate_compression_algo algorithm = gdeflate::getCompressionAlgo(opts.algorithm);

  // Single-chunk-per-window sizing (batch_size == 1). Fixed for every window, so size once.
  size_t temp_bytes = 0;
  size_t max_out = 0;
  compute_streaming_window_sizing(opts, temp_bytes, max_out, stream);

  // The caller must provide at least the device workspace gzipStreamingCompressTempSize() reports;
  // the windows are carved from it, so a short buffer would be an out-of-bounds write.
  const size_t required_device_temp = streaming_total_device_bytes(temp_bytes, max_out);

  if (device_temp_bytes < required_device_temp)
  {
    throw NVCompException(nvcompErrorInvalidValue, "gzipStreamingCompress: device temp buffer too small");
  }

  // Write out header before any compression
  write_or_throw(output, MINIMAL_GZIP_HEADER, GZIP_HEADER_BYTES);

  // Carve the running-CRC32 cell off the front of the caller temp; the windows get the rest. This
  // mirrors the layout assumed by streaming_total_device_bytes().
  uint8_t *const device_temp_base = static_cast<uint8_t *>(device_temp);
  unsigned int *const d_crc = reinterpret_cast<unsigned int *>(device_temp_base);
  void *const windows_device_temp = device_temp_base + roundUpTo(sizeof(uint32_t), sizeof(size_t));

  StreamGzipCompressionContext streaming_context(GZIP_STREAMING_WINDOW_SIZE, stream, d_crc);

  StreamGzipPipeline pipeline(streaming_context, input, output, max_out, temp_bytes, algorithm, windows_device_temp);

  std::thread reader([&]() { pipeline.run_reader(); });
  std::thread writer;
  try
  {
    writer = std::thread([&]() { pipeline.run_writer(); });
  }
  catch (const std::system_error &e)
  {
    // The writer failed to start (e.g. thread/resource exhaustion). Close the channels so the running
    // reader unblocks (it would otherwise wait forever for windows the writer never recycles)
    pipeline.fail();
    reader.join();
    throw NVCompException(
      nvcompErrorInternal,
      std::string("gzipStreamingCompress: failed to start writer thread: ") + e.what()
    );
  }
  pipeline.run_compress();

  reader.join();
  writer.join();
  pipeline.rethrow_first_error();

  // Terminal empty BFINAL=1 block, then the CRC32 + ISIZE footer.
  static const uint8_t TERMINAL_BTYPE1_BLOCK[2] = {0x03, 0x00};
  write_or_throw(output, TERMINAL_BTYPE1_BLOCK, 2);

  const uint32_t crc = streaming_context.finalize_crc(streaming_context.compute_stream);
  const uint32_t isize = static_cast<uint32_t>(streaming_context.isize);
  uint8_t footer[GZIP_FOOTER_BYTES] = {};
  assemble_gzip_footer(footer, crc, isize);
  write_or_throw(output, footer, GZIP_FOOTER_BYTES);
}

} // namespace nvcomp
