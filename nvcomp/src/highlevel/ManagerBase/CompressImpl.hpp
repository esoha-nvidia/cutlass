/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

#pragma once

#include <algorithm>

// Out-of-line member-function definitions for the ManagerBase class template.
// Included at the bottom of ManagerBase.hpp; included here as well so this file
// is self-contained, but it cannot be used without the ManagerBase declaration.

#include "common_utils.hpp"
#include "highlevel/ManagerBase.hpp"

namespace nvcomp
{

#define TEMPLATE                                                                                                       \
  template <                                                                                                           \
    typename FormatSpecHeader,                                                                                         \
    typename DecompressFn_t,                                                                                           \
    typename DecompressScratchFn_t,                                                                                    \
    typename DecompressSizeFn_t,                                                                                       \
    typename CompressFn_t,                                                                                             \
    typename CompressScratchFn_t,                                                                                      \
    typename MaxCompChunkSizeFn_t,                                                                                     \
    typename CompressOpts_t,                                                                                           \
    typename DecompressOpts_t,                                                                                         \
    nvcompFormatType_t format_type>

#define HEAD                                                                                                           \
  ManagerBase<                                                                                                         \
    FormatSpecHeader,                                                                                                  \
    DecompressFn_t,                                                                                                    \
    DecompressScratchFn_t,                                                                                             \
    DecompressSizeFn_t,                                                                                                \
    CompressFn_t,                                                                                                      \
    CompressScratchFn_t,                                                                                               \
    MaxCompChunkSizeFn_t,                                                                                              \
    CompressOpts_t,                                                                                                    \
    DecompressOpts_t,                                                                                                  \
    format_type>

TEMPLATE
void HEAD::init_format_spec()
{
  // Make sure the size of FormatSpecHeader is always less than the CompressOpts_t,
  // otherwise the memcpy wont be correct and the compressed buffer header will be inaccurate
  static_assert(sizeof(FormatSpecHeader) <= sizeof(CompressOpts_t));

  // Zero the whole header first so the reserved bytes stay clear and are never
  // accidentally filled from CompressOpts beyond the matching FormatSpec prefix.
  std::memset(&format_spec, 0, sizeof(format_spec));

  // Copy only the active FormatSpec prefix; reserved bytes remain zero from memset.
  // Member order of that prefix must match CompressOpts_t.
  std::memcpy(&format_spec, &(this->format_opts), offsetof(FormatSpecHeader, reserved));

  assert(reserved_bytes_all_zero(format_spec.reserved));
}

TEMPLATE
void HEAD::compress_chunked_single(
  const uint8_t *uncomp_buffer,
  uint8_t *comp_buffer,
  const CompressionConfig &comp_config,
  size_t *comp_size
)
{
  // Allocate the pinned host scratch (if necessary)
  allocate_host_scratch();

  // Update the scratch buffer as needed to work for this particular compression API
  allocate_gpu_scratch(compute_compress_scratch_buffer_size(comp_config.num_chunks, true));

  // prepare header
  CommonHeader *common_header = reinterpret_cast<CommonHeader *>(comp_buffer);
  FormatSpecHeader *comp_format_header = reinterpret_cast<FormatSpecHeader *>(common_header + 1);

  comp_buffer += sizeof(CommonHeader) + sizeof(FormatSpecHeader);

  // prepare API pointers
  const size_t num_chunks = comp_config.num_chunks;

  // Scratch memory layout
  // Note: max_comp_chunk_size has been rounded up to min_alignment.
  // [ Compressed chunks temporary location (num_chunks-1 * max_comp_chunk_size, bytes) ]
  // [ Uncompressed buffer pointers (num_chunks, uint8_t*) ]
  // [ Comp buffer pointers (num_chunks, uint8_t*) ]
  // [ Uncompressed sizes (num_chunks, size_t) ]
  // [ Compression statuses (num_chunks, nvcompStatus_t) ]
  // [ Compression scratch for the entire batch (bytes) ]
  uint8_t *free_scratch_buffer = scratch_buffer;

  // Aligning to the format's strongest required alignment (min_alignment).
  uint8_t *scratch_comp_buffer =
    reinterpret_cast<uint8_t *>(roundUpTo(reinterpret_cast<uintptr_t>(free_scratch_buffer), min_alignment));
  free_scratch_buffer = scratch_comp_buffer + (max_comp_chunk_size * (num_chunks - 1));
  assert(max_comp_chunk_size % min_alignment == 0);

  const uint8_t **scratch_uncomp_buffers = reinterpret_cast<const uint8_t **>(free_scratch_buffer);
  free_scratch_buffer += num_chunks * sizeof(uint8_t *);

  uint8_t **scratch_comp_buffers = reinterpret_cast<uint8_t **>(free_scratch_buffer);
  free_scratch_buffer += num_chunks * sizeof(uint8_t *);

  size_t *scratch_uncomp_sizes = reinterpret_cast<size_t *>(free_scratch_buffer);
  free_scratch_buffer += num_chunks * sizeof(size_t);

  nvcompStatus_t *scratch_statuses = reinterpret_cast<nvcompStatus_t *>(free_scratch_buffer);
  free_scratch_buffer += num_chunks * sizeof(nvcompStatus_t);
  free_scratch_buffer =
    reinterpret_cast<uint8_t *>(roundUpTo(reinterpret_cast<uintptr_t>(free_scratch_buffer), min_alignment));

  // Note: free_scratch_buffer now is aligned to min_alignment,
  //       this is required for the worst-case compression scratch space
  assert(reinterpret_cast<uintptr_t>(free_scratch_buffer) % min_alignment == 0);

  const size_t free_scratch_size = scratch_buffer_size - (uintptr_t(free_scratch_buffer) - uintptr_t(scratch_buffer));
  assert(free_scratch_size >= compute_lowlevel_compress_scratch_size(num_chunks, uncomp_chunk_size));

  size_t *comp_chunk_offsets = roundUpToAlignment<size_t>(comp_buffer);
  size_t *comp_sizes = comp_chunk_offsets + num_chunks;

  comp_buffer = reinterpret_cast<uint8_t *>(comp_sizes + num_chunks);

  uint32_t *comp_chunk_checksum = nullptr;
  uint32_t *decomp_chunk_checksum = nullptr;
  if (comp_config.compute_checksums)
  {
    // Should be aligned to 8-byte boundary
    comp_chunk_checksum = reinterpret_cast<uint32_t *>(roundUpToAlignment<uint64_t>(comp_buffer));
    decomp_chunk_checksum = comp_chunk_checksum + num_chunks;
    comp_buffer = reinterpret_cast<uint8_t *>(decomp_chunk_checksum + num_chunks);
  }

  // The compressed-data region must satisfy the format's strongest required
  // alignment.
  comp_buffer = reinterpret_cast<uint8_t *>(roundUpTo(reinterpret_cast<uintptr_t>(comp_buffer), min_alignment));

  DeviceGuard device_guard(user_stream);

  constexpr int NUM_WARPS_PER_COMPACT_BLOCK = 16;
  int maxBlocks;
  int blockSize = WARP_SIZE * NUM_WARPS_PER_COMPACT_BLOCK;
  int num_SMs;

  // Determine the maximum number of active blocks per SM
  CUDA_CHECK(
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocks, compact_comp_buffers_and_header_output, blockSize, 0)
  );

  // Get the number of streaming multiprocessors on the device
  num_SMs = CudaUtils::get_sm_count(user_stream);

  // Calculate the total maximum number of active blocks
  int maxActiveBlocks = maxBlocks * num_SMs;

  //If file size is very small, launch multiple CTA per compressed chunk to perform compaction
  size_t num_CTAs_per_chunk;
  num_CTAs_per_chunk = (maxActiveBlocks + (num_chunks - 1)) / num_chunks;
  num_CTAs_per_chunk = std::min(
    static_cast<size_t>(64),
    num_CTAs_per_chunk
  ); // This number can probably be optimized further, but performance would depend on file size, and the difference would be very minor

  init_format_spec();

  setup_comp_llif_buffers<<<roundUpDiv(narrow_cast<int>(num_chunks), WARP_SIZE), WARP_SIZE, 0, user_stream>>>(
    format_spec,
    comp_format_header,
    scratch_comp_buffer,
    comp_buffer,
    uncomp_buffer,
    scratch_uncomp_buffers,
    scratch_comp_buffers,
    scratch_uncomp_sizes,
    comp_config.uncompressed_buffer_size,
    num_chunks,
    max_comp_chunk_size,
    uncomp_chunk_size,
    common_header
  );
  CUDA_CHECK(cudaGetLastError());

  ManagerBase::check<FnType::Compress>(compress_fn(
    reinterpret_cast<const void *const *>(scratch_uncomp_buffers),
    scratch_uncomp_sizes,
    num_chunks > 1 ? uncomp_chunk_size : comp_config.uncompressed_buffer_size,
    num_chunks,
    free_scratch_buffer,
    free_scratch_size,
    reinterpret_cast<void *const *>(scratch_comp_buffers),
    comp_sizes,
    format_opts,
    scratch_statuses,
    user_stream
  ));

  // Perform a max reduction for the device status
  max_reduce_device_status_single(scratch_statuses, common_status_cpu, num_chunks, user_stream);

  // Note:
  // Offsets are calculated such that the compacted chunks are aligned according to `min_alignment`.
  // The gaps between the compacted chunks are filled with 0x00 bytes.
  round_up_alignment_kernel<<<roundUpDiv(narrow_cast<int>(num_chunks), WARP_SIZE), WARP_SIZE, 0, user_stream>>>(
    comp_sizes,
    comp_chunk_offsets,
    num_chunks,
    min_alignment
  );
  CUDA_CHECK(cudaGetLastError());

  cub_exclusive_sum(
    comp_chunk_offsets,
    comp_chunk_offsets,
    num_chunks,
    free_scratch_buffer,
    free_scratch_size,
    user_stream
  );

  compact_comp_buffers_and_header_output<<<
    cuda_dim_cast(std::max(size_t(1), num_chunks - 1), num_CTAs_per_chunk, 1),
    blockSize,
    0,
    user_stream>>>(
    comp_buffer,
    scratch_comp_buffers,
    comp_sizes,
    common_header,
    comp_chunk_offsets,
    num_chunks,
    comp_config.uncompressed_buffer_size,
    uncomp_chunk_size,
    format_type,
    comp_size,
    min_alignment
  );
  CUDA_CHECK(cudaGetLastError());

  if (comp_config.compute_checksums)
  {
    store_all_checksums(
      comp_chunk_offsets,
      comp_sizes,
      comp_buffer,
      uncomp_buffer,
      uncomp_chunk_size,
      comp_chunk_checksum,
      decomp_chunk_checksum,
      scratch_buffer,
      common_header,
      comp_config,
      user_stream
    );
  }

  // Propagate the status to the config
  CUDA_CHECK(cudaLaunchHostLambda(
    user_stream,
    [comp_config, pinned_common_status = common_status_cpu]() {
      // Note: even if on the caller side `comp_config` goes out of scope,
      //       the lambda holds a copy of the object, and the underlying shared pointer.
      *comp_config.get_status() = *pinned_common_status;
    },
    false /* force_sync*/
  ));
}
TEMPLATE
void HEAD::compress_chunked_batched(
  const uint8_t *const *input_uncomp_buffers,
  uint8_t *const *output_comp_buffers,
  const std::vector<CompressionConfig> &comp_configs,
  size_t *comp_sizes
)
{
  size_t batch_count = comp_configs.size();
  size_t total_num_chunks = 0;
  size_t max_chunk_size = 0;
  size_t max_num_chunks = 0;

  // Pinned memory layout
  // [ Compression configs (batch_size, CompressionConfig) ]
  // [ Uncompressed buffer pointers (batch_size, uint8_t*) ]
  // [ Compressed buffer pointers (batch_size, uint8_t*) ]
  // [ Global chunk offsets (batch_size, size_t) ]
  // [ Compression config statuses (batch_size, nvcompStatus_t) ]
  size_t pinned_mem_size =
    batch_count * (sizeof(CompressionConfig) + 2 * sizeof(uint8_t *) + sizeof(size_t) + sizeof(nvcompStatus_t)) +
    alignof(uint8_t *) - 1;

  HostPinnedGuard guard(pinned_mem_size, alignof(CompressionConfig), user_stream);

  CompressionConfig *pinned_compression_configs = static_cast<CompressionConfig *>(guard.get_ptr());
  uint8_t **pinned_input_uncomp_buffers = roundUpToAlignment<uint8_t *>(pinned_compression_configs + batch_count);
  uint8_t **pinned_output_comp_buffers = pinned_input_uncomp_buffers + batch_count;
  size_t *pinned_chunk_offsets = reinterpret_cast<size_t *>(pinned_output_comp_buffers + batch_count);
  nvcompStatus_t *pinned_comp_config_statuses = reinterpret_cast<nvcompStatus_t *>(pinned_chunk_offsets + batch_count);

  for (int idx = 0; idx < batch_count; ++idx)
  {
    size_t batch_size = comp_configs[idx].num_chunks;
    total_num_chunks += batch_size;
    max_num_chunks = std::max(max_num_chunks, batch_size);

    if (batch_size == 1)
    {
      max_chunk_size = std::max(max_chunk_size, comp_configs[idx].uncompressed_buffer_size);
    }
    else
    {
      max_chunk_size = uncomp_chunk_size;
    }
  }

  // Sanity checks
  if (total_num_chunks == 0)
  {
    throw NVCompException(nvcompErrorInvalidValue, "The total number of chunks in the batch cannot be zero.");
  }
  else if (max_chunk_size == 0)
  {
    throw NVCompException(nvcompErrorInvalidValue, "The maximum uncompressed chunk size cannot be zero.");
  }

  CUDA_CHECK(cudaLaunchHostLambda(
    user_stream,
    [batch_count,
     comp_configs,
     input_uncomp_bufs = std::vector<const uint8_t *>(input_uncomp_buffers, input_uncomp_buffers + batch_count),
     output_comp_bufs = std::vector<uint8_t *>(output_comp_buffers, output_comp_buffers + batch_count),
     pinned_compression_configs,
     pinned_input_uncomp_buffers,
     pinned_output_comp_buffers,
     pinned_chunk_offsets]() {
      size_t running_offset = 0;
      for (size_t idx = 0; idx < batch_count; ++idx)
      {
        pinned_chunk_offsets[idx] = running_offset;
        running_offset += comp_configs[idx].num_chunks;
      }
      std::memcpy(pinned_input_uncomp_buffers, input_uncomp_bufs.data(), batch_count * sizeof(uint8_t *));
      std::memcpy(pinned_output_comp_buffers, output_comp_bufs.data(), batch_count * sizeof(uint8_t *));
      std::memcpy(pinned_compression_configs, comp_configs.data(), batch_count * sizeof(CompressionConfig));
    },
    false /* force_sync*/
  ));

  // Allocate the pinned host scratch (if necessary)
  allocate_host_scratch();

  // Update the scratch buffer as needed to work for this particular compression API
  allocate_gpu_scratch(compute_batched_compress_scratch_buffer_size(comp_configs, total_num_chunks));

  // Declare necessary vectors
  std::vector<uint8_t *> host_this_batch_comp_buffer(batch_count);
  std::vector<CommonHeader *> host_common_header(batch_count);
  std::vector<FormatSpecHeader *> host_comp_format_header(batch_count);
  std::vector<size_t *> host_this_batch_comp_chunk_offsets(batch_count);
  std::vector<size_t *> host_this_batch_comp_sizes(batch_count);

  // Scratch memory layout
  // Note: max_comp_chunk_size has been rounded up to `min_alignment`.
  // [ Compressed data (max_comp_chunk_size * (total_num_chunks - batch_count), bytes) ]
  // [ Uncompressed input chunk pointers (total_num_chunks, uint8_t*) ]
  // [ Compressed output chunk pointers (total_num_chunks, uint8_t*) ]
  // [ Uncompressed chunk sizes (total_num_chunks, size_t) ]
  // [ Compressed chunk offsets (total_num_chunks, size_t) ]
  // [ Compressed chunk sizes (total_num_chunks, size_t) ]
  // [ Compression statuses (total_num_chunks, nvcompStatus_t) ]
  // [ Compression scratch for the entire batch (bytes) ]
  uint8_t *free_scratch_buffer = scratch_buffer;

  //Scratch space for storing compressed data temporarily
  //There are total_num_chunks across all batches.
  //But the first chunk of each batch writes to the main output buffer.
  //So we only need scratch for total_num_chunks - batch_count (number of batches)
  uint8_t *scratch_comp_buffer =
    reinterpret_cast<uint8_t *>(roundUpTo(reinterpret_cast<uintptr_t>(free_scratch_buffer), min_alignment));
  free_scratch_buffer =
    reinterpret_cast<uint8_t *>(scratch_comp_buffer + (max_comp_chunk_size * (total_num_chunks - batch_count)));
  assert(max_comp_chunk_size % min_alignment == 0);

  const uint8_t **scratch_uncomp_chunk_buffers = reinterpret_cast<const uint8_t **>(free_scratch_buffer);
  free_scratch_buffer += total_num_chunks * sizeof(uint8_t *);

  uint8_t **scratch_comp_chunk_buffers = reinterpret_cast<uint8_t **>(free_scratch_buffer);
  free_scratch_buffer += total_num_chunks * sizeof(uint8_t *);

  size_t *scratch_uncomp_chunk_sizes = reinterpret_cast<size_t *>(free_scratch_buffer);
  free_scratch_buffer += total_num_chunks * sizeof(size_t);

  size_t *scratch_comp_chunk_offsets = reinterpret_cast<size_t *>(free_scratch_buffer);
  free_scratch_buffer += total_num_chunks * sizeof(size_t);

  size_t *scratch_comp_chunk_sizes = reinterpret_cast<size_t *>(free_scratch_buffer);
  free_scratch_buffer += total_num_chunks * sizeof(size_t);

  nvcompStatus_t *statuses = reinterpret_cast<nvcompStatus_t *>(free_scratch_buffer);
  free_scratch_buffer += total_num_chunks * sizeof(nvcompStatus_t);

  free_scratch_buffer =
    reinterpret_cast<uint8_t *>(roundUpTo(reinterpret_cast<uintptr_t>(free_scratch_buffer), min_alignment));
  // Note:
  // free_scratch_buffer is now aligned to min_alignment which is
  // required for the worst case compression scratch alignment
  assert(reinterpret_cast<uintptr_t>(free_scratch_buffer) % min_alignment == 0);

  const size_t free_scratch_size = scratch_buffer_size - (uintptr_t(free_scratch_buffer) - uintptr_t(scratch_buffer));
  assert(free_scratch_size >= compute_lowlevel_compress_scratch_size(total_num_chunks, uncomp_chunk_size));

  init_format_spec();

  setup_batched_comp_llif_buffers<<<roundUpDiv(narrow_cast<int>(total_num_chunks), WARP_SIZE), WARP_SIZE, 0, user_stream>>>(
    pinned_input_uncomp_buffers, //Array of pointers to input uncompressed buffer of each batch
    pinned_output_comp_buffers, //Array of pointers to output compressed buffer of each batch
    pinned_compression_configs, //Array of pointers to compression config of each batch
    total_num_chunks, //Total number of chunks across all batches
    batch_count, //Number of batches
    format_spec, //Format Spec to copy for each batch
    scratch_comp_buffer, //Start of scratch memory where we are storing output temporarily
    scratch_uncomp_chunk_buffers, //Pointer to start of each uncompressed chunk (Scratch)
    scratch_comp_chunk_buffers, //Pointer to start of each compressed chunk (Scratch)
    scratch_uncomp_chunk_sizes, //Pointer to size of each uncompressed chunk(Scratch)
    max_comp_chunk_size, //Maximum compressed size of each chunk
    uncomp_chunk_size, //Uncompressed size of each chunk
    min_alignment //Strongest required buffer alignment for the format
  );
  CUDA_CHECK(cudaGetLastError());

  ManagerBase::check<FnType::Compress>(compress_fn(
    reinterpret_cast<const void *const *>(scratch_uncomp_chunk_buffers),
    scratch_uncomp_chunk_sizes,
    max_chunk_size,
    total_num_chunks,
    free_scratch_buffer,
    free_scratch_size,
    reinterpret_cast<void *const *>(scratch_comp_chunk_buffers),
    scratch_comp_chunk_sizes,
    format_opts,
    statuses,
    user_stream
  ));

  // Perform a max reduction for the device statuses
  max_reduce_device_status_batched(
    statuses,
    pinned_comp_config_statuses,
    pinned_chunk_offsets,
    total_num_chunks,
    max_num_chunks,
    batch_count,
    user_stream
  );

  DeviceGuard device_guard(user_stream);

  constexpr int NUM_WARPS_PER_COMPACT_BLOCK = 16;
  int maxBlocks;
  int blockSize = WARP_SIZE * NUM_WARPS_PER_COMPACT_BLOCK;
  int num_SMs;

  // Determine the maximum number of active blocks per SM
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &maxBlocks,
    batched_compact_comp_buffers_and_header_output,
    blockSize,
    0
  ));

  // Get the number of streaming multiprocessors on the device
  num_SMs = CudaUtils::get_sm_count(user_stream);

  // Calculate the total maximum number of active blocks
  size_t maxActiveBlocks = maxBlocks * num_SMs;

  //If file size is very small, launch multiple CTA per compressed chunk to perform compaction
  size_t num_CTAs_per_chunk;
  num_CTAs_per_chunk = roundUpDiv(maxActiveBlocks, total_num_chunks);
  num_CTAs_per_chunk = std::min(
    static_cast<size_t>(64),
    num_CTAs_per_chunk
  ); // This number can probably be optimized further, but performance would depend on file size, and the difference would be very minor

  round_up_alignment_kernel<<<roundUpDiv(narrow_cast<int>(total_num_chunks), WARP_SIZE), WARP_SIZE, 0, user_stream>>>(
    scratch_comp_chunk_sizes,
    scratch_comp_chunk_offsets,
    total_num_chunks,
    min_alignment
  );
  CUDA_CHECK(cudaGetLastError());

  cub_exclusive_sum(
    scratch_comp_chunk_offsets,
    scratch_comp_chunk_offsets,
    total_num_chunks,
    free_scratch_buffer,
    free_scratch_size,
    user_stream
  );

  int header_size = sizeof(CommonHeader) + sizeof(FormatSpecHeader);

  batched_compact_comp_buffers_and_header_output<<<
    cuda_dim_cast(total_num_chunks, num_CTAs_per_chunk, 1),
    blockSize,
    0,
    user_stream>>>(
    batch_count,
    total_num_chunks,
    pinned_output_comp_buffers,
    pinned_compression_configs,
    scratch_comp_chunk_buffers,
    scratch_comp_chunk_sizes,
    scratch_comp_chunk_offsets,
    uncomp_chunk_size,
    format_type,
    comp_sizes,
    header_size,
    min_alignment
  );
  CUDA_CHECK(cudaGetLastError());

  // Propagate the statuses to the configs
  CUDA_CHECK(cudaLaunchHostLambda(
    user_stream,
    [batch_count, comp_configs, pinned_comp_config_statuses]() {
      // Note: even if on the caller side `comp_configs` goes out of scope,
      //       the lambda holds a copy of the object, and the underlying shared pointers.
      for (size_t idx = 0; idx < batch_count; ++idx)
      {
        *comp_configs[idx].get_status() = pinned_comp_config_statuses[idx];
      }
    },
    false /* force_sync*/
  ));
}
TEMPLATE
void HEAD::compress_raw_batched(
  const uint8_t *const *uncomp_buffers,
  uint8_t *const *comp_buffers,
  const std::vector<CompressionConfig> &comp_configs,
  size_t *comp_sizes
)
{
  size_t batch_size = comp_configs.size();

  //Find max size
  size_t max_uncomp_size = 0;
  std::vector<size_t> host_uncomp_sizes(batch_size);
  for (size_t idx = 0; idx < batch_size; ++idx)
  {
    host_uncomp_sizes[idx] = comp_configs[idx].uncompressed_buffer_size;
    max_uncomp_size = host_uncomp_sizes[idx] > max_uncomp_size ? host_uncomp_sizes[idx] : max_uncomp_size;
  }

  // Get maximum scratch requirement for compression
  size_t compress_scratch_req = compute_lowlevel_compress_scratch_size(batch_size, max_uncomp_size);

  // With ExecutionPolicy::Latency the pointer / size / status arrays that the codec kernel
  // dereferences per chunk are staged in device memory.
  const bool stage_metadata_on_device = execution_policy == ExecutionPolicy::Latency;
  if (stage_metadata_on_device)
  {
    compress_scratch_req +=
      batch_size * sizeof(size_t) + // Uncompressed sizes
      batch_size * sizeof(const uint8_t *) + // Uncompressed buffer pointers
      batch_size * sizeof(uint8_t *) + // Compressed buffer pointers
      roundUpTo(batch_size * sizeof(nvcompStatus_t), sizeof(size_t)) + // Compression statuses (aligned)
      min_alignment - 1; // Worst case alignment bytes
  }

  // Worst case alignment bytes
  compress_scratch_req += min_alignment - 1;

  //allocate scratch and assign API pointers
  allocate_gpu_scratch(compress_scratch_req);

  // Scratch memory layout
  // - for ExecutionPolicy::Latency:
  //   [ Uncompressed sizes (batch_size, size_t) ]
  //   [ Uncompressed buffer pointers (batch_size, const uint8_t*) ]
  //   [ Compressed buffer pointers (batch_size, uint8_t*) ]
  //   [ Compression statuses (batch_size, nvcompStatus_t) ]
  //
  // [ Compression scratch for the entire batch (bytes) ]
  uint8_t *free_scratch_buffer =
    reinterpret_cast<uint8_t *>(roundUpTo(reinterpret_cast<uintptr_t>(scratch_buffer), min_alignment));

  size_t *device_uncomp_sizes = nullptr;
  const uint8_t **device_uncomp_buffers = nullptr;
  uint8_t **device_comp_buffers = nullptr;
  nvcompStatus_t *device_comp_statuses = nullptr;
  if (stage_metadata_on_device)
  {
    device_uncomp_sizes = reinterpret_cast<size_t *>(free_scratch_buffer);
    free_scratch_buffer += batch_size * sizeof(size_t);

    device_uncomp_buffers = reinterpret_cast<const uint8_t **>(free_scratch_buffer);
    free_scratch_buffer += batch_size * sizeof(const uint8_t *);

    device_comp_buffers = reinterpret_cast<uint8_t **>(free_scratch_buffer);
    free_scratch_buffer += batch_size * sizeof(uint8_t *);

    device_comp_statuses = reinterpret_cast<nvcompStatus_t *>(free_scratch_buffer);
    free_scratch_buffer += roundUpTo(batch_size * sizeof(nvcompStatus_t), sizeof(size_t));

    // The staged arrays are only size_t-aligned, so realign before the codec scratch.
    free_scratch_buffer =
      reinterpret_cast<uint8_t *>(roundUpTo(reinterpret_cast<uintptr_t>(free_scratch_buffer), min_alignment));
  }

  // Note:
  // free_scratch_buffer now is aligned to min_alignment
  // as the worst-case scratch buffer requirement for compression.
  assert(reinterpret_cast<uintptr_t>(free_scratch_buffer) % min_alignment == 0);
  assert(free_scratch_buffer <= scratch_buffer + scratch_buffer_size);

  const size_t free_scratch_size = scratch_buffer_size - (uintptr_t(free_scratch_buffer) - uintptr_t(scratch_buffer));
  assert(free_scratch_size >= compute_lowlevel_compress_scratch_size(batch_size, max_uncomp_size));

  // Pinned memory layout
  // [ Uncompressed sizes (batch_size, size_t) ]
  // [ Uncompressed buffer pointers (batch_size, uint8_t*) ]
  // [ Compressed buffer pointers (batch_size, uint8_t*) ]
  // [ Compression config statuses (batch_size, nvcompStatus_t) ]
  size_t pinned_mem_size = batch_size *
                           (sizeof(size_t) + sizeof(const uint8_t *) + sizeof(uint8_t *) + sizeof(nvcompStatus_t));

  HostPinnedGuard guard(pinned_mem_size, sizeof(size_t), user_stream);

  size_t *pinned_uncomp_sizes = static_cast<size_t *>(guard.get_ptr());
  const uint8_t **pinned_uncomp_buffers = reinterpret_cast<const uint8_t **>(pinned_uncomp_sizes + batch_size);
  uint8_t **pinned_comp_buffers =
    reinterpret_cast<uint8_t **>(reinterpret_cast<void *>(pinned_uncomp_buffers + batch_size));
  nvcompStatus_t *pinned_comp_statuses = reinterpret_cast<nvcompStatus_t *>(pinned_comp_buffers + batch_size);

  if (bitstream_kind == BitstreamKind::WITH_UNCOMPRESSED_SIZE)
  {
    static_assert(UNCOMPRESSED_SIZE_OFFSET == 2 * sizeof(size_t));
    for (size_t idx = 0; idx < batch_size; ++idx)
    {
      // Avoid uninitialized memory
      const size_t prefix[2] = {comp_configs[idx].uncompressed_buffer_size, 0};
      CUDA_CHECK(cudaMemcpyAsync(comp_buffers[idx], prefix, sizeof(prefix), cudaMemcpyHostToDevice, user_stream));
    }
  }

  const size_t comp_buf_offset = (bitstream_kind == BitstreamKind::WITH_UNCOMPRESSED_SIZE) ? UNCOMPRESSED_SIZE_OFFSET
                                                                                           : 0;
  assert(UNCOMPRESSED_SIZE_OFFSET % min_alignment == 0);
  CUDA_CHECK(cudaLaunchHostLambda(
    user_stream,
    [batch_size,
     comp_buf_offset,
     host_uncomp_sizes,
     uncomp_bufs = std::vector<const uint8_t *>(uncomp_buffers, uncomp_buffers + batch_size),
     comp_bufs = std::vector<uint8_t *>(comp_buffers, comp_buffers + batch_size),
     pinned_uncomp_sizes,
     pinned_uncomp_buffers,
     pinned_comp_buffers]() {
      std::memcpy(pinned_uncomp_sizes, host_uncomp_sizes.data(), batch_size * sizeof(size_t));
      std::memcpy(pinned_uncomp_buffers, uncomp_bufs.data(), batch_size * sizeof(const uint8_t *));
      for (size_t idx = 0; idx < batch_size; ++idx)
      {
        pinned_comp_buffers[idx] = comp_bufs[idx] + comp_buf_offset;
      }
    },
    false /* force_sync*/
  ));

  if (stage_metadata_on_device)
  {
    // Stream-ordered, so these run after the host lambda above has populated the pinned arrays.
    CUDA_CHECK(cudaMemcpyAsync(
      device_uncomp_sizes,
      pinned_uncomp_sizes,
      batch_size * sizeof(size_t),
      cudaMemcpyHostToDevice,
      user_stream
    ));
    CUDA_CHECK(cudaMemcpyAsync(
      device_uncomp_buffers,
      pinned_uncomp_buffers,
      batch_size * sizeof(const uint8_t *),
      cudaMemcpyHostToDevice,
      user_stream
    ));
    CUDA_CHECK(cudaMemcpyAsync(
      device_comp_buffers,
      pinned_comp_buffers,
      batch_size * sizeof(uint8_t *),
      cudaMemcpyHostToDevice,
      user_stream
    ));
  }

  const uint8_t **kernel_uncomp_buffers = stage_metadata_on_device ? device_uncomp_buffers : pinned_uncomp_buffers;
  size_t *kernel_uncomp_sizes = stage_metadata_on_device ? device_uncomp_sizes : pinned_uncomp_sizes;
  uint8_t **kernel_comp_buffers = stage_metadata_on_device ? device_comp_buffers : pinned_comp_buffers;
  nvcompStatus_t *kernel_comp_statuses = stage_metadata_on_device ? device_comp_statuses : pinned_comp_statuses;

  ManagerBase::check<FnType::Compress>(compress_fn(
    reinterpret_cast<const void *const *>(kernel_uncomp_buffers),
    kernel_uncomp_sizes,
    max_uncomp_size,
    batch_size,
    free_scratch_buffer,
    free_scratch_size,
    reinterpret_cast<void *const *>(kernel_comp_buffers),
    comp_sizes,
    format_opts,
    kernel_comp_statuses,
    user_stream
  ));

  if (stage_metadata_on_device)
  {
    CUDA_CHECK(cudaMemcpyAsync(
      pinned_comp_statuses,
      device_comp_statuses,
      batch_size * sizeof(nvcompStatus_t),
      cudaMemcpyDeviceToHost,
      user_stream
    ));
  }

  if (bitstream_kind == BitstreamKind::WITH_UNCOMPRESSED_SIZE)
  {
    increase_array_by(comp_sizes, batch_size, UNCOMPRESSED_SIZE_OFFSET, user_stream);
  }

  // Propagate the statuses to the configs
  CUDA_CHECK(cudaLaunchHostLambda(
    user_stream,
    [batch_size, comp_configs, pinned_comp_statuses]() {
      // Note: even if on the caller side `comp_configs` goes out of scope,
      //       the lambda holds a copy of the object, and the underlying shared pointers.
      for (size_t idx = 0; idx < batch_size; ++idx)
      {
        *comp_configs[idx].get_status() = pinned_comp_statuses[idx];
      }
    },
    false /* force_sync*/
  ));
}

#undef HEAD
#undef TEMPLATE

} // namespace nvcomp
