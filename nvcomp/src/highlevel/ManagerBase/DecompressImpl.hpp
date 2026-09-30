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

// Out-of-line member-function definitions for the ManagerBase class template.
// Included at the bottom of ManagerBase.hpp; included here as well so this file
// is self-contained, but it cannot be used without the ManagerBase declaration.

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
void HEAD::decompress_chunked_single(
  uint8_t *decomp_buffer,
  const uint8_t *comp_buffer,
  const DecompressionConfig &decomp_config
)
{
  // Allocate the pinned host scratch (if necessary)
  allocate_host_scratch();

  // Update the scratch buffer as needed to work for this particular compression API
  allocate_gpu_scratch(compute_decompress_scratch_buffer_size(decomp_config.num_chunks));

  const CommonHeader *common_header = reinterpret_cast<const CommonHeader *>(comp_buffer);
  comp_buffer += sizeof(CommonHeader) + sizeof(FormatSpecHeader);

  auto &checksums_present = decomp_config.checksums_present;
  bool verify_checksums = checksums_present && (checksum_policy != ChecksumPolicy::NoComputeNoVerify) &&
                          (checksum_policy != ChecksumPolicy::ComputeAndNoVerify);

  const size_t num_chunks = decomp_config.num_chunks;

  // Scratch memory layout
  // [ Decompressed chunk pointers (num_chunks, uint8_t*) ]
  // [ Compressed chunk pointers (num_chunks, uint8_t*) ]
  // [ Decompression buffer sizes (num_chunks, size_t) ]
  // [ Decompressed actual sizes (num_chunks, size_t) ]
  // [ Decompression statuses (num_chunks, nvcompStatus_t) ]
  // [ Decompression scratch for the entire batch (bytes) ]
  uint8_t *free_scratch_buffer = scratch_buffer;

  uint8_t **scratch_uncomp_buffers = roundUpToAlignment<uint8_t *>(free_scratch_buffer);
  free_scratch_buffer = reinterpret_cast<uint8_t *>(scratch_uncomp_buffers + num_chunks);

  const uint8_t **scratch_comp_buffers = reinterpret_cast<const uint8_t **>(free_scratch_buffer);
  free_scratch_buffer += num_chunks * sizeof(const uint8_t *);

  size_t *scratch_uncomp_sizes = reinterpret_cast<size_t *>(free_scratch_buffer);
  free_scratch_buffer += num_chunks * sizeof(size_t);

  size_t *scratch_actual_uncomp_sizes = reinterpret_cast<size_t *>(free_scratch_buffer);
  free_scratch_buffer += num_chunks * sizeof(size_t);

  nvcompStatus_t *scratch_statuses = reinterpret_cast<nvcompStatus_t *>(free_scratch_buffer);
  free_scratch_buffer += num_chunks * sizeof(nvcompStatus_t);
  free_scratch_buffer =
    reinterpret_cast<uint8_t *>(roundUpTo(reinterpret_cast<uintptr_t>(free_scratch_buffer), min_alignment));

  // Note:
  // free_scratch_buffer now must be aligned to min_alignment
  // as the worst case alignment requirement for decompression
  assert(reinterpret_cast<uintptr_t>(free_scratch_buffer) % min_alignment == 0);

  const size_t free_scratch_size = scratch_buffer_size - (uintptr_t(free_scratch_buffer) - uintptr_t(scratch_buffer));
  assert(free_scratch_size >= compute_lowlevel_decompress_scratch_size(num_chunks, uncomp_chunk_size));

  const size_t *comp_chunk_offsets = roundUpToAlignment<size_t>(comp_buffer);
  const size_t *comp_sizes = comp_chunk_offsets + num_chunks;

  const size_t *provided_uncomp_chunk_offsets = nullptr;
  const size_t *provided_uncomp_sizes = nullptr;

  const bool uncomp_sizes_and_offsets_provided = decomp_config.impl->uncompressed_sizes_and_offsets_provided;
  if (uncomp_sizes_and_offsets_provided)
  {
    provided_uncomp_chunk_offsets = comp_sizes + num_chunks;
    provided_uncomp_sizes = provided_uncomp_chunk_offsets + num_chunks;
    comp_buffer = reinterpret_cast<const uint8_t *>(provided_uncomp_sizes + num_chunks);
  }
  else
  {
    comp_buffer = reinterpret_cast<const uint8_t *>(comp_sizes + num_chunks);
  }

  const uint32_t *comp_chunk_checksum = nullptr;
  const uint32_t *decomp_chunk_checksum = nullptr;

  if (checksums_present)
  {
    // Should be aligned to 8-byte boundary
    comp_chunk_checksum = reinterpret_cast<const uint32_t *>(roundUpToAlignment<uint64_t>(comp_buffer));
    decomp_chunk_checksum = comp_chunk_checksum + num_chunks;
    comp_buffer = reinterpret_cast<const uint8_t *>(decomp_chunk_checksum + num_chunks);
  }

  comp_buffer = reinterpret_cast<const uint8_t *>(roundUpTo(reinterpret_cast<uintptr_t>(comp_buffer), min_alignment));

  setup_decomp_llif_buffers<<<roundUpDiv(narrow_cast<int>(num_chunks), WARP_SIZE), WARP_SIZE, 0, user_stream>>>(
    common_header,
    decomp_buffer,
    comp_buffer,
    scratch_uncomp_buffers,
    scratch_comp_buffers,
    comp_chunk_offsets,
    scratch_uncomp_sizes,
    provided_uncomp_chunk_offsets,
    provided_uncomp_sizes,
    uncomp_sizes_and_offsets_provided
  );
  CUDA_CHECK(cudaGetLastError());

  ManagerBase::check<FnType::Decompress>(decompress_fn(
    reinterpret_cast<const void *const *>(scratch_comp_buffers),
    comp_sizes,
    scratch_uncomp_sizes,
    scratch_actual_uncomp_sizes,
    num_chunks,
    free_scratch_buffer,
    free_scratch_size,
    reinterpret_cast<void *const *>(scratch_uncomp_buffers),
    scratch_statuses,
    format_opts,
    decompress_opts,
    user_stream,
    nullptr /*host_comp_buffers*/
  ));

  // Perform a max reduction for the device status
  max_reduce_device_status_single(scratch_statuses, common_status_cpu, num_chunks, user_stream);

  if (verify_checksums)
  {
    verify_all_checksums(
      comp_chunk_offsets,
      comp_sizes,
      comp_buffer,
      decomp_buffer,
      uncomp_chunk_size,
      comp_chunk_checksum,
      decomp_chunk_checksum,
      scratch_buffer,
      common_header,
      decomp_config,
      common_status_cpu,
      user_stream
    );
  }

  // Propagate the status to the config
  CUDA_CHECK(cudaLaunchHostLambda(
    user_stream,
    [decomp_config, pinned_common_status = common_status_cpu]() {
      // Note: even if on the caller side `decomp_config` goes out of scope,
      //       the lambda holds a copy of the object, and the underlying shared pointer.
      *decomp_config.get_status() = *pinned_common_status;
    },
    false /* force_sync*/
  ));
}
TEMPLATE
void HEAD::compute_decompress_hlif_mem_size(
  bool host_setup_mode,
  bool decomp_config_used,
  size_t batch_count,
  size_t &total_num_chunks,
  size_t &max_num_chunks,
  size_t &pinned_mem_size,
  size_t &total_decompress_scratch_req,
  const uint8_t *const *host_comp_buffers,
  const std::vector<DecompressionConfig> &decomp_configs
)
{
  total_decompress_scratch_req = 2 * (min_alignment - 1); //Alignment bytes

  // Pinned memory layout
  // [ Compressed input buffer pointers (batch_count, uint8_t*) ]
  // [ Decompressed output buffer pointers (batch_count, uint8_t*) ]
  pinned_mem_size = 2 * batch_count * sizeof(uint8_t *); //For input_comp_buffers and output_decomp_buffers

  // Determine total number of chunks and max number of chunks
  for (size_t batch_id = 0; batch_id < batch_count; ++batch_id)
  {
    size_t batch_num_chunks = 0;

    if (host_setup_mode)
    {
      const CommonHeader *const common_header = reinterpret_cast<const CommonHeader *>(host_comp_buffers[batch_id]);
      batch_num_chunks = common_header->num_chunks;
    }
    else
    {
      batch_num_chunks = decomp_configs[batch_id].num_chunks;
    }
    total_num_chunks += batch_num_chunks;
    max_num_chunks = std::max(max_num_chunks, batch_num_chunks);
  }

  if (total_num_chunks == 0)
  {
    throw NVCompException(nvcompErrorInvalidValue, "The total number of chunks in the batch cannot be zero.");
  }

  if (host_setup_mode)
  {
    pinned_mem_size += (3 * total_num_chunks *
                        sizeof(uint8_t *)) //For uncomp_chunk_buffers, comp_chunk_buffers, and host_comp_chunk_buffers
                       + (2 * total_num_chunks * sizeof(size_t)) //For uncomp_sizes, comp_chunk_sizes
                       + (total_num_chunks * sizeof(nvcompStatus_t)); // For decompression statuses
    total_decompress_scratch_req += total_num_chunks * sizeof(size_t); //For actual_uncomp_chunk_sizes
  }
  else
  {
    pinned_mem_size += batch_count * sizeof(size_t) // For host_global_chunk_offset
                       + batch_count * sizeof(nvcompStatus_t); // For the decompression config statuses

    total_decompress_scratch_req +=
      total_num_chunks *
        (5 * sizeof(size_t) +
         sizeof(nvcompStatus_t)) // 5 * sizeof(size_t) represents the API requirements for the HLIF call
      + roundUpTo(
          batch_count * sizeof(nvcompStatus_t),
          min_alignment
        ); //To store the max device status for each item in the batch
  }
  //Calculate the scratch size required for the decompression kernels
  size_t format_decomp_scratch_req = compute_lowlevel_decompress_scratch_size(total_num_chunks, uncomp_chunk_size);
  total_decompress_scratch_req += format_decomp_scratch_req;
}
TEMPLATE
void HEAD::run_decompress_chunked_batched_host_path(
  const uint8_t *const *host_comp_buffers,
  uint8_t **pinned_input_comp_buffers,
  uint8_t **pinned_output_decomp_buffers,
  size_t batch_count,
  size_t total_num_chunks,
  int header_size,
  bool decomp_config_used,
  const std::vector<DecompressionConfig> &decomp_configs
)
{
  uint8_t *free_scratch_buffer = reinterpret_cast<uint8_t *>(roundUpToAlignment<uint8_t *>(scratch_buffer));

  // Pinned layout: [input_comp][output_decomp][uncomp_chunk_buffers...]
  uint8_t **pinned_uncomp_chunk_buffers = reinterpret_cast<uint8_t **>(pinned_output_decomp_buffers + batch_count);
  const uint8_t **pinned_comp_chunk_buffers =
    reinterpret_cast<const uint8_t **>(reinterpret_cast<void *>(pinned_uncomp_chunk_buffers + total_num_chunks));
  const uint8_t **pinned_host_comp_chunk_buffers =
    reinterpret_cast<const uint8_t **>(pinned_comp_chunk_buffers + total_num_chunks);
  size_t *pinned_uncomp_chunk_sizes = reinterpret_cast<size_t *>(pinned_host_comp_chunk_buffers + total_num_chunks);
  size_t *pinned_comp_chunk_sizes = reinterpret_cast<size_t *>(pinned_uncomp_chunk_sizes + total_num_chunks);
  nvcompStatus_t *pinned_statuses = reinterpret_cast<nvcompStatus_t *>(pinned_comp_chunk_sizes + total_num_chunks);

  size_t *scratch_actual_uncomp_chunk_sizes =
    reinterpret_cast<size_t *>(roundUpToAlignment<size_t *>(free_scratch_buffer));
  free_scratch_buffer += total_num_chunks * sizeof(size_t);
  free_scratch_buffer =
    reinterpret_cast<uint8_t *>(roundUpTo(reinterpret_cast<uintptr_t>(free_scratch_buffer), min_alignment));
  const size_t free_scratch_size = scratch_buffer_size - (uintptr_t(free_scratch_buffer) - uintptr_t(scratch_buffer));
  assert(free_scratch_size >= compute_lowlevel_decompress_scratch_size(total_num_chunks, uncomp_chunk_size));

  // Local copy so the host lambda captures a plain variable (a member cannot be
  // captured by name without also capturing `this`).
  const size_t min_alignment_local = min_alignment;
  CUDA_CHECK(cudaLaunchHostLambda(
    user_stream,
    [pinned_input_comp_buffers,
     host_comp_buffers,
     pinned_output_decomp_buffers,
     header_size,
     pinned_uncomp_chunk_buffers,
     pinned_comp_chunk_buffers,
     pinned_host_comp_chunk_buffers,
     pinned_uncomp_chunk_sizes,
     pinned_comp_chunk_sizes,
     batch_count,
     min_alignment_local]() {
      setup_batched_decomp_llif_buffers_host(
        pinned_input_comp_buffers,
        host_comp_buffers,
        pinned_output_decomp_buffers,
        header_size,
        pinned_uncomp_chunk_buffers,
        pinned_comp_chunk_buffers,
        pinned_host_comp_chunk_buffers,
        pinned_uncomp_chunk_sizes,
        pinned_comp_chunk_sizes,
        batch_count,
        min_alignment_local
      );
    },
    true /* force_sync*/
  ));

  ManagerBase::check<FnType::Decompress>(decompress_fn(
    reinterpret_cast<const void *const *>(pinned_comp_chunk_buffers),
    pinned_comp_chunk_sizes,
    pinned_uncomp_chunk_sizes,
    scratch_actual_uncomp_chunk_sizes,
    total_num_chunks,
    free_scratch_buffer,
    free_scratch_size,
    reinterpret_cast<void *const *>(pinned_uncomp_chunk_buffers),
    pinned_statuses,
    format_opts,
    decompress_opts,
    user_stream,
    reinterpret_cast<const void *const *>(pinned_host_comp_chunk_buffers)
  ));

  // If decomp_config is used, run max reduction in a host callback after the decompress kernel.
  if (decomp_config_used)
  {
    CUDA_CHECK(cudaLaunchHostLambda(
      user_stream,
      [pinned_statuses, decomp_configs, batch_count]() {
        max_reduce_host_status_batched(pinned_statuses, decomp_configs, batch_count);
      },
      true /* force_sync*/
    ));
  }
}
TEMPLATE
void HEAD::run_decompress_chunked_batched_device_path(
  uint8_t **pinned_input_comp_buffers,
  uint8_t **pinned_output_decomp_buffers,
  size_t batch_count,
  size_t total_num_chunks,
  size_t max_num_chunks,
  size_t *pinned_global_chunk_offset,
  nvcompStatus_t *pinned_decomp_config_statuses,
  int header_size,
  const std::vector<DecompressionConfig> &decomp_configs
)
{
  // GPU Scratch memory layout
  // [ Decompressed chunk pointers (total_num_chunks, uint8_t*) ]
  // [ Compressed chunk pointers (total_num_chunks, uint8_t*) ]
  // [ Decompressed chunk buffer sizes (total_num_chunks, size_t) ]
  // [ Decompressed actual chunk sizes (total_num_chunks, size_t) ]
  // [ Compressed chunk sizes (total_num_chunks, size_t) ]
  // [ Decompression statuses (total_num_chunks, nvcompStatus_t) ]
  // [ Decompression scratch (bytes) ]
  uint8_t *free_scratch_buffer = scratch_buffer;

  uint8_t **scratch_uncomp_chunk_buffers = roundUpToAlignment<uint8_t *>(free_scratch_buffer);
  free_scratch_buffer = reinterpret_cast<uint8_t *>(scratch_uncomp_chunk_buffers + total_num_chunks);

  const uint8_t **scratch_comp_chunk_buffers = reinterpret_cast<const uint8_t **>(free_scratch_buffer);
  free_scratch_buffer += total_num_chunks * sizeof(const uint8_t *);

  size_t *scratch_uncomp_chunk_sizes = reinterpret_cast<size_t *>(free_scratch_buffer);
  free_scratch_buffer += total_num_chunks * sizeof(size_t);

  size_t *scratch_actual_uncomp_chunk_sizes = reinterpret_cast<size_t *>(free_scratch_buffer);
  free_scratch_buffer += total_num_chunks * sizeof(size_t);

  size_t *scratch_comp_chunk_sizes = reinterpret_cast<size_t *>(free_scratch_buffer);
  free_scratch_buffer += total_num_chunks * sizeof(size_t);

  nvcompStatus_t *scratch_statuses = reinterpret_cast<nvcompStatus_t *>(free_scratch_buffer);
  free_scratch_buffer += total_num_chunks * sizeof(nvcompStatus_t);
  free_scratch_buffer =
    reinterpret_cast<uint8_t *>(roundUpTo(reinterpret_cast<uintptr_t>(free_scratch_buffer), min_alignment));

  // Note:
  // free_scratch_buffer must be aligned to min_alignment to accomodate
  // the worst-case decompression scratch alignment.

  assert(reinterpret_cast<uintptr_t>(free_scratch_buffer) % min_alignment == 0);
  assert(free_scratch_buffer <= scratch_buffer + scratch_buffer_size);
  const size_t free_scratch_size = scratch_buffer_size - (uintptr_t(free_scratch_buffer) - uintptr_t(scratch_buffer));
  assert(free_scratch_size >= compute_lowlevel_decompress_scratch_size(total_num_chunks, uncomp_chunk_size));

  // The pinned memory is already allocated and populated in decompress_chunked_batched.

  DeviceGuard device_guard(user_stream);

  constexpr int NUM_WARPS_PER_BATCH = 8;
  int maxBlocks;
  int blockSize = WARP_SIZE * NUM_WARPS_PER_BATCH;
  int num_SMs;

  //Determine the maximum number of active blocks per SM
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocks, setup_batched_decomp_llif_buffers, blockSize, 0));
  num_SMs = CudaUtils::get_sm_count(user_stream);
  //Calculate the total maximum number of active blocks
  size_t maxActiveBlocks = maxBlocks * num_SMs;

  //If batch size is very small, launch multiple CTA per compressed chunk to perform compaction
  size_t num_CTAs_per_batch;
  num_CTAs_per_batch = roundUpDiv(maxActiveBlocks, batch_count);
  num_CTAs_per_batch = std::min(
    static_cast<size_t>(64),
    num_CTAs_per_batch
  ); // This number can probably be optimized further, but performance would depend on batch size, and the difference would be very minor

  setup_batched_decomp_llif_buffers<<<cuda_dim_cast(num_CTAs_per_batch, batch_count, 1), blockSize, 0, user_stream>>>(
    pinned_global_chunk_offset,
    pinned_input_comp_buffers,
    pinned_output_decomp_buffers,
    header_size,
    scratch_uncomp_chunk_buffers,
    scratch_comp_chunk_buffers,
    scratch_uncomp_chunk_sizes,
    scratch_comp_chunk_sizes,
    min_alignment
  );
  CUDA_CHECK(cudaGetLastError());

  ManagerBase::check<FnType::Decompress>(decompress_fn(
    reinterpret_cast<const void *const *>(scratch_comp_chunk_buffers),
    scratch_comp_chunk_sizes,
    scratch_uncomp_chunk_sizes,
    scratch_actual_uncomp_chunk_sizes,
    total_num_chunks,
    free_scratch_buffer,
    free_scratch_size,
    reinterpret_cast<void *const *>(scratch_uncomp_chunk_buffers),
    scratch_statuses,
    format_opts,
    decompress_opts,
    user_stream,
    nullptr /*host_comp_chunk_buffers*/
  ));

  // Perform a max reduction for the device statuses
  max_reduce_device_status_batched(
    scratch_statuses,
    pinned_decomp_config_statuses,
    pinned_global_chunk_offset,
    total_num_chunks,
    max_num_chunks,
    batch_count,
    user_stream
  );

  // Propagate the statuses to the configs
  CUDA_CHECK(cudaLaunchHostLambda(
    user_stream,
    [batch_count, decomp_configs, pinned_decomp_config_statuses]() {
      for (size_t idx = 0; idx < batch_count; ++idx)
      {
        *decomp_configs[idx].get_status() = pinned_decomp_config_statuses[idx];
      }
    },
    false /* force_sync*/
  ));
}
TEMPLATE
void HEAD::decompress_chunked_batched(
  uint8_t *const *output_decomp_buffers,
  const uint8_t *const *input_comp_buffers,
  const std::vector<DecompressionConfig> &decomp_configs,
  const size_t batch_count,
  const uint8_t *const *host_comp_buffers
)
{
  size_t total_num_chunks = 0;
  size_t max_num_chunks = 0;
  size_t pinned_mem_size = 0;
  size_t total_decompress_scratch_req = 0;
  const int header_size = sizeof(CommonHeader) + sizeof(FormatSpecHeader);

  bool host_setup_mode = host_comp_buffers != nullptr;
  bool force_sync = host_setup_mode;
  bool decomp_config_used = (decomp_configs.size() == batch_count); //Always true in device path

  compute_decompress_hlif_mem_size(
    host_setup_mode,
    decomp_config_used,
    batch_count,
    total_num_chunks,
    max_num_chunks,
    pinned_mem_size,
    total_decompress_scratch_req,
    host_comp_buffers,
    decomp_configs
  );

  allocate_gpu_scratch(total_decompress_scratch_req);

  // Copy data from the original host buffers to the pinned memory. Required for both host and device setup.
  HostPinnedGuard guard(pinned_mem_size, alignof(size_t), user_stream);

  uint8_t **pinned_input_comp_buffers = static_cast<uint8_t **>(guard.get_ptr());
  uint8_t **pinned_output_decomp_buffers = reinterpret_cast<uint8_t **>(pinned_input_comp_buffers + batch_count);
  size_t *pinned_global_chunk_offset = reinterpret_cast<size_t *>(pinned_output_decomp_buffers + batch_count);
  nvcompStatus_t *pinned_decomp_config_statuses =
    reinterpret_cast<nvcompStatus_t *>(pinned_global_chunk_offset + batch_count);

  CUDA_CHECK(cudaLaunchHostLambda(
    user_stream,
    [batch_count,
     host_setup_mode,
     decomp_config_used,
     decomp_configs,
     input_comp_bufs = std::vector<const uint8_t *>(input_comp_buffers, input_comp_buffers + batch_count),
     output_decomp_bufs = std::vector<uint8_t *>(output_decomp_buffers, output_decomp_buffers + batch_count),
     pinned_input_comp_buffers,
     pinned_output_decomp_buffers,
     pinned_global_chunk_offset]() {
      std::memcpy(pinned_input_comp_buffers, input_comp_bufs.data(), batch_count * sizeof(uint8_t *));
      std::memcpy(pinned_output_decomp_buffers, output_decomp_bufs.data(), batch_count * sizeof(uint8_t *));
      if (decomp_config_used)
      {
        size_t running_offset = 0;
        for (size_t batch_id = 0; batch_id < batch_count; ++batch_id)
        {
          if (!host_setup_mode)
          {
            pinned_global_chunk_offset[batch_id] = running_offset;
            running_offset += decomp_configs[batch_id].num_chunks;
          }
        }
      }
    },
    force_sync
  ));

  if (host_setup_mode)
  {
    run_decompress_chunked_batched_host_path(
      host_comp_buffers,
      pinned_input_comp_buffers,
      pinned_output_decomp_buffers,
      batch_count,
      total_num_chunks,
      header_size,
      decomp_config_used,
      decomp_configs
    );
  }
  else
  {
    run_decompress_chunked_batched_device_path(
      pinned_input_comp_buffers,
      pinned_output_decomp_buffers,
      batch_count,
      total_num_chunks,
      max_num_chunks,
      pinned_global_chunk_offset,
      pinned_decomp_config_statuses,
      header_size,
      decomp_configs
    );
  }
}
TEMPLATE
void HEAD::decompress_raw_batched(
  uint8_t *const *decomp_buffers,
  const uint8_t *const *comp_buffers,
  const std::vector<DecompressionConfig> &decomp_configs,
  const size_t *comp_sizes
)
{
  size_t batch_size = decomp_configs.size();

  //Find max size
  size_t max_decomp_size = 0;
  std::vector<size_t> host_decomp_sizes(batch_size);
  for (size_t idx = 0; idx < batch_size; ++idx)
  {
    host_decomp_sizes[idx] = decomp_configs[idx].decomp_data_size;
    max_decomp_size = host_decomp_sizes[idx] > max_decomp_size ? host_decomp_sizes[idx] : max_decomp_size;
  }

  // Get maximum scratch requirement for decompression
  size_t decompress_scratch_req = compute_lowlevel_decompress_scratch_size(batch_size, max_decomp_size);

  // Device scratch for actual_decomp_sizes and statuses only
  decompress_scratch_req += batch_size * sizeof(size_t);

  if (bitstream_kind == BitstreamKind::WITH_UNCOMPRESSED_SIZE)
  {
    decompress_scratch_req += sizeof(size_t) * batch_size; // mutable_comp_sizes
  }

  // With ExecutionPolicy::Latency the pointer / size / status arrays that the codec kernel
  // dereferences per chunk are staged in device memory.
  const bool stage_metadata_on_device = execution_policy == ExecutionPolicy::Latency;
  if (stage_metadata_on_device)
  {
    decompress_scratch_req +=
      batch_size * sizeof(size_t) + // Decompressed sizes array
      batch_size * sizeof(uint8_t *) + // Decompressed buffer pointers
      batch_size * sizeof(const uint8_t *) + // Compressed buffer pointers
      roundUpTo(batch_size * sizeof(nvcompStatus_t), sizeof(size_t)) + // Decompression statuses (rounded)
      min_alignment - 1; // Alignment padding
  }

  // Alignment bytes: cover the min_alignment round-up of the decompression
  // scratch region (see decompress_raw_batched).
  decompress_scratch_req += min_alignment - 1;

  allocate_gpu_scratch(decompress_scratch_req);

  // Scratch memory layout
  // [ Actual decompressed sizes (batch_size, size_t) ]
  //
  // - for BitstreamKind::WITH_UNCOMPRESSED_SIZE:
  //   [ Mutable compressed sizes (batch_size, size_t) ]
  //
  // - for ExecutionPolicy::Latency:
  //   [ Decompressed sizes (batch_size, size_t) ]
  //   [ Decompressed buffer pointers (batch_size, uint8_t*) ]
  //   [ Compressed buffer pointers (batch_size, const uint8_t*) ]
  //   [ Decompression statuses (batch_size, nvcompStatus_t) ]
  //
  // [ Decompression scratch space (bytes)]
  uint8_t *free_scratch_buffer = scratch_buffer;

  size_t *actual_decomp_sizes = roundUpToAlignment<size_t>(free_scratch_buffer);
  free_scratch_buffer = reinterpret_cast<uint8_t *>(actual_decomp_sizes + batch_size);

  if (bitstream_kind == BitstreamKind::WITH_UNCOMPRESSED_SIZE)
  {
    size_t *mutable_comp_sizes = reinterpret_cast<size_t *>(free_scratch_buffer);
    free_scratch_buffer += batch_size * sizeof(size_t);

    CUDA_CHECK(
      cudaMemcpyAsync(mutable_comp_sizes, comp_sizes, sizeof(size_t) * batch_size, cudaMemcpyDeviceToDevice, user_stream)
    );
    decrease_array_by(mutable_comp_sizes, batch_size, UNCOMPRESSED_SIZE_OFFSET, user_stream);
    comp_sizes = mutable_comp_sizes;
  }
  free_scratch_buffer =
    reinterpret_cast<uint8_t *>(roundUpTo(reinterpret_cast<uintptr_t>(free_scratch_buffer), min_alignment));

  size_t *device_decomp_sizes = nullptr;
  uint8_t **device_decomp_buffers = nullptr;
  const uint8_t **device_comp_buffers = nullptr;
  nvcompStatus_t *device_decomp_statuses = nullptr;
  if (stage_metadata_on_device)
  {
    device_decomp_sizes = reinterpret_cast<size_t *>(free_scratch_buffer);
    free_scratch_buffer += batch_size * sizeof(size_t);

    device_decomp_buffers = reinterpret_cast<uint8_t **>(free_scratch_buffer);
    free_scratch_buffer += batch_size * sizeof(uint8_t *);

    device_comp_buffers = reinterpret_cast<const uint8_t **>(free_scratch_buffer);
    free_scratch_buffer += batch_size * sizeof(const uint8_t *);

    device_decomp_statuses = reinterpret_cast<nvcompStatus_t *>(free_scratch_buffer);
    free_scratch_buffer += roundUpTo(batch_size * sizeof(nvcompStatus_t), sizeof(size_t));

    // The staged arrays are only size_t-aligned, so realign before the codec scratch.
    free_scratch_buffer =
      reinterpret_cast<uint8_t *>(roundUpTo(reinterpret_cast<uintptr_t>(free_scratch_buffer), min_alignment));
  }

  // Note:
  // free_scratch_buffer must be aligned to min_alignment to
  // accomodate the worst-case decompression scratch requirement.
  assert(reinterpret_cast<uintptr_t>(free_scratch_buffer) % min_alignment == 0);
  assert(free_scratch_buffer <= scratch_buffer + scratch_buffer_size);

  const size_t free_scratch_size = scratch_buffer_size - (uintptr_t(free_scratch_buffer) - uintptr_t(scratch_buffer));
  assert(free_scratch_size >= compute_lowlevel_decompress_scratch_size(batch_size, max_decomp_size));

  // Pinned memory layout
  // [ Decompressed sizes (batch_size, size_t) ]
  // [ Decompressed buffer pointers (batch_size, uint8_t*) ]
  // [ Compressed buffer pointers (batch_size, const uint8_t*) ]
  // [ Decompression config statuses (batch_size, nvcompStatus_t) ]
  size_t pinned_mem_size = batch_size *
                           (sizeof(size_t) + sizeof(uint8_t *) + sizeof(const uint8_t *) + sizeof(nvcompStatus_t));

  HostPinnedGuard guard(pinned_mem_size, sizeof(size_t), user_stream);

  size_t *pinned_decomp_sizes = static_cast<size_t *>(guard.get_ptr());
  uint8_t **pinned_decomp_buffers = reinterpret_cast<uint8_t **>(pinned_decomp_sizes + batch_size);
  const uint8_t **pinned_comp_buffers =
    reinterpret_cast<const uint8_t **>(reinterpret_cast<void *>(pinned_decomp_buffers + batch_size));
  nvcompStatus_t *pinned_decomp_statuses =
    reinterpret_cast<nvcompStatus_t *>(reinterpret_cast<void *>(pinned_comp_buffers + batch_size));

  const size_t comp_buf_offset = (bitstream_kind == BitstreamKind::WITH_UNCOMPRESSED_SIZE) ? UNCOMPRESSED_SIZE_OFFSET
                                                                                           : 0;
  CUDA_CHECK(cudaLaunchHostLambda(
    user_stream,
    [batch_size,
     comp_buf_offset,
     host_decomp_sizes,
     decomp_bufs = std::vector<uint8_t *>(decomp_buffers, decomp_buffers + batch_size),
     comp_bufs = std::vector<const uint8_t *>(comp_buffers, comp_buffers + batch_size),
     pinned_decomp_sizes,
     pinned_decomp_buffers,
     pinned_comp_buffers]() {
      std::memcpy(pinned_decomp_sizes, host_decomp_sizes.data(), batch_size * sizeof(size_t));
      std::memcpy(pinned_decomp_buffers, decomp_bufs.data(), batch_size * sizeof(uint8_t *));
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
      device_decomp_sizes,
      pinned_decomp_sizes,
      batch_size * sizeof(size_t),
      cudaMemcpyHostToDevice,
      user_stream
    ));
    CUDA_CHECK(cudaMemcpyAsync(
      device_decomp_buffers,
      pinned_decomp_buffers,
      batch_size * sizeof(uint8_t *),
      cudaMemcpyHostToDevice,
      user_stream
    ));
    CUDA_CHECK(cudaMemcpyAsync(
      device_comp_buffers,
      pinned_comp_buffers,
      batch_size * sizeof(const uint8_t *),
      cudaMemcpyHostToDevice,
      user_stream
    ));
  }

  const uint8_t **kernel_comp_buffers = stage_metadata_on_device ? device_comp_buffers : pinned_comp_buffers;
  size_t *kernel_decomp_sizes = stage_metadata_on_device ? device_decomp_sizes : pinned_decomp_sizes;
  uint8_t **kernel_decomp_buffers = stage_metadata_on_device ? device_decomp_buffers : pinned_decomp_buffers;
  nvcompStatus_t *kernel_decomp_statuses = stage_metadata_on_device ? device_decomp_statuses : pinned_decomp_statuses;

  ManagerBase::check<FnType::Decompress>(decompress_fn(
    reinterpret_cast<const void *const *>(kernel_comp_buffers),
    comp_sizes,
    kernel_decomp_sizes,
    actual_decomp_sizes,
    batch_size,
    free_scratch_buffer,
    free_scratch_size,
    reinterpret_cast<void *const *>(kernel_decomp_buffers),
    kernel_decomp_statuses,
    format_opts,
    decompress_opts,
    user_stream,
    nullptr /*host_comp_chunk_buffers*/
  ));

  if (stage_metadata_on_device)
  {
    // Bring the statuses back so that the propagation lambda below is unchanged.
    CUDA_CHECK(cudaMemcpyAsync(
      pinned_decomp_statuses,
      device_decomp_statuses,
      batch_size * sizeof(nvcompStatus_t),
      cudaMemcpyDeviceToHost,
      user_stream
    ));
  }

  // Propagate the statuses to the configs
  CUDA_CHECK(cudaLaunchHostLambda(
    user_stream,
    [batch_size, decomp_configs, pinned_decomp_statuses]() {
      // Note: even if on the caller side `decomp_configs` goes out of scope,
      //       the lambda holds a copy of the object, and the underlying shared pointers.
      for (size_t idx = 0; idx < batch_size; ++idx)
      {
        *decomp_configs[idx].get_status() = pinned_decomp_statuses[idx];
      }
    },
    false /* force_sync*/
  ));
}

#undef HEAD
#undef TEMPLATE

} // namespace nvcomp
