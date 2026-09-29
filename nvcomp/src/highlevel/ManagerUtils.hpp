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

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "highlevel/CompressionConfigs.hpp"
#include "nvcomp/shared_types.h"
#include "nvcomp_common_deps/hlif_shared_types.hpp"

namespace nvcomp
{

__global__ void setup_decomp_llif_buffers(
  const CommonHeader *common_header,
  uint8_t *decomp_buffer,
  const uint8_t *comp_buffer,
  uint8_t **uncomp_buffers,
  const uint8_t **comp_buffers,
  const size_t *comp_offsets,
  size_t *uncomp_sizes,
  const size_t *provided_uncomp_chunk_offsets,
  const size_t *provided_uncomp_sizes,
  const bool include_uncomp_offsets_and_sizes
);

__global__ void setup_batched_decomp_llif_buffers(
  size_t *batch_chunks_exclusive_sum,
  uint8_t **device_input_comp_buffers,
  uint8_t **device_output_decomp_buffers,
  int header_size,
  uint8_t **uncomp_chunk_buffers,
  const uint8_t **comp_chunk_buffers,
  size_t *uncomp_sizes,
  size_t *comp_chunk_sizes,
  size_t min_alignment
);

void setup_batched_decomp_llif_buffers_host(
  uint8_t **device_input_comp_buffers,
  const uint8_t *const *host_input_comp_buffers,
  uint8_t **output_decomp_buffers,
  int header_size,
  uint8_t **uncomp_chunk_buffers,
  const uint8_t **comp_chunk_buffers,
  const uint8_t **host_comp_chunk_buffers,
  size_t *uncomp_sizes,
  size_t *comp_chunk_sizes,
  size_t batch_count,
  size_t min_alignment
);

__global__ void compact_comp_buffers_and_header_output(
  uint8_t *__restrict__ comp_buffer,
  const uint8_t *const *__restrict__ comp_buffers,
  const size_t *__restrict__ comp_sizes,
  CommonHeader *__restrict__ header,
  size_t *__restrict__ chunk_offset_buffer,
  const size_t num_chunks,
  const size_t decomp_buffer_size,
  const size_t uncomp_chunk_size,
  const nvcompFormatType_t comp_format,
  size_t *__restrict__ comp_size,
  size_t alignment
);

__global__ void batched_compact_comp_buffers_and_header_output(
  size_t batch_count,
  size_t total_num_chunks,
  uint8_t **output_comp_buffers,
  CompressionConfig *compression_configs,
  const uint8_t *const *comp_chunk_buffers,
  const size_t *comp_sizes,
  size_t *chunk_offset_buffer,
  const size_t uncomp_chunk_size,
  const nvcompFormatType_t comp_format,
  size_t *comp_size,
  int header_size,
  size_t min_alignment
);

__global__ void
round_up_alignment_kernel(const size_t *in_array, size_t *out_array, const size_t n_elems, const size_t alignment);

void cub_exclusive_sum_scratch_compute_size_t(size_t &scratch_buffer_req, const size_t num_items, cudaStream_t stream);

void cub_exclusive_sum(
  const size_t *input,
  size_t *output,
  size_t num_items,
  uint8_t *scratch_buffer,
  const size_t scratch_buffer_size,
  cudaStream_t stream
);

void increase_array_by(size_t *input, size_t num_elements, size_t increment, cudaStream_t stream);

void decrease_array_by(size_t *input, size_t num_elements, size_t decrement, cudaStream_t stream);

void max_reduce_device_status_single(
  nvcompStatus_t *d_statuses,
  nvcompStatus_t *h_status,
  const size_t num_chunks,
  cudaStream_t stream
);

void max_reduce_device_status_batched(
  nvcompStatus_t *d_statuses,
  nvcompStatus_t *h_statuses,
  const size_t *num_chunk_offsets,
  const size_t total_num_chunks,
  const size_t max_num_chunks,
  const size_t batch_count,
  cudaStream_t stream
);

void max_reduce_host_status_batched(
  nvcompStatus_t *pinned_statuses,
  const std::vector<DecompressionConfig> &decomp_configs,
  size_t batch_count
);

} // namespace nvcomp
