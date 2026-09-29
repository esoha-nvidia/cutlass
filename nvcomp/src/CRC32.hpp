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

#pragma once

#include "nvcomp/nvcompManager.hpp"
#include "nvcomp/shared_types.h"
#include "nvcomp_common_deps/hlif_shared_types.hpp"

// Guard agains unaligned scratch buffers
static constexpr size_t CHECKSUM_TEMP_SIZE = 2 * sizeof(uint32_t) + alignof(uint32_t) - 1;

namespace nvcomp
{

void compute_uncomp_chunk_checksums(
  size_t batch_size,
  const void *const *device_uncompressed_ptrs,
  const size_t *device_uncompressed_bytes,
  uint32_t *device_CRC32_ptrs,
  cudaStream_t stream = 0
);

std::vector<Checksum_t>
compute_uncomp_chunk_checksums(const std::vector<std::vector<uint8_t>> &uncompressed_chunks, cudaStream_t stream = 0);

void store_decomp_chunk_checksums(
  uint32_t num_chunks,
  size_t decomp_chunk_size,
  size_t decomp_buffer_size,
  const uint8_t *decomp_buffer,
  uint32_t *decomp_chunk_checksums,
  CommonHeader *common_header,
  cudaStream_t stream = 0
);

void store_comp_chunk_checksums(
  uint32_t num_chunks,
  const size_t *comp_chunk_sizes,
  const size_t *comp_chunk_offsets,
  const uint8_t *comp_buffer,
  uint32_t *comp_chunk_checksums,
  CommonHeader *common_header,
  cudaStream_t stream = 0
);

void verify_decomp_chunk_checksums(
  uint32_t num_chunks,
  size_t decomp_chunk_size,
  size_t decomp_buffer_size,
  const uint8_t *decomp_buffer,
  const uint32_t *decomp_chunk_checksums,
  nvcompStatus_t *status,
  cudaStream_t stream = 0
);

void verify_comp_chunk_checksums(
  uint32_t num_chunks,
  const size_t *comp_chunk_sizes,
  const size_t *comp_chunk_offsets,
  const uint8_t *comp_buffer,
  const uint32_t *comp_chunk_checksums,
  nvcompStatus_t *status,
  cudaStream_t stream = 0
);

void store_single_checksum(
  const uint8_t *file,
  size_t file_size,
  const size_t *file_size_ptr,
  CommonHeader *common_header,
  uint32_t *crc_dst,
  bool is_chunk_level_checksum,
  cudaStream_t stream = 0
);

void verify_single_checksum(
  const uint8_t *file,
  size_t file_size,
  const size_t *file_size_ptr,
  uint8_t *scratch_buffer,
  nvcompStatus_t *status,
  const uint32_t *restricted_crc_address,
  bool is_chunk_level_checksum,
  cudaStream_t stream = 0
);

void verify_all_checksums(
  const size_t *comp_chunk_offsets,
  const size_t *comp_chunk_sizes,
  const uint8_t *comp_data_buffer,
  const uint8_t *decomp_buffer,
  size_t uncomp_chunk_size,
  const uint32_t *comp_chunk_checksums,
  const uint32_t *decomp_chunk_checksums,
  uint8_t *scratch_buffer,
  const CommonHeader *common_header,
  const DecompressionConfig &config,
  nvcompStatus_t *status,
  cudaStream_t stream
);

void store_all_checksums(
  const size_t *comp_chunk_offsets,
  const size_t *comp_chunk_sizes,
  const uint8_t *comp_data_buffer,
  const uint8_t *decomp_buffer,
  size_t uncomp_chunk_size,
  uint32_t *comp_chunk_checksums,
  uint32_t *decomp_chunk_checksums,
  uint8_t *scratch_buffer,
  CommonHeader *common_header,
  const CompressionConfig &config,
  cudaStream_t stream
);

void cuCRC32_permuted_test(unsigned int n_msg, uint32_t *crc_dst, uint32_t chunk_size, uint8_t *buf);

} // namespace nvcomp
