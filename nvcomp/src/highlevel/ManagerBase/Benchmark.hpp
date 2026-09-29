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
std::vector<uint8_t> HEAD::do_replicate(const std::vector<uint8_t> &input_data, int extra_rep_count)
{
  // This is done in host data
  int total_rep_count = extra_rep_count + 1;
  std::vector<uint8_t> new_data(input_data.size() * total_rep_count);

  // Header
  const size_t header_size = sizeof(CommonHeader) + sizeof(FormatSpecHeader);

  uint8_t *comp_data = new_data.data();
  const uint8_t *orig_comp_data = input_data.data();
  memcpy(comp_data, orig_comp_data, header_size);
  CommonHeader *common_header = reinterpret_cast<CommonHeader *>(comp_data);

  if (common_header->magic_number != MAGIC_NUMBER && common_header->magic_number != PARQUET_HLIF_MAGIC_NUMBER)
  {
    throw NVCompException(nvcompErrorNotSupported, "The given compressed buffer cannot be replicated.");
  }
  else if (common_header->magic_number == MAGIC_NUMBER &&
           common_header->decomp_data_size % common_header->uncomp_chunk_size != 0)
  {
    throw NVCompException(
      nvcompErrorNotSupported,
      "Replication of HLIF buffers only supported with the ordinary magic number "
      "if the uncompressed data size is divisible by the selected uncompressed chunk size."
    );
  }

  common_header->decomp_data_size *= total_rep_count;

  comp_data += roundUpTo(header_size, sizeof(size_t));
  orig_comp_data += roundUpTo(header_size, sizeof(size_t));

  const size_t orig_num_chunks = common_header->num_chunks;
  size_t num_chunks = orig_num_chunks * total_rep_count;
  common_header->num_chunks = num_chunks;

  // Per-chunk fields
  const size_t *orig_comp_chunk_offsets = roundUpToAlignment<size_t>(orig_comp_data);
  const size_t *orig_comp_sizes = orig_comp_chunk_offsets + orig_num_chunks;

  // The memcpy takes the original offsets / sizes from the input data
  size_t *comp_chunk_offsets = roundUpToAlignment<size_t>(comp_data);
  size_t *comp_sizes = comp_chunk_offsets + num_chunks;
  memcpy(comp_chunk_offsets, orig_comp_chunk_offsets, orig_num_chunks * sizeof(size_t));
  memcpy(comp_sizes, orig_comp_sizes, orig_num_chunks * sizeof(size_t));

  size_t *iter_comp_sizes = comp_sizes + orig_num_chunks;
  size_t *iter_comp_offsets = comp_chunk_offsets + orig_num_chunks;
  size_t full_rep_comp_data_size =
    roundUpTo(comp_chunk_offsets[orig_num_chunks - 1] + comp_sizes[orig_num_chunks - 1], sizeof(size_t));
  size_t comp_rep_offset = full_rep_comp_data_size;
  for (int ix_rep = 0; ix_rep < extra_rep_count; ++ix_rep)
  {
    memcpy(iter_comp_sizes, comp_sizes, sizeof(size_t) * orig_num_chunks);
    memcpy(iter_comp_offsets, comp_chunk_offsets, sizeof(size_t) * orig_num_chunks);
    for (int ix_chunk = 0; ix_chunk < orig_num_chunks; ++ix_chunk)
    {
      iter_comp_offsets[ix_chunk] += comp_rep_offset;
    }

    iter_comp_offsets += orig_num_chunks;
    iter_comp_sizes += orig_num_chunks;
    comp_rep_offset += full_rep_comp_data_size;
  }
  common_header->comp_data_size = comp_rep_offset;

  comp_data += num_chunks * 2 * sizeof(size_t);
  orig_comp_data += orig_num_chunks * 2 * sizeof(size_t);

  if (common_header->magic_number == PARQUET_HLIF_MAGIC_NUMBER)
  {
    // HLIF buffers with the PARQUET_HLIF_MAGIC_NUMBER also contain
    // the uncompressed offsets and sizes

    const size_t *orig_provided_uncomp_chunk_offsets = reinterpret_cast<const size_t *>(orig_comp_data);
    const size_t *orig_provided_uncomp_sizes = orig_provided_uncomp_chunk_offsets + orig_num_chunks;

    size_t *provided_uncomp_chunk_offsets = reinterpret_cast<size_t *>(comp_data);
    size_t *provided_uncomp_sizes = provided_uncomp_chunk_offsets + num_chunks;

    memcpy(provided_uncomp_chunk_offsets, orig_provided_uncomp_chunk_offsets, orig_num_chunks * sizeof(size_t));
    memcpy(provided_uncomp_sizes, orig_provided_uncomp_sizes, orig_num_chunks * sizeof(size_t));

    size_t *iter_uncomp_size = provided_uncomp_sizes + orig_num_chunks;
    size_t *iter_uncomp_offsets = provided_uncomp_chunk_offsets + orig_num_chunks;
    size_t iter_uncomp_rep_offset = roundUpTo(
      orig_provided_uncomp_chunk_offsets[orig_num_chunks - 1] + orig_provided_uncomp_sizes[orig_num_chunks - 1],
      sizeof(size_t)
    );
    size_t uncomp_rep_offset = iter_uncomp_rep_offset;

    for (int ix_rep = 0; ix_rep < extra_rep_count; ++ix_rep)
    {
      memcpy(iter_uncomp_size, orig_provided_uncomp_sizes, sizeof(size_t) * orig_num_chunks);
      memcpy(iter_uncomp_offsets, orig_provided_uncomp_chunk_offsets, sizeof(size_t) * orig_num_chunks);
      for (int ix_chunk = 0; ix_chunk < orig_num_chunks; ++ix_chunk)
      {
        iter_uncomp_offsets[ix_chunk] += uncomp_rep_offset;
      }
      iter_uncomp_offsets += orig_num_chunks;
      iter_uncomp_size += orig_num_chunks;
      uncomp_rep_offset += iter_uncomp_rep_offset;
    }

    // Then copy all the compressed chunks
    comp_data += num_chunks * 2 * sizeof(size_t);
    orig_comp_data += orig_num_chunks * 2 * sizeof(size_t);
  }

  for (int ix_rep = 0; ix_rep < total_rep_count; ++ix_rep)
  {
    memcpy(comp_data, orig_comp_data, full_rep_comp_data_size);
    comp_data += full_rep_comp_data_size;
  }

  return new_data;
}
TEMPLATE
void HEAD::recompress(
  const uint8_t *decomp_buffer,
  const uint8_t *comp_buffer,
  const DecompressionConfig &decomp_config,
  size_t &recompress_size,
  float &recompress_throughput,
  float &recompress_ratio
)
{
  // We've already decompressed the comp buffer to the decomp buffer
  // Now we'll set up some llif buffers to time recompression with the manager's format
  // We'll set this up so that the llif buffers correspond 1:1 to the chunks in the original compressed buffer
  // We can also use this compare GPU ratios against CPU ratios (if the original data was compressed by CPU)

  const size_t num_chunks = decomp_config.num_chunks;
  uint8_t *recomp_buffer;

  std::vector<size_t> cpu_uncomp_sizes(num_chunks);
  std::vector<size_t> cpu_uncomp_offsets(num_chunks);

  comp_buffer += roundUpTo(sizeof(CommonHeader) + sizeof(FormatSpecHeader), sizeof(size_t));
  comp_buffer += 2 * num_chunks * sizeof(size_t); // comp chunk offsets and comp sizes

  // We need a header with PARQUET_HLIF_MAGIC_NUMBER
  // so that we can retrieve the uncompressed offsets & sizes.
  const size_t *provided_uncomp_chunk_offsets = reinterpret_cast<const size_t *>(comp_buffer);
  const size_t *provided_uncomp_sizes = provided_uncomp_chunk_offsets + num_chunks;
  CUDA_CHECK(cudaMemcpy(
    cpu_uncomp_offsets.data(),
    provided_uncomp_chunk_offsets,
    num_chunks * sizeof(size_t),
    cudaMemcpyDeviceToHost
  ));
  CUDA_CHECK(
    cudaMemcpy(cpu_uncomp_sizes.data(), provided_uncomp_sizes, num_chunks * sizeof(size_t), cudaMemcpyDeviceToHost)
  );

  size_t total_max_comp_size = 0;
  size_t total_uncomp_size = 0;

  std::vector<size_t> cpu_comp_buffer_offsets(num_chunks);
  max_comp_chunk_size = 0;
  for (size_t ix_chunk = 0; ix_chunk < num_chunks; ++ix_chunk)
  {
    cpu_comp_buffer_offsets[ix_chunk] = total_max_comp_size;
    size_t this_max_comp_chunk_size = 0;
    ManagerBase::check<FnType::MaxCompChunkSize>(
      max_comp_size_fn(cpu_uncomp_sizes[ix_chunk], format_opts, &this_max_comp_chunk_size)
    );
    total_uncomp_size += cpu_uncomp_sizes[ix_chunk];
    this_max_comp_chunk_size = roundUpTo(this_max_comp_chunk_size, min_alignment);
    total_max_comp_size += this_max_comp_chunk_size;
    max_comp_chunk_size = max(max_comp_chunk_size, this_max_comp_chunk_size);
  }

  // Note:
  // We are only interested in the statistics of the data, and hence no compaction is necessary
  allocate_gpu_scratch(compute_compress_scratch_buffer_size(num_chunks, false));

  // We'll use uncomp_chunk_sizes and uncomp_chunk_offsets from the header
  CUDA_CHECK(cudaMalloc(&recomp_buffer, total_max_comp_size));

  // Scratch memory layout
  // Note: max_comp_chunk_size was rounded up to min_alignment
  // [ Uncompressed buffer pointers (num_chunks, uint8_t*) ]
  // [ Compressed buffer pointers (num_chunks, uint8_t*) ]
  // [ Compressed sizes (num_chunks, size_t) ]
  // [ Compression scratch buffer (bytes) ]
  uint8_t *free_scratch_buffer = scratch_buffer;

  const uint8_t **uncomp_buffers = roundUpToAlignment<const uint8_t *>(free_scratch_buffer);
  free_scratch_buffer = reinterpret_cast<uint8_t *>(uncomp_buffers + num_chunks);

  uint8_t **comp_buffers = reinterpret_cast<uint8_t **>(free_scratch_buffer);
  free_scratch_buffer += num_chunks * sizeof(size_t);

  size_t *comp_sizes = reinterpret_cast<size_t *>(free_scratch_buffer);
  free_scratch_buffer += num_chunks * sizeof(size_t);
  free_scratch_buffer =
    reinterpret_cast<uint8_t *>(roundUpTo(reinterpret_cast<uintptr_t>(free_scratch_buffer), min_alignment));

  // Note:
  // free_scratch_buffer needs to be aligned to min_alignment to accomodate
  // the worst-case compression scratch alignment requirement.
  assert(reinterpret_cast<uintptr_t>(free_scratch_buffer) % min_alignment == 0);

  const size_t free_scratch_size = scratch_buffer_size - (uintptr_t(free_scratch_buffer) - uintptr_t(scratch_buffer));
  assert(
    free_scratch_size >= compute_lowlevel_compress_scratch_size(
                           num_chunks,
                           *std::max_element(cpu_uncomp_sizes.begin(), cpu_uncomp_sizes.end())
                         )
  );

  std::vector<uint8_t *> comp_results(num_chunks);
  std::vector<const uint8_t *> decomp_inputs(num_chunks);
  for (size_t ix_chunk = 0; ix_chunk < num_chunks; ++ix_chunk)
  {
    comp_results[ix_chunk] = recomp_buffer + cpu_comp_buffer_offsets[ix_chunk];
    decomp_inputs[ix_chunk] = decomp_buffer + cpu_uncomp_offsets[ix_chunk];
  }

  CUDA_CHECK(cudaMemcpy(uncomp_buffers, decomp_inputs.data(), sizeof(size_t) * num_chunks, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(comp_buffers, comp_results.data(), sizeof(size_t) * num_chunks, cudaMemcpyHostToDevice));

  cudaEvent_t start, end;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));

  CUDA_CHECK(cudaStreamSynchronize(user_stream));
  CUDA_CHECK(cudaEventRecord(start, user_stream));

  ManagerBase::check<FnType::Compress>(compress_fn(
    reinterpret_cast<const void *const *>(uncomp_buffers),
    provided_uncomp_sizes,
    max_comp_chunk_size,
    num_chunks,
    free_scratch_buffer,
    free_scratch_size,
    reinterpret_cast<void *const *>(comp_buffers),
    comp_sizes,
    format_opts,
    nullptr,
    user_stream
  ));

  // Note:
  // the .recompress() API is internal, and has no CompressionConfig argument as of today where
  // we could propagate the reduced compression device status.

  CUDA_CHECK(cudaEventRecord(end, user_stream));
  CUDA_CHECK(cudaStreamSynchronize(user_stream));
  float compress_ms;
  CUDA_CHECK(cudaEventElapsedTime(&compress_ms, start, end));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(end));

  std::vector<size_t> cpu_comp_sizes(num_chunks);
  CUDA_CHECK(cudaMemcpy(cpu_comp_sizes.data(), comp_sizes, num_chunks * sizeof(size_t), cudaMemcpyDeviceToHost));

  // Output metrics
  recompress_size = std::accumulate(cpu_comp_sizes.begin(), cpu_comp_sizes.end(), size_t(0));
  if (recompress_size == 0)
  {
    throw NVCompException(nvcompErrorInvalidValue, "Compressed size summed up to zero. Something must have gone wrong.");
  }
  recompress_ratio = static_cast<float>(decomp_config.decomp_data_size) / recompress_size;
  recompress_throughput = static_cast<float>(decomp_config.decomp_data_size) / (compress_ms * 1e-3f) / 1e9f;

  CUDA_CHECK(cudaFree(recomp_buffer));
}

#undef HEAD
#undef TEMPLATE

} // namespace nvcomp
