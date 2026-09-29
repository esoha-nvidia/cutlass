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
size_t HEAD::compute_lowlevel_compress_scratch_size(const size_t batch_size, const size_t max_uncomp_chunk_size)
{
  size_t compress_scratch_req;
  ManagerBase::check<FnType::CompressScratch>(comp_scratch_size_fn(
    batch_size,
    max_uncomp_chunk_size,
    format_opts,
    &compress_scratch_req,
    batch_size * max_uncomp_chunk_size
  ));
  return compress_scratch_req;
}

TEMPLATE
size_t HEAD::compute_compress_scratch_buffer_size(const size_t batch_size, const bool with_compaction)
{
  // Get maximum scratch requirement for decompression / compression
  size_t compress_scratch_req = compute_lowlevel_compress_scratch_size(batch_size, uncomp_chunk_size);

  // Need to allow at least the scratch required for a cub inclusive sum
  size_t cub_scratch_req = 0;
  cub_exclusive_sum_scratch_compute_size_t(cub_scratch_req, batch_size, user_stream);

  compress_scratch_req = std::max(compress_scratch_req, cub_scratch_req);

  if (with_compaction)
  {
    // (batch size - 1) * max_comp_chunk_size allows us to compress to a temporary
    // buffer before compacting (for all but the first chunk)
    compress_scratch_req += (batch_size - 1) * max_comp_chunk_size;
  }

  // 3 * sizeof(size_t) represents the API requirements for the HLIF call
  compress_scratch_req += batch_size * 3 * sizeof(size_t);

  // To store device statuses
  compress_scratch_req += roundUpTo(batch_size * sizeof(nvcompStatus_t), min_alignment);

  // Alignment bytes
  compress_scratch_req += min_alignment - 1;

  return CHECKSUM_TEMP_SIZE + compress_scratch_req;
}
TEMPLATE
size_t HEAD::compute_batched_compress_scratch_buffer_size(
  const std::vector<CompressionConfig> &comp_configs,
  size_t total_num_chunks
)
{
  size_t batch_count = comp_configs.size();

  size_t sum_compress_scratch_req = 0;
  size_t total_compress_scratch_req = compute_lowlevel_compress_scratch_size(total_num_chunks, uncomp_chunk_size);
  size_t total_cub_scan_scratch_req = 0;

  cub_exclusive_sum_scratch_compute_size_t(total_cub_scan_scratch_req, total_num_chunks, user_stream);

  sum_compress_scratch_req += std::max(
    total_compress_scratch_req,
    total_cub_scan_scratch_req
  ); //Scratch space required for compression/Exclusive Sum
  sum_compress_scratch_req +=
    (total_num_chunks - batch_count) *
    max_comp_chunk_size; //Scratch space required to store compressed buffer temporarily before compaction
  sum_compress_scratch_req += total_num_chunks * 5 * sizeof(size_t); //Scratch space required for HLIF bookkeeping
  sum_compress_scratch_req += roundUpTo(
    total_num_chunks * sizeof(nvcompStatus_t),
    min_alignment
  ); // For storing device statuses for all chunks

  // Alignment bytes: need to cover the min_alignment round-up for
  // scratch_comp_buffer.
  sum_compress_scratch_req += min_alignment - 1;

  return sum_compress_scratch_req;
}

TEMPLATE
size_t HEAD::compute_lowlevel_decompress_scratch_size(const size_t batch_size, const size_t max_uncomp_chunk_size)
{
  size_t decompress_scratch_req;
  ManagerBase::check<FnType::DecompressScratch>(decomp_scratch_size_fn(
    batch_size,
    max_uncomp_chunk_size,
    decompress_opts,
    &decompress_scratch_req,
    batch_size * max_uncomp_chunk_size
  ));
  return decompress_scratch_req;
}

TEMPLATE
size_t HEAD::compute_decompress_scratch_buffer_size(const size_t batch_size)
{
  size_t decomp_scratch_req = compute_lowlevel_decompress_scratch_size(batch_size, uncomp_chunk_size);

  // 4 * sizeof(size_t) represents the API requirements for the HLIF call
  decomp_scratch_req += batch_size * (4 * sizeof(size_t) + sizeof(nvcompStatus_t));

  // Alignment bytes
  decomp_scratch_req += 2 * (min_alignment - 1);

  return CHECKSUM_TEMP_SIZE + decomp_scratch_req;
}
TEMPLATE
void HEAD::allocate_gpu_scratch(size_t req_scratch_memory)
{
  if (scratch_buffer_size < req_scratch_memory)
  {
    if (scratch_buffer_size > 0)
    {
      deallocator(scratch_buffer, scratch_buffer_size);
    }
    scratch_buffer = reinterpret_cast<uint8_t *>(allocator(req_scratch_memory));
    scratch_buffer_size = req_scratch_memory;
  }
}
TEMPLATE
void HEAD::deallocate_gpu_scratch()
{
  if (scratch_buffer_size > 0)
  {
    deallocator(scratch_buffer, scratch_buffer_size);
    scratch_buffer_size = 0;
  }
}
TEMPLATE
void HEAD::allocate_host_scratch()
{
  if (common_header_cpu)
  {
    return;
  }

  auto pinned_allocator = get_pinned_memory_resource();
  void *pinned_memory_ptr = pinned_allocator.allocate(
    user_stream,
    sizeof(CommonHeader) + alignof(nvcompStatus_t) - 1 + sizeof(nvcompStatus_t),
    alignof(CommonHeader)
  );
  common_header_cpu = static_cast<CommonHeader *>(pinned_memory_ptr);
  common_status_cpu = roundUpToAlignment<nvcompStatus_t>(common_header_cpu + 1);
}
TEMPLATE
void HEAD::deallocate_host_scratch()
{
  if (common_header_cpu)
  {
    auto pinned_allocator = get_pinned_memory_resource();
    pinned_allocator.deallocate(
      user_stream,
      common_header_cpu,
      sizeof(CommonHeader) + alignof(nvcompStatus_t) - 1 + sizeof(nvcompStatus_t),
      alignof(CommonHeader)
    );
    common_header_cpu = nullptr;
  }
}
TEMPLATE
void HEAD::deallocate_gpu_mem()
{
  // Note:
  // Although this is just pinned host space,
  // the `user_stream` needs to be available upon destruction,
  // hence we are also deallocating this.
  deallocate_host_scratch();
  deallocate_gpu_scratch();
}

#undef HEAD
#undef TEMPLATE

} // namespace nvcomp
