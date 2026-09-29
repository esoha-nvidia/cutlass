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
size_t HEAD::get_compressed_output_size(const uint8_t *comp_buffer)
{
  return get_compressed_output_size(&comp_buffer, 1)[0];
}
TEMPLATE
std::vector<size_t> HEAD::get_compressed_output_size(const uint8_t *const *comp_buffers, size_t batch_size)
{
  if (bitstream_kind != BitstreamKind::NVCOMP_NATIVE)
  {
    throw NVCompException(
      nvcompErrorNotSupported,
      "get_compressed_output_size can only be called if bitstream kind is NVCOMP_NATIVE"
    );
  }

  std::vector<size_t> output_sizes;
  output_sizes.reserve(batch_size);

  // Check if the compressed buffers are accessible on the host
  // Currently we are only checking if the array of pointers is accessible on the host, and also the very first pointer.
  // The rationale behind the first pointer in the array is that it is unlikely the user would provide a mixed array.
  bool host_setup_mode = CudaUtils::is_host_pointer(comp_buffers) && CudaUtils::is_host_pointer(comp_buffers[0]);

  if (host_setup_mode)
  {
    // Sync to ensure stream ordered access of the compressed buffers
    CUDA_CHECK(cudaStreamSynchronize(user_stream));
    for (size_t idx = 0; idx < batch_size; ++idx)
    {
      const CommonHeader *common_header = reinterpret_cast<const CommonHeader *>(comp_buffers[idx]);
      output_sizes.push_back(common_header->comp_data_size + common_header->comp_data_offset);
    }
  }
  else
  {
    allocate_host_scratch();
    for (size_t idx = 0; idx < batch_size; ++idx)
    {
      CUDA_CHECK(
        cudaMemcpyAsync(common_header_cpu, comp_buffers[idx], sizeof(CommonHeader), cudaMemcpyDefault, user_stream)
      );
      CUDA_CHECK(cudaStreamSynchronize(user_stream));
      output_sizes.push_back(common_header_cpu->comp_data_size + common_header_cpu->comp_data_offset);
    }
  }

  return output_sizes;
}
TEMPLATE
size_t HEAD::get_decompressed_output_size(const uint8_t *comp_buffer)
{
  return get_decompressed_output_size(&comp_buffer, 1)[0];
}
TEMPLATE
std::vector<size_t> HEAD::get_decompressed_output_size(const uint8_t *const *comp_buffers, size_t batch_size)
{
  if (bitstream_kind != BitstreamKind::NVCOMP_NATIVE)
  {
    throw NVCompException(
      nvcompErrorNotSupported,
      "get_decompressed_output_size can only be called if bitstream kind is NVCOMP_NATIVE"
    );
  }

  std::vector<size_t> decompressed_output_sizes;
  decompressed_output_sizes.reserve(batch_size);

  // Check if the compressed buffers are accessible on the host
  // Currently we are only checking if the array of pointers is accessible on the host, and also the very first pointer.
  // The rationale behind the first pointer in the array is that it is unlikely the user would provide a mixed array.
  bool host_setup_mode = CudaUtils::is_host_pointer(comp_buffers) && CudaUtils::is_host_pointer(comp_buffers[0]);

  if (host_setup_mode)
  {
    // Sync to ensure stream ordered access of the compressed buffers
    CUDA_CHECK(cudaStreamSynchronize(user_stream));
    for (size_t idx = 0; idx < batch_size; ++idx)
    {
      const CommonHeader *common_header = reinterpret_cast<const CommonHeader *>(comp_buffers[idx]);
      decompressed_output_sizes.push_back(common_header->decomp_data_size);
    }
  }
  else
  {
    allocate_host_scratch();
    for (size_t idx = 0; idx < batch_size; ++idx)
    {
      CUDA_CHECK(
        cudaMemcpyAsync(common_header_cpu, comp_buffers[idx], sizeof(CommonHeader), cudaMemcpyDefault, user_stream)
      );
      CUDA_CHECK(cudaStreamSynchronize(user_stream));
      decompressed_output_sizes.push_back(common_header_cpu->decomp_data_size);
    }
  }

  return decompressed_output_sizes;
}
TEMPLATE
CompressionConfig HEAD::configure_compression(const size_t uncomp_buffer_size)
{
  if (bitstream_kind != BitstreamKind::NVCOMP_NATIVE)
  {
    return configure_compression(std::vector{uncomp_buffer_size})[0];
  }

  CompressionConfig comp_config{uncomp_buffer_size};

  comp_config.num_chunks = roundUpDiv(uncomp_buffer_size, uncomp_chunk_size);
  comp_config.compute_checksums = (checksum_policy == ComputeAndNoVerify) || (checksum_policy == ComputeAndVerify) ||
                                  (checksum_policy == ComputeAndVerifyIfPresent);

  size_t max_comp_buff_size = 0;
  if (comp_config.num_chunks > 1)
  {
    max_comp_buff_size = comp_config.num_chunks * max_comp_chunk_size;
  }
  else
  {
    ManagerBase::check<FnType::MaxCompChunkSize>(max_comp_size_fn(uncomp_buffer_size, format_opts, &max_comp_buff_size));
    max_comp_buff_size = roundUpTo(max_comp_buff_size, sizeof(size_t));
  }

  size_t offsets_size = comp_config.num_chunks * sizeof(size_t);
  size_t sizes_size = comp_config.num_chunks * sizeof(size_t);
  size_t checksums_size = comp_config.compute_checksums ? comp_config.num_chunks * 2 * sizeof(uint32_t) : 0;
  comp_config.max_compressed_buffer_size =
    max_comp_buff_size + roundUpTo(
                           offsets_size + sizes_size + checksums_size + sizeof(CommonHeader) + sizeof(FormatSpecHeader),
                           min_alignment
                         );

  return comp_config;
}
TEMPLATE
std::vector<CompressionConfig> HEAD::configure_compression(const std::vector<size_t> &uncomp_buffer_sizes)
{
  std::vector<CompressionConfig> comp_configs;
  comp_configs.reserve(uncomp_buffer_sizes.size());

  if (bitstream_kind == BitstreamKind::NVCOMP_NATIVE)
  {
    for (const auto &buffer_size : uncomp_buffer_sizes)
    {
      comp_configs.push_back(configure_compression(buffer_size));
    }

    return comp_configs;
  }

  size_t max_size = *std::max_element(uncomp_buffer_sizes.begin(), uncomp_buffer_sizes.end());
  size_t max_compressed_size;
  ManagerBase::check<FnType::MaxCompChunkSize>(max_comp_size_fn(max_size, format_opts, &max_compressed_size));

  if (bitstream_kind == BitstreamKind::WITH_UNCOMPRESSED_SIZE)
  {
    max_compressed_size += UNCOMPRESSED_SIZE_OFFSET;
    assert(UNCOMPRESSED_SIZE_OFFSET % min_alignment == 0);
  }

  for (const auto &buffer_size : uncomp_buffer_sizes)
  {
    comp_configs.emplace_back(buffer_size);
    comp_configs.back().num_chunks = 1;
    comp_configs.back().compute_checksums = false;
    comp_configs.back().max_compressed_buffer_size = max_compressed_size;
  }

  return comp_configs;
}
TEMPLATE
DecompressionConfig HEAD::extract_decomp_config(const CommonHeader *common_header)
{
  DecompressionConfig decomp_config;

  validate_hlif_magic_number(common_header->magic_number);
  if (common_header->format != format_type)
  {
    throw NVCompException(nvcompErrorCannotDecompress, "The passed data was not compressed with requested codec");
  }

  decomp_config.decomp_data_size = common_header->decomp_data_size;
  decomp_config.num_chunks = common_header->num_chunks;

  if (!common_header->include_per_chunk_comp_buffer_checksums ||
      !common_header->include_per_chunk_decomp_buffer_checksums)
  {
    if (checksum_policy == ComputeAndVerify)
    {
      throw NVCompException(
        nvcompErrorCannotVerifyChecksums,
        "Cannot verify chunk checksums - not computed during compression phase.\
        Consider setting the checksum policy to VerifyIfIncluded.\n"
      );
    }
    decomp_config.checksums_present = false;
  }
  else
  {
    decomp_config.checksums_present = true;
  }

  bool include_uncomp_offsets_and_sizes = common_header->magic_number == PARQUET_HLIF_MAGIC_NUMBER;
  decomp_config.impl->uncompressed_sizes_and_offsets_provided = include_uncomp_offsets_and_sizes;

  return decomp_config;
}
TEMPLATE
DecompressionConfig HEAD::configure_decompression(const uint8_t *comp_buffer, const size_t *comp_size)
{
  if (bitstream_kind != BitstreamKind::NVCOMP_NATIVE)
  {
    return configure_decompression(&comp_buffer, 1, comp_size)[0];
  }
  const CommonHeader *common_header = reinterpret_cast<const CommonHeader *>(comp_buffer);

  // Allocate the pinned host scratch (if necessary)
  allocate_host_scratch();

  CUDA_CHECK(cudaMemcpyAsync(common_header_cpu, common_header, sizeof(CommonHeader), cudaMemcpyDefault, user_stream));
  CUDA_CHECK(cudaStreamSynchronize(user_stream));
  return extract_decomp_config(common_header_cpu);
}
TEMPLATE
std::vector<DecompressionConfig>
HEAD::configure_decompression(const uint8_t *const *comp_buffers, size_t batch_size, const size_t *comp_sizes)
{
  if (bitstream_kind == BitstreamKind::NVCOMP_NATIVE)
  {
    std::vector<DecompressionConfig> decomp_configs;
    decomp_configs.reserve(batch_size);

    for (size_t idx = 0; idx < batch_size; ++idx)
    {
      decomp_configs.push_back(configure_decompression(comp_buffers[idx]));
    }

    return decomp_configs;
  }

  std::vector<DecompressionConfig> decomp_configs(batch_size);
  for (size_t idx = 0; idx < batch_size; ++idx)
  {
    decomp_configs[idx].checksums_present = false;
    decomp_configs[idx].num_chunks = 1;
  }

  if (bitstream_kind == BitstreamKind::RAW)
  {
    if (comp_sizes == nullptr)
    {
      throw NVCompException(nvcompErrorNotSupported, "comp_sizes must be provided when using RAW bitstream kind");
    }

    // Pinned memory layout
    // [ Uncompressed chunk sizes (batch_size, size_t) ]
    // [ Compressed buffer pointers (batch_size, uint8_t*) ]
    size_t pinned_mem_size = batch_size * sizeof(size_t) + batch_size * sizeof(const uint8_t *);
    HostPinnedGuard guard(pinned_mem_size, alignof(size_t), user_stream);

    size_t *pinned_uncomp_chunk_sizes = static_cast<size_t *>(guard.get_ptr());
    const uint8_t **pinned_comp_buffers = reinterpret_cast<const uint8_t **>(pinned_uncomp_chunk_sizes + batch_size);

    bool force_sync = true; // Always sync because this API already syncs the stream.
    CUDA_CHECK(cudaLaunchHostLambda(
      user_stream,
      [batch_size,
       comp_bufs = std::vector<const uint8_t *>(comp_buffers, comp_buffers + batch_size),
       pinned_comp_buffers]() {
        std::memcpy(pinned_comp_buffers, comp_bufs.data(), batch_size * sizeof(const uint8_t *));
      },
      force_sync
    ));

    ManagerBase::check<FnType::DecompressSize>(decomp_size_fn(
      reinterpret_cast<const void *const *>(pinned_comp_buffers),
      comp_sizes,
      pinned_uncomp_chunk_sizes,
      batch_size,
      user_stream
    ));

    CUDA_CHECK(cudaStreamSynchronize(user_stream));

    for (size_t idx = 0; idx < batch_size; ++idx)
    {
      decomp_configs[idx].decomp_data_size = pinned_uncomp_chunk_sizes[idx];
    }
  }
  else
  { // BitstreamKind::WITH_UNCOMPRESSED_SIZE
    for (size_t idx = 0; idx < batch_size; ++idx)
    {
      CUDA_CHECK(cudaMemcpyAsync(
        &decomp_configs[idx].decomp_data_size,
        comp_buffers[idx],
        sizeof(size_t),
        cudaMemcpyDeviceToHost,
        user_stream
      ));
    }
    CUDA_CHECK(cudaStreamSynchronize(user_stream));
  }

  return decomp_configs;
}
TEMPLATE
DecompressionConfig HEAD::configure_decompression(const CompressionConfig &comp_config)
{
  DecompressionConfig decomp_config;

  decomp_config.decomp_data_size = comp_config.uncompressed_buffer_size;
  decomp_config.num_chunks = comp_config.num_chunks;
  decomp_config.checksums_present = comp_config.compute_checksums;

  return decomp_config;
}
TEMPLATE
std::vector<DecompressionConfig> HEAD::configure_decompression(const std::vector<CompressionConfig> &comp_configs)
{
  std::vector<DecompressionConfig> decomp_configs;
  decomp_configs.reserve(comp_configs.size());

  for (const auto &comp_config : comp_configs)
  {
    decomp_configs.push_back(configure_decompression(comp_config));
  }

  return decomp_configs;
}
TEMPLATE
void HEAD::set_scratch_allocators(const AllocFn_t &alloc_fn, const DeAllocFn_t &dealloc_fn)
{
  // Note: the pinned host scratch is not affected by custom allocators.
  deallocate_gpu_scratch();
  allocator = alloc_fn;
  deallocator = dealloc_fn;
}

#undef HEAD
#undef TEMPLATE

} // namespace nvcomp
