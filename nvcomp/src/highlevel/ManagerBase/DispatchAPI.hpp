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
void HEAD::compress(
  const uint8_t *uncomp_buffer,
  uint8_t *comp_buffer,
  const CompressionConfig &comp_config,
  size_t *comp_size
)
{
  ManagerBase::check_buffer_alignment(uncomp_buffer, compress_alignment.input, "Compression input");
  ManagerBase::check_buffer_alignment(comp_buffer, compress_alignment.output, "Compression output");

  if (bitstream_kind == BitstreamKind::NVCOMP_NATIVE)
  {
    compress_chunked_single(uncomp_buffer, comp_buffer, comp_config, comp_size);
  }
  else
  {
    if (comp_size == nullptr)
    {
      throw NVCompException(nvcompErrorNotSupported, "comp_size can only be nullptr if bitstream kind is NVCOMP_NATIVE");
    }
    compress_raw_batched(&uncomp_buffer, &comp_buffer, {comp_config}, comp_size);
  }
}
TEMPLATE
void HEAD::compress(
  const uint8_t *const *uncomp_buffers,
  uint8_t *const *comp_buffers,
  const std::vector<CompressionConfig> &comp_configs,
  size_t *comp_sizes
)
{
  size_t batch_count = comp_configs.size();

  for (size_t idx = 0; idx < batch_count; ++idx)
  {
    ManagerBase::check_buffer_alignment(uncomp_buffers[idx], compress_alignment.input, "Compression input");
    ManagerBase::check_buffer_alignment(comp_buffers[idx], compress_alignment.output, "Compression output");
  }

  // Check if the compression configuration is the same across all batches
  bool checksum_enabled = false;
  for (size_t idx = 0; idx < batch_count; ++idx)
  {
    if (comp_configs[idx].compute_checksums)
    {
      checksum_enabled = true;
      break;
    }
  }

  if (bitstream_kind == BitstreamKind::NVCOMP_NATIVE)
  {
    if (checksum_enabled)
    {
      for (size_t idx = 0; idx < batch_count; ++idx)
      {
        compress_chunked_single(
          uncomp_buffers[idx],
          comp_buffers[idx],
          comp_configs[idx],
          comp_sizes == nullptr ? nullptr : comp_sizes + idx
        );
      }
    }
    else
    {
      compress_chunked_batched(uncomp_buffers, comp_buffers, comp_configs, comp_sizes);
    }
  }
  else
  {
    if (comp_sizes == nullptr)
    {
      throw NVCompException(
        nvcompErrorNotSupported,
        "comp_sizes can only be nullptr if bitstream kind is NVCOMP_NATIVE"
      );
    }
    compress_raw_batched(uncomp_buffers, comp_buffers, comp_configs, comp_sizes);
  }
}
TEMPLATE
void HEAD::decompress(
  uint8_t *decomp_buffer,
  const uint8_t *comp_buffer,
  const DecompressionConfig &decomp_config,
  size_t *comp_size
)
{
  ManagerBase::check_buffer_alignment(comp_buffer, decompress_alignment.input, "Decompression input");
  ManagerBase::check_buffer_alignment(decomp_buffer, decompress_alignment.output, "Decompression output");

  if (bitstream_kind == BitstreamKind::NVCOMP_NATIVE)
  {
    decompress_chunked_single(decomp_buffer, comp_buffer, decomp_config);
  }
  else
  {
    if (comp_size == nullptr)
    {
      throw NVCompException(nvcompErrorNotSupported, "comp_size can only be nullptr if bitstream kind is NVCOMP_NATIVE");
    }
    decompress_raw_batched(&decomp_buffer, &comp_buffer, {decomp_config}, comp_size);
  }
}
TEMPLATE
void HEAD::decompress(
  uint8_t *const *decomp_buffers,
  const uint8_t *const *comp_buffers,
  const std::vector<DecompressionConfig> &decomp_configs,
  const size_t *comp_sizes
)
{
  size_t batch_count = decomp_configs.size();
  decompress(decomp_buffers, comp_buffers, decomp_configs, comp_sizes, batch_count, nullptr);
}
TEMPLATE
void HEAD::decompress(
  uint8_t *const *decomp_buffers,
  const uint8_t *const *comp_buffers,
  const std::vector<DecompressionConfig> &decomp_configs,
  const size_t *comp_sizes,
  const size_t batch_count,
  const uint8_t *const *host_comp_buffers
)
{
  for (size_t idx = 0; idx < batch_count; ++idx)
  {
    ManagerBase::check_buffer_alignment(comp_buffers[idx], decompress_alignment.input, "Decompression input");
    ManagerBase::check_buffer_alignment(decomp_buffers[idx], decompress_alignment.output, "Decompression output");
  }

  //Check if the decompression configurations for all the batches are the same.
  bool checksum_enabled = false;
  bool host_setup_mode = host_comp_buffers != nullptr;

  if ((!host_setup_mode) && (decomp_configs.size() == 0))
  {
    throw NVCompException(
      nvcompErrorInvalidValue,
      "decomp_configs must be provided when host_comp_buffers is not provided"
    );
  }

  for (const DecompressionConfig &decomp_config : decomp_configs)
  {
    if (decomp_config.checksums_present == true)
    {
      checksum_enabled = true;
      break;
    }
  }

  if (bitstream_kind == BitstreamKind::NVCOMP_NATIVE)
  {
    if (checksum_enabled == true)
    {
      for (size_t idx = 0; idx < batch_count; ++idx)
      {
        if ((host_setup_mode) && (decomp_configs.size() != batch_count))
        {
          // Compute the decompression config from the common header
          const CommonHeader *this_batch_common_header = reinterpret_cast<const CommonHeader *>(host_comp_buffers[idx]);
          DecompressionConfig this_batch_decomp_config = extract_decomp_config(this_batch_common_header);
          decompress_chunked_single(decomp_buffers[idx], comp_buffers[idx], this_batch_decomp_config);
        }
        else
        {
          decompress_chunked_single(decomp_buffers[idx], comp_buffers[idx], decomp_configs[idx]);
        }
      }
    }
    else
    {
      if (host_setup_mode)
      {
        decompress_chunked_batched(decomp_buffers, comp_buffers, decomp_configs, batch_count, host_comp_buffers);
      }
      else
      {
        decompress_chunked_batched(decomp_buffers, comp_buffers, decomp_configs, batch_count);
      }
    }
  }
  else
  {
    if (comp_sizes == nullptr)
    {
      throw NVCompException(
        nvcompErrorNotSupported,
        "comp_sizes can only be nullptr if bitstream kind is NVCOMP_NATIVE"
      );
    }
    decompress_raw_batched(decomp_buffers, comp_buffers, decomp_configs, comp_sizes);
  }
}

#undef HEAD
#undef TEMPLATE

} // namespace nvcomp
