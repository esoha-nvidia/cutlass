/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
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

#include "nvcomp/nvcompManager.hpp"
#include "nvcomp_common_deps/hlif_shared_types.hpp"

namespace nvcomp::detail
{

PimplManager::PimplManager(std::unique_ptr<nvcompManagerInternalBase> p) noexcept
    : impl(std::move(p))
{}

nvcompAlignmentRequirements_t PimplManager::get_required_compression_alignments() const
{
  return impl->get_required_compression_alignments();
}

nvcompAlignmentRequirements_t PimplManager::get_required_decompression_alignments() const
{
  return impl->get_required_decompression_alignments();
}

CompressionConfig PimplManager::configure_compression(const size_t uncomp_buffer_size)
{
  return impl->configure_compression(uncomp_buffer_size);
}

std::vector<CompressionConfig> PimplManager::configure_compression(const std::vector<size_t> &uncomp_buffer_sizes)
{
  return impl->configure_compression(uncomp_buffer_sizes);
}

void PimplManager::compress(
  const uint8_t *uncomp_buffer,
  uint8_t *comp_buffer,
  const CompressionConfig &comp_config,
  size_t *comp_size
)
{
  return impl->compress(uncomp_buffer, comp_buffer, comp_config, comp_size);
}

void PimplManager::compress(
  const uint8_t *const *uncomp_buffers,
  uint8_t *const *comp_buffers,
  const std::vector<CompressionConfig> &comp_configs,
  size_t *comp_sizes
)
{
  return impl->compress(uncomp_buffers, comp_buffers, comp_configs, comp_sizes);
}

DecompressionConfig PimplManager::configure_decompression(const uint8_t *comp_buffer, const size_t *comp_size)
{
  return impl->configure_decompression(comp_buffer, comp_size);
}

std::vector<DecompressionConfig>
PimplManager::configure_decompression(const uint8_t *const *comp_buffers, size_t batch_size, const size_t *comp_sizes)
{
  return impl->configure_decompression(comp_buffers, batch_size, comp_sizes);
}

DecompressionConfig PimplManager::configure_decompression(const CompressionConfig &comp_config)
{
  return impl->configure_decompression(comp_config);
}

std::vector<DecompressionConfig>
PimplManager::configure_decompression(const std::vector<CompressionConfig> &comp_configs)
{
  return impl->configure_decompression(comp_configs);
}

void PimplManager::decompress(
  uint8_t *decomp_buffer,
  const uint8_t *comp_buffer,
  const DecompressionConfig &decomp_config,
  size_t *comp_size
)
{
  return impl->decompress(decomp_buffer, comp_buffer, decomp_config, comp_size);
}

void PimplManager::decompress(
  uint8_t *const *decomp_buffers,
  const uint8_t *const *comp_buffers,
  const std::vector<DecompressionConfig> &decomp_configs,
  const size_t *comp_sizes
)
{
  return impl->decompress(decomp_buffers, comp_buffers, decomp_configs, comp_sizes);
}

void PimplManager::decompress(
  uint8_t *const *decomp_buffers,
  const uint8_t *const *comp_buffers,
  const std::vector<DecompressionConfig> &decomp_configs,
  const size_t *comp_sizes,
  const size_t batch_count,
  const uint8_t *const *host_comp_buffers
)
{
  return impl->decompress(decomp_buffers, comp_buffers, decomp_configs, comp_sizes, batch_count, host_comp_buffers);
}

void PimplManager::set_scratch_allocators(const AllocFn_t &alloc_fn, const DeAllocFn_t &dealloc_fn)
{
  return impl->set_scratch_allocators(alloc_fn, dealloc_fn);
}

size_t PimplManager::get_compressed_output_size(const uint8_t *comp_buffer)
{
  return impl->get_compressed_output_size(comp_buffer);
}

std::vector<size_t> PimplManager::get_compressed_output_size(const uint8_t *const *comp_buffers, size_t batch_size)
{
  return impl->get_compressed_output_size(comp_buffers, batch_size);
}

size_t PimplManager::get_decompressed_output_size(const uint8_t *comp_buffer)
{
  return impl->get_decompressed_output_size(comp_buffer);
}

std::vector<size_t> PimplManager::get_decompressed_output_size(const uint8_t *const *comp_buffers, size_t batch_size)
{
  return impl->get_decompressed_output_size(comp_buffers, batch_size);
}

void PimplManager::deallocate_gpu_mem() { impl->deallocate_gpu_mem(); }

} // namespace nvcomp::detail
