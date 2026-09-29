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

#include "CompressionConfigs.hpp"

namespace nvcomp
{

CompressionConfig::CompressionConfigImpl::CompressionConfigImpl()
    : status(nvcompSuccess)
{}

nvcompStatus_t *CompressionConfig::CompressionConfigImpl::get_status() const { return &status; }

CompressionConfig::CompressionConfig()
    : impl(std::make_shared<CompressionConfig::CompressionConfigImpl>())
    , uncompressed_buffer_size(0)
    , max_compressed_buffer_size(0)
    , num_chunks(0)
    , compute_checksums(false)
{}

CompressionConfig::CompressionConfig(size_t uncompressed_buffer_size)
    : impl(std::make_shared<CompressionConfig::CompressionConfigImpl>())
    , uncompressed_buffer_size(uncompressed_buffer_size)
    , max_compressed_buffer_size(0)
    , num_chunks(0)
    , compute_checksums(false)
{}

CompressionConfig::CompressionConfig(CompressionConfig &&other)
    : impl(std::move(other.impl))
    , uncompressed_buffer_size(other.uncompressed_buffer_size)
    , max_compressed_buffer_size(other.max_compressed_buffer_size)
    , num_chunks(other.num_chunks)
    , compute_checksums(other.compute_checksums)
{}

CompressionConfig::CompressionConfig(const CompressionConfig &other)
    : impl(other.impl)
    , uncompressed_buffer_size(other.uncompressed_buffer_size)
    , max_compressed_buffer_size(other.max_compressed_buffer_size)
    , num_chunks(other.num_chunks)
    , compute_checksums(other.compute_checksums)
{}

CompressionConfig &CompressionConfig::operator=(const CompressionConfig &other)
{
  impl = other.impl;
  uncompressed_buffer_size = other.uncompressed_buffer_size;
  max_compressed_buffer_size = other.max_compressed_buffer_size;
  num_chunks = other.num_chunks;
  compute_checksums = other.compute_checksums;
  return *this;
}

CompressionConfig &CompressionConfig::operator=(CompressionConfig &&other)
{
  impl = std::move(other.impl);
  uncompressed_buffer_size = other.uncompressed_buffer_size;
  max_compressed_buffer_size = other.max_compressed_buffer_size;
  num_chunks = other.num_chunks;
  compute_checksums = other.compute_checksums;
  return *this;
}

CompressionConfig::~CompressionConfig() {}

nvcompStatus_t *CompressionConfig::get_status() const { return impl->get_status(); }

DecompressionConfig::DecompressionConfigImpl::DecompressionConfigImpl()
    : status(nvcompSuccess)
    , uncompressed_sizes_and_offsets_provided()
{}

nvcompStatus_t *DecompressionConfig::DecompressionConfigImpl::get_status() const { return &status; }

DecompressionConfig::DecompressionConfig()
    : impl(std::make_shared<DecompressionConfig::DecompressionConfigImpl>())
    , decomp_data_size(0)
    , num_chunks(0)
    , checksums_present(false)
{}

DecompressionConfig::DecompressionConfig(const DecompressionConfig &other)
    : impl(other.impl)
    , decomp_data_size(other.decomp_data_size)
    , num_chunks(other.num_chunks)
    , checksums_present(other.checksums_present)
{}

DecompressionConfig::DecompressionConfig(DecompressionConfig &&other)
    : impl(std::move(other.impl))
    , decomp_data_size(other.decomp_data_size)
    , num_chunks(other.num_chunks)
    , checksums_present(other.checksums_present)
{}

DecompressionConfig::~DecompressionConfig() {}

DecompressionConfig &DecompressionConfig::operator=(const DecompressionConfig &other)
{
  impl = other.impl;
  decomp_data_size = other.decomp_data_size;
  num_chunks = other.num_chunks;
  checksums_present = other.checksums_present;
  return *this;
}

DecompressionConfig &DecompressionConfig::operator=(DecompressionConfig &&other)
{
  impl = std::move(other.impl);
  decomp_data_size = other.decomp_data_size;
  num_chunks = other.num_chunks;
  checksums_present = other.checksums_present;
  return *this;
}

nvcompStatus_t *DecompressionConfig::get_status() const { return impl->get_status(); }

} // namespace nvcomp
