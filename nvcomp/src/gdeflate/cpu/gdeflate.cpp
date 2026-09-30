/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
*/

#include <cassert>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "common.h"
#include "gdeflate_constants.h"
#include "nvcomp.hpp"
#include "nvcomp/native/gdeflate_cpu.h"

#include <libdeflate.h>
#define WITH_GDEFLATE
#include "deflate_constants.h"

namespace nvcomp::gdeflate
{

// Verify public & internal constants
static_assert(nvcompGdeflateCPUCompressionMaxAllowedChunkSize == GDEFLATE_PAGE_SIZE);
static_assert(
  (nvcompGdeflateCPURequiredCompressionAlignment & (nvcompGdeflateCPURequiredCompressionAlignment - 1)) == 0
);
static_assert(
  (nvcompGdeflateCPURequiredDecompressionAlignment & (nvcompGdeflateCPURequiredDecompressionAlignment - 1)) == 0
);

namespace
{

using GdeflateCompressorHandle =
  std::unique_ptr<libdeflate_gdeflate_compressor, decltype(&libdeflate_free_gdeflate_compressor)>;
using GdeflateDecompressorHandle =
  std::unique_ptr<libdeflate_gdeflate_decompressor, decltype(&libdeflate_free_gdeflate_decompressor)>;

void validate_max_uncompressed_chunk_bytes(size_t max_uncompressed_chunk_bytes)
{
  if (max_uncompressed_chunk_bytes > nvcompGdeflateCPUCompressionMaxAllowedChunkSize)
  {
    throw NVCompException(
      nvcompErrorInvalidValue,
      "Maximum allowed chunk size for Gdeflate CPU is " +
        std::to_string(nvcompGdeflateCPUCompressionMaxAllowedChunkSize) + " bytes"
    );
  }
}

} // namespace

void compressCPUGetMaxOutputChunkSize(size_t max_uncompressed_chunk_bytes, size_t *max_compressed_chunk_bytes)
{

  if (max_compressed_chunk_bytes == nullptr)
  {
    throw NVCompException(nvcompErrorInvalidValue, "max_compressed_chunk_bytes must not be null");
  }
  validate_max_uncompressed_chunk_bytes(max_uncompressed_chunk_bytes);

  // TODO: Check how/why this differs from our gdeflate::compressGetMaxOutputChunkSize implementation .
  size_t npages;
  *max_compressed_chunk_bytes = libdeflate_gdeflate_compress_bound(nullptr, max_uncompressed_chunk_bytes, &npages);
  assert(npages == 1);
}

void decompressCPU(
  const void *const *in_ptr,
  const size_t *in_bytes,
  size_t batch_size,
  void *const *out_ptr,
  size_t *out_buffer_bytes,
  size_t *out_bytes
)
{

  if (in_ptr == nullptr)
  {
    throw NVCompException(nvcompErrorInvalidValue, "in_ptr must not be null");
  }
  if (in_bytes == nullptr)
  {
    throw NVCompException(nvcompErrorInvalidValue, "in_bytes must not be null");
  }
  if (out_ptr == nullptr)
  {
    throw NVCompException(nvcompErrorInvalidValue, "out_ptr must not be null");
  }
  if (out_buffer_bytes == nullptr)
  {
    throw NVCompException(nvcompErrorInvalidValue, "out_buffer_bytes must not be null");
  }
  if (out_bytes == nullptr)
  {
    throw NVCompException(nvcompErrorInvalidValue, "out_bytes must not be null");
  }

  GdeflateDecompressorHandle decompressor(
    libdeflate_alloc_gdeflate_decompressor(),
    libdeflate_free_gdeflate_decompressor
  );
  if (decompressor == nullptr)
  {
    throw NVCompException(nvcompErrorInternal, "Failed to allocate Gdeflate CPU decompressor");
  }

  // TODO: why not parallel?
  for (size_t i = 0; i < batch_size; ++i)
  {
    libdeflate_gdeflate_in_page page;
    page.data = in_ptr[i];
    page.nbytes = in_bytes[i];

    // Note: out_bytes and out_buffer_bytes might point to the same memory
    //       location, so we cache the value present and use the cached value
    //       for the available bytes going forward.
    size_t out_buffer_bytes_i = out_buffer_bytes[i];
    // Note: the decompress API only increments the size, hence we need to
    //       zero it out ourselves.
    out_bytes[i] = 0;
    libdeflate_result result =
      libdeflate_gdeflate_decompress(decompressor.get(), &page, 1, out_ptr[i], out_buffer_bytes_i, &out_bytes[i]);
    if (result != LIBDEFLATE_SUCCESS)
    {
      throw NVCompException(nvcompErrorCannotDecompress, "Failed to decompress chunk");
    }
  }
}

void compressCPU(
  const void *const *in_ptr,
  const size_t *in_bytes,
  const size_t max_uncompressed_chunk_bytes,
  size_t batch_size,
  void *const *out_ptr,
  size_t *out_bytes,
  int compression_level
)
{

  if (in_ptr == nullptr)
  {
    throw NVCompException(nvcompErrorInvalidValue, "in_ptr must not be null");
  }
  if (in_bytes == nullptr)
  {
    throw NVCompException(nvcompErrorInvalidValue, "in_bytes must not be null");
  }
  if (out_ptr == nullptr)
  {
    throw NVCompException(nvcompErrorInvalidValue, "out_ptr must not be null");
  }
  if (out_bytes == nullptr)
  {
    throw NVCompException(nvcompErrorInvalidValue, "out_bytes must not be null");
  }
  validate_max_uncompressed_chunk_bytes(max_uncompressed_chunk_bytes);
  if (compression_level < nvcompGdeflateCPUMinCompressionLevel ||
      compression_level > nvcompGdeflateCPUMaxCompressionLevel)
  {
    throw NVCompException(
      nvcompErrorInvalidValue,
      "Compression level must be between " + std::to_string(nvcompGdeflateCPUMinCompressionLevel) + " and " +
        std::to_string(nvcompGdeflateCPUMaxCompressionLevel) + ", both inclusive"
    );
  }

  size_t npages;
  size_t max_comp_len = libdeflate_gdeflate_compress_bound(nullptr, max_uncompressed_chunk_bytes, &npages);
  assert(npages == 1);

  GdeflateCompressorHandle compressor(
    libdeflate_alloc_gdeflate_compressor(compression_level),
    libdeflate_free_gdeflate_compressor
  );
  if (compressor == nullptr)
  {
    throw NVCompException(nvcompErrorInternal, "Failed to allocate Gdeflate CPU compressor");
  }

  // TODO: why not parallel?
  for (size_t i = 0; i < batch_size; ++i)
  {
    if (in_bytes[i] > max_uncompressed_chunk_bytes)
    {
      throw NVCompException(
        nvcompErrorInvalidValue,
        "max_uncompressed_chunk_bytes cannot be lower than any single chunk size"
      );
    }

    // Create a single libdeflate page from each chunk
    libdeflate_gdeflate_out_page page;
    page.data = out_ptr[i];
    page.nbytes = max_comp_len;

    size_t compressedSize = libdeflate_gdeflate_compress(compressor.get(), in_ptr[i], in_bytes[i], &page, 1);

    if (compressedSize == 0)
    {
      throw NVCompException(
        nvcompErrorCannotCompress,
        "Failed to compress chunk, check if max_uncompressed_chunk_bytes is set correctly"
      );
    }
    out_bytes[i] = page.nbytes;
  }
}

} // namespace nvcomp::gdeflate
