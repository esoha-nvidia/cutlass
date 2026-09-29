/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES.
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

#include "nvcomp.hpp"

typedef uint64_t ChunkStartOffset_t;
typedef uint32_t Checksum_t;

// 16B offset to ensure that bumping a compression format's required alignment from 8B to 16B doesn't break the format inadvertently.
static constexpr size_t UNCOMPRESSED_SIZE_OFFSET = 16;

/* The value was choosen by random number generator, then bumped for the
 * packed 56-byte CommonHeader wire format (last byte 38 -> 39).
 * When divided into bytes (Big endian) it is equal to:
 * in decimal
 * 82 | 145 | 19 | 39
 * in binary:
 * 01010010 | 10010001 | 00010011 | 00100111
 * ASCII decode:
 * R | æ | DC3 | '
 * MAGIC_NUMBER for pre-v6.0 HLIF buffers = 1385239334 (82 | 145 | 19 | 38)
*/
static constexpr uint32_t LEGACY_MAGIC_NUMBER = 1385239334;
static constexpr uint32_t MAGIC_NUMBER = 1385239335;

// This magic number is used to indicate we're doing parquet conversion to hlif,
// which provides the uncompressed offset and size of each chunk.
static constexpr uint32_t LEGACY_PARQUET_HLIF_MAGIC_NUMBER = LEGACY_MAGIC_NUMBER ^ 0xff;
static constexpr uint32_t PARQUET_HLIF_MAGIC_NUMBER = MAGIC_NUMBER ^ 0xff;

/*
 * HLIF buffer content with magic_number = MAGIC_NUMBER:
 *
 * {
 *   CommonHeader - fixed size
 *   FormatSpecHeader - its size is changing on the format
 *   Compressed chunk offsets [num_chunks * size_t]
 *   Compressed chunk sizes [num_chunks * size_t]
 *   Compressed chunk checksums [num_chunks * uint32_t] optional, aligned to 8-byte boundary
 *   Decompressed chunk checksums [num_chunks * uint32_t] optional
 * }
 * { Compressed chunk 0 }
 * { Compressed chunk 1 }
 * ...
 * { Compressed chunk num_chunks - 1 }
 *
 *
 * HLIF buffer content with magic_number = PARQUET_HLIF_MAGIC_NUMBER:
 *
 * {
 *   CommonHeader - fixed size
 *   FormatSpecHeader - its size is changing on the format
 *   Compressed chunk offsets [ num_chunks * size_t ]
 *   Compressed chunk sizes [ num_chunks * size_t ]
 *   Uncompressed chunk offsets [ num_chunks * size_t ]
 *   Uncompressed chunk sizes [ num_chunks * size_t ]
 *   Compressed chunk checksums [num_chunks * uint32_t] optional, aligned to 8-byte boundary
 *   Decompressed chunk checksums [num_chunks * uint32_t] optional
 * }
 * { Compressed chunk 0 }
 * { Compressed chunk 1 }
 * ...
 * { Compressed chunk num_chunks - 1 }
 */

// Packed HLIF common header (56 bytes). Member order is intentional
// to minimize ABI padding; This reordering breaks the ABI and version above nvCOMP v6.0
// will not be able to decode HLIF buffers compressed by older versions. Changing the
// magic number will make sure we can error-out nicely when we try to decompress older buffers.
struct CommonHeader
{
  uint32_t magic_number;
  uint8_t major_version;
  uint8_t minor_version;
  nvcomp::nvcompFormatType_t format;
  uint64_t comp_data_size;
  uint64_t decomp_data_size;
  size_t num_chunks;
  size_t uncomp_chunk_size;
  uint32_t comp_data_offset;
  Checksum_t full_comp_buffer_checksum;
  Checksum_t decomp_buffer_checksum;
  bool include_chunk_starts;
  bool include_per_chunk_comp_buffer_checksums;
  bool include_per_chunk_decomp_buffer_checksums;
};

static_assert(sizeof(CommonHeader) == 56, "CommonHeader must be 56 bytes");

struct CompressArgs
{
  CommonHeader *common_header;
  const uint8_t *decomp_buffer;
  size_t decomp_buffer_size;
  uint8_t *comp_buffer;
  uint8_t *scratch_buffer;
  size_t uncomp_chunk_size;
  size_t *ix_output;
  uint32_t *ix_chunk;
  size_t num_chunks;
  size_t max_comp_chunk_size;
  size_t *comp_chunk_offsets;
  size_t *comp_chunk_sizes;
  nvcompStatus_t *output_status;
};

namespace nvcomp
{

// Reject pre-v6.0 (legacy magic) and non-NVCOMP_NATIVE HLIF buffers.
inline void validate_hlif_magic_number(const uint32_t magic_number)
{
  if (magic_number == 0 || magic_number == LEGACY_MAGIC_NUMBER || magic_number == LEGACY_PARQUET_HLIF_MAGIC_NUMBER)
  {
    throw NVCompException(
      nvcompErrorInvalidValue,
      "The passed data was compressed with an old version of nvCOMP (<=5.3.0).\n"
      "The support for that version is deprecated.\n"
      "Please decompress the data using nvCOMP <=5.3.0 or\n"
      "compress the data with this nvCOMP version."
    );
  }
  if (magic_number != MAGIC_NUMBER && magic_number != PARQUET_HLIF_MAGIC_NUMBER)
  {
    throw NVCompException(
      nvcompErrorInvalidValue,
      "The passed data was not compressed with NVCOMP_NATIVE bitstream "
      "kind."
    );
  }
}

namespace detail
{

/**
 * @brief Interface class that ManagerBase inherits from
 * (and by extension all <Format>Manager classes)
 * Having this private allows us to add private abstract helper methods
 * that aren't exposed to the user
 */
struct nvcompManagerInternalBase : nvcompManagerBase
{

  virtual void recompress(
    const uint8_t *decomp_buffer,
    const uint8_t *comp_buffer,
    const DecompressionConfig &decomp_config,
    size_t &recompress_size,
    float &recompress_throughput,
    float &recompress_ratio
  ) = 0;

  virtual std::vector<uint8_t> do_replicate(const std::vector<uint8_t> &input_data, int extra_rep_count) = 0;
};

} // namespace detail
} // namespace nvcomp
