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

#include <cstddef>
#include <cstdint>

#include "CudaConstants.h"

namespace ans_gpu_lib
{

// Type for in-chunk sizes, symbol counts and byte offsets in device code. A chunk
// is capped at nvcompANSCompressionMaxAllowedChunkSize (1 << 24) on compress and
// nvcompANSDecompressionMaxAllowedChunkSize (1 << 25) on decompress, so every
// intra-chunk quantity fits in 32 bits with room to spare. Values that cross the
// host API boundary or land in the bitstream stay size_t: the chunk size prefix
// carries a flag in bit 63, and the ANS_sub_chunk_header fields are format-fixed.
using IndexT = uint32_t;

// Note:
// symbols are collected on the byte level [0, ..., 255]
constexpr uint32_t NV_MAX_SYMBOL_VALUE = 255;
constexpr uint32_t NV_SYMBOL_COUNT = NV_MAX_SYMBOL_VALUE + 1;
constexpr uint32_t MIN_TABLELOG = 5;
constexpr uint32_t MAX_TABLELOG = 11;
constexpr uint32_t DEFAULT_TABLELOG = 10;
constexpr uint32_t DEFAULT_MAX_PATTERN_SIZE = 1 << DEFAULT_TABLELOG;
constexpr int MIN_SUB_CHUNKS_PER_CHUNK = 4;
constexpr int MAX_SUB_CHUNKS_PER_CHUNK = 64;
constexpr int MIN_SUB_CHUNK_SIZE = 2048;
constexpr uint32_t DEFAULT_DEVICE_SUB_CHUNK_SIZE = 4096;
constexpr int NUM_COMP_WARPS_PER_CTA = 4;
constexpr int NUM_COMP_THREADS_PER_CTA = NUM_COMP_WARPS_PER_CTA * WARP_SIZE;
constexpr int NUM_DECOMP_WARPS_PER_CTA = 4;
constexpr int ANS_FUSED_DEFRAG_UINT4S_PER_THREAD = 3;
constexpr size_t SYMBOL_TABLE_SIZE = 1024;
constexpr size_t CDF_TABLE_SIZE = 514;
// Shared encoding table size (shared across the CTA, irrespective of warp count)
constexpr size_t SHARED_ENCODING_TABLE_SIZE = 3072;
// nvCOMPDx FP16 per-warp staging buffer (WARP_SIZE * sizeof(uint2)).
constexpr size_t SHARED_STAGING_BUFFER_SIZE_PER_WARP = 256;
// Shared decoding buffer size (128 * sizeof(uint16_t))
constexpr size_t SHARED_DECODING_BUFFER_SIZE_PER_WARP = 256;
constexpr size_t NUM_ENCODES_PER_LOOP = 64;
constexpr size_t NUM_ENCODES_PER_THREAD = 2;
constexpr size_t BITS_PER_BYTE = 8;

} // namespace ans_gpu_lib
