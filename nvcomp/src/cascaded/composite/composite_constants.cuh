/*
 * Copyright (c) 2021-2026, NVIDIA CORPORATION. All rights reserved.
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

#include <algorithm>
#include <cstdint>

#include "nvcomp/cascaded.h"
#include "nvcomp/utils.hpp"

namespace nvcomp::cascaded::composite
{
// TODO (uhofmann): all those constexpr variables here have the wrong case

constexpr int default_chunk_size = 4096;

constexpr int max_chunk_metadata_size = 64;

constexpr int MAX_NUM_RLE_LAYERS = 7;

// Resulting function from rearranging the formula in "get_chunk_metadata_size"
// to solve for the number of delta layers, given the max chunk metadata size and the number of RLE layers.
constexpr int get_max_num_delta_layers(int num_RLEs, int max_chunk_metadata_size)
{
  return static_cast<int>(
    (max_chunk_metadata_size - nvcomp::roundUpTo(static_cast<size_t>(4 + 4 * (num_RLEs + 1)), sizeof(size_t))) /
    sizeof(size_t)
  );
}

// Since max number of RLE layers is 7, and max_chunk_metadata_size is 64,
// the function get_max_num_delta_layers constrains the max number of delta layers to be 3, assuming the data has a maximum size of 8B.
constexpr int MAX_NUM_DELTA_LAYERS = get_max_num_delta_layers(MAX_NUM_RLE_LAYERS, max_chunk_metadata_size);

// Maximum optional encoding stages considered by asymmetric Cascaded.
constexpr uint32_t ADAPTIVE_MAX_NUM_STAGES = 8;

struct AdaptiveStageCounts
{
  uint32_t num_RLEs;
  uint32_t num_deltas;
};

/**
 * @brief Map public Cascaded compress options to the RLE/Delta stage budget.
 *
 * Host-callable single source of truth for compressAsync and tests.
 */
inline AdaptiveStageCounts
map_adaptive_stage_counts(const uint8_t compression_level, const uint64_t fine_grained_encoding_flags)
{
  const uint32_t adaptive_stages = std::min<uint32_t>(compression_level, ADAPTIVE_MAX_NUM_STAGES);
  const bool use_rle = (fine_grained_encoding_flags & NVCOMP_CASCADED_FINE_GRAINED_ENCODING_RLE) != 0u;
  const bool use_delta = (fine_grained_encoding_flags & NVCOMP_CASCADED_FINE_GRAINED_ENCODING_DELTA) != 0u;
  if (use_rle && use_delta)
  {
    const uint32_t alternating_stages = std::min(adaptive_stages, static_cast<uint32_t>(2 * MAX_NUM_DELTA_LAYERS + 1));
    return {(alternating_stages + 1) / 2, alternating_stages / 2};
  }
  return {
    use_rle ? std::min(adaptive_stages, static_cast<uint32_t>(MAX_NUM_RLE_LAYERS)) : 0,
    use_delta ? std::min(adaptive_stages, static_cast<uint32_t>(MAX_NUM_DELTA_LAYERS)) : 0
  };
}

constexpr int composite_compress_threadblock_size = 128;
constexpr int composite_decompress_threadblock_size = 128;

} // namespace nvcomp::cascaded::composite
