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

#include <cassert>
#include <cstdint>
#include <cstring>

#include "cascaded/modules/bitpack.cuh"
#include "cascaded/modules/delta.cuh"
#include "cascaded/modules/rle.cuh"
#include "composite_constants.cuh"
#include "Reduction.cuh"

namespace nvcomp::cascaded::composite
{

// Adaptive chunks have a fixed two-word prefix. The first word stores the
// record size and two stage bitmaps; bit zero represents the first optional
// stage. The RLE bitmap is a subset of the applied-stage bitmap, and an applied
// stage without its RLE bit is a Delta stage. The second word stores the final
// stream size.
constexpr uint32_t ADAPTIVE_CHUNK_PREFIX_SIZE = 2u * sizeof(uint32_t);
constexpr uint32_t ADAPTIVE_CHUNK_SIZE_MASK = 0xffffu;
constexpr uint32_t ADAPTIVE_CHUNK_STAGES_SHIFT = 16u;
constexpr uint32_t ADAPTIVE_CHUNK_STAGES_MASK = 0xffu;
constexpr uint32_t ADAPTIVE_CHUNK_RLE_STAGES_SHIFT = 24u;

struct AdaptiveCompressionOptions
{
  uint32_t num_RLEs;
  uint32_t num_deltas;
  bool use_bp;
};

template <typename data_type>
inline __device__ void serialize_value(uint32_t *const destination, const data_type &value)
{
  memcpy(destination, &value, sizeof(data_type));
}

template <typename data_type>
inline __device__ data_type deserialize_value(const uint32_t *const source)
{
  data_type value;
  memcpy(&value, source, sizeof(data_type));
  return value;
}

inline __device__ uint32_t
pack_adaptive_chunk_size(const uint32_t chunk_size_bytes, const uint32_t applied_stages, const uint32_t rle_stages)
{
  assert(chunk_size_bytes <= ADAPTIVE_CHUNK_SIZE_MASK);
  assert(applied_stages <= ADAPTIVE_CHUNK_STAGES_MASK);
  assert((rle_stages & ~applied_stages) == 0u);
  return chunk_size_bytes | (applied_stages << ADAPTIVE_CHUNK_STAGES_SHIFT) |
         (rle_stages << ADAPTIVE_CHUNK_RLE_STAGES_SHIFT);
}

inline __device__ uint32_t get_adaptive_chunk_size(const uint32_t packed_chunk_size)
{
  return packed_chunk_size & ADAPTIVE_CHUNK_SIZE_MASK;
}

inline __device__ uint32_t get_adaptive_applied_stages(const uint32_t packed_chunk_size)
{
  return (packed_chunk_size >> ADAPTIVE_CHUNK_STAGES_SHIFT) & ADAPTIVE_CHUNK_STAGES_MASK;
}

inline __device__ uint32_t get_adaptive_rle_stages(const uint32_t packed_chunk_size)
{
  return (packed_chunk_size >> ADAPTIVE_CHUNK_RLE_STAGES_SHIFT) & ADAPTIVE_CHUNK_STAGES_MASK;
}

/**
 * Helper function to calculate the size in byte of the chunk metadata. The size
 * is guaranteed to be a multiple of the data type size, and a multiple of 4.
 */
template <typename data_type>
__device__ int get_chunk_metadata_size(const int num_RLEs, const int num_deltas)
{
  const int chunk_metadata_size = roundUpTo(4 + 4 * (num_RLEs + 1), sizeof(data_type)) +
                                  roundUpTo(sizeof(data_type) * num_deltas, 4);

  // We currently assume, for example in composite_comp_kernels.cuh:264, that it
  // is a multiple of 4.
  assert(chunk_metadata_size % sizeof(uint32_t) == 0);
  return chunk_metadata_size;
}

/**
 * @brief Helper function to determine the shared memory requirement
 * based on the chunk size and datatype...
 * @tparam chunk_size Size of the chunk in bytes
 * @tparam width Width of the datatype being used (in bytes)
 * @tparam storage_width Width in bytes of the storage type used
 * to compute offset.  This should be a minimum of 4 bytes, or 8
 * bytes if the width is also 8 bytes.
 */
template <int chunk_size, int width, int storage_width>
constexpr __device__ int compute_decompress_smem_size()
{
  constexpr int storage_num_elts = roundUpDiv(chunk_size + 4 + width, storage_width);
  constexpr int tot_elt_storage = 2 * (storage_num_elts * storage_width);
  constexpr int run_width = 2;
  constexpr int chunk_num_elements = chunk_size / width;
  constexpr int tot_count_bytes = 2 * (chunk_num_elements * run_width);

  return 64 + tot_elt_storage + tot_count_bytes + (4 * 8);
}

} // namespace nvcomp::cascaded::composite
