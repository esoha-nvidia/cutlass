// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once

#include <cuda/std/array>
#include <cuda/std/functional>
#include <cuda/std/tuple>

#include <cstdint>

#include "CudaConstants.h"
#include "nvcomp/utils.hpp"

namespace nvcomp
{
namespace lz4
{

// This restricts us to 4GB chunk sizes (total buffer can be up to
// max(size_t)). We actually artificially restrict it to much less, to
// limit what we have to test, as well as to encourage users to exploit some
// parallelism. If this changes, nvcompLZ4DecompressionMaxAllowedChunkSize in lz4.h should be updated to match.
using position_type = uint32_t;

// Limits lookback to 64 KB
using offset_type = uint16_t;

struct sequence
{
  uint32_t distance;
  uint32_t match_length;
  uint32_t literal_length;
};

using LZ4DecompressBuffer = cuda::std::array<uint8_t, WARP_SIZE_U * sizeof(uint64_t)>;

struct LZ4DecompressWarpMemory
{
  // hack to expose buffer size since size() is not a static member of array
  static constexpr auto BUFFER_SIZE = cuda::std::tuple_size<LZ4DecompressBuffer>::value;

  alignas(8) LZ4DecompressBuffer buffer;
  int ix_output[WARP_SIZE_U];
};

/**
 * @brief Get the minimum of two numbers
 * @note This template function is necessary, because we compile
 *       the device API / nvcompDx with CTK 12.6.x where cuda::std::min
 *       is a private function, and cannot be properly included.
 *
 * @param[in] a The first argument to consider
 * @param[in] b The second argument to consider
 *
 * @return The minimum of a and b
 */
template <typename T>
constexpr __host__ __device__ T min(const T &a, const T &b)
{
  return a < b ? a : b;
}

/**
 * @brief Get the size of the hash table needed for the given maximum chunk
 * size.
 *
 * @param[in] max_uncomp_chunk_size The maximum chunk size to process.
 *
 * @return The number of elements/slots required in the hashtable.
 */
constexpr __host__ __device__ size_t getHashTableSize(const size_t max_uncomp_chunk_size)
{
  constexpr const size_t MAX_HASH_TABLE_SIZE = 1U << 14;

  // when chunk size is smaller than the max hashtable size round the
  // hashtable size up to the nearest power of 2 of the chunk size.
  // The lower load factor from a significantly larger hashtable size compared
  // to the chunk size doesn't increase performance, however having a smaller
  // hashtable which yields much high cache utilization does.
  return lz4::min(roundUpPow2(max_uncomp_chunk_size), MAX_HASH_TABLE_SIZE);
}

} // namespace lz4
} // namespace nvcomp
