/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or its
 * affiliates is strictly prohibited.
 */

#pragma once

#include <cassert>
#include <cstdint>
#include <type_traits>

#include "cascaded/next/primitives/kibi_cascader/kibi_cascader_dx_uint32.cuh"
#include "cascaded/next/terminal_codecs/terminal_cascaded/terminal_cascaded_io.cuh"
#include "cascaded/next/terminal_codecs/terminal_cascaded/terminal_cascaded_validate.cuh"
#include "cascaded/universal/universal_header.cuh"
#include "CudaConstants.h"
#include "nvcomp.h"

namespace nvcomp::cascaded::next::terminal_cascaded
{

template <typename T, kibi_cascader::kibi_cascader_optimization_search_space SearchSpace>
struct CompressStorage
{
  uint32_t sizes[WARPS_PER_CTA];
  uint32_t tile_kc_buffers_bytes;
  uint32_t shared_metadata[WARPS_PER_CTA][kibi_cascader::MAX_KC_METADATA_WORDS];
  uint32_t shared_input[WARPS_PER_CTA][WORDS_PER_KC];
  union
  {
    uint32_t shared_workspace[WARPS_PER_CTA][WORDS_PER_KC];
    alignas(16) uint8_t scratch[COMPRESS_SCRATCH_BYTES];
  };
};

inline __device__ uint32_t thread_exclusive_scan_kc_buffer_sizes(uint32_t *const sizes)
{
  uint32_t total_size = 0u;
  for (uint32_t warp = 0u; warp < WARPS_PER_CTA; ++warp)
  {
    const uint32_t kc_buffer_size = sizes[warp];
    sizes[warp] = total_size;
    total_size += kc_buffer_size;
  }
  return total_size;
}

template <kibi_cascader::kibi_cascader_optimization_search_space SearchSpace>
inline __device__ void compress_chunk_u32(
  const uint8_t *__restrict__ const in,
  const uint32_t uncompressed_bytes,
  uint8_t *__restrict__ const out,
  const nvcompType_t data_type,
  CompressStorage<uint32_t, SearchSpace> &storage,
  uint32_t &compressed_bytes
)
{
  assert(uncompressed_bytes % sizeof(uint32_t) == 0u);

  const uint32_t num_kc_batches = num_kc_batches_for(uncompressed_bytes);
  const uint32_t total_wds = total_words(uncompressed_bytes);
  const uint32_t num_sub = num_sub_units_for(num_kc_batches);
  const uint32_t warp_id = threadIdx.x / WARP_SIZE;
  const uint32_t lane = lane_id();
  uint32_t *const shared_input = storage.shared_input[warp_id];
  uint32_t *const shared_metadata = storage.shared_metadata[warp_id];
  uint32_t *const shared_workspace = storage.shared_workspace[warp_id];

  if (threadIdx.x == 0u)
  {
    universal_header::write_cascaded_next_header(
      out,
      data_type,
      uncompressed_bytes,
      universal_header::CompressionMode::Symmetric,
      universal_header::TerminalCodec::CascadedBitpack
    );
  }

  uint32_t sub_unit_start = preamble_size(num_sub);

  for (uint32_t tile = 0u; tile < num_kc_batches; tile += WARPS_PER_CTA)
  {
    const uint32_t ix_batch = tile + warp_id;
    const bool is_batch_in_bounds = ix_batch < num_kc_batches;

    uint32_t bitpack_input[WORDS_PER_THREAD];
    kibi_cascader::KibiCascaderCompressionPlan plan{};
    uint32_t metadata_size_bytes = 0u;
    uint32_t input_size_bytes = 0u;
    uint32_t kc_buffer_size_bytes = 0u;

    if (is_batch_in_bounds)
    {
      input_size_bytes = kc_size_bytes(uncompressed_bytes, ix_batch);
      const uint32_t na = num_active_for(kc_words(total_wds, ix_batch));
      const bool is_full_batch = input_size_bytes == KC_SIZE_BYTES;
      if (is_full_batch)
      {
        load_full_kc_coalesced(in, ix_batch, lane, bitpack_input, shared_input);
      }
      else
      {
        load_kc(in, uncompressed_bytes, ix_batch, lane, na, bitpack_input);
#pragma unroll
        for (uint32_t word = 0u; word < WORDS_PER_THREAD; ++word)
        {
          const uint32_t logical_word = lane * WORDS_PER_THREAD + word;
          shared_input[kibi_cascader::shared_input_word_index(logical_word)] = bitpack_input[word];
        }
        __syncwarp();
      }

      plan = kibi_cascader::plan<uint32_t, SearchSpace>(
        shared_input,
        bitpack_input,
        shared_metadata,
        shared_workspace,
        input_size_bytes
      );
      metadata_size_bytes = get_serialized_metadata_size_bytes<SearchSpace>(shared_metadata);
      kc_buffer_size_bytes = KC_BUFFER_PREFIX_BYTES + metadata_size_bytes + plan.payload_size_bytes;
    }

    if (lane == 0u)
    {
      storage.sizes[warp_id] = kc_buffer_size_bytes;
    }
    __syncthreads();

    if (threadIdx.x == 0u)
    {
      storage.tile_kc_buffers_bytes = thread_exclusive_scan_kc_buffer_sizes(storage.sizes);
    }
    __syncthreads();

    const uint32_t tile_kc_buffers_bytes = storage.tile_kc_buffers_bytes;
    const uint32_t n = min(WARPS_PER_CTA, num_kc_batches - tile);

    if (is_batch_in_bounds)
    {
      const uint32_t kc_buffer_offset = n * KC_BUFFER_OFFSET_BYTES + storage.sizes[warp_id];
      uint8_t *kc_buffer = out + sub_unit_start + kc_buffer_offset;

      if (lane == 0u)
      {
        store_unaligned<uint16_t>(
          out + sub_unit_start + warp_id * KC_BUFFER_OFFSET_BYTES,
          static_cast<uint16_t>(kc_buffer_offset)
        );
        kc_buffer[0u] = static_cast<uint8_t>(metadata_size_bytes);
      }
      serialize_u32_kc_metadata<SearchSpace>(shared_metadata, kc_buffer + KC_BUFFER_PREFIX_BYTES);
      kibi_cascader::encode<uint32_t, SearchSpace>(
        bitpack_input,
        shared_metadata,
        kc_buffer + KC_BUFFER_PREFIX_BYTES + metadata_size_bytes,
        plan
      );
    }

    if (threadIdx.x == 0u)
    {
      const uint32_t sub_unit_size = n * KC_BUFFER_OFFSET_BYTES + tile_kc_buffers_bytes;
      store_unaligned<uint16_t>(
        out + universal_header::HEADER_SIZE_BYTES + (tile / WARPS_PER_CTA) * sizeof(uint16_t),
        static_cast<uint16_t>(sub_unit_size)
      );
    }

    __syncthreads();
    const uint32_t sub_unit_size = n * KC_BUFFER_OFFSET_BYTES + tile_kc_buffers_bytes;
    sub_unit_start += sub_unit_size;
  }

  if (threadIdx.x == 0u)
  {
    compressed_bytes = sub_unit_start;
  }
}

template <typename T, kibi_cascader::kibi_cascader_optimization_search_space SearchSpace>
inline __device__ void compress_chunk(
  const uint8_t *__restrict__ const in,
  const uint32_t uncompressed_bytes,
  uint8_t *__restrict__ const out,
  const nvcompType_t data_type,
  CompressStorage<T, SearchSpace> &storage,
  uint32_t &compressed_bytes
)
{
  if constexpr (std::is_same_v<T, uint32_t>)
  {
    compress_chunk_u32<SearchSpace>(in, uncompressed_bytes, out, data_type, storage, compressed_bytes);
  }
  else
  {
    static_assert(std::is_same_v<T, uint64_t>);
    if (uncompressed_bytes % sizeof(T) != 0u)
    {
      if (threadIdx.x == 0u)
      {
        compressed_bytes = 0u;
      }
      return;
    }

    const uint32_t num_kc_batches = num_kc_batches_for(uncompressed_bytes);
    const uint32_t num_sub = num_sub_units_for(num_kc_batches);
    const uint32_t warp_id = threadIdx.x / WARP_SIZE;
    const uint32_t lane = lane_id();
    constexpr auto PLANE_SEARCH_SPACE = SearchSpace;
    uint32_t *const shared_input = storage.shared_input[warp_id];
    uint32_t *const shared_metadata = storage.shared_metadata[warp_id];
    uint32_t *const shared_workspace = storage.shared_workspace[warp_id];

    if (threadIdx.x == 0u)
    {
      universal_header::write_cascaded_next_header(
        out,
        data_type,
        uncompressed_bytes,
        universal_header::CompressionMode::Symmetric,
        universal_header::TerminalCodec::CascadedBitpack
      );
    }

    uint32_t sub_unit_start = preamble_size(num_sub);

    for (uint32_t tile = 0u; tile < num_kc_batches; tile += WARPS_PER_CTA)
    {
      const uint32_t ix_batch = tile + warp_id;
      const bool is_batch_in_bounds = ix_batch < num_kc_batches;
      const bool is_paired_batch = is_batch_in_bounds && ((ix_batch | 1u) < num_kc_batches);

      uint32_t bitpack_input[WORDS_PER_THREAD];
      kibi_cascader::KibiCascaderCompressionPlan plan{};
      uint32_t metadata_size_bytes = 0u;
      uint32_t input_size_bytes = 0u;
      uint32_t kc_buffer_size_bytes = 0u;

      if (is_paired_batch)
      {
        uint32_t *const pair_scratch = storage.shared_input[warp_id & ~1u];
        stage_uint64_pair_input(in, uncompressed_bytes, ix_batch, lane, pair_scratch);
      }
      __syncthreads();

      if (is_batch_in_bounds)
      {
        if (is_paired_batch)
        {
          const uint32_t num_values = uint64_pair_num_values(uncompressed_bytes, ix_batch & ~1u);
          const uint32_t *const pair_scratch = storage.shared_input[warp_id & ~1u];
          input_size_bytes = num_values * sizeof(uint32_t);
          load_uint64_pair_plane(pair_scratch, num_values, ix_batch & 1u, lane, bitpack_input);
        }
        else
        {
          input_size_bytes = kc_size_bytes(uncompressed_bytes, ix_batch);
          load_uint64_unpaired_contiguous(in, uncompressed_bytes, ix_batch, lane, shared_input, bitpack_input);
        }
        __syncwarp();
#pragma unroll
        for (uint32_t word = 0u; word < WORDS_PER_THREAD; ++word)
        {
          const uint32_t logical_word = lane * WORDS_PER_THREAD + word;
          shared_input[kibi_cascader::shared_input_word_index(logical_word)] = bitpack_input[word];
        }
        __syncwarp();
        plan = kibi_cascader::plan<uint32_t, PLANE_SEARCH_SPACE>(
          shared_input,
          bitpack_input,
          shared_metadata,
          shared_workspace,
          input_size_bytes
        );
        metadata_size_bytes = get_serialized_metadata_size_bytes<PLANE_SEARCH_SPACE>(shared_metadata);
        kc_buffer_size_bytes = KC_BUFFER_PREFIX_BYTES + metadata_size_bytes + plan.payload_size_bytes;
      }

      if (lane == 0u)
      {
        storage.sizes[warp_id] = kc_buffer_size_bytes;
      }
      __syncthreads();

      if (threadIdx.x == 0u)
      {
        storage.tile_kc_buffers_bytes = thread_exclusive_scan_kc_buffer_sizes(storage.sizes);
      }
      __syncthreads();

      const uint32_t tile_kc_buffers_bytes = storage.tile_kc_buffers_bytes;
      const uint32_t n = min(WARPS_PER_CTA, num_kc_batches - tile);

      if (is_batch_in_bounds)
      {
        const uint32_t kc_buffer_offset = n * KC_BUFFER_OFFSET_BYTES + storage.sizes[warp_id];
        uint8_t *kc_buffer = storage.scratch + kc_buffer_offset;

        if (lane == 0u)
        {
          store_unaligned<uint16_t>(
            storage.scratch + warp_id * KC_BUFFER_OFFSET_BYTES,
            static_cast<uint16_t>(kc_buffer_offset)
          );
          kc_buffer[0u] = static_cast<uint8_t>(metadata_size_bytes);
        }

        serialize_u32_kc_metadata<PLANE_SEARCH_SPACE>(shared_metadata, kc_buffer + KC_BUFFER_PREFIX_BYTES);
        kibi_cascader::encode<uint32_t, PLANE_SEARCH_SPACE>(
          bitpack_input,
          shared_metadata,
          kc_buffer + KC_BUFFER_PREFIX_BYTES + metadata_size_bytes,
          plan
        );
      }

      if (threadIdx.x == 0u)
      {
        const uint32_t sub_unit_size = n * KC_BUFFER_OFFSET_BYTES + tile_kc_buffers_bytes;
        store_unaligned<uint16_t>(
          out + universal_header::HEADER_SIZE_BYTES + (tile / WARPS_PER_CTA) * sizeof(uint16_t),
          static_cast<uint16_t>(sub_unit_size)
        );
      }

      __syncthreads();
      const uint32_t sub_unit_size = n * KC_BUFFER_OFFSET_BYTES + tile_kc_buffers_bytes;
      copy_compressed_sub_unit(out + sub_unit_start, storage.scratch, sub_unit_size);
      __syncthreads();

      sub_unit_start += sub_unit_size;
    }

    if (threadIdx.x == 0u)
    {
      compressed_bytes = sub_unit_start;
    }
  }
}

template <typename T, kibi_cascader::kibi_cascader_optimization_search_space SearchSpace>
__launch_bounds__(THREADS_PER_CTA, 4) __global__ void compress_kernel(
  const void *const *__restrict__ device_uncompressed_chunk_ptrs,
  const size_t *__restrict__ device_uncompressed_chunk_bytes,
  void *const *__restrict__ device_compressed_chunk_ptrs,
  size_t *__restrict__ device_compressed_chunk_bytes,
  const nvcompType_t data_type,
  nvcompStatus_t *__restrict__ device_statuses
)
{
  static_assert(sizeof(CompressStorage<T, SearchSpace>) <= 18u * 1024u);
  const uint32_t chunk = blockIdx.x;
  const size_t input_bytes = device_uncompressed_chunk_bytes[chunk];
  if (input_bytes == 0u || input_bytes > nvcompCascadedCompressionMaxAllowedChunkSize || input_bytes % sizeof(T) != 0u)
  {
    if (threadIdx.x == 0u)
    {
      device_compressed_chunk_bytes[chunk] = 0u;
      if (device_statuses != nullptr)
      {
        device_statuses[chunk] = nvcompErrorInvalidValue;
      }
    }
    return;
  }
  __shared__ CompressStorage<T, SearchSpace> storage;
  __shared__ uint32_t compressed_bytes;
  compress_chunk<T, SearchSpace>(
    reinterpret_cast<const uint8_t *>(device_uncompressed_chunk_ptrs[chunk]),
    static_cast<uint32_t>(input_bytes),
    reinterpret_cast<uint8_t *>(device_compressed_chunk_ptrs[chunk]),
    data_type,
    storage,
    compressed_bytes
  );
  if (threadIdx.x == 0u)
  {
    device_compressed_chunk_bytes[chunk] = compressed_bytes;
    if (device_statuses != nullptr)
    {
      device_statuses[chunk] = compressed_bytes == 0u ? nvcompErrorNotSupported : nvcompSuccess;
    }
  }
}

template <typename T, kibi_cascader::kibi_cascader_optimization_search_space SearchSpace>
inline __device__ void decompress_chunk(
  const uint32_t chunk,
  const void *const *__restrict__ device_compressed_chunk_ptrs,
  const size_t *__restrict__ device_compressed_chunk_bytes,
  const size_t *__restrict__ device_uncompressed_buffer_bytes,
  size_t *__restrict__ device_uncompressed_chunk_bytes,
  void *const *__restrict__ device_uncompressed_chunk_ptrs,
  const nvcompType_t data_type,
  nvcompStatus_t *__restrict__ device_statuses,
  uint32_t *output_transpose,
  uint32_t *metadata_scratch,
  uint32_t &validation_failed
)
{
  static_assert(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>);
  const uint8_t *comp = reinterpret_cast<const uint8_t *>(device_compressed_chunk_ptrs[chunk]);

  if (device_compressed_chunk_bytes[chunk] < universal_header::HEADER_SIZE_BYTES ||
      !universal_header::is_cascaded_next_compressed(comp) || !universal_header::has_supported_preamble(comp) ||
      universal_header::get_mode(comp) != universal_header::CompressionMode::Symmetric ||
      universal_header::get_data_type(comp) != data_type)
  {
    if (threadIdx.x == 0u)
    {
      if (device_statuses != nullptr)
      {
        device_statuses[chunk] = nvcompErrorCannotDecompress;
      }
      device_uncompressed_chunk_bytes[chunk] = 0u;
    }
    return;
  }

  const uint32_t uncompressed_bytes = universal_header::get_uncompressed_size(comp);
  if (uncompressed_bytes == 0u || uncompressed_bytes > nvcompCascadedCompressionMaxAllowedChunkSize ||
      uncompressed_bytes % sizeof(T) != 0u)
  {
    if (threadIdx.x == 0u)
    {
      if (device_statuses != nullptr)
      {
        device_statuses[chunk] = nvcompErrorCannotDecompress;
      }
      device_uncompressed_chunk_bytes[chunk] = 0u;
    }
    return;
  }

  const uint32_t num_kc_batches = num_kc_batches_for(uncompressed_bytes);
  const uint32_t total_wds = total_words(uncompressed_bytes);
  const uint32_t num_sub = num_sub_units_for(num_kc_batches);

  if (device_compressed_chunk_bytes[chunk] < preamble_size(num_sub))
  {
    if (threadIdx.x == 0u)
    {
      if (device_statuses != nullptr)
      {
        device_statuses[chunk] = nvcompErrorCannotDecompress;
      }
      device_uncompressed_chunk_bytes[chunk] = 0u;
    }
    return;
  }

  uint8_t *out = reinterpret_cast<uint8_t *>(device_uncompressed_chunk_ptrs[chunk]);

  const bool output_too_small = uncompressed_bytes > device_uncompressed_buffer_bytes[chunk];
  const bool valid_chunk = !output_too_small && cta_validate_compressed_chunk<T, SearchSpace>(
                                                  comp,
                                                  device_compressed_chunk_bytes[chunk],
                                                  uncompressed_bytes,
                                                  validation_failed
                                                );
  if (!valid_chunk)
  {
    if (threadIdx.x == 0u)
    {
      if (device_statuses != nullptr)
      {
        device_statuses[chunk] = nvcompErrorCannotDecompress;
      }
      device_uncompressed_chunk_bytes[chunk] = 0u;
    }
    return;
  }

  if (threadIdx.x == 0u)
  {
    if (device_statuses != nullptr)
    {
      device_statuses[chunk] = nvcompSuccess;
    }
    device_uncompressed_chunk_bytes[chunk] = uncompressed_bytes;
  }
  __syncthreads();

  const uint32_t warp_id = threadIdx.x / WARP_SIZE;
  // lane_id() seems to increase register pressure here. TODO: investigate
  const uint32_t lane = threadIdx.x % WARP_SIZE;
  uint32_t *const shared_metadata = metadata_scratch + warp_id * kibi_cascader::MAX_KC_METADATA_WORDS;
  uint32_t *const shared_workspace = output_transpose + warp_id * OUTPUT_TRANSPOSE_WORDS_PER_WARP;

  uint32_t sub_unit_start = preamble_size(num_sub);
  if constexpr (std::is_same_v<T, uint64_t>)
  {
    constexpr auto PLANE_SEARCH_SPACE = SearchSpace;
    for (uint32_t tile = 0u; tile < num_kc_batches; tile += WARPS_PER_CTA)
    {
      const uint32_t ix_batch = tile + warp_id;
      const bool is_batch_in_bounds = ix_batch < num_kc_batches;
      const bool is_paired_batch = is_batch_in_bounds && ((ix_batch | 1u) < num_kc_batches);
      uint32_t decoded_output[WORDS_PER_THREAD]{};

      if (is_batch_in_bounds)
      {
        const uint32_t ix_local_batch = ix_batch - tile;
        const uint32_t kc_buffer_offset =
          load_unaligned<uint16_t>(comp + sub_unit_start + ix_local_batch * KC_BUFFER_OFFSET_BYTES);
        const uint8_t *const kc_buffer = comp + sub_unit_start + kc_buffer_offset;
        const uint32_t metadata_size_bytes = kc_buffer[0u];
        if (is_paired_batch)
        {
          const uint32_t num_values = uint64_pair_num_values(uncompressed_bytes, ix_batch & ~1u);
          deserialize_kc_metadata<uint32_t, PLANE_SEARCH_SPACE>(
            kc_buffer + KC_BUFFER_PREFIX_BYTES,
            shared_metadata,
            metadata_size_bytes
          );
          kibi_cascader::decode<uint32_t, PLANE_SEARCH_SPACE>(
            kc_buffer + KC_BUFFER_PREFIX_BYTES + metadata_size_bytes,
            decoded_output,
            shared_metadata,
            shared_workspace,
            num_values * sizeof(uint32_t)
          );
          uint32_t *const pair_scratch = output_transpose + (warp_id & ~1u) * OUTPUT_TRANSPOSE_WORDS_PER_WARP;
          stage_uint64_pair_plane(pair_scratch, num_values, ix_batch & 1u, lane, decoded_output);
        }
        else
        {
          const uint32_t output_size_bytes = kc_size_bytes(uncompressed_bytes, ix_batch);
          deserialize_kc_metadata<uint32_t, PLANE_SEARCH_SPACE>(
            kc_buffer + KC_BUFFER_PREFIX_BYTES,
            shared_metadata,
            metadata_size_bytes
          );
          kibi_cascader::decode<uint32_t, PLANE_SEARCH_SPACE>(
            kc_buffer + KC_BUFFER_PREFIX_BYTES + metadata_size_bytes,
            decoded_output,
            shared_metadata,
            shared_workspace,
            output_size_bytes
          );
          load_uint64_kc_from_contiguous(
            decoded_output,
            output_size_bytes / sizeof(uint64_t),
            lane,
            shared_workspace,
            decoded_output
          );
        }
      }

      __syncthreads();
      if (is_paired_batch)
      {
        const uint32_t num_values = uint64_pair_num_values(uncompressed_bytes, ix_batch & ~1u);
        const uint32_t *const pair_scratch = output_transpose + (warp_id & ~1u) * OUTPUT_TRANSPOSE_WORDS_PER_WARP;
        load_uint64_kc_from_pair(pair_scratch, num_values, ix_batch & 1u, lane, decoded_output);
      }
      __syncthreads();

      if (is_batch_in_bounds)
      {
        const uint32_t output_size_bytes = kc_size_bytes(uncompressed_bytes, ix_batch);
        const uint32_t na = num_active_for(kc_words(total_wds, ix_batch));
        // Caching this in a bool seems to increase register pressure. TODO: investigate
        if (output_size_bytes == KC_SIZE_BYTES)
        {
          store_full_kc_coalesced(
            out,
            ix_batch,
            lane,
            decoded_output,
            output_transpose + warp_id * OUTPUT_TRANSPOSE_WORDS_PER_WARP
          );
        }
        else
        {
          store_kc(out, uncompressed_bytes, ix_batch, lane, na, decoded_output);
        }
      }

      __syncthreads();
      const uint32_t k = tile / WARPS_PER_CTA;
      const uint32_t sub_unit_size = __shfl_sync(
        0xFFFFFFFFu,
        lane == 0u ? load_unaligned<uint16_t>(comp + universal_header::HEADER_SIZE_BYTES + k * sizeof(uint16_t)) : 0u,
        0u
      );
      sub_unit_start += sub_unit_size;
    }
  }
  else
  {
    const uint32_t num_warps = blockDim.x / WARP_SIZE;
    constexpr auto SEARCH_SPACE = SearchSpace;
    for (uint32_t ix_batch = warp_id; ix_batch < num_kc_batches; ix_batch += num_warps)
    {
      const uint32_t k = ix_batch / WARPS_PER_CTA;
      const uint32_t ix_local_batch = ix_batch % WARPS_PER_CTA;

      const uint32_t kc_buffer_offset =
        load_unaligned<uint16_t>(comp + sub_unit_start + ix_local_batch * KC_BUFFER_OFFSET_BYTES);
      const uint8_t *kc_buffer = comp + sub_unit_start + kc_buffer_offset;
      const uint32_t metadata_size_bytes = kc_buffer[0u];
      deserialize_kc_metadata<T, SEARCH_SPACE>(kc_buffer + KC_BUFFER_PREFIX_BYTES, shared_metadata, metadata_size_bytes);

      const uint32_t na = num_active_for(kc_words(total_wds, ix_batch));
      const uint32_t output_size_bytes = kc_size_bytes(uncompressed_bytes, ix_batch);
      uint32_t decoded_output[WORDS_PER_THREAD];
      kibi_cascader::decode<T, SEARCH_SPACE>(
        kc_buffer + KC_BUFFER_PREFIX_BYTES + metadata_size_bytes,
        decoded_output,
        shared_metadata,
        shared_workspace,
        output_size_bytes
      );

      // Caching this in a bool seems to increase register pressure. TODO: investigate
      if (output_size_bytes == KC_SIZE_BYTES)
      {
        store_full_kc_coalesced(
          out,
          ix_batch,
          lane,
          decoded_output,
          output_transpose + warp_id * OUTPUT_TRANSPOSE_WORDS_PER_WARP
        );
      }
      else
      {
        store_kc(out, uncompressed_bytes, ix_batch, lane, na, decoded_output);
      }

      const uint32_t sub_unit_size = __shfl_sync(
        0xFFFFFFFFu,
        lane == 0u ? load_unaligned<uint16_t>(comp + universal_header::HEADER_SIZE_BYTES + k * sizeof(uint16_t)) : 0u,
        0u
      );
      sub_unit_start += sub_unit_size;
    }
  }
}

template <typename T, kibi_cascader::kibi_cascader_optimization_search_space SearchSpace>
__global__ void decompress_kernel(
  const void *const *__restrict__ device_compressed_chunk_ptrs,
  const size_t *__restrict__ device_compressed_chunk_bytes,
  const size_t *__restrict__ device_uncompressed_buffer_bytes,
  size_t *__restrict__ device_uncompressed_chunk_bytes,
  void *const *__restrict__ device_uncompressed_chunk_ptrs,
  const nvcompType_t data_type,
  nvcompStatus_t *__restrict__ device_statuses
)
{
  __shared__ uint32_t output_transpose[WARPS_PER_CTA][OUTPUT_TRANSPOSE_WORDS_PER_WARP];
  __shared__ uint32_t metadata_scratch[WARPS_PER_CTA][kibi_cascader::MAX_KC_METADATA_WORDS];
  __shared__ uint32_t validation_failed;
  decompress_chunk<T, SearchSpace>(
    blockIdx.x,
    device_compressed_chunk_ptrs,
    device_compressed_chunk_bytes,
    device_uncompressed_buffer_bytes,
    device_uncompressed_chunk_bytes,
    device_uncompressed_chunk_ptrs,
    data_type,
    device_statuses,
    output_transpose[0u],
    metadata_scratch[0u],
    validation_failed
  );
}

template <typename T>
inline __device__ bool header_type_matches_width(const nvcompType_t header_type)
{
  if constexpr (std::is_same_v<T, uint32_t>)
  {
    return header_type == NVCOMP_TYPE_INT || header_type == NVCOMP_TYPE_UINT;
  }
  else
  {
    return header_type == NVCOMP_TYPE_LONGLONG || header_type == NVCOMP_TYPE_ULONGLONG;
  }
}

inline __device__ void set_terminal_chunk_failure(
  const uint32_t chunk,
  size_t *__restrict__ device_uncompressed_chunk_bytes,
  nvcompStatus_t *__restrict__ device_statuses
)
{
  if (threadIdx.x == 0u)
  {
    if (device_statuses != nullptr)
    {
      device_statuses[chunk] = nvcompErrorCannotDecompress;
    }
    device_uncompressed_chunk_bytes[chunk] = 0u;
  }
}

inline __device__ void set_terminal_empty_chunk_outputs(
  const uint32_t chunk,
  const size_t compressed_bytes,
  size_t *__restrict__ device_uncompressed_chunk_bytes,
  nvcompStatus_t *__restrict__ device_statuses
)
{
  if (threadIdx.x == 0u)
  {
    device_uncompressed_chunk_bytes[chunk] = 0u;
    if (device_statuses != nullptr)
    {
      device_statuses[chunk] = compressed_bytes == 0u ? nvcompSuccess : nvcompErrorCannotDecompress;
    }
  }
}

template <typename T>
inline __device__ void dispatch_decompress_by_search_space(
  const uint32_t search_space,
  const uint32_t chunk,
  const void *const *__restrict__ device_compressed_chunk_ptrs,
  const size_t *__restrict__ device_compressed_chunk_bytes,
  const size_t *__restrict__ device_uncompressed_buffer_bytes,
  size_t *__restrict__ device_uncompressed_chunk_bytes,
  void *const *__restrict__ device_uncompressed_chunk_ptrs,
  const nvcompType_t header_type,
  nvcompStatus_t *__restrict__ device_statuses,
  uint32_t *output_transpose,
  uint32_t *metadata_scratch,
  uint32_t &validation_failed
)
{
  switch (search_space)
  {
    case kibi_cascader::SEARCH_NONE:
      decompress_chunk<T, kibi_cascader::SEARCH_NONE>(
        chunk,
        device_compressed_chunk_ptrs,
        device_compressed_chunk_bytes,
        device_uncompressed_buffer_bytes,
        device_uncompressed_chunk_bytes,
        device_uncompressed_chunk_ptrs,
        header_type,
        device_statuses,
        output_transpose,
        metadata_scratch,
        validation_failed
      );
      return;
    case kibi_cascader::SEARCH_DELTA:
      decompress_chunk<T, kibi_cascader::SEARCH_DELTA>(
        chunk,
        device_compressed_chunk_ptrs,
        device_compressed_chunk_bytes,
        device_uncompressed_buffer_bytes,
        device_uncompressed_chunk_bytes,
        device_uncompressed_chunk_ptrs,
        header_type,
        device_statuses,
        output_transpose,
        metadata_scratch,
        validation_failed
      );
      return;
    case kibi_cascader::SEARCH_FOR:
      decompress_chunk<T, kibi_cascader::SEARCH_FOR>(
        chunk,
        device_compressed_chunk_ptrs,
        device_compressed_chunk_bytes,
        device_uncompressed_buffer_bytes,
        device_uncompressed_chunk_bytes,
        device_uncompressed_chunk_ptrs,
        header_type,
        device_statuses,
        output_transpose,
        metadata_scratch,
        validation_failed
      );
      return;
    case kibi_cascader::SEARCH_DELTA | kibi_cascader::SEARCH_FOR:
      decompress_chunk<T, kibi_cascader::SEARCH_DELTA | kibi_cascader::SEARCH_FOR>(
        chunk,
        device_compressed_chunk_ptrs,
        device_compressed_chunk_bytes,
        device_uncompressed_buffer_bytes,
        device_uncompressed_chunk_bytes,
        device_uncompressed_chunk_ptrs,
        header_type,
        device_statuses,
        output_transpose,
        metadata_scratch,
        validation_failed
      );
      return;
    case kibi_cascader::SEARCH_RLE:
      decompress_chunk<T, kibi_cascader::SEARCH_RLE>(
        chunk,
        device_compressed_chunk_ptrs,
        device_compressed_chunk_bytes,
        device_uncompressed_buffer_bytes,
        device_uncompressed_chunk_bytes,
        device_uncompressed_chunk_ptrs,
        header_type,
        device_statuses,
        output_transpose,
        metadata_scratch,
        validation_failed
      );
      return;
    case kibi_cascader::SEARCH_DELTA | kibi_cascader::SEARCH_RLE:
      decompress_chunk<T, kibi_cascader::SEARCH_DELTA | kibi_cascader::SEARCH_RLE>(
        chunk,
        device_compressed_chunk_ptrs,
        device_compressed_chunk_bytes,
        device_uncompressed_buffer_bytes,
        device_uncompressed_chunk_bytes,
        device_uncompressed_chunk_ptrs,
        header_type,
        device_statuses,
        output_transpose,
        metadata_scratch,
        validation_failed
      );
      return;
    case kibi_cascader::SEARCH_FOR | kibi_cascader::SEARCH_RLE:
      decompress_chunk<T, kibi_cascader::SEARCH_FOR | kibi_cascader::SEARCH_RLE>(
        chunk,
        device_compressed_chunk_ptrs,
        device_compressed_chunk_bytes,
        device_uncompressed_buffer_bytes,
        device_uncompressed_chunk_bytes,
        device_uncompressed_chunk_ptrs,
        header_type,
        device_statuses,
        output_transpose,
        metadata_scratch,
        validation_failed
      );
      return;
    case TERMINAL_CASCADED_SEARCH_ALL:
      decompress_chunk<T, TERMINAL_CASCADED_SEARCH_ALL>(
        chunk,
        device_compressed_chunk_ptrs,
        device_compressed_chunk_bytes,
        device_uncompressed_buffer_bytes,
        device_uncompressed_chunk_bytes,
        device_uncompressed_chunk_ptrs,
        header_type,
        device_statuses,
        output_transpose,
        metadata_scratch,
        validation_failed
      );
      return;
    default:
      set_terminal_chunk_failure(chunk, device_uncompressed_chunk_bytes, device_statuses);
      return;
  }
}

template <typename T, bool ERR_ON_MODE_MISMATCH>
inline __device__ void decompress_chunk_self_describing(
  const void *const *__restrict__ device_compressed_chunk_ptrs,
  const size_t *__restrict__ device_compressed_chunk_bytes,
  const size_t *__restrict__ device_uncompressed_buffer_bytes,
  size_t *__restrict__ device_uncompressed_chunk_bytes,
  void *const *__restrict__ device_uncompressed_chunk_ptrs,
  const nvcompType_t expected_data_type,
  nvcompStatus_t *__restrict__ device_statuses,
  uint32_t *output_transpose,
  uint32_t *metadata_scratch,
  uint32_t &validation_failed
)
{
  const uint32_t chunk = blockIdx.x;
  const size_t compressed_bytes = device_compressed_chunk_bytes[chunk];

  if (compressed_bytes < universal_header::HEADER_SIZE_BYTES)
  {
    set_terminal_empty_chunk_outputs(chunk, compressed_bytes, device_uncompressed_chunk_bytes, device_statuses);
    return;
  }

  const uint8_t *comp = reinterpret_cast<const uint8_t *>(device_compressed_chunk_ptrs[chunk]);
  if (!universal_header::is_cascaded_next_compressed(comp))
  {
    if constexpr (ERR_ON_MODE_MISMATCH)
    {
      set_terminal_chunk_failure(chunk, device_uncompressed_chunk_bytes, device_statuses);
    }
    return;
  }

  if (!universal_header::has_supported_preamble(comp))
  {
    set_terminal_chunk_failure(chunk, device_uncompressed_chunk_bytes, device_statuses);
    return;
  }

  const auto mode = universal_header::get_mode(comp);
  if (mode == universal_header::CompressionMode::Asymmetric)
  {
    if constexpr (ERR_ON_MODE_MISMATCH)
    {
      set_terminal_chunk_failure(chunk, device_uncompressed_chunk_bytes, device_statuses);
    }
    return;
  }
  if (mode != universal_header::CompressionMode::Symmetric)
  {
    set_terminal_chunk_failure(chunk, device_uncompressed_chunk_bytes, device_statuses);
    return;
  }

  const nvcompType_t header_type = universal_header::get_data_type(comp);
  if (expected_data_type != NVCOMP_TYPE_BITS && header_type != expected_data_type)
  {
    set_terminal_chunk_failure(chunk, device_uncompressed_chunk_bytes, device_statuses);
    return;
  }
  if (!header_type_matches_width<T>(header_type))
  {
    const bool supported_other_width = header_type == NVCOMP_TYPE_INT || header_type == NVCOMP_TYPE_UINT ||
                                       header_type == NVCOMP_TYPE_LONGLONG || header_type == NVCOMP_TYPE_ULONGLONG;
    if (supported_other_width)
    {
      return;
    }
    set_terminal_chunk_failure(chunk, device_uncompressed_chunk_bytes, device_statuses);
    return;
  }

  decompress_chunk<T, TERMINAL_CASCADED_SEARCH_ALL>(
    chunk,
    device_compressed_chunk_ptrs,
    device_compressed_chunk_bytes,
    device_uncompressed_buffer_bytes,
    device_uncompressed_chunk_bytes,
    device_uncompressed_chunk_ptrs,
    header_type,
    device_statuses,
    output_transpose,
    metadata_scratch,
    validation_failed
  );
}

template <typename T, bool ERR_ON_MODE_MISMATCH>
__global__ void self_dispatch_decompress_kernel(
  const void *const *__restrict__ device_compressed_chunk_ptrs,
  const size_t *__restrict__ device_compressed_chunk_bytes,
  const size_t *__restrict__ device_uncompressed_buffer_bytes,
  size_t *__restrict__ device_uncompressed_chunk_bytes,
  void *const *__restrict__ device_uncompressed_chunk_ptrs,
  const nvcompType_t expected_data_type,
  nvcompStatus_t *__restrict__ device_statuses
)
{
  __shared__ uint32_t output_transpose[WARPS_PER_CTA][OUTPUT_TRANSPOSE_WORDS_PER_WARP];
  __shared__ uint32_t metadata_scratch[WARPS_PER_CTA][kibi_cascader::MAX_KC_METADATA_WORDS];
  __shared__ uint32_t validation_failed;
  if (threadIdx.x == 0u)
  {
    validation_failed = 0u;
  }
  __syncthreads();
  decompress_chunk_self_describing<T, ERR_ON_MODE_MISMATCH>(
    device_compressed_chunk_ptrs,
    device_compressed_chunk_bytes,
    device_uncompressed_buffer_bytes,
    device_uncompressed_chunk_bytes,
    device_uncompressed_chunk_ptrs,
    expected_data_type,
    device_statuses,
    output_transpose[0u],
    metadata_scratch[0u],
    validation_failed
  );
}

} // namespace nvcomp::cascaded::next::terminal_cascaded
