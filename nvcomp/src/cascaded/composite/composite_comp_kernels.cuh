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

#include <cuda/std/utility>

#include <type_traits>

#include "cascaded/common/cascaded_utils.cuh"
#include "cascaded/composite/composite_batch_optimizer.cuh"
#include "cascaded/modules/bitpack.cuh"
#include "cascaded/modules/delta.cuh"
#include "cascaded/modules/rle.cuh"
#include "cascaded/universal/universal_header.cuh"
#include "common.h"
#include "composite_constants.cuh"
#include "composite_types.cuh"
#include "composite_utils.cuh"
#include "CudaUtils.h"
#include "exception.hpp"
#include "lowlevel/Check.h"
#include "nvcomp.h"
#include "nvcomp.hpp"
#include "nvcomp/cascaded.h"
#include "type_macros.h"

namespace nvcomp::cascaded::composite
{

/**
 * Write a buffer with a single threadblock, optionally with bitpacking.
 *
 * For the current implementation, this function is used to write a layer
 * output from the shared memory to the global memory.
 *
 * @param[in] input Pointer to the uncompressed input buffer.
 * @param[in] num_elements Number of input elements.
 * @param[out] output Pointer to the output buffer.
 * @param[in] output_limit Pointer to the location just past the end of the
 * output buffer. If the output need to overflow beyond this pointer,
 * `BlockIOStatus::out_of_bound` will be returned. This argument must belong to
 * the same array as \p output. Otherwise, the behavior is undefined.
 * @param[out] out_bytes Number of bytes written to \p output. This argument
 * should be unique to each thread.
 * @param[in] temp_storage Temporary storage used for holding the bitpacked
 * data. Users should guarantee this storage has enough space for the combined
 * of the bitpacked metadata and the bitpacked data.
 * @param[in] use_bp Whether bitpacking should be used.
 * @param[in] data_is_deltas Whether the data pointed to by input is deltas.
 *
 * All threads in the block must call this function. The caller must synchronize
 * the block after it returns before reusing \p input or \p temp_storage.
 */
template <typename data_type, typename size_type, int threadblock_size>
__device__ BlockIOStatus block_write(
  const data_type *input,
  size_type num_elements,
  uint32_t *output,
  const uint32_t *output_limit,
  size_type *out_bytes,
  uint32_t *temp_storage,
  bool use_bp,
  bool data_is_deltas
)
{
  const uint32_t *source = nullptr;

  if (use_bp)
  {
    modules::block_bitpack<data_type, size_type, threadblock_size>(
      input,
      num_elements,
      data_is_deltas,
      temp_storage,
      out_bytes
    );
    __syncthreads();
    source = temp_storage;
  }
  else
  {
    *out_bytes = num_elements * sizeof(data_type);
    source = reinterpret_cast<const uint32_t *>(input);
  }

  const size_type padded_out_bytes = roundUpTo(*out_bytes, sizeof(uint32_t));
  if (output + padded_out_bytes / sizeof(uint32_t) > output_limit)
  {
    return BlockIOStatus::out_of_bound;
  }

  for (int element_idx = threadIdx.x; element_idx < padded_out_bytes / sizeof(uint32_t); element_idx += blockDim.x)
  {
    output[element_idx] = source[element_idx];
  }

  return BlockIOStatus::success;
}

template <typename data_type, typename size_type, int threadblock_size>
__device__ size_type block_encoded_size(
  const data_type *const input,
  const size_type num_elements,
  uint32_t *const bitpacking_metadata,
  const bool use_bitpacking,
  const bool data_is_deltas
)
{
  size_type encoded_bytes;
  if (use_bitpacking)
  {
    auto frame_of_reference = reinterpret_cast<data_type *>(bitpacking_metadata);
    uint32_t *const bitwidth_ptr = roundUpToAlignment<uint32_t>(frame_of_reference + 1);
    if (std::is_signed_v<data_type> || data_is_deltas)
    {
      using signed_data_type = std::make_signed_t<data_type>;
      modules::get_for_bitwidth<signed_data_type, size_type, threadblock_size>(
        reinterpret_cast<const signed_data_type *>(input),
        num_elements,
        reinterpret_cast<signed_data_type *>(frame_of_reference),
        bitwidth_ptr
      );
    }
    else
    {
      modules::get_for_bitwidth<data_type, size_type, threadblock_size>(
        input,
        num_elements,
        frame_of_reference,
        bitwidth_ptr
      );
    }
    __syncthreads();

    const uint32_t bitwidth = *bitwidth_ptr >> 16;
    // The frame of reference is followed by a 4B bitwidth/element-count word.
    // Padding aligns that word and the subsequent packed uint32_t data to both
    // 4B and data_type, including when data_type is only 1B or 2B.
    encoded_bytes = roundUpTo(sizeof(data_type) + sizeof(uint32_t), max(size_t{4}, sizeof(data_type))) +
                    roundUpDiv(num_elements * bitwidth, sizeof(uint32_t) * modules::num_bits_per_byte) *
                      sizeof(uint32_t);
    __syncthreads();
  }
  else
  {
    encoded_bytes = num_elements * sizeof(data_type);
  }
  return encoded_bytes;
}

template <typename data_type, typename size_type>
struct ChunkEncodeState
{
  data_type *shared_input_buffer;
  data_type *shared_output_buffer;
  uint32_t *current_output_ptr;
  size_type num_elements;
  size_type best_final_bytes;
  int consecutive_rejections;
  uint32_t applied_stages;
  bool current_is_signed;
  bool final_output_written;
};

template <typename data_type, typename size_type>
struct StageCandidate
{
  data_type *input;
  data_type *output;
  uint32_t *next_stage_ptr;
  size_type num_elements;
  size_type final_bytes;
};

struct SizedFieldLayout
{
  uint32_t *size_ptr;
  uint32_t *payload_ptr;
  uint32_t *next_ptr;
};

__device__ __forceinline__ SizedFieldLayout reserve_sized_field(uint32_t *const base, const size_t payload_bytes)
{
  uint32_t *const payload = base + 1;
  return {base, payload, payload + roundUpDiv(payload_bytes, sizeof(uint32_t))};
}

template <typename data_type, typename size_type>
__device__ __forceinline__ void accept_stage(
  ChunkEncodeState<data_type, size_type> &state,
  const StageCandidate<data_type, size_type> &candidate,
  const uint32_t stage_mask
)
{
  state.applied_stages |= stage_mask;
  state.consecutive_rejections = 0;
  state.current_output_ptr = candidate.next_stage_ptr;
  state.shared_input_buffer = candidate.input;
  state.shared_output_buffer = candidate.output;
  state.num_elements = candidate.num_elements;
  state.best_final_bytes = candidate.final_bytes;
  state.final_output_written = false;
}

/**
 * @brief Batched cascaded compression kernel.
 *
 * All cascaded compression layers are fused together in this single kernel.
 *
 * @tparam data_type Data type of each element.
 * @tparam threadblock_size Number of threads in a threadblock. This argument
 * must match the configuration specified when launching this kernel.
 * @tparam chunk_size Input size that is loaded into shared memory at a time.
 * This argument must be a multiple of the size of `data_type`.
 *
 * @param[in] batch_size Number of partitions to compress.
 * @param[in] uncompressed_data Array with size \p batch_size of pointers to
 * input uncompressed partitions.
 * @param[in] uncompressed_bytes Sizes of input uncompressed partitions in
 * bytes.
 * @param[out] compressed_data Array with size \p batch_size of output locations
 * of the compressed buffers. Each compressed buffer must start at a location
 * aligned with both 4B and the data type.
 * @param[out] compressed_bytes Number of bytes decompressed of all partitions.
 * @param[in] comp_opts Compression format used.
 */
template <typename data_type, typename size_type, int threadblock_size, int chunk_size = default_chunk_size>
__device__ void do_composite_compression(
  int batch_size,
  int batch_start,
  int batch_stride,
  const data_type *const *uncompressed_data,
  const size_type *uncompressed_bytes,
  void *const *compressed_data,
  size_type *compressed_bytes,
  AdaptiveCompressionOptions comp_opts
)
{
  using run_type = uint16_t;
  constexpr int CHUNK_NUM_ELEMENTS = chunk_size / sizeof(data_type);

  // We need to guarantee the `CHUNK_NUM_ELEMENTS` is smaller than the limit of
  // uint16_t for two reasons:
  // 1. We use uint16_t to represent run counts.
  // 2. We use 16 bits to represent number of elements in the bitpacking layer.
  assert(CHUNK_NUM_ELEMENTS < 65536);

  // `shared_element_storage_0` and `shared_element_storage_1` are shared memory
  // storage used for holding input and output of the current layer.
  // `shared_storage_type` is used to make sure the storage is both 4B and
  // data_type aligned.
  typedef larger_t<data_type, uint32_t> shared_storage_type;

  // Allocate `4 + sizeof(data_type)` in addition to `chunk_size` to accommodate
  // bitpacking metadata. The extra `sizeof(data_type)` is used for holding
  // frame of reference, while the extra 4B is used to hold the bitwidth and the
  // number of elements.
  constexpr size_t storage_num_elements = roundUpDiv(chunk_size + 4 + sizeof(data_type), sizeof(shared_storage_type));

  __shared__ shared_storage_type shared_element_storage_0[storage_num_elements];
  data_type *shared_element_buffer_0 = reinterpret_cast<data_type *>(shared_element_storage_0);

  __shared__ shared_storage_type shared_element_storage_1[storage_num_elements];
  data_type *shared_element_buffer_1 = reinterpret_cast<data_type *>(shared_element_storage_1);

  constexpr int shared_counts_storage_size = roundUpTo(CHUNK_NUM_ELEMENTS * sizeof(run_type), 4);
  // Shared memory buffer used by RLE for holding run counts
  __shared__ uint32_t shared_count_buffer[shared_counts_storage_size / sizeof(uint32_t)];
  // Temporary storage used by RLE
  // Allocate extra 8B for holding bitpacking metadata. The metadata consists of
  // frame of reference (2B) and bit-width and number of elements (4B). So, in
  // total the metadata needs 6B. 8B is allocated for 4B alignment.
  __shared__ uint32_t shared_tmp_buffer[shared_counts_storage_size / sizeof(uint32_t) + 8 / sizeof(uint32_t)];

  // Number of output elements for the RLE layer
  __shared__ size_type num_outputs;
  size_type out_bytes;

  for (int partition_idx = batch_start; partition_idx < batch_size; partition_idx += batch_stride)
  {
    const auto input_buffer = uncompressed_data[partition_idx];
    const auto input_bytes = uncompressed_bytes[partition_idx];
    assert(input_bytes <= UINT32_MAX);
    const size_type num_input_elements = input_bytes / sizeof(data_type);

    if (input_buffer == nullptr || input_bytes == 0)
    {
      if (threadIdx.x == 0)
      {
        // Zero compressed bytes fully represents empty input; no output buffer
        // or header is required.
        compressed_bytes[partition_idx] = 0;
      }
      continue;
    }

    auto output_buffer = static_cast<uint32_t *>(compressed_data[partition_idx]);
    // `output_limit` points to the end of the output compressed buffer of the
    // current partition. The size of the compressed buffer should be at least
    // 8B larger than the input uncompressed buffer. It needs to be 8B larger
    // because in the fallback path, the compressed buffer still needs to store
    // the metadata. It is users responsibility to guarantee this requirement.
    uint32_t *output_limit = output_buffer + roundUpDiv(universal_header::HEADER_SIZE_BYTES, sizeof(uint32_t)) +
                             roundUpDiv(input_bytes, sizeof(uint32_t));

    // Global flag on whether we will compress the current partition. If
    // compressed size is larger than the uncompressed size (i.e. compression
    // ratio < 1), we will use the fallback path of directly copying from the
    // input buffer to the compressed buffer, and set this flag to false.
    bool use_compression = comp_opts.num_RLEs != 0 || comp_opts.num_deltas != 0 || comp_opts.use_bp != 0;

    // Pointer to the first chunk of the current partition
    auto current_output_ptr = reinterpret_cast<uint32_t *>(
      roundUpToAlignment<data_type>(output_buffer + roundUpDiv(universal_header::HEADER_SIZE_BYTES, sizeof(uint32_t)))
    );

    const int num_chunks = roundUpDiv(num_input_elements, CHUNK_NUM_ELEMENTS);

    for (int chunk_idx = 0; chunk_idx < num_chunks && use_compression; chunk_idx++)
    {
      // Save a pointer at the start of the chunk
      uint32_t *chunk_start_ptr = current_output_ptr;

      // Adaptive chunks begin with a fixed two-word prefix containing the
      // packed chunk size / accepted-round count and final stream size.
      current_output_ptr += ADAPTIVE_CHUNK_PREFIX_SIZE / sizeof(uint32_t);

      auto input_buffer_current_chunk = input_buffer + CHUNK_NUM_ELEMENTS * chunk_idx;
      size_type num_elements_current_chunk =
        min(num_input_elements - chunk_idx * CHUNK_NUM_ELEMENTS, static_cast<size_type>(CHUNK_NUM_ELEMENTS));

      // Threadblock collectively loads current chunk from input uncompressed
      // buffer to shared memory buffer
      for (int element_idx = threadIdx.x; element_idx < num_elements_current_chunk; element_idx += blockDim.x)
      {
        shared_element_buffer_0[element_idx] = input_buffer_current_chunk[element_idx];
      }
      __syncthreads();

      const bool use_rle = comp_opts.num_RLEs > 0;
      const bool use_delta = comp_opts.num_deltas > 0;
      const uint32_t max_stages = min(comp_opts.num_RLEs + comp_opts.num_deltas, ADAPTIVE_MAX_NUM_STAGES);
      ChunkEncodeState<data_type, size_type> state{
        shared_element_buffer_0,
        shared_element_buffer_1,
        current_output_ptr,
        num_elements_current_chunk,
        0,
        0,
        0,
        false,
        false
      };
      if (max_stages > 0)
      {
        state.best_final_bytes = block_encoded_size<data_type, size_type, threadblock_size>(
          state.shared_input_buffer,
          state.num_elements,
          shared_tmp_buffer,
          comp_opts.use_bp,
          false
        );
      }

      uint32_t rle_stage_mask = 0u;
      for (uint32_t stage_idx = 0; stage_idx < max_stages && state.num_elements > 0; ++stage_idx)
      {
        auto best_final_ptr = reinterpret_cast<uint32_t *>(roundUpToAlignment<data_type>(state.current_output_ptr));
        const auto final_output_status = block_write<data_type, size_type, threadblock_size>(
          state.shared_input_buffer,
          state.num_elements,
          best_final_ptr,
          output_limit,
          &state.best_final_bytes,
          reinterpret_cast<uint32_t *>(state.shared_output_buffer),
          comp_opts.use_bp,
          state.current_is_signed
        );
        __syncthreads();
        state.final_output_written = final_output_status == BlockIOStatus::success;

        const bool stage_is_rle = use_rle && (!use_delta || (stage_idx & 1) == 0);
        if (stage_is_rle)
        {
          rle_stage_mask |= get_adaptive_stage_mask(stage_idx);
        }
        size_type candidate_elements = state.num_elements;
        data_type *candidate_input = state.shared_input_buffer;
        data_type *candidate_output = state.shared_output_buffer;

        if (stage_is_rle)
        {
          size_type candidate_count_bytes;
          modules::block_rle_compress<data_type, size_type, run_type, threadblock_size>(
            candidate_input,
            candidate_elements,
            candidate_output,
            reinterpret_cast<run_type *>(shared_count_buffer),
            &num_outputs,
            reinterpret_cast<run_type *>(shared_tmp_buffer)
          );
          __syncthreads();
          candidate_elements = num_outputs;
          assert(candidate_elements <= (storage_num_elements * sizeof(shared_storage_type)) / sizeof(data_type));
          cuda::std::swap(candidate_input, candidate_output);
          candidate_count_bytes = block_encoded_size<run_type, size_type, threadblock_size>(
            reinterpret_cast<run_type *>(shared_count_buffer),
            candidate_elements,
            shared_tmp_buffer,
            comp_opts.use_bp,
            false
          );
          size_type candidate_final_bytes;
          candidate_final_bytes = block_encoded_size<data_type, size_type, threadblock_size>(
            candidate_input,
            candidate_elements,
            shared_tmp_buffer,
            comp_opts.use_bp,
            state.current_is_signed
          );
          const auto counts = reserve_sized_field(state.current_output_ptr, candidate_count_bytes);
          const auto candidate_final_ptr = reinterpret_cast<uint32_t *>(roundUpToAlignment<data_type>(counts.next_ptr));
          const size_type best_storage_bytes = get_adaptive_record_size<data_type, size_type>(
            reinterpret_cast<uintptr_t>(state.current_output_ptr),
            reinterpret_cast<uintptr_t>(best_final_ptr),
            state.best_final_bytes
          );
          const size_type candidate_storage_bytes = get_adaptive_record_size<data_type, size_type>(
            reinterpret_cast<uintptr_t>(state.current_output_ptr),
            reinterpret_cast<uintptr_t>(candidate_final_ptr),
            candidate_final_bytes
          );
          if (candidate_storage_bytes >= best_storage_bytes)
          {
            if (++state.consecutive_rejections == 2)
            {
              break;
            }
            continue;
          }

          const auto count_write_status = block_write<run_type, size_type, threadblock_size>(
            reinterpret_cast<run_type *>(shared_count_buffer),
            candidate_elements,
            counts.payload_ptr,
            output_limit,
            &out_bytes,
            shared_tmp_buffer,
            comp_opts.use_bp,
            false
          );
          __syncthreads();
          if (count_write_status != BlockIOStatus::success)
          {
            use_compression = false;
            break;
          }
          assert(out_bytes == candidate_count_bytes);
          if (threadIdx.x == 0)
          {
            *counts.size_ptr = static_cast<uint32_t>(out_bytes);
          }
          accept_stage(
            state,
            StageCandidate<data_type, size_type>{
              candidate_input,
              candidate_output,
              counts.next_ptr,
              candidate_elements,
              candidate_final_bytes
            },
            get_adaptive_stage_mask(stage_idx)
          );
          continue;
        }

        data_type candidate_delta_first{};
        if (threadIdx.x == 0)
        {
          candidate_delta_first = candidate_input[0];
        }
        modules::block_delta_compress<data_type, size_type>(candidate_input, candidate_elements, candidate_output);
        __syncthreads();
        cuda::std::swap(candidate_input, candidate_output);
        --candidate_elements;

        size_type candidate_final_bytes;
        candidate_final_bytes = block_encoded_size<data_type, size_type, threadblock_size>(
          candidate_input,
          candidate_elements,
          shared_tmp_buffer,
          comp_opts.use_bp,
          true
        );
        uint32_t *const candidate_delta_ptr = state.current_output_ptr;
        auto candidate_stage_ptr = candidate_delta_ptr + roundUpDiv(sizeof(data_type), sizeof(uint32_t));
        const auto candidate_final_ptr =
          reinterpret_cast<uint32_t *>(roundUpToAlignment<data_type>(candidate_stage_ptr));
        const size_type best_storage_bytes = get_adaptive_record_size<data_type, size_type>(
          reinterpret_cast<uintptr_t>(state.current_output_ptr),
          reinterpret_cast<uintptr_t>(best_final_ptr),
          state.best_final_bytes
        );
        const size_type candidate_storage_bytes = get_adaptive_record_size<data_type, size_type>(
          reinterpret_cast<uintptr_t>(state.current_output_ptr),
          reinterpret_cast<uintptr_t>(candidate_final_ptr),
          candidate_final_bytes
        );
        if (candidate_storage_bytes < best_storage_bytes)
        {
          if (threadIdx.x == 0)
          {
            serialize_value(candidate_delta_ptr, candidate_delta_first);
          }
          state.current_is_signed = true;
          accept_stage(
            state,
            StageCandidate<data_type, size_type>{
              candidate_input,
              candidate_output,
              candidate_stage_ptr,
              candidate_elements,
              candidate_final_bytes
            },
            get_adaptive_stage_mask(stage_idx)
          );
          continue;
        }

        ++state.consecutive_rejections;
        const bool can_look_ahead = use_rle && stage_idx + 1 < max_stages && candidate_elements > 0;
        if (!can_look_ahead)
        {
          if (state.consecutive_rejections == 2)
          {
            break;
          }
          continue;
        }
        rle_stage_mask |= get_adaptive_stage_mask(stage_idx + 1);

        // A locally losing Delta can expose long runs. Evaluate the following
        // RLE exactly before rejecting both stages; if the pair also loses, the
        // two-rejection rule terminates the search, so the overwritten working
        // buffer is no longer needed.
        modules::block_rle_compress<data_type, size_type, run_type, threadblock_size>(
          candidate_input,
          candidate_elements,
          candidate_output,
          reinterpret_cast<run_type *>(shared_count_buffer),
          &num_outputs,
          reinterpret_cast<run_type *>(shared_tmp_buffer)
        );
        __syncthreads();
        candidate_elements = num_outputs;
        cuda::std::swap(candidate_input, candidate_output);

        size_type candidate_count_bytes;
        candidate_count_bytes = block_encoded_size<run_type, size_type, threadblock_size>(
          reinterpret_cast<run_type *>(shared_count_buffer),
          candidate_elements,
          shared_tmp_buffer,
          comp_opts.use_bp,
          false
        );
        candidate_final_bytes = block_encoded_size<data_type, size_type, threadblock_size>(
          candidate_input,
          candidate_elements,
          shared_tmp_buffer,
          comp_opts.use_bp,
          true
        );
        const auto counts = reserve_sized_field(candidate_stage_ptr, candidate_count_bytes);
        candidate_stage_ptr = counts.next_ptr;
        const auto pair_final_ptr = reinterpret_cast<uint32_t *>(roundUpToAlignment<data_type>(candidate_stage_ptr));
        const size_type pair_storage_bytes = get_adaptive_record_size<data_type, size_type>(
          reinterpret_cast<uintptr_t>(state.current_output_ptr),
          reinterpret_cast<uintptr_t>(pair_final_ptr),
          candidate_final_bytes
        );
        if (pair_storage_bytes >= best_storage_bytes)
        {
          if (!state.final_output_written)
          {
            use_compression = false;
          }
          break;
        }

        const auto count_write_status = block_write<run_type, size_type, threadblock_size>(
          reinterpret_cast<run_type *>(shared_count_buffer),
          candidate_elements,
          counts.payload_ptr,
          output_limit,
          &out_bytes,
          shared_tmp_buffer,
          comp_opts.use_bp,
          false
        );
        __syncthreads();
        if (count_write_status != BlockIOStatus::success)
        {
          use_compression = false;
          break;
        }
        if (threadIdx.x == 0)
        {
          serialize_value(candidate_delta_ptr, candidate_delta_first);
          *counts.size_ptr = static_cast<uint32_t>(out_bytes);
        }
        state.current_is_signed = true;
        accept_stage(
          state,
          StageCandidate<data_type, size_type>{
            candidate_input,
            candidate_output,
            candidate_stage_ptr,
            candidate_elements,
            candidate_final_bytes
          },
          get_adaptive_stage_mask(stage_idx, 2)
        );
        ++stage_idx;
      }

      if (!use_compression)
      {
        break;
      }

      auto final_output_ptr = reinterpret_cast<uint32_t *>(roundUpToAlignment<data_type>(state.current_output_ptr));
      if (!state.final_output_written)
      {
        const auto final_write_status = block_write<data_type, size_type, threadblock_size>(
          state.shared_input_buffer,
          state.num_elements,
          final_output_ptr,
          output_limit,
          &out_bytes,
          reinterpret_cast<uint32_t *>(state.shared_output_buffer),
          comp_opts.use_bp,
          state.current_is_signed
        );
        __syncthreads();
        if (final_write_status != BlockIOStatus::success)
        {
          use_compression = false;
          break;
        }
      }
      if (state.final_output_written)
      {
        out_bytes = state.best_final_bytes;
      }
      current_output_ptr = final_output_ptr + roundUpDiv(out_bytes, 4);
      current_output_ptr = reinterpret_cast<uint32_t *>(roundUpToAlignment<data_type>(current_output_ptr));

      // Flush the fixed chunk prefix. Per-round metadata is already stored
      // inline with each accepted stage.
      if (threadIdx.x == 0)
      {
        const uint32_t chunk_output_size = reinterpret_cast<uintptr_t>(current_output_ptr) -
                                           reinterpret_cast<uintptr_t>(chunk_start_ptr);
        chunk_start_ptr[0] =
          pack_adaptive_chunk_size(chunk_output_size, state.applied_stages, state.applied_stages & rle_stage_mask);
        chunk_start_ptr[1] = static_cast<uint32_t>(out_bytes);
      }

      __syncthreads();
    }

    // Reserve the exact uncompressed-size encoding for the raw fallback so
    // decompression can distinguish it without another universal-header flag.
    if (use_compression && current_output_ptr >= output_limit)
    {
      use_compression = false;
    }

    if (!use_compression)
    {
      // Compressed size is larger than uncompressed size, so we fallback to
      // directly copy input array to output
      data_type *direct_output_buffer =
        roundUpToAlignment<data_type>(output_buffer + universal_header::HEADER_SIZE_WORDS);
      for (int element_idx = threadIdx.x; element_idx < num_input_elements; element_idx += blockDim.x)
      {
        direct_output_buffer[element_idx] = input_buffer[element_idx];
      }
    }

    // Save the metadata of the current partition
    if (threadIdx.x == 0)
    {
      auto partition_metadata_ptr = reinterpret_cast<uint8_t *>(output_buffer);

      const uint32_t uncompressed_size_bytes = static_cast<uint32_t>(num_input_elements * sizeof(data_type));
      if (use_compression)
      {
        universal_header::write_cascaded_next_header(
          partition_metadata_ptr,
          d_TypeOf<data_type>(),
          uncompressed_size_bytes,
          universal_header::CompressionMode::Asymmetric,
          universal_header::TerminalCodec::CascadedBitpack
        );

        compressed_bytes[partition_idx] = reinterpret_cast<uintptr_t>(current_output_ptr) -
                                          reinterpret_cast<uintptr_t>(output_buffer);
      }
      else
      {
        universal_header::write_dummy_header(
          partition_metadata_ptr,
          cascaded::d_TypeOf<data_type>(),
          uncompressed_size_bytes,
          universal_header::CompressionMode::Asymmetric
        );

        compressed_bytes[partition_idx] = roundUpTo(universal_header::HEADER_SIZE_BYTES, sizeof(data_type)) +
                                          roundUpTo(num_input_elements * sizeof(data_type), 4);
      }
    }
  }
}

/**
 * @brief Batched cascaded compression kernel.
 *
 * All cascaded compression layers are fused together in this single kernel.
 *
 * @tparam data_type Data type of each element.
 * @tparam threadblock_size Number of threads in a threadblock. This argument
 * must match the configuration specified when launching this kernel.
 * @tparam chunk_size Input size that is loaded into shared memory at a time.
 * This argument must be a multiple of the size of `data_type`.
 *
 * @param[in] batch_size Number of partitions to compress.
 * @param[in] uncompressed_data Array with size \p batch_size of pointers to
 * input uncompressed partitions.
 * @param[in] uncompressed_bytes Sizes of input uncompressed partitions in
 * bytes.
 * @param[out] compressed_data Array with size \p batch_size of output locations
 * of the compressed buffers. Each compressed buffer must start at a location
 * aligned with both 4B and the data type.
 * @param[out] compressed_bytes Number of bytes decompressed of all partitions.
 * @param[in] comp_opts Compression format used.
 */
template <typename data_type, typename size_type, int threadblock_size, int chunk_size = default_chunk_size>
__global__ void composite_compression_kernel(
  int batch_size,
  const data_type *const *uncompressed_data,
  const size_type *uncompressed_bytes,
  void *const *compressed_data,
  size_type *compressed_bytes,
  AdaptiveCompressionOptions comp_opts
)
{
  do_composite_compression<data_type, size_type, threadblock_size, chunk_size>(
    batch_size,
    blockIdx.x,
    gridDim.x,
    uncompressed_data,
    uncompressed_bytes,
    compressed_data,
    compressed_bytes,
    comp_opts
  );
}

template <typename data_type>
void composite_batched_compression_typed(
  const AdaptiveCompressionOptions format_opts,
  const void *const *device_uncompressed_chunk_ptrs,
  const size_t *device_uncompressed_bytes,
  size_t batch_size,
  void *const *device_compressed_ptrs,
  size_t *device_compressed_bytes,
  nvcompStatus_t *device_statuses,
  cudaStream_t stream
)
{
  constexpr int threadblock_size = composite_compress_threadblock_size;
  composite_compression_kernel<data_type, size_t, threadblock_size>
    <<<cuda_dim_cast(batch_size), threadblock_size, 0, stream>>>(
      narrow_cast<int>(batch_size),
      reinterpret_cast<const data_type *const *>(device_uncompressed_chunk_ptrs),
      device_uncompressed_bytes,
      device_compressed_ptrs,
      device_compressed_bytes,
      format_opts
    );
  CUDA_CHECK(cudaGetLastError());

  // mark compression successful
  try_clear_device_statuses(batch_size, device_statuses, stream);
}

nvcompStatus_t compressAsync(
  const void *const *device_uncompressed_chunk_ptrs,
  const size_t *device_uncompressed_chunk_bytes,
  size_t num_chunks,

  void *const *device_compressed_chunk_ptrs,
  size_t *device_compressed_chunk_bytes,

  const nvcompBatchedCascadedCompressOpts_t comp_opts,
  nvcompStatus_t *device_statuses,
  cudaStream_t stream
)
{
  if (num_chunks == 0)
  {
    // The batch size doubles as the grid size, and an empty grid is an invalid
    // launch configuration. There is nothing to compress either way.
    return nvcompSuccess;
  }

  try
  {
    const AdaptiveStageCounts stage_counts =
      map_adaptive_stage_counts(comp_opts.compression_level, comp_opts.fine_grained_encoding_flags);
    const AdaptiveCompressionOptions adaptive_opts{stage_counts.num_RLEs, stage_counts.num_deltas, true};
    NVCOMP_TYPE_ONE_SWITCH(
      comp_opts.common_opts.data_type,
      composite_batched_compression_typed,
      adaptive_opts,
      device_uncompressed_chunk_ptrs,
      device_uncompressed_chunk_bytes,
      num_chunks,
      device_compressed_chunk_ptrs,
      device_compressed_chunk_bytes,
      device_statuses,
      stream
    );
  }
  catch (const std::exception &e)
  {
    LOG_ERROR("{}", e.what());
    return Check::exception_to_error(e, "nvcompBatchedCascadedCompressAsync()");
  }

  return nvcompSuccess;
}

} // namespace nvcomp::cascaded::composite
