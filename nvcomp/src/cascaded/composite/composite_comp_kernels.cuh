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

#include "cascaded/common/cascaded_utils.cuh"
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

using nvcomp::Check;
using nvcomp::CudaUtils;
using nvcomp::NVCompException;
using nvcomp::roundUpDiv;
using nvcomp::roundUpTo;
using nvcomp::roundUpToAlignment;

namespace composite
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
  nvcompBatchedCascadedCompressOpts_t comp_opts
)
{
  using run_type = uint16_t;
  constexpr int chunk_num_elements = chunk_size / sizeof(data_type);

  // We need to guarantee the `chunk_num_elements` is smaller than the limit of
  // uint16_t for two reasons:
  // 1. We use uint16_t to represent run counts.
  // 2. We use 16 bits to represent number of elements in the bitpacking layer.
  assert(chunk_num_elements < 65536);

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

  constexpr int shared_counts_storage_size = roundUpTo(chunk_num_elements * sizeof(run_type), 4);
  // Shared memory buffer used by RLE for holding run counts
  __shared__ uint32_t shared_count_buffer[shared_counts_storage_size / sizeof(uint32_t)];
  // Temporary storage used by RLE
  // Allocate extra 8B for holding bitpacking metadata. The metadata consists of
  // frame of reference (2B) and bit-width and number of elements (4B). So, in
  // total the metadata needs 6B. 8B is allocated for 4B alignment.
  __shared__ uint32_t shared_tmp_buffer[shared_counts_storage_size / sizeof(uint32_t) + 8 / sizeof(uint32_t)];

  // `chunk_metadata` is a shared-memory staging buffer for the metadata of the
  // current chunk before flushing it to global memory. Here we assume the chunk
  // metadata is at most 64B large.
  // This must be aligned to at least sizeof(data_type) to avoid delta_header
  // below being shifted depending on the alignment.
  __shared__ alignas(8) uint32_t chunk_metadata[max_chunk_metadata_size / sizeof(uint32_t)];
  const int chunk_metadata_size = get_chunk_metadata_size<data_type>(comp_opts.num_RLEs, comp_opts.num_deltas);
  assert(chunk_metadata_size <= max_chunk_metadata_size);

  // Pointer to the delta section of chunk metadata in shared memory. Padding
  // will be added if necessary to make the pointer `data_type` aligned.
  // Explanation of the math here: from the start of a chunk metadata, we need
  // to skip the size of the chunk (4B), and (num_RLEs + 1) RLE offsets
  // (4B each) to get to the start of the delta header.
  data_type *const delta_header = roundUpToAlignment<data_type>(chunk_metadata + 1 + comp_opts.num_RLEs + 1);

  // Number of output elements for the RLE layer
  __shared__ size_type num_outputs;
  size_type out_bytes;

  for (int partition_idx = batch_start; partition_idx < batch_size; partition_idx += batch_stride)
  {
    const auto input_buffer = uncompressed_data[partition_idx];
    const auto input_bytes = uncompressed_bytes[partition_idx];
    assert(input_bytes <= UINT32_MAX);
    const size_type num_input_elements = input_bytes / sizeof(data_type);
    auto output_buffer = static_cast<uint32_t *>(compressed_data[partition_idx]);
    // `output_limit` points to the end of the output compressed buffer of the
    // current partition. The size of the compressed buffer should be at least
    // 8B larger than the input uncompressed buffer. It needs to be 8B larger
    // because in the fallback path, the compressed buffer still needs to store
    // the metadata. It is users responsibility to guarantee this requirement.
    uint32_t *output_limit = output_buffer + roundUpDiv(universal_header::header_size_bytes, sizeof(uint32_t)) +
                             roundUpDiv(input_bytes, sizeof(uint32_t));

    if (input_buffer == nullptr || input_bytes == 0)
    {
      if (threadIdx.x == 0)
      {
        compressed_bytes[partition_idx] = 0;
      }
      continue;
    }

    // Global flag on whether we will compress the current partition. If
    // compressed size is larger than the uncompressed size (i.e. compression
    // ratio < 1), we will use the fallback path of directly copying from the
    // input buffer to the compressed buffer, and set this flag to false.
    bool use_compression = true;

    if (comp_opts.num_RLEs == 0 && comp_opts.num_deltas == 0 && comp_opts.use_bp == 0)
    {
      use_compression = false;
    }

    // Pointer to the first chunk of the current partition
    auto current_output_ptr = reinterpret_cast<uint32_t *>(
      roundUpToAlignment<data_type>(output_buffer + roundUpDiv(universal_header::header_size_bytes, sizeof(uint32_t)))
    );

    const int num_chunks = roundUpDiv(num_input_elements, chunk_num_elements);

    for (int chunk_idx = 0; chunk_idx < num_chunks && use_compression; chunk_idx++)
    {
      // Save a pointer at the start of the chunk
      uint32_t *chunk_start_ptr = current_output_ptr;

      // Move current output pointer as the end of chunk metadata
      current_output_ptr += chunk_metadata_size / sizeof(uint32_t);

      auto input_buffer_current_chunk = input_buffer + chunk_num_elements * chunk_idx;
      size_type num_elements_current_chunk =
        min(num_input_elements - chunk_idx * chunk_num_elements, static_cast<size_type>(chunk_num_elements));

      // Threadblock collectively loads current chunk from input uncompressed
      // buffer to shared memory buffer
      for (int element_idx = threadIdx.x; element_idx < num_elements_current_chunk; element_idx += blockDim.x)
      {
        shared_element_buffer_0[element_idx] = input_buffer_current_chunk[element_idx];
      }
      __syncthreads();

      int rle_remaining = comp_opts.num_RLEs;
      int delta_remaining = comp_opts.num_deltas;
      int delta_skipped = 0;

      data_type *shared_input_buffer = shared_element_buffer_0;
      data_type *shared_output_buffer = shared_element_buffer_1;

      for (int layer_idx = 0; layer_idx < max(comp_opts.num_RLEs, comp_opts.num_deltas); layer_idx++)
      {
        if (rle_remaining > 0)
        {
          // Run RLE
          modules::block_rle_compress<data_type, size_type, run_type, threadblock_size>(
            shared_input_buffer,
            num_elements_current_chunk,
            shared_output_buffer,
            reinterpret_cast<run_type *>(shared_count_buffer),
            &num_outputs,
            reinterpret_cast<run_type *>(shared_tmp_buffer)
          );
          __syncthreads();
          assert(num_outputs <= (storage_num_elements * sizeof(shared_storage_type)) / sizeof(data_type));

          // Save run counts to the compressed buffer
          if (block_write<run_type, size_type, threadblock_size>(
                reinterpret_cast<run_type *>(shared_count_buffer),
                num_outputs,
                current_output_ptr,
                output_limit,
                &out_bytes,
                shared_tmp_buffer,
                comp_opts.use_bp,
                false
              ) != BlockIOStatus::success)
          {
            use_compression = false;
            goto afterlastchunk;
          }

          current_output_ptr += roundUpDiv(out_bytes, 4);

          // Store the size into chunk metadata
          if (threadIdx.x == 0)
          {
            chunk_metadata[comp_opts.num_RLEs - rle_remaining + 1] = out_bytes;
          }

          // Revert the role of input and ouput buffer
          auto temp_ptr = shared_output_buffer;
          shared_output_buffer = shared_input_buffer;
          shared_input_buffer = temp_ptr;

          num_elements_current_chunk = num_outputs;

          rle_remaining--;
        }

        if (delta_remaining > 0)
        {
          // A previous delta pass may have removed the last element,
          // but delta needs at least one element, so skip if none
          if (num_elements_current_chunk == 0)
          {
            ++delta_skipped;
            if (threadIdx.x == 0)
            {
              // Arbitrary zero, so that it's initialized
              delta_header[comp_opts.num_deltas - delta_remaining] = 0;
            }
            --delta_remaining;
            // Even though there are no elements left, to maintain compatibility
            // with cases before this fix was introduced, still do any remaining
            // RLE passes, because they may write a frame of reference value for
            // bit packing, and they initialize entries in chunk_metadata
            continue;
          }

          // Run Delta
          assert(num_elements_current_chunk <= (storage_num_elements * sizeof(shared_storage_type)) / sizeof(data_type));
          modules::block_delta_compress<data_type, size_type>(
            shared_input_buffer,
            num_elements_current_chunk,
            shared_output_buffer
          );

          if (threadIdx.x == 0)
          {
            delta_header[comp_opts.num_deltas - delta_remaining] = shared_input_buffer[0];
          }

          // Revert the role of input and ouput buffer
          auto temp_ptr = shared_output_buffer;
          shared_output_buffer = shared_input_buffer;
          shared_input_buffer = temp_ptr;

          // Number of elements is decreased by 1 since the first element is
          // excluded for the subsequent operations.
          num_elements_current_chunk -= 1;

          delta_remaining--;
        }

        __syncthreads();
      }

      // Save final output to output buffer
      auto final_output_ptr = reinterpret_cast<uint32_t *>(roundUpToAlignment<data_type>(current_output_ptr));

      if (delta_skipped != 0)
      {
        assert(num_elements_current_chunk == 0);
        // A negative out_bytes value is used to indicate that some number of
        // delta passes were skipped.
        out_bytes = size_type(-delta_skipped);
        current_output_ptr = final_output_ptr;
        if (reinterpret_cast<uintptr_t>(current_output_ptr) > reinterpret_cast<uintptr_t>(output_limit))
        {
          use_compression = false;
          goto afterlastchunk; // This is set to avoid flushing the chunk header below
        }
      }
      else
      {
        if (block_write<data_type, size_type, threadblock_size>(
              shared_input_buffer,
              num_elements_current_chunk,
              final_output_ptr,
              output_limit,
              &out_bytes,
              reinterpret_cast<uint32_t *>(shared_output_buffer),
              comp_opts.use_bp,
              comp_opts.num_deltas != 0
            ) != BlockIOStatus::success)
        {
          use_compression = false;
          break;
        }

        current_output_ptr = final_output_ptr + roundUpDiv(out_bytes, 4);
        current_output_ptr = reinterpret_cast<uint32_t *>(roundUpToAlignment<data_type>(current_output_ptr));
      }

      // Flush chunk header from shared memory to output buffer
      if (threadIdx.x == 0)
      {
        const uint32_t chunk_output_size = reinterpret_cast<uintptr_t>(current_output_ptr) -
                                           reinterpret_cast<uintptr_t>(chunk_start_ptr);
        chunk_metadata[0] = chunk_output_size;
        chunk_metadata[comp_opts.num_RLEs + 1] = out_bytes;

        for (int idx = 0; idx < chunk_metadata_size / 4; idx++)
        {
          chunk_start_ptr[idx] = chunk_metadata[idx];
        }
      }

      __syncthreads();
    }

  afterlastchunk:
    if (!use_compression)
    {
      // Compressed size is larger than uncompressed size, so we fallback to
      // directly copy input array to output
      data_type *direct_output_buffer =
        roundUpToAlignment<data_type>(output_buffer + universal_header::header_size_words);
      for (int element_idx = threadIdx.x; element_idx < num_input_elements; element_idx += blockDim.x)
      {
        direct_output_buffer[element_idx] = input_buffer[element_idx];
      }
    }

    // Save the metadata of the current partition
    if (threadIdx.x == 0)
    {
      auto partition_metadata_ptr = reinterpret_cast<uint8_t *>(output_buffer);

      uint32_t uncompressed_size_bytes = static_cast<uint32_t>(num_input_elements * sizeof(data_type));
      if (use_compression)
      {
        universal_header::write_legacy_header(
          partition_metadata_ptr,
          cascaded::d_TypeOf<data_type>(),
          uncompressed_size_bytes, // swap into 3rd slot
          comp_opts.num_RLEs,
          comp_opts.num_deltas,
          comp_opts.use_bp
        );

        compressed_bytes[partition_idx] = reinterpret_cast<uintptr_t>(current_output_ptr) -
                                          reinterpret_cast<uintptr_t>(output_buffer);
      }
      else
      {
        universal_header::write_dummy_header(
          partition_metadata_ptr,
          cascaded::d_TypeOf<data_type>(),
          uncompressed_size_bytes
        );

        compressed_bytes[partition_idx] = roundUpTo(universal_header::header_size_bytes, sizeof(data_type)) +
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
  nvcompBatchedCascadedCompressOpts_t comp_opts
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
  const nvcompBatchedCascadedCompressOpts_t format_opts,
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
    <<<nvcomp::cuda_dim_cast(batch_size), threadblock_size, 0, stream>>>(
      nvcomp::narrow_cast<int>(batch_size),
      reinterpret_cast<const data_type *const *>(device_uncompressed_chunk_ptrs),
      device_uncompressed_bytes,
      device_compressed_ptrs,
      device_compressed_bytes,
      format_opts
    );
  CUDA_CHECK(cudaGetLastError());

  // mark compression successful
  nvcomp::try_clear_device_statuses(batch_size, device_statuses, stream);
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
  try
  {
    NVCOMP_TYPE_ONE_SWITCH(
      comp_opts.data_type,
      composite_batched_compression_typed,
      comp_opts,
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

} // namespace composite