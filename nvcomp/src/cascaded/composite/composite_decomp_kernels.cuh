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
#include "nvcomp/cascaded.h"
#include "type_macros.h"

using nvcomp::Check;
using nvcomp::CudaUtils;
using nvcomp::isAligned;
using nvcomp::roundUpDiv;
using nvcomp::roundUpTo;
using nvcomp::roundUpToAlignment;

namespace composite
{

/**
 * Read a buffer with a single threadblock, optionally with bitunpacking.
 *
 * For the current implementation, this function is used to load compressed data
 * from the global memory to the shared memory.
 *
 * @param[in] input Compressed input buffer.
 * @param[in] in_byte Number of bytes of the compressed input.
 * @param[in] input_limit Pointer past the end of the input buffer. If this
 * function needs to load beyond this pointer, the function will return
 * `BlockIOStatus::out_of_bound`. This pointer must belong to the same array as
 * \p input, otherwise the behavior is undefined.
 * @param[out] output Pointer to the output buffer.
 * @param[out] out_num_elements Number of `data_type` elements written to the
 * output buffer. This argument should be unique to each thread.
 * @param[in] temp_storage Temporary storage to hold input data before
 * bitpacking. User needs to guarantee that this buffer has size at least
 * `in_byte` (rounded up to a multiple of 4) bytes.
 * @param[in] use_bp Whether bitpacking should be used.
 */
template <typename data_type, typename size_type, int threadblock_size>
__device__ BlockIOStatus block_read(
  const uint32_t *input,
  size_type in_byte,
  const uint32_t *input_limit,
  data_type *output,
  size_type *out_num_elements,
  uint32_t *temp_storage,
  bool use_bp
)
{
  if (input_limit && input + roundUpDiv(in_byte, 4) > input_limit)
  {
    return BlockIOStatus::out_of_bound;
  }

  uint32_t *dest_ptr;
  if (use_bp)
  {
    dest_ptr = temp_storage;
  }
  else
  {
    dest_ptr = reinterpret_cast<uint32_t *>(output);
  }

  for (int element_idx = threadIdx.x; element_idx < roundUpDiv(in_byte, 4); element_idx += threadblock_size)
  {
    dest_ptr[element_idx] = input[element_idx];
  }
  __syncthreads();

  if (use_bp)
  {
    modules::block_bitunpack<data_type, size_type>(temp_storage, output, out_num_elements);
  }
  else
  {
    if (out_num_elements != nullptr)
    {
      *out_num_elements = in_byte / sizeof(data_type);
    }
  }

  return BlockIOStatus::success;
}

/**
 * @brief Device function to perform batched cascaded decompression for
 * a given datatype
 *
 * @tparam data_type Data type of each uncompressed element.
 * @tparam threadblock_size Number of threads in a threadblock. This argument
 * must match the configuration specified when launching this kernel.
 * @tparam chunk_size Number of bytes for each uncompressed chunk to fit inside
 * shared memory. This argument must match the chunk size specified during
 * compression.
 *
 * @param[in] batch_size Number of partitions to decompress.
 * @param[in] compressed_data Array of size \p batch_size where each element is
 * a pointer to the compressed data of a partition.
 * @param[in] compressed_bytes Sizes of the compressed buffers corresponding to
 * \p compressed_data.
 * @param[out] decompressed_data Pointers to the output decompressed buffers.
 * @param[in] decompressed_buffer_bytes Sizes of the decompressed buffers in
 * bytes.
 * @param[out] actual_decompressed_bytes Actual number of bytes decompressed for
 * all partitions.
 * @param[in] shmem Allocated shared memory buffer for use in decompression.
 * @param[out] statuses Whether the compressions are successful.
 */
template <typename data_type, typename size_type, int threadblock_size, int chunk_size = default_chunk_size>
__device__ void composite_decompression_fcn(
  int batch_size,
  int batch_start,
  int batch_stride,
  const void *const *compressed_data,
  const size_type *compressed_bytes,
  void *const *decompressed_data,
  const size_type *decompressed_buffer_bytes,
  size_type *actual_decompressed_bytes,
  void *shmem,
  nvcompStatus_t *statuses
)
{
  using run_type = uint16_t;
  constexpr int chunk_num_elements = chunk_size / sizeof(data_type);

  // Shared memory storage for chunk metadata. Chunk metadata consists of
  // 1. size of the chunk (4B)
  // 2. The sizes in bytes of all RLE count arrays (4B per RLE layer)
  // 3. The size in byte of the final array (4B)
  // 4. The first elements of each delta layer (data type size per Delta layer)
  // We assume the data type is at most 8B large, so we use uint64_t here to
  // make sure the storage starts at an 8-byte alignment location.
  // Here we assume the metadata is at most 64B.
  uint64_t *chunk_metadata_storage = static_cast<uint64_t *>(shmem);
  auto chunk_metadata = reinterpret_cast<uint32_t *>(chunk_metadata_storage);
  shmem = static_cast<void *>((static_cast<uint8_t *>(shmem)) + 64);

  // `shared_element_storage_0` and `shared_element_storage_1` are shared memory
  // storage used for the data arrays of the input and the output of the current
  // layer. `shared_storage_type` is used to make sure the storage is both 4B
  // and data_type aligned.
  typedef larger_t<data_type, uint32_t> shared_storage_type;

  // Allocate `4 + sizeof(data_type)` in addition to `chunk_size` to accommodate
  // bitpacking metadata.
  constexpr size_t storage_num_elements = roundUpDiv(chunk_size + 4 + sizeof(data_type), sizeof(shared_storage_type));

  shared_storage_type *shared_element_storage_0 = static_cast<shared_storage_type *>(shmem);
  shmem = static_cast<void *>(static_cast<shared_storage_type *>(shmem) + storage_num_elements);
  data_type *shared_element_buffer_0 = reinterpret_cast<data_type *>(shared_element_storage_0);

  shared_storage_type *shared_element_storage_1 = static_cast<shared_storage_type *>(shmem);
  shmem = static_cast<void *>(static_cast<shared_storage_type *>(shmem) + storage_num_elements);

  data_type *shared_element_buffer_1 = reinterpret_cast<data_type *>(shared_element_storage_1);

  // `count_array` is the shared memory storage for RLE count arrays.
  // `temp_count_array` is used for bit-unpacking when loading RLE counts. Since
  // run_type should be no larger than 4B, we use `uint32_t` to guarantee 4B
  // aligned (which implies run_type aligned as well).
  uint32_t *count_array = static_cast<uint32_t *>(shmem);
  shmem = static_cast<void *>(static_cast<uint8_t *>(shmem) + (chunk_num_elements * sizeof(run_type)));

  uint32_t *temp_count_array = static_cast<uint32_t *>(shmem);
  shmem = static_cast<void *>(static_cast<uint8_t *>(shmem) + (chunk_num_elements * sizeof(run_type)));

  // RLE offsets
  uint32_t *rle_offsets = static_cast<uint32_t *>(shmem);

  for (int partition_idx = batch_start; partition_idx < batch_size; partition_idx += batch_stride)
  {
    if (compressed_data[partition_idx] == nullptr ||
        compressed_bytes[partition_idx] < universal_header::header_size_bytes)
    {
      // Compressed buffer should at least have enough space for partition
      // metadata.
      if (threadIdx.x == 0)
      {
        // Special case for zero-byte chunks: they compress to zero bytes,
        // so they're valid to be decompressed to zero bytes.
        const bool is_empty = compressed_bytes[partition_idx] == 0;
        statuses[partition_idx] = is_empty ? nvcompSuccess : nvcompErrorCannotDecompress;
        actual_decompressed_bytes[partition_idx] = 0;
      }
      continue;
    }

    if (!universal_header::is_legacy_compressed(reinterpret_cast<const uint8_t *>(compressed_data[partition_idx])))
    {
      // This buffer was compressed using a compression mode other than LEGACY.
      // This kernel cannot decompress the buffer.
      if (threadIdx.x == 0)
      {
        statuses[partition_idx] = nvcompErrorCannotDecompress;
        actual_decompressed_bytes[partition_idx] = 0;
      }
      continue;
    }

    const uint32_t *partition_start_ptr = static_cast<const uint32_t *>(compressed_data[partition_idx]);
    const uint32_t *partition_end_ptr = partition_start_ptr + compressed_bytes[partition_idx] / 4;
    data_type *decompressed_ptr = reinterpret_cast<data_type *const *>(decompressed_data)[partition_idx];
    size_type decompressed_num_elements = 0;

    const uint8_t *partition_metadata_ptr = reinterpret_cast<const uint8_t *>(partition_start_ptr);
    int num_RLEs = universal_header::get_num_rles(partition_metadata_ptr);
    int num_deltas = universal_header::get_num_deltas(partition_metadata_ptr);
    int bitpacking = universal_header::get_use_bitpack(partition_metadata_ptr);

    // Max number of RLE layers is 7
    assert(num_RLEs <= max_num_rle_layers);
    assert(num_deltas <= max_num_delta_layers);

    const uint32_t num_uncompressed_elements =
      universal_header::get_uncompressed_size(reinterpret_cast<const uint8_t *>(partition_start_ptr)) /
      sizeof(data_type);

    if (decompressed_buffer_bytes[partition_idx] < sizeof(data_type) * num_uncompressed_elements)
    {
      // The output buffer is not large enough to hold all uncompressed
      // elements, so we report failure.
      if (threadIdx.x == 0)
      {
        actual_decompressed_bytes[partition_idx] = 0;
        statuses[partition_idx] = nvcompErrorCannotDecompress;
      }
      continue;
    }

    if (num_RLEs == 0 && num_deltas == 0 && bitpacking == 0)
    {
      // No compression is used. This could be the result of user specification
      // or compression ratio less than 1. In this case, we copy the compressed
      // data directly to output buffer.

      if (compressed_bytes[partition_idx] < roundUpTo(universal_header::header_size_bytes, sizeof(data_type)) +
                                              sizeof(data_type) * num_uncompressed_elements)
      {
        // Compressed buffer does not have enough space to hold all uncompressed
        // data, so we report failure.
        if (threadIdx.x == 0)
        {
          actual_decompressed_bytes[partition_idx] = 0;
          statuses[partition_idx] = nvcompErrorCannotDecompress;
        }
      }
      else
      {
        const data_type *direct_compressed_buffer =
          roundUpToAlignment<data_type>(partition_start_ptr + universal_header::header_size_words);
        for (int element_idx = threadIdx.x; element_idx < num_uncompressed_elements; element_idx += blockDim.x)
        {
          decompressed_ptr[element_idx] = direct_compressed_buffer[element_idx];
        }
        if (threadIdx.x == 0)
        {
          actual_decompressed_bytes[partition_idx] = sizeof(data_type) * num_uncompressed_elements;
          statuses[partition_idx] = nvcompSuccess;
        }
      }
      continue;
    }

    // Start location of the first elements of delta layers in shared memory
    // storage of chunk metadata.
    const data_type *const delta_header = roundUpToAlignment<data_type>(chunk_metadata + 1 + num_RLEs + 1);

    // `chunk_ptr` points to the start location of the current chunk in global
    // memory. Here we initialize it to the start location of the first chunk.
    const uint32_t *chunk_ptr = reinterpret_cast<const uint32_t *>(
      roundUpToAlignment<data_type>(partition_start_ptr + universal_header::header_size_words)
    );

    bool is_decompression_successful = true;

    while (chunk_ptr < partition_end_ptr)
    {
      // Load chunk metadata to the shared memory storage
      const int chunk_metadata_size = get_chunk_metadata_size<data_type>(num_RLEs, num_deltas);
      if (chunk_ptr + chunk_metadata_size / 4 > partition_end_ptr)
      {
        // Compressed buffer does not have enough space for the current chunk
        // metadata. This means the compressed data is corrupt, so we report
        // failure.
        is_decompression_successful = false;
        break;
      }
      for (int element_idx = threadIdx.x; element_idx < chunk_metadata_size / 4; element_idx += threadblock_size)
      {
        chunk_metadata[element_idx] = chunk_ptr[element_idx];
      }

      __syncthreads();

      // Chunk size is the first element of metadata
      const int compressed_chunk_size = chunk_metadata[0];

      // Calculate RLE count array / final array location offsets from array
      // sizes. The calculation is a prefix sum on array sizes with alignment
      // paddings.
      if (threadIdx.x == 0)
      {
        rle_offsets[0] = 0;
        if (num_RLEs > 0)
        {
          for (int rle_idx = 0; rle_idx < num_RLEs - 1; rle_idx++)
          {
            // The count arrays start at alignment of 4.
            rle_offsets[rle_idx + 1] = roundUpTo(rle_offsets[rle_idx] + chunk_metadata[rle_idx + 1], 4);
          }
          // The final array start at location both aligned with data_type and
          // aligned with 4B.
          rle_offsets[num_RLEs] = roundUpTo(
            rle_offsets[num_RLEs - 1] + chunk_metadata[num_RLEs],
            max(static_cast<size_t>(4), sizeof(data_type))
          );
        }
      }
      __syncthreads();

      data_type *shared_input_buffer = shared_element_buffer_0;
      data_type *shared_output_buffer = shared_element_buffer_1;
      const uint32_t *rle0_ptr = chunk_ptr + chunk_metadata_size / 4;

      // Load array after final layer to shared memory
      const uint32_t *final_array_ptr = rle0_ptr + rle_offsets[num_RLEs] / 4;
      const uint32_t in_bytes = chunk_metadata[1 + num_RLEs];
      size_type num_elements;
      int delta_remaining;
      if (int32_t(in_bytes) >= 0)
      {
        if (block_read<data_type, size_type, threadblock_size>(
              final_array_ptr,
              in_bytes,
              partition_end_ptr,
              shared_input_buffer,
              &num_elements,
              reinterpret_cast<uint32_t *>(shared_output_buffer),
              bitpacking
            ) != BlockIOStatus::success)
        {
          is_decompression_successful = false;
          break;
        }
        __syncthreads();

        delta_remaining = num_deltas;
      }
      else
      {
        // Negative byte count for this sub-chunk indicates that there are
        // no elements and that at least one delta pass was skipped due
        // to running out of elements during compression, so they must be
        // skipped during decompression, too.
        delta_remaining = num_deltas + int32_t(in_bytes);
        num_elements = 0;
      }

      int rle_remaining = num_RLEs;

      for (int layer_idx = 0; layer_idx < max(num_RLEs, num_deltas); layer_idx++)
      {
        if (delta_remaining > 0 && delta_remaining >= rle_remaining)
        {
          // Decompress the delta layer
          modules::block_delta_decompress<data_type, size_type, threadblock_size>(
            shared_input_buffer,
            delta_header[delta_remaining - 1],
            num_elements,
            shared_output_buffer
          );
          __syncthreads();

          // Revert the role of input and ouput buffer
          auto temp_ptr = shared_output_buffer;
          shared_output_buffer = shared_input_buffer;
          shared_input_buffer = temp_ptr;

          // Decompressing delta layer adds one extra element (the first
          // element).
          num_elements++;
          delta_remaining--;
        }

        if (rle_remaining > 0 && rle_remaining > delta_remaining)
        {
          // Load the count array from global memory to shared memory
          if (block_read<run_type, size_type, threadblock_size>(
                rle0_ptr + rle_offsets[rle_remaining - 1] / 4,
                chunk_metadata[rle_remaining],
                partition_end_ptr,
                reinterpret_cast<run_type *>(count_array),
                nullptr,
                temp_count_array,
                bitpacking
              ) != BlockIOStatus::success)
          {
            is_decompression_successful = false;
            goto afterlastchunk;
          }
          __syncthreads();

          // Decompress the RLE layer
          size_type output_num_elements;
          modules::block_rle_decompress<data_type, size_type, run_type, threadblock_size>(
            shared_input_buffer,
            reinterpret_cast<run_type *>(count_array),
            num_elements,
            shared_output_buffer,
            &output_num_elements
          );
          num_elements = output_num_elements;

          // Revert the role of input and ouput buffer
          auto temp_ptr = shared_output_buffer;
          shared_output_buffer = shared_input_buffer;
          shared_input_buffer = temp_ptr;

          rle_remaining--;
        }
      }

      // Save the current chunk to the output buffer

      if (decompressed_num_elements + num_elements > num_uncompressed_elements)
      {
        // If the number of decompressed elements after the current chunk is
        // more than the total number of uncompressed elements, the compressed
        // data must be corrupted, so we report failure.
        is_decompression_successful = false;
        break;
      }

      for (int element_idx = threadIdx.x; element_idx < num_elements; element_idx += threadblock_size)
      {
        decompressed_ptr[element_idx] = shared_input_buffer[element_idx];
      }
      decompressed_ptr += num_elements;
      decompressed_num_elements += num_elements;

      // Update `chunk_ptr` to the start location of the next chunk
      chunk_ptr =
        reinterpret_cast<const uint32_t *>(roundUpToAlignment<data_type>(chunk_ptr + compressed_chunk_size / 4));
    }

  afterlastchunk:
    if (num_uncompressed_elements != decompressed_num_elements)
    {
      // The number of decompressed elements does not match the uncompressed
      // element stored in the compressed buffer. This means the compressed
      // data is corrupted, so we report failure.
      is_decompression_successful = false;
    }

    if (threadIdx.x == 0)
    {
      if (is_decompression_successful)
      {
        actual_decompressed_bytes[partition_idx] = decompressed_num_elements * sizeof(data_type);
        statuses[partition_idx] = nvcompSuccess;
      }
      else
      {
        actual_decompressed_bytes[partition_idx] = 0;
        statuses[partition_idx] = nvcompErrorCannotDecompress;
      }
    }
  }
}

/**
 * @brief Kernel to perform batched cascaded decompression. Extracts the
 * datatype from the metadata of the compressed buffer, then checks of the
 * templated call type matches.  If it matches, it allocates the correct amount
 * of shared memory and runs decompression.  Otherwise, it just exits.
 *
 * @tparam bitwidth_test Data type to use for underlying decompression.  If
 * datatype found in metadata matches, perform compression, else exit.
 * @tparam size_type Data type used for size measures, typically size_t is used.
 * @tparam threadblock_size Number of threads in a threadblock. This argument
 * must match the configuration specified when launching this kernel.
 * @tparam chunk_size Number of bytes for each uncompressed chunk to fit inside
 * shared memory. This argument must match the chunk size specified during
 * compression.
 *
 * @param[in] batch_size Number of partitions to decompress.
 * @param[in] compressed_data Array of size \p batch_size where each element is
 * a pointer to the compressed data of a partition.
 * @param[in] compressed_bytes Sizes of the compressed buffers corresponding to
 * \p compressed_data.
 * @param[out] decompressed_data Pointers to the output decompressed buffers.
 * @param[in] decompressed_buffer_bytes Sizes of the decompressed buffers in
 * bytes.
 * @param[out] actual_decompressed_bytes Actual number of bytes decompressed for
 * all partitions.
 */
template <int bitwidth_test, typename size_type, int threadblock_size, int chunk_size = default_chunk_size>
__global__ void type_checked_composite_decompression_kernel(
  int batch_size,
  const void *const *compressed_data,
  const size_type *compressed_bytes,
  void *const *decompressed_data,
  const size_type *decompressed_buffer_bytes,
  size_type *actual_decompressed_bytes,
  nvcompStatus_t *statuses
)
{
  // This kernel assumes all chunks have the same data type.
  // TODO: is there a reason to require all chunks be the same data type?

  // Find the first non-null compressed buffer with a readable header. Assume that buffer's type is also my buffers type
  int i = 0;
  nvcompType_t type;
  while (i < batch_size)
  {
    if (compressed_data[i] != nullptr && compressed_bytes[i] >= universal_header::header_size_bytes)
    {
      const auto partition_metadata_ptr = reinterpret_cast<const uint8_t *>(compressed_data[i]);
      type = universal_header::get_data_type(partition_metadata_ptr);
      assert(nvcomp::isValidNvcompType(type));
      break;
    }
    if (i == batch_size - 1)
    {
      return; // No readable headers; nothing to do
    }
    ++i;
  }

  using data_type = std::conditional_t<
    bitwidth_test == 1,
    uint8_t,
    std::conditional_t<bitwidth_test == 2, uint16_t, std::conditional_t<bitwidth_test == 4, uint32_t, uint64_t>>>;

  constexpr int shmem_size = compute_decompress_smem_size<chunk_size, bitwidth_test, ((bitwidth_test == 8) ? 8 : 4)>();
  // This must be aligned to at least sizeof(data_type) and sizeof(uint32_t)
  // to avoid misaligned access crashes depending on its alignment.
  __shared__ alignas(8) uint8_t shmem[shmem_size];

  const nvcompType_t signed_type = nvcomp::TypeOfConst<std::make_signed_t<data_type>>();
  const nvcompType_t unsigned_type = nvcomp::TypeOfConst<std::make_unsigned_t<data_type>>();
  if (type == signed_type || type == unsigned_type)
  {
    composite_decompression_fcn<data_type, size_type, threadblock_size>(
      batch_size,
      blockIdx.x,
      gridDim.x,
      compressed_data,
      compressed_bytes,
      decompressed_data,
      decompressed_buffer_bytes,
      actual_decompressed_bytes,
      reinterpret_cast<void *>(shmem),
      statuses
    );
  }
}

nvcompStatus_t DecompressAsync(
  const void *const *device_compressed_chunk_ptrs,
  const size_t *device_compressed_chunk_bytes,
  const size_t *device_uncompressed_buffer_bytes,
  size_t *device_uncompressed_chunk_bytes,
  size_t num_chunks,
  void *const *device_uncompressed_chunk_ptrs,
  nvcompStatus_t *device_statuses,
  nvcompType_t type,
  cudaStream_t stream
)
{
  constexpr int threadblock_size = composite_decompress_threadblock_size;
  try
  {
    switch (type)
    {
      case NVCOMP_TYPE_CHAR:
      case NVCOMP_TYPE_UCHAR:
        type_checked_composite_decompression_kernel<1, size_t, threadblock_size>
          <<<nvcomp::cuda_dim_cast(num_chunks), threadblock_size, 0, stream>>>(
            nvcomp::narrow_cast<int>(num_chunks),
            device_compressed_chunk_ptrs,
            device_compressed_chunk_bytes,
            device_uncompressed_chunk_ptrs,
            device_uncompressed_buffer_bytes,
            device_uncompressed_chunk_bytes,
            device_statuses
          );
        break;
      case NVCOMP_TYPE_SHORT:
      case NVCOMP_TYPE_USHORT:
        type_checked_composite_decompression_kernel<2, size_t, threadblock_size>
          <<<nvcomp::cuda_dim_cast(num_chunks), threadblock_size, 0, stream>>>(
            nvcomp::narrow_cast<int>(num_chunks),
            device_compressed_chunk_ptrs,
            device_compressed_chunk_bytes,
            device_uncompressed_chunk_ptrs,
            device_uncompressed_buffer_bytes,
            device_uncompressed_chunk_bytes,
            device_statuses
          );
        break;
      case NVCOMP_TYPE_INT:
      case NVCOMP_TYPE_UINT:
        type_checked_composite_decompression_kernel<4, size_t, threadblock_size>
          <<<nvcomp::cuda_dim_cast(num_chunks), threadblock_size, 0, stream>>>(
            nvcomp::narrow_cast<int>(num_chunks),
            device_compressed_chunk_ptrs,
            device_compressed_chunk_bytes,
            device_uncompressed_chunk_ptrs,
            device_uncompressed_buffer_bytes,
            device_uncompressed_chunk_bytes,
            device_statuses
          );
        break;
      case NVCOMP_TYPE_LONGLONG:
      case NVCOMP_TYPE_ULONGLONG:
        type_checked_composite_decompression_kernel<8, size_t, threadblock_size>
          <<<nvcomp::cuda_dim_cast(num_chunks), threadblock_size, 0, stream>>>(
            nvcomp::narrow_cast<int>(num_chunks),
            device_compressed_chunk_ptrs,
            device_compressed_chunk_bytes,
            device_uncompressed_chunk_ptrs,
            device_uncompressed_buffer_bytes,
            device_uncompressed_chunk_bytes,
            device_statuses
          );
        break;
      default:
        return nvcompErrorInvalidValue;
    }
    CUDA_CHECK(cudaGetLastError());
  }
  catch (const std::exception &e)
  {
    LOG_ERROR("{}", e.what());
    return Check::exception_to_error(e, "nvcompBatchedCascadedDecompressAsync()");
  }

  return nvcompSuccess;
}

} // namespace composite