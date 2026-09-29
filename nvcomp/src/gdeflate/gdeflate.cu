/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <thrust/scan.h>

#include <cuda_runtime.h>

#include <ciso646>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "deflate.h"
#include "exception.hpp"
#include "gdeflate/common.h"
#include "gdeflate/gdeflate.h"
#include "gdeflate/gdeflate_compress.h"
#include "gdeflate/gdeflate_constants.h"
#include "gdeflate/gdeflate_decompress.h"
#include "gdeflate/huffman.h"
#include "gdeflate/lz.h"
#include "nvcomp/utils.hpp"
#include "optimal_parse/sa.cuh"

using namespace nvcomp;

namespace gdeflate
{

/**
  * @brief Mapping between external and internal compression algo enums
  */
gdeflate_compression_internal_algo map_to_internal_algo(const gdeflate_compression_algo algo)
{
  switch (algo)
  {
    case HIGH_COMPRESSION:
      return OPTIMAL_PARSE;
    case HIGH_THROUGHPUT:
      return HASH_BASED;
    case ENTROPY_ONLY:
      return HUFFMAN_ONLY;
    case MEDIUM_COMPRESSION:
      return HASH_WITH_CHAIN;
    case L6_OP_COMPRESSION:
      return OPTIMAL_PARSE_L6;
  }
  return HASH_BASED;
}

/**
 * @brief Get the amount of temp space required on the GPU for decompression.
 *
 * @param num_chunks The number of items in the batch.
 * @param max_uncompressed_chunk_size The size of the largest chunk when uncompressed.
 * @param temp_bytes The amount of temporary GPU space that will be required to
 * decompress.
 *
 */
void decompressGetTempSize(size_t num_chunks, size_t max_uncompressed_chunk_size, size_t *temp_bytes)
{
  *temp_bytes = 0;
}

__device__ void zeroOutputCalulation(size_t out_bytes, size_t batch_size, void *const *device_out_ptr)
{
  uint8_t *out_ptr = (uint8_t *)device_out_ptr[blockIdx.x];

  const size_t alignment_mask = sizeof(uint4) - 1;
  uint4 *aligned_ptr = (uint4 *)(((uintptr_t)out_ptr + alignment_mask) & ~alignment_mask);

  uint8_t init_align_bytes = min((size_t)((uint8_t *)aligned_ptr - out_ptr), out_bytes);
  size_t aligned_elems = (out_bytes - init_align_bytes) / sizeof(uint4);
  uint8_t end_align_bytes = out_bytes - (init_align_bytes + aligned_elems * sizeof(uint4));
  uint8_t *end_ptr = out_ptr + (init_align_bytes + aligned_elems * sizeof(uint4));

  // Write out to init and end of the stream first
  if (threadIdx.x < init_align_bytes)
  {
    out_ptr[threadIdx.x] = 0;
  }
  if (threadIdx.x < end_align_bytes)
  {
    end_ptr[threadIdx.x] = 0;
  }

  // Write to aligned portion of the output stream
  uint4 zero = {0, 0, 0, 0};
  for (size_t i = threadIdx.x; i < aligned_elems; i += blockDim.x)
  {
    aligned_ptr[i] = zero;
  }
}

__global__ void DeflateCompressGetMaxOutputChunkSizeKernel(
  const size_t *chunk_uncompressed_size_bytes,
  size_t *chunk_max_compressed_size,
  size_t num_chunks_host,
  const int *device_num_chunks
)
{
  const size_t num_chunks = device_num_chunks ? static_cast<size_t>(*device_num_chunks) : num_chunks_host;
  int chunk_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (chunk_idx >= num_chunks)
  {
    return;
  }
  nvcomp_deflate::DeflateCompressGetMaxOutputChunkSize(
    chunk_uncompressed_size_bytes[chunk_idx],
    &chunk_max_compressed_size[chunk_idx]
  );
}

__global__ void zeroOutput(
  const size_t *device_out_bytes,
  size_t host_batch_size,
  void *const *device_out_ptr,
  const int *device_num_deflate_blocks
)
{
  // This kernel is kind of slow for huge amount of data, but on ncu takes still only 3% of runtime.
  const size_t batch_size = device_num_deflate_blocks ? static_cast<size_t>(*device_num_deflate_blocks)
                                                      : host_batch_size;
  if (blockIdx.x >= batch_size)
  {
    return;
  }
  size_t out_bytes = device_out_bytes[blockIdx.x];
  zeroOutputCalulation(out_bytes, batch_size, device_out_ptr);
}

/**
 * @brief Perform decompression.
 *
 * @param device_compressed_ptr The pointers on the GPU, to the compressed chunks.
 * @param device_compressed_bytes The size of each compressed chunk on the GPU.
 * @param device_uncompressed_bytes The size of each uncompressed chunk on the GPU (max available space).
 * @param device_actual_uncompressed_bytes Actual bytes of uncompressed chunk data. Can be set to nullptr to turn off bounds checking.
 * @param batch_size The number of batch items.
 * @param device_uncompressed_ptr The pointers on the GPU, to where to uncompress each chunk (output).
 * @param device_statuses Pointer to per chunk decompression status (success or failure). Can be set to nullptr to turn off bounds checking.
 * @param stream The stream to operate on.
 *
 */
void decompressAsync(
  const void *const *device_compressed_ptr,
  const size_t *device_compressed_bytes,
  const size_t *device_uncompressed_bytes,
  size_t *device_actual_uncompressed_bytes,
  size_t batch_size,
  void *const *device_uncompressed_ptr,
  gdeflateStatus_t *device_statuses,
  cudaStream_t stream
)
{

  bool check_bounds = true;
  if ((device_actual_uncompressed_bytes == nullptr) || (device_statuses == nullptr))
  {
    check_bounds = false;
  }

#ifndef SIMPLE_STORES
  // Zero out output buffer
  // Doing it here since we don't have access to the output buffer size
  // inside the gdeflate kernel
  zeroOutput<<<cuda_dim_cast(batch_size), 128, 0, stream>>>(
    device_uncompressed_bytes,
    narrow_cast<unsigned int>(batch_size),
    device_uncompressed_ptr
  );
  CUDA_CHECK(cudaGetLastError());
#endif

  constexpr unsigned int warps_per_cta = 4;
  auto grid = cuda_dim_cast(roundUpDiv(batch_size, warps_per_cta));
  dim3 block = {WARP_SIZE_U, warps_per_cta, 1};

  gdeflate_trace *trace = nullptr;
  switch (check_bounds)
  {
    case true:
      gdeflateDecompress<warps_per_cta><<<grid, block, 0, stream>>>(
        (const uint32_t *const *)device_compressed_ptr,
        (uint8_t *const *)device_uncompressed_ptr,
        device_compressed_bytes,
        device_uncompressed_bytes,
        narrow_cast<unsigned int>(batch_size),
        device_actual_uncompressed_bytes,
        device_statuses,
        trace
      );
      break;
    case false:
      gdeflateDecompress<warps_per_cta, false, true><<<grid, block, 0, stream>>>(
        (const uint32_t *const *)device_compressed_ptr,
        (uint8_t *const *)device_uncompressed_ptr,
        device_compressed_bytes,
        device_uncompressed_bytes,
        narrow_cast<unsigned int>(batch_size),
        device_actual_uncompressed_bytes,
        device_statuses,
        trace
      );
      break;
  }
  CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Calculates the decompressed size of each chunk asynchronously. This is
 * needed when we do not know the expected output size. All pointers must be GPU
 * accessible. Note, if the stream is corrupt, the sizes will be garbage.
 *
 * @param device_compress_ptrs The compressed chunks of data. List of pointers
 * must be GPU accessible along with each chunk.
 * @param device_compressed_bytes The size of each compressed chunk. Must be GPU
 * accessible.
 * @param device_uncompressed_bytes The calculated decompressed size of each
 * chunk. Must be GPU accessible.
 * @param batch_size The number of chunks
 * @param stream The stream to operate on.
 */
void getDecompressSizeAsync(
  const void *const *device_compressed_ptr,
  const size_t *device_compressed_bytes,
  size_t *device_actual_uncompressed_bytes,
  size_t batch_size,
  cudaStream_t stream
)
{

  // Note: in practice these never happen as they are checked in "nvcompBatchedGdeflateGetDecompressSizeAsync"
  if (device_compressed_ptr == nullptr)
  {
    throw NVCompException(nvcompErrorInvalidValue, "device_compressed_ptr must not be null");
  }
  if (device_compressed_bytes == nullptr)
  {
    throw NVCompException(nvcompErrorInvalidValue, "device_compressed_bytes must not be null");
  }
  if (device_actual_uncompressed_bytes == nullptr)
  {
    throw NVCompException(nvcompErrorInvalidValue, "device_actual_uncompressed_bytes must not be null");
  }

  constexpr unsigned int warps_per_cta = 4;
  auto grid = cuda_dim_cast(roundUpDiv(batch_size, warps_per_cta));
  dim3 block = {WARP_SIZE_U, warps_per_cta, 1};

  uint8_t *const *device_uncompressed_ptr = nullptr;
  const size_t *device_uncompressed_bytes = nullptr;
  gdeflateStatus_t *device_statuses = nullptr;

  gdeflate_trace *trace = nullptr;
  gdeflateDecompress<warps_per_cta, true, false><<<grid, block, 0, stream>>>(
    (const uint32_t *const *)device_compressed_ptr,
    device_uncompressed_ptr,
    device_compressed_bytes,
    device_uncompressed_bytes,
    narrow_cast<unsigned int>(batch_size),
    device_actual_uncompressed_bytes,
    device_statuses,
    trace
  );
  CUDA_CHECK(cudaGetLastError());
}

template <typename offset_type>
void compressGetTempSizeTemplated(
  size_t batch_size,
  size_t max_chunk_size,
  size_t *temp_bytes,
  gdeflate_compression_algo algo
)
{
  constexpr size_t length_size = sizeof(offset_type);
  constexpr size_t length_ptr_size = sizeof(offset_type *);

  size_t total_input_bytes = max_chunk_size * batch_size;

  *temp_bytes = roundUpTo(batch_size * length_ptr_size, sizeof(void *)) + // length_ptrs
                roundUpTo(batch_size * length_ptr_size, sizeof(void *)) + // distance_ptrs
                roundUpTo(batch_size * sizeof(uint8_t *), sizeof(void *)); // literal_ptrs

  const gdeflate_compression_internal_algo algo_ = map_to_internal_algo(algo);
  switch (algo_)
  {
    case OPTIMAL_PARSE:
      *temp_bytes += roundUpTo(batch_size * sizeof(uint32_t *), sizeof(void *)); // cost_ptrs
      // Reuse LZ & optimal parse temp with SA temp
      *temp_bytes += max(
        roundUpTo(sa::saGetTempStorageSize<offset_type>(max_chunk_size, batch_size), sizeof(void *)),
        roundUpTo(batch_size * sizeof(unsigned int), sizeof(void *)) + // num_symbols
          roundUpTo(batch_size * sizeof(unsigned int), sizeof(void *)) + // num_literals
          roundUpTo(total_input_bytes * length_size, sizeof(void *)) + // length
          roundUpTo(total_input_bytes * length_size, sizeof(void *)) + // distance
          roundUpTo(total_input_bytes * sizeof(uint8_t), sizeof(void *)) + // literals
          roundUpTo(total_input_bytes * sizeof(uint32_t), sizeof(void *))
      ); // costs
      *temp_bytes += roundUpTo(batch_size * length_ptr_size, sizeof(void *)) + // sa ptrs
                     roundUpTo(total_input_bytes * length_size, sizeof(void *)); // sa
      *temp_bytes += roundUpTo(batch_size * length_ptr_size, sizeof(void *)) + // inv sa ptrs
                     roundUpTo(total_input_bytes * length_size, sizeof(void *)); // inv sa
      // The largest value written to the lcp is deflateMaxMatchLength-1
      static_assert(deflateMaxMatchLength - 1 <= std::numeric_limits<sa::U16>::max());
      *temp_bytes += roundUpTo(batch_size * sizeof(sa::U16 *), sizeof(void *)) + // lcp ptrs
                     roundUpTo(total_input_bytes * sizeof(sa::U16), sizeof(void *)); // lcp
      break;
    case OPTIMAL_PARSE_L6:
      *temp_bytes += roundUpTo(batch_size * sizeof(uint32_t *), sizeof(void *)); // cost_ptrs
      // Reuse LZ & optimal parse temp with SA temp
      *temp_bytes += max(
        roundUpTo(sa::saGetTempStorageSize<offset_type>(max_chunk_size, batch_size), sizeof(void *)),
        roundUpTo(batch_size * sizeof(unsigned int), sizeof(void *)) + // num_symbols
          roundUpTo(batch_size * sizeof(unsigned int), sizeof(void *)) + // num_literals
          roundUpTo(total_input_bytes * length_size, sizeof(void *)) + // length
          roundUpTo(total_input_bytes * length_size, sizeof(void *)) + // distance
          roundUpTo(total_input_bytes * sizeof(uint8_t), sizeof(void *)) + // literals
          roundUpTo(total_input_bytes * sizeof(uint32_t), sizeof(void *))
      ); // costs
      *temp_bytes += roundUpTo(batch_size * length_ptr_size, sizeof(void *)) + // sa ptrs
                     roundUpTo(total_input_bytes * length_size, sizeof(void *)); // sa
      *temp_bytes += roundUpTo(batch_size * length_ptr_size, sizeof(void *)) + // inv sa ptrs
                     roundUpTo(total_input_bytes * length_size, sizeof(void *)); // inv sa
      *temp_bytes += roundUpTo(batch_size * sizeof(sa::U8 *), sizeof(void *)) + // lcp ptrs
                     roundUpTo(total_input_bytes * sizeof(sa::U8), sizeof(void *)); // lcp
      break;
    case HASH_BASED:
      *temp_bytes += roundUpTo(batch_size * sizeof(unsigned int), sizeof(void *)) + // num_symbols
                     roundUpTo(batch_size * sizeof(unsigned int), sizeof(void *)) + // num_literals
                     roundUpTo(total_input_bytes * length_size, sizeof(void *)) + // length
                     roundUpTo(total_input_bytes * length_size, sizeof(void *)) + // distance
                     roundUpTo(total_input_bytes * sizeof(uint8_t), sizeof(void *)); // literals
      *temp_bytes += roundUpTo(batch_size * HASH_TABLE_SIZE * length_size, sizeof(void *)); // Hash tables
      break;
    case HASH_WITH_CHAIN:
      *temp_bytes += roundUpTo(batch_size * sizeof(unsigned int), sizeof(void *)) + // num_symbols
                     roundUpTo(batch_size * sizeof(unsigned int), sizeof(void *)) + // num_literals
                     roundUpTo(total_input_bytes * length_size, sizeof(void *)) + // length
                     roundUpTo(total_input_bytes * length_size, sizeof(void *)) + // distance
                     roundUpTo(total_input_bytes * sizeof(uint8_t), sizeof(void *)); // literals
      *temp_bytes += roundUpTo(
        batch_size * HASH_TABLE_SIZE_WITH_CHAIN * sizeof(hash_chain<offset_type>),
        sizeof(void *)
      ); // Hash tables
      break;
    case ENTROPY_ONLY:
      // No temp space required for entropy only compression mode
      *temp_bytes = 0;
      break;
  }
}

/**
 * @brief Get temporary space required for compression.
 *
 * @param batch_size The number of items in the batch.
 * @param max_chunk_size The maximum size of a chunk in the batch.
 * @param temp_bytes The size of the required GPU workspace for compression
 * (output).
 */
void compressGetTempSize(size_t batch_size, size_t max_chunk_size, size_t *temp_bytes, gdeflate_compression_algo algo)
{

  if (max_chunk_size > gdeflateMaxChunkSize)
  {
    throw NVCompException(
      nvcompErrorChunkSizeTooLarge,
      std::string("Maximum allowed chunk size for gdeflate is ") + gdeflate::gdeflateMaxChunkSizeText
    );
  }

  if (max_chunk_size <= (1 << 16))
  {
    compressGetTempSizeTemplated<uint16_t>(batch_size, max_chunk_size, temp_bytes, algo);
  }
  else
  {
    compressGetTempSizeTemplated<uint32_t>(batch_size, max_chunk_size, temp_bytes, algo);
  }
}

/**
 * @brief Get the maximum size any chunk could compress to in the batch. That is, the minimum amount of output memory required to be given compressAsync() for each batch item.
 *
 * @param max_chunk_size The maximum size of a chunk in the batch.
 * @param max_compressed_size The maximum compressed size of the largest chunk (output).
 *
 */
void compressGetMaxOutputChunkSize(size_t max_chunk_size, size_t *max_compressed_size)
{
  if (max_chunk_size > gdeflateMaxChunkSize)
  {
    throw NVCompException(
      nvcompErrorChunkSizeTooLarge,
      std::string("Maximum allowed chunk size for gdeflate is ") + gdeflate::gdeflateMaxChunkSizeText
    );
  }

  // TODO: reduce temp_factor to make upper bound tighter
  constexpr float temp_factor = 2.f;

  // 286 bytes: upper bound on size of huffman table,
  // 2 * WARP_SIZE * sizeof(uint32_t): lower bound on compressed stream size (32-way interleaved)
  const size_t minimum_buffer_size = max(size_t{2 * WARP_SIZE * sizeof(uint32_t)}, size_t{286});
  *max_compressed_size =
    roundUpTo(static_cast<size_t>(std::ceil(temp_factor * max_chunk_size)) + minimum_buffer_size, sizeof(uint32_t));
}

template <typename LCP_t = sa::U16, typename offset_type>
__global__ void computeTempPointers(
  offset_type *const device_lengths,
  offset_type *const device_distances,
  uint8_t *const device_literals,
  uint32_t *const device_costs,
  offset_type *const device_sa,
  offset_type *const device_inv_sa,
  LCP_t *const device_lcp,
  const size_t max_chunk_size,
  const size_t host_batch_size,
  offset_type **device_length_ptrs,
  offset_type **device_distance_ptrs,
  uint8_t **device_literal_ptrs,
  uint32_t **device_cost_ptrs,
  offset_type **device_sa_ptrs,
  offset_type **device_inv_sa_ptrs,
  LCP_t **device_lcp_ptrs,
  const int *device_num_deflate_blocks
)
{
  const size_t batch_size = device_num_deflate_blocks ? static_cast<size_t>(*device_num_deflate_blocks)
                                                      : host_batch_size;
  size_t start = threadIdx.x + blockIdx.x * blockDim.x;
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = start; i < batch_size; i += stride)
  {
    device_length_ptrs[i] = device_lengths + i * max_chunk_size;
    device_distance_ptrs[i] = device_distances + i * max_chunk_size;
    device_literal_ptrs[i] = device_literals + i * max_chunk_size;
    if (device_costs)
    {
      device_cost_ptrs[i] = device_costs + i * max_chunk_size;
    }
    if (device_sa)
    {
      device_sa_ptrs[i] = device_sa + i * max_chunk_size;
    }
    if (device_inv_sa)
    {
      device_inv_sa_ptrs[i] = device_inv_sa + i * max_chunk_size;
    }
    if (device_lcp)
    {
      device_lcp_ptrs[i] = device_lcp + i * max_chunk_size;
    }
  }
}

template <typename offset_type>
void compressAsyncTemplated(
  const void *const *device_uncompressed_ptr,
  const size_t *device_uncompressed_bytes,
  const size_t max_chunk_size,
  unsigned int batch_size, // batch_size is always passed to algo specific function, which all accepts only unsigned int
  void *temp_ptr,
  size_t temp_bytes,
  void *const *device_compressed_ptr,
  size_t *device_compressed_bytes,
  uint8_t *device_compressed_pad_bits,
  gdeflate_compression_algo algo,
  cudaStream_t stream,
  bool standard,
  const int *device_num_deflate_blocks
)
{
#ifdef GDEFLATE_ENABLE_DEFLATE64
  [[maybe_unused]] constexpr bool deflate64 = true;
#else
  [[maybe_unused]] constexpr bool deflate64 = false;
#endif

  offset_type **device_length_ptrs = nullptr;
  offset_type **device_distance_ptrs = nullptr;
  uint8_t **device_literal_ptrs = nullptr;
  uint32_t **device_cost_ptrs = nullptr;
  unsigned int *num_symbols = nullptr;
  unsigned int *num_literals = nullptr;
  offset_type *device_lengths = nullptr;
  offset_type *device_distances = nullptr;
  uint8_t *device_literals = nullptr;
  uint32_t *device_costs = nullptr;
  offset_type *device_hash_tables = nullptr;
  hash_chain<offset_type> *device_hash_tables_with_chain = nullptr;
  [[maybe_unused]] uint8_t *device_sa_temp = nullptr;
  offset_type **device_sa_ptrs = nullptr;
  offset_type *device_sa = nullptr;
  sa::U16 **device_lcp_ptrs_u16 = nullptr;
  sa::U16 *device_lcp_u16 = nullptr;
  sa::U8 **device_lcp_ptrs_u8 = nullptr;
  sa::U8 *device_lcp_u8 = nullptr;
  offset_type **device_inv_sa_ptrs = nullptr;
  offset_type *device_inv_sa = nullptr;

  const gdeflate_compression_internal_algo algo_ = map_to_internal_algo(algo);

  // to make sure we don't overflow during multiplications
  const size_t batch_size_u64 = batch_size;

  // Compute pointers for all the temp variables
  auto scratch = reinterpret_cast<uint8_t *>(temp_ptr);
  size_t temp_counter = 0;
  size_t total_input_bytes = max_chunk_size * batch_size_u64;

  device_length_ptrs = reinterpret_cast<offset_type **>(scratch + temp_counter);
  temp_counter += roundUpTo(batch_size_u64 * sizeof(offset_type *), sizeof(void *));

  device_distance_ptrs = reinterpret_cast<offset_type **>(scratch + temp_counter);
  temp_counter += roundUpTo(batch_size_u64 * sizeof(offset_type *), sizeof(void *));

  device_literal_ptrs = reinterpret_cast<uint8_t **>(scratch + temp_counter);
  temp_counter += roundUpTo(batch_size_u64 * sizeof(uint8_t *), sizeof(void *));

  if (algo_ == OPTIMAL_PARSE or algo_ == OPTIMAL_PARSE_L6)
  {
    device_cost_ptrs = reinterpret_cast<uint32_t **>(scratch + temp_counter);
    temp_counter += roundUpTo(batch_size_u64 * sizeof(uint32_t *), sizeof(void *));
  }

  const size_t ptrs_end = temp_counter;

  num_symbols = reinterpret_cast<unsigned int *>(scratch + temp_counter);
  temp_counter += roundUpTo(batch_size_u64 * sizeof(unsigned int), sizeof(void *));

  num_literals = reinterpret_cast<unsigned int *>(scratch + temp_counter);
  temp_counter += roundUpTo(batch_size_u64 * sizeof(unsigned int), sizeof(void *));

  device_lengths = reinterpret_cast<offset_type *>(scratch + temp_counter);
  temp_counter += roundUpTo(total_input_bytes * sizeof(offset_type), sizeof(void *));

  device_distances = reinterpret_cast<offset_type *>(scratch + temp_counter);
  temp_counter += roundUpTo(total_input_bytes * sizeof(offset_type), sizeof(void *));

  device_literals = reinterpret_cast<uint8_t *>(scratch + temp_counter);
  temp_counter += roundUpTo(total_input_bytes * sizeof(uint8_t), sizeof(void *));

  switch (algo_)
  {
    case OPTIMAL_PARSE: {
      device_costs = reinterpret_cast<uint32_t *>(scratch + temp_counter);
      temp_counter += roundUpTo(total_input_bytes * sizeof(uint32_t), sizeof(void *));

      // The suffix array sorting shares the temp space we use for the LZ buffers
      // because we sort before we do LZ and so that memory is free to use
      device_sa_temp = scratch + ptrs_end;

      size_t lz_temp_space = temp_counter - ptrs_end;
      size_t sa_temp_space =
        roundUpTo(sa::saGetTempStorageSize<offset_type>(max_chunk_size, batch_size), sizeof(void *));

      temp_counter = ptrs_end + max(lz_temp_space, sa_temp_space);

      device_sa_ptrs = reinterpret_cast<offset_type **>(scratch + temp_counter);
      temp_counter += roundUpTo(batch_size_u64 * sizeof(offset_type *), sizeof(void *));

      device_sa = reinterpret_cast<offset_type *>(scratch + temp_counter);
      temp_counter += roundUpTo(total_input_bytes * sizeof(offset_type), sizeof(void *));

      device_lcp_ptrs_u16 = reinterpret_cast<sa::U16 **>(scratch + temp_counter);
      temp_counter += roundUpTo(batch_size_u64 * sizeof(sa::U16 *), sizeof(void *));

      device_lcp_u16 = reinterpret_cast<sa::U16 *>(scratch + temp_counter);
      temp_counter += roundUpTo(total_input_bytes * sizeof(sa::U16), sizeof(void *));

      device_inv_sa_ptrs = reinterpret_cast<offset_type **>(scratch + temp_counter);
      temp_counter += roundUpTo(batch_size_u64 * sizeof(offset_type *), sizeof(void *));

      device_inv_sa = reinterpret_cast<offset_type *>(scratch + temp_counter);
      temp_counter += roundUpTo(total_input_bytes * sizeof(offset_type), sizeof(void *));
      break;
    }
    case OPTIMAL_PARSE_L6: {
      device_costs = reinterpret_cast<uint32_t *>(scratch + temp_counter);
      temp_counter += roundUpTo(total_input_bytes * sizeof(uint32_t), sizeof(void *));

      // The suffix array sorting shares the temp space we use for the LZ buffers
      // because we sort before we do LZ and so that memory is free to use
      device_sa_temp = scratch + ptrs_end;
      size_t lz_temp_space = temp_counter - ptrs_end;
      size_t sa_temp_space =
        roundUpTo(sa::saGetTempStorageSize<offset_type>(max_chunk_size, batch_size), sizeof(void *));

      temp_counter = ptrs_end + max(lz_temp_space, sa_temp_space);

      device_sa_ptrs = reinterpret_cast<offset_type **>(scratch + temp_counter);
      temp_counter += roundUpTo(batch_size_u64 * sizeof(offset_type *), sizeof(void *));

      device_sa = reinterpret_cast<offset_type *>(scratch + temp_counter);
      temp_counter += roundUpTo(total_input_bytes * sizeof(offset_type), sizeof(void *));

      device_lcp_ptrs_u8 = reinterpret_cast<sa::U8 **>(scratch + temp_counter);
      temp_counter += roundUpTo(batch_size_u64 * sizeof(sa::U8 *), sizeof(void *));

      device_lcp_u8 = reinterpret_cast<sa::U8 *>(scratch + temp_counter);
      temp_counter += roundUpTo(total_input_bytes * sizeof(sa::U8), sizeof(void *));

      device_inv_sa_ptrs = reinterpret_cast<offset_type **>(scratch + temp_counter);
      temp_counter += roundUpTo(batch_size_u64 * sizeof(offset_type *), sizeof(void *));

      device_inv_sa = reinterpret_cast<offset_type *>(scratch + temp_counter);
      temp_counter += roundUpTo(total_input_bytes * sizeof(offset_type), sizeof(void *));
      break;
    }
    case HASH_BASED:
      device_hash_tables = reinterpret_cast<offset_type *>(scratch + temp_counter);
      temp_counter += roundUpTo(batch_size_u64 * sizeof(offset_type) * HASH_TABLE_SIZE, sizeof(void *)); // Hash tables
      break;
    case HASH_WITH_CHAIN:
      device_hash_tables_with_chain = reinterpret_cast<hash_chain<offset_type> *>(scratch + temp_counter);
      temp_counter += roundUpTo(
        batch_size_u64 * sizeof(hash_chain<offset_type>) * HASH_TABLE_SIZE_WITH_CHAIN,
        sizeof(void *)
      ); // Hash tables
      break;
    case HUFFMAN_ONLY:
      temp_counter = 0; // No temp storage for entropy only
      break;
  }

  if (temp_counter > temp_bytes)
  {
    std::cerr << "Required temp bytes  : " << temp_counter << "\nAvailable temp bytes : " << temp_bytes << std::endl;
    throw NVCompException(nvcompErrorInternal, "Insufficient temporary workspace");
  }

  // Compute the temp pointers for each chunk
  if (algo_ != HUFFMAN_ONLY)
  {
    int block = 128;
    auto grid = cuda_dim_cast(roundUpDiv(batch_size, block));
    if (algo_ == OPTIMAL_PARSE_L6)
    {
      computeTempPointers<sa::U8, offset_type><<<grid, block, 0, stream>>>(
        device_lengths,
        device_distances,
        device_literals,
        device_costs,
        device_sa,
        device_inv_sa,
        device_lcp_u8,
        max_chunk_size,
        batch_size,
        device_length_ptrs,
        device_distance_ptrs,
        device_literal_ptrs,
        device_cost_ptrs,
        device_sa_ptrs,
        device_inv_sa_ptrs,
        device_lcp_ptrs_u8,
        device_num_deflate_blocks
      );
    }
    else
    {
      computeTempPointers<sa::U16, offset_type><<<grid, block, 0, stream>>>(
        device_lengths,
        device_distances,
        device_literals,
        device_costs,
        device_sa,
        device_inv_sa,
        device_lcp_u16,
        max_chunk_size,
        batch_size,
        device_length_ptrs,
        device_distance_ptrs,
        device_literal_ptrs,
        device_cost_ptrs,
        device_sa_ptrs,
        device_inv_sa_ptrs,
        device_lcp_ptrs_u16,
        device_num_deflate_blocks
      );
    }
    CUDA_CHECK(cudaGetLastError());
  }
  // Set max match length to standard deflate. No deflate64 for compression
  [[maybe_unused]] constexpr int maxMatchLength = deflateMaxMatchLength;
  switch (algo_)
  {
    case OPTIMAL_PARSE: {
      // Compress using LZ and save to intermediate representation
      sa::saBuild<deflateMaxMatchLength, sa::U16>(
        device_sa_temp,
        reinterpret_cast<const uint8_t *const *>(device_uncompressed_ptr),
        device_uncompressed_bytes,
        max_chunk_size,
        batch_size,
        device_sa_ptrs,
        device_inv_sa_ptrs,
        device_lcp_ptrs_u16,
        stream
      );
      // suffixes_per_cta determines the amount of work per CTA. Increasing will decrease parallelism across
      // CTAS, and decreasing will not give enough work to steal from to load balance.
      // 256 was a good balance to give best performance.
      constexpr size_t suffixes_per_cta = 256;
      size_t ctas_per_chunk = ceilDiv(max_chunk_size, suffixes_per_cta);

      if (ctas_per_chunk > 0)
      {
        lz_compress_longest_matches<<<cuda_dim_cast(batch_size_u64 * ctas_per_chunk), dim3(128, 1), 0, stream>>>(
          reinterpret_cast<const unsigned char *const *>(device_uncompressed_ptr),
          device_uncompressed_bytes,
          maxMatchLength,
          device_length_ptrs,
          device_distance_ptrs,
          batch_size,
          device_sa_ptrs,
          device_inv_sa_ptrs,
          device_lcp_ptrs_u16,
          narrow_cast<unsigned int>(ctas_per_chunk),
          device_num_deflate_blocks
        );
        CUDA_CHECK(cudaGetLastError());
      }

      lz_compress_optimal_parse<deflate64><<<batch_size, dim3(32, 1), 0, stream>>>(
        reinterpret_cast<const unsigned char *const *>(device_uncompressed_ptr),
        device_uncompressed_bytes,
        maxMatchLength,
        device_cost_ptrs,
        device_length_ptrs,
        device_distance_ptrs,
        device_literal_ptrs,
        num_symbols,
        num_literals,
        batch_size,
        device_num_deflate_blocks
      );
      break;
    }
    case HASH_BASED: {
      // Compress using LZ and save to intermediate representation
      lz_compress_greedy_hash<<<batch_size, WARP_SIZE, 0, stream>>>(
        reinterpret_cast<const unsigned char *const *>(device_uncompressed_ptr),
        device_uncompressed_bytes,
        device_hash_tables,
        device_length_ptrs,
        device_distance_ptrs,
        device_literal_ptrs,
        num_symbols,
        num_literals,
        batch_size,
        device_num_deflate_blocks
      );
      break;
    }
    case HASH_WITH_CHAIN: {
      // Compress using LZ and save to intermediate representation
      lz_compress_greedy_hash_with_chain<<<batch_size, WARP_SIZE, 0, stream>>>(
        reinterpret_cast<const unsigned char *const *>(device_uncompressed_ptr),
        device_uncompressed_bytes,
        device_hash_tables_with_chain,
        device_length_ptrs,
        device_distance_ptrs,
        device_literal_ptrs,
        num_symbols,
        num_literals,
        batch_size,
        device_num_deflate_blocks
      );
      break;
    }
    case OPTIMAL_PARSE_L6: {
      // Compress using LZ and save to intermediate representation
      sa::saBuild<deflateL6MaxMatchLength, sa::U8>(
        device_sa_temp,
        reinterpret_cast<const uint8_t *const *>(device_uncompressed_ptr),
        device_uncompressed_bytes,
        max_chunk_size,
        batch_size,
        device_sa_ptrs,
        device_inv_sa_ptrs,
        device_lcp_ptrs_u8,
        stream
      );

      // suffixes_per_cta determines the amount of work per CTA. Increasing will decrease parallelism across
      // CTAS, and decreasing will not give enough work to steal from to load balance.
      // 256 was a good balance to give best performance.
      constexpr size_t suffixes_per_cta = 256;
      size_t ctas_per_chunk = ceilDiv(max_chunk_size, suffixes_per_cta);

      if (ctas_per_chunk > 0)
      {
        lz_compress_longest_matches<sa::U8>
          <<<cuda_dim_cast(batch_size_u64 * ctas_per_chunk), dim3(128, 1), 0, stream>>>(
            reinterpret_cast<const unsigned char *const *>(device_uncompressed_ptr),
            device_uncompressed_bytes,
            maxMatchLength,
            device_length_ptrs,
            device_distance_ptrs,
            batch_size,
            device_sa_ptrs,
            device_inv_sa_ptrs,
            device_lcp_ptrs_u8,
            narrow_cast<unsigned int>(ctas_per_chunk),
            device_num_deflate_blocks
          );
        CUDA_CHECK(cudaGetLastError());
      }

      lz_compress_optimal_parse<deflate64><<<batch_size, dim3(32, 1), 0, stream>>>(
        reinterpret_cast<const unsigned char *const *>(device_uncompressed_ptr),
        device_uncompressed_bytes,
        maxMatchLength,
        device_cost_ptrs,
        device_length_ptrs,
        device_distance_ptrs,
        device_literal_ptrs,
        num_symbols,
        num_literals,
        batch_size,
        device_num_deflate_blocks
      );

      break;
    }
    case HUFFMAN_ONLY: {
      // Compress using Huffman only and skip LZ phase
      huffman_encode_dispatch(
        reinterpret_cast<uint32_t *const *>(device_compressed_ptr),
        device_compressed_bytes,
        device_compressed_pad_bits,
        reinterpret_cast<const unsigned char *const *>(device_uncompressed_ptr),
        device_uncompressed_bytes,
        batch_size,
        standard,
        stream,
        device_num_deflate_blocks
      );
      break;
    }
  }
  CUDA_CHECK(cudaGetLastError());

  if (algo_ != HUFFMAN_ONLY)
  {
    // Huffman encode LZ representation based and write out in swizzled order
    if (standard)
    {
      gdeflate_encode_standard<<<batch_size, WARP_SIZE, 0, stream>>>(
        reinterpret_cast<uint32_t *const *>(device_compressed_ptr),
        device_compressed_bytes,
        device_compressed_pad_bits,
        device_length_ptrs,
        device_distance_ptrs,
        device_literal_ptrs,
        num_symbols,
        num_literals,
        batch_size,
        device_num_deflate_blocks
      );
    }
    else
    {
      gdeflate_encode<<<batch_size, WARP_SIZE, 0, stream>>>(
        reinterpret_cast<uint32_t *const *>(device_compressed_ptr),
        device_compressed_bytes,
        device_length_ptrs,
        device_distance_ptrs,
        device_literal_ptrs,
        num_symbols,
        num_literals,
        batch_size,
        device_num_deflate_blocks
      );
    }
    CUDA_CHECK(cudaGetLastError());
  }
}

/**
 * @brief Perform compression.
 *
 * @param device_uncompressed_ptr The pointers on the GPU, to uncompressed batched items.
 * @param device_uncompressed_bytes The size of each uncompressed batch item on the GPU.
 * @param max_chunk_size The maximum size of a chunk.
 * @param batch_size The number of batch items.
 * @param temp_ptr The temporary GPU workspace.
 * @param temp_bytes The size of the temporary GPU workspace.
 * @param device_compressed_ptr The pointers on the GPU, to the output location for each compressed batch item (output).
 * @param device_compressed_bytes The compressed size of each chunk on the GPU (output).
 * @param algo Algorithm to use for compression
 * @param device_statuses The status of each compression batch item on the GPU.
 * @param stream The stream to operate on.
 * @param standard true for deflate, false for gdeflate
 * @param device_compressed_pad_bits amount of unused bits in last byte of each deflate block. nullptr for untracked.
 * @param device_num_deflate_blocks Optional GPU pointer to the number of deflate blocks to output,
 *                                  used for GPU-calculated kernel dim setting in gzip kernels.
 *                                  nullptr means `batch_size` is used as the number of blocks.
 */
void compressAsync(
  const void *const *device_uncompressed_ptr,
  const size_t *device_uncompressed_bytes,
  const size_t max_chunk_size,
  size_t batch_size,
  void *temp_ptr,
  size_t temp_bytes,
  void *const *device_compressed_ptr,
  size_t *device_compressed_bytes,
  gdeflate_compression_algo algo,
  nvcompStatus_t *device_statuses,
  cudaStream_t stream,
  bool standard,
  uint8_t *device_compressed_pad_bits,
  const int *device_num_deflate_blocks
)
{
  // Note: in practice this never happens as it is checked in "nvcompBatchedGdeflateCompressAsync"
  if (max_chunk_size > gdeflateMaxChunkSize)
  {
    throw NVCompException(
      nvcompErrorChunkSizeTooLarge,
      std::string("Maximum allowed chunk size for gdeflate is ") + gdeflate::gdeflateMaxChunkSizeText
    );
  }

  auto batch_size_u32 = static_cast<unsigned int>(batch_size);

  if (standard)
  {
    DeflateCompressGetMaxOutputChunkSizeKernel<<<cuda_dim_cast(roundUpDiv(batch_size, WARP_SIZE)), WARP_SIZE, 0, stream>>>(
      device_uncompressed_bytes,
      device_compressed_bytes,
      batch_size,
      device_num_deflate_blocks
    );
    CUDA_CHECK(cudaGetLastError());

    // The standard bitwriter packs bits with atomicOr, which never clears bits, so the output
    // must be pre-zeroed or partially-written words keep garbage from the caller's buffer.
    zeroOutput<<<batch_size_u32, 128, 0, stream>>>(
      device_compressed_bytes,
      batch_size,
      device_compressed_ptr,
      device_num_deflate_blocks
    );
    CUDA_CHECK(cudaGetLastError());
  }

  // mark compression successful
  try_clear_device_statuses(batch_size, device_statuses, stream);

  if (max_chunk_size <= (1 << 16))
  {
    compressAsyncTemplated<uint16_t>(
      device_uncompressed_ptr,
      device_uncompressed_bytes,
      max_chunk_size,
      batch_size_u32,
      temp_ptr,
      temp_bytes,
      device_compressed_ptr,
      device_compressed_bytes,
      device_compressed_pad_bits,
      algo,
      stream,
      standard,
      device_num_deflate_blocks
    );
  }
  else
  {
    compressAsyncTemplated<uint32_t>(
      device_uncompressed_ptr,
      device_uncompressed_bytes,
      max_chunk_size,
      batch_size_u32,
      temp_ptr,
      temp_bytes,
      device_compressed_ptr,
      device_compressed_bytes,
      device_compressed_pad_bits,
      algo,
      stream,
      standard,
      device_num_deflate_blocks
    );
  }
}

} // namespace gdeflate

// #endif
