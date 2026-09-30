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

#include <cub/cub.cuh>

#include <cuda_runtime.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>

#include "bitcomp_private.h"
#include "common.h"
#include "CorrectnessChecks.cuh"
#include "exception.hpp"
#include "HWDecompress.hpp"
#include "loadStore.h"
#include "LZ4CompressionKernels.h"
#include "LZ4Kernels.cuh"
#include "nvcomp/lz4.h"
#include "nvcomp/native/bitcomp.h"
#include "transpose.h"

#include <cooperative_groups.h>

using double_word_type = uint64_t;
using item_type = uint32_t;

namespace nvcomp
{

namespace cg = cooperative_groups;

namespace lowlevel
{

template <typename T, bool READ_INPUT_FROM_TEMP_SPACE>
__global__ void lz4CompressBatchKernel(
  const uint8_t *const *device_in_ptr,
  const size_t *const device_in_bytes,
  uint8_t *const *const device_out_ptr,
  size_t *const device_out_bytes,
  offset_type *const temp_space,
  const position_type hash_table_size,
  const unsigned int max_chunk_size
)
{
  const int bidx = blockIdx.x * blockDim.y + threadIdx.y;

  const size_t decomp_length = device_in_bytes[bidx];

  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<32>(block);

  uint8_t *const comp_ptr = device_out_ptr[bidx];
  size_t *const comp_length = device_out_bytes + bidx;

  // we can run with smaller hash tables for performance, but obviously not
  // bigger
  assert(hash_table_size <= sizeof(lz4::offset_type) * lz4::getHashTableSize(nvcompLZ4CompressionMaxAllowedChunkSize));
  const size_t tmp_size_warp = lz4::getHashTableSize(nvcompLZ4CompressionMaxAllowedChunkSize);

  if constexpr (READ_INPUT_FROM_TEMP_SPACE)
  {
    const unsigned int rounded_max_chunk_size = roundUpTo(max_chunk_size, LZ4_BITSHUFFLE_BYTES_PER_THREAD);

    const uint8_t *decomp_ptr = reinterpret_cast<const uint8_t *>(temp_space) + bidx * rounded_max_chunk_size;
    assert(reinterpret_cast<uintptr_t>(decomp_ptr) % sizeof(T) == 0 && "Input buffer not aligned");

    // Skip the first num_chunks (gridDim.x) * max_chunk_size bytes of the temp space
    // (where transformed input data is stored) to get to the hash table
    offset_type *const hash_table =
      reinterpret_cast<offset_type *>(reinterpret_cast<uint8_t *>(temp_space) + (gridDim.x * rounded_max_chunk_size)) +
      (bidx * tmp_size_warp);

    compressStream(
      comp_ptr,
      reinterpret_cast<const T *>(decomp_ptr),
      hash_table,
      hash_table_size,
      decomp_length,
      comp_length,
      warp
    );
  }
  else
  {
    const uint8_t *decomp_ptr = device_in_ptr[bidx];
    assert(reinterpret_cast<uintptr_t>(decomp_ptr) % sizeof(T) == 0 && "Input buffer not aligned");

    offset_type *const hash_table = temp_space + bidx * tmp_size_warp;

    compressStream(
      comp_ptr,
      reinterpret_cast<const T *>(decomp_ptr),
      hash_table,
      hash_table_size,
      decomp_length,
      comp_length,
      warp
    );
  }
}

template <bool CORRECTNESS_CHECK, bool WRITE_DECOMPRESSED_OUTPUT>
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1200)
__launch_bounds__(LZ4_DECOMP_THREADS_PER_CHUNK *LZ4_DECOMP_CHUNKS_PER_BLOCK)
#endif
  __global__ void lz4DecompressBatchKernel(
    const uint8_t *const *const device_in_ptrs,
    const size_t *const device_in_bytes,
    const size_t *const device_out_bytes,
    const size_t batch_size,
    uint8_t *const *const device_out_ptrs,
    size_t *device_uncompressed_bytes,
    nvcompStatus_t *device_status_ptrs,
    LZ4CorrectnessChecker<CORRECTNESS_CHECK> *device_correctness_ptrs
  )
{
  assert(blockDim.x == WARP_SIZE_U); // code below assumes there is only a single warp on dim x
  const int bid = blockIdx.x * LZ4_DECOMP_CHUNKS_PER_BLOCK + threadIdx.y;

  __shared__ lz4::LZ4DecompressWarpMemory shared_mem[LZ4_DECOMP_CHUNKS_PER_BLOCK];

  /*
  __shared__ __align__(8) uint8_t buffer[DECOMP_INPUT_BUFFER_SIZE * LZ4_DECOMP_CHUNKS_PER_BLOCK];
  //0..15 bits represent distance, 16..31 bits represent match length, 32..65 bits represent output buffer offset.
  __shared__ sequence sequences[WARP_SIZE*LZ4_DECOMP_CHUNKS_PER_BLOCK]; 
  __shared__ unsigned int ix_literal[WARP_SIZE * LZ4_DECOMP_CHUNKS_PER_BLOCK];
  __shared__ int ix_output[WARP_SIZE * LZ4_DECOMP_CHUNKS_PER_BLOCK];
  */

  assert(!WRITE_DECOMPRESSED_OUTPUT || device_out_ptrs != nullptr);
  // device_uncompressed_bytes must always be present.
  assert(device_uncompressed_bytes != nullptr);

  if constexpr (CORRECTNESS_CHECK)
  {
    assert(device_correctness_ptrs != nullptr);
  }

  if (bid < batch_size)
  {
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    uint8_t *const decomp_ptr = device_out_ptrs == nullptr ? nullptr : device_out_ptrs[bid];
    const uint8_t *const comp_ptr = device_in_ptrs[bid];
    const position_type chunk_length = static_cast<position_type>(device_in_bytes[bid]);
    const position_type output_buf_length = WRITE_DECOMPRESSED_OUTPUT
                                              ? static_cast<position_type>(device_out_bytes[bid])
                                              : UINT_MAX;

    decompressStream<CORRECTNESS_CHECK>(
      shared_mem[threadIdx.y],
      decomp_ptr,
      comp_ptr,
      chunk_length,
      output_buf_length,
      device_uncompressed_bytes + bid,
      WRITE_DECOMPRESSED_OUTPUT ? device_status_ptrs + bid : nullptr,
      WRITE_DECOMPRESSED_OUTPUT,
      warp,
      device_correctness_ptrs ? device_correctness_ptrs + bid : nullptr
    );
  }
}

/******************************************************************************
 * PUBLIC FUNCTIONS ***********************************************************
 *****************************************************************************/

void lz4BatchCompress(
  const uint8_t *const *decomp_data_device,
  const size_t *const decomp_sizes_device,
  const unsigned int max_chunk_size,
  const size_t batch_size,
  void *const temp_data,
  const size_t temp_bytes,
  uint8_t *const *const comp_data_device,
  size_t *const comp_sizes_device,
  nvcompType_t data_type,
  bool read_input_from_temp_space,
  nvcompStatus_t *device_statuses,
  cudaStream_t stream
)
{

  position_type HT_size = narrow_cast<position_type>(lz4::getHashTableSize(static_cast<size_t>(max_chunk_size)));

  const size_t total_required_temp = batch_size * HT_size * sizeof(offset_type);
  if (temp_bytes < total_required_temp)
  {
    throw NVCompException(
      nvcompErrorInternal,
      "Insufficient temp space: got " + std::to_string(temp_bytes) + " bytes, but need " +
        std::to_string(total_required_temp) + " bytes."
    );
  }

  // mark compression successful
  try_clear_device_statuses(batch_size, device_statuses, stream);

  const dim3 grid(nvcomp::cuda_dim_cast(batch_size));
  const dim3 block(LZ4_COMP_THREADS_PER_CHUNK, LZ4_COMP_CHUNKS_PER_BLOCK);

  decltype(&lz4CompressBatchKernel<uint8_t, true>) compress_kernel;
  switch (data_type)
  {
    case NVCOMP_TYPE_BITS:
    case NVCOMP_TYPE_CHAR:
    case NVCOMP_TYPE_UCHAR:
      compress_kernel = read_input_from_temp_space ? lz4CompressBatchKernel<uint8_t, true>
                                                   : lz4CompressBatchKernel<uint8_t, false>;
      break;
    case NVCOMP_TYPE_SHORT:
    case NVCOMP_TYPE_USHORT:
      compress_kernel = read_input_from_temp_space ? lz4CompressBatchKernel<uint16_t, true>
                                                   : lz4CompressBatchKernel<uint16_t, false>;
      break;
    case NVCOMP_TYPE_INT:
    case NVCOMP_TYPE_UINT:
      compress_kernel = read_input_from_temp_space ? lz4CompressBatchKernel<uint32_t, true>
                                                   : lz4CompressBatchKernel<uint32_t, false>;
      break;
    default:
      throw NVCompException(nvcompErrorNotSupported, "Unsupported input data type");
  }
  compress_kernel<<<grid, block, 0, stream>>>(
    decomp_data_device,
    decomp_sizes_device,
    comp_data_device,
    comp_sizes_device,
    static_cast<offset_type *>(temp_data),
    HT_size,
    max_chunk_size
  );
  CUDA_CHECK(cudaGetLastError());
}

__global__ void init_cudecomp_sort(
  const void *const *device_in_ptr,
  const size_t *device_in_bytes,
  const size_t *device_buffer_bytes,
  nvcompStatus_t *device_statuses,
  size_t num_chunks,
  int *sort_keys,
  int *sort_vals
)
{
  const int ix_chunk = blockIdx.x * blockDim.x + threadIdx.x;
  if (ix_chunk >= num_chunks)
  {
    return;
  }
  sort_vals[ix_chunk] = ix_chunk;

  // Use buffer size for sorting since LZ4 cannot get uncompressed size from compressed data
  // Use 4 byte values to improve sort efficiency --
  // we won't be calling decomp engine with > 2 GB buffers anyway
  sort_keys[ix_chunk] = static_cast<int>(device_buffer_bytes[ix_chunk]);
}

__global__ void fill_cudecomp_params(
  const void *const *device_in_ptr,
  const size_t *device_in_bytes,
  void *const *device_out_ptr,
  size_t *device_out_bytes,
  CUmemDecompressParams *de_params,
  nvcompStatus_t *device_statuses,
  size_t num_chunks,
  int *sort_vals
)
{
  const int ix_chunk = blockIdx.x * blockDim.x + threadIdx.x;
  if (ix_chunk >= num_chunks)
  {
    return;
  }

  // If sort_vals is nullptr, use original index, otherwise use sorted index
  int sorted_ix_chunk = (sort_vals != nullptr) ? sort_vals[ix_chunk] : ix_chunk;
  const size_t dev_in_bytes = device_in_bytes[sorted_ix_chunk];

  // LZ4 doesn't have a header to parse, so we use the entire compressed data
  de_params[ix_chunk].src = device_in_ptr[sorted_ix_chunk];
  de_params[ix_chunk].dst = device_out_ptr[sorted_ix_chunk];
  device_out_bytes[sorted_ix_chunk] = 0; // WAR for HW only setting 32 lsb for this value
  de_params[ix_chunk].dstActBytes = reinterpret_cast<cuuint32_t *>(&device_out_bytes[sorted_ix_chunk]);
  de_params[ix_chunk].srcNumBytes = dev_in_bytes;
  de_params[ix_chunk].dstNumBytes = 0;
  de_params[ix_chunk].algo = CU_MEM_DECOMPRESS_ALGORITHM_LZ4;

  device_statuses[sorted_ix_chunk] = nvcompSuccess;
}

void LZ4FillCuDecompParams(
  const void *const *device_in_ptr,
  const size_t *device_in_bytes,
  void *const *device_out_ptr,
  CUmemDecompressParams *de_params,
  size_t *device_out_bytes,
  const size_t *device_buffer_bytes,
  size_t num_chunks,
  nvcompStatus_t *device_statuses,
  uint8_t *scratch_allocation,
  size_t scratch_size,
  bool use_sorting,
  cudaStream_t stream
)
{
  int block_size = 32;
  int grid_dim = nvcomp::roundUpDiv(nvcomp::narrow_cast<int>(num_chunks), block_size);

  int *sort_ix_out = nullptr;
  if (use_sorting)
  {
    int *sort_keys_in = roundUpToAlignment<int>(scratch_allocation);
    int *sort_keys_out = sort_keys_in + num_chunks;
    int *sort_ix_in = sort_keys_out + num_chunks;
    int *sort_ix_out = sort_ix_in + num_chunks;
    uint8_t *sort_scratch = reinterpret_cast<uint8_t *>(sort_ix_out + num_chunks);
    scratch_size -= (4 * sizeof(int) * num_chunks + sizeof(int) - 1);

    init_cudecomp_sort<<<grid_dim, block_size, 0, stream>>>(
      device_in_ptr,
      device_in_bytes,
      device_buffer_bytes,
      device_statuses,
      num_chunks,
      sort_keys_in,
      sort_ix_in
    );
    CUDA_CHECK(cudaGetLastError());

    cub::DeviceRadixSort::SortPairsDescending(
      sort_scratch,
      scratch_size,
      sort_keys_in,
      sort_keys_out,
      sort_ix_in,
      sort_ix_out,
      num_chunks,
      0,
      32,
      stream
    );
  }

  fill_cudecomp_params<<<grid_dim, block_size, 0, stream>>>(
    device_in_ptr,
    device_in_bytes,
    device_out_ptr,
    device_out_bytes,
    de_params,
    device_statuses,
    num_chunks,
    sort_ix_out
  );
  CUDA_CHECK(cudaGetLastError());
}

template <bool CORRECTNESS_CHECK>
void lz4BatchDecompress(
  const uint8_t *const *const device_in_ptrs,
  const size_t *const device_in_bytes,
  const size_t *const device_out_bytes,
  const size_t batch_size,
  uint8_t *const *const device_out_ptrs,
  size_t *device_actual_uncompressed_bytes,
  nvcompStatus_t *device_status_ptrs,
  void *device_correctness_ptrs,
  cudaStream_t stream
)
{
  const dim3 grid(nvcomp::cuda_dim_cast(roundUpDiv(batch_size, LZ4_DECOMP_CHUNKS_PER_BLOCK)));
  const dim3 block(LZ4_DECOMP_THREADS_PER_CHUNK, LZ4_DECOMP_CHUNKS_PER_BLOCK);

  if constexpr (CORRECTNESS_CHECK)
  {
    // The entire correctness-checker array is copied to the host wholesale in
    // printErrors(). Zero it first so that chunks which record no error (and any
    // struct padding) are not flagged by `compute-sanitizer --tool initcheck` as
    // uninitialized cudaMemcpy-source reads. Each chunk's Initialize() (run by
    // the kernel below) then sets line_number_ = -1 to mark "no error".
    CUDA_CHECK(
      cudaMemsetAsync(device_correctness_ptrs, 0, batch_size * sizeof(LZ4CorrectnessChecker<CORRECTNESS_CHECK>), stream)
    );
  }

  lz4DecompressBatchKernel<CORRECTNESS_CHECK, true><<<grid, block, 0, stream>>>(
    device_in_ptrs,
    device_in_bytes,
    device_out_bytes,
    batch_size,
    device_out_ptrs,
    device_actual_uncompressed_bytes,
    device_status_ptrs,
    reinterpret_cast<LZ4CorrectnessChecker<CORRECTNESS_CHECK> *>(device_correctness_ptrs)
  );
  CUDA_CHECK(cudaGetLastError());

  LZ4CorrectnessChecker<CORRECTNESS_CHECK>::printErrors(
    reinterpret_cast<LZ4CorrectnessChecker<CORRECTNESS_CHECK> *>(device_correctness_ptrs),
    batch_size,
    stream
  );
}

template void lz4BatchDecompress<true>(
  const uint8_t *const *const,
  const size_t *const,
  const size_t *const,
  const size_t,
  uint8_t *const *const,
  size_t *,
  nvcompStatus_t *,
  void *,
  cudaStream_t
);

template void lz4BatchDecompress<false>(
  const uint8_t *const *const,
  const size_t *const,
  const size_t *const,
  const size_t,
  uint8_t *const *const,
  size_t *,
  nvcompStatus_t *,
  void *,
  cudaStream_t
);

void lz4BatchGetDecompressSizes(
  const uint8_t *const *device_compressed_ptrs,
  const size_t *device_compressed_bytes,
  size_t *device_uncompressed_bytes,
  size_t batch_size,
  cudaStream_t stream
)
{
  const dim3 grid(nvcomp::cuda_dim_cast(roundUpDiv(batch_size, LZ4_DECOMP_CHUNKS_PER_BLOCK)));
  const dim3 block(LZ4_DECOMP_THREADS_PER_CHUNK, LZ4_DECOMP_CHUNKS_PER_BLOCK);

  lz4DecompressBatchKernel<false, false><<<grid, block, 0, stream>>>(
    device_compressed_ptrs,
    device_compressed_bytes,
    nullptr,
    batch_size,
    nullptr,
    device_uncompressed_bytes,
    nullptr,
    nullptr
  );
  CUDA_CHECK(cudaGetLastError());
}

size_t lz4ComputeChunksInBatch(const size_t *const decomp_data_size, const size_t batch_size, const size_t chunk_size)
{
  size_t num_chunks = 0;

  for (size_t i = 0; i < batch_size; ++i)
  {
    num_chunks += roundUpDiv(decomp_data_size[i], chunk_size);
  }

  return num_chunks;
}

size_t lz4BatchCompressComputeTempSize(const size_t max_uncomp_chunk_size, const size_t batch_size)
{
  // Note:
  // It seems, for performance reasons, nvcompLZ4CompressionMaxAllowedChunkSize
  // was hardcoded in the old nvCOMP Device API, and through the device API
  // this trickled down to both the temp size calculation and also to the
  // lz4CompressBatchKernel CUDA kernel.
  (void)max_uncomp_chunk_size;
  const size_t tmp_size_per_group =
    sizeof(lz4::offset_type) *
    lz4::getHashTableSize(/* max_uncomp_chunk_size */ nvcompLZ4CompressionMaxAllowedChunkSize);
  return batch_size * tmp_size_per_group;
}

size_t lz4ComputeMaxSize(const size_t size)
{
  if (size > lz4MaxChunkSize())
  {
    throw NVCompException(
      nvcompErrorChunkSizeTooLarge,
      "Maximum chunk size for LZ4 is " + std::to_string(lz4MaxChunkSize())
    );
  }
  return maxSizeOfStream(size);
}

size_t lz4MaxChunkSize() { return nvcompLZ4CompressionMaxAllowedChunkSize; }

template <typename T>
inline __device__ void loadInputToSharedMemoryForBitPlanTransposeBack(
  uint8_t *out_ptr,
  uint4 *sm,
  size_t num_bytes,
  size_t num_elements,
  bool aligned
)
{
  // Because inverse bitshuffle loads 4 bytes at a time from shared memory,
  // by padding with 0 to the next multiple of 4/sizeof(T), we can load the next
  // 4/sizeof(T) bytes without changing the transpose logic.
  // The effect would be that the unbitshuffled data is padded with 0s.
  // We then would not copy those bytes from shared memory to the output.
  // num_elements is guaranteed below int range, so this cast is safe.
  int num_bytes_per_bit_plane = static_cast<int>(num_elements / 8);
  const int bytes_per_thread_per_bitplane = 4 / sizeof(T);
  int num_padding_bytes = roundUpTo(num_bytes_per_bit_plane, bytes_per_thread_per_bitplane) - num_bytes_per_bit_plane;

  if (num_padding_bytes == 0)
  {
    bitcomp::loadStore::loadInputToShared<T, BITCOMP_LOSSLESS, BITCOMP_DEFAULT_FORMAT>(out_ptr, sm, num_bytes, aligned);
  }
  else
  {
    // Load the data byte by byte for each bitplane
    const int elem_bits = sizeof(T) * 8;
    int num_sub_blocks = roundUpDiv(num_bytes_per_bit_plane, LZ4_BITSHUFFLE_THREADS_PER_BLOCK);
    uint8_t *sm_bytes = reinterpret_cast<uint8_t *>(sm);
    for (int i = 0; i < elem_bits; i++) // loop over bit plane
    {
      uint8_t *bit_plane_ptr = out_ptr + i * num_bytes_per_bit_plane;
      uint8_t *sm_bit_plane_ptr = sm_bytes + i * (num_bytes_per_bit_plane + num_padding_bytes);
      for (int j = 0; j < num_sub_blocks; j++)
      {
        int offset = threadIdx.x + j * LZ4_BITSHUFFLE_THREADS_PER_BLOCK;
        if (offset < num_bytes_per_bit_plane)
        {
          sm_bit_plane_ptr[offset] = bit_plane_ptr[offset];
        }
      }
      if (threadIdx.x < num_padding_bytes)
      {
        sm_bit_plane_ptr[num_bytes_per_bit_plane + threadIdx.x] = 0;
      }
    }
  }
}

template <typename T>
inline __device__ void
storeSharedToOutputForBitPlanTranspose(uint8_t *out_ptr, uint4 *sm, size_t num_bytes, size_t num_elements, bool aligned)
{
  int num_bytes_per_bit_plane = num_elements / 8;
  const int bytes_per_thread_per_bitplane = 4 / sizeof(T);
  int num_padding_bytes = roundUpTo(num_bytes_per_bit_plane, bytes_per_thread_per_bitplane) - num_bytes_per_bit_plane;

  if (num_padding_bytes == 0)
  {
    bitcomp::loadStore::storeSharedToOutput<T, BITCOMP_LOSSLESS, BITCOMP_DEFAULT_FORMAT, false>(
      out_ptr,
      sm,
      0,
      num_bytes,
      aligned
    );
  }
  else
  {
    // Load the data byte by byte for each bitplane
    const int elem_bits = sizeof(T) * 8;
    // number of sub-blocks in the bitplane
    int num_sub_blocks = roundUpDiv(num_bytes_per_bit_plane, LZ4_BITSHUFFLE_THREADS_PER_BLOCK);
    uint8_t *sm_bytes = reinterpret_cast<uint8_t *>(sm);
    for (int i = 0; i < elem_bits; i++) // loop over bit plane
    {
      uint8_t *bit_plane_ptr = out_ptr + i * num_bytes_per_bit_plane;
      uint8_t *sm_bit_plane_ptr = sm_bytes + i * (num_bytes_per_bit_plane + num_padding_bytes);
      for (int j = 0; j < num_sub_blocks; j++)
      {
        int offset = threadIdx.x + j * LZ4_BITSHUFFLE_THREADS_PER_BLOCK;
        if (offset < num_bytes_per_bit_plane)
        {
          bit_plane_ptr[offset] = sm_bit_plane_ptr[offset];
        }
      }
    }
  }
}

template <typename T>
__global__ void inverseBitshuffleKernel(
  uint8_t *const *device_out_ptrs,
  size_t *device_uncompressed_chunk_bytes,
  size_t num_chunks,
  bool msb_first
)
{
  uint8_t *out_ptr = device_out_ptrs[blockIdx.x];
  // uncompressed size in bytes
  size_t num_bytes = device_uncompressed_chunk_bytes[blockIdx.x];

  // One thread in loadStore loads 2 x uint4 (32 bytes) values from global to shared memory
  // and we have 256 threads per block => 8192 bytes loaded into shared memory per block
  __shared__ uint4 sm[LZ4_BITSHUFFLE_SHARED_MEMORY_SIZE_IN_UINT4];

  int blockStart = 0;
  int blockEnd = 0;

  int num_blocks = num_bytes / LZ4_BITSHUFFLE_MAX_BLOCK_SIZE;
  int num_leftover_bytes = num_bytes % (LZ4_BITSHUFFLE_BLOCK_ELEMENTS_MULTIPLIER * sizeof(T));
  size_t last_block_size = (num_bytes % LZ4_BITSHUFFLE_MAX_BLOCK_SIZE) - num_leftover_bytes;

  // Process each full block
  for (int i = 0; i < num_blocks; i++)
  {
    blockEnd = blockStart + LZ4_BITSHUFFLE_MAX_BLOCK_SIZE;
    bool aligned = (((uintptr_t)out_ptr & 0xf) | (LZ4_BITSHUFFLE_MAX_BLOCK_SIZE & 0xf)) == 0;

    // Load input to shared memory
    bitcomp::loadStore::loadInputToShared<T, BITCOMP_LOSSLESS, BITCOMP_DEFAULT_FORMAT>(
      out_ptr,
      sm,
      LZ4_BITSHUFFLE_MAX_BLOCK_SIZE,
      aligned
    );

    __syncthreads();

    // Backward bit-plan transpose in shared memory
    if (msb_first)
    {
      bitcomp::transpose::bitPlanTransposeBack<sizeof(T), true>(sm, LZ4_BITSHUFFLE_THREADS_PER_BLOCK);
    }
    else
    {
      bitcomp::transpose::bitPlanTransposeBack<sizeof(T), false>(sm, LZ4_BITSHUFFLE_THREADS_PER_BLOCK);
    }

    __syncthreads();

    // Save to global memory
    bitcomp::loadStore::storeSharedToOutput<T, BITCOMP_LOSSLESS, BITCOMP_DEFAULT_FORMAT, false>(
      out_ptr,
      sm,
      blockStart,
      blockEnd,
      aligned
    );

    __syncthreads();

    out_ptr += LZ4_BITSHUFFLE_MAX_BLOCK_SIZE;
  }

  // Process the partial block
  if (last_block_size > 0)
  {
    size_t num_elements_to_process = last_block_size / sizeof(T);
    blockEnd = blockStart + last_block_size;
    int worker_threads = roundUpDiv(last_block_size, LZ4_INVERSE_BITSHUFFLE_BYTES_PER_THREAD);
    bool aligned = (((uintptr_t)out_ptr & 0xf) | (last_block_size & 0xf)) == 0;

    // this would load the bitshuffled data with bit-plane padding
    loadInputToSharedMemoryForBitPlanTransposeBack<T>(out_ptr, sm, last_block_size, num_elements_to_process, aligned);

    __syncthreads();

    // Backward bit-plan transpose in shared memory
    if (msb_first)
    {
      bitcomp::transpose::bitPlanTransposeBack<sizeof(T), true>(sm, worker_threads);
    }
    else
    {
      bitcomp::transpose::bitPlanTransposeBack<sizeof(T), false>(sm, worker_threads);
    }

    __syncthreads();

    // Save to global memory
    bitcomp::loadStore::storeSharedToOutput<T, BITCOMP_LOSSLESS, BITCOMP_DEFAULT_FORMAT, false>(
      out_ptr,
      sm,
      blockStart,
      blockEnd,
      aligned
    );

    __syncthreads();

    out_ptr += last_block_size;
  }

  // Nothing to do for leftover bytes - they will be in the output as is.
}

template <typename T>
__global__ void bitshuffleKernel(
  const uint8_t *const *device_in_ptrs,
  const size_t *const device_uncompressed_chunk_bytes,
  void *device_temp_ptr,
  size_t num_chunks,
  const unsigned int max_chunk_size,
  bool msb_first
)
{
  const int ix_chunk = blockIdx.x;

  const uint8_t *in_ptr = device_in_ptrs[ix_chunk];
  const size_t num_bytes = device_uncompressed_chunk_bytes[ix_chunk];

  // With bitshuffle, we are writing to scratch space with each chunk equal to the
  // size of the max chunk size rounded up to the next multiple of LZ4_BITSHUFFLE_BYTES_PER_THREAD
  const unsigned int rounded_max_chunk_size = roundUpTo(max_chunk_size, LZ4_BITSHUFFLE_BYTES_PER_THREAD);
  uint8_t *out_ptr = static_cast<uint8_t *>(device_temp_ptr) + ix_chunk * rounded_max_chunk_size;

  __shared__ uint4 sm[LZ4_BITSHUFFLE_SHARED_MEMORY_SIZE_IN_UINT4];

  int blockStart = 0;
  int blockEnd = 0;

  int num_blocks = num_bytes / LZ4_BITSHUFFLE_MAX_BLOCK_SIZE;
  int num_leftover_bytes = num_bytes % (LZ4_BITSHUFFLE_BLOCK_ELEMENTS_MULTIPLIER * sizeof(T));
  size_t last_block_size = (num_bytes % LZ4_BITSHUFFLE_MAX_BLOCK_SIZE) - num_leftover_bytes;

  for (int i = 0; i < num_blocks; i++)
  {
    blockEnd = blockStart + LZ4_BITSHUFFLE_MAX_BLOCK_SIZE;
    bool aligned = (((uintptr_t)in_ptr & 0xf) | (LZ4_BITSHUFFLE_MAX_BLOCK_SIZE & 0xf)) == 0;

    // Load input to shared memory
    bitcomp::loadStore::loadInputToShared<T, BITCOMP_LOSSLESS, BITCOMP_DEFAULT_FORMAT>(
      in_ptr,
      sm,
      LZ4_BITSHUFFLE_MAX_BLOCK_SIZE,
      aligned
    );

    __syncthreads();

    // Backward bit-plan transpose in shared memory
    if (msb_first)
    {
      bitcomp::transpose::bitPlanTranspose<sizeof(T), false, true, true>(sm, LZ4_BITSHUFFLE_THREADS_PER_BLOCK);
    }
    else
    {
      bitcomp::transpose::bitPlanTranspose<sizeof(T), false, false, true>(sm, LZ4_BITSHUFFLE_THREADS_PER_BLOCK);
    }

    __syncthreads();

    // Save to global memory
    aligned = (((uintptr_t)out_ptr & 0xf) | (LZ4_BITSHUFFLE_MAX_BLOCK_SIZE & 0xf)) == 0;
    bitcomp::loadStore::storeSharedToOutput<T, BITCOMP_LOSSLESS, BITCOMP_DEFAULT_FORMAT, false>(
      out_ptr,
      sm,
      blockStart,
      blockEnd,
      aligned
    );

    __syncthreads();

    in_ptr += LZ4_BITSHUFFLE_MAX_BLOCK_SIZE;
    out_ptr += LZ4_BITSHUFFLE_MAX_BLOCK_SIZE;
  }

  // Process the partial block
  // Since each thread is processing LZ4_BITSHUFFLE_BYTES_PER_THREAD bytes,
  // we need to proceed as usual until the bitPlanTranspose is done.
  // For the copy to output, we need to use the loadSharedToOutputForBitPlanTranspose function as it
  // allows us to discard the zeros in each bit plane.
  if (last_block_size > 0)
  {
    size_t num_elements_to_process = last_block_size / sizeof(T);
    blockEnd = blockStart + last_block_size;
    const int worker_threads = roundUpDiv(last_block_size, LZ4_BITSHUFFLE_BYTES_PER_THREAD);
    bool aligned = (((uintptr_t)in_ptr & 0xf) | (last_block_size & 0xf)) == 0;

    // Load input to shared memory with
    bitcomp::loadStore::loadInputToShared<T, BITCOMP_LOSSLESS, BITCOMP_DEFAULT_FORMAT>(
      in_ptr,
      sm,
      last_block_size,
      aligned
    );

    __syncthreads();

    // Backward bit-plan transpose in shared memory
    if (msb_first)
    {
      bitcomp::transpose::bitPlanTranspose<sizeof(T), false, true, true>(sm, worker_threads);
    }
    else
    {
      bitcomp::transpose::bitPlanTranspose<sizeof(T), false, false, true>(sm, worker_threads);
    }

    __syncthreads();

    // Save to global memory
    aligned = (((uintptr_t)out_ptr & 0xf) | (last_block_size & 0xf)) == 0;
    storeSharedToOutputForBitPlanTranspose<T>(out_ptr, sm, last_block_size, num_elements_to_process, aligned);

    __syncthreads();

    in_ptr += last_block_size;
    out_ptr += last_block_size;
  }

  // Process the leftover bytes (<8)
  if (threadIdx.x < num_leftover_bytes)
  {
    out_ptr[threadIdx.x] = in_ptr[threadIdx.x];
  }
}

void doInverseBitshuffle(
  uint8_t *const *device_out_ptrs,
  size_t *device_uncompressed_chunk_bytes,
  size_t num_chunks,
  nvcompType_t data_type,
  bool msb_first,
  cudaStream_t stream
)
{
  const dim3 grid(nvcomp::cuda_dim_cast(num_chunks));
  const dim3 block(LZ4_BITSHUFFLE_THREADS_PER_BLOCK);

  decltype(&inverseBitshuffleKernel<uint8_t>) kernel;
  switch (data_type)
  {
    case NVCOMP_TYPE_BITS:
    case NVCOMP_TYPE_CHAR:
    case NVCOMP_TYPE_UCHAR:
      kernel = inverseBitshuffleKernel<uint8_t>;
      break;
    case NVCOMP_TYPE_SHORT:
    case NVCOMP_TYPE_USHORT:
      kernel = inverseBitshuffleKernel<uint16_t>;
      break;
    case NVCOMP_TYPE_INT:
    case NVCOMP_TYPE_UINT:
      kernel = inverseBitshuffleKernel<uint32_t>;
      break;
    default:
      throw NVCompException(nvcompErrorNotSupported, "Unsupported data type");
  }
  kernel<<<grid, block, 0, stream>>>(device_out_ptrs, device_uncompressed_chunk_bytes, num_chunks, msb_first);
  CUDA_CHECK(cudaGetLastError());
}

void doBitshuffle(
  const uint8_t *const *device_in_ptrs,
  const size_t *const device_uncompressed_chunk_bytes,
  void *device_temp_ptr,
  size_t num_chunks,
  const unsigned int max_chunk_size,
  nvcompType_t data_type,
  bool msb_first,
  cudaStream_t stream
)
{
  const dim3 grid(nvcomp::cuda_dim_cast(num_chunks));
  const dim3 block(LZ4_BITSHUFFLE_THREADS_PER_BLOCK);

  decltype(&bitshuffleKernel<uint8_t>) kernel;
  switch (data_type)
  {
    case NVCOMP_TYPE_BITS:
    case NVCOMP_TYPE_CHAR:
    case NVCOMP_TYPE_UCHAR:
      kernel = bitshuffleKernel<uint8_t>;
      break;
    case NVCOMP_TYPE_SHORT:
    case NVCOMP_TYPE_USHORT:
      kernel = bitshuffleKernel<uint16_t>;
      break;
    case NVCOMP_TYPE_INT:
    case NVCOMP_TYPE_UINT:
      kernel = bitshuffleKernel<uint32_t>;
      break;
    default:
      throw NVCompException(nvcompErrorNotSupported, "Unsupported data type");
  }
  kernel<<<grid, block, 0, stream>>>(
    device_in_ptrs,
    device_uncompressed_chunk_bytes,
    device_temp_ptr,
    num_chunks,
    max_chunk_size,
    msb_first
  );
  CUDA_CHECK(cudaGetLastError());
}

} // namespace lowlevel
} // namespace nvcomp
