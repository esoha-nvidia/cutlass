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

#include "common.h"
#include "CorrectnessChecks.cuh"
#include "CudaUtils.h"
#include "exception.hpp"
#include "lowlevel/SnappyBatchKernels.h"
#include "snappy/constants.cuh"
#include "snappy/util.cuh"
#include "SnappyKernels.cuh"

namespace nvcomp
{

/**
 * @brief Snappy compression kernel
 * See http://github.com/google/snappy/blob/master/format_description.txt
 *
 * @param[in] inputs Source/Destination buffer information per block
 * @param[out] outputs Compression status per block
 * @param[in] count Number of blocks to compress
 **/
__global__ void __launch_bounds__(COMP_THREADS_PER_BLOCK) snap_kernel(
  const void *const *__restrict__ device_in_ptr,
  const uint64_t *__restrict__ device_in_bytes,
  void *const *__restrict__ device_out_ptr,
  const uint64_t *__restrict__ device_out_available_bytes,
  nvcompStatus_t *__restrict__ outputs,
  uint64_t *device_out_bytes
)
{
  const int ix_chunk = blockIdx.x;
  do_snap(
    reinterpret_cast<const uint8_t *>(device_in_ptr[ix_chunk]),
    device_in_bytes[ix_chunk],
    reinterpret_cast<uint8_t *>(device_out_ptr[ix_chunk]),
    device_out_available_bytes ? device_out_available_bytes[ix_chunk] : 0,
    outputs ? &outputs[ix_chunk] : nullptr,
    &device_out_bytes[ix_chunk]
  );
}

// TODO: Check the error in snappy block and if -1 return 0 for uncompressed chunk sizes
__global__ void __launch_bounds__(32) get_uncompressed_sizes_kernel(
  const void *const *__restrict__ device_in_ptr,
  const uint64_t *__restrict__ device_in_bytes,
  uint64_t *__restrict__ device_out_bytes
)
{
  int t = threadIdx.x;
  int strm_id = blockIdx.x;

  if (t == 0)
  {
    uint32_t uncompressed_size = 0;
    const uint8_t *cur = reinterpret_cast<const uint8_t *>(device_in_ptr[strm_id]);
    const uint8_t *end = cur + device_in_bytes[strm_id];
    if (cur < end)
    {
      int32_t error;
      uint32_t header_bytes;
      uncompressed_size = snappy::get_uncompressed_size(cur, device_in_bytes[strm_id], header_bytes, error);
    }
    device_out_bytes[strm_id] = uncompressed_size;
  }
}

/**
 * @brief Snappy decompression kernel
 * See http://github.com/google/snappy/blob/master/format_description.txt
 *
 * blockDim {DECOMP_THREADS_PER_BLOCK,1,1}
 *
 * @param[in] inputs Source & destination information per block
 * @param[out] outputs Decompression status per block
 **/
template <bool CORRECTNESS_CHECK>
__global__ void __launch_bounds__(DECOMP_THREADS_PER_BLOCK) unsnap_kernel(
  const void *const *__restrict__ device_in_ptr,
  const uint64_t *__restrict__ device_in_bytes,
  void *const *__restrict__ device_out_ptr,
  const uint64_t *__restrict__ device_out_available_bytes,
  nvcompStatus_t *const __restrict__ outputs,
  snappy::SnappyCorrectnessChecker<CORRECTNESS_CHECK> *__restrict__ device_correctness_ptrs,
  uint64_t *__restrict__ device_out_bytes
)
{
  const int ix_chunk = blockIdx.x;
  if constexpr (CORRECTNESS_CHECK)
  {
    assert(device_correctness_ptrs != nullptr and outputs != nullptr);
  }
  snappy::do_unsnap<CORRECTNESS_CHECK>(
    reinterpret_cast<const uint8_t *>(device_in_ptr[ix_chunk]),
    device_in_bytes[ix_chunk],
    reinterpret_cast<uint8_t *>(device_out_ptr[ix_chunk]),
    device_out_available_bytes ? device_out_available_bytes[ix_chunk] : 0,
    &outputs[ix_chunk],
    device_correctness_ptrs ? &device_correctness_ptrs[ix_chunk] : nullptr,
    &device_out_bytes[ix_chunk]
  );
}

void gpu_snap(
  const void *const *device_in_ptr,
  const size_t *device_in_bytes,
  void *const *device_out_ptr,
  const size_t *device_out_available_bytes,
  nvcompStatus_t *outputs,
  size_t *device_out_bytes,
  size_t count,
  cudaStream_t stream
)
{
  dim3 dim_block(COMP_THREADS_PER_BLOCK, 1);
  dim3 dim_grid(nvcomp::cuda_dim_cast(count), 1);
  if (count > 0)
  {
    snap_kernel<<<dim_grid, dim_block, 0, stream>>>(
      device_in_ptr,
      device_in_bytes,
      device_out_ptr,
      device_out_available_bytes,
      outputs,
      device_out_bytes
    );
  }
  CUDA_CHECK(cudaGetLastError());
}

template <bool CORRECTNESS_CHECK>
void gpu_unsnap(
  const void *const *device_in_ptr,
  const size_t *device_in_bytes,
  void *const *device_out_ptr,
  const size_t *device_out_available_bytes,
  nvcompStatus_t *outputs,
  void *device_correctness_ptrs,
  size_t *device_out_bytes,
  size_t count,
  cudaStream_t stream
)
{
  dim3 dim_block(DECOMP_THREADS_PER_BLOCK, 1);
  dim3 dim_grid(nvcomp::cuda_dim_cast(count), 1);

  if constexpr (CORRECTNESS_CHECK)
  {
    // The entire correctness-checker array is copied to the host wholesale in
    // printErrors(). Zero it first so that chunks which record no error (and any
    // struct padding) are not flagged by `compute-sanitizer --tool initcheck` as
    // uninitialized cudaMemcpy-source reads. Each chunk's Initialize() (run by
    // the kernel below) then sets line_number_ = -1 to mark "no error".
    CUDA_CHECK(cudaMemsetAsync(
      device_correctness_ptrs,
      0,
      count * sizeof(snappy::SnappyCorrectnessChecker<CORRECTNESS_CHECK>),
      stream
    ));
  }

  unsnap_kernel<CORRECTNESS_CHECK><<<dim_grid, dim_block, 0, stream>>>(
    device_in_ptr,
    device_in_bytes,
    device_out_ptr,
    device_out_available_bytes,
    outputs,
    reinterpret_cast<snappy::SnappyCorrectnessChecker<CORRECTNESS_CHECK> *>(device_correctness_ptrs),
    device_out_bytes
  );
  CUDA_CHECK(cudaGetLastError());

  snappy::SnappyCorrectnessChecker<CORRECTNESS_CHECK>::printErrors(
    reinterpret_cast<snappy::SnappyCorrectnessChecker<CORRECTNESS_CHECK> *>(device_correctness_ptrs),
    count,
    stream
  );
}

template void gpu_unsnap<false>(
  const void *const *device_in_ptr,
  const size_t *device_in_bytes,
  void *const *device_out_ptr,
  const size_t *device_out_available_bytes,
  nvcompStatus_t *outputs,
  void *device_correctness_ptrs,
  size_t *device_out_bytes,
  size_t count,
  cudaStream_t stream
);

template void gpu_unsnap<true>(
  const void *const *device_in_ptr,
  const size_t *device_in_bytes,
  void *const *device_out_ptr,
  const size_t *device_out_available_bytes,
  nvcompStatus_t *outputs,
  void *device_correctness_ptrs,
  size_t *device_out_bytes,
  size_t count,
  cudaStream_t stream
);

void gpu_get_uncompressed_sizes(
  const void *const *device_in_ptr,
  const size_t *device_in_bytes,
  size_t *device_out_bytes,
  size_t count,
  cudaStream_t stream
)
{
  dim3 dim_block(32, 1);
  dim3 dim_grid(nvcomp::cuda_dim_cast(count), 1);

  get_uncompressed_sizes_kernel<<<dim_grid, dim_block, 0, stream>>>(device_in_ptr, device_in_bytes, device_out_bytes);
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

  // Parse the input buffer to determine the number of bytes to skip
  // First byte with a 0 msb indicates no more bytes in the header
  const uint8_t *cur = reinterpret_cast<const uint8_t *>(device_in_ptr[ix_chunk]);
  uint32_t header_bytes;
  int32_t error;
  uint32_t uncompressed_size = snappy::get_uncompressed_size(cur, device_in_bytes[ix_chunk], header_bytes, error);

  if (error)
  {
    // Error
    device_statuses[ix_chunk] = nvcompStatus_t::nvcompErrorCannotDecompress;

    sort_keys[ix_chunk] = 0;
  }
  else
  {
    sort_keys[ix_chunk] = uncompressed_size;
  }
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

  // Parse the input buffer to determine the number of bytes to skip
  // First byte with a 0 msb indicates no more bytes in the header
  const uint8_t *cur = reinterpret_cast<const uint8_t *>(device_in_ptr[sorted_ix_chunk]);
  uint32_t header_bytes;
  int32_t error = 0;
  uint32_t uncompressed_size = snappy::get_uncompressed_size(cur, dev_in_bytes, header_bytes, error);
  if (error)
  {
    return;
  }

  de_params[ix_chunk].src = reinterpret_cast<const void *>(cur + header_bytes);
  de_params[ix_chunk].dst = device_out_ptr[sorted_ix_chunk];
  device_out_bytes[sorted_ix_chunk] = 0; // WAR for HW only setting 32 lsb for this value
  de_params[ix_chunk].dstActBytes = reinterpret_cast<cuuint32_t *>(&device_out_bytes[sorted_ix_chunk]);
  de_params[ix_chunk].srcNumBytes = dev_in_bytes - header_bytes;
  de_params[ix_chunk].dstNumBytes = 0;
  de_params[ix_chunk].algo = CU_MEM_DECOMPRESS_ALGORITHM_SNAPPY;

  device_statuses[sorted_ix_chunk] = nvcompSuccess;
}

void gpu_fill_cudecomp_params(
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

} // namespace nvcomp
