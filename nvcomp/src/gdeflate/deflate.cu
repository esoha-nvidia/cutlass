/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <cub/device/device_radix_sort.cuh>

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

#include "CorrectnessChecks.cuh"
#include "exception.hpp"
#include "gdeflate/common.h"
#include "gdeflate/deflate.h"
#include "gdeflate/deflate_decompress.cuh"
#include "gdeflate/gdeflate_constants.h"
#include "gdeflate/gzip_util.cuh"
#include "HWDecompress.hpp"
#include "nvcomp/utils.hpp"

using nvcomp::cuda_dim_cast;
using nvcomp::roundUpDiv;

namespace nvcomp_deflate
{

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

  // Use 4 byte values to improve sort efficiency --
  // we won't be calling decomp engine with > 2 GB buffers anyway
  sort_keys[ix_chunk] = static_cast<int>(device_buffer_bytes[ix_chunk]);
  sort_vals[ix_chunk] = ix_chunk;
}

__global__ void fill_params_kernel(
  CUmemDecompressParams *params,
  const void *const *device_compressed_chunk_ptrs,
  const size_t *device_compressed_chunk_bytes,
  size_t *device_uncompressed_chunk_bytes,
  void *const *device_uncompressed_chunk_ptrs,
  const bool do_gzip_header_parse,
  nvcompStatus_t *device_statuses,
  const size_t num_chunks,
  const int *sort_vals
)
{
  const size_t ix_chunk = blockIdx.x * blockDim.x + threadIdx.x;
  if (ix_chunk >= num_chunks)
  {
    return;
  }

  // If sort_vals is nullptr, use original index, otherwise use sorted index
  const int sorted_ix_chunk = (sort_vals != nullptr) ? sort_vals[ix_chunk] : ix_chunk;

  size_t src_size = device_compressed_chunk_bytes[sorted_ix_chunk];
  const uint8_t *src = reinterpret_cast<const uint8_t *>(device_compressed_chunk_ptrs[sorted_ix_chunk]);
  if (do_gzip_header_parse)
  {
    int header_size = deflate::parse_gzip_header(src, src_size);
    src_size = (src_size >= 8) ? src_size - 8 : 0; // ignore footer
    if (header_size >= 0)
    {
      src += header_size;
      src_size -= header_size;
    }
  }
  params[ix_chunk].src = src;
  params[ix_chunk].dst = device_uncompressed_chunk_ptrs[sorted_ix_chunk];
  params[ix_chunk].dstNumBytes = 0;
  params[ix_chunk].srcNumBytes = src_size;
  params[ix_chunk].dstActBytes = reinterpret_cast<cuuint32_t *>(&device_uncompressed_chunk_bytes[sorted_ix_chunk]);
  device_uncompressed_chunk_bytes[sorted_ix_chunk] = 0;
  params[ix_chunk].algo = CU_MEM_DECOMPRESS_ALGORITHM_DEFLATE;
  if (device_statuses != nullptr)
  {
    device_statuses[sorted_ix_chunk] = nvcompSuccess;
  }
}

void DeflateFillCuDecompParams(
  CUmemDecompressParams *params,
  const void *const *device_compressed_chunk_ptrs,
  const size_t *device_compressed_chunk_bytes,
  size_t *device_uncompressed_chunk_bytes,
  const size_t *device_uncompressed_buffer_bytes,
  void *const *device_uncompressed_chunk_ptrs,
  bool do_gzip_header_parse,
  const size_t num_chunks,
  nvcompStatus_t *device_statuses,
  uint8_t *scratch_allocation,
  size_t scratch_size,
  bool use_sorting,
  cudaStream_t stream
)
{
  int block_size = 32;
  auto grid_dim = cuda_dim_cast(roundUpDiv(num_chunks, block_size));

  int *sort_ix_out = nullptr;
  if (use_sorting)
  {
    int *sort_keys_in = nvcomp::roundUpToAlignment<int>(scratch_allocation);
    int *sort_keys_out = sort_keys_in + num_chunks;
    int *sort_ix_in = sort_keys_out + num_chunks;
    sort_ix_out = sort_ix_in + num_chunks;
    uint8_t *sort_scratch = reinterpret_cast<uint8_t *>(sort_ix_out + num_chunks);
    scratch_size -= (4 * sizeof(int) * num_chunks + sizeof(int) - 1);

    init_cudecomp_sort<<<grid_dim, block_size, 0, stream>>>(
      device_compressed_chunk_ptrs,
      device_compressed_chunk_bytes,
      device_uncompressed_buffer_bytes,
      device_statuses,
      num_chunks,
      sort_keys_in,
      sort_ix_in
    );
    CUDA_CHECK(cudaGetLastError());

    nvcomp::cub::DeviceRadixSort::SortPairsDescending(
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

  fill_params_kernel<<<grid_dim, block_size, 0, stream>>>(
    params,
    device_compressed_chunk_ptrs,
    device_compressed_chunk_bytes,
    device_uncompressed_chunk_bytes,
    device_uncompressed_chunk_ptrs,
    do_gzip_header_parse,
    device_statuses,
    num_chunks,
    sort_ix_out
  );
  CUDA_CHECK(cudaGetLastError());
}

template <bool CORRECTNESS_CHECK>
cudaError_t __host__ DeflateDecompressAsync(
  const void *const *device_compressed_ptrs,
  const size_t *device_compressed_bytes,
  void *const *device_uncompressed_ptrs,
  const size_t *device_uncompressed_bytes,
  size_t *device_actual_uncompressed_bytes,
  deflateStatus_t *device_statuses,
  void *device_correctness_ptrs,
  size_t batch_size,
  cudaStream_t stream,
  bool gzip_header_parser
)
{
  if constexpr (CORRECTNESS_CHECK)
  {
    assert(device_correctness_ptrs != nullptr and device_statuses != nullptr);
  }

  if (batch_size > 0)
  {
    if constexpr (CORRECTNESS_CHECK)
    {
      // The entire correctness-checker array is copied to the host wholesale in
      // printErrorsInternal(). Zero it first so that chunks which record no error
      // (and any struct padding) are not flagged by `compute-sanitizer --tool
      // initcheck` as uninitialized cudaMemcpy-source reads. Each chunk's
      // Initialize() (run by the kernel below) then sets line_number_ = -1 to
      // mark "no error".
      CUDA_CHECK(cudaMemsetAsync(
        device_correctness_ptrs,
        0,
        batch_size * sizeof(DeflateCorrectnessChecker<CORRECTNESS_CHECK>),
        stream
      ));
    }
    auto grid_dim = cuda_dim_cast(batch_size);
    auto kernel = device_actual_uncompressed_bytes
                    ? inflate_kernel<true /* should_output */, true /* get_decomp_bytes */, CORRECTNESS_CHECK>
                    : inflate_kernel<true /* should_output */, false /* get_decomp_bytes */, CORRECTNESS_CHECK>;
    kernel<<<grid_dim, NUMTHREADS, 0, stream>>>(
      device_compressed_ptrs,
      device_compressed_bytes,
      device_uncompressed_ptrs,
      device_uncompressed_bytes,
      device_actual_uncompressed_bytes,
      device_statuses,
      reinterpret_cast<DeflateCorrectnessChecker<CORRECTNESS_CHECK> *>(device_correctness_ptrs),
      gzip_header_parser
    );
    CUDA_CHECK(cudaGetLastError());
  }

  DeflateCorrectnessChecker<CORRECTNESS_CHECK>::printErrors(
    reinterpret_cast<DeflateCorrectnessChecker<CORRECTNESS_CHECK> *>(device_correctness_ptrs),
    batch_size,
    stream
  );
  return cudaSuccess;
}

// Explicit instantiations
template cudaError_t __host__ DeflateDecompressAsync<false>(
  const void *const *device_compressed_ptrs,
  const size_t *device_compressed_bytes,
  void *const *device_uncompressed_ptrs,
  const size_t *device_uncompressed_bytes,
  size_t *device_actual_uncompressed_bytes,
  deflateStatus_t *device_statuses,
  void *device_correctness_ptrs,
  size_t batch_size,
  cudaStream_t stream,
  bool gzip_header_parser
);

template cudaError_t __host__ DeflateDecompressAsync<true>(
  const void *const *device_compressed_ptrs,
  const size_t *device_compressed_bytes,
  void *const *device_uncompressed_ptrs,
  const size_t *device_uncompressed_bytes,
  size_t *device_actual_uncompressed_bytes,
  deflateStatus_t *device_statuses,
  void *device_correctness_ptrs,
  size_t batch_size,
  cudaStream_t stream,
  bool gzip_header_parser
);

cudaError_t __host__ DeflateDecompressSizeAsync(
  const void *const *device_compressed_ptrs,
  const size_t *device_compressed_bytes,
  void *const *device_uncompressed_ptrs,
  const size_t *device_uncompressed_bytes,
  size_t *device_actual_uncompressed_bytes,
  deflateStatus_t *device_statuses,
  size_t batch_size,
  cudaStream_t stream,
  bool gzip_header_parser
)
{
  if (batch_size > 0)
  {
    auto grid_dim = cuda_dim_cast(batch_size);
    inflate_kernel<false /* should_output */, true /* get_decomp_bytes */, false /* check_correctness */>
      <<<grid_dim, NUMTHREADS, 0, stream>>>(
        device_compressed_ptrs,
        device_compressed_bytes,
        device_uncompressed_ptrs,
        device_uncompressed_bytes,
        device_actual_uncompressed_bytes,
        device_statuses,
        nullptr,
        gzip_header_parser
      );
    CUDA_CHECK(cudaGetLastError());
  }
  return cudaSuccess;
}
} // namespace nvcomp_deflate
