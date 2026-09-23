/***************************************************************************************************
 * Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*
  This example demonstrates how to call a CUTLASS GEMM kernel and provides a naive reference
  matrix multiply kernel to verify its correctness.

  The CUTLASS Gemm template is instantiated in the function CutlassSgemmNN. This is kernel computes
  the general matrix product (GEMM) using single-precision floating-point arithmetic and assumes
  all matrices have column-major layout.

  The threadblock tile size is chosen as 128x128x8 which offers good performance for large matrices.
  See the CUTLASS Parallel for All blog post for more exposition on the tunable parameters available
  in CUTLASS.

  https://devblogs.nvidia.com/cutlass-linear-algebra-cuda/

  Aside from defining and launching the SGEMM kernel, this example does not use any other components
  or utilities within CUTLASS. Such utilities are demonstrated elsewhere in other examples and are
  prevalent in the CUTLASS unit tests.

  This example has delibrately been kept similar to the basic_gemm example from cutlass-1.3 to
  highlight the minimum amount of differences needed to transition to cutlass-2.0.

  Cutlass-1.3 sgemm: https://github.com/NVIDIA/cutlass/blob/master/examples/00_basic_gemm/basic_gemm.cu
*/

// Standard Library includes
#include <algorithm>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <vector>

#include <cublas_v2.h>
#include <nvtx3/nvToolsExt.h>
#include <nvcompdx.hpp>

// Helper methods to check for errors
#include "helper.h"

//
// CUTLASS includes needed for single-precision GEMM kernel
//

// Defines cutlass::gemm::device::Gemm, the generic Gemm computation template class.
#include "cutlass/gemm/device/gemm.h"

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Tile-level ANS via nvCOMPDx
//
// LinearCombination is a per-thread functor and never holds a whole 128x128 tile. After it writes
// D, a follow-up kernel (one CTA per CUTLASS tile, 8 warps) packs the strided column-major tile
// into a contiguous 64 KiB chunk and runs block-level ANS with nvCOMPDx.
//
///////////////////////////////////////////////////////////////////////////////////////////////////

enum {
  kAnsTileM = 128,
  kAnsTileN = 128,
  kAnsWarps = 8,
  kAnsThreads = kAnsWarps * 32
};

static constexpr unsigned kAnsChunkBytes =
    static_cast<unsigned>(kAnsTileM) * kAnsTileN * sizeof(float);

using AnsCompressor = decltype(
    nvcompdx::Algorithm<nvcompdx::algorithm::ans>() +
    nvcompdx::DataType<nvcompdx::datatype::uint8>() +
    nvcompdx::Direction<nvcompdx::direction::compress>() +
    nvcompdx::MaxUncompChunkSize<kAnsChunkBytes>() +
    nvcompdx::Block() +
    nvcompdx::BlockWarp<kAnsWarps, true>() +
    nvcompdx::SM<1000>());

static size_t align_up_bytes(size_t value, size_t alignment) {
  if (alignment == 0) {
    return value;
  }
  return (value + alignment - 1) / alignment * alignment;
}

/// Pack one CUTLASS output tile, then ANS-compress it. All 256 threads must call execute().
template <typename Compressor>
__global__ void compress_cutlass_tiles_nvcompdx_kernel(
    float const *C,
    int ldc,
    int M,
    int N,
    float *packed_tiles,
    size_t packed_stride_elems,
    char *compressed_tiles,
    size_t compressed_stride,
    unsigned long long *comp_sizes,
    unsigned char *tmp_global) {
  NVCOMPDX_SKIP_IF_NOT_APPLICABLE_SM(Compressor);

  int const tile_m = static_cast<int>(blockIdx.x);
  int const tile_n = static_cast<int>(blockIdx.y);
  int const tiles_m = static_cast<int>(gridDim.x);
  int const m0 = tile_m * kAnsTileM;
  int const n0 = tile_n * kAnsTileN;
  int const remain_m = M - m0;
  int const remain_n = N - n0;
  int const rows = remain_m < kAnsTileM ? remain_m : kAnsTileM;
  int const cols = remain_n < kAnsTileN ? remain_n : kAnsTileN;
  size_t const tile_id =
      static_cast<size_t>(tile_m) + static_cast<size_t>(tile_n) * static_cast<size_t>(tiles_m);

  float *packed = packed_tiles + tile_id * packed_stride_elems;
  int const tile_elems = kAnsTileM * kAnsTileN;
  for (int i = static_cast<int>(threadIdx.x); i < tile_elems; i += static_cast<int>(blockDim.x)) {
    int const row = i % kAnsTileM;
    int const col = i / kAnsTileM;
    float value = 0.f;
    if (row < rows && col < cols) {
      value = C[(m0 + row) + (n0 + col) * ldc];
    }
    packed[row + col * kAnsTileM] = value;
  }
  __syncthreads();

  auto compressor = Compressor();
  extern __shared__ __align__(Compressor::shmem_alignment()) unsigned char shared_scratch[];
  unsigned char *tmp = tmp_global;
  if (tmp != nullptr && compressor.tmp_size_group() > 0) {
    tmp += compressor.tmp_size_group() * tile_id;
  }

  compressor.execute(
      packed,
      compressed_tiles + tile_id * compressed_stride,
      kAnsChunkBytes,
      comp_sizes + tile_id,
      shared_scratch,
      tmp);
}

/// ANS-compress every CUTLASS output tile with nvCOMPDx. C is left unchanged for the GEMM check.
cudaError_t CompressOutputTilesAns(int M, int N, float const *C, int ldc) {
  if (M <= 0 || N <= 0) {
    return cudaSuccess;
  }

  int device = 0;
  cudaError_t err = cudaGetDevice(&device);
  if (err != cudaSuccess) {
    return err;
  }
  int major = 0;
  int minor = 0;
  err = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
  if (err != cudaSuccess) {
    return err;
  }
  err = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
  if (err != cudaSuccess) {
    return err;
  }
  if (major != 10) {
    std::cerr << "nvCOMPDx kernel in this example is compiled for SM100 (GB200), got SM"
              << major << minor << std::endl;
    return cudaErrorInvalidDevice;
  }

  int const tiles_m = (M + kAnsTileM - 1) / kAnsTileM;
  int const tiles_n = (N + kAnsTileN - 1) / kAnsTileN;
  size_t const num_chunks = static_cast<size_t>(tiles_m) * static_cast<size_t>(tiles_n);

  auto compressor = AnsCompressor();
  size_t const chunk_bytes = kAnsChunkBytes;
  size_t const shmem_bytes = static_cast<size_t>(compressor.shmem_size_group());
  size_t const tmp_bytes = static_cast<size_t>(compressor.tmp_size_total(num_chunks));
  size_t const packed_stride = align_up_bytes(
      chunk_bytes, std::max(static_cast<size_t>(compressor.input_alignment()), size_t(256)));
  size_t const compressed_stride = align_up_bytes(
      static_cast<size_t>(compressor.max_comp_chunk_size()),
      std::max(static_cast<size_t>(compressor.output_alignment()), size_t(256)));

  float *d_packed = nullptr;
  char *d_compressed = nullptr;
  unsigned long long *d_comp_sizes = nullptr;
  unsigned char *d_temp = nullptr;

  auto free_all = [&]() {
    cudaFree(d_packed);
    cudaFree(d_compressed);
    cudaFree(d_comp_sizes);
    cudaFree(d_temp);
  };

  err = cudaMalloc(&d_packed, packed_stride * num_chunks);
  if (err != cudaSuccess) {
    return err;
  }
  err = cudaMalloc(&d_compressed, compressed_stride * num_chunks);
  if (err != cudaSuccess) {
    free_all();
    return err;
  }
  err = cudaMalloc(&d_comp_sizes, num_chunks * sizeof(unsigned long long));
  if (err != cudaSuccess) {
    free_all();
    return err;
  }
  if (tmp_bytes) {
    err = cudaMalloc(&d_temp, tmp_bytes);
    if (err != cudaSuccess) {
      free_all();
      return err;
    }
  }

  err = cudaFuncSetAttribute(
      compress_cutlass_tiles_nvcompdx_kernel<AnsCompressor>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(shmem_bytes));
  if (err != cudaSuccess) {
    free_all();
    return err;
  }

  nvtxRangePushA("nvcompdx_ans_tiles");
  compress_cutlass_tiles_nvcompdx_kernel<AnsCompressor>
      <<<dim3(tiles_m, tiles_n), kAnsThreads, shmem_bytes>>>(
          C,
          ldc,
          M,
          N,
          d_packed,
          packed_stride / sizeof(float),
          d_compressed,
          compressed_stride,
          d_comp_sizes,
          d_temp);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    nvtxRangePop();
    free_all();
    return err;
  }
  err = cudaDeviceSynchronize();
  nvtxRangePop();
  if (err != cudaSuccess) {
    free_all();
    return err;
  }

  std::vector<unsigned long long> h_comp_sizes(num_chunks);
  err = cudaMemcpy(
      h_comp_sizes.data(),
      d_comp_sizes,
      num_chunks * sizeof(unsigned long long),
      cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    free_all();
    return err;
  }

  size_t compressed_bytes = 0;
  for (size_t i = 0; i < num_chunks; ++i) {
    compressed_bytes += static_cast<size_t>(h_comp_sizes[i]);
  }
  size_t const uncompressed_bytes = num_chunks * chunk_bytes;
  std::cout << "nvCOMPDx ANS: " << num_chunks
            << " tiles of " << chunk_bytes << " B, uncompressed "
            << uncompressed_bytes << " B, compressed " << compressed_bytes
            << " B, ratio "
            << (compressed_bytes ? static_cast<double>(uncompressed_bytes) / compressed_bytes : 0.0)
            << std::endl;

  free_all();
  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// This function defines a CUTLASS GEMM kernel instantiation, constructs its parameters object,
// and launches it on the CUDA device.
//
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Define a CUTLASS GEMM template and launch a GEMM kernel.
cudaError_t CutlassSgemmNN(
  int M,
  int N,
  int K,
  float alpha,
  float const *A,
  int lda,
  float const *B,
  int ldb,
  float beta,
  float *C,
  int ldc) {

  // Define type definition for single-precision CUTLASS GEMM with column-major
  // input matrices and 128x128x8 threadblock tile size (chosen by default).
  //
  // To keep the interface manageable, several helpers are defined for plausible compositions
  // including the following example for single-precision GEMM. Typical values are used as
  // default template arguments. See `cutlass/gemm/device/default_gemm_configuration.h` for more details.
  //
  // To view the full gemm device API interface, see `cutlass/gemm/device/gemm.h`

  using ColumnMajor = cutlass::layout::ColumnMajor;

  using CutlassGemm = cutlass::gemm::device::Gemm<float,        // Data-type of A matrix
                                                  ColumnMajor,  // Layout of A matrix
                                                  float,        // Data-type of B matrix
                                                  ColumnMajor,  // Layout of B matrix
                                                  float,        // Data-type of C matrix
                                                  ColumnMajor>; // Layout of C matrix

  // Define a CUTLASS GEMM type
  CutlassGemm gemm_operator;

  // Construct the CUTLASS GEMM arguments object.
  //
  // One of CUTLASS's design patterns is to define gemm argument objects that are constructible
  // in host code and passed to kernels by value. These may include pointers, strides, scalars,
  // and other arguments needed by Gemm and its components.
  //
  // The benefits of this pattern are (1.) a structured, composable strategy for passing host-constructible
  // arguments to kernels and (2.) minimized initialization overhead on kernel entry.
  //
  CutlassGemm::Arguments args({M , N, K},  // Gemm Problem dimensions
                              {A, lda},    // Tensor-ref for source matrix A
                              {B, ldb},    // Tensor-ref for source matrix B
                              {C, ldc},    // Tensor-ref for source matrix C
                              {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                              {alpha, beta}); // Scalars used in the Epilogue

  //
  // Launch the CUTLASS GEMM kernel.
  //

  nvtxRangePushA("cutlass_gemm");
  cutlass::Status status = gemm_operator(args);
  cudaError_t sync_status = cudaDeviceSynchronize();
  nvtxRangePop();

  //
  // Return a cudaError_t if the CUTLASS GEMM operator returned an error code.
  //

  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  return sync_status;
}

/// Column-major SGEMM via cuBLAS. NVTX covers only the GEMM launch.
cudaError_t CublasSgemmNN(
  int M,
  int N,
  int K,
  float alpha,
  float const *A,
  int lda,
  float const *B,
  int ldb,
  float beta,
  float *C,
  int ldc) {

  cublasHandle_t handle;
  cublasStatus_t status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return cudaErrorUnknown;
  }

  nvtxRangePushA("cublas_gemm");
  status = cublasSgemm(
    handle,
    CUBLAS_OP_N,
    CUBLAS_OP_N,
    M,
    N,
    K,
    &alpha,
    A,
    lda,
    B,
    ldb,
    &beta,
    C,
    ldc);
  cudaError_t sync_status = cudaDeviceSynchronize();
  nvtxRangePop();

  cublasDestroy(handle);

  if (status != CUBLAS_STATUS_SUCCESS) {
    return cudaErrorUnknown;
  }
  return sync_status;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// The source code after this point in the file is generic CUDA using the CUDA Runtime API
// and simple CUDA kernels to initialize matrices and compute the general matrix product.
//
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Kernel to initialize a matrix with small integers.
__global__ void InitializeMatrix_kernel(
  float *matrix,
  int rows,
  int columns,
  int seed = 0) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < rows && j < columns) {
    int offset = i + j * rows;

    // Generate arbitrary elements.
    int const k = 16807;
    int const m = 16;
    float value = float(((offset + seed) * k % m) - m / 2);

    matrix[offset] = value;
  }
}

/// Simple function to initialize a matrix to arbitrary small integers.
cudaError_t InitializeMatrix(float *matrix, int rows, int columns, int seed = 0) {

  dim3 block(16, 16);
  dim3 grid(
    (rows + block.x - 1) / block.x,
    (columns + block.y - 1) / block.y
  );

  InitializeMatrix_kernel<<< grid, block >>>(matrix, rows, columns, seed);

  return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Allocates device memory for a matrix then fills with arbitrary small integers.
cudaError_t AllocateMatrix(float **matrix, int rows, int columns, int seed = 0) {
  cudaError_t result;

  size_t sizeof_matrix = sizeof(float) * rows * columns;

  // Allocate device memory.
  result = cudaMalloc(reinterpret_cast<void **>(matrix), sizeof_matrix);

  if (result != cudaSuccess) {
    std::cerr << "Failed to allocate matrix: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  // Clear the allocation.
  result = cudaMemset(*matrix, 0, sizeof_matrix);

  if (result != cudaSuccess) {
    std::cerr << "Failed to clear matrix device memory: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  // Initialize matrix elements to arbitrary small integers.
  result = InitializeMatrix(*matrix, rows, columns, seed);

  if (result != cudaSuccess) {
    std::cerr << "Failed to initialize matrix: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Naive reference GEMM computation.
__global__ void ReferenceGemm_kernel(
  int M,
  int N,
  int K,
  float alpha,
  float const *A,
  int lda,
  float const *B,
  int ldb,
  float beta,
  float *C,
  int ldc) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < M && j < N) {
    float accumulator = 0;

    for (int k = 0; k < K; ++k) {
      accumulator += A[i + k * lda] * B[k + j * ldb];
    }

    C[i + j * ldc] = alpha * accumulator + beta * C[i + j * ldc];
  }
}

/// Reference GEMM computation.
cudaError_t ReferenceGemm(
  int M,
  int N,
  int K,
  float alpha,
  float const *A,
  int lda,
  float const *B,
  int ldb,
  float beta,
  float *C,
  int ldc) {

  dim3 block(16, 16);
  dim3 grid(
    (M + block.x - 1) / block.x,
    (N + block.y - 1) / block.y
  );

  nvtxRangePushA("reference_gemm");
  ReferenceGemm_kernel<<< grid, block >>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  cudaError_t sync_status = cudaDeviceSynchronize();
  nvtxRangePop();

  if (sync_status != cudaSuccess) {
    return sync_status;
  }
  return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Allocate several matrices in GPU device memory and call a single-precision
/// CUTLASS GEMM kernel.
cudaError_t TestCutlassGemm(int M, int N, int K, float alpha, float beta) {
  cudaError_t result;

  //
  // Define several matrices to be used as operands to GEMM kernels.
  //

  // Compute leading dimensions for each matrix.
  int lda = M;
  int ldb = K;
  int ldc = M;

  // Compute size in bytes of the C matrix.
  size_t sizeof_C = sizeof(float) * ldc * N;

  // Define pointers to matrices in GPU device memory.
  float *A;
  float *B;
  float *C_cutlass;
  float *C_reference;

  //
  // Allocate matrices in GPU device memory with arbitrary seeds.
  //

  result = AllocateMatrix(&A, M, K, 0);

  if (result !=  cudaSuccess) {
    return result;
  }

  result = AllocateMatrix(&B, K, N, 17);

  if (result !=  cudaSuccess) {
    cudaFree(A);
    return result;
  }

  result = AllocateMatrix(&C_cutlass, M, N, 101);

  if (result != cudaSuccess) {
    cudaFree(A);
    cudaFree(B);
    return result;
  }

  result = AllocateMatrix(&C_reference, M, N, 101);

  if (result != cudaSuccess) {
    cudaFree(A);
    cudaFree(B);
    cudaFree(C_cutlass);
    return result;
  }

  result = cudaMemcpy(C_reference, C_cutlass, sizeof_C, cudaMemcpyDeviceToDevice);

  if (result != cudaSuccess) {
    std::cerr << "Failed to copy C_cutlass matrix to C_reference: "
      << cudaGetErrorString(result) << std::endl;

    cudaFree(C_reference);
    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);

    return result;
  }

  //
  // Launch cuBLAS GEMM (NVTX range "cublas_gemm" is inside CublasSgemmNN).
  //

  result = CublasSgemmNN(M, N, K, alpha, A, lda, B, ldb, beta, C_reference, ldc);

  if (result != cudaSuccess) {
    std::cerr << "cuBLAS GEMM kernel failed: "
      << cudaGetErrorString(result) << std::endl;

    cudaFree(C_reference);
    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);

    return result;
  }

  // Restore C_reference to the original C so the naive kernel remains the
  // correctness check for CUTLASS (not TF32 cuBLAS).
  result = cudaMemcpy(C_reference, C_cutlass, sizeof_C, cudaMemcpyDeviceToDevice);

  if (result != cudaSuccess) {
    std::cerr << "Failed to restore C_reference after cuBLAS: "
      << cudaGetErrorString(result) << std::endl;

    cudaFree(C_reference);
    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);

    return result;
  }

  //
  // Launch CUTLASS GEMM.
  //

  result = CutlassSgemmNN(M, N, K, alpha, A, lda, B, ldb, beta, C_cutlass, ldc);

  if (result != cudaSuccess) {
    std::cerr << "CUTLASS GEMM kernel failed: "
      << cudaGetErrorString(result) << std::endl;

    cudaFree(C_reference);
    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);

    return result;
  }

  result = CompressOutputTilesAns(M, N, C_cutlass, ldc);

  if (result != cudaSuccess) {
    std::cerr << "nvCOMPDx ANS tile compression failed: "
      << cudaGetErrorString(result) << std::endl;

    cudaFree(C_reference);
    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);

    return result;
  }

  //
  // Verify.
  //

  // Launch reference GEMM
  result = ReferenceGemm(M, N, K, alpha, A, lda, B, ldb, beta, C_reference, ldc);

  if (result != cudaSuccess) {
    std::cerr << "Reference GEMM kernel failed: "
      << cudaGetErrorString(result) << std::endl;

    cudaFree(C_reference);
    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);

    return result;
  }

  // Copy to host and verify equivalence.
  std::vector<float> host_cutlass(ldc * N, 0);
  std::vector<float> host_reference(ldc * N, 0);

  result = cudaMemcpy(host_cutlass.data(), C_cutlass, sizeof_C, cudaMemcpyDeviceToHost);

  if (result != cudaSuccess) {
    std::cerr << "Failed to copy CUTLASS GEMM results: "
      << cudaGetErrorString(result) << std::endl;

    cudaFree(C_reference);
    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);

    return result;
  }

  result = cudaMemcpy(host_reference.data(), C_reference, sizeof_C, cudaMemcpyDeviceToHost);

  if (result != cudaSuccess) {
    std::cerr << "Failed to copy Reference GEMM results: "
      << cudaGetErrorString(result) << std::endl;

    cudaFree(C_reference);
    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);

    return result;
  }

  //
  // Free device memory allocations.
  //

  cudaFree(C_reference);
  cudaFree(C_cutlass);
  cudaFree(B);
  cudaFree(A);

  //
  // Test for bit equivalence of results.
  //

  if (host_cutlass != host_reference) {
    std::cerr << "CUTLASS results incorrect." << std::endl;

    return cudaErrorUnknown;
  }

  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Entry point to basic_gemm example.
//
// usage:
//
//   00_basic_gemm <M> <N> <K> <alpha> <beta>
//
int main(int argc, const char *arg[]) {

  //
  // Parse the command line to obtain GEMM dimensions and scalar values.
  //

  // GEMM problem dimensions.
  // CUTLASS threadblock tile is 128x128x8, so a 128^3 problem launches 1 CTA.
  // 4096x4096 uses a 32x32 grid (1024 CTAs). K does not change CTA count.
  int problem[3] = { 4096, 4096, 1024 };

  for (int i = 1; i < argc && i < 4; ++i) {
    std::stringstream ss(arg[i]);
    ss >> problem[i - 1];
  }

  // Scalars used for linear scaling the result of the matrix product.
  float scalars[2] = { 1, 0 };

  for (int i = 4; i < argc && i < 6; ++i) {
    std::stringstream ss(arg[i]);
    ss >> scalars[i - 4];
  }

  //
  // Run the CUTLASS GEMM test.
  //

  std::cout << "Running GEMM: M=" << problem[0]
            << " N=" << problem[1]
            << " K=" << problem[2]
            << " (CUTLASS tile 128x128 => "
            << ((problem[0] + 127) / 128) * ((problem[1] + 127) / 128)
            << " CTAs)" << std::endl;

  cudaError_t result = TestCutlassGemm(
    problem[0],     // GEMM M dimension
    problem[1],     // GEMM N dimension
    problem[2],     // GEMM K dimension
    scalars[0],     // alpha
    scalars[1]      // beta
  );

  if (result == cudaSuccess) {
    std::cout << "Passed." << std::endl;
  }

  // Exit.
  return result == cudaSuccess ? 0 : -1;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
