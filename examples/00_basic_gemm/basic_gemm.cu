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
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

#include <cublas_v2.h>
#include <nvtx3/nvToolsExt.h>
#include <nvcomp/ans.h>
#include <nvcompdx.hpp>

// Helper methods to check for errors
#include "helper.h"

//
// CUTLASS includes needed for single-precision GEMM kernel
//

// Defines cutlass::gemm::device::Gemm, the generic Gemm computation template class.
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Tile-level ANS fused into the CUTLASS GEMM kernel via nvCOMPDx
//
// LinearCombination is a per-thread functor and never holds a whole 128x128 tile. After it writes
// D, the same CTA packs that strided column-major tile into a contiguous 64 KiB chunk and runs
// block-level ANS. device::Gemm launches Kernel<GemmKernel> with no hook after the epilogue, so
// this example launches Kernel<GemmFusedAns> itself (same grid, 256 threads, dynamic smem reused
// after the epilogue).
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

static cudaError_t CheckNvcompdxSm() {
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
  return cudaSuccess;
}

static cudaError_t PrintAnsRatio(
    unsigned long long const *d_comp_sizes,
    size_t num_chunks,
    size_t chunk_bytes,
    char const *label) {
  std::vector<unsigned long long> h_comp_sizes(num_chunks);
  cudaError_t err = cudaMemcpy(
      h_comp_sizes.data(),
      d_comp_sizes,
      num_chunks * sizeof(unsigned long long),
      cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    return err;
  }

  size_t compressed_bytes = 0;
  for (size_t i = 0; i < num_chunks; ++i) {
    compressed_bytes += static_cast<size_t>(h_comp_sizes[i]);
  }
  size_t const uncompressed_bytes = num_chunks * chunk_bytes;
  std::cout << label << ": " << num_chunks
            << " tiles of " << chunk_bytes << " B, uncompressed "
            << uncompressed_bytes << " B, compressed " << compressed_bytes
            << " B, ratio "
            << (compressed_bytes ? static_cast<double>(uncompressed_bytes) / compressed_bytes : 0.0)
            << std::endl;
  return cudaSuccess;
}

/// LinearCombination plus device pointers so the fused kernel can ANS-compress
/// the CTA's output tile after the GEMM epilogue.
struct AnsEpilogueOp : public cutlass::epilogue::thread::LinearCombination<float, 1, float, float> {
  using Base = cutlass::epilogue::thread::LinearCombination<float, 1, float, float>;

  struct Params : public Base::Params {
    float *packed = nullptr;
    size_t packed_stride_elems = 0;
    char *compressed = nullptr;
    size_t compressed_stride = 0;
    unsigned long long *comp_sizes = nullptr;
    unsigned char *tmp = nullptr;
    int orig_M = 0;
    int orig_N = 0;

    CUTLASS_HOST_DEVICE
    Params() = default;

    CUTLASS_HOST_DEVICE
    Params(float alpha, float beta = 0.f) : Base::Params(alpha, beta) {}
  };

  CUTLASS_HOST_DEVICE
  AnsEpilogueOp() : Base(Params()) {}

  CUTLASS_HOST_DEVICE
  AnsEpilogueOp(Params const &params) : Base(params) {}

  CUTLASS_HOST_DEVICE
  AnsEpilogueOp(Params const &params, int group_idx) : Base(params, group_idx) {}
};

// Column-major device::Gemm swaps A/B and treats C/D as RowMajor with problem {N, M, K}.
using CutlassGemm = cutlass::gemm::device::Gemm<
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::ColumnMajor,
    float,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm70,
    cutlass::gemm::GemmShape<128, 128, 8>,
    cutlass::gemm::GemmShape<32, 64, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    AnsEpilogueOp>;

using CutlassGemmKernel = typename CutlassGemm::GemmKernel;
static_assert(CutlassGemmKernel::kThreadCount == kAnsThreads,
              "nvCOMPDx BlockWarp<8> requires the 256-thread CUTLASS GEMM CTA");

/// GEMM mainloop + LinearCombination, then pack this CTA's 128x128 tile and ANS-compress it.
struct GemmFusedAns {
  using Params = typename CutlassGemmKernel::Params;
  using SharedStorage = typename CutlassGemmKernel::SharedStorage;
  static int const kThreadCount = CutlassGemmKernel::kThreadCount;

  CUTLASS_DEVICE
  void compress_tile(Params const &params, unsigned char *shared_scratch) {
    NVCOMPDX_SKIP_IF_NOT_APPLICABLE_SM(AnsCompressor);

    typename CutlassGemmKernel::ThreadblockSwizzle threadblock_swizzle;
    cutlass::gemm::GemmCoord tb =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    // After the ColumnMajor A/B swap, tb.m() walks original columns and tb.n() original rows.
    int const m0 = tb.n() * kAnsTileM;
    int const n0 = tb.m() * kAnsTileN;
    size_t const tile_id =
        static_cast<size_t>(tb.m()) +
        static_cast<size_t>(tb.n()) * static_cast<size_t>(params.grid_tiled_shape.m());

    AnsEpilogueOp::Params const &ans = params.output_op;
    float *packed = ans.packed + tile_id * ans.packed_stride_elems;
    float const *C = params.ref_D.data();
    int const ldc = static_cast<int>(params.ref_D.stride(0));
    int const remain_m = ans.orig_M - m0;
    int const remain_n = ans.orig_N - n0;
    int const rows = remain_m < kAnsTileM ? remain_m : kAnsTileM;
    int const cols = remain_n < kAnsTileN ? remain_n : kAnsTileN;
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

    uintptr_t const align = static_cast<uintptr_t>(AnsCompressor::shmem_alignment());
    uintptr_t scratch_addr = reinterpret_cast<uintptr_t>(shared_scratch);
    if (align > 1) {
      scratch_addr = (scratch_addr + align - 1) & ~(align - 1);
    }
    unsigned char *aligned_scratch = reinterpret_cast<unsigned char *>(scratch_addr);

    auto compressor = AnsCompressor();
    unsigned char *tmp = ans.tmp;
    if (tmp != nullptr && compressor.tmp_size_group() > 0) {
      tmp += compressor.tmp_size_group() * tile_id;
    }

    compressor.execute(
        packed,
        ans.compressed + tile_id * ans.compressed_stride,
        kAnsChunkBytes,
        ans.comp_sizes + tile_id,
        aligned_scratch,
        tmp);
  }

  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {
    CutlassGemmKernel gemm;
    gemm(params, shared_storage);

    typename CutlassGemmKernel::ThreadblockSwizzle threadblock_swizzle;
    cutlass::gemm::GemmCoord tb =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    if (params.grid_tiled_shape.m() <= tb.m() ||
        params.grid_tiled_shape.n() <= tb.n()) {
      return;
    }

    __syncthreads();
    compress_tile(params, reinterpret_cast<unsigned char *>(&shared_storage));
  }
};

/// Pack one column-major 128x128 tile and ANS-compress it. No GEMM.
template <typename Compressor>
__global__ void compress_tiles_nvcompdx_kernel(
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

__global__ void count_ans_tile_mismatches_kernel(
    unsigned char const *packed,
    size_t packed_stride,
    unsigned char const *decompressed,
    size_t decompressed_stride,
    size_t const *decomp_sizes,
    size_t num_chunks,
    size_t chunk_bytes,
    unsigned long long *mismatch_chunks) {
  size_t const tile_id = static_cast<size_t>(blockIdx.x);
  if (tile_id >= num_chunks) {
    return;
  }

  __shared__ int tile_mismatch;
  if (threadIdx.x == 0) {
    tile_mismatch = (decomp_sizes[tile_id] != chunk_bytes);
  }
  __syncthreads();

  unsigned char const *ref = packed + tile_id * packed_stride;
  unsigned char const *got = decompressed + tile_id * decompressed_stride;
  for (size_t i = threadIdx.x; i < chunk_bytes; i += blockDim.x) {
    if (ref[i] != got[i]) {
      tile_mismatch = 1;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0 && tile_mismatch) {
    atomicAdd(mismatch_chunks, 1ull);
  }
}

/// After compress iterations: nvCOMP LLIF batched ANS decompress, then compare to packed tiles.
cudaError_t ValidateAnsCompression(
    void const *packed,
    size_t packed_stride,
    void const *compressed,
    size_t compressed_stride,
    unsigned long long const *comp_sizes,
    size_t num_chunks,
    size_t chunk_bytes) {
  if (num_chunks == 0) {
    return cudaSuccess;
  }

  nvcompBatchedANSDecompressOpts_t const opts = nvcompBatchedANSDecompressDefaultOpts;
  size_t temp_bytes = 0;
  nvcompStatus_t nvst = nvcompBatchedANSDecompressGetTempSizeAsync(
      num_chunks,
      chunk_bytes,
      opts,
      &temp_bytes,
      num_chunks * chunk_bytes);
  if (nvst != nvcompSuccess) {
    std::cerr << "nvcompBatchedANSDecompressGetTempSizeAsync failed: "
              << nvcompGetStatusString(nvst) << std::endl;
    return cudaErrorUnknown;
  }

  size_t const decompressed_stride = align_up_bytes(
      chunk_bytes, std::max(nvcompANSRequiredDecompressionAlignment, size_t(256)));

  unsigned char *d_decompressed = nullptr;
  size_t *d_decomp_sizes = nullptr;
  size_t *d_comp_bytes = nullptr;
  size_t *d_out_caps = nullptr;
  void **d_in_ptrs = nullptr;
  void **d_out_ptrs = nullptr;
  void *d_temp = nullptr;
  nvcompStatus_t *d_statuses = nullptr;
  unsigned long long *d_mismatches = nullptr;

  auto free_all = [&]() {
    cudaFree(d_mismatches);
    cudaFree(d_statuses);
    cudaFree(d_temp);
    cudaFree(d_out_ptrs);
    cudaFree(d_in_ptrs);
    cudaFree(d_out_caps);
    cudaFree(d_comp_bytes);
    cudaFree(d_decomp_sizes);
    cudaFree(d_decompressed);
  };

  cudaError_t err = cudaMalloc(&d_decompressed, decompressed_stride * num_chunks);
  if (err != cudaSuccess) {
    return err;
  }
  err = cudaMalloc(&d_decomp_sizes, num_chunks * sizeof(size_t));
  if (err != cudaSuccess) {
    free_all();
    return err;
  }
  err = cudaMalloc(&d_comp_bytes, num_chunks * sizeof(size_t));
  if (err != cudaSuccess) {
    free_all();
    return err;
  }
  err = cudaMalloc(&d_out_caps, num_chunks * sizeof(size_t));
  if (err != cudaSuccess) {
    free_all();
    return err;
  }
  err = cudaMalloc(&d_in_ptrs, num_chunks * sizeof(void *));
  if (err != cudaSuccess) {
    free_all();
    return err;
  }
  err = cudaMalloc(&d_out_ptrs, num_chunks * sizeof(void *));
  if (err != cudaSuccess) {
    free_all();
    return err;
  }
  err = cudaMalloc(&d_statuses, num_chunks * sizeof(nvcompStatus_t));
  if (err != cudaSuccess) {
    free_all();
    return err;
  }
  err = cudaMalloc(&d_mismatches, sizeof(unsigned long long));
  if (err != cudaSuccess) {
    free_all();
    return err;
  }
  if (temp_bytes) {
    err = cudaMalloc(&d_temp, temp_bytes);
    if (err != cudaSuccess) {
      free_all();
      return err;
    }
  }

  std::vector<unsigned long long> h_comp_ull(num_chunks);
  err = cudaMemcpy(
      h_comp_ull.data(),
      comp_sizes,
      num_chunks * sizeof(unsigned long long),
      cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    free_all();
    return err;
  }
  std::vector<size_t> h_comp_bytes(num_chunks);
  std::vector<size_t> h_out_caps(num_chunks, chunk_bytes);
  std::vector<void *> h_in_ptrs(num_chunks);
  std::vector<void *> h_out_ptrs(num_chunks);
  char *comp_base = static_cast<char *>(const_cast<void *>(compressed));
  for (size_t i = 0; i < num_chunks; ++i) {
    h_comp_bytes[i] = static_cast<size_t>(h_comp_ull[i]);
    h_in_ptrs[i] = comp_base + i * compressed_stride;
    h_out_ptrs[i] = d_decompressed + i * decompressed_stride;
  }
  err = cudaMemcpy(d_comp_bytes, h_comp_bytes.data(), num_chunks * sizeof(size_t), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    free_all();
    return err;
  }
  err = cudaMemcpy(d_out_caps, h_out_caps.data(), num_chunks * sizeof(size_t), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    free_all();
    return err;
  }
  err = cudaMemcpy(d_in_ptrs, h_in_ptrs.data(), num_chunks * sizeof(void *), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    free_all();
    return err;
  }
  err = cudaMemcpy(d_out_ptrs, h_out_ptrs.data(), num_chunks * sizeof(void *), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    free_all();
    return err;
  }
  err = cudaMemset(d_decomp_sizes, 0, num_chunks * sizeof(size_t));
  if (err != cudaSuccess) {
    free_all();
    return err;
  }
  err = cudaMemset(d_mismatches, 0, sizeof(unsigned long long));
  if (err != cudaSuccess) {
    free_all();
    return err;
  }

  nvtxRangePushA("nvcomp_ans_validate");
  nvst = nvcompBatchedANSDecompressAsync(
      reinterpret_cast<const void *const *>(d_in_ptrs),
      d_comp_bytes,
      d_out_caps,
      d_decomp_sizes,
      num_chunks,
      d_temp,
      temp_bytes,
      d_out_ptrs,
      opts,
      d_statuses,
      0);
  if (nvst != nvcompSuccess) {
    nvtxRangePop();
    std::cerr << "nvcompBatchedANSDecompressAsync failed: "
              << nvcompGetStatusString(nvst) << std::endl;
    free_all();
    return cudaErrorUnknown;
  }
  count_ans_tile_mismatches_kernel<<<static_cast<unsigned>(num_chunks), kAnsThreads>>>(
      static_cast<unsigned char const *>(packed),
      packed_stride,
      d_decompressed,
      decompressed_stride,
      d_decomp_sizes,
      num_chunks,
      chunk_bytes,
      d_mismatches);
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

  std::vector<nvcompStatus_t> h_statuses(num_chunks);
  err = cudaMemcpy(
      h_statuses.data(), d_statuses, num_chunks * sizeof(nvcompStatus_t), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    free_all();
    return err;
  }
  for (size_t i = 0; i < num_chunks; ++i) {
    if (h_statuses[i] != nvcompSuccess) {
      std::cerr << "nvCOMP LLIF decompress status tile " << i << ": "
                << nvcompGetStatusString(h_statuses[i]) << std::endl;
      free_all();
      return cudaErrorUnknown;
    }
  }

  unsigned long long mismatches = 0;
  err = cudaMemcpy(&mismatches, d_mismatches, sizeof(mismatches), cudaMemcpyDeviceToHost);
  free_all();
  if (err != cudaSuccess) {
    return err;
  }

  if (mismatches != 0) {
    std::cerr << "ANS decompress validation failed: " << mismatches << " / " << num_chunks
              << " tiles did not round-trip" << std::endl;
    return cudaErrorUnknown;
  }
  std::cout << "ANS decompress validation (nvCOMP LLIF): passed (" << num_chunks
            << " tiles)" << std::endl;
  return cudaSuccess;
}

/// ANS-compress every 128x128 tile of C. C is not modified.
cudaError_t CompressOutputTilesAns(int M, int N, float const *C, int ldc, int iterations) {
  if (M <= 0 || N <= 0) {
    return cudaErrorInvalidValue;
  }

  cudaError_t err = CheckNvcompdxSm();
  if (err != cudaSuccess) {
    return err;
  }

  int const tiles_m = (M + kAnsTileM - 1) / kAnsTileM;
  int const tiles_n = (N + kAnsTileN - 1) / kAnsTileN;
  size_t const num_chunks = static_cast<size_t>(tiles_m) * static_cast<size_t>(tiles_n);

  size_t const chunk_bytes = kAnsChunkBytes;
  size_t const shmem_bytes = static_cast<size_t>(AnsCompressor::shmem_size_group());
  size_t const tmp_bytes = static_cast<size_t>(AnsCompressor::tmp_size_total(num_chunks));
  size_t const packed_stride = align_up_bytes(
      chunk_bytes, std::max(static_cast<size_t>(AnsCompressor::input_alignment()), size_t(256)));
  size_t const compressed_stride = align_up_bytes(
      static_cast<size_t>(AnsCompressor::max_comp_chunk_size()),
      std::max(static_cast<size_t>(AnsCompressor::output_alignment()), size_t(256)));

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

  err = cudaMemset(d_comp_sizes, 0, num_chunks * sizeof(unsigned long long));
  if (err != cudaSuccess) {
    free_all();
    return err;
  }

  err = cudaFuncSetAttribute(
      compress_tiles_nvcompdx_kernel<AnsCompressor>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(shmem_bytes));
  if (err != cudaSuccess) {
    free_all();
    return err;
  }

  nvtxRangePushA("nvcompdx_ans_tiles");
  for (int iter = 0; iter < iterations; ++iter) {
    compress_tiles_nvcompdx_kernel<AnsCompressor>
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
  }
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

  err = PrintAnsRatio(d_comp_sizes, num_chunks, chunk_bytes, "nvCOMPDx ANS (only)");
  if (err == cudaSuccess) {
    err = ValidateAnsCompression(
        d_packed,
        packed_stride,
        d_compressed,
        compressed_stride,
        d_comp_sizes,
        num_chunks,
        chunk_bytes);
  }
  free_all();
  return err;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// This function defines a CUTLASS GEMM kernel instantiation, constructs its parameters object,
// and launches it on the CUDA device.
//
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Define a CUTLASS GEMM template and launch a GEMM kernel.
/// With fuse_nvcomp, the same CTA also ANS-compresses its 128x128 output tile.
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
  int ldc,
  bool fuse_nvcomp,
  int iterations) {

  if (!fuse_nvcomp) {
    using ColumnMajor = cutlass::layout::ColumnMajor;
    using GemmUnfused = cutlass::gemm::device::Gemm<float, ColumnMajor, float, ColumnMajor,
                                                   float, ColumnMajor>;
    GemmUnfused gemm_operator;
    typename GemmUnfused::Arguments args({M, N, K},
                                         {A, lda},
                                         {B, ldb},
                                         {C, ldc},
                                         {C, ldc},
                                         {alpha, beta});
    nvtxRangePushA("cutlass_gemm");
    cutlass::Status status = cutlass::Status::kSuccess;
    for (int iter = 0; iter < iterations; ++iter) {
      status = gemm_operator(args);
      if (status != cutlass::Status::kSuccess) {
        break;
      }
    }
    cudaError_t sync_status = cudaDeviceSynchronize();
    nvtxRangePop();
    if (status != cutlass::Status::kSuccess) {
      return cudaErrorUnknown;
    }
    return sync_status;
  }

  // Same 128x128x8 SIMT SGEMM as the 6-arg ColumnMajor device::Gemm, plus AnsEpilogueOp so
  // compression pointers travel in kernel Params. The ColumnMajor specialization swaps A/B
  // and launches a RowMajor kernel on problem {N, M, K}.

  cudaError_t err = CheckNvcompdxSm();
  if (err != cudaSuccess) {
    return err;
  }

  using ThreadblockSwizzle = typename CutlassGemmKernel::ThreadblockSwizzle;
  ThreadblockSwizzle threadblock_swizzle;

  CutlassGemm::Arguments args({M, N, K},
                              {A, lda},
                              {B, ldb},
                              {C, ldc},
                              {C, ldc},
                              {alpha, beta});

  auto underlying_args = CutlassGemm::to_underlying_arguments(args);
  cutlass::gemm::GemmCoord grid_tiled_shape = threadblock_swizzle.get_tiled_shape(
      underlying_args.problem_size,
      {CutlassGemm::ThreadblockShape::kM,
       CutlassGemm::ThreadblockShape::kN,
       CutlassGemm::ThreadblockShape::kK},
      underlying_args.split_k_slices);

  size_t const num_chunks =
      static_cast<size_t>(grid_tiled_shape.m()) * static_cast<size_t>(grid_tiled_shape.n());

  size_t const chunk_bytes = kAnsChunkBytes;
  size_t const nvcomp_shmem = static_cast<size_t>(AnsCompressor::shmem_size_group());
  size_t const gemm_shmem = sizeof(typename CutlassGemmKernel::SharedStorage);
  size_t const shmem_align = static_cast<size_t>(AnsCompressor::shmem_alignment());
  size_t const dyn_smem = std::max(gemm_shmem, nvcomp_shmem + shmem_align);
  size_t const tmp_bytes = static_cast<size_t>(AnsCompressor::tmp_size_total(num_chunks));
  size_t const packed_stride = align_up_bytes(
      chunk_bytes, std::max(static_cast<size_t>(AnsCompressor::input_alignment()), size_t(256)));
  size_t const compressed_stride = align_up_bytes(
      static_cast<size_t>(AnsCompressor::max_comp_chunk_size()),
      std::max(static_cast<size_t>(AnsCompressor::output_alignment()), size_t(256)));

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

  err = cudaMemset(d_comp_sizes, 0, num_chunks * sizeof(unsigned long long));
  if (err != cudaSuccess) {
    free_all();
    return err;
  }

  args.epilogue.packed = d_packed;
  args.epilogue.packed_stride_elems = packed_stride / sizeof(float);
  args.epilogue.compressed = d_compressed;
  args.epilogue.compressed_stride = compressed_stride;
  args.epilogue.comp_sizes = d_comp_sizes;
  args.epilogue.tmp = d_temp;
  args.epilogue.orig_M = M;
  args.epilogue.orig_N = N;
  underlying_args = CutlassGemm::to_underlying_arguments(args);

  typename CutlassGemmKernel::Params params{
      underlying_args.problem_size,
      grid_tiled_shape,
      underlying_args.ref_A.non_const_ref(),
      underlying_args.ref_B.non_const_ref(),
      underlying_args.ref_C.non_const_ref(),
      underlying_args.ref_D,
      underlying_args.epilogue,
      nullptr,
      underlying_args.gather_A_indices,
      underlying_args.gather_B_indices,
      underlying_args.scatter_D_indices};

  err = cudaFuncSetAttribute(
      cutlass::Kernel<GemmFusedAns>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(dyn_smem));
  if (err != cudaSuccess) {
    free_all();
    return err;
  }

  dim3 grid = threadblock_swizzle.get_grid_shape(grid_tiled_shape);
  dim3 block(CutlassGemmKernel::kThreadCount, 1, 1);

  nvtxRangePushA("cutlass_gemm");
  for (int iter = 0; iter < iterations; ++iter) {
    cutlass::Kernel<GemmFusedAns><<<grid, block, dyn_smem>>>(params);
  }
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

  err = PrintAnsRatio(d_comp_sizes, num_chunks, chunk_bytes, "nvCOMPDx ANS (fused)");
  if (err == cudaSuccess) {
    err = ValidateAnsCompression(
        d_packed,
        packed_stride,
        d_compressed,
        compressed_stride,
        d_comp_sizes,
        num_chunks,
        chunk_bytes);
  }
  free_all();
  return err;
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
  int ldc,
  int iterations) {

  cublasHandle_t handle;
  cublasStatus_t status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return cudaErrorUnknown;
  }

  nvtxRangePushA("cublas_gemm");
  for (int iter = 0; iter < iterations; ++iter) {
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
    if (status != CUBLAS_STATUS_SUCCESS) {
      break;
    }
  }
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
  int ldc,
  int iterations) {

  dim3 block(16, 16);
  dim3 grid(
    (M + block.x - 1) / block.x,
    (N + block.y - 1) / block.y
  );

  nvtxRangePushA("reference_gemm");
  for (int iter = 0; iter < iterations; ++iter) {
    ReferenceGemm_kernel<<< grid, block >>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  }
  cudaError_t sync_status = cudaDeviceSynchronize();
  nvtxRangePop();

  if (sync_status != cudaSuccess) {
    return sync_status;
  }
  return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Allocate several matrices in GPU device memory and call a single-precision
/// CUTLASS GEMM kernel. --nvcomp-only skips GEMM and compresses an M x N matrix.
cudaError_t TestCutlassGemm(
    int M,
    int N,
    int K,
    float alpha,
    float beta,
    bool fuse_nvcomp,
    bool nvcomp_only,
    int iterations) {
  cudaError_t result;

  if (nvcomp_only) {
    float *C = nullptr;
    result = AllocateMatrix(&C, M, N, 101);
    if (result != cudaSuccess) {
      return result;
    }
    result = CompressOutputTilesAns(M, N, C, M, iterations);
    cudaFree(C);
    return result;
  }

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

  result = CublasSgemmNN(M, N, K, alpha, A, lda, B, ldb, beta, C_reference, ldc, iterations);

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

  result = CutlassSgemmNN(M, N, K, alpha, A, lda, B, ldb, beta, C_cutlass, ldc, fuse_nvcomp, iterations);

  if (result != cudaSuccess) {
    std::cerr << "CUTLASS GEMM kernel failed: "
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
  result = ReferenceGemm(M, N, K, alpha, A, lda, B, ldb, beta, C_reference, ldc, iterations);

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
//   00_basic_gemm [M] [N] [K] [alpha] [beta] [--fuse-nvcomp] [--nvcomp-only] [--iters N]
//
static bool is_opt(const char *arg, const char *hyphen, const char *underscore) {
  return std::strcmp(arg, hyphen) == 0 || std::strcmp(arg, underscore) == 0;
}

static void PrintUsage(std::ostream &os) {
  os << "Usage: 00_basic_gemm [M] [N] [K] [alpha] [beta] [options]\n"
     << "  --fuse-nvcomp   ANS-compress each CUTLASS 128x128 output tile in the GEMM CTA\n"
     << "  --nvcomp-only   Run only tile ANS (no GEMM); uses M x N as the matrix\n"
     << "  --iters N       Launch each kernel N times (default 10)\n";
}

int main(int argc, const char *arg[]) {

  //
  // Parse the command line to obtain GEMM dimensions, scalars, and optional flags.
  //

  // GEMM problem dimensions.
  // CUTLASS threadblock tile is 128x128x8, so a 128^3 problem launches 1 CTA.
  // 4096x4096 uses a 32x32 grid (1024 CTAs). K does not change CTA count.
  int problem[3] = { 4096, 4096, 1024 };
  float scalars[2] = { 1, 0 };
  bool fuse_nvcomp = false;
  bool nvcomp_only = false;
  int iterations = 10;
  int positional = 0;

  for (int i = 1; i < argc; ++i) {
    if (is_opt(arg[i], "--fuse-nvcomp", "--fuse_nvcomp")) {
      fuse_nvcomp = true;
      continue;
    }
    if (is_opt(arg[i], "--nvcomp-only", "--nvcomp_only")) {
      nvcomp_only = true;
      continue;
    }
    if (is_opt(arg[i], "--iters", "--iterations")) {
      if (i + 1 >= argc) {
        std::cerr << arg[i] << " requires an integer argument\n";
        PrintUsage(std::cerr);
        return -1;
      }
      std::stringstream ss(arg[++i]);
      ss >> iterations;
      if (ss.fail() || iterations < 1) {
        std::cerr << "Invalid --iters value\n";
        return -1;
      }
      continue;
    }
    if (std::strncmp(arg[i], "--iters=", 8) == 0) {
      std::stringstream ss(arg[i] + 8);
      ss >> iterations;
      if (ss.fail() || iterations < 1) {
        std::cerr << "Invalid --iters value\n";
        return -1;
      }
      continue;
    }
    if (is_opt(arg[i], "--help", "-h") || std::strcmp(arg[i], "-h") == 0) {
      PrintUsage(std::cout);
      return 0;
    }
    if (arg[i][0] == '-') {
      std::cerr << "Unknown option: " << arg[i] << "\n";
      PrintUsage(std::cerr);
      return -1;
    }
    std::stringstream ss(arg[i]);
    if (positional < 3) {
      ss >> problem[positional];
    } else if (positional < 5) {
      ss >> scalars[positional - 3];
    }
    ++positional;
  }

  if (fuse_nvcomp && nvcomp_only) {
    std::cerr << "--fuse-nvcomp and --nvcomp-only are mutually exclusive\n";
    return -1;
  }

  //
  // Run the CUTLASS GEMM test.
  //

  if (nvcomp_only) {
    std::cout << "Running nvCOMP only: M=" << problem[0]
              << " N=" << problem[1]
              << " (tiles 128x128 => "
              << ((problem[0] + 127) / 128) * ((problem[1] + 127) / 128)
              << " CTAs, iters=" << iterations << ")" << std::endl;
  } else {
    std::cout << "Running GEMM: M=" << problem[0]
              << " N=" << problem[1]
              << " K=" << problem[2]
              << " (CUTLASS tile 128x128 => "
              << ((problem[0] + 127) / 128) * ((problem[1] + 127) / 128)
              << " CTAs, nvCOMP fusion "
              << (fuse_nvcomp ? "on" : "off")
              << ", iters=" << iterations << ")" << std::endl;
  }

  cudaError_t result = TestCutlassGemm(
    problem[0],     // GEMM M dimension
    problem[1],     // GEMM N dimension
    problem[2],     // GEMM K dimension
    scalars[0],     // alpha
    scalars[1],     // beta
    fuse_nvcomp,
    nvcomp_only,
    iterations
  );

  if (result == cudaSuccess) {
    std::cout << "Passed." << std::endl;
  }

  // Exit.
  return result == cudaSuccess ? 0 : -1;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
