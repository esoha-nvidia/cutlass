/***************************************************************************************************
 * Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

/*! \file
    \brief Blackwell SM100 FP16 GEMM (tcgen05) followed by unfused nvCOMP LLIF ANS.

    Same CUTLASS 3.x Blackwell kernel family as 70_blackwell_fp16_gemm, with a 1SM 128x128
    MMA tile so C is a grid of 128x128 tiles. After the GEMM, each tile is packed and
    ANS-compressed by stock compress_kernel (NVCOMP_TYPE_FLOAT16). This is the unfused
    path: GEMM CTA and ANS CTA are separate. In-CTA fusion needs a custom Blackwell epilogue.

    Usage:
      $ ./examples/70_blackwell_gemm/70_blackwell_fp16_gemm_nvcomp --m=8192 --n=8192 --k=2048
*/

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

#include <nvtx3/nvToolsExt.h>
#include <nvcomp/ans.h>
#include <nvcomp/ans_device.cuh>

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"

#include "helper.h"

using namespace cute;

enum {
  kAnsTileM = 128,
  kAnsTileN = 128,
  kAnsThreads = 256
};

static constexpr size_t kAnsChunkBytes =
    static_cast<size_t>(kAnsTileM) * kAnsTileN * sizeof(cutlass::half_t);

static nvcompBatchedANSCompressOpts_t AnsFp16CompressOpts() {
  nvcompBatchedANSCompressOpts_t opts = nvcompBatchedANSCompressDefaultOpts;
  opts.data_type = NVCOMP_TYPE_FLOAT16;
  return opts;
}

static nvcompBatchedANSDecompressOpts_t AnsFp16DecompressOpts() {
  nvcompBatchedANSDecompressOpts_t opts = nvcompBatchedANSDecompressDefaultOpts;
  opts.data_type = NVCOMP_TYPE_FLOAT16;
  return opts;
}

static size_t align_up_bytes(size_t value, size_t alignment) {
  if (alignment == 0) {
    return value;
  }
  return (value + alignment - 1) / alignment * alignment;
}

static cudaError_t PrintAnsRatio(
    size_t const *d_comp_sizes,
    size_t num_chunks,
    size_t chunk_bytes,
    char const *label) {
  std::vector<size_t> h_comp_sizes(num_chunks);
  cudaError_t err = cudaMemcpy(
      h_comp_sizes.data(), d_comp_sizes, num_chunks * sizeof(size_t), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    return err;
  }
  size_t compressed_bytes = 0;
  for (size_t i = 0; i < num_chunks; ++i) {
    compressed_bytes += h_comp_sizes[i];
  }
  size_t const uncompressed_bytes = num_chunks * chunk_bytes;
  std::cout << label << ": " << num_chunks << " tiles of " << chunk_bytes
            << " B, uncompressed " << uncompressed_bytes << " B, compressed "
            << compressed_bytes << " B, ratio "
            << (compressed_bytes ? static_cast<double>(uncompressed_bytes) / compressed_bytes : 0.0)
            << std::endl;
  return cudaSuccess;
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
  size_t const tile_id = blockIdx.x;
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

struct UnfusedAnsWorkspace {
  size_t num_chunks = 0;
  size_t chunk_bytes = 0;
  size_t packed_stride = 0;
  size_t compressed_stride = 0;
  nvcompBatchedANSCompressOpts_t compress_opts{};
  cutlass::half_t *d_packed = nullptr;
  char *d_compressed = nullptr;
  size_t *d_comp_sizes = nullptr;
  size_t *d_uncomp_bytes = nullptr;
  void **d_in_ptrs = nullptr;
  void **d_out_ptrs = nullptr;

  void release() {
    cudaFree(d_out_ptrs);
    cudaFree(d_in_ptrs);
    cudaFree(d_uncomp_bytes);
    cudaFree(d_packed);
    cudaFree(d_compressed);
    cudaFree(d_comp_sizes);
    d_out_ptrs = nullptr;
    d_in_ptrs = nullptr;
    d_uncomp_bytes = nullptr;
    d_packed = nullptr;
    d_compressed = nullptr;
    d_comp_sizes = nullptr;
  }

  cudaError_t allocate(int M, int N) {
    if (M <= 0 || N <= 0) {
      return cudaErrorInvalidValue;
    }
    int const tiles_m = (M + kAnsTileM - 1) / kAnsTileM;
    int const tiles_n = (N + kAnsTileN - 1) / kAnsTileN;
    num_chunks = static_cast<size_t>(tiles_m) * static_cast<size_t>(tiles_n);
    chunk_bytes = kAnsChunkBytes;
    compress_opts = AnsFp16CompressOpts();
    size_t max_comp_chunk = 0;
    nvcompStatus_t nvst = nvcompBatchedANSCompressGetMaxOutputChunkSize(
        chunk_bytes, compress_opts, &max_comp_chunk);
    if (nvst != nvcompSuccess) {
      std::cerr << "nvcompBatchedANSCompressGetMaxOutputChunkSize failed: "
                << nvcompGetStatusString(nvst) << std::endl;
      return cudaErrorUnknown;
    }
    packed_stride =
        align_up_bytes(chunk_bytes, std::max(nvcompANSRequiredCompressionAlignment, size_t(256)));
    compressed_stride =
        align_up_bytes(max_comp_chunk, std::max(nvcompANSRequiredCompressionAlignment, size_t(256)));

    cudaError_t err = cudaMalloc(&d_packed, packed_stride * num_chunks);
    if (err != cudaSuccess) {
      return err;
    }
    err = cudaMalloc(&d_compressed, compressed_stride * num_chunks);
    if (err != cudaSuccess) {
      release();
      return err;
    }
    err = cudaMemset(d_compressed, 0, compressed_stride * num_chunks);
    if (err != cudaSuccess) {
      release();
      return err;
    }
    err = cudaMalloc(&d_comp_sizes, num_chunks * sizeof(size_t));
    if (err != cudaSuccess) {
      release();
      return err;
    }
    err = cudaMalloc(&d_uncomp_bytes, num_chunks * sizeof(size_t));
    if (err != cudaSuccess) {
      release();
      return err;
    }
    err = cudaMalloc(&d_in_ptrs, num_chunks * sizeof(void *));
    if (err != cudaSuccess) {
      release();
      return err;
    }
    err = cudaMalloc(&d_out_ptrs, num_chunks * sizeof(void *));
    if (err != cudaSuccess) {
      release();
      return err;
    }

    std::vector<size_t> h_uncomp_bytes(num_chunks, chunk_bytes);
    std::vector<void *> h_in_ptrs(num_chunks);
    std::vector<void *> h_out_ptrs(num_chunks);
    for (size_t i = 0; i < num_chunks; ++i) {
      h_in_ptrs[i] = reinterpret_cast<char *>(d_packed) + i * packed_stride;
      h_out_ptrs[i] = d_compressed + i * compressed_stride;
    }
    err = cudaMemcpy(d_uncomp_bytes, h_uncomp_bytes.data(), num_chunks * sizeof(size_t),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      release();
      return err;
    }
    err = cudaMemcpy(d_in_ptrs, h_in_ptrs.data(), num_chunks * sizeof(void *), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      release();
      return err;
    }
    err = cudaMemcpy(d_out_ptrs, h_out_ptrs.data(), num_chunks * sizeof(void *), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      release();
      return err;
    }
    return cudaSuccess;
  }

  nvcompStatus_t launch(cutlass::half_t const *C, int ldc, int M, int N) {
    return nvcompBatchedANSCompressFromColMajorTilesAsync(
        C,
        ldc,
        M,
        N,
        kAnsTileM,
        kAnsTileN,
        /*pack_mn_swapped=*/0,
        reinterpret_cast<const void *const *>(d_in_ptrs),
        d_uncomp_bytes,
        chunk_bytes,
        num_chunks,
        d_out_ptrs,
        d_comp_sizes,
        compress_opts,
        nullptr,
        0);
  }

  cudaError_t validate_roundtrip(char const *ratio_label) {
    cudaError_t err = PrintAnsRatio(d_comp_sizes, num_chunks, chunk_bytes, ratio_label);
    if (err != cudaSuccess || num_chunks == 0) {
      return err;
    }

    nvcompBatchedANSDecompressOpts_t const opts = AnsFp16DecompressOpts();
    size_t temp_bytes = 0;
    nvcompStatus_t nvst = nvcompBatchedANSDecompressGetTempSize(
        num_chunks, chunk_bytes, opts, &temp_bytes, num_chunks * chunk_bytes, 0);
    if (nvst != nvcompSuccess) {
      std::cerr << "nvcompBatchedANSDecompressGetTempSize failed: "
                << nvcompGetStatusString(nvst) << std::endl;
      return cudaErrorUnknown;
    }

    size_t const decompressed_stride = align_up_bytes(
        chunk_bytes, std::max(nvcompANSRequiredDecompressionAlignment, size_t(256)));
    unsigned char *d_decompressed = nullptr;
    size_t *d_decomp_sizes = nullptr;
    size_t *d_comp_bytes = nullptr;
    size_t *d_out_caps = nullptr;
    void **d_in = nullptr;
    void **d_out = nullptr;
    void *d_temp = nullptr;
    nvcompStatus_t *d_statuses = nullptr;
    unsigned long long *d_mismatches = nullptr;

    auto free_all = [&]() {
      cudaFree(d_mismatches);
      cudaFree(d_statuses);
      cudaFree(d_temp);
      cudaFree(d_out);
      cudaFree(d_in);
      cudaFree(d_out_caps);
      cudaFree(d_comp_bytes);
      cudaFree(d_decomp_sizes);
      cudaFree(d_decompressed);
    };

    err = cudaMalloc(&d_decompressed, decompressed_stride * num_chunks);
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
    err = cudaMalloc(&d_in, num_chunks * sizeof(void *));
    if (err != cudaSuccess) {
      free_all();
      return err;
    }
    err = cudaMalloc(&d_out, num_chunks * sizeof(void *));
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

    std::vector<size_t> h_comp_bytes(num_chunks);
    err = cudaMemcpy(h_comp_bytes.data(), d_comp_sizes, num_chunks * sizeof(size_t),
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      free_all();
      return err;
    }
    std::vector<size_t> h_out_caps(num_chunks, chunk_bytes);
    std::vector<void *> h_in(num_chunks);
    std::vector<void *> h_out(num_chunks);
    for (size_t i = 0; i < num_chunks; ++i) {
      h_in[i] = d_compressed + i * compressed_stride;
      h_out[i] = d_decompressed + i * decompressed_stride;
    }
    err = cudaMemcpy(d_comp_bytes, h_comp_bytes.data(), num_chunks * sizeof(size_t),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      free_all();
      return err;
    }
    err = cudaMemcpy(d_out_caps, h_out_caps.data(), num_chunks * sizeof(size_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      free_all();
      return err;
    }
    err = cudaMemcpy(d_in, h_in.data(), num_chunks * sizeof(void *), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      free_all();
      return err;
    }
    err = cudaMemcpy(d_out, h_out.data(), num_chunks * sizeof(void *), cudaMemcpyHostToDevice);
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

    nvst = nvcompBatchedANSDecompressAsync(
        reinterpret_cast<const void *const *>(d_in),
        d_comp_bytes,
        d_out_caps,
        d_decomp_sizes,
        num_chunks,
        d_temp,
        temp_bytes,
        d_out,
        opts,
        d_statuses,
        0);
    if (nvst != nvcompSuccess) {
      std::cerr << "nvcompBatchedANSDecompressAsync failed: "
                << nvcompGetStatusString(nvst) << std::endl;
      free_all();
      return cudaErrorUnknown;
    }
    count_ans_tile_mismatches_kernel<<<static_cast<unsigned>(num_chunks), kAnsThreads>>>(
        reinterpret_cast<unsigned char const *>(d_packed),
        packed_stride,
        d_decompressed,
        decompressed_stride,
        d_decomp_sizes,
        num_chunks,
        chunk_bytes,
        d_mismatches);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      free_all();
      return err;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      free_all();
      return err;
    }

    std::vector<nvcompStatus_t> h_statuses(num_chunks);
    err = cudaMemcpy(h_statuses.data(), d_statuses, num_chunks * sizeof(nvcompStatus_t),
                     cudaMemcpyDeviceToHost);
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
};

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations: Blackwell tcgen05, 1SM 128x128 tile (matches ANS tiles).
/////////////////////////////////////////////////////////////////////////////////////////////////

using         ElementA    = half_t;
using         LayoutA     = cutlass::layout::RowMajor;
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;

using         ElementB    = half_t;
using         LayoutB     = cutlass::layout::ColumnMajor;
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;

using         ElementC    = half_t;
using         LayoutC     = cutlass::layout::ColumnMajor;
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;

using ElementAccumulator  = float;
using ArchTag             = cutlass::arch::Sm100;
using OperatorClass       = cutlass::arch::OpClassTensorOp;

using MmaTileShape_MNK = Shape<_128,_128,_64>;
using ClusterShape_MNK = Shape<_1,_1,_1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, AlignmentC,
    ElementC, LayoutC, AlignmentC,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using DeviceGemmReference = cutlass::reference::device::Gemm<
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  ElementAccumulator>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

StrideA stride_A;
StrideB stride_B;
StrideC stride_C;
StrideD stride_D;

cutlass::DeviceAllocation<typename Gemm::ElementA> block_A;
cutlass::DeviceAllocation<typename Gemm::ElementB> block_B;
cutlass::DeviceAllocation<typename Gemm::ElementC> block_C;
cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput> block_D;
cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput> block_ref_D;

#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

struct Options {
  bool help;
  float alpha, beta;
  int iterations;
  int m, n, k;
  int swizzle;

  Options():
    help(false),
    m(8192), n(8192), k(2048),
    alpha(1.f), beta(0.f),
    iterations(10),
    swizzle(0)
  { }

  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);
    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }
    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("iterations", iterations);
    cmd.get_cmd_line_argument("swizzle", swizzle);
  }

  std::ostream & print_usage(std::ostream &out) const {
    out << "70_blackwell_fp16_gemm_nvcomp\n\n"
      << "  Blackwell FP16 tcgen05 GEMM, then unfused nvCOMP ANS on 128x128 C tiles.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM\n"
      << "  --n=<int>                   Sets the N extent of the GEMM\n"
      << "  --k=<int>                   Sets the K extent of the GEMM\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n\n"
      << "  --swizzle=<int>             Cluster rasterization swizzle\n\n"
      << "  --iterations=<int>          Number of profiling iterations (GEMM+ANS each).\n\n";
    return out;
  }

  double gflops(double runtime_s) const {
    uint64_t flop = uint64_t(2) * m * n * k;
    double gflop = double(flop) / double(1.0e9);
    return gflop / runtime_s;
  }
};

struct Result {
  double avg_runtime_ms;
  double gflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  Result(
    double avg_runtime_ms = 0,
    double gflops = 0,
    cutlass::Status status = cutlass::Status::kSuccess,
    cudaError_t error = cudaSuccess)
  :
    avg_runtime_ms(avg_runtime_ms), gflops(gflops), status(status), error(error), passed(false)
  {}
};

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

template <class Element>
__global__ void initialize_small_ints_kernel(Element *ptr, size_t n, int seed) {
  size_t const idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  int const k = 16807;
  int const m = 16;
  float value = float(((static_cast<int>(idx) + seed) * k % m) - m / 2);
  ptr[idx] = Element(value);
}

// Same small-integer fill as 00_basic_gemm so ANS FLOAT16 sees a similar C histogram.
template <class Element>
bool initialize_block(
  cutlass::DeviceAllocation<Element>& block,
  int seed=0) {

  size_t const n = block.size();
  int const threads = 256;
  int const blocks = static_cast<int>((n + static_cast<size_t>(threads) - 1) / static_cast<size_t>(threads));
  initialize_small_ints_kernel<<<blocks, threads>>>(block.get(), n, seed);
  CUDA_CHECK(cudaGetLastError());
  return true;
}

void initialize(const Options &options) {
  stride_A = cutlass::make_cute_packed_stride(StrideA{}, {options.m, options.k, 1});
  stride_B = cutlass::make_cute_packed_stride(StrideB{}, {options.n, options.k, 1});
  stride_C = cutlass::make_cute_packed_stride(StrideC{}, {options.m, options.n, 1});
  stride_D = cutlass::make_cute_packed_stride(StrideD{}, {options.m, options.n, 1});

  block_A.reset(options.m * options.k);
  block_B.reset(options.k * options.n);
  block_C.reset(options.m * options.n);
  block_D.reset(options.m * options.n);
  block_ref_D.reset(options.m * options.n);

  initialize_block(block_A, 0);
  initialize_block(block_B, 17);
  initialize_block(block_C, 101);
}

typename Gemm::Arguments args_from_options(const Options &options) {
  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {options.m, options.n, options.k, 1},
    {block_A.get(), stride_A, block_B.get(), stride_B},
    {{options.alpha, options.beta}, block_C.get(), stride_C, block_D.get(), stride_D}
  };
  arguments.scheduler.max_swizzle_size = options.swizzle;
  return arguments;
}

bool verify(const Options &options) {
  cutlass::TensorRef ref_A(block_A.get(), Gemm::LayoutA::packed({options.m, options.k}));
  cutlass::TensorRef ref_B(block_B.get(), Gemm::LayoutB::packed({options.k, options.n}));
  cutlass::TensorRef ref_C(block_C.get(), Gemm::LayoutC::packed({options.m, options.n}));
  cutlass::TensorRef ref_D(block_ref_D.get(), Gemm::LayoutD::packed({options.m, options.n}));

  DeviceGemmReference gemm_reference;
  gemm_reference(
    {options.m, options.n, options.k},
    ElementAccumulator(options.alpha),
    ref_A,
    ref_B,
    ElementAccumulator(options.beta),
    ref_C,
    ref_D);
  CUDA_CHECK(cudaDeviceSynchronize());
  return cutlass::reference::device::BlockCompareEqual(block_ref_D.get(), block_D.get(), block_D.size());
}

template <typename GemmT>
int run(Options &options) {
  initialize(options);

  GemmT gemm;
  auto arguments = args_from_options(options);
  size_t workspace_size = GemmT::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  CUTLASS_CHECK(gemm.can_implement(arguments));
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));

  nvtxRangePushA("cutlass_gemm");
  CUTLASS_CHECK(gemm.run());
  CUDA_CHECK(cudaDeviceSynchronize());
  nvtxRangePop();

  Result result;
  result.passed = verify(options);
  std::cout << "  GEMM disposition: " << (result.passed ? "Passed" : "Failed") << std::endl;
  if (!result.passed) {
    return -1;
  }

  UnfusedAnsWorkspace ans;
  cudaError_t ans_err = ans.allocate(options.m, options.n);
  if (ans_err != cudaSuccess) {
    std::cerr << "ANS workspace allocate failed: " << cudaGetErrorString(ans_err) << std::endl;
    return -1;
  }

  int const ldc = options.m;
  nvtxRangePushA("nvcomp_ans");
  nvcompStatus_t nvst = ans.launch(block_D.get(), ldc, options.m, options.n);
  CUDA_CHECK(cudaDeviceSynchronize());
  nvtxRangePop();
  if (nvst != nvcompSuccess) {
    std::cerr << "nvcompBatchedANSCompressFromColMajorTilesAsync failed: "
              << nvcompGetStatusString(nvst) << std::endl;
    ans.release();
    return -1;
  }
  ans_err = ans.validate_roundtrip("nvCOMP LLIF ANS (unfused, Blackwell GEMM)");
  if (ans_err != cudaSuccess) {
    ans.release();
    return -1;
  }

  if (options.iterations > 0) {
    GpuTimer timer;
    timer.start();
    for (int iter = 0; iter < options.iterations; ++iter) {
      CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
      nvtxRangePushA("cutlass_gemm");
      CUTLASS_CHECK(gemm.run());
      nvtxRangePop();
      nvtxRangePushA("nvcomp_ans");
      nvst = ans.launch(block_D.get(), ldc, options.m, options.n);
      nvtxRangePop();
      if (nvst != nvcompSuccess) {
        break;
      }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    timer.stop();
    if (nvst != nvcompSuccess) {
      std::cerr << "nvcompBatchedANSCompressFromColMajorTilesAsync failed: "
                << nvcompGetStatusString(nvst) << std::endl;
      ans.release();
      return -1;
    }

    float elapsed_ms = timer.elapsed_millis();
    result.avg_runtime_ms = double(elapsed_ms) / double(options.iterations);
    result.gflops = options.gflops(result.avg_runtime_ms / 1000.0);

    std::cout << "  Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << std::endl;
    std::cout << "  Avg runtime (GEMM+ANS): " << result.avg_runtime_ms << " ms" << std::endl;
    std::cout << "  GEMM GFLOPS (ignores ANS): " << result.gflops << std::endl;
  }

  ans.release();
  return 0;
}

#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

int main(int argc, char const **args) {
  if (__CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 8)) {
    std::cerr << "This example requires CUDA 12.8 or newer." << std::endl;
    return 0;
  }

  cudaDeviceProp props;
  int current_device_id;
  CUDA_CHECK(cudaGetDevice(&current_device_id));
  CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));

  if (props.major != 10 || props.minor != 0) {
    std::cerr << "This example requires a GPU with compute capability 100a." << std::endl;
    return 0;
  }

  Options options;
  options.parse(argc, args);
  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  return run<Gemm>(options);
#else
  std::cerr << "CUTLASS_ARCH_MMA_SM100_SUPPORTED is not set; rebuild with -DCUTLASS_NVCC_ARCHS=100a."
            << std::endl;
  return 0;
#endif
}
