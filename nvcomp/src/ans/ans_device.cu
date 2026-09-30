/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Relocatable device entry for in-kernel ANS LLIF compress. Kept in its own
 * translation unit so the rest of nvCOMP can be compiled without -rdc (gdeflate
 * headers define non-inline __device__ tables that nvlink otherwise duplicates).
 */

#include "nvcomp/ans.h"
#include "nvcomp/ans_device.cuh"
#include "ans/compress_kernels_llif.cuh"

__device__ void nvcompDeviceANSCompressChunk(
  void *compressed,
  const void *uncompressed,
  size_t uncompressed_bytes,
  size_t *compressed_size,
  int max_sub_chunk_size,
  uint32_t slot_words,
  void *smem
)
{
  using Policy = ans_gpu_lib::detail::CharEncodePolicy<ans_gpu_lib::detail::CharX2EncodeImpl>;
  static_assert(
    NVCOMP_DEVICE_ANS_COMPRESS_BLOCK_THREADS == static_cast<int>(ans_gpu_lib::NUM_COMP_THREADS_PER_CTA),
    "device ANS compress CTA width must match the LLIF compressor"
  );
  using Workspace = ans_gpu_lib::detail::DeviceCompressSmem<Policy>;
  auto &workspace = *reinterpret_cast<Workspace *>(smem);
  ans_gpu_lib::detail::compress_chunk<Policy, /*Sampled=*/false>(
    compressed,
    uncompressed,
    static_cast<ans_gpu_lib::IndexT>(uncompressed_bytes),
    compressed_size,
    max_sub_chunk_size,
    nullptr,
    slot_words,
    /*histogram_reduction_log2=*/0u,
    workspace.workspace,
    workspace.packed_chunk_size_bytes
  );
}
