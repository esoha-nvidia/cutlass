/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Relocatable device entry for in-kernel ANS LLIF compress. Kept in its own
 * translation unit so the rest of nvCOMP can be compiled without -rdc (gdeflate
 * headers define non-inline __device__ tables that nvlink otherwise duplicates).
 */

#include "ans/compress_kernels_llif.cuh"
#include "nvcomp/ans_device.cuh"

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
  using Policy = ans_gpu_lib::detail::CharEncodePolicy;
  constexpr int kThreads = NVCOMP_DEVICE_ANS_COMPRESS_BLOCK_THREADS;
  using Smem = ans_gpu_lib::detail::CompressSmem<Policy, kThreads / WARP_SIZE>;
  auto &workspace = *reinterpret_cast<Smem *>(smem);
  ans_gpu_lib::detail::compress_chunk<Policy, kThreads>(
    compressed,
    uncompressed,
    static_cast<ans_gpu_lib::IndexT>(uncompressed_bytes),
    *compressed_size,
    max_sub_chunk_size,
    slot_words,
    workspace
  );
}
