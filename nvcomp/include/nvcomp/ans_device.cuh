/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * Host helpers for in-kernel ANS LLIF compress. compress_chunk is a header
 * device function (ans_device_chunk.cuh); instantiate it in the caller TU so
 * fused GEMM does not need RDC. The host batched API remains
 * nvcompBatchedANSCompressAsync.
 */

#ifndef NVCOMP_ANS_DEVICE_H
#define NVCOMP_ANS_DEVICE_H

#include "nvcomp/ans.h"

#ifdef __cplusplus

/// Thread count required by the inlined ANS compressor (CUTLASS 128x128 SIMT CTA).
static constexpr int NVCOMP_DEVICE_ANS_COMPRESS_BLOCK_THREADS = 256;

extern "C" nvcompStatus_t nvcompBatchedANSCompressGetDeviceLaunchParams(
  size_t num_chunks,
  size_t max_uncompressed_chunk_bytes,
  nvcompBatchedANSCompressOpts_t compress_opts,
  int *max_sub_chunk_size,
  uint32_t *slot_words,
  size_t *smem_bytes,
  size_t *smem_alignment,
  int *block_threads
);

/**
 * @brief Host launch of compress_kernel that first packs a column-major C tile
 * into each uncompressed chunk buffer. FLOAT16 copies 16-bit elements (fp16 or
 * bf16). pack_mn_swapped=0 matches the 128x128 column-major gather used by
 * --nvcomp-only. Uncompressed chunk pointers are the pack destinations.
 */
extern "C" nvcompStatus_t nvcompBatchedANSCompressFromColMajorTilesAsync(
  const void *device_C,
  int ldc,
  int M,
  int N,
  int tile_m,
  int tile_n,
  int pack_mn_swapped,
  const void *const *device_uncompressed_chunk_ptrs,
  const size_t *device_uncompressed_chunk_bytes,
  size_t max_uncompressed_chunk_bytes,
  size_t num_chunks,
  void *const *device_compressed_chunk_ptrs,
  size_t *device_compressed_chunk_bytes,
  nvcompBatchedANSCompressOpts_t compress_opts,
  nvcompStatus_t *device_statuses,
  cudaStream_t stream
);

#endif // __cplusplus

#endif // NVCOMP_ANS_DEVICE_H
