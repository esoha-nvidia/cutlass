/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * Device-callable ANS LLIF compress. The host batched API remains
 * nvcompBatchedANSCompressAsync (one CTA per chunk, same thread count). This
 * entry point runs the same algorithm inside a caller kernel; the CTA must
 * have NVCOMP_DEVICE_ANS_COMPRESS_BLOCK_THREADS threads.
 */

#ifndef NVCOMP_ANS_DEVICE_H
#define NVCOMP_ANS_DEVICE_H

#include "nvcomp/ans.h"

#ifdef __cplusplus

/// Thread count required by nvcompDeviceANSCompressChunk (CUTLASS 128x128 SIMT CTA).
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
 * @brief Host launch of compress_kernel that first packs a column-major float
 * tile into each uncompressed chunk buffer. pack_mn_swapped=0 matches the
 * 128x128 column-major gather used by --nvcomp-only. Uncompressed chunk pointers
 * are the pack destinations.
 */
extern "C" nvcompStatus_t nvcompBatchedANSCompressFromColMajorTilesAsync(
  const float *device_C,
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

#ifdef __CUDACC__

/**
 * @brief Compress one uncompressed buffer with char/uint8 rANS LLIF
 * (two states per lane, exact histogram).
 *
 * Must be called by every thread of a CTA whose width is
 * NVCOMP_DEVICE_ANS_COMPRESS_BLOCK_THREADS. `smem` must be at least
 * `smem_bytes` from nvcompBatchedANSCompressGetDeviceLaunchParams and aligned
 * to `smem_alignment`. `compressed` must satisfy
 * nvcompANSRequiredCompressionAlignment.
 *
 * @param[out] compressed Destination bitstream for this chunk.
 * @param[in] uncompressed Source bytes.
 * @param[in] uncompressed_bytes Source size in bytes.
 * @param[out] compressed_size Written compressed size (device pointer).
 * @param[in] max_sub_chunk_size From GetDeviceLaunchParams.
 * @param[in] slot_words Sub-chunk compressed slot size in bytes
 *            (from GetDeviceLaunchParams).
 * @param[in] smem CTA shared workspace.
 */
extern __device__ void nvcompDeviceANSCompressChunk(
  void *compressed,
  const void *uncompressed,
  size_t uncompressed_bytes,
  size_t *compressed_size,
  int max_sub_chunk_size,
  uint32_t slot_words,
  void *smem
);

#endif // __CUDACC__
#endif // __cplusplus

#endif // NVCOMP_ANS_DEVICE_H
