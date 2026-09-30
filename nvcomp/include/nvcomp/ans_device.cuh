/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * Device-callable ANS LLIF compress. The host batched API remains
 * nvcompBatchedANSCompressAsync (one 128-thread CTA per chunk). This entry
 * point runs the same algorithm inside a caller kernel; the CTA must have
 * NVCOMP_DEVICE_ANS_COMPRESS_BLOCK_THREADS threads.
 */

#ifndef NVCOMP_ANS_DEVICE_H
#define NVCOMP_ANS_DEVICE_H

#include "nvcomp/ans.h"

#ifdef __cplusplus

/// Thread count required by nvcompDeviceANSCompressChunk (CUTLASS 128x128 SIMT CTA).
static constexpr int NVCOMP_DEVICE_ANS_COMPRESS_BLOCK_THREADS = 256;

#ifdef __CUDACC__

/**
 * @brief Compress one uncompressed buffer with char/uint8 rANS LLIF.
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
 * @param[in] slot_words From GetDeviceLaunchParams.
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
