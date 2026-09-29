/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
*/

#pragma once

#include "nvcomp/native/bitcomp.h"

#ifdef __cplusplus
extern "C" {
#endif
typedef enum bitcompIntFormat_t
{
  BITCOMP_DEFAULT_FORMAT = 0,
  BITCOMP_CUSTOM_INTEGER
} bitcompIntFormat_t;

typedef struct
{
  bitcompDataType_t dataType;
  bitcompMode_t mode;
  bitcompAlgorithm_t algo;
  bitcompIntFormat_t ifmt;
  int error;
} batchCompInfo_t;
#ifdef __cplusplus
}
#endif

namespace bitcomp
{

typedef unsigned long long uint64;
typedef unsigned int uint;

constexpr int NOMINAL_BLOCK_SIZE = 8192;

bool validUncompressedBufferAlignment(void *adr, bitcompDataType_t type);

bool validCompressedBufferAlignment(void *adr);

bitcompResult_t getCompressionInfo(
  const void *compressedData,
  size_t &sizeComp,
  size_t &sizeUncomp,
  bitcompAlgorithm_t &algo,
  bitcompDataType_t &dataType,
  bitcompMode_t &compMode,
  bitcompIntFormat_t &ifmt
);

int getDevInfo(
  void *compressedData,
  size_t *sizeComp,
  size_t *sizeUncomp,
  size_t *nblocks,
  size_t *nblocksNotcomp,
  int *minBlockSize,
  int *maxBlockSize,
  int *avgBlockSize
);

template <typename T, bitcompDataType_t typeId, bitcompMode_t compMode, bitcompIntFormat_t ifmt>
nvcompStatus_t launchBatchEncoder(
  const void *const *in,
  void *const *out,
  bitcompAlgorithm_t algo,
  const size_t *nbytes,
  size_t *outputSizes,
  size_t nbatch,
  T delta,
  cudaStream_t stream
);

nvcompStatus_t launchBatchDecoder(
  const void *const *input,
  void *const *output,
  const size_t *output_buffer_sizes,
  nvcompStatus_t *statuses,
  size_t *uncompressed_sizes,
  const batchCompInfo_t &comp_info,
  size_t batch_size,
  cudaStream_t stream
);

} // namespace bitcomp

#ifdef __cplusplus
extern "C" {
#endif

typedef struct bitcompContext
{
  size_t uncompressedSize; // Not use in batch mode, in bytes
  size_t batches; // 0 unless batch mode is used
  void *deviceCounter;
  bitcompDataType_t dataType;
  bitcompAlgorithm_t algo;
  bitcompMode_t compMode;
  bitcompIntFormat_t ifmt;
  cudaStream_t stream;
  bitcompResult_t lastError;
} bitcompContext;

nvcompStatus_t bitcompBatchGetUncompressedSizesAsync(
  const void *const *compressedData,
  size_t *uncompressedSizes,
  size_t batch,
  cudaStream_t stream
);

#ifdef __cplusplus
}
#endif
