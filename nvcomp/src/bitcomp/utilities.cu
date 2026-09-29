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

#include <cuda_fp16.h>

#include "bitcomp_private.h"
#include "common.h"
#include "header.h"
#include "utilities.h"

#include <nvcomp/native/bitcomp.h>

namespace bitcomp
{

namespace utilities
{

// *******************************************************************************************************************
int getSizeofBitcompType(bitcompDataType_t type)
{
  switch (type)
  {
    case BITCOMP_UNSIGNED_8BIT:
    case BITCOMP_SIGNED_8BIT:
      return sizeof(uint8_t);
    case BITCOMP_UNSIGNED_16BIT:
    case BITCOMP_SIGNED_16BIT:
    case BITCOMP_FP16_DATA:
      return sizeof(uint16_t);
    case BITCOMP_UNSIGNED_32BIT:
    case BITCOMP_SIGNED_32BIT:
      return sizeof(uint32_t);
    case BITCOMP_UNSIGNED_64BIT:
    case BITCOMP_SIGNED_64BIT:
      return sizeof(uint64_t);
    case BITCOMP_FP32_DATA:
      return sizeof(float);
    case BITCOMP_FP64_DATA:
      return sizeof(double);
    default:
      return -1; // Invalid type
  }
}

// *******************************************************************************************************************
bool validBufferAlignment(const void *adr, const size_t alignment)
{
  return (reinterpret_cast<uintptr_t>(adr) % alignment == 0);
}

// *******************************************************************************************************************
// We expect the uncompressed buffer to be aligned to the type.

bool validUncompressedBufferAlignment(const void *adr, bitcompDataType_t type)
{
  const int type_size = getSizeofBitcompType(type);
  return type_size > 0 ? validBufferAlignment(adr, type_size) : false;
}

// *******************************************************************************************************************
// We expect the compressed buffer to be 64-bit aligned.

bool validCompressedBufferAlignment(const void *adr) { return validBufferAlignment(adr, alignof(void *)); }

// *******************************************************************************************************************

void resetHandle(bitcompHandle_t handle) { memset(handle, 0, sizeof(bitcompContext)); }

// *******************************************************************************************************************
// Read header info (from the host), copying the data locally if it's not accessible.

bitcompResult_t getCompressionInfo(const void *compressedData, bitcompContext *context, size_t *lcomp)
{
  struct cudaPointerAttributes attr;
  uint64 hostHeader[header::globalHeaderValues];
  uint64 *hdr;

  if (compressedData == NULL)
  {
    return BITCOMP_INVALID_PARAMETER;
  }

  if (cudaPointerGetAttributes(&attr, compressedData) == cudaSuccess && attr.hostPointer == NULL &&
      attr.devicePointer != NULL)
  {
    if (cudaMemcpy(hostHeader, compressedData, header::globalHeaderValues * sizeof(uint64), cudaMemcpyDefault) !=
        cudaSuccess)
    {
      return BITCOMP_INVALID_PARAMETER;
    }
    hdr = hostHeader;
  }
  else
  {
    hdr = (uint64 *)compressedData;
  }

  if (!header::hasValidMagicNumber(hdr))
  {
    return BITCOMP_INVALID_COMPRESSED_DATA;
  }

  *lcomp = header::getTotalCompressedSize(hdr);
  utilities::resetHandle(context);
  context->uncompressedSize = header::getUncompressedSize(hdr);
  context->algo = header::getAlgorithm(hdr);
  context->dataType = header::getDataType(hdr);
  context->compMode = header::getCompMode(hdr);
  context->ifmt = header::getIntFormat(hdr);

  return BITCOMP_SUCCESS;
}

// *******************************************************************************************************************
// Getting developer info about compressed data

int getDevInfo(
  void *compressedData,
  size_t *sizeComp,
  size_t *sizeUncomp,
  size_t *nblocks,
  size_t *incompressibleBlocks,
  int *minBlockSize,
  int *maxBlockSize,
  int *avgBlockSize
)
{
  struct cudaPointerAttributes attr;
  uint64 hostHeader[header::globalHeaderValues];
  uint64 *hdr;

  if (compressedData == NULL)
  {
    return BITCOMP_INVALID_PARAMETER;
  }

  if (cudaPointerGetAttributes(&attr, compressedData) != cudaSuccess)
  {
    return BITCOMP_INVALID_PARAMETER;
  }

  if (attr.hostPointer == NULL && attr.devicePointer != NULL)
  {
    // Device pointer, not visible on the host -> copy
    if (cudaMemcpy(hostHeader, compressedData, header::globalHeaderValues * sizeof(uint64), cudaMemcpyDefault) !=
        cudaSuccess)
    {
      return BITCOMP_INVALID_PARAMETER;
    }
    size_t hdrlen = header::getHeaderLength(hostHeader);
    hdr = (uint64 *)malloc(hdrlen);
    if (cudaMemcpy(hdr, compressedData, hdrlen, cudaMemcpyDefault) != cudaSuccess)
    {
      free(hdr);
      return BITCOMP_INVALID_PARAMETER;
    }
  }
  else
  {
    hdr = (uint64 *)compressedData;
  }

  if (!header::hasValidMagicNumber((void *)hdr))
  {
    if (hdr != compressedData)
    {
      free(hdr);
    }
    return BITCOMP_INVALID_COMPRESSED_DATA;
  }

  *sizeUncomp = header::getUncompressedSize((void *)hdr);
  *sizeComp = header::getTotalCompressedSize((void *)hdr);
  *nblocks = header::getNumBlocks((void *)hdr);
  int counter = 0;
  int mini = NOMINAL_BLOCK_SIZE;
  int maxi = 0;
  size_t avg = 0;
  for (size_t iblock = 0; iblock < *nblocks; iblock++)
  {
    bool overflow;
    uint lcomp;
    uint64 offset;
    if (iblock > std::numeric_limits<int>::max())
    {
      if (hdr != compressedData)
      {
        free(hdr);
      }
      return BITCOMP_INVALID_COMPRESSED_DATA;
    }
    header::getBlockInfo(hdr, static_cast<int>(iblock), offset, lcomp, overflow);
    if (overflow)
    {
      counter++;
    }
    mini = min(mini, lcomp * 4);
    maxi = max(maxi, lcomp * 4);
    avg += lcomp * 4;
  }
  *incompressibleBlocks = counter;
  *minBlockSize = mini;
  *maxBlockSize = maxi;
  *avgBlockSize = (int)(avg / *nblocks);

  if (hdr != compressedData)
  {
    free(hdr);
  }

  return BITCOMP_SUCCESS;
}

// *******************************************************************************************************************
// Retrieve compression info (type, mode, algo, integer format) from batched compressed data
// The compressed data and the compinfo structure must be device-visible

__global__ void batch_query_compinfo(const void *const *compressedData, size_t batch, batchCompInfo_t *compinfo)
{
  size_t index = blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  if (index >= batch)
  {
    return;
  }

  const void *hdr = compressedData[index];
  if (!header::hasValidMagicNumber(hdr))
  {
    atomicExch(&compinfo->error, 1);
    return;
  }
  bitcompDataType_t dataType = header::getDataType(hdr);
  bitcompMode_t mode = header::getCompMode(hdr);
  bitcompAlgorithm_t algo = header::getAlgorithm(hdr);
  bitcompIntFormat_t ifmt = header::getIntFormat(hdr);
  // First thread writes the info to the output structure
  if (index == 0)
  {
    compinfo->dataType = dataType;
    compinfo->mode = mode;
    compinfo->algo = algo;
    compinfo->ifmt = ifmt;
  }
  else
  // Other threads make sure their value match their lower neighbor's
  {
    const void *hdrm1 = compressedData[index - 1];
    bitcompDataType_t dataType_m1 = header::getDataType(hdrm1);
    bitcompMode_t mode_m1 = header::getCompMode(hdrm1);
    bitcompAlgorithm_t algo_m1 = header::getAlgorithm(hdrm1);
    bitcompIntFormat_t ifmt_m1 = header::getIntFormat(hdrm1);
    if (dataType != dataType_m1 || mode != mode_m1 || algo != algo_m1 || ifmt != ifmt_m1)
    {
      atomicExch(&compinfo->error, 1);
    }
  }
}

nvcompStatus_t bitcompGetBatchCompressedInfo(
  const void *const *compressedData,
  size_t batches,
  batchCompInfo_t &host_comp_info,
  cudaStream_t stream,
  void *const device_temp_ptr
)
{
  batchCompInfo_t *device_comp_info = static_cast<batchCompInfo_t *>(device_temp_ptr);

  CUDA_CHECK(cudaMemsetAsync(device_comp_info, 0, sizeof(*device_comp_info), stream));

  constexpr int BLOCK_SIZE = 512;
  batch_query_compinfo<<<nvcomp::cuda_dim_cast(nvcomp::roundUpDiv(batches, BLOCK_SIZE)), BLOCK_SIZE, 0, stream>>>(
    compressedData,
    batches,
    device_comp_info
  );
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpyAsync(&host_comp_info, device_comp_info, sizeof(host_comp_info), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  if (host_comp_info.error != 0)
  {
    return nvcompErrorCannotDecompress;
  }
  return nvcompSuccess;
}

} // namespace utilities

} // namespace bitcomp

extern "C" {

using namespace bitcomp;

// *******************************************************************************************************************
// Worst case scenario: header + all blocks uncompressed at 8KB.
size_t bitcompMaxBuflen(size_t nbytes)
{
  return (header::computeHeaderLength((uint64)nbytes) + header::computeNumBlocks((uint64)nbytes) * 8192ULL);
}

// Deprecated version
size_t bitcomp_max_buflen(size_t nbytes) { return bitcompMaxBuflen(nbytes); }

// *******************************************************************************************************************
bitcompResult_t bitcompGetCompressedSize(const void *compressedData, size_t *size)
{
  bitcompContext context;
  size_t lcomp;
  bitcompResult_t ier = utilities::getCompressionInfo(compressedData, &context, &lcomp);
  if (ier != BITCOMP_SUCCESS)
  {
    *size = 0;
    return ier;
  }
  *size = lcomp;
  return BITCOMP_SUCCESS;
}

// *******************************************************************************************************************
// Kernel to read the compressed size asynchronously

__global__ void compsize_query(const void *hdr, size_t *lcomp)
{
  size_t size = 0;
  if (header::hasValidMagicNumber(hdr))
  {
    size = header::getTotalCompressedSize(hdr);
  }
  *lcomp = size;
}

bitcompResult_t bitcompGetCompressedSizeAsync(const void *compressedData, size_t *size, cudaStream_t stream)
{
  if (compressedData == NULL)
  {
    return BITCOMP_INVALID_PARAMETER;
  }
  compsize_query<<<1, 1, 0, stream>>>(compressedData, size);
  return cudaGetLastError() == cudaSuccess ? BITCOMP_SUCCESS : BITCOMP_CUDA_KERNEL_LAUNCH_ERROR;
}

// *******************************************************************************************************************
bitcompResult_t bitcompGetUncompressedSize(const void *compressedData, size_t *size)
{
  bitcompContext context;
  size_t lcomp;
  bitcompResult_t ier = utilities::getCompressionInfo(compressedData, &context, &lcomp);
  if (ier != BITCOMP_SUCCESS)
  {
    *size = 0;
    return ier;
  }
  *size = context.uncompressedSize;
  return BITCOMP_SUCCESS;
}

// *******************************************************************************************************************
bitcompResult_t bitcompGetCompressedInfo(
  const void *compressedData,
  size_t *compressedSize, // Size of the compressed data (input/output)
  size_t *uncompressedSize, // Size of the uncompressed data in bytes
  bitcompDataType_t *dataType, // Type of the data
  bitcompMode_t *mode, // Compression mode (lossy or lossless)
  bitcompAlgorithm_t *algo
) // Algoirhtm (default or sparse)
{
  if (*compressedSize < header::globalHeaderValues * 8)
  {
    return BITCOMP_INVALID_PARAMETER;
  }
  bitcompContext context;
  size_t lcomp;
  bitcompResult_t ier = utilities::getCompressionInfo(compressedData, &context, &lcomp);
  if (ier != BITCOMP_SUCCESS)
  {
    return ier;
  }
  if (*compressedSize < lcomp)
  {
    return BITCOMP_INVALID_PARAMETER;
  }
  *compressedSize = lcomp;
  *uncompressedSize = context.uncompressedSize;
  *dataType = context.dataType;
  *mode = context.compMode;
  *algo = context.algo;
  return BITCOMP_SUCCESS;
}

// *******************************************************************************************************************

__global__ void
batch_query_uncompressed_sizes(const void *const *compressedData, size_t *uncompressedSizes, size_t batch)
{
  size_t index = blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  if (index >= batch)
  {
    return;
  }
  const void *hdr = compressedData[index];
  size_t uncompSize = 0;
  if (header::hasValidMagicNumber(hdr))
  {
    uncompSize = header::getUncompressedSize(hdr);
  }
  uncompressedSizes[index] = uncompSize;
}

nvcompStatus_t bitcompBatchGetUncompressedSizesAsync(
  const void *const *compressedData,
  size_t *uncompressedSizes,
  size_t batch_size,
  cudaStream_t stream
)
{
  if (batch_size == 0)
  {
    return nvcompSuccess;
  }

  constexpr int BLOCK_SIZE = 512;
  batch_query_uncompressed_sizes<<<nvcomp::cuda_dim_cast(nvcomp::roundUpDiv(batch_size, BLOCK_SIZE)), BLOCK_SIZE, 0, stream>>>(
    compressedData,
    uncompressedSizes,
    batch_size
  );
  return cudaGetLastError() == cudaSuccess ? nvcompSuccess : nvcompErrorCudaError;
}

// *******************************************************************************************************************
}
