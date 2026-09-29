/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026 NVIDIA CORPORATION & AFFILIATES.
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

#include <cuda/std/bit>
#include <cuda/type_traits>

#include <cassert>
#include <limits>

#include "bitcomp_private.h"
#include "exception.hpp" // nvcomp::NVCompException

namespace bitcomp
{
namespace utilities
{
template <typename T>
__host__ __device__ T zeroMantissaBits(const T delta)
{
  static_assert(cuda::is_floating_point_v<T>, "Template parameter T must be a floating point type");
  // Remove mantissa from delta
  if constexpr (cuda::std::is_same_v<T, float>)
  {
    uint32_t delta_as_int = cuda::std::bit_cast<uint32_t>(delta);
    delta_as_int &= 0xFF800000; // keep sign and exponent bits only
    return cuda::std::bit_cast<T>(delta_as_int);
  }
  else if constexpr (cuda::std::is_same_v<T, double>)
  {
    uint64_t delta_as_int = cuda::std::bit_cast<uint64_t>(delta);
    delta_as_int &= 0xFFF0000000000000; // keep sign and exponent bits only
    return cuda::std::bit_cast<T>(delta_as_int);
  }
  else
  {
    uint16_t delta_as_int = cuda::std::bit_cast<uint16_t>(delta);
    delta_as_int &= 0xFC00; // keep sign and exponent bits only
    return cuda::std::bit_cast<T>(delta_as_int);
  }
}

bool validBufferAlignment(const void *adr, const size_t alignment);

int getSizeofBitcompType(bitcompDataType_t type);

// Map an nvcomp data type to the corresponding Bitcomp data type. Kept inline in
// this header so it can be reused (e.g. from tests) without exporting an internal
// symbol from the shared library.
inline bitcompDataType_t nvcomp_to_bitcomp_data_type(nvcompType_t data_type)
{
  switch (data_type)
  {
    case NVCOMP_TYPE_BITS:
    case NVCOMP_TYPE_UCHAR:
      return BITCOMP_UNSIGNED_8BIT;
    case NVCOMP_TYPE_CHAR:
      return BITCOMP_SIGNED_8BIT;
    case NVCOMP_TYPE_USHORT:
      return BITCOMP_UNSIGNED_16BIT;
    case NVCOMP_TYPE_SHORT:
      return BITCOMP_SIGNED_16BIT;
    case NVCOMP_TYPE_UINT:
      return BITCOMP_UNSIGNED_32BIT;
    case NVCOMP_TYPE_INT:
      return BITCOMP_SIGNED_32BIT;
    case NVCOMP_TYPE_ULONGLONG:
      return BITCOMP_UNSIGNED_64BIT;
    case NVCOMP_TYPE_LONGLONG:
      return BITCOMP_SIGNED_64BIT;
    case NVCOMP_TYPE_FLOAT16:
      return BITCOMP_FP16_DATA;
    case NVCOMP_TYPE_FLOAT32:
      return BITCOMP_FP32_DATA;
    case NVCOMP_TYPE_FLOAT64:
      return BITCOMP_FP64_DATA;
    default:
      throw nvcomp::NVCompException(nvcompErrorNotSupported, "Unsupported nvcomp data type.");
  }
}

bool validUncompressedBufferAlignment(const void *adr, bitcompDataType_t type);

bool validCompressedBufferAlignment(const void *adr);

void resetHandle(bitcompHandle_t handle);

bitcompResult_t getCompressionInfo(const void *compressedData, bitcompHandle_t handle, size_t *lcomp);

int getDevInfo(
  void *compressedData,
  size_t *sizeComp,
  size_t *sizeUncomp,
  size_t *nblocks,
  size_t *incompressibleBlocks,
  int *minBlockSize,
  int *maxBlockSize,
  int *avgBlockSize
);

nvcompStatus_t bitcompGetBatchCompressedInfo(
  const void *const *compressed_data,
  size_t batches,
  batchCompInfo_t &host_comp_info,
  cudaStream_t stream,
  void *device_temp_ptr
);

template <bool batch = false>
bool valid_handle(const bitcompHandle_t &handle)
{
  switch (handle->dataType)
  {
    case BITCOMP_UNSIGNED_8BIT:
    case BITCOMP_UNSIGNED_16BIT:
    case BITCOMP_UNSIGNED_32BIT:
    case BITCOMP_UNSIGNED_64BIT:
      if (handle->compMode != BITCOMP_LOSSLESS || handle->ifmt != BITCOMP_DEFAULT_FORMAT)
      {
        return false;
      }
      break;

    case BITCOMP_SIGNED_8BIT:
    case BITCOMP_SIGNED_16BIT:
    case BITCOMP_SIGNED_32BIT:
    case BITCOMP_SIGNED_64BIT:
      if (handle->compMode != BITCOMP_LOSSLESS || handle->ifmt != BITCOMP_CUSTOM_INTEGER)
      {
        return false;
      }
      break;

    case BITCOMP_FP16_DATA:
    case BITCOMP_FP32_DATA:
    case BITCOMP_FP64_DATA:
      if (handle->compMode == BITCOMP_LOSSLESS ||
          (handle->compMode == BITCOMP_LOSSY_FP_TO_SIGNED && handle->ifmt != BITCOMP_CUSTOM_INTEGER) ||
          (handle->compMode == BITCOMP_LOSSY_FP_TO_UNSIGNED && handle->ifmt != BITCOMP_DEFAULT_FORMAT))
      {
        return false;
      }
  }

  if (batch != (handle->batches != 0))
  {
    return false;
  }

  return true;
}

} // namespace utilities

} // namespace bitcomp
