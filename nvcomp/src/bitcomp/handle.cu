#include "bitcomp_private.h"
#include "utilities.h"

#include <nvcomp/native/bitcomp.h>

extern "C" {
using namespace bitcomp;

bitcompResult_t bitcompCreatePlan(
  bitcompHandle_t *handle, // Plan handle
  size_t n, // Size of the uncompressed data in bytes
  bitcompDataType_t dataType, // Type of the data to compress
  bitcompMode_t mode, // Compression mode (lossy or lossless)
  bitcompAlgorithm_t algo
) // Compression algorithm

{
  bitcompContext *context = (bitcompContext *)malloc(sizeof(bitcompContext));
  if (context == NULL)
  {
    return BITCOMP_UNKNOWN_ERROR;
  }
  utilities::resetHandle(context);

  context->uncompressedSize = n;
  context->compMode = mode;
  context->algo = algo;
  context->dataType = dataType;
  switch (dataType)
  {
    case BITCOMP_UNSIGNED_8BIT:
    case BITCOMP_UNSIGNED_16BIT:
    case BITCOMP_UNSIGNED_32BIT:
    case BITCOMP_UNSIGNED_64BIT:
      // Unsigned integral types:
      // Can only be compressed in lossless mode
      if (mode != BITCOMP_LOSSLESS)
      {
        free(context);
        return BITCOMP_INVALID_PARAMETER;
      }
      // 2s complement not an issue -> default format
      context->ifmt = BITCOMP_DEFAULT_FORMAT;
      break;

    case BITCOMP_SIGNED_8BIT:
    case BITCOMP_SIGNED_16BIT:
    case BITCOMP_SIGNED_32BIT:
    case BITCOMP_SIGNED_64BIT:
      // Signed integral types:
      // Can only be compressed in lossless mode
      if (mode != BITCOMP_LOSSLESS)
      {
        free(context);
        return BITCOMP_INVALID_PARAMETER;
      }
      // 2s complement is an issue -> use custom format
      context->ifmt = BITCOMP_CUSTOM_INTEGER;
      break;

    case BITCOMP_FP16_DATA:
    case BITCOMP_FP32_DATA:
    case BITCOMP_FP64_DATA:
      if (mode == BITCOMP_LOSSLESS)
      {
        // FP data is usually compressed in lossy fashion.
        // If lossless is requested, treat as integral type
        if (dataType == BITCOMP_FP16_DATA)
        {
          context->dataType = BITCOMP_UNSIGNED_16BIT;
        }
        else if (dataType == BITCOMP_FP32_DATA)
        {
          context->dataType = BITCOMP_UNSIGNED_32BIT;
        }
        else if (dataType == BITCOMP_FP64_DATA)
        {
          context->dataType = BITCOMP_UNSIGNED_64BIT;
        }
        context->ifmt = BITCOMP_DEFAULT_FORMAT;
      }
      else if (mode == BITCOMP_LOSSY_FP_TO_SIGNED)
      {
        context->ifmt = BITCOMP_CUSTOM_INTEGER;
      }
      else
      {
        context->ifmt = BITCOMP_DEFAULT_FORMAT;
      }
      break;

    default: // Unknown type?
      free(context);
      return BITCOMP_INVALID_PARAMETER;
  }

  *handle = context;
  return BITCOMP_SUCCESS;
}

// bitcompCreatePlanFromCompressedData: Use the compressed data header.
bitcompResult_t bitcompCreatePlanFromCompressedData(
  bitcompHandle_t *handle, // Plan handle
  const void *data
) // Compressed data from which the plan will be read
{
  if (data == NULL)
  {
    return BITCOMP_INVALID_PARAMETER;
  }
  bitcompContext *context = (bitcompContext *)malloc(sizeof(bitcompContext));
  utilities::resetHandle(context);
  size_t lcomp;
  if (utilities::getCompressionInfo(data, context, &lcomp) != BITCOMP_SUCCESS)
  {
    free(context);
    return BITCOMP_INVALID_PARAMETER;
  }
  *handle = context;
  return BITCOMP_SUCCESS;
}

// bitcompDestroyPlan: Detroy an existing bitcomp plan
bitcompResult_t bitcompDestroyPlan(bitcompHandle_t handle)
{
  if (handle == NULL)
  {
    return BITCOMP_INVALID_PARAMETER;
  }
  if (handle->deviceCounter != nullptr)
  {
    // Wait for the associated stream to finish only if a device counter is used
    if (cudaStreamSynchronize(handle->stream) != cudaSuccess)
    {
      return BITCOMP_CUDA_API_ERROR;
    }
    if (cudaFree(handle->deviceCounter) != cudaSuccess)
    {
      return BITCOMP_CUDA_API_ERROR;
    }
  }
  free(handle);
  return BITCOMP_SUCCESS;
}

//***********************************************************************************************
// Modification of plan attributes

// bitcompSetStream: Associate the bitcomp plan to a particular stream
bitcompResult_t bitcompSetStream(bitcompHandle_t handle, cudaStream_t stream)
{
  if (handle == NULL)
  {
    return BITCOMP_INVALID_PARAMETER;
  }
  handle->stream = stream;
  return BITCOMP_SUCCESS;
}

// bitcompAccelerateRemoteCompression:
// Accelerate compression when the compressed output is not in the local device's global memory
bitcompResult_t bitcompAccelerateRemoteCompression(bitcompHandle_t handle)
{
  if (handle == NULL)
  {
    return BITCOMP_INVALID_PARAMETER;
  }
  // If the pointer has already been set, ignore.
  if (handle->deviceCounter != nullptr)
  {
    return BITCOMP_SUCCESS;
  }
  // Allocate a device-memory counter for faster atomics
  if (handle->batches == 0)
  {
    if (cudaMalloc((void **)&handle->deviceCounter, sizeof(uint64)) != cudaSuccess)
    {
      handle->deviceCounter = nullptr;
      return BITCOMP_CUDA_API_ERROR;
    }
  }
  else
  {
    // Batch mode: no atomics, no device pointer needed
  }
  return BITCOMP_SUCCESS;
}

//***********************************************************************************************
// Handle utilities

// bitcompGetUncompressedSizeFromHandle:
// Get the uncompressed size associated with a handle
bitcompResult_t bitcompGetUncompressedSizeFromHandle(const bitcompHandle_t handle, size_t *bytes)
{
  if (handle == NULL)
  {
    return BITCOMP_INVALID_PARAMETER;
  }
  *bytes = handle->uncompressedSize;
  return BITCOMP_SUCCESS;
}

// bitcompGetdataTypeFromHandle:
// Get the data type associated with a handle
bitcompResult_t bitcompGetDataTypeFromHandle(const bitcompHandle_t handle, bitcompDataType_t *dataType)
{
  if (handle == NULL)
  {
    return BITCOMP_INVALID_PARAMETER;
  }
  *dataType = handle->dataType;
  return BITCOMP_SUCCESS;
}
}
