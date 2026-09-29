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

#include <cstdint>

#include "bitcomp_private.h"
#include "common.h"
#include "nvcomp/native/bitcomp.h"
#include "rle.h"
#include "utilities.h"
#include "zbmap.h"

#include <stdio.h>

namespace bitcomp
{

// *******************************************************************************************************************
// Generic encoder kernel

template <bitcompAlgorithm_t algo, typename T, bitcompDataType_t typeId, bitcompMode_t compMode, bitcompIntFormat_t ifmt>
#if defined(__CUDA_ARCH__)
#if __CUDA_ARCH__ == 1200
__launch_bounds__(256, (algo == BITCOMP_DEFAULT_ALGO) ? 5 : 6)
#else
__launch_bounds__(256)
#endif
#else
__launch_bounds__(256)
#endif
  __global__ void encoder_kernel(
    const void *__restrict__ in,
    void *__restrict__ out,
    void *__restrict__ counter,
    size_t nbytes,
    T delta
  )
{
  // Special case for zero-sized blocks
  if (nbytes == 0)
  {
    if (threadIdx.x == 0)
    {
      header::setFlags<typeId, compMode, algo, ifmt>(out);
      header::setUncompressedSize(out, 0);
    }
    return;
  }
  const char *p_in = reinterpret_cast<const char *>(in);
  uint *p_out = reinterpret_cast<uint *>(out);
  unsigned long long *p_counter = reinterpret_cast<unsigned long long *>(counter);
  uint64 blockOffset; // Not used. Only used for batch processing
  if (algo == BITCOMP_DEFAULT_ALGO)
  {
    rle::encoder<T, typeId, compMode, ifmt, true>(p_in, p_out, p_counter, blockOffset, (uint64)nbytes, blockIdx.x, delta);
  }
  if (algo == BITCOMP_SPARSE_ALGO)
  {
    zbmap::encoder<T, typeId, compMode, ifmt, true>(
      p_in,
      p_out,
      p_counter,
      blockOffset,
      (uint64)nbytes,
      blockIdx.x,
      delta
    );
  }
}

// *******************************************************************************************************************
template <typename T>
inline __device__ T getDelta(T scale)
{
  return utilities::zeroMantissaBits(scale);
}

// *******************************************************************************************************************
// Generic batch encoder kernel, with a unique (scalar) delta
template <bitcompAlgorithm_t algo, typename T, bitcompDataType_t typeId, bitcompMode_t compMode, bitcompIntFormat_t ifmt>
#if defined(__CUDA_ARCH__)
#if __CUDA_ARCH__ == 800 || __CUDA_ARCH__ == 900 || __CUDA_ARCH__ == 1000
__launch_bounds__(256, (algo == BITCOMP_DEFAULT_ALGO) ? 5 : 8)
#elif __CUDA_ARCH__ == 860 || __CUDA_ARCH__ == 890 || __CUDA_ARCH__ == 1200
__launch_bounds__(256, (algo == BITCOMP_DEFAULT_ALGO) ? 5 : 6)
#else
__launch_bounds__(256)
#endif
#else
__launch_bounds__(256)
#endif
  __global__ void batch_encoder_kernel(
    const void *const *__restrict__ in,
    void *const *__restrict__ out,
    const size_t *__restrict__ nbytes,
    size_t *__restrict__ outputSizes,
    T scalar_delta
  )
{
  static_assert(sizeof(void *) == sizeof(T *));
  const char *p_in = static_cast<const char *>(in[blockIdx.x]);
  static_assert(sizeof(void *) == sizeof(uint *));
  uint *p_out = static_cast<uint *>(out[blockIdx.x]); // Header for this batch
  uint64 blockBytes = nbytes[blockIdx.x]; // Uncompressed size for this batch
  uint64 blockOffset = 0; // No need to set the counter to zero before the batch kernel, using this as counter.
  int nblocks = (blockBytes + 8191) >> 13;
  T delta = T(0.0);
  if constexpr (compMode != BITCOMP_LOSSLESS)
  {
    delta = getDelta(scalar_delta);
  }
  // Encode all the 8KB blocks sequentially
  for (int iblock = 0; iblock < nblocks; iblock++)
  {
    if (algo == BITCOMP_DEFAULT_ALGO)
    {
      rle::encoder<T, typeId, compMode, ifmt, false>(p_in, p_out, nullptr, blockOffset, blockBytes, iblock, delta);
    }
    if (algo == BITCOMP_SPARSE_ALGO)
    {
      zbmap::encoder<T, typeId, compMode, ifmt, false>(p_in, p_out, nullptr, blockOffset, blockBytes, iblock, delta);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0)
  {
    if (nblocks == 0)
    {
      // Special case for zero-byte chunks, header not written by the encoder
      header::setFlags<typeId, compMode, algo, ifmt>(p_out);
      header::setUncompressedSize(p_out, 0);
    }
    // Write the total length
    uint64 *headerCounter = reinterpret_cast<uint64 *>(header::getCompressedLengthAddress(p_out));
    *headerCounter = blockOffset;
    // Write the total compressed size in bytes to the dedicated device-visible array
    outputSizes[blockIdx.x] = header::getTotalCompressedSize(nblocks, blockOffset);
  }
}

// *******************************************************************************************************************
// Generic encoder kernel launcher

template <typename T, bitcompDataType_t typeId, bitcompMode_t compMode, bitcompIntFormat_t ifmt>
bitcompResult_t launchEncoder(
  const void *in,
  void *out,
  void *externalCounter,
  bitcompAlgorithm_t algo,
  size_t nbytes,
  T delta,
  cudaStream_t stream
)
{
  // If no counter is provided, use the internal header counter. A device-memory counter is better for atomics.
  void *headerCounter = header::getCompressedLengthAddress(out);
  void *counter;
  if (externalCounter == NULL)
  {
    counter = headerCounter;
  }
  else
  {
    counter = externalCounter;
  }
  if (cudaMemsetAsync(counter, 0, sizeof(uint64), stream) != cudaSuccess)
  {
    return BITCOMP_CUDA_API_ERROR;
  }

  // Launching at least 1 block, even if the size is 0.
  const unsigned int blocks = max(1, nvcomp::cuda_dim_cast((nbytes + 8191) >> 13));
  const unsigned int threads = 256;

  // Zero out mantissa bits of delta
  if constexpr (compMode != BITCOMP_LOSSLESS)
  {
    delta = utilities::zeroMantissaBits(delta);
  }

  if (algo == BITCOMP_DEFAULT_ALGO)
  {
    encoder_kernel<BITCOMP_DEFAULT_ALGO, T, typeId, compMode, ifmt>
      <<<blocks, threads, 0, stream>>>(in, out, counter, nbytes, delta);
  }
  else if (algo == BITCOMP_SPARSE_ALGO)
  {
    encoder_kernel<BITCOMP_SPARSE_ALGO, T, typeId, compMode, ifmt>
      <<<blocks, threads, 0, stream>>>(in, out, counter, nbytes, delta);
  }
  if (cudaGetLastError() != cudaSuccess)
  {
    return BITCOMP_CUDA_KERNEL_LAUNCH_ERROR;
  }

  if (externalCounter != NULL)
  {
    if (cudaMemcpyAsync(headerCounter, externalCounter, sizeof(uint64), cudaMemcpyDefault, stream) != cudaSuccess)
    {
      return BITCOMP_CUDA_API_ERROR;
    }
  }
  return BITCOMP_SUCCESS;
}

// *******************************************************************************************************************
// Generic batch encoder kernel launcher

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
)
{
  if (nbatch == 0)
  {
    return nvcompSuccess;
  }

  const unsigned int blocks = nvcomp::cuda_dim_cast(nbatch);
  const unsigned int threads = 256;

  if (algo == BITCOMP_DEFAULT_ALGO)
  {
    batch_encoder_kernel<BITCOMP_DEFAULT_ALGO, T, typeId, compMode, ifmt>
      <<<blocks, threads, 0, stream>>>(in, out, nbytes, outputSizes, delta);
  }
  else if (algo == BITCOMP_SPARSE_ALGO)
  {
    batch_encoder_kernel<BITCOMP_SPARSE_ALGO, T, typeId, compMode, ifmt>
      <<<blocks, threads, 0, stream>>>(in, out, nbytes, outputSizes, delta);
  }
  return cudaGetLastError() == cudaSuccess ? nvcompSuccess : nvcompErrorCudaError;
}

// The LLIF calls this template from another translation unit, so emit every
// supported specialization here to provide linker-visible definitions.
#define BITCOMP_INSTANTIATE_BATCH_ENCODER(T, type_id, mode, int_format)                                                \
  template nvcompStatus_t launchBatchEncoder<T, type_id, mode, int_format>(                                            \
    const void *const *,                                                                                               \
    void *const *,                                                                                                     \
    bitcompAlgorithm_t,                                                                                                \
    const size_t *,                                                                                                    \
    size_t *,                                                                                                          \
    size_t,                                                                                                            \
    T,                                                                                                                 \
    cudaStream_t                                                                                                       \
  )

BITCOMP_INSTANTIATE_BATCH_ENCODER(unsigned char, BITCOMP_UNSIGNED_8BIT, BITCOMP_LOSSLESS, BITCOMP_DEFAULT_FORMAT);
BITCOMP_INSTANTIATE_BATCH_ENCODER(char, BITCOMP_SIGNED_8BIT, BITCOMP_LOSSLESS, BITCOMP_CUSTOM_INTEGER);
BITCOMP_INSTANTIATE_BATCH_ENCODER(unsigned short, BITCOMP_UNSIGNED_16BIT, BITCOMP_LOSSLESS, BITCOMP_DEFAULT_FORMAT);
BITCOMP_INSTANTIATE_BATCH_ENCODER(short, BITCOMP_SIGNED_16BIT, BITCOMP_LOSSLESS, BITCOMP_CUSTOM_INTEGER);
BITCOMP_INSTANTIATE_BATCH_ENCODER(uint, BITCOMP_UNSIGNED_32BIT, BITCOMP_LOSSLESS, BITCOMP_DEFAULT_FORMAT);
BITCOMP_INSTANTIATE_BATCH_ENCODER(int, BITCOMP_SIGNED_32BIT, BITCOMP_LOSSLESS, BITCOMP_CUSTOM_INTEGER);
BITCOMP_INSTANTIATE_BATCH_ENCODER(unsigned long long, BITCOMP_UNSIGNED_64BIT, BITCOMP_LOSSLESS, BITCOMP_DEFAULT_FORMAT);
BITCOMP_INSTANTIATE_BATCH_ENCODER(long long, BITCOMP_SIGNED_64BIT, BITCOMP_LOSSLESS, BITCOMP_CUSTOM_INTEGER);
BITCOMP_INSTANTIATE_BATCH_ENCODER(half, BITCOMP_FP16_DATA, BITCOMP_LOSSY_FP_TO_SIGNED, BITCOMP_CUSTOM_INTEGER);
BITCOMP_INSTANTIATE_BATCH_ENCODER(half, BITCOMP_FP16_DATA, BITCOMP_LOSSY_FP_TO_UNSIGNED, BITCOMP_DEFAULT_FORMAT);
BITCOMP_INSTANTIATE_BATCH_ENCODER(float, BITCOMP_FP32_DATA, BITCOMP_LOSSY_FP_TO_SIGNED, BITCOMP_CUSTOM_INTEGER);
BITCOMP_INSTANTIATE_BATCH_ENCODER(float, BITCOMP_FP32_DATA, BITCOMP_LOSSY_FP_TO_UNSIGNED, BITCOMP_DEFAULT_FORMAT);
BITCOMP_INSTANTIATE_BATCH_ENCODER(double, BITCOMP_FP64_DATA, BITCOMP_LOSSY_FP_TO_SIGNED, BITCOMP_CUSTOM_INTEGER);
BITCOMP_INSTANTIATE_BATCH_ENCODER(double, BITCOMP_FP64_DATA, BITCOMP_LOSSY_FP_TO_UNSIGNED, BITCOMP_DEFAULT_FORMAT);

#undef BITCOMP_INSTANTIATE_BATCH_ENCODER

// *******************************************************************************************************************
// Template for all Lossy Floating point functions
template <typename T, bitcompDataType_t typeId>
bitcompResult_t bitcompCompressLossy(
  bitcompHandle_t handle, // Bitcomp handle
  const void *in, // Uncompressed FP input
  void *out, // Compressed output
  const T delta
) // Quantization delta
{
  if (handle == nullptr || in == nullptr || out == nullptr)
  {
    return BITCOMP_INVALID_PARAMETER;
  }

  size_t nbytes = handle->uncompressedSize;
  void *counter = handle->deviceCounter;

  if (!utilities::validUncompressedBufferAlignment(in, handle->dataType) ||
      !utilities::validCompressedBufferAlignment(out))
  {
    return BITCOMP_INVALID_ALIGNMENT;
  }

  if (!utilities::valid_handle(handle))
  {
    return BITCOMP_INVALID_PARAMETER;
  }

  switch (handle->compMode)
  {
    case BITCOMP_LOSSY_FP_TO_SIGNED:
      return launchEncoder<T, typeId, BITCOMP_LOSSY_FP_TO_SIGNED, BITCOMP_CUSTOM_INTEGER>(
        in,
        out,
        counter,
        handle->algo,
        nbytes,
        delta,
        handle->stream
      );
    case BITCOMP_LOSSY_FP_TO_UNSIGNED:
      return launchEncoder<T, typeId, BITCOMP_LOSSY_FP_TO_UNSIGNED, BITCOMP_DEFAULT_FORMAT>(
        in,
        out,
        counter,
        handle->algo,
        nbytes,
        delta,
        handle->stream
      );
    default:
      return BITCOMP_INVALID_PARAMETER;
  }
}

} // namespace bitcomp

extern "C" {

using namespace bitcomp;

// *******************************************************************************************************************
bitcompResult_t bitcompCompressLossy_fp16(
  const bitcompHandle_t handle, // Bitcomp handle
  const half *input, // Uncompressed FP16 input
  void *output, // Compressed output
  half delta
) // Quantization delta
{
  return (bitcompCompressLossy<half, BITCOMP_FP16_DATA>(handle, input, output, delta));
}

// *******************************************************************************************************************
// bitcompCompressLossy_fp32: Compression of single precision floating point data
bitcompResult_t bitcompCompressLossy_fp32(
  const bitcompHandle_t handle, // Bitcomp handle
  const float *input, // Uncompressed FP32 input
  void *output, // Compressed output
  float delta
) // Quantization delta
{
  return (bitcompCompressLossy<float, BITCOMP_FP32_DATA>(handle, input, output, delta));
}
// *******************************************************************************************************************
// bitcompCompressLossy_fp64: Compression of double precision floating point data
bitcompResult_t bitcompCompressLossy_fp64(
  const bitcompHandle_t handle, // Bitcomp handle
  const double *input, // Uncompressed FP64 input
  void *output, // Compressed output
  double delta
) // Quantization delta
{
  return (bitcompCompressLossy<double, BITCOMP_FP64_DATA>(handle, input, output, delta));
}

// *******************************************************************************************************************
// bitcompCompressLossless: Lossless compression of integral data types
bitcompResult_t bitcompCompressLossless(
  const bitcompHandle_t handle, // Bitcomp handle
  const void *input, // Uncompressed FP64 input
  void *output
) // Compressed output
{
  if (!utilities::validUncompressedBufferAlignment(input, handle->dataType) ||
      !utilities::validCompressedBufferAlignment(output))
  {
    return BITCOMP_INVALID_ALIGNMENT;
  }

  if (!utilities::valid_handle(handle))
  {
    return BITCOMP_INVALID_PARAMETER;
  }

  size_t nbytes = handle->uncompressedSize;
  bitcompAlgorithm_t algo = handle->algo;
  cudaStream_t stream = handle->stream;
  void *counter = handle->deviceCounter;

  switch (handle->dataType)
  {
    case BITCOMP_UNSIGNED_8BIT:
      return launchEncoder<unsigned char, BITCOMP_UNSIGNED_8BIT, BITCOMP_LOSSLESS, BITCOMP_DEFAULT_FORMAT>(
        input,
        output,
        counter,
        algo,
        nbytes,
        (unsigned char)0,
        stream
      );
    case BITCOMP_SIGNED_8BIT:
      return launchEncoder<char, BITCOMP_SIGNED_8BIT, BITCOMP_LOSSLESS, BITCOMP_CUSTOM_INTEGER>(
        input,
        output,
        counter,
        algo,
        nbytes,
        '\0',
        stream
      );
    case BITCOMP_UNSIGNED_16BIT:
      return launchEncoder<unsigned short, BITCOMP_UNSIGNED_16BIT, BITCOMP_LOSSLESS, BITCOMP_DEFAULT_FORMAT>(
        input,
        output,
        counter,
        algo,
        nbytes,
        0,
        stream
      );
    case BITCOMP_SIGNED_16BIT:
      return launchEncoder<short, BITCOMP_SIGNED_16BIT, BITCOMP_LOSSLESS, BITCOMP_CUSTOM_INTEGER>(
        input,
        output,
        counter,
        algo,
        nbytes,
        0,
        stream
      );
    case BITCOMP_UNSIGNED_32BIT:
      return launchEncoder<uint, BITCOMP_UNSIGNED_32BIT, BITCOMP_LOSSLESS, BITCOMP_DEFAULT_FORMAT>(
        input,
        output,
        counter,
        algo,
        nbytes,
        0,
        stream
      );
    case BITCOMP_SIGNED_32BIT:
      return launchEncoder<int, BITCOMP_SIGNED_32BIT, BITCOMP_LOSSLESS, BITCOMP_CUSTOM_INTEGER>(
        input,
        output,
        counter,
        algo,
        nbytes,
        0,
        stream
      );
    case BITCOMP_UNSIGNED_64BIT:
      return launchEncoder<unsigned long long, BITCOMP_UNSIGNED_64BIT, BITCOMP_LOSSLESS, BITCOMP_DEFAULT_FORMAT>(
        input,
        output,
        counter,
        algo,
        nbytes,
        0,
        stream
      );
    case BITCOMP_SIGNED_64BIT:
      return launchEncoder<long long, BITCOMP_SIGNED_64BIT, BITCOMP_LOSSLESS, BITCOMP_CUSTOM_INTEGER>(
        input,
        output,
        counter,
        algo,
        nbytes,
        0,
        stream
      );
    default:
      return BITCOMP_INVALID_PARAMETER;
  }
}

// ******************************************************************************************************************
} // extern "C"
