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

#include <cuda.h>
#include <cuda/type_traits>
#include <cuda_runtime.h>

#include "bitcomp_private.h"
#include "cpu_quantize.h"
#include "cpu_rle.h"
#include "cpu_transpose.h"
#include "cpu_zbmap.h"
#include "nvcomp/utils.hpp"
#include "utilities.h"

#include <math.h>
#include <nvcomp/native/bitcomp.h>
#include <stdio.h>

namespace bitcomp
{

// *******************************************************************************************************************
// Generic decoder for CPU

template <typename T>
inline void cpu_decoder(
  const void *__restrict__ in,
  void *__restrict__ out,
  bitcompMode_t compMode,
  bitcompAlgorithm_t algo,
  bitcompIntFormat_t ifmt,
  size_t start,
  size_t length
)
{
  constexpr int elemBytes = sizeof(T);
  // Note:
  // buf1 and buf2 are used by functions with AVX below
  // to speed up computation, hence the 32-byte alignment
  // requirement. We are keeping 16-byte alignment for potential
  // NEON vectorizations.
#ifdef __x86_64
  alignas(32) unsigned char buf1[NOMINAL_BLOCK_SIZE];
  alignas(32) unsigned char buf2[NOMINAL_BLOCK_SIZE];
#else
  alignas(16) unsigned char buf1[NOMINAL_BLOCK_SIZE];
  alignas(16) unsigned char buf2[NOMINAL_BLOCK_SIZE];
#endif

  const uint *in32 = reinterpret_cast<const uint *>(in);
  unsigned char *out8 = reinterpret_cast<unsigned char *>(out);

  T delta = header::getScalingDelta<T>(in);
  uint64 hdrlenw = header::getHeaderLengthInWords(in);

  // Starting decompression on an 8KB boundary
  uint64 blockAlignedStart = start & ~8191L;

  // Loop on all the uncompressed blocks of 8KB
  for (uint64 offset = blockAlignedStart; offset < start + length; offset += NOMINAL_BLOCK_SIZE)
  {
    uint64 blockOffset;
    uint lcompw;
    bool overflow; // Get the offset of the compressed data for the current block
    int iblock = nvcomp::narrow_cast<int>(offset / NOMINAL_BLOCK_SIZE);
    header::getBlockInfo(in, iblock, blockOffset, lcompw, overflow);

    const unsigned char *compPtr = reinterpret_cast<const unsigned char *>(in32 + hdrlenw + blockOffset);

    // Intersection of output with current block
    uint64 blockStart = 0;
    if (offset < (uint64)start)
    {
      blockStart = start - offset;
    }
    uint64 blockEnd = NOMINAL_BLOCK_SIZE;
    if (offset + NOMINAL_BLOCK_SIZE > start + length)
    {
      blockEnd = start + length - offset;
    }
    uint64 blockSize = blockEnd - blockStart;

    // If the block was not compressed, restore the original data.
    if (overflow)
    {
      memcpy(out8, compPtr + blockStart, blockSize);
    }
    else
    {
      // Decode the data in -> buf1
      if (algo == BITCOMP_DEFAULT_ALGO)
      {
        rle::decoder(compPtr, lcompw, buf1);
      }
      else if (algo == BITCOMP_SPARSE_ALGO)
      {
        zbm::decoder(compPtr, buf1);
      }

      // Backward bit-plan transpose buf1 -> buf2
      bitTransposeBwd<elemBytes>(buf1, buf2);

      // (Optional) Go back to 2's complement for signed types, in-place (buf2)
      if (ifmt == BITCOMP_CUSTOM_INTEGER)
      {
        customizeIntegerBwd<elemBytes, BITCOMP_CUSTOM_INTEGER>(buf2);
      }

      // (optional) Backward quantize, in-place (buf2) - only for floating-point types
      if constexpr (cuda::is_floating_point_v<T>)
      {
        const bool pow_two_delta = utilities::zeroMantissaBits(delta) == delta;
        if (compMode == BITCOMP_LOSSY_FP_TO_SIGNED)
        {
          if (pow_two_delta)
          {
            quantizeBwd<T, BITCOMP_LOSSY_FP_TO_SIGNED, true>(buf2, delta);
          }
          else
          {
            quantizeBwd<T, BITCOMP_LOSSY_FP_TO_SIGNED, false>(buf2, delta);
          }
        }
        else if (compMode == BITCOMP_LOSSY_FP_TO_UNSIGNED)
        {
          if (pow_two_delta)
          {
            quantizeBwd<T, BITCOMP_LOSSY_FP_TO_UNSIGNED, true>(buf2, delta);
          }
          else
          {
            quantizeBwd<T, BITCOMP_LOSSY_FP_TO_UNSIGNED, false>(buf2, delta);
          }
        }
      }

      // Copy result buf2 -> final output
      memcpy(out8, buf2 + blockStart, blockSize);
    }

    // Move output pointer for the next block
    out8 += blockSize;

  } // End loop on all the blocks
}

} // namespace bitcomp

extern "C" {

using namespace bitcomp;

// CPU decoder, using the parameters stored in the handle to figure out how to call the launcher
bitcompResult_t bitcompHostUncompress(
  const bitcompHandle_t handle, // Bitcomp handle
  const void *input, // Compressed input
  void *output
) // Uncompressed output
{
  // Sync the stream if it is non-NULL, before even looking at the compressed data
  if (handle->stream != 0)
  {
    cudaStreamSynchronize(handle->stream);
  }

  if (!utilities::valid_handle(handle) ||
      !header::hasValidFlags(input, handle->dataType, handle->compMode, handle->algo, handle->ifmt))
  {
    return BITCOMP_INVALID_PARAMETER;
  }

  uint64 nbytes = handle->uncompressedSize;

  // Launch the decoder kernel based on data type
  // Exact type (signed or unsigned) does not matter
  switch (handle->dataType)
  {
    case BITCOMP_UNSIGNED_8BIT:
    case BITCOMP_SIGNED_8BIT:
      cpu_decoder<char>(input, output, handle->compMode, handle->algo, handle->ifmt, 0, nbytes);
      break;

    case BITCOMP_UNSIGNED_16BIT:
    case BITCOMP_SIGNED_16BIT:
      cpu_decoder<short>(input, output, handle->compMode, handle->algo, handle->ifmt, 0, nbytes);
      break;

    case BITCOMP_UNSIGNED_32BIT:
    case BITCOMP_SIGNED_32BIT:
      cpu_decoder<int>(input, output, handle->compMode, handle->algo, handle->ifmt, 0, nbytes);
      break;

    case BITCOMP_UNSIGNED_64BIT:
    case BITCOMP_SIGNED_64BIT:
      cpu_decoder<long long>(input, output, handle->compMode, handle->algo, handle->ifmt, 0, nbytes);
      break;

    case BITCOMP_FP16_DATA:
      cpu_decoder<half>(input, output, handle->compMode, handle->algo, handle->ifmt, 0, nbytes);
      break;

    case BITCOMP_FP32_DATA:
      cpu_decoder<float>(input, output, handle->compMode, handle->algo, handle->ifmt, 0, nbytes);
      break;

    case BITCOMP_FP64_DATA:
      cpu_decoder<double>(input, output, handle->compMode, handle->algo, handle->ifmt, 0, nbytes);
      break;

    default:
      return BITCOMP_INVALID_PARAMETER;
  }

  return BITCOMP_SUCCESS;
}

//bitcompHostPartialUncompress: Partial decompression of compressed data on CPU
bitcompResult_t bitcompHostPartialUncompress(
  const bitcompHandle_t handle, // Bitcomp handle
  const void *input, // Compressed input
  void *output, // Uncompressed output
  size_t start, // Start offset in bytes of the uncompressed data
  size_t length
) // Length in bytes of the partial decompression
{
  // Sync the stream if it is non-NULL, before even looking at the compressed data
  if (handle->stream != 0)
  {
    cudaStreamSynchronize(handle->stream);
  }

  if (!utilities::valid_handle(handle) ||
      !header::hasValidFlags(input, handle->dataType, handle->compMode, handle->algo, handle->ifmt))
  {
    return BITCOMP_INVALID_PARAMETER;
  }

  // Make sure the offset and length are OK
  uint64 nbytes = handle->uncompressedSize;
  if (start + length > nbytes)
  {
    return BITCOMP_INVALID_PARAMETER;
  }

  // Launch the decoder kernel based on data type
  // Exact type (signed or unsigned) does not matter
  switch (handle->dataType)
  {
    case BITCOMP_UNSIGNED_8BIT:
    case BITCOMP_SIGNED_8BIT:
      cpu_decoder<char>(input, output, handle->compMode, handle->algo, handle->ifmt, start, length);
      break;

    case BITCOMP_UNSIGNED_16BIT:
    case BITCOMP_SIGNED_16BIT:
      cpu_decoder<short>(input, output, handle->compMode, handle->algo, handle->ifmt, start, length);
      break;

    case BITCOMP_UNSIGNED_32BIT:
    case BITCOMP_SIGNED_32BIT:
      cpu_decoder<int>(input, output, handle->compMode, handle->algo, handle->ifmt, start, length);
      break;

    case BITCOMP_UNSIGNED_64BIT:
    case BITCOMP_SIGNED_64BIT:
      cpu_decoder<long long>(input, output, handle->compMode, handle->algo, handle->ifmt, start, length);
      break;

    case BITCOMP_FP16_DATA:
      cpu_decoder<half>(input, output, handle->compMode, handle->algo, handle->ifmt, start, length);
      break;

    case BITCOMP_FP32_DATA:
      cpu_decoder<float>(input, output, handle->compMode, handle->algo, handle->ifmt, start, length);
      break;

    case BITCOMP_FP64_DATA:
      cpu_decoder<double>(input, output, handle->compMode, handle->algo, handle->ifmt, start, length);
      break;

    default:
      return BITCOMP_INVALID_PARAMETER;
  }

  return BITCOMP_SUCCESS;
}
}
