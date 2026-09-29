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

// ************************************************************************************************
// Generic encoder for CPU

template <typename T, bitcompDataType_t typeId, bitcompMode_t compMode, bitcompIntFormat_t ifmt>
inline void
cpu_encoder(const void *__restrict__ in, void *__restrict__ out, bitcompAlgorithm_t algo, size_t nbytes, T delta)
{
  const int lbuf_rle = 8436; // Worst case scenario for RLE
  const int lbuf_zbm = 9348; // Worst case scenario for ZBMAP
  constexpr int lbuf = lbuf_rle > lbuf_zbm ? lbuf_rle : lbuf_zbm;
  constexpr int elemBytes = sizeof(T);
  // Note:
  // buf1 and buf2 are used by functions with AVX below
  // to speed up computation, hence the 32-byte alignment
  // requirement. We are keeping 16-byte alignment for potential
  // NEON vectorizations.
#ifdef __x86_64
  alignas(32) unsigned char buf1[lbuf];
  alignas(32) unsigned char buf2[lbuf];
#else
  alignas(16) unsigned char buf1[lbuf];
  alignas(16) unsigned char buf2[lbuf];
#endif

  unsigned long long outputOffset = 0;
  uint *out32 = (uint *)out;
  unsigned char *in8 = (unsigned char *)in;

  // Zero out mantissa bits of delta
  if constexpr (compMode != BITCOMP_LOSSLESS)
  {
    delta = utilities::zeroMantissaBits(delta);
  }

  // Initialize the header
  if (algo == BITCOMP_DEFAULT_ALGO)
  {
    header::setFlags<typeId, compMode, BITCOMP_DEFAULT_ALGO, ifmt>(out);
  }
  else if (algo == BITCOMP_SPARSE_ALGO)
  {
    header::setFlags<typeId, compMode, BITCOMP_SPARSE_ALGO, ifmt>(out);
  }
  header::setUncompressedSize(out, nbytes);
  header::setScalingDelta(out, delta);
  uint64 hdrlenw = header::computeHeaderLengthInWords(nbytes);

  // Loop on all the blocks of 8KB
  for (uint64 offset = 0; offset < nbytes; offset += NOMINAL_BLOCK_SIZE)
  {
    int iblock = nvcomp::narrow_cast<int>(offset / NOMINAL_BLOCK_SIZE);

    // Copy the data in buf1, pad with zeroes if needed
    int blockSize = NOMINAL_BLOCK_SIZE;
    if (offset + NOMINAL_BLOCK_SIZE <= nbytes)
    {
      memcpy(buf1, in8 + offset, NOMINAL_BLOCK_SIZE);
    }
    else
    {
      blockSize = (int)(nbytes - offset);
      memcpy(buf1, in8 + offset, blockSize);
      memset(buf1 + blockSize, 0, NOMINAL_BLOCK_SIZE - blockSize);
    }

    // Optional quantization to integer
    int overflow = 0;
    if constexpr (compMode != BITCOMP_LOSSLESS)
    {
      overflow = cpu_quantize<compMode, T>(buf1, delta);
    }

    if (!overflow)
    {
      // Optional custom integer format
      customizeInteger<elemBytes, ifmt>(buf1);

      // Bit-plan transpose buf1 -> buf2
      bitTransposeFwd<elemBytes>(buf1, buf2);

      int lcomp = 0;
      // RLE or ZBMAP encoding, lcomp is the compressed length in 32-bit words
      if (algo == BITCOMP_DEFAULT_ALGO)
      {
        lcomp = rle::encoder(buf2, buf1);
      }
      else if (algo == BITCOMP_SPARSE_ALGO)
      {
        lcomp = zbm::encoder(buf2, buf1);
      }

      // If compression worked, fill the header and copy the compressed data
      if (lcomp < 2048)
      {
        uint *buf = (uint *)buf1;
        for (int i = 0; i < lcomp; i++)
        {
          out32[outputOffset + hdrlenw + i] = buf[i];
        }
        header::setBlockInfo(out, iblock, outputOffset, lcomp);
        outputOffset += lcomp;
      }
      else
      {
        overflow = 1;
      }
    }

    if (overflow)
    {
      // Copy the original data (using fixed size of 2048 for uncompressed blocks, like the GPU code)
      header::setBlockInfoOverflow(out, iblock, outputOffset, 2048);
      memcpy(out32 + hdrlenw + outputOffset, in8 + offset, blockSize);
      // Write zeroes if blockSize != NOMINAL_BLOCK_SIZE, for a repeatable output
      if (blockSize != NOMINAL_BLOCK_SIZE)
      {
        assert(blockSize < NOMINAL_BLOCK_SIZE);
        memset((char *)(out32 + hdrlenw + outputOffset) + blockSize, 0, NOMINAL_BLOCK_SIZE - blockSize);
      }
      outputOffset += 2048;
    }

  } // End loop on all the blocks

  // Write the total length in the header
  *((unsigned long long *)header::getCompressedLengthAddress(out)) = outputOffset;
}

// ************************************************************************************************
// Template for all Lossy Floating point functions
template <typename T, bitcompDataType_t typeId>
bitcompResult_t bitcompHostCompressLossy(
  bitcompHandle_t handle, // Bitcomp handle
  const void *in, // Uncompressed FP16 input
  void *out, // Compressed output
  T delta
) // Quantization delta
{
  size_t nbytes = handle->uncompressedSize;

  if (!utilities::valid_handle(handle))
  {
    return BITCOMP_INVALID_PARAMETER;
  }

  // Sync the stream if it is non-NULL, before compressing
  if (handle->stream != 0)
  {
    cudaStreamSynchronize(handle->stream);
  }

  if (handle->compMode == BITCOMP_LOSSY_FP_TO_SIGNED)
  {
    cpu_encoder<T, typeId, BITCOMP_LOSSY_FP_TO_SIGNED, BITCOMP_CUSTOM_INTEGER>(in, out, handle->algo, nbytes, delta);
  }
  else if (handle->compMode == BITCOMP_LOSSY_FP_TO_UNSIGNED)
  {
    cpu_encoder<T, typeId, BITCOMP_LOSSY_FP_TO_UNSIGNED, BITCOMP_DEFAULT_FORMAT>(in, out, handle->algo, nbytes, delta);
  }
  return BITCOMP_SUCCESS;
}
} // namespace bitcomp

extern "C" {

using namespace bitcomp;

// ************************************************************************************************
// bitcompHostCompressLossy_fp16: Compression of half precision floating point data
bitcompResult_t bitcompHostCompressLossy_fp16(
  const bitcompHandle_t handle, // Bitcomp handle
  const half *input, // Uncompressed FP16 input
  void *output, // Compressed output
  half delta
) // Quantization delta
{
  return (bitcompHostCompressLossy<half, BITCOMP_FP16_DATA>(handle, input, output, delta));
}

// ************************************************************************************************
// bitcompHostCompressLossy_fp32: Compression of single precision floating point data
bitcompResult_t bitcompHostCompressLossy_fp32(
  const bitcompHandle_t handle, // Bitcomp handle
  const float *input, // Uncompressed FP32 input
  void *output, // Compressed output
  float delta
) // Quantization delta
{
  return (bitcompHostCompressLossy<float, BITCOMP_FP32_DATA>(handle, input, output, delta));
}

// ************************************************************************************************
// bitcompHostCompressLossy_fp64: Compression of double precision floating point data
bitcompResult_t bitcompHostCompressLossy_fp64(
  const bitcompHandle_t handle, // Bitcomp handle
  const double *input, // Uncompressed FP64 input
  void *output, // Compressed output
  double delta
) // Quantization delta
{
  return (bitcompHostCompressLossy<double, BITCOMP_FP64_DATA>(handle, input, output, delta));
}

// ************************************************************************************************
// bitcompHostCompressLossless: Lossless compression of integral data types
bitcompResult_t bitcompHostCompressLossless(
  const bitcompHandle_t handle, // Bitcomp handle
  const void *input, // Uncompressed FP64 input
  void *output
) // Compressed output
{
  size_t nbytes = handle->uncompressedSize;

  // Sync the stream if it is non-NULL, before compressing
  if (handle->stream != 0)
  {
    cudaStreamSynchronize(handle->stream);
  }

  if (!utilities::valid_handle(handle))
  {
    return BITCOMP_INVALID_PARAMETER;
  }

  switch (handle->dataType)
  {
    case BITCOMP_UNSIGNED_8BIT:
      cpu_encoder<unsigned char, BITCOMP_UNSIGNED_8BIT, BITCOMP_LOSSLESS, BITCOMP_DEFAULT_FORMAT>(
        input,
        output,
        handle->algo,
        nbytes,
        (unsigned char)0
      );
      break;
    case BITCOMP_SIGNED_8BIT:
      cpu_encoder<char, BITCOMP_SIGNED_8BIT, BITCOMP_LOSSLESS, BITCOMP_CUSTOM_INTEGER>(
        input,
        output,
        handle->algo,
        nbytes,
        '\0'
      );
      break;
    case BITCOMP_UNSIGNED_16BIT:
      cpu_encoder<unsigned short, BITCOMP_UNSIGNED_16BIT, BITCOMP_LOSSLESS, BITCOMP_DEFAULT_FORMAT>(
        input,
        output,
        handle->algo,
        nbytes,
        0
      );
      break;
    case BITCOMP_SIGNED_16BIT:
      cpu_encoder<short, BITCOMP_SIGNED_16BIT, BITCOMP_LOSSLESS, BITCOMP_CUSTOM_INTEGER>(
        input,
        output,
        handle->algo,
        nbytes,
        0
      );
      break;
    case BITCOMP_UNSIGNED_32BIT:
      cpu_encoder<uint, BITCOMP_UNSIGNED_32BIT, BITCOMP_LOSSLESS, BITCOMP_DEFAULT_FORMAT>(
        input,
        output,
        handle->algo,
        nbytes,
        0
      );
      break;
    case BITCOMP_SIGNED_32BIT:
      cpu_encoder<int, BITCOMP_SIGNED_32BIT, BITCOMP_LOSSLESS, BITCOMP_CUSTOM_INTEGER>(
        input,
        output,
        handle->algo,
        nbytes,
        0
      );
      break;
    case BITCOMP_UNSIGNED_64BIT:
      cpu_encoder<unsigned long long, BITCOMP_UNSIGNED_64BIT, BITCOMP_LOSSLESS, BITCOMP_DEFAULT_FORMAT>(
        input,
        output,
        handle->algo,
        nbytes,
        0
      );
      break;
    case BITCOMP_SIGNED_64BIT:
      cpu_encoder<long long, BITCOMP_SIGNED_64BIT, BITCOMP_LOSSLESS, BITCOMP_CUSTOM_INTEGER>(
        input,
        output,
        handle->algo,
        nbytes,
        0
      );
      break;
    // Other types should not be compressed with the lossless function
    default:
      return BITCOMP_INVALID_PARAMETER;
  }
  return BITCOMP_SUCCESS;
}
}
