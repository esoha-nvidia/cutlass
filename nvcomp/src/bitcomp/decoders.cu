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
#include "rle.h"
#include "utilities.h"
#include "zbmap.h"

#include <nvcomp/native/bitcomp.h>
#include <stdio.h>

namespace bitcomp
{

// *******************************************************************************************************************
// Generic decoder kernel

template <bitcompAlgorithm_t algo, typename T, bitcompMode_t compMode, bitcompIntFormat_t ifmt>
#if defined(__CUDA_ARCH__)
#if __CUDA_ARCH__ == 860 || __CUDA_ARCH__ == 890 || __CUDA_ARCH__ == 1200
__launch_bounds__(256, 6)
#elif __CUDA_ARCH__ == 900
__launch_bounds__(256, 8)
#else
__launch_bounds__(256)
#endif
#else
__launch_bounds__(256)
#endif
  __global__
  void decoder_kernel(const void *__restrict__ in, void *__restrict__ out, int firstBlock, uint64 start, uint64 end)
{

  // Read the header
  T delta;
  if (compMode != BITCOMP_LOSSLESS)
  {
    delta = header::getScalingDelta<T>(in);
  }
  uint64 hdrlenw = header::getHeaderLengthInWords(in);
  // uint64 nbytes = header::getUncompressedSize(in);
  uint64 blockOffset;
  uint lcompw;
  bool overflow;
  header::getBlockInfo(reinterpret_cast<const void *>(in), (firstBlock + blockIdx.x), blockOffset, lcompw, overflow);

  // Offsets assuming full block decompression. Will be adjusted if doing a partial decomp.
  uint64 outputOffset = (firstBlock + blockIdx.x) * 8192ULL;
  const uint *p_in = reinterpret_cast<const uint *>(in) + hdrlenw + blockOffset;
  char *p_out = reinterpret_cast<char *>(out) + outputOffset;

  // Partial blocks
  int blockStart = start < outputOffset ? 0 : static_cast<int>(start - outputOffset);
  int blockEnd = (int)(min(8192ULL, end - outputOffset));
  int blockBytes = blockEnd - blockStart;

  // 16-byte alignment based on pointers and number of bytes
  bool aligned = ((((uintptr_t)p_out + blockStart) & 0xf) | (blockStart & 0xf) | (blockBytes & 0xf)) == 0;

  // If the block was not compressed, restore data as is and return.
  if (overflow)
  {
    loadStore::restoreIncompressibleBlock<T>(p_in, p_out, blockStart, blockEnd, aligned);
    return;
  }

  if (algo == BITCOMP_DEFAULT_ALGO)
  {
    rle::decoder<T, compMode, ifmt>(p_in, p_out, blockStart, blockEnd, aligned, delta, lcompw);
  }
  if (algo == BITCOMP_SPARSE_ALGO)
  {
    zbmap::decoder<T, compMode, ifmt>(p_in, p_out, blockStart, blockEnd, aligned, delta);
  }
}

__device__ void setChunkStatus(nvcompStatus_t *statuses, int chunk_id, nvcompStatus_t status)
{
  assert(statuses != nullptr);
  if (threadIdx.x == 0)
  {
    statuses[chunk_id] = status;
  }
}

// *******************************************************************************************************************
// Generic batch decoder kernel. Does not support partial decompression
// Each CTA works on one batch and decompresses all the 8KB blocks sequentially

template <bitcompAlgorithm_t ALGORITHM, typename T, bitcompMode_t COMP_MODE, bitcompIntFormat_t INT_FORMAT>
#if defined(__CUDA_ARCH__)
#if __CUDA_ARCH__ == 750
// Note: 750 is the Geforce line of Turing that only allowed a maximum of 1024 resident threads per SM
__launch_bounds__(256, 4)
#elif __CUDA_ARCH__ == 800 || __CUDA_ARCH__ == 900 || __CUDA_ARCH__ == 1000
__launch_bounds__(256, 8)
#else
__launch_bounds__(256, 6)
#endif
#else
__launch_bounds__(256, 6)
#endif
  __global__ void batch_decoder_kernel(
    const void *const *__restrict__ input,
    void *const *__restrict__ output,
    const size_t *output_buffer_sizes,
    size_t *uncompressed_sizes,
    nvcompStatus_t *statuses
  )
{
  const int chunk_id = static_cast<int>(blockIdx.x);
  const void *const p_in = input[chunk_id];
  char *p_out = reinterpret_cast<char *const *>(output)[chunk_id];
  assert(header::getAlgorithm(p_in) == ALGORITHM);
  assert(header::getCompMode(p_in) == COMP_MODE);
  assert(header::getIntFormat(p_in) == INT_FORMAT);

  T delta;
  if constexpr (COMP_MODE != BITCOMP_LOSSLESS)
  {
    delta = header::getScalingDelta<T>(p_in);
  }
  const uint64 hdrlenw = header::getHeaderLengthInWords(p_in);
  const uint64 uncompressed_bytes = header::getUncompressedSize(p_in);

  if (!header::hasValidMagicNumber(p_in))
  {
    setChunkStatus(statuses, chunk_id, nvcompErrorCannotDecompress);
    return;
  }
  if (output_buffer_sizes[chunk_id] < uncompressed_bytes)
  {
    setChunkStatus(statuses, chunk_id, nvcompErrorOutputBufferTooSmall);
    return;
  }

  int nblocks = nvcomp::roundUpDiv(uncompressed_bytes, NOMINAL_BLOCK_SIZE);

  // Loop on all the 8KB blocks sequentially
  for (int iblock = 0; iblock < nblocks; iblock++)
  {
    // Query the block info (offset of input, compressed length, overflow status)
    uint64 blockOffset;
    uint lcompw;
    bool overflow;
    header::getBlockInfo(p_in, iblock, blockOffset, lcompw, overflow);
    const uint *p_block_in = reinterpret_cast<const uint *>(p_in) + hdrlenw + blockOffset;
    const uint64 remaining_bytes = uncompressed_bytes - iblock * static_cast<uint64>(NOMINAL_BLOCK_SIZE);
    const int blockBytes =
      static_cast<int>(remaining_bytes < NOMINAL_BLOCK_SIZE ? remaining_bytes : NOMINAL_BLOCK_SIZE);

    // 16-byte alignment based on pointers and number of bytes
    bool aligned = (((uintptr_t)p_out & 0xf) | (blockBytes & 0xf)) == 0;

    // If the block was not compressed, restore data as is, otherwise decompress
    if (overflow)
    {
      loadStore::restoreIncompressibleBlock<T>(p_block_in, p_out, 0, blockBytes, aligned);
    }
    else
    {
      if (ALGORITHM == BITCOMP_DEFAULT_ALGO)
      {
        rle::decoder<T, COMP_MODE, INT_FORMAT>(p_block_in, p_out, 0, blockBytes, aligned, delta, lcompw);
      }
      if (ALGORITHM == BITCOMP_SPARSE_ALGO)
      {
        zbmap::decoder<T, COMP_MODE, INT_FORMAT>(p_block_in, p_out, 0, blockBytes, aligned, delta);
      }
    }
    p_out += NOMINAL_BLOCK_SIZE;
    __syncthreads();
  }
  if (threadIdx.x == 0)
  {
    uncompressed_sizes[chunk_id] = uncompressed_bytes;
  }
  setChunkStatus(statuses, chunk_id, nvcompSuccess);
}

// *******************************************************************************************************************
// Template decoder kernel launcher

template <typename T, bitcompMode_t mode, bitcompIntFormat_t ifmt>
bitcompResult_t
launchDecoder(const void *in, void *out, bitcompAlgorithm_t algo, size_t startOffset, size_t nbytes, cudaStream_t stream)
{
  if (nbytes == 0)
  {
    return BITCOMP_SUCCESS;
  }
  uint64 start = startOffset;
  uint64 end = start + nbytes;
  const unsigned int firstBlock = nvcomp::cuda_dim_cast(start >> 13);
  const unsigned int lastBlock = nvcomp::cuda_dim_cast((end - 1) >> 13);
  const unsigned int blocks = lastBlock - firstBlock + 1;
  const unsigned int threads = 256;

  if ((uintptr_t)in % sizeof(T) != 0 || (uintptr_t)out % sizeof(T) != 0 || start % sizeof(T) != 0)
  {
    return BITCOMP_INVALID_ALIGNMENT;
  }
  else if (nbytes % sizeof(T) != 0)
  {
    return BITCOMP_INVALID_INPUT_LENGTH;
  }

  if (algo == BITCOMP_DEFAULT_ALGO)
  {
    decoder_kernel<BITCOMP_DEFAULT_ALGO, T, mode, ifmt>
      <<<blocks, threads, 0, stream>>>(in, out, firstBlock, start, end);
  }
  else if (algo == BITCOMP_SPARSE_ALGO)
  {
    decoder_kernel<BITCOMP_SPARSE_ALGO, T, mode, ifmt><<<blocks, threads, 0, stream>>>(in, out, firstBlock, start, end);
  }
  return cudaGetLastError() == cudaSuccess ? BITCOMP_SUCCESS : BITCOMP_CUDA_KERNEL_LAUNCH_ERROR;
}

// *******************************************************************************************************************
// Template batch decoder kernel launcher

nvcompStatus_t launchBatchDecoder(
  const void *const *input,
  void *const *output,
  const size_t *output_buffer_sizes,
  nvcompStatus_t *statuses,
  size_t *uncompressed_sizes,
  const batchCompInfo_t &comp_info,
  size_t batch_size,
  cudaStream_t stream
)
{
  if (batch_size == 0)
  {
    return nvcompSuccess;
  }

  bitcompContext context{};
  context.batches = batch_size;
  context.dataType = comp_info.dataType;
  context.algo = comp_info.algo;
  context.compMode = comp_info.mode;
  context.ifmt = comp_info.ifmt;
  if (!utilities::valid_handle<true>(&context))
  {
    return nvcompErrorCannotDecompress;
  }

  const unsigned int blocks = nvcomp::cuda_dim_cast(batch_size);
  constexpr unsigned int THREADS = 256;

#define BITCOMP_LAUNCH_BATCH_DECODER(ALGORITHM, T, COMP_MODE, INT_FORMAT)                                              \
  do                                                                                                                   \
  {                                                                                                                    \
    batch_decoder_kernel<ALGORITHM, T, COMP_MODE, INT_FORMAT>                                                          \
      <<<blocks, THREADS, 0, stream>>>(input, output, output_buffer_sizes, uncompressed_sizes, statuses);              \
  } while (0)

#define BITCOMP_DISPATCH_BATCH_DECODER(ALGORITHM)                                                                      \
  do                                                                                                                   \
  {                                                                                                                    \
    switch (comp_info.dataType)                                                                                        \
    {                                                                                                                  \
      case BITCOMP_UNSIGNED_8BIT:                                                                                      \
        BITCOMP_LAUNCH_BATCH_DECODER(ALGORITHM, uint8_t, BITCOMP_LOSSLESS, BITCOMP_DEFAULT_FORMAT);                    \
        break;                                                                                                         \
      case BITCOMP_SIGNED_8BIT:                                                                                        \
        BITCOMP_LAUNCH_BATCH_DECODER(ALGORITHM, int8_t, BITCOMP_LOSSLESS, BITCOMP_CUSTOM_INTEGER);                     \
        break;                                                                                                         \
      case BITCOMP_UNSIGNED_16BIT:                                                                                     \
        BITCOMP_LAUNCH_BATCH_DECODER(ALGORITHM, uint16_t, BITCOMP_LOSSLESS, BITCOMP_DEFAULT_FORMAT);                   \
        break;                                                                                                         \
      case BITCOMP_SIGNED_16BIT:                                                                                       \
        BITCOMP_LAUNCH_BATCH_DECODER(ALGORITHM, int16_t, BITCOMP_LOSSLESS, BITCOMP_CUSTOM_INTEGER);                    \
        break;                                                                                                         \
      case BITCOMP_UNSIGNED_32BIT:                                                                                     \
        BITCOMP_LAUNCH_BATCH_DECODER(ALGORITHM, uint32_t, BITCOMP_LOSSLESS, BITCOMP_DEFAULT_FORMAT);                   \
        break;                                                                                                         \
      case BITCOMP_SIGNED_32BIT:                                                                                       \
        BITCOMP_LAUNCH_BATCH_DECODER(ALGORITHM, int32_t, BITCOMP_LOSSLESS, BITCOMP_CUSTOM_INTEGER);                    \
        break;                                                                                                         \
      case BITCOMP_UNSIGNED_64BIT:                                                                                     \
        BITCOMP_LAUNCH_BATCH_DECODER(ALGORITHM, uint64_t, BITCOMP_LOSSLESS, BITCOMP_DEFAULT_FORMAT);                   \
        break;                                                                                                         \
      case BITCOMP_SIGNED_64BIT:                                                                                       \
        BITCOMP_LAUNCH_BATCH_DECODER(ALGORITHM, int64_t, BITCOMP_LOSSLESS, BITCOMP_CUSTOM_INTEGER);                    \
        break;                                                                                                         \
      case BITCOMP_FP16_DATA:                                                                                          \
        if (comp_info.mode == BITCOMP_LOSSY_FP_TO_SIGNED)                                                              \
        {                                                                                                              \
          BITCOMP_LAUNCH_BATCH_DECODER(ALGORITHM, half, BITCOMP_LOSSY_FP_TO_SIGNED, BITCOMP_CUSTOM_INTEGER);           \
        }                                                                                                              \
        else if (comp_info.mode == BITCOMP_LOSSY_FP_TO_UNSIGNED)                                                       \
        {                                                                                                              \
          BITCOMP_LAUNCH_BATCH_DECODER(ALGORITHM, half, BITCOMP_LOSSY_FP_TO_UNSIGNED, BITCOMP_DEFAULT_FORMAT);         \
        }                                                                                                              \
        break;                                                                                                         \
      case BITCOMP_FP32_DATA:                                                                                          \
        if (comp_info.mode == BITCOMP_LOSSY_FP_TO_SIGNED)                                                              \
        {                                                                                                              \
          BITCOMP_LAUNCH_BATCH_DECODER(ALGORITHM, float, BITCOMP_LOSSY_FP_TO_SIGNED, BITCOMP_CUSTOM_INTEGER);          \
        }                                                                                                              \
        else if (comp_info.mode == BITCOMP_LOSSY_FP_TO_UNSIGNED)                                                       \
        {                                                                                                              \
          BITCOMP_LAUNCH_BATCH_DECODER(ALGORITHM, float, BITCOMP_LOSSY_FP_TO_UNSIGNED, BITCOMP_DEFAULT_FORMAT);        \
        }                                                                                                              \
        break;                                                                                                         \
      case BITCOMP_FP64_DATA:                                                                                          \
        if (comp_info.mode == BITCOMP_LOSSY_FP_TO_SIGNED)                                                              \
        {                                                                                                              \
          BITCOMP_LAUNCH_BATCH_DECODER(ALGORITHM, double, BITCOMP_LOSSY_FP_TO_SIGNED, BITCOMP_CUSTOM_INTEGER);         \
        }                                                                                                              \
        else if (comp_info.mode == BITCOMP_LOSSY_FP_TO_UNSIGNED)                                                       \
        {                                                                                                              \
          BITCOMP_LAUNCH_BATCH_DECODER(ALGORITHM, double, BITCOMP_LOSSY_FP_TO_UNSIGNED, BITCOMP_DEFAULT_FORMAT);       \
        }                                                                                                              \
        break;                                                                                                         \
      default:                                                                                                         \
        return nvcompErrorCannotDecompress;                                                                            \
    }                                                                                                                  \
  } while (0)

  switch (comp_info.algo)
  {
    case BITCOMP_DEFAULT_ALGO:
      BITCOMP_DISPATCH_BATCH_DECODER(BITCOMP_DEFAULT_ALGO);
      break;
    case BITCOMP_SPARSE_ALGO:
      BITCOMP_DISPATCH_BATCH_DECODER(BITCOMP_SPARSE_ALGO);
      break;
    default:
      return nvcompErrorCannotDecompress;
  }

#undef BITCOMP_DISPATCH_BATCH_DECODER
#undef BITCOMP_LAUNCH_BATCH_DECODER

  return cudaGetLastError() == cudaSuccess ? nvcompSuccess : nvcompErrorCudaError;
}
} // namespace bitcomp

// *******************************************************************************************************************

extern "C" {

using namespace bitcomp;

// GPU decoder, using the parameters stored in the handle to figure out how to call the launcher,
// supporting a partial decompression.
bitcompResult_t bitcompPartialUncompress(
  const bitcompHandle_t handle, // Bitcomp handle
  const void *input, // Compressed input
  void *output, // Uncompressed output
  size_t start,
  size_t length
)
{
  if (!utilities::validCompressedBufferAlignment(input) ||
      !utilities::validUncompressedBufferAlignment(output, handle->dataType))
  {
    return BITCOMP_INVALID_ALIGNMENT;
  }

  size_t nbytes = handle->uncompressedSize;
  if (start + length < nbytes || handle->batches > 0)
  {
    return BITCOMP_INVALID_PARAMETER;
  }

  bitcompAlgorithm_t algo = handle->algo;
  cudaStream_t stream = handle->stream;

  if (!utilities::valid_handle(handle))
  {
    return BITCOMP_INVALID_PARAMETER;
  }

  // Launch the decoder kernel based on handle information
  switch (handle->dataType)
  {
    case BITCOMP_UNSIGNED_8BIT:
      return launchDecoder<unsigned char, BITCOMP_LOSSLESS, BITCOMP_DEFAULT_FORMAT>(
        input,
        output,
        algo,
        start,
        length,
        stream
      );
    case BITCOMP_SIGNED_8BIT:
      return launchDecoder<char, BITCOMP_LOSSLESS, BITCOMP_CUSTOM_INTEGER>(input, output, algo, start, length, stream);
    case BITCOMP_UNSIGNED_16BIT:
      return launchDecoder<unsigned short, BITCOMP_LOSSLESS, BITCOMP_DEFAULT_FORMAT>(
        input,
        output,
        algo,
        start,
        length,
        stream
      );
    case BITCOMP_SIGNED_16BIT:
      return launchDecoder<short, BITCOMP_LOSSLESS, BITCOMP_CUSTOM_INTEGER>(input, output, algo, start, length, stream);
    case BITCOMP_UNSIGNED_32BIT:
      return launchDecoder<uint, BITCOMP_LOSSLESS, BITCOMP_DEFAULT_FORMAT>(input, output, algo, start, length, stream);
    case BITCOMP_SIGNED_32BIT:
      return launchDecoder<int, BITCOMP_LOSSLESS, BITCOMP_CUSTOM_INTEGER>(input, output, algo, start, length, stream);
    case BITCOMP_UNSIGNED_64BIT:
      return launchDecoder<unsigned long long, BITCOMP_LOSSLESS, BITCOMP_DEFAULT_FORMAT>(
        input,
        output,
        algo,
        start,
        length,
        stream
      );
    case BITCOMP_SIGNED_64BIT:
      return launchDecoder<long long, BITCOMP_LOSSLESS, BITCOMP_CUSTOM_INTEGER>(
        input,
        output,
        algo,
        start,
        length,
        stream
      );
    case BITCOMP_FP16_DATA:
      if (handle->compMode == BITCOMP_LOSSY_FP_TO_SIGNED)
      {
        return launchDecoder<half, BITCOMP_LOSSY_FP_TO_SIGNED, BITCOMP_CUSTOM_INTEGER>(
          input,
          output,
          algo,
          start,
          length,
          stream
        );
      }
      else if (handle->compMode == BITCOMP_LOSSY_FP_TO_UNSIGNED)
      {
        return launchDecoder<half, BITCOMP_LOSSY_FP_TO_UNSIGNED, BITCOMP_DEFAULT_FORMAT>(
          input,
          output,
          algo,
          start,
          length,
          stream
        );
      }
      break;
    case BITCOMP_FP32_DATA:
      if (handle->compMode == BITCOMP_LOSSY_FP_TO_SIGNED)
      {
        return launchDecoder<float, BITCOMP_LOSSY_FP_TO_SIGNED, BITCOMP_CUSTOM_INTEGER>(
          input,
          output,
          algo,
          start,
          length,
          stream
        );
      }
      else if (handle->compMode == BITCOMP_LOSSY_FP_TO_UNSIGNED)
      {
        return launchDecoder<float, BITCOMP_LOSSY_FP_TO_UNSIGNED, BITCOMP_DEFAULT_FORMAT>(
          input,
          output,
          algo,
          start,
          length,
          stream
        );
      }
      break;
    case BITCOMP_FP64_DATA:
      if (handle->compMode == BITCOMP_LOSSY_FP_TO_SIGNED)
      {
        return launchDecoder<double, BITCOMP_LOSSY_FP_TO_SIGNED, BITCOMP_CUSTOM_INTEGER>(
          input,
          output,
          algo,
          start,
          length,
          stream
        );
      }
      else if (handle->compMode == BITCOMP_LOSSY_FP_TO_UNSIGNED)
      {
        return launchDecoder<double, BITCOMP_LOSSY_FP_TO_UNSIGNED, BITCOMP_DEFAULT_FORMAT>(
          input,
          output,
          algo,
          start,
          length,
          stream
        );
      }
      break;
  }
  return BITCOMP_INVALID_PARAMETER;
}

// Decompression of the whole compressed data
bitcompResult_t bitcompUncompress(
  const bitcompHandle_t handle, // Bitcomp handle
  const void *input, // Compressed input
  void *output
) // Uncompressed output
{
  size_t nbytes = handle->uncompressedSize;
  return bitcompPartialUncompress(handle, input, output, 0, nbytes);
}
}
