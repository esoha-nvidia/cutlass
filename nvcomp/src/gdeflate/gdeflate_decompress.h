/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include <cuda.h>

#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

#include "bitreader.h"
#include "common.h"
#include "gdeflate.h"
#include "gdeflate_constants.h"
#include "huffman.h"

#include <stdio.h>

// #define IGNORE_COPIES                   // Turn off copies to measure only decode and literal performance
// #define VECTORIZED_DEPENDENT_COPIES     // Output dependent copy data in vectorized form

// #define GDEFLATE_LOG_TRACE              // Collect decode trace and write to gdeflate_trace.dat

#define BRANCHLESS_DECODING

// #define SPECIALIZED_LAUNCH_CONFIG_INSTANTIATIONS // Instantiate template specialized kernels for all possible launch configs

namespace gdeflate
{

// Each warp needs 1536 bytes of smem (1.5KB)
static constexpr const int MAX_WARPS_PER_CTA = 32;
static const int MAX_COPIES = 25600;

struct gdeflate_trace
{
  bool isCopy;
  uint16_t length;
  uint16_t symbol;
  uint32_t distance;
  int position;
  bool active;
  uint16_t output_bytes;
};

// Read code length code lengths
template <typename Treader>
__device__ void read_lencodes(uint16_t *lencode, Treader &br, unsigned int hclen, bool &corrupted)
{
  assert(hclen > 0 && hclen <= gdeflate_alphabet_size);

  bool active = threadIdx.x < hclen;
  uint16_t len = (uint16_t)br.read(3, active);
  // Order in which code length code length codes are in the stream
  if (threadIdx.x < gdeflate_alphabet_size)
  {
    lencode[map[threadIdx.x]] = len;
  }
  corrupted |= br.is_corrupted();
}

// Read and decode code lengths
template <typename Treader>
__device__ void unpack_codelens(
  uint16_t *codelen,
  unsigned int count,
  Treader &br,
  const uint16_t *lencode,
  uint16_t *counts,
  uint16_t *symbols,
  uint16_t *offset,
  uint32_t *basecode,
  bool &corrupted
)
{
  // Note: in testing this never happens
  if (corrupted)
  {
    return;
  }

  // Create the code length decoder
  // From RFC, alphabet size of 19, max code length of 7
  warp_decoder<gdeflate_alphabet_size, gdeflate_codelen_max_codelen> dec;
  dec.init(gdeflate_alphabet_size, lencode, counts, symbols, offset, basecode);

  // Fill with zeros initially to avoid processing codes 17 and 18
  for (unsigned int i = threadIdx.x; i < gdeflate_total_symbols_smem; i += WARP_SIZE_U)
  {
    codelen[i] = 0;
  }

  unsigned int outpos = 0;
  while (count)
  {
    uint32_t bits = br.peek(gdeflate_codelen_max_codelen + gdeflate_codelen_max_codelen, true);

    uint32_t code = __brev(bits);
    uint16_t len = dec.len4code(code);
    assert(len <= gdeflate_codelen_max_codelen);
    uint16_t sym = dec.sym4code(code, len);
    assert(sym < gdeflate_alphabet_size);

    unsigned int xlen, n;

    // Note: in practice, sym < 16, since "pack_codelens" does not do RLE.
    switch (sym)
    {
      case 16:
        // Copy the prev code n times
        xlen = 2;
        n = 3 + ((bits >> len) & 3);
        break;
      case 17:
        // n zeros
        xlen = 3;
        n = 3 + ((bits >> len) & 7);
        break;
      case 18:
        // n zeros
        xlen = 7;
        n = 11 + ((bits >> len) & 127);
        break;
      default:
        // Single lencode
        xlen = 0;
        n = 1;
        break;
    }
    unsigned int n_post = postfixSum(n, WARP_ALL);
    outpos += n_post - n;
    bool active = count >= n_post;
    unsigned int ballot = __ballot_sync(WARP_ALL, active);
    unsigned int last_tid = 32 - __ffs(__brev(ballot));

    // Write all lencodes < 16
    if (active && (sym < 16))
    {
      codelen[outpos] = sym;
    }
    __syncwarp();

    // Serialize all threads with sym == 16
    for (unsigned int tid = 0; tid < WARP_SIZE_U; ++tid)
    {
      // Note: always false since sym < 16
      if (active && (threadIdx.x == tid) && (sym == 16))
      {
        uint16_t last = codelen[outpos - 1];
        for (unsigned int ii = outpos; ii < outpos + n; ++ii)
        {
          codelen[ii] = last;
        }
      }
      __syncwarp();
    }

    outpos = __shfl_sync(WARP_ALL, outpos + n, last_tid);
    count = __shfl_sync(WARP_ALL, count - n_post, last_tid);
    br.eat(len + xlen, active);
    corrupted |= br.is_corrupted();

    if (corrupted || (ballot != WARP_ALL))
    {
      break;
    }
  }
}

// Generate code lengths for fixed Huffman block
// based on the DEFLATE RFC
inline __device__ void fixed_codelens(uint16_t *codelen)
{
  for (unsigned int i = threadIdx.x; i < 288 + 32; i += WARP_SIZE_U)
  {
    if (i < 144)
    {
      codelen[i] = 8;
    }
    else if (i < 256)
    {
      codelen[i] = 9;
    }
    else if (i < 280)
    {
      codelen[i] = 7;
    }
    else if (i < 288)
    {
      codelen[i] = 8;
    }
    else
    {
      codelen[i] = 5;
    }
  }
  __syncwarp();
}

#ifndef BRANCHLESS_DECODING
template <bool deflate64>
__device__ inline uint16_t
decode_litlen(const warp_decoder<288, 15> &dec, uint32_t bits, uint32_t code, uint16_t &sym, uint16_t &length)
{
  uint16_t len = dec.len4code(code);
  assert(len <= gdeflate_max_codelen);
  sym = dec.sym4code(code, len);
  assert(sym < gdeflate_valid_litlen_symbols);

  uint16_t *lenbase = deflate64 ? lenbase64 : lenbase32;
  uint8_t *xlenbits = deflate64 ? xlenbits64 : xlenbits32;
  unsigned char xlen = sym > 256 ? xlenbits[sym - 257] : 0;
  uint16_t lengthBase = sym > 256 ? lenbase[sym - 257] : 0;
  length = sym > 256 ? lengthBase + (uint16_t)((bits >> len) & mask<uint32_t>(xlen)) : 1;
  return len + xlen;
}
#endif // BRANCHLESS_DECODING

template <bool deflate64>
__device__ inline uint16_t
decode_distance(const warp_decoder<32, 15> &dec, uint32_t bits, uint32_t code, uint16_t &sym, uint32_t &distance)
{
  uint16_t len = dec.len4code(code);
  assert(len <= gdeflate_max_codelen);
  sym = dec.sym4code(code, len);
  assert(sym < gdeflate_valid_distance_symbols);

  uint16_t *distanceTable = deflate64 ? distanceTable64 : distanceTable32;
  uint8_t *xdistbits = deflate64 ? xdistbits64 : xdistbits32;
  unsigned char xdist = xdistbits[sym];
  distance = distanceTable[sym] + (uint32_t)((bits >> len) & mask<uint32_t>(xdist));
  return len + xdist;
}

template <bool deflate64>
class symbol_translator
{
public:
  __device__ uint16_t
  translate(uint32_t bits, uint32_t code, uint16_t sym, uint16_t len, bool isCopy, uint16_t &value) const
  {
    uint16_t lut_index = isCopy ? sym : (sym > 256) ? sym - 257 : 31;
    uint16_t *baseTable = isCopy ? (deflate64 ? distanceTable64 : distanceTable32)
                                 : (deflate64 ? lenbase64 : lenbase32);
    uint8_t *xbits = isCopy ? (deflate64 ? xdistbits64 : xdistbits32) : (deflate64 ? xlenbits64 : xlenbits32);
    uint8_t xlen = xbits[lut_index];
    value = baseTable[lut_index] + ((bits >> len) & mask<uint32_t>(xlen));
    return xlen;
  }
};

#define SIMPLE_LOADS
#define SIMPLE_STORES
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900)
// Turn off unroll optimization for Hopper
// This is a WAR to get around a compiler error that causes
// OOB writes that are always < 4 bytes.
// TODO: remove this WAR after deeper analysis of generated SASS
#define SIMPLE_STORES_UNROLL_OPT
#endif
// #define COALESCE_STORES

#ifdef SIMPLE_LOADS
inline __device__ uint4 loadBytes(const uint8_t *src, const uint32_t bytes)
{
  uint4 regs;
  uint8_t *buf = reinterpret_cast<uint8_t *>(&regs);
  for (uint32_t i = 0; i < bytes; ++i)
  {
    buf[i] = src[i];
  }
  return regs;
}
#else
inline __device__ uint4 loadBytes(const uint8_t *src, const uint32_t bytes)
{
  uint32_t *startAligned = (uint32_t *)((size_t)src & ~(3));
  int bytesToLoad = bytes + (int)((size_t)src & 3);

  uint4 regs = {0, 0, 0, 0};
  uint32_t tail;

  regs.x = startAligned[0];
  if (bytesToLoad > 4)
  {
    regs.y = startAligned[1];
  }
  if (bytesToLoad > 8)
  {
    regs.z = startAligned[2];
  }
  if (bytesToLoad > 12)
  {
    regs.w = startAligned[3];
  }
  if (bytesToLoad > 16)
  {
    tail = startAligned[4];
  }

  uint32_t shiftAmt = ((size_t)src & 3) * 8;

  // Shift the 5 registers down by the shift amount
  regs.x = __funnelshift_r(regs.x, regs.y, shiftAmt);
  regs.y = __funnelshift_r(regs.y, regs.z, shiftAmt);
  regs.z = __funnelshift_r(regs.z, regs.w, shiftAmt);
  regs.w = __funnelshift_r(regs.w, tail, shiftAmt);

  // unused bytes need to be zeroed out for vectorized storeBytes
  uint32_t bytes8 = bytes * 8;
  if (bytes < 16)
  {
    regs.w &= (uint32_t)WARP_ALL >> (16 * 8 - bytes8);
  }
  if (bytes < 12)
  {
    regs.z &= (uint32_t)WARP_ALL >> (12 * 8 - bytes8);
  }
  if (bytes < 8)
  {
    regs.y &= (uint32_t)WARP_ALL >> (8 * 8 - bytes8);
  }
  if (bytes < 4)
  {
    regs.x &= (uint32_t)WARP_ALL >> (4 * 8 - bytes8);
  }

  // Return lower 4 registers
  return regs;
}
#endif

#ifdef SIMPLE_STORES
inline __device__ void storeBytes(uint8_t *dst, uint4 bytes, const uint32_t count, uint8_t *endp, const int laneId)
{
#ifndef SIMPLE_STORES_UNROLL_OPT
  uint32_t real_count = count; // (endp - dst) < count ? (endp - dst) : count;
  uint8_t *real_bytes = reinterpret_cast<uint8_t *>(&bytes);
  for (uint32_t i = 0; i < real_count; ++i)
  {
    dst[i] = real_bytes[i];
  }
#else
  uint32_t real_count = count; //(endp - dst) < count ? (endp - dst) : count;

#pragma unroll
  for (uint32_t i = 0; i < 4; ++i)
  {
    if (i < real_count)
    {
      dst[i] = (uint8_t)(bytes.x >> (i * 8));
    }
  }
  if (real_count > 4)
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i)
    {
      if ((i + 4) < real_count)
      {
        dst[i + 4] = (uint8_t)(bytes.y >> (i * 8));
      }
    }
  if (real_count > 8)
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i)
    {
      if ((i + 8) < real_count)
      {
        dst[i + 8] = (uint8_t)(bytes.z >> (i * 8));
      }
    }
  if (real_count > 12)
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i)
    {
      if ((i + 12) < real_count)
      {
        dst[i + 12] = (uint8_t)(bytes.w >> (i * 8));
      }
    }
#endif
}
#else
#ifndef COALESCE_STORES
inline __device__ void storeBytes(uint8_t *dst, uint4 bytes, const uint32_t count, uint8_t *endp, const int laneId)
{
  uint32_t *startAligned = (uint32_t *)((size_t)dst & ~(3));
  uint32_t lsbs = ((size_t)dst) & 3;
  uint32_t bytesToStore = count + lsbs;
  uint32_t shiftAmt = lsbs * 8;

  if (count > 0)
  {
    atomicOr(startAligned, bytes.x << shiftAmt);
  }
  if (bytesToStore > 4)
  {
    atomicOr(startAligned + 1, __funnelshift_l(bytes.x, bytes.y, shiftAmt));
  }
  if (bytesToStore > 8)
  {
    atomicOr(startAligned + 2, __funnelshift_l(bytes.y, bytes.z, shiftAmt));
  }
  if (bytesToStore > 12)
  {
    atomicOr(startAligned + 3, __funnelshift_l(bytes.z, bytes.w, shiftAmt));
  }
  if (bytesToStore > 16)
  {
    atomicOr(startAligned + 4, bytes.w >> (32 - shiftAmt));
  }
}
#else
inline __device__ void storeBytes(uint8_t *outp, uint4 bytes, uint32_t cnt, uint8_t *endp, const int laneId)
{
  // Compute aligned destination pointers and the number of bytes to store
  uint32_t *start = (uint32_t *)((size_t)outp & ~(3));
  uint32_t lsbs = (size_t)outp & 3;
  uint32_t bytesToStore = cnt + lsbs;
  uint32_t shiftAmt = lsbs * 8;

  uint32_t bytes0 = bytes.x << shiftAmt;
  uint32_t bytes1 = __funnelshift_l(bytes.x, bytes.y, shiftAmt);
  uint32_t bytes2 = __funnelshift_l(bytes.y, bytes.z, shiftAmt);
  uint32_t bytes3 = __funnelshift_l(bytes.z, bytes.w, shiftAmt);
  uint32_t bytes4 = bytes.w >> (32 - shiftAmt);

  // Should compile to 4 ISETP
  bool isLast0 = bytesToStore <= 4;
  bool isLast4 = bytesToStore > 16;
  bool isLast1 = bytesToStore <= 8 && !isLast0;
  bool isLast3 = bytesToStore > 12 && !isLast4;

  uint32_t startBytes = bytes0;

  // End of the range
  uint32_t *end = start + ((bytesToStore - 1) >> 2);
  uint32_t endBytes = isLast0 ? bytes0 : (isLast1 ? bytes1 : (isLast3 ? bytes3 : (isLast4 ? bytes4 : bytes2)));

  // Look at the end position of the thread before me
  uint32_t prevEnd = __shfl_up_sync(WARP_ALL, (uint32_t)end, 1);

  // Look at start positions and data of 3 threads ahead of me (we can only coalesce with up to 3 consecutive threads)
  uint32_t nextStart1 = __shfl_down_sync(WARP_ALL, (uint32_t)start, 1);
  uint32_t nextStart2 = __shfl_down_sync(WARP_ALL, (uint32_t)start, 2);
  uint32_t nextStart3 = __shfl_down_sync(WARP_ALL, (uint32_t)start, 3);

  uint32_t nextStartBytes1 = __shfl_down_sync(WARP_ALL, startBytes, 1);
  uint32_t nextStartBytes2 = __shfl_down_sync(WARP_ALL, startBytes, 2);
  uint32_t nextStartBytes3 = __shfl_down_sync(WARP_ALL, startBytes, 3);

  // Coalesce from threads ahead of me
  if ((uint32_t)end == nextStart1)
  {
    endBytes |= nextStartBytes1;
  }
  if ((uint32_t)end == nextStart2)
  {
    endBytes |= nextStartBytes2;
  }
  if ((uint32_t)end == nextStart3)
  {
    endBytes |= nextStartBytes3;
  }

  // If coalescing group starts at our end position, perform the write
  if (end != start || laneId)
  {
    if (end <= (uint32_t *)((size_t)endp & ~(3)))
    {
      atomicOr(
        end,
        endBytes
      ); // OPT: this only needs to be atomic for thread 0 in the block (if we're continuing previous block)
    }
  }
  else
  {
    startBytes |= endBytes; // else, coalesce start and end
  }

  // If coalescing group starts at our start position, perform the write
  if ((uint32_t)start != prevEnd)
  {
    atomicOr(start, startBytes);
  }

  // Store intermediate words
  if (isLast4)
  {
    start[3] = bytes3;
  }
  if (isLast3 || isLast4)
  {
    start[2] = bytes2;
  }
  if (!isLast1 && !isLast0)
  {
    start[1] = bytes1;
  }
}
#endif
#endif

template <bool check_bounds, bool should_output>
__device__ void write_output(
  unsigned char *output,
  uint16_t sym,
  uint16_t length,
  uint32_t distance,
  bool isCopy,
  bool active,
  unsigned char *output_end_ptr,
  bool &corrupted
)
{
  if (!should_output)
  {
    return;
  }
  if (check_bounds)
  {
    corrupted |= (active && (sym != 256)) ? (output + length > output_end_ptr) : false;
    corrupted = (__any_sync(WARP_ALL, corrupted) != 0);
    if (corrupted)
    {
      return;
    }
  }

  bool isLiteral = (active && !isCopy && (sym < 256));
  uint4 buf = {0, 0, 0, 0};
  uint16_t bytesCopied = 0;
  if (isLiteral)
  {
    buf.x = (uint32_t)sym;
    bytesCopied = 1;
  }

#ifdef IGNORE_COPIES
  storeBytes(output, buf, bytesCopied, output_end_ptr, threadIdx.x);
#else

  // Independent copies
  int copyMask = __ballot_sync(WARP_ALL, isCopy); // Collect a ballot of pending matches
  int first_copy = __ffs(copyMask) - 1; // Get thread id of the first copy thread
  unsigned char *copyhwm = (unsigned char *)( // Get copy high water mark to
    __shfl_sync(WARP_ALL, (uintptr_t)(output), first_copy)
  ); // find independent copies

  bool isIndependentMatch = false;
  unsigned char *matchEndPtr = output - distance + 16; // Compute the distance of the match copy end
  isIndependentMatch = isCopy && (matchEndPtr < copyhwm); // Check if match copy ends before the high water mark
  // TODO: Add stricter condition to only check intersection with other matches in this CW
  if (isIndependentMatch)
  {
    bytesCopied = min(length, 16); // we can only copy 16B max in one shot
    // This condition should be unnecessary, need further investigate the reason.
    if (output_end_ptr - output + distance > bytesCopied)
    {
      buf = loadBytes(output - distance, bytesCopied); // Copy independent match data
    }
  }

  // flush to the output stream - dense copy
  uint16_t storeBytesToCopy = (isIndependentMatch || isLiteral) ? bytesCopied : 0;
  storeBytes(output, buf, storeBytesToCopy, output_end_ptr, threadIdx.x);
  __syncwarp();
  // now process all pending matches cooperatively one-by-one, including tails of long independent copies
  bool dependentMatch = isCopy && (!(isIndependentMatch) || (bytesCopied < length));
  int matchCopyMask = __ballot_sync(WARP_ALL, dependentMatch); // Collect a ballot of pending matches
  while (matchCopyMask)
  {
    int tid = __ffs(matchCopyMask) - 1;
    int currOffset = __shfl_sync(WARP_ALL, distance, tid); // Broadcast this distance
    int currLength = __shfl_sync(WARP_ALL, length, tid); // Broadcast this length
    int currBytesCopied = __shfl_sync(WARP_ALL, bytesCopied, tid); // Broadcast this bytesCopied
    uint8_t *currDst = (uint8_t *)(__shfl_sync(WARP_ALL, (uintptr_t)(output), tid)); // Broadcast this output

    // cooperatively copy data
    uint8_t *currSrc = &currDst[0 - currOffset];
    // for long independent matches (>16B) we already copied the first 16 bytes, skip those
#ifdef VECTORIZED_DEPENDENT_COPIES
    bool isNotLoopingMatch = (currOffset >= currLength);
    if (isNotLoopingMatch)
    {
      // Use vectorized reads and writes for non looping matches
      // Assign bytes evenly to each thread except thread 0 that
      // handles the extra initial (up to 3) bytes until a word boundary
      uint8_t *currSrcAligned = (uint8_t *)((size_t)(currSrc + 3) & ~(3)); // Get higher 4B aligned pointer
      uint8_t unalignedBytes = (uint8_t)(currSrcAligned - currSrc);
      int bytesPerThread = (currLength - currBytesCopied - unalignedBytes + WARP_SIZE - 1) / WARP_SIZE;
      int start = currBytesCopied + threadIdx.x * bytesPerThread + ((threadIdx.x > 0) ? unalignedBytes : 0);
      bytesPerThread += ((threadIdx.x == 0) ? unalignedBytes : 0);
      int bytes = (start < currLength) ? min(bytesPerThread, currLength - start) : 0;
      buf = loadBytes(currSrc + start, bytes);
      storeBytes(currDst + start, buf, bytes, output_end_ptr, threadIdx.x);
    }
    else
    {
      for (int i = threadIdx.x + currBytesCopied; i < currLength; i += WARP_SIZE_U)
      {
        currDst[i] = currSrc[i % currOffset]; // take care of the overlapping copies with modulo
      }
    }
#else // VECTORIZED_DEPENDENT_COPIES
    if (currLength - currBytesCopied <= 32)
    {
      // Process "short" copies to avoid any additional logic
      int i = threadIdx.x + currBytesCopied;
      if (i < currLength)
      {
        currDst[i] = currSrc[i % currOffset]; // take care of the overlapping copies with modulo
      }
    }
    else
    {
      // Branch for "long" copies
      // Manually unrolled loop twice for better ILP
      // Note: using intentionally unsigned WARP_SIZE_U, as WARP_SIZE leads to significantly
      //       more unrolled loops in the resulting PTX and worse performance.
      for (int i = threadIdx.x + currBytesCopied; i < currLength; i += 2 * WARP_SIZE_U)
      {
        uint8_t v1 = 0;
        uint8_t v2 = 0;

        // Load both values
        v1 = currSrc[i % currOffset]; // take care of the overlapping copies with modulo
        if (i + WARP_SIZE_U < currLength)
        {
          v2 = currSrc[(i + WARP_SIZE_U) % currOffset]; // take care of the overlapping copies with modulo
        }

        // Write to output
        currDst[i] = v1;
        if (i + WARP_SIZE_U < currLength)
        {
          currDst[i + WARP_SIZE_U] = v2;
        }
      }
    }
#endif // VECTORIZED_DEPENDENT_COPIES

    matchCopyMask = matchCopyMask & (WARP_ALL << (tid + 1));
    __syncwarp();
  }
#endif
}

template <bool check_bounds, bool should_output>
__device__ void write_output_without_copies(
  unsigned char *output,
  uint16_t sym,
  uint16_t length,
  uint32_t distance,
  bool isCopy,
  bool active,
  unsigned char *output_end_ptr,
  uint32_t *copies,
  bool &corrupted
)
{

  if (!should_output)
  {
    return;
  }
  if (check_bounds)
  {
    corrupted |= (active && (sym != 256)) ? (output + length > output_end_ptr) : false;
    corrupted = __shfl_sync(WARP_ALL, corrupted, 0);
    if (corrupted)
    {
      return;
    }
  }
  bool isLiteral = (active && !isCopy && (sym < 256));
  uint4 buf = {0, 0, 0, 0};
  uint16_t bytesCopied = 0;
  if (isLiteral)
  {
    buf.x = (uint32_t)sym;
    bytesCopied = 1;
  }

#ifdef IGNORE_COPIES
  storeBytes(output, buf, bytesCopied, output_end_ptr, 0); //TODO reduce
#else

  // Independent copies
  bool isIndependentMatch = false;
  unsigned char *matchEndPtr = output - distance + 16; // Compute the distance of the match copy end
  isIndependentMatch = isCopy && (matchEndPtr < output); // Check if match copy ends before the high water mark
  // TODO: Add stricter condition to only check intersection with other matches in this CW
  if (isIndependentMatch)
  {
    bytesCopied = min(length, 16); // we can only copy 16B max in one shot
    buf = loadBytes(output - distance, bytesCopied); // Copy independent match data
  }

  // flush to the output stream - dense copy
  uint16_t storeBytesToCopy = (isIndependentMatch || isLiteral) ? bytesCopied : 0;
  storeBytes(output, buf, storeBytesToCopy, output_end_ptr, 0); //TODO reduce
  __syncwarp();

  // now process all pending matches cooperatively one-by-one, including tails of long independent copies
  bool dependentMatch = isCopy && (!(isIndependentMatch) || (bytesCopied < length));
  dependentMatch = __shfl_sync(WARP_ALL, dependentMatch, 0);
  //TODO reduce
  if (dependentMatch)
  {
    int tid = 0;
    int currOffset = __shfl_sync(WARP_ALL, distance, tid); // Broadcast this distance
    int currLength = __shfl_sync(WARP_ALL, length, tid); // Broadcast this length
    int currBytesCopied = __shfl_sync(WARP_ALL, bytesCopied, tid); // Broadcast this bytesCopied
    uint8_t *currDst = (uint8_t *)(__shfl_sync(WARP_ALL, (uintptr_t)(output), tid)); // Broadcast this output

    // cooperatively copy data
    uint8_t *currSrc = &currDst[0 - currOffset];
    // for long independent matches (>16B) we already copied the first 16 bytes, skip those
#ifdef VECTORIZED_DEPENDENT_COPIES
    bool isNotLoopingMatch = (currOffset >= currLength);
    if (isNotLoopingMatch)
    {
      // Use vectorized reads and writes for non looping matches
      // Assign bytes evenly to each thread except thread 0 that
      // handles the extra initial (up to 3) bytes until a word boundary
      uint8_t *currSrcAligned = (uint8_t *)((size_t)(currSrc + 3) & ~(3)); // Get higher 4B aligned pointer
      uint8_t unalignedBytes = (uint8_t)(currSrcAligned - currSrc);
      int bytesPerThread = (currLength - currBytesCopied - unalignedBytes + WARP_SIZE - 1) / WARP_SIZE;
      int start = currBytesCopied + threadIdx.x * bytesPerThread + ((threadIdx.x > 0) ? unalignedBytes : 0);
      bytesPerThread += ((threadIdx.x == 0) ? unalignedBytes : 0);
      int bytes = (start < currLength) ? min(bytesPerThread, currLength - start) : 0;
      buf = loadBytes(currSrc + start, bytes);
      storeBytes(currDst + start, buf, bytes, output_end_ptr, threadIdx.x);
    }
    else
    {
      for (int i = threadIdx.x + currBytesCopied; i < currLength; i += WARP_SIZE_U)
      {
        currDst[i] = currSrc[i % currOffset]; // take care of the overlapping copies with modulo
      }
    }
#else // VECTORIZED_DEPENDENT_COPIES
    if (currLength - currBytesCopied <= 32)
    {
      // Process "short" copies to avoid any additional logic
      int i = threadIdx.x + currBytesCopied;
      if (i < currLength)
      {
        currDst[i] = currSrc[i % currOffset]; // take care of the overlapping copies with modulo
      }
    }
    else
    {
      // Branch for "long" copies
      // Manually unrolled loop twice for better ILP
      for (int i = threadIdx.x + currBytesCopied; i < currLength; i += 2 * WARP_SIZE_U)
      {
        uint8_t v1 = 0;
        uint8_t v2 = 0;

        // Load both values
        v1 = currSrc[i % currOffset]; // take care of the overlapping copies with modulo
        if (i + WARP_SIZE_U < currLength)
        {
          v2 = currSrc[(i + WARP_SIZE_U) % currOffset]; // take care of the overlapping copies with modulo
        }

        // Write to output
        currDst[i] = v1;
        if (i + WARP_SIZE_U < currLength)
        {
          currDst[i + WARP_SIZE_U] = v2;
        }
      }
    }
#endif // VECTORIZED_DEPENDENT_COPIES
    __syncwarp();
  }

#endif
}

template <bool check_bounds, bool should_output, bool deflate64, typename Treader>
__device__ unsigned char *parse_symbols(
  unsigned char *output,
  unsigned char *output_end_ptr,
  Treader &br,
  unsigned int hlit,
  unsigned int hdist,
  const uint16_t *codelen,
  uint16_t *counts,
  uint16_t *symbols,
  uint16_t *offset,
  uint32_t *basecode,
  bool &corrupted,
  gdeflate_trace *&trace
)
{
  if (check_bounds && corrupted)
  {
    return nullptr;
  }

  warp_decoder<gdeflate_litlen_symbols, gdeflate_max_codelen> lit;
  lit.init(hlit, codelen, counts, symbols, offset, basecode);

  warp_decoder<gdeflate_distance_symbols, gdeflate_max_codelen> dist;
  dist.init(hdist, codelen + hlit, counts + 16, symbols + gdeflate_litlen_symbols, offset + 16, basecode + 16);

#ifdef BRANCHLESS_DECODING
  // Create the branchless decoder object using the two warp_decoder above for bootstrapping
  // symbols assumed to be 288+32 contiguous elements for lit/len+distance respectively
  // offset and basecode assumed to be 16+16 contiguous elements for lit/len+distance respectively
  warp_dual_decoder<> dec(symbols, offset, basecode);
  // Create branchless symbol translator that stores the length and distance tables in registers
  symbol_translator<deflate64> st;
#endif // BRANCHLESS_DECODING

  bool block_end = false;
  bool isCopy = false;
  uint16_t length = 0;
  unsigned char *outputptr = output;
  unsigned char *copyptr = output;
  do
  {
    // Get literal/length or distance code
    uint32_t bits = br.peek(gdeflate_max_codelen + gdeflate_max_extra_bits, true);
    uint32_t code = __brev(bits);

    uint16_t sym;
    uint32_t distance = 0;
#ifdef BRANCHLESS_DECODING
    uint16_t base = isCopy ? 16 : 0;
    // Unified decode for both literal/length and distance symbols
    uint16_t numbits = dec.len4code(code, base);
    sym = dec.sym4code(code, numbits, base);

    // Translate symbol to value (no op for literals, xtra bits for length and distance)
    uint16_t value = 0;
    numbits += st.translate(bits, code, sym, numbits, isCopy, value);
    length = isCopy ? length : value;
    distance = isCopy ? value : 0;
#else
    uint16_t numbits = isCopy ? decode_distance<deflate64>(dist, bits, code, sym, distance)
                              : decode_litlen<deflate64>(lit, bits, code, sym, length);
#endif
    uint16_t output_bytes = isCopy ? 0 : length;

    // Check for block end symbol 256
    unsigned int block_end_ballot = __ballot_sync(WARP_ALL, sym == 256);
    block_end = (block_end_ballot > 0);

    // Check if thread is after block end
    unsigned int last_active_thread = block_end_ballot ? __ffs(block_end_ballot) : 32;
    unsigned int active_mask = mask<unsigned int>(last_active_thread);
    bool active = isCopy | ((active_mask >> threadIdx.x) > 0);

    outputptr += prefixSum(output_bytes, WARP_ALL);

    // Write literals and process copies
    write_output<check_bounds, should_output>(
      isCopy ? copyptr : outputptr,
      sym,
      length,
      distance,
      isCopy,
      active,
      output_end_ptr,
      corrupted
    );

    // Save copy pointer for next decode step
    if (sym > 256)
    {
      copyptr = outputptr;
    }

    br.eat(numbits, active);

    if (check_bounds)
    {
      corrupted |= br.is_corrupted();
      if (corrupted)
      {
        return nullptr;
      }
    }

#ifdef GDEFLATE_LOG_TRACE
    trace[threadIdx.x] = {isCopy, length, sym, distance, (int)(outputptr - output), active, output_bytes};
    trace += WARP_SIZE_U;
#endif

    // Update output pointer and copy flag
    outputptr = (unsigned char *)(__shfl_sync(WARP_ALL, (uintptr_t)(outputptr + output_bytes), last_active_thread - 1));
    isCopy = active && (sym > 256);

  } while (!block_end);

  // Last round for copies
  {
    uint32_t bits = br.peek(gdeflate_max_codelen + gdeflate_max_extra_bits, true);
    uint32_t code = __brev(bits);

    uint16_t sym;
    uint32_t distance = 0;
    uint16_t numbits = isCopy ? decode_distance<deflate64>(dist, bits, code, sym, distance) : 0;

    // Write literals and process copies
    write_output<check_bounds, should_output>(copyptr, sym, length, distance, isCopy, isCopy, output_end_ptr, corrupted);

    br.eat(numbits, isCopy);

    if (check_bounds)
    {
      corrupted |= br.is_corrupted();
      if (corrupted)
      {
        return nullptr;
      }
    }

#ifdef GDEFLATE_LOG_TRACE
    trace[threadIdx.x] = {isCopy, length, sym, distance, (int)(outputptr - output), isCopy, length};
    trace += WARP_SIZE_U;
#endif
  }

  return outputptr - 1; // Account for last thread adding 1
}

template <bool check_bounds, bool should_output, typename Treader>
__device__ unsigned char *
copy_uncompressed(unsigned char *output, unsigned char *output_end_ptr, Treader &br, uint32_t size, bool &corrupted)
{

  if (check_bounds && should_output && (output + size > output_end_ptr))
  {
    corrupted = true;
    return output + size;
  }

  // TODO: Optimize should_output == false case to skip reading data
  for (uint32_t i = threadIdx.x; i < WARP_SIZE_U * (size / WARP_SIZE_U); i += WARP_SIZE_U)
  {
    unsigned char c = br.read(8, true);
    if (should_output)
    {
      output[i] = c;
    }
  }

  uint32_t remainder = size % WARP_SIZE_U;
  if (remainder > 0)
  {
    unsigned char c = br.read(8, threadIdx.x < remainder);
    if (threadIdx.x < remainder)
    {
      if (should_output)
      {
        output[WARP_SIZE_U * (size / WARP_SIZE_U) + threadIdx.x] = c;
      }
    }
  }

  if (check_bounds)
  {
    corrupted |= br.is_corrupted();
  }
  return output + size;
}

template <
  bool check_bounds = true,
  bool should_output = true,
  typename status_t = gdeflateStatus_t,
  status_t success = gdeflateSuccess,
  status_t failure = gdeflateErrorCannotDecompress>
__device__ void decompress_tile(
  const uint32_t *input,
  unsigned char *output,
  const uint32_t input_size,
  const uint32_t output_size,
  uint16_t *codelen,
  uint16_t *symbols,
  uint16_t *counts,
  uint16_t *offset,
  uint32_t *basecode,
  size_t *decomp_size,
  status_t *decomp_status,
  gdeflate_trace *trace
)
{
  warp_bitreader<check_bounds, uint32_t> br(input, input + (input_size / sizeof(uint32_t)));
  uint32_t header, bfinal = 0, btype;
  bool tid0 = threadIdx.x == 0;
#ifdef GDEFLATE_ENABLE_DEFLATE64
  constexpr bool deflate64 = true;
#else
  constexpr bool deflate64 = false;
#endif

  unsigned char *output_start_ptr = output;
  unsigned char *output_end_ptr = output + output_size;
  bool corrupted = false;
  do
  {
    // Only thread 0 has the header
    header = __shfl_sync(WARP_ALL, br.peek(3 + 14, tid0), 0);
    bfinal = header & 1;
    btype = (header >> 1) & 3;
    br.eat(3, tid0);

    assert(btype != 3);

    switch (btype)
    {
      case 2: {
        // Read header from thread 0
        header >>= 3;
        br.eat(14, tid0);

        // TODO: h{lit,dist,clen} are confusing names. These are the names
        // of fields in the dynamic deflate header before the offsets (257,
        // 1, and 4, respectively) are applied. Better names for post-offset
        // quantities would be n{lit,dist,clen}, like in our deflate
        // implementation, or similar. This would need to be fixed in many places.
        unsigned int hlit = ((header >> 0) & mask<uint32_t>(5)) + 257;
        unsigned int hdist = ((header >> 5) & mask<uint32_t>(5)) + 1;
        unsigned int hclen = ((header >> 10) & mask<uint32_t>(4)) + 4;

        // NOTE: The deflate specification states that what we call hlit should
        // be less or equal to 286. In the header, values up to 288 can be
        // specified. We allow this as it does not break the decoding as long
        // as the associated codelengths are zero so that the extra two symbols
        // do not appear. It is also likely a common violation of the deflate
        // specification, in part due to it allowing hdist values that cause
        // inclusion of invalid distance symbols in the codelength tables,
        // but not hlit. This violation was also perpetrated by our previous
        // implementation of GDeflate compression.
        // TODO: Check that codelen[i] is zero for all 286 <= i < hlit, mark
        // as corrupted if not.

        read_lencodes(codelen, br, hclen, corrupted);
        unpack_codelens(codelen, hlit + hdist, br, codelen, counts, symbols, offset, basecode, corrupted);
        output = parse_symbols<check_bounds, should_output, deflate64>(
          output,
          output_end_ptr,
          br,
          hlit,
          hdist,
          codelen,
          counts,
          symbols,
          offset,
          basecode,
          corrupted,
          trace
        );
        break;
      }
      case 1: {
        fixed_codelens(codelen);
        output = parse_symbols<check_bounds, should_output, deflate64>(
          output,
          output_end_ptr,
          br,
          288,
          32,
          codelen,
          counts,
          symbols,
          offset,
          basecode,
          corrupted,
          trace
        );
        break;
      }
      case 0: {
#ifdef GDEFLATE_ENABLE_DEFLATE64
        uint32_t size = br.read(16, tid0);
#else
        uint32_t size = br.read_aligned(16, tid0);
        uint32_t nsize = br.read_aligned(16, tid0);
        assert((!tid0) || (size == (~nsize & 0xffff)));
#endif

        // Broadcast size to all other threads and copy data
        size = __shfl_sync(WARP_ALL, size, 0);
        output = copy_uncompressed<check_bounds, should_output>(output, output_end_ptr, br, size, corrupted);
        break;
      }
    }

    if (check_bounds && corrupted)
    {
      if (tid0)
      {
        if (should_output)
        {
          *decomp_status = failure;
        }
        if (decomp_size)
        {
          *decomp_size = 0;
        }
      }
      return;
    }
  } while (!bfinal);

  __syncwarp();
  size_t decomp_size_reg = output - output_start_ptr;
  if (tid0)
  {
    // if (check_bounds && decomp_size) *decomp_size = output - output_start_ptr;
    if (check_bounds && decomp_size)
    {
      *decomp_size = decomp_size_reg;
    }
    if (check_bounds && should_output)
    {
      *decomp_status = success;
    }
  }
  __syncwarp();
}

template <
  int WARPS_PER_CTA,
  bool check_bounds = true,
  bool should_output = true,
  typename status_t = gdeflateStatus_t,
  status_t success = gdeflateSuccess,
  status_t failure = gdeflateErrorCannotDecompress>
__launch_bounds__(WARP_SIZE *WARPS_PER_CTA) __global__ void gdeflateDecompress(
  const uint32_t *const *data_ptr,
  uint8_t *const *dest_ptr,
  const size_t *data_size,
  const size_t *dest_size,
  unsigned int num_streams,
  size_t *decomp_size,
  status_t *decomp_status,
  gdeflate_trace *trace
)
{
  static_assert(check_bounds || should_output, "gdeflateDecompress: either check_bounds or should_output must be true!");

  __shared__ uint16_t codelen[WARPS_PER_CTA][gdeflate_total_symbols_smem];
  __shared__ uint16_t symbols[WARPS_PER_CTA][gdeflate_total_symbols_smem];
  __shared__ uint16_t counts[WARPS_PER_CTA][32];
  __shared__ uint16_t offset[WARPS_PER_CTA][32];
  __shared__ uint32_t basecode[WARPS_PER_CTA][32];

  unsigned int block_id = blockIdx.x;
  unsigned int stride = gridDim.x * blockDim.y;
  for (unsigned int bid = block_id * blockDim.y + threadIdx.y; bid < num_streams; bid += stride)
  {
    decompress_tile<check_bounds, should_output, status_t, success, failure>(
      data_ptr[bid],
      should_output ? dest_ptr[bid] : nullptr,
      (uint32_t)data_size[bid],
      should_output ? (uint32_t)dest_size[bid] : 0,
      codelen[threadIdx.y],
      symbols[threadIdx.y],
      counts[threadIdx.y],
      offset[threadIdx.y],
      basecode[threadIdx.y],
      check_bounds ? decomp_size + bid : nullptr,
      (check_bounds && should_output) ? decomp_status + bid : nullptr,
      trace
    );
  }
}

#ifdef SPECIALIZED_LAUNCH_CONFIG_INSTANTIATIONS
template <int warpsPerSM>
void gdeflateDecompressTemplatedWrapper(
  const uint32_t *const *data_ptr,
  uint8_t *const *dest_ptr,
  const size_t *data_size,
  const size_t *dest_size,
  unsigned int num_streams,
  size_t *decomp_size,
  gdeflateStatus_t *decomp_status,
  gdeflate_trace *trace,
  int grid,
  dim3 block,
  int &maxActiveBlocks,
  cudaEvent_t &eventStart,
  cudaEvent_t &eventStop,
  cudaStream_t stream
)
{
  int threads = block.x * block.y * block.z;
  // NOTE(pgmerek): If this API is re-activated, we need to consider adding a device guard here
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, gdeflateDecompress<warpsPerSM>, threads, 0);
  cudaEventRecord(eventStart, stream);
  gdeflateDecompress<warpsPerSM, false, true><<<grid, block, 0, stream>>>(
    data_ptr,
    dest_ptr,
    data_size,
    dest_size,
    num_streams,
    decomp_size,
    decomp_status,
    trace
  );
  cudaEventRecord(eventStop, stream);
}

// Statically build a list of warp configurations
// using variadic templates
template <int... WarpList>
struct warplist
{
  typedef warplist<WarpList..., sizeof...(WarpList) + 1> next;
};

// Helper to build the warp configuration list
template <int N>
struct build_warplist
{
  typedef typename build_warplist<N - 1>::type::next type;
};
template <>
struct build_warplist<1>
{
  typedef warplist<1> type;
};

// Wrapper to instantiate and run the kernel with the required warp config
template <int... WarpList>
void gdeflateDecompressVariadic_(
  const uint32_t *const *data_ptr,
  uint8_t *const *dest_ptr,
  const size_t *data_size,
  const size_t *dest_size,
  unsigned int num_streams,
  size_t *decomp_size,
  gdeflateStatus_t *decomp_status,
  gdeflate_trace *trace,
  int grid,
  dim3 block,
  int &maxActiveblocks,
  cudaEvent_t &eventStart,
  cudaEvent_t &eventStop,
  cudaStream_t stream,
  int index,
  warplist<WarpList...>
)
{
  typedef void (*element_type)(
    const uint32_t *const *,
    uint8_t *const *,
    const size_t *,
    const size_t *,
    unsigned int,
    size_t *,
    gdeflateStatus_t *,
    gdeflate_trace *,
    int,
    dim3,
    int &,
    cudaEvent_t &,
    cudaEvent_t &,
    cudaStream_t
  );

  // Statically create a function pointer array for each warp configuration
  static constexpr element_type table[] = {gdeflateDecompressTemplatedWrapper<WarpList>...};

  // Run the version with the required warp configuration
  table[index - 1](
    data_ptr,
    dest_ptr,
    data_size,
    dest_size,
    num_streams,
    decomp_size,
    decomp_status,
    trace,
    grid,
    block,
    maxActiveblocks,
    eventStart,
    eventStop,
    stream
  );
}

// This is the wrapper to call
inline void gdeflateDecompressVariadic(
  const uint32_t *const *data_ptr,
  uint8_t *const *dest_ptr,
  const size_t *data_size,
  const size_t *dest_size,
  unsigned int num_streams,
  size_t *decomp_size,
  gdeflateStatus_t *decomp_status,
  gdeflate_trace *trace,
  int grid,
  dim3 block,
  int &maxActiveblocks,
  cudaEvent_t &eventStart,
  cudaEvent_t &eventStop,
  cudaStream_t stream,
  int index
)
{
  typedef typename build_warplist<MAX_WARPS_PER_CTA>::type warplist_type;

  gdeflateDecompressVariadic_(
    data_ptr,
    dest_ptr,
    data_size,
    dest_size,
    num_streams,
    decomp_size,
    decomp_status,
    trace,
    grid,
    block,
    maxActiveblocks,
    eventStart,
    eventStop,
    stream,
    index,
    warplist_type{}
  );
}
#endif // SPECIALIZED_LAUNCH_CONFIG_INSTANTIATIONS

} // namespace gdeflate
