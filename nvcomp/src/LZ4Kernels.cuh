/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES.
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

#include <cub/cub.cuh>

#include <cuda_runtime_api.h>

#include <cassert>
#include <vector>

#include "CorrectnessChecks.cuh"
#include "LZ4Constants.cuh"
#include "LZ4Types.cuh"
#include "LZ77_decomp.cuh"
#include "nvcomp/shared_types.h"

namespace nvcomp
{

using position_type = lz4::position_type;
using offset_type = lz4::offset_type;

// ideally this would fit in a quad-word -- right now though it spills into
// 24-bytes (instead of 16-bytes).
struct chunk_header
{
  const uint8_t *src;
  uint8_t *dst;
  uint32_t size;
};

struct compression_chunk_header
{
  const uint8_t *src;
  uint8_t *dst;
  offset_type *hash;
  size_t *comp_size;
  uint32_t size;
};

using sequence = lz4::sequence;

/******************************************************************************
 * DEVICE FUNCTIONS AND KERNELS ***********************************************
 *****************************************************************************/

inline __device__ __host__ size_t maxSizeOfStream(const size_t max_uncomp_chunk_size)
{
  const size_t expansion = max_uncomp_chunk_size + 1 + roundUpDiv(max_uncomp_chunk_size, 255);
  return roundUpTo(expansion, sizeof(size_t));
}

inline __device__ int warpBallot(int vote) { return __ballot_sync(WARP_ALL, vote); }

template <typename T>
inline __device__ void writeWord(uint8_t *const address, const T word)
{
#pragma unroll
  for (size_t i = 0; i < sizeof(T); ++i)
  {
    address[i] = static_cast<uint8_t>((word >> (8 * i)) & 0xff);
  }
}

template <typename T, typename S>
inline __device__ T readWord(const S *const address)
{
  T word = 0;
#pragma unroll
  for (size_t i = 0; i < sizeof(T) / sizeof(S); ++i)
  {
    word |= address[i] << (8 * sizeof(S) * i);
  }
  return word;
}

template <int BLOCK_SIZE, typename CG>
inline __device__ void writeLSIC(uint8_t *const out, const position_type number, CG &cg)
{
  assert(BLOCK_SIZE == cg.size());

  const position_type num = (number / 0xffu) + 1;
  const uint8_t leftOver = number % 0xffu;
  for (position_type i = cg.thread_rank(); i < num; i += BLOCK_SIZE)
  {
    const uint8_t val = i + 1 < num ? 0xffu : leftOver;
    out[i] = val;
  }
}

struct token_type
{
  position_type num_literals;
  position_type num_matches;

  __device__ bool hasNumLiteralsOverflow() const { return num_literals >= 15; }

  __device__ bool hasNumMatchesOverflow() const { return num_matches >= 19; }

  __device__ position_type numLiteralsOverflow() const
  {
    assert(hasNumLiteralsOverflow());
    return num_literals - 15;
  }

  __device__ uint8_t numLiteralsForHeader() const
  {
    if (hasNumLiteralsOverflow())
    {
      return 15;
    }
    else
    {
      return num_literals;
    }
  }

  __device__ position_type numMatchesOverflow() const
  {
    assert(num_matches >= 19);
    return num_matches - 19;
  }

  __device__ uint8_t numMatchesForHeader() const
  {
    if (hasNumMatchesOverflow())
    {
      return 15;
    }
    else
    {
      return num_matches - 4;
    }
  }
  __device__ position_type lengthOfLiteralEncoding() const
  {
    if (hasNumLiteralsOverflow())
    {
      const position_type num = numLiteralsOverflow();
      const position_type length = (num / 0xff) + 1;
      return length;
    }
    return 0;
  }

  __device__ position_type lengthOfMatchEncoding() const
  {
    assert(hasNumMatchesOverflow());
    const position_type num = numMatchesOverflow();
    const position_type length = (num / 0xff) + 1;
    return length;
  }
};

class BufferControl
{
public:
  __device__ BufferControl(
    uint8_t *const buffer, // points to a buffer in shared memory
    const uint8_t *const compData, // points to a compressed chunk
    const position_type length
  )
      : // the compressed chunk's length
      m_offset(0)
      , m_length(length)
      , m_buffer(buffer)
      , m_compData(compData)
  {
    // do nothing
  }

#ifdef WARP_READ_LSIC
  // this is currently unused as its slower
  template <typename CG>
  inline __device__ position_type queryLSIC(const position_type idx, CG &cg) const
  {
    if (idx + WARP_SIZE_U <= end())
    {
      // most likely case
      const uint8_t byte = rawAt(idx)[cg.thread_rank()];

      uint32_t mask = warpBallot(byte != 0xff);
      mask = __brev(mask);

      const position_type fullBytes = __clz(mask);

      if (fullBytes < WARP_SIZE_U)
      {
        return fullBytes * 0xff + rawAt(idx)[fullBytes];
      }
      else
      {
        return WARP_SIZE_U * 0xff;
      }
    }
    else
    {
      uint8_t byte;
      if (idx + cg.thread_rank() < end())
      {
        byte = rawAt(idx)[cg.thread_rank()];
      }
      else
      {
        byte = m_compData[idx + cg.thread_rank()];
      }

      uint32_t mask = warpBallot(byte != 0xff);
      mask = __brev(mask);

      const position_type fullBytes = __clz(mask);

      if (fullBytes < WARP_SIZE_U)
      {
        return fullBytes * 0xff + __shfl_sync(WARP_ALL, byte, fullBytes);
      }
      else
      {
        return WARP_SIZE_U * 0xff;
      }
    }
  }
#endif

  template <bool CORRECTNESS_CHECK>
  inline __device__ position_type
  readLSIC(position_type &idx, LZ4CorrectnessChecker<CORRECTNESS_CHECK> *correctness_checker) const
  {
#ifdef WARP_READ_LSIC
    position_type num = 0;
    while (true)
    {
      const position_type block = queryLSIC(idx);
      num += block;

      if (block < WARP_SIZE_U * 0xff)
      {
        idx += (block / 0xff) + 1;
        break;
      }
      else
      {
        idx += WARP_SIZE_U;
      }
    }
    return num;
#else
    using Checker = LZ4CorrectnessChecker<CORRECTNESS_CHECK>;
    position_type num = 0;
    uint8_t next = 0xff;
    // read from the buffer
    while (next == 0xff && idx < end())
    {
      // Check for OOB access
      if (Checker::errorIfTrue(
            idx >= m_length,
            LZ4CorrectnessErrorTypes::INVALID_LSIC,
            idx,
            0,
            __LINE__,
            __func__,
            correctness_checker
          ))
      {
        return 0;
      }
      next = rawAt(idx)[0];
      ++idx;
      // Check for overflow
      if (Checker::errorIfTrue(
            num + next < num,
            LZ4CorrectnessErrorTypes::INVALID_LSIC,
            idx,
            0,
            __LINE__,
            __func__,
            correctness_checker
          ))
      {
        return 0;
      }
      num += next;
    }
    // read from global memory
    while (next == 0xff)
    {
      // Check for OOB access
      if (Checker::errorIfTrue(
            idx >= m_length,
            LZ4CorrectnessErrorTypes::INVALID_LSIC,
            idx,
            0,
            __LINE__,
            __func__,
            correctness_checker
          ))
      {
        return 0;
      }
      next = m_compData[idx];
      ++idx;
      // Check for overflow
      if (Checker::errorIfTrue(
            num + next < num,
            LZ4CorrectnessErrorTypes::INVALID_LSIC,
            idx,
            0,
            __LINE__,
            __func__,
            correctness_checker
          ))
      {
        return 0;
      }
      num += next;
    }
    return num;
#endif
  }

  inline const __device__ uint8_t *raw() const { return m_buffer; }

  inline const __device__ uint8_t *rawAt(const position_type i) const { return raw() + (i - begin()); }

  inline __device__ uint8_t operator[](const position_type i) const
  {
    if (i < end())
    {
      return rawAt(i)[0];
    }
    else
    {
      return m_compData[i];
    }
  }

  inline __device__ void setAndAlignOffset(const position_type offset)
  {
    static_assert(sizeof(size_t) == sizeof(const uint8_t *), "Size of pointer must be equal to size_t.");

    const uint8_t *const alignedPtr = reinterpret_cast<const uint8_t *>(
      (reinterpret_cast<size_t>(m_compData + offset) / sizeof(double_word_type)) * sizeof(double_word_type)
    );

    // Note:
    // Even if the argument `offset` is 0, m_offset can become non-zero
    // AND/OR potentially negative, because the required LZ4 decompression
    // input alignment is 1 byte.
    // A negative m_offset would translate to buffer underaddressing, if
    // we used 8-byte reads unconditionally.
    m_offset = alignedPtr - m_compData;
  }

  template <bool potentially_underaddressing, typename CG>
  inline __device__ void loadAt(const position_type offset, CG &cg)
  {
    setAndAlignOffset(offset);

    do
    {
      // Note: Never touched since IMPLICIT_BUFFER_ALIGNMENT is set
      if constexpr (potentially_underaddressing)
      {
        if (m_offset < 0)
        {
          loadWithNegativeOffset(cg);
          break;
        }
      }
      loadWithNonNegativeOffset(cg);
    } while (0);

    cg.sync();
  }

  inline __device__ position_type begin() const { return m_offset; }

  inline __device__ position_type end() const { return m_offset + DECOMP_INPUT_BUFFER_SIZE; }

private:
  // Handling loads where the read offset is negative
  template <typename CG>
  inline __device__ void loadWithNegativeOffset(CG &cg)
  {
    int64_t rd_offset = m_offset;
    constexpr position_type bytes_to_load = DECOMP_INPUT_BUFFER_SIZE - sizeof(double_word_type);

    // Note:
    // We'd need to load the first few valid bytes (out of 8) to align the next
    // read location to 8 bytes without underaddressing the input buffer
    int64_t rd_ix = rd_offset + cg.thread_rank();
    if (cg.thread_rank() < sizeof(double_word_type))
    {
      m_buffer[cg.thread_rank()] = rd_ix >= 0 && rd_ix < m_length ? m_compData[rd_ix] : 0x00;
    }
    rd_offset += sizeof(double_word_type);

    // Can we fill the entire shared memory buffer so that we remain within the chunk?
    if (rd_offset + bytes_to_load <= m_length)
    {
      assert(reinterpret_cast<size_t>(m_compData + rd_offset) % sizeof(double_word_type) == 0);
      assert(DECOMP_INPUT_BUFFER_SIZE == WARP_SIZE_U * sizeof(double_word_type));
      const double_word_type *const word_data = reinterpret_cast<const double_word_type *>(m_compData + rd_offset);
      double_word_type *const word_buffer = reinterpret_cast<double_word_type *>(m_buffer + sizeof(double_word_type));
      // Note: We might want to increase the shared memory buffer size to 256+8 bytes,
      //       so that this branch can be eliminated. It might be unnecessary though,
      //       given this is used only once.
      if (cg.thread_rank() < (bytes_to_load / sizeof(double_word_type)))
      {
        word_buffer[cg.thread_rank()] = word_data[cg.thread_rank()];
      }
    }
    else
    {
      const uint8_t *byte_data = m_compData + rd_offset;
      uint8_t *byte_buffer = m_buffer + sizeof(double_word_type);
#pragma unroll
      for (int i = cg.thread_rank(); i < bytes_to_load; i += WARP_SIZE)
      {
        if ((rd_offset + i) < m_length)
        {
          byte_buffer[i] = byte_data[i];
        }
      }
    }
  }

  // Handling loads where the read offset is non-negative
  // Note: when IMPLICIT_BUFFER_ALIGNMENT is defined, we are only relying on this
  //       loading variant, as no underaddressing can happen.
  template <typename CG>
  inline __device__ void loadWithNonNegativeOffset(CG &cg)
  {
    if (m_offset + DECOMP_INPUT_BUFFER_SIZE <= m_length)
    {
      assert(reinterpret_cast<size_t>(m_compData + m_offset) % sizeof(double_word_type) == 0);
      assert(DECOMP_INPUT_BUFFER_SIZE == WARP_SIZE_U * sizeof(double_word_type));
      const double_word_type *const word_data = reinterpret_cast<const double_word_type *>(m_compData + m_offset);
      double_word_type *const word_buffer = reinterpret_cast<double_word_type *>(m_buffer);
      word_buffer[cg.thread_rank()] = word_data[cg.thread_rank()];
    }
    else
    {
#pragma unroll
      for (int i = cg.thread_rank(); i < DECOMP_INPUT_BUFFER_SIZE; i += WARP_SIZE)
      {
        if (m_offset + i < m_length)
        {
          m_buffer[i] = m_compData[m_offset + i];
        }
      }
    }
  }

  int64_t m_offset;
  const position_type m_length;
  uint8_t *const m_buffer;
  const uint8_t *const m_compData;
}; // End BufferControl Class

inline __device__ position_type hash(const word_type key, position_type hash_table_size)
{
  // needs to be 12 bits
  return (__brev(key) + (key ^ 0xc375)) & (hash_table_size - 1);
}

inline __device__ uint8_t encodePair(const uint8_t t1, const uint8_t t2) { return ((t1 & 0x0f) << 4) | (t2 & 0x0f); }

inline __device__ token_type decodePair(const uint8_t num)
{
  return token_type{static_cast<uint8_t>((num & 0xf0) >> 4), static_cast<uint8_t>(num & 0x0f)};
}

template <int BLOCK_SIZE, typename CG>
inline __device__ void copyLiterals(uint8_t *const dest, const uint8_t *const source, const position_type length, CG &cg)
{
  assert(BLOCK_SIZE == cg.size());
  for (position_type i = cg.thread_rank(); i < length; i += BLOCK_SIZE)
  {
    dest[i] = source[i];
  }
}

constexpr __host__ __device__ size_t divRoundUp(size_t x, size_t y) { return (x + y - 1) / y; }

template <typename T, typename CG>
inline __device__ position_type lengthOfMatch(
  const T *const data,
  const position_type prev_location,
  const position_type next_location,
  const position_type length,
  CG &cg
)
{
  assert(prev_location < next_location);

  constexpr position_type min_ending_literals = divRoundUp(MIN_ENDING_LITERALS_BYTES, sizeof(T));

  position_type match_length = length - next_location - min_ending_literals;
  for (position_type j = 0; j + next_location + min_ending_literals < length; j += cg.size())
  {
    const position_type i = cg.thread_rank() + j;
    int no_matches = i + next_location + min_ending_literals < length
                       ? (data[prev_location + i] != data[next_location + i])
                       : 1;
    no_matches = warpBallot(no_matches);
    if (no_matches)
    {
      match_length = j + __clz(__brev(no_matches));
      break;
    }
  }

  return match_length;
}

template <typename T>
inline __device__ position_type convertIdx(const offset_type offset, const position_type pos)
{
  // We store (offset % TYPED_HASH_OFFSET_INTERVAL) in the hash table
  constexpr const position_type TYPED_HASH_OFFSET_INTERVAL = (MAX_OFFSET + 1) / sizeof(T);

  assert(offset <= pos);

  // Computing "realPos" as the position of the possible match, where we assume that
  // the offset is the shortest possible given all possible values that could lead to the same
  // (offset % TYPED_HASH_OFFSET_INTERVAL) result
  position_type realPos = (pos / TYPED_HASH_OFFSET_INTERVAL) * TYPED_HASH_OFFSET_INTERVAL + offset;
  if (realPos >= pos)
  {
    realPos -= TYPED_HASH_OFFSET_INTERVAL;
  }

  assert(realPos < pos);

  return realPos;
}

template <typename T>
inline __device__ bool isValidHash(
  const T *const data,
  const offset_type *const hashTable,
  const position_type key,
  const position_type hashPos,
  const position_type decomp_idx,
  position_type &offset
)
{
  constexpr const position_type MAX_TYPED_OFFSET = (MAX_OFFSET + 1) / sizeof(T) - 1;
  const offset_type hashed_offset = hashTable[hashPos];

  if (hashed_offset == NULL_OFFSET)
  {
    return false;
  }

  offset = convertIdx<T>(hashed_offset, decomp_idx);

  if (decomp_idx - offset > MAX_TYPED_OFFSET)
  {
    // can't match current position, ahead, or NULL_OFFSET
    return false;
  }

  const word_type hashKey = readWord<word_type>(data + offset);

  if (hashKey != key)
  {
    return false;
  }

  return true;
}

template <int BLOCK_SIZE, typename CG>
inline __device__ void writeSequenceData(
  uint8_t *const compData,
  const uint8_t *const decompData,
  const token_type token,
  const offset_type offset,
  const position_type decomp_idx,
  position_type &comp_idx,
  CG &cg
)
{
  assert(token.num_matches == 0 || token.num_matches >= 4);

  // -> add token
  if (cg.thread_rank() == 0)
  {
    compData[comp_idx] = encodePair(token.numLiteralsForHeader(), token.numMatchesForHeader());
  }
  ++comp_idx;

  // -> add literal length
  const position_type literalEncodingLength = token.lengthOfLiteralEncoding();
  if (literalEncodingLength)
  {
    writeLSIC<BLOCK_SIZE, CG>(compData + comp_idx, token.numLiteralsOverflow(), cg);
    comp_idx += literalEncodingLength;
  }

  // -> add literals
  copyLiterals<BLOCK_SIZE, CG>(compData + comp_idx, decompData + decomp_idx, token.num_literals, cg);
  comp_idx += token.num_literals;

  // -> add offset
  if (token.num_matches > 0)
  {
    assert(offset > 0);

    writeWord(compData + comp_idx, offset);
    comp_idx += sizeof(offset);

    // -> add match length
    if (token.hasNumMatchesOverflow())
    {
      writeLSIC<BLOCK_SIZE, CG>(compData + comp_idx, token.numMatchesOverflow(), cg);
      comp_idx += token.lengthOfMatchEncoding();
    }
  }
}

inline __device__ int numValidThreadsToMask(const int numValidThreads)
{
  return WARP_ALL >> (WARP_SIZE - numValidThreads);
}

template <typename T, typename CG>
inline __device__ void insertHashTableWarp(
  offset_type *hashTable,
  const position_type hashTableSize,
  const offset_type pos,
  const word_type next,
  const int numValidThreads,
  CG &cg
)
{
  position_type hashPos = hash(next, hashTableSize);

  if (cg.thread_rank() < numValidThreads)
  {
    const int match = __match_any_sync(numValidThreadsToMask(numValidThreads), hashPos);
    if (!match || 31 - __clz(match) == cg.thread_rank())
    {
      // I'm the last match -- can insert
      constexpr const position_type TYPED_HASH_OFFSET_INTERVAL = (MAX_OFFSET + 1) / sizeof(T);
      hashTable[hashPos] = pos % TYPED_HASH_OFFSET_INTERVAL;
    }
  }

  cg.sync();
}

template <typename ALIGNMENT>
inline __device__ word_type shuffleLiterals(word_type literals);

/**
 * Each thread needs a four byte word, but only separated by sizeof(ALIGNMENT)
 * e.g.: for two threads, the five bytes [ 0x12 0x34 0x56 0x78 0x9a ] would
 * be assigned as [0x78563412 0x9a785634 ] to the two threads
 * (little-endian). That means when reading 32 bytes, we can only fill
 * the first 29 thread's 4-byte words.
 */
template <>
inline __device__ word_type shuffleLiterals<uint8_t>(word_type literals)
{
  // collect first byte
  word_type next = literals;
  // collect second byte
  next |= __shfl_down_sync(WARP_ALL, next, 1) << 8;
  // collect third and fourth bytes
  next |= __shfl_down_sync(WARP_ALL, next, 2) << 16;
  return next;
}

/**
 * We shuffle on 16-bit alignment, so six bytes [ 0x12 0x34 0x56 0x78 0x9a 0x0b ]
 * would be assigned [0x78563412 0x0b9a7856 ] to the two threads. We only fill
 * the first 31 threads because the 32'nd thread wouldn't have a complete 4 byte
 * word to process.
 */
template <>
inline __device__ word_type shuffleLiterals<uint16_t>(word_type literals)
{
  // collect first and second bytes
  word_type next = literals;
  // collect third and fourth byte
  next |= __shfl_down_sync(WARP_ALL, next, 1) << 16;
  return next;
}

/**
 * We shuffle on 32-bit alignment, so since each thread already reads 32-bits
 * we don't need to shuffle - so this function is no-op.
 */
template <>
inline __device__ word_type shuffleLiterals<uint32_t>(word_type literals)
{
  return literals;
}

template <typename T, typename CG>
__device__ void compressStream(
  uint8_t *compData,
  const T *decompData,
  offset_type *const hashTable,
  const position_type hash_table_size,
  const position_type length,
  size_t *comp_length,
  CG &cg
)
{
  assert(cg.size() == LZ4_COMP_THREADS_PER_CHUNK);

  static_assert(sizeof(T) <= 4, "Max alignment support is 4 bytes");

  static_assert(
    LZ4_COMP_THREADS_PER_CHUNK <= WARP_SIZE,
    "Compression can be done with at "
    "most one warp"
  );

  position_type decomp_idx = 0;
  position_type comp_idx = 0;
  const position_type typed_length = divRoundUp(length, sizeof(T));

  for (position_type i = cg.thread_rank(); i < hash_table_size; i += LZ4_COMP_THREADS_PER_CHUNK)
  {
    hashTable[i] = NULL_OFFSET;
  }

  cg.sync();

  constexpr position_type last_valid_match = divRoundUp(LAST_VALID_MATCH_BYTES, sizeof(T));
  constexpr position_type min_ending_literals = divRoundUp(MIN_ENDING_LITERALS_BYTES, sizeof(T));

  // otherwise ceil of typed_length can result in illegal memory access
  static_assert(last_valid_match > 0, "Must be rounded up");
  static_assert(min_ending_literals > 0, "Must be rounded up");

  while (decomp_idx < typed_length)
  {
    const position_type tokenStart = decomp_idx;
    while (true)
    {
      if (decomp_idx + last_valid_match >= typed_length)
      {
        // jump to end
        decomp_idx = typed_length;

        // no match -- literals to the end
        token_type tok;
        tok.num_literals = length - (tokenStart * sizeof(T));
        tok.num_matches = 0;
        // TODO: write sizeof(T) aligned sequences, needs padding and
        // decompressor to be aware of alignment
        writeSequenceData<LZ4_COMP_THREADS_PER_CHUNK, CG>(
          compData,
          reinterpret_cast<const uint8_t *>(decompData),
          tok,
          0,
          tokenStart * sizeof(T),
          comp_idx,
          cg
        );
        break;
      }

      // begin adding tokens to the hash table until we find a match
      word_type next = 0;
      if (decomp_idx + min_ending_literals + cg.thread_rank() < typed_length)
      {
        next = decompData[decomp_idx + cg.thread_rank()];
      }

      next = shuffleLiterals<T>(next);

      // the number of threads which won't have enough literals - read
      // shuffleLiterals comment for more details.
      constexpr int invalid_threads = 3 / sizeof(T);
      // if we're at the end of the data, mark them as inactive.
      const int numValidThreads = min(
        static_cast<int>(LZ4_COMP_THREADS_PER_CHUNK - invalid_threads),
        static_cast<int>(typed_length - decomp_idx - last_valid_match)
      );

      // first try to find a local match
      position_type match_location = typed_length;
      int match_mask_self = 0;
      if (cg.thread_rank() < numValidThreads)
      {
        match_mask_self = __match_any_sync(numValidThreadsToMask(numValidThreads), next);
      }

      // each thread has a mask of other threads with matches, next we need
      // to find the first thread with a match before it
      const int match_mask_warp = warpBallot(match_mask_self && __clz(__brev(match_mask_self)) != cg.thread_rank());

      int first_match_thread;
      if (match_mask_warp)
      {
        // find the byte offset (thread id) within the warp where the first
        // match is located
        first_match_thread = __clz(__brev(match_mask_warp));

        // determine the global position for the finding thread
        match_location = __clz(__brev(match_mask_self)) + decomp_idx;
        // comunicate the global position of the match to other threads
        match_location = __shfl_sync(WARP_ALL, match_location, first_match_thread);
      }
      else
      {
        first_match_thread = numValidThreads;
      }

      {
        // go to hash table for an earlier match
        position_type hashPos = hash(next, hash_table_size);
        word_type offset = decomp_idx;
        const int match_found =
          cg.thread_rank() < first_match_thread
            ? isValidHash<T>(decompData, hashTable, next, hashPos, decomp_idx + cg.thread_rank(), offset)
            : 0;

        // determine the first thread to find a match
        const int match = warpBallot(match_found);
        const int candidate_first_match_thread = __clz(__brev(match));

        assert(candidate_first_match_thread != cg.thread_rank() || match_found);
        assert(!match_found || candidate_first_match_thread <= cg.thread_rank());

        if (candidate_first_match_thread < first_match_thread)
        {
          // if we found a valid match, and it occurs before a previously found
          // match, use that
          first_match_thread = candidate_first_match_thread;
          match_location = __shfl_sync(WARP_ALL, offset, first_match_thread);
        }
      }

      if (match_location != typed_length)
      {
        // insert up to the match into the hash table
        insertHashTableWarp<T>(hashTable, hash_table_size, decomp_idx + cg.thread_rank(), next, first_match_thread, cg);

        const position_type pos = decomp_idx + first_match_thread;
        assert(match_location < pos);
        assert(pos - match_location <= MAX_OFFSET);

        // we found a match
        const offset_type match_offset = pos - match_location;
        assert(match_offset > 0);
        assert(match_offset <= pos);
        const position_type num_literals = pos - tokenStart;

        // compute match length
        const position_type num_matches = lengthOfMatch(decompData, match_location, pos, typed_length, cg);

        // -> write our token and literal length
        token_type tok;
        tok.num_literals = num_literals * sizeof(T);
        tok.num_matches = num_matches * sizeof(T);
        // update our position
        decomp_idx = tokenStart + num_matches + num_literals;

        // insert only the literals into the hash table
        writeSequenceData<LZ4_COMP_THREADS_PER_CHUNK, CG>(
          compData,
          reinterpret_cast<const uint8_t *>(decompData),
          tok,
          match_offset * sizeof(T),
          tokenStart * sizeof(T),
          comp_idx,
          cg
        );
        break;
      }

      // insert everything into hash table
      insertHashTableWarp<T>(hashTable, hash_table_size, decomp_idx + cg.thread_rank(), next, numValidThreads, cg);

      decomp_idx += numValidThreads;
    }
  }

  if (cg.thread_rank() == 0)
  {
    // An empty input should be compressed to a single zero byte
    // See https://github.com/lz4/lz4/blob/dev/doc/lz4_Block_format.md#end-of-block-conditions .
    if (length == 0)
    {
      comp_idx = 1;
      compData[0] = 0;
    }
    *comp_length = static_cast<size_t>(comp_idx);
  }
}

template <
  bool CORRECTNESS_CHECK,
  typename CG,
#ifndef NDEBUG
  bool OOB_CHECKING = true>
#else
  bool OOB_CHECKING = false>
#endif
inline __device__ void decompressStream(
  lz4::LZ4DecompressWarpMemory &shared_mem,
  uint8_t *decompData,
  const uint8_t *compData,
  const position_type comp_end, // chunk length procesed by the warp (note, that each warp processes an individual chunk)
  const position_type buf_end,
  size_t *decompSize,
  nvcompStatus_t *decompStatus, // Can be null if called from nvCOMPDx
  bool output_decompressed,
  CG &cg,
  LZ4CorrectnessChecker<CORRECTNESS_CHECK> *correctness_checker
)
{
  assert(cg.size() == LZ4_DECOMP_THREADS_PER_CHUNK);
  BufferControl ctrl(shared_mem.buffer.data(), compData, comp_end);
#ifdef IMPLICIT_BUFFER_ALIGNMENT
  // Note: with an implicit buffer alignment guarantee we don't have to worry about underaddressing a given buffer,
  //       because it is guaranteed by cudaMalloc/cudaMallocAsync/etc. that it is going to be 128/256-byte aligned.
  //       The pointer we receive might not be well-aligned for an 8-byte load, but we are definitely not going
  //       outside of buffer boundaries.
  ctrl.loadAt<false>(0, cg);
#else
  ctrl.loadAt<true>(0, cg);
#endif // IMPLICIT_BUFFER_ALIGNMENT
  position_type decomp_idx = 0;
  position_type comp_idx = 0; // where we are in the chunk

  uint8_t seq_index = 0;
  const int t = cg.thread_rank();

  using Checker = LZ4CorrectnessChecker<CORRECTNESS_CHECK>;
  Checker::Initialize(&cg, correctness_checker);

  bool corrupted_sequence = Checker::checkEmptyChunk(comp_end, *ctrl.rawAt(0), __LINE__, __func__, correctness_checker);

  const auto logResult = [&]() {
    if (t == 0)
    {
      decompSize[0] = corrupted_sequence ? 0 : decomp_idx;
      if (output_decompressed && decompStatus != nullptr)
      {
        decompStatus[0] = corrupted_sequence ? nvcompErrorCannotDecompress : nvcompSuccess;
      }
    }
  };

  if (corrupted_sequence)
  {
    logResult();
    return;
  }

  while (comp_idx < comp_end)
  {
    // Decoded token information for given thread t
    unsigned int ix_literal_t = 0;
    sequence sequence_t{};

    while (seq_index < WARP_SIZE_U)
    {
      if (comp_idx + DECOMP_BUFFER_PREFETCH_DIST > ctrl.end())
      {
        ctrl.loadAt<false>(comp_idx, cg);
      }

      // read token byte
      // never OOB since chunk is at least 1 byte long.
      token_type tok = decodePair(*ctrl.rawAt(comp_idx));
      ++comp_idx;

      // read the length of the literals
      position_type num_literals = tok.num_literals;
      if (tok.num_literals == 15)
      {
        num_literals += ctrl.readLSIC<CORRECTNESS_CHECK>(comp_idx, correctness_checker);
        if (Checker::hasError(correctness_checker))
        {
          corrupted_sequence = true;
          logResult();
          return;
        }
      }

      // prevent OOB access when decompressing non-lz4 streams
      // Note this will never be hit when calculating the decomp size
      // since buf_end == UINT_MAX
      if constexpr (OOB_CHECKING or CORRECTNESS_CHECK)
      {
        if ((decomp_idx + num_literals > buf_end) or (comp_idx + num_literals > comp_end))
        {
          corrupted_sequence = true;
          Checker::setError(
            LZ4CorrectnessErrorTypes::LITERAL_SEQUENCE_TOO_LONG,
            comp_idx,
            decomp_idx,
            __LINE__,
            __func__,
            correctness_checker
          );
          logResult();
          return;
        }
      }

      // save the literals sequence info to the buffer
      if (output_decompressed)
      {
        if (t == seq_index)
        {
          ix_literal_t = comp_idx;
          sequence_t.literal_length = num_literals;
          shared_mem.ix_output[seq_index] = decomp_idx;
        }
      }

      comp_idx += num_literals;
      decomp_idx += num_literals;

      // Note that the last sequence stops right after literals field.
      // There are specific parsing rules to respect to be compatible with the
      // reference decoder : 1) The last 5 bytes are always literals 2) The last
      // match cannot start within the last 12 bytes. Consequently, a file with
      // less then 13 bytes can only be represented as literals. These rules are in
      // place to benefit speed and ensure buffer limits are never crossed.
      // This means that for a valid compressed buffer, it cannot happen that
      // comp_idx < comp_end at this point, but becomes comp_idx >= comp_end
      // after reading the offset and match length fields.
      // Therefore, we only need to check here.
      if (comp_idx < comp_end)
      {

        // read the offset
        offset_type offset;

        // Check for OOB access when reading offset
        corrupted_sequence = Checker::errorIfTrue(
          comp_idx + sizeof(offset_type) > comp_end,
          LZ4CorrectnessErrorTypes::UNEXPECTED_END_OF_CHUNK,
          comp_idx,
          decomp_idx,
          __LINE__,
          __func__,
          correctness_checker
        );
        // Check that the match does not start within the last 12 bytes
        Checker::setMatchStart(decomp_idx, correctness_checker);

        if (corrupted_sequence)
        {
          logResult();
          return;
        }
        offset = readWord<offset_type>(compData + comp_idx);

        comp_idx += sizeof(offset_type);

        // read the match length
        position_type match = 4 + tok.num_matches;
        if (tok.num_matches == 15)
        {
          match += ctrl.readLSIC<CORRECTNESS_CHECK>(comp_idx, correctness_checker);
          if (Checker::hasError(correctness_checker))
          {
            corrupted_sequence = true;
            logResult();
            return;
          }
        }
        cg.sync();

        corrupted_sequence = Checker::errorIfTrue(
          comp_idx >= comp_end,
          LZ4CorrectnessErrorTypes::INVALID_LAST_SEQUENCE,
          comp_idx,
          decomp_idx,
          __LINE__,
          __func__,
          correctness_checker
        );

        // According to the LZ4 block format, the presence of a 0 offset value denotes an invalid (corrupted) block.
        if constexpr (OOB_CHECKING or CORRECTNESS_CHECK)
        {
          if (decomp_idx < offset or decomp_idx + match > buf_end or offset == 0)
          {
            corrupted_sequence = true;
            Checker::setError(
              LZ4CorrectnessErrorTypes::INVALID_MATCH,
              comp_idx,
              decomp_idx,
              __LINE__,
              __func__,
              correctness_checker
            );
          }
        }
        if (corrupted_sequence)
        {
          logResult();
          return;
        }
        if (output_decompressed)
        {
          if (t == seq_index)
          {
            sequence_t.distance = offset;
            sequence_t.match_length = match;
          }
          seq_index++;
        }

        decomp_idx += match;
      }
      else
      {
        if (output_decompressed)
        {
          // The last sequence must contain only literals, and be at least 5-bytes long (excluding token). If the input is less than 5 bytes, the first and only sequence contains all bytes as literals.
          corrupted_sequence = Checker::errorIfTrue(
            (comp_end > MIN_ENDING_LITERALS_BYTES) and (num_literals < MIN_ENDING_LITERALS_BYTES),
            LZ4CorrectnessErrorTypes::LAST_LITERAL_SEQUENCE_TOO_SHORT,
            comp_idx,
            decomp_idx,
            __LINE__,
            __func__,
            correctness_checker
          );
          if (corrupted_sequence)
          {
            logResult();
            return;
          }

          // That the last block's match length (since there can't be a match) should be 0 is not enforced by the standard.

          if (t == seq_index)
          {
            sequence_t.distance = 0;
            sequence_t.match_length = 0;
          }
          seq_index++;
        }
        // If we are here, it means that we have reached the end of the stream,
        // and since we don't check for that at the beginning of the inner while loop,
        // we must break out of it here.
        break;
      }
    }
    // No sync is required here because the specific thread
    // will read the content which is only written by itself.
    if (output_decompressed)
    {
      bool active = t < seq_index;
      bool is_last = comp_idx >= comp_end;
      unsigned distance = active ? sequence_t.distance : 1;
      unsigned match_length = active ? sequence_t.match_length : 0;
      unsigned literal_length = active ? sequence_t.literal_length : 0;

      do_literal_copies(compData, ix_literal_t, literal_length, active, shared_mem.ix_output[t], decompData, is_last);

      // Note, that the last sequence cannot end on a match copy,
      // hence we don't need to guard against vectorized loads potentially
      // overaddressing the input buffer.
      do_match_copies<false>(match_length, distance, active, shared_mem.ix_output, decompData, seq_index, false);
      seq_index = 0;
    }
  }
  corrupted_sequence = corrupted_sequence or Checker::errorIfTrue(
                                               comp_idx != comp_end,
                                               LZ4CorrectnessErrorTypes::UNEXPECTED_END_OF_CHUNK,
                                               comp_idx,
                                               decomp_idx,
                                               __LINE__,
                                               __func__,
                                               correctness_checker
                                             );

  corrupted_sequence = corrupted_sequence or
                       Checker::checkLastMatchValidity(comp_idx, decomp_idx, __LINE__, __func__, correctness_checker);

  logResult();
}

} // namespace nvcomp
