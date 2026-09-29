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

#include <cub/cub.cuh>

#include <cassert>
#include <cstdint>

#include "../common.h"
#include "common.h"
#include "gdeflate_constants.h"

namespace gdeflate
{

using word_type = uint32_t;

// This restricts us to 4GB chunk sizes (total buffer can be up to
// max(size_t)). We actually artificially restrict it to much less, to
// limit what we have to test, as well as to encourage users to exploit some
// parallelism.
using position_type = uint32_t;
using double_word_type = uint64_t;
using item_type = uint32_t;

/**
 * @brief The number of threads to use per chunk in compression.
 */
constexpr const int COMP_THREADS_PER_CHUNK = WARP_SIZE;

/**
 * @brief The number of elements in the hash table to use while performing
 * compression. Ref zlib.
 */
constexpr const position_type HASH_TABLE_SIZE = 1U << 14;
constexpr const position_type HASH_TABLE_SIZE_WITH_CHAIN = 1U << 15;

constexpr const uint8_t MAX_CHAIN = 4;
constexpr const uint8_t GOOD_LEN = 8;
constexpr const uint16_t TOO_FAR = 4096;

template <typename offset_type>
struct hash_chain
{
  uint8_t head;
  offset_type chain[MAX_CHAIN];
};

/**
 * @brief The value used to explicitly represent and invalid offset. This
 * denotes an empty slot in the hashtable.
 */
template <typename offset_type>
constexpr const offset_type NULL_OFFSET = static_cast<offset_type>(-1);

/**
 * @brief The maximum size of a valid offset.
 */
constexpr const position_type MAX_OFFSET = (1U << 15) - 1;

/**
 * @brief The maximum length of a valid match.
 */
constexpr const position_type MAX_MATCH_LENGTH = 258; // TODO: update to include deflate64
constexpr const uint8_t min_match = 3;
//Same as zlib
constexpr const uint8_t hash_shift = (15 + min_match - 1) / min_match;
/**
 * @brief The maximum size of an uncompressed chunk.
 */
constexpr const size_t MAX_CHUNK_SIZE = 1U << 16; // 64 KB

inline __device__ int numValidThreadsToMask(const int numValidThreads)
{
  assert(numValidThreads > 0 && numValidThreads <= 32);
  return WARP_ALL >> (WARP_SIZE - numValidThreads);
}

inline __device__ int warpBallot(int vote) { return __ballot_sync(WARP_ALL, vote); }

struct token_type
{
  position_type num_literals;
  position_type num_matches;
};

//For deflate default (algorithm 0) compression algorithm
//Faster than the defalte algorithm 3
inline __device__ position_type hash(const word_type key)
{
  // needs to be 12 bits
  return (__brev(key) + (key ^ 0xc375)) & (HASH_TABLE_SIZE - 1);
}

//For defalte algorithm 3, slower than the algorithm 0
//but has around 14% higher compresstion ratio
inline __device__ position_type hash_zlib(uint32_t ins_h, uint32_t key_next)
{
  return (((ins_h << hash_shift) ^ key_next) & (HASH_TABLE_SIZE_WITH_CHAIN - 1));
}

template <typename offset_type>
inline __device__ position_type convertIdx(const offset_type offset, const position_type pos, bool &repeat)
{
  constexpr const position_type OFFSET_SIZE = MAX_OFFSET + 1;

  // Note: During testing this never happens
  if (offset > pos)
  {
    repeat = true;
    return 0;
  }

  position_type realPos = (pos / OFFSET_SIZE) * OFFSET_SIZE + offset;

  if (realPos == pos)
  {
    repeat = true;
    return realPos;
  }

  if (realPos >= pos)
  {
    realPos -= OFFSET_SIZE;
  }

  assert(realPos < pos);

  return realPos;
}

template <typename offset_type>
inline __device__ position_type convertIdx(const offset_type offset, const position_type pos)
{
  constexpr const position_type OFFSET_SIZE = MAX_OFFSET + 1;

  assert(offset <= pos);

  position_type realPos = (pos / OFFSET_SIZE) * OFFSET_SIZE + offset;

  if (realPos >= pos)
  {
    realPos -= OFFSET_SIZE;
  }

  assert(realPos < pos);

  return realPos;
}

template <typename T>
inline __device__ T readWord(const uint8_t *const address)
{
  T word = 0;
  for (size_t i = 0; i < sizeof(T); ++i)
  {
    word |= address[i] << (8 * i);
  }

  return word;
}

template <uint8_t MAX_CHAIN_L, typename offset_type>
inline __device__ bool isValidHash(
  const uint8_t *const data,
  const hash_chain<offset_type> hashed_offset,
  const position_type key,
  const position_type decomp_idx,
  int *offset,
  uint32_t &offset_id
)
{
  bool is_valid = false;
  for (int i = 0; i < MAX_CHAIN_L; i++)
  {
    if (hashed_offset.chain[i] == NULL_OFFSET<offset_type>)
    {
      break;
    }

    bool repeat = false;
    int cur_offset = convertIdx(hashed_offset.chain[i], decomp_idx, repeat);

    if (decomp_idx - cur_offset > MAX_OFFSET || repeat)
    {
      // can't match current position, ahead, or NULL_OFFSET
      continue;
    }

    const word_type hashKey = readWord<word_type>(data + cur_offset);

    if (hashKey != key)
    {
      continue;
    }

    is_valid = true;
    offset[i] = cur_offset;
    if (offset_id == 0)
    {
      offset_id = i + 1; //to not conflict with i=0 case.
    }
  }
  //lower 16 bits save the info of the first match position in the offset buffer.
  //higher 16 bits save the head info.
  offset_id = offset_id | (hashed_offset.head << 16);
  return is_valid;
}

template <typename offset_type>
inline __device__ bool isValidHash(
  const uint8_t *const data,
  const offset_type hashed_offset,
  const position_type key,
  const position_type decomp_idx,
  position_type &offset
)
{
  if (hashed_offset == NULL_OFFSET<offset_type>)
  {
    return false;
  }

  offset = convertIdx(hashed_offset, decomp_idx);

  if (decomp_idx - offset > MAX_OFFSET)
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
template <uint8_t MAX_CHAIN_L, typename offset_type>
inline __device__ void insertHashTableWarp(
  hash_chain<offset_type> *hashTable,
  const offset_type pos,
  const word_type key,
  const word_type key_next,
  const int numValidThreads
)
{
  position_type hashPos = hash_zlib(key, key_next);

  if (threadIdx.x < numValidThreads)
  {
    const int match = __match_any_sync(numValidThreadsToMask(numValidThreads), hashPos);
    if (!match || 31 - __clz(match) == threadIdx.x)
    {
      // I'm the last match -- can insert
      hash_chain<offset_type> *chain = &hashTable[hashPos];
      chain->chain[chain->head] = pos & MAX_OFFSET;
      chain->head = (chain->head + 1) & (MAX_CHAIN_L - 1);
    }
  }
  __syncwarp();
}

template <typename offset_type>
inline __device__ void
insertHashTableWarp(offset_type *hashTable, const offset_type pos, const word_type next, const int numValidThreads)
{
  position_type hashPos = hash(next);

  if (threadIdx.x < numValidThreads)
  {
    const int match = __match_any_sync(numValidThreadsToMask(numValidThreads), hashPos);
    if (!match || 31 - __clz(match) == threadIdx.x)
    {
      // I'm the last match -- can insert
      hashTable[hashPos] = pos & MAX_OFFSET;
    }
  }

  __syncwarp();
}
template <uint8_t MAX_CHAIN_L, typename offset_type>
inline __device__ position_type lengthOfMatch(
  const uint8_t *const data,
  int *offsets,
  uint8_t &head,
  offset_type &match_offset,
  int first_match_thread,
  const position_type next_location,
  const position_type length,
  int prev_length = 0
)
{
  position_type longest_match_length = 0;
  position_type match_length;
  position_type final_match_len = 0;
  uint8_t chain_length = MAX_CHAIN_L;

  // Note: in practice prev_length is never set, and so this never happens
  if (prev_length >= GOOD_LEN)
  {
    chain_length >>= 2;
  }
  // int best_len = min_match;
  //Search for the longest match within the chain
  //Start with the nearest offset.
  for (int ci = 0; ci < chain_length; ci++)
  {
    head = (head - 1) & (chain_length - 1);
    int prev_location = __shfl_sync(WARP_ALL, offsets[head], first_match_thread);
    if (prev_location < 0)
    {
      continue;
    }
    assert(prev_location < next_location);
    position_type max_match_length = min(length - next_location - 5, MAX_MATCH_LENGTH);
    match_length = max_match_length;

    for (position_type j = 0; j < max_match_length; j += blockDim.x)
    {
      const position_type i = threadIdx.x + j;
      int match = i < max_match_length ? (data[prev_location + i] != data[next_location + i]) : 1;
      match = warpBallot(match);
      if (match)
      {
        match_length = j + __clz(__brev(match));
        if (match_length > longest_match_length)
        {
          longest_match_length = match_length;
          match_offset = next_location - prev_location;
        }
        break;
      }
    }
    final_match_len = (longest_match_length ? longest_match_length : match_length);
  }
  return final_match_len;
}

inline __device__ position_type lengthOfMatch(
  const uint8_t *const data,
  const position_type prev_location,
  position_type next_location,
  const position_type length
)
{
  assert(prev_location < next_location);

  position_type max_match_length = min(length - next_location - 5, MAX_MATCH_LENGTH);
  position_type match_length = max_match_length;
  for (position_type j = 0; j < max_match_length; j += blockDim.x)
  {
    const position_type i = threadIdx.x + j;
    int match = i < max_match_length ? (data[prev_location + i] != data[next_location + i]) : 1;
    match = warpBallot(match);
    if (match)
    {
      match_length = j + __clz(__brev(match));
      break;
    }
  }

  return match_length;
}

template <int BLOCK_SIZE, typename offset_type>
inline __device__ void writeSequenceData(
  uint8_t *const literals,
  offset_type *const length,
  offset_type *const distance,
  const uint8_t *const decompData,
  const token_type token,
  const offset_type offset,
  const position_type decomp_idx,
  unsigned int &nliterals,
  unsigned int &nsymbols
)
{
  assert(token.num_matches == 0 || token.num_matches >= min_match);

  // Copy literals to the literal stream
  for (position_type i = threadIdx.x; i < token.num_literals; i += BLOCK_SIZE)
  {
    literals[nliterals + i] = decompData[decomp_idx + i];
    length[nsymbols + i] = 1;
    distance[nsymbols + i] = nliterals + i;
  }
  nliterals += token.num_literals;
  nsymbols += token.num_literals;

  // -> add offset
  if (token.num_matches > 0)
  {
    assert(offset > 0);

    if (threadIdx.x == 0)
    {
      length[nsymbols] = token.num_matches;
      distance[nsymbols] = offset;
    }
    nsymbols++;
  }
}

template <uint8_t MAX_CHAIN_L, typename offset_type, bool HASH_WITH_CHAIN = true>
__device__ void lz_compress_tile_greedy_hash_with_chain_fast(
  const unsigned char *decompData,
  const position_type decompBytes,
  hash_chain<offset_type> *const hashTable,
  offset_type *length,
  offset_type *distance,
  uint8_t *literals,
  unsigned int *num_symbols,
  unsigned int *num_literals
)
{
  assert(blockDim.x == COMP_THREADS_PER_CHUNK);
  static_assert(
    COMP_THREADS_PER_CHUNK <= 32,
    "Compression can be done with at "
    "most one warp"
  );

  position_type decomp_idx = 0;

  unsigned int nsymbols = 0;
  unsigned int nliterals = 0;

  // Initialize the hash table
  for (position_type i = threadIdx.x; i < HASH_TABLE_SIZE_WITH_CHAIN; i += COMP_THREADS_PER_CHUNK)
  {
    hashTable[i].head = 0;
    for (int j = 0; j < MAX_CHAIN_L; j++)
    {
      hashTable[i].chain[j] = NULL_OFFSET<offset_type>;
    }
  }

  __syncwarp();

  while (decomp_idx < decompBytes)
  {
    const position_type tokenStart = decomp_idx;
    while (true)
    {
      if (decomp_idx + 5 + 4 >= decompBytes)
      {
        // jump to end
        decomp_idx = decompBytes;

        // no match -- literals to the end
        token_type tok;
        tok.num_literals = decompBytes - tokenStart;
        tok.num_matches = 0;
        writeSequenceData<COMP_THREADS_PER_CHUNK>(
          literals,
          length,
          distance,
          decompData,
          tok,
          static_cast<offset_type>(0),
          tokenStart,
          nliterals,
          nsymbols
        );
        break;
      }

      // begin adding tokens to the hash table until we find a match
      uint8_t byte = 0;
      if (decomp_idx + 5 + threadIdx.x < decompBytes)
      {
        byte = decompData[decomp_idx + threadIdx.x];
      }
      // each thread needs a four byte word, but only separated by a byte e.g.:
      // for two threads, the five bytes [ 0x12 0x34 0x56 0x78 0x9a ] would
      // be assigned as [0x78563412 0x9a785634 ] to the two threads
      // (little-endian). That means when reading 32 bytes, we can only fill
      // the first 29 thread's 4-byte words.
      word_type cur_word = byte;
      // collect second byte
      cur_word |= __shfl_down_sync(WARP_ALL, byte, 1) << 8;
      // collect third and fourth bytes
      cur_word |= __shfl_down_sync(WARP_ALL, cur_word, 2) << 16;

      //reduce the 4 bytes min match to 3 bytes, keep it for further tunning
      // word_type word3 = cur_word & 0xFFFFFF;
      // since we do not have valid data for the last 3 threads (or more if
      // we're at the end of the data), mark them as inactive.
      const int numValidThreads =
        min(static_cast<int>(COMP_THREADS_PER_CHUNK - 2), static_cast<int>(decompBytes - decomp_idx - 9));

      // first try to find a local match
      position_type match_location = decompBytes;
      int match_mask_self = 0;
      if (threadIdx.x < numValidThreads)
      {
        match_mask_self = __match_any_sync(numValidThreadsToMask(numValidThreads), cur_word);
      }

      // each thread has a mask of other threads with matches, next we need
      // to find the first thread with a match before it
      const int match_mask_warp = warpBallot(match_mask_self && __clz(__brev(match_mask_self)) != threadIdx.x);

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

      //better than one byte shift.
      //The next next byte is a better input for the hash function
      //comparing with the next byte.
      uint8_t next_byte = (cur_word >> 16) & 255;
      position_type hashPos = hash_zlib(byte, next_byte);

      //first four bytes represent chain's head pos
      //last four bytes represet the match pos.
      uint32_t offset_id = 0;
      int offset[MAX_CHAIN_L];
      for (int i = 0; i < MAX_CHAIN_L; i++)
      {
        offset[i] = -1;
      }

      int match_found = threadIdx.x < first_match_thread ? isValidHash<MAX_CHAIN_L, offset_type>(
                                                             decompData,
                                                             hashTable[hashPos],
                                                             cur_word,
                                                             decomp_idx + threadIdx.x,
                                                             offset,
                                                             offset_id
                                                           )
                                                         : 0;
      // determine the first thread to find a match
      const int match = warpBallot(match_found);
      const int candidate_first_match_thread = __clz(__brev(match));

      assert(candidate_first_match_thread != threadIdx.x || match_found);
      assert(!match_found || candidate_first_match_thread <= threadIdx.x);
      if (candidate_first_match_thread < first_match_thread)
      {

        // if we found a valid match, and it occurs before a previously found
        // match, use that
        first_match_thread = candidate_first_match_thread;
        offset_id = __shfl_sync(WARP_ALL, offset_id, first_match_thread);
        // //mask out the first match postion in the offset buffer.
        position_type first_match_location = offset[(offset_id & 0xFFFF) - 1];
        match_location = __shfl_sync(WARP_ALL, first_match_location, first_match_thread);
      }

      //there is a match
      if (match_location != decompBytes)
      {
        const position_type pos = decomp_idx + first_match_thread;
        assert(match_location < pos);
        assert(pos - match_location <= MAX_OFFSET);

        // we found a match
        offset_type match_offset = pos - match_location;
        const position_type num_lits = pos - tokenStart;

        // compute match length
        position_type num_matches = 0;
        //mask out the head info
        uint8_t head_info = offset_id >> 16;
        //check if there is a valid hash found.
        if (offset_id & 0xFFFF)
        {
          num_matches =
            lengthOfMatch<MAX_CHAIN_L>(decompData, offset, head_info, match_offset, first_match_thread, pos, decompBytes);
        }
        else
        {
          assert(match_offset > 0);
          assert(match_offset <= pos);
          num_matches = lengthOfMatch(decompData, match_location, pos, decompBytes);
        }
        // -> write our token and literal length
        token_type tok;
        tok.num_literals = num_lits;
        tok.num_matches = num_matches;

        insertHashTableWarp<MAX_CHAIN_L, offset_type>(
          hashTable,
          decomp_idx + threadIdx.x,
          byte,
          next_byte,
          min(numValidThreads, first_match_thread + num_matches)
        );

        // update our position
        decomp_idx = tokenStart + num_matches + num_lits;
        // insert only the literals into the hash table
        writeSequenceData<COMP_THREADS_PER_CHUNK>(
          literals,
          length,
          distance,
          decompData,
          tok,
          match_offset,
          tokenStart,
          nliterals,
          nsymbols
        );
        break;
      }

      // insert everything into hash table
      insertHashTableWarp<MAX_CHAIN_L, offset_type>(
        hashTable,
        decomp_idx + threadIdx.x,
        byte,
        next_byte,
        numValidThreads
      );

      decomp_idx += numValidThreads;
    }
  }

  if (threadIdx.x == 0)
  {
    *num_literals = nliterals;
    *num_symbols = nsymbols;
  }
  __syncwarp();
}

template <typename offset_type>
__device__ void lz_compress_tile_greedy_hash(
  const unsigned char *decompData,
  const position_type decompBytes,
  offset_type *const hashTable,
  offset_type *length,
  offset_type *distance,
  uint8_t *literals,
  unsigned int *num_symbols,
  unsigned int *num_literals
)
{
  assert(blockDim.x == COMP_THREADS_PER_CHUNK);
  static_assert(
    COMP_THREADS_PER_CHUNK <= 32,
    "Compression can be done with at "
    "most one warp"
  );

  position_type decomp_idx = 0;

  unsigned int nsymbols = 0;
  unsigned int nliterals = 0;

  // Initialize the hash table
  for (position_type i = threadIdx.x; i < HASH_TABLE_SIZE; i += COMP_THREADS_PER_CHUNK)
  {
    hashTable[i] = NULL_OFFSET<offset_type>;
  }
  __syncwarp();

  while (decomp_idx < decompBytes)
  {
    const position_type tokenStart = decomp_idx;
    while (true)
    {
      if (decomp_idx + 5 + 4 >= decompBytes)
      {
        // jump to end
        decomp_idx = decompBytes;

        // no match -- literals to the end
        token_type tok;
        tok.num_literals = decompBytes - tokenStart;
        tok.num_matches = 0;
        writeSequenceData<COMP_THREADS_PER_CHUNK>(
          literals,
          length,
          distance,
          decompData,
          tok,
          static_cast<offset_type>(0),
          tokenStart,
          nliterals,
          nsymbols
        );
        break;
      }

      // begin adding tokens to the hash table until we find a match
      uint8_t byte = 0;
      if (decomp_idx + 5 + threadIdx.x < decompBytes)
      {
        byte = decompData[decomp_idx + threadIdx.x];
      }

      // each thread needs a four byte word, but only separated by a byte e.g.:
      // for two threads, the five bytes [ 0x12 0x34 0x56 0x78 0x9a ] would
      // be assigned as [0x78563412 0x9a785634 ] to the two threads
      // (little-endian). That means when reading 32 bytes, we can only fill
      // the first 29 thread's 4-byte words.
      word_type next = byte;
      // collect second byte
      next |= __shfl_down_sync(WARP_ALL, byte, 1) << 8;
      // collect third and fourth bytes
      next |= __shfl_down_sync(WARP_ALL, next, 2) << 16;

      // since we do not have valid data for the last 3 threads (or more if
      // we're at the end of the data), mark them as inactive.
      const int numValidThreads =
        min(static_cast<int>(COMP_THREADS_PER_CHUNK - 3), static_cast<int>(decompBytes - decomp_idx - 9));

      // first try to find a local match
      position_type match_location = decompBytes;
      int match_mask_self = 0;
      if (threadIdx.x < numValidThreads)
      {
        match_mask_self = __match_any_sync(numValidThreadsToMask(numValidThreads), next);
      }

      // each thread has a mask of other threads with matches, next we need
      // to find the first thread with a match before it
      const int match_mask_warp = warpBallot(match_mask_self && __clz(__brev(match_mask_self)) != threadIdx.x);

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
        position_type hashPos = hash(next);
        position_type offset = decomp_idx;
        const int match_found =
          threadIdx.x < first_match_thread
            ? isValidHash<offset_type>(decompData, hashTable[hashPos], next, decomp_idx + threadIdx.x, offset)
            : 0;
        // determine the first thread to find a match
        const int match = warpBallot(match_found);
        const int candidate_first_match_thread = __clz(__brev(match));

        assert(candidate_first_match_thread != threadIdx.x || match_found);
        assert(!match_found || candidate_first_match_thread <= threadIdx.x);

        if (candidate_first_match_thread < first_match_thread)
        {
          // if we found a valid match, and it occurs before a previously found
          // match, use that
          first_match_thread = candidate_first_match_thread;
          match_location = __shfl_sync(WARP_ALL, offset, first_match_thread);
        }
      }

      if (match_location != decompBytes)
      {
        // insert up to the match into the hash table
        insertHashTableWarp(hashTable, static_cast<offset_type>(decomp_idx + threadIdx.x), next, first_match_thread);

        const position_type pos = decomp_idx + first_match_thread;
        assert(match_location < pos);
        assert(pos - match_location <= MAX_OFFSET);

        // we found a match
        const offset_type match_offset = pos - match_location;
        assert(match_offset > 0);
        assert(match_offset <= pos);
        const position_type num_lits = pos - tokenStart;

        // compute match length
        const position_type num_matches = lengthOfMatch(decompData, match_location, pos, decompBytes);

        // -> write our token and literal length
        token_type tok;
        tok.num_literals = num_lits;
        tok.num_matches = num_matches;

        // update our position
        decomp_idx = tokenStart + num_matches + num_lits;

        // insert only the literals into the hash table
        writeSequenceData<COMP_THREADS_PER_CHUNK>(
          literals,
          length,
          distance,
          decompData,
          tok,
          match_offset,
          tokenStart,
          nliterals,
          nsymbols
        );
        break;
      }

      // insert everything into hash table
      insertHashTableWarp(hashTable, static_cast<offset_type>(decomp_idx + threadIdx.x), next, numValidThreads);

      decomp_idx += numValidThreads;
    }
  }

  if (threadIdx.x == 0)
  {
    *num_literals = nliterals;
    *num_symbols = nsymbols;
  }
  __syncwarp();
}

template <typename offset_type>
__global__ void lz_compress_greedy_hash(
  const unsigned char *const *input_ptrs,
  const size_t *input_bytes,
  offset_type *hash_tables,
  offset_type **length_ptrs,
  offset_type **distance_ptrs,
  uint8_t **literal_ptrs,
  unsigned int *num_symbols,
  unsigned int *num_literals,
  const unsigned int num_tiles_host,
  const int *device_num_tiles
)
{
  const unsigned int num_tiles = device_num_tiles ? static_cast<unsigned int>(*device_num_tiles) : num_tiles_host;
  unsigned int stride = gridDim.x * blockDim.y;
  for (unsigned int bid = blockIdx.x * blockDim.y + threadIdx.y; bid < num_tiles; bid += stride)
  {
    lz_compress_tile_greedy_hash(
      input_ptrs[bid],
      (position_type)input_bytes[bid],
      hash_tables + bid * HASH_TABLE_SIZE,
      length_ptrs[bid],
      distance_ptrs[bid],
      literal_ptrs[bid],
      num_symbols + bid,
      num_literals + bid
    );
  }
}

template <typename offset_type>
__global__ void lz_compress_greedy_hash_with_chain(
  const unsigned char *const *input_ptrs,
  const size_t *input_bytes,
  hash_chain<offset_type> *hash_tables,
  offset_type **length_ptrs,
  offset_type **distance_ptrs,
  uint8_t **literal_ptrs,
  unsigned int *num_symbols,
  unsigned int *num_literals,
  const unsigned int num_tiles_host,
  const int *device_num_tiles
)
{
  const unsigned int num_tiles = device_num_tiles ? static_cast<unsigned int>(*device_num_tiles) : num_tiles_host;
  unsigned int stride = gridDim.x * blockDim.y;
  for (unsigned int bid = blockIdx.x * blockDim.y + threadIdx.y; bid < num_tiles; bid += stride)
  {
    lz_compress_tile_greedy_hash_with_chain_fast<MAX_CHAIN, offset_type>(
      input_ptrs[bid],
      (position_type)input_bytes[bid],
      hash_tables + bid * HASH_TABLE_SIZE_WITH_CHAIN,
      length_ptrs[bid],
      distance_ptrs[bid],
      literal_ptrs[bid],
      num_symbols + bid,
      num_literals + bid
    );
  }
}

} // namespace gdeflate
