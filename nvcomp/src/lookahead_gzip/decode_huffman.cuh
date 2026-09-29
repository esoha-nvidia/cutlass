/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

// Functions related to the decoding of Huffman streams into O2OVectors.

#pragma once

#include "huffman_table.cuh"
#include "lookback.cuh"
#include "offset_to_overlap.cuh"
#include "ring_buffer.cuh"

#include <stdint.h>

namespace decode_huffman
{

// Write the results of the decode into an overlap structure.
template <typename overlap_base_t>
inline __device__ void set_copy(
  UncompressedSizeOverlap<overlap_base_t> *overlap,
  const uint iteration_offset,
  const uint bits_decoded,
  const uint uncompressed_size,
  const uint huffman_tree_index
)
{
  overlap->set_copy(huffman_tree_index, iteration_offset + bits_decoded);
  overlap->set_uncompressed_size(uncompressed_size);
}

// Write the number of bits beyond the MAX_OFFSET as an overlap.
template <typename overlap_base_t>
inline __device__ void set_todo(
  UncompressedSizeOverlap<overlap_base_t> *overlap,
  const uint iteration_offset,
  const uint bits_decoded,
  const uint uncompressed_size,
  const uint huffman_tree_index
)
{
  overlap->set_overlap(huffman_tree_index, iteration_offset + bits_decoded - MAX_OFFSET);
  overlap->set_uncompressed_size(uncompressed_size);
}

template <typename overlap_base_t>
inline __device__ void todo_to_overlap(UncompressedSizeOverlap<overlap_base_t> *overlap, const size_t stride_bits)
{
  overlap->set_overlap(overlap->get_huffman_tree_index(), overlap->get_overlap() - (stride_bits - MAX_OFFSET));
  // Uncompressed size is unchanged.
}

// Write the results of the decode into an overlap structure.
template <typename overlap_base_t>
inline __device__ void set_overlap(
  UncompressedSizeOverlap<overlap_base_t> *overlap,
  const bool error_or_end_of_block,
  const uint iteration_offset,
  const uint bits_decoded,
  const uint32_t uncompressed_size,
  const uint huffman_tree_index,
  const size_t stride_bits
)
{
  if (error_or_end_of_block)
  {
    // If it's an error, bits_decoded has a 0 already.  If
    // it's an end of block then bits_decoded has the number of bits
    // decoded, which is what we want for an end-of-block anyway.  We
    // must set the MSB for these situations.
    overlap->set_error_or_end_of_block(iteration_offset, bits_decoded);
  }
  else
  {
    // Not an error so we need to report the overlap and also if we
    // need to now decode a length code.
    overlap->set_overlap(huffman_tree_index, iteration_offset + bits_decoded - stride_bits);
  }
  overlap->set_uncompressed_size(uncompressed_size);
}

/* This will walk from the root of the Huffman tree to a symbol.  */
inline __device__ uint16_t walk_huffman_tree(
  const HuffmanTable<MAX_SYMBOL_BITS, LITLEN_COUNT> *huffman_trees,
  uint &huffman_tree_index,
  const uint32_t bit_buffer,
  unsigned int &bits_decoded,
  bool &error_or_end_of_block,
  uint &uncompressed_size,
  uint16_t &distance
)
{
  uint32_t bit_buffer_reversed = __brev(bit_buffer);
  const auto &current_huffman = huffman_trees[huffman_tree_index];
  const auto bit_length = current_huffman.bits_to_length(bit_buffer_reversed);
  const auto symbols_offset = current_huffman.get_offset(bit_length, bit_buffer_reversed);
  const bool invalid_offset = symbols_offset >= current_huffman.get_symbols_count();
  // Avoid over-indexing in get_symbol (symbols[]) and in get_extra_bits (EXTRA_BITS[table][symbol])
  // by falling back to the end-of-block symbol.
  const auto symbol = invalid_offset ? static_cast<uint16_t>(256) : current_huffman.get_symbol(symbols_offset);
  __builtin_assume(huffman_tree_index < 2);
  const auto extra_bits = invalid_offset ? 0u : get_extra_bits(huffman_tree_index, symbol);
  uncompressed_size += invalid_offset ? 0u : get_uncompressed_size(huffman_tree_index, symbol);
  if (huffman_tree_index == 1 && !invalid_offset)
  {
    distance = get_distance_base(symbol);
    distance += (bit_buffer >> bit_length) & ((1 << extra_bits) - 1);
  }
  huffman_tree_index = (symbol > 256) ? 1 : 0;
  bits_decoded += invalid_offset ? 0u : (bit_length + extra_bits);
  error_or_end_of_block = (symbol == 256);
  if (symbol > 256)
  {
    uncompressed_size += (bit_buffer >> bit_length) & ((1 << extra_bits) - 1);
  }
  return symbol;
}

template <bool prune, bool BOUNDS_CHECK = true, typename deflate_stream_t, class overlap_t>
inline __device__ void decode_one(
  const deflate_stream_t &deflate_stream,
  const size_t stride_bit_offset,
  const HuffmanTable<MAX_SYMBOL_BITS, LITLEN_COUNT> *huffmans,
  const uint huffman_tree_start_index,
  const uint iteration_offset,
  overlap_t *overlap,
  const size_t stride_bits,
  const size_t data_size = 0
)
{
  // Was the previous code a length?
  uint huffman_tree_index = huffman_tree_start_index;
  // This is the overlap that we want to report.  We will subtract
  // 8*STRIDE_BYTES from it before we store it.  It counts up how
  // many bytes have been processed or skipped due to the
  // iteration_offset.
  unsigned int bits_decoded = 0;
  unsigned int uncompressed_size = 0; // This is in bytes.
  bool error_or_end_of_block = false;
  if (prune)
  {
    // Decode just one to see if we can prune.  We do this outside of
    // the loop so we don't have to do a slow branch inside the while
    // loop on each iteration.
    while (true)
    {
      uint16_t unused_distance;
      uint32_t bit_buffer =
        get_aligned_32_bits<BOUNDS_CHECK>(deflate_stream, stride_bit_offset + bits_decoded, 0, data_size);
      walk_huffman_tree(
        huffmans,
        huffman_tree_index,
        bit_buffer,
        bits_decoded,
        error_or_end_of_block,
        uncompressed_size,
        unused_distance
      );
      // We finished a single decode.  Have we reached a condition
      // where we can set the overlap?
      if (error_or_end_of_block)
      {
        // We hit an error or end of block.  Let's store our result.
        set_overlap(
          overlap,
          error_or_end_of_block,
          iteration_offset,
          bits_decoded,
          uncompressed_size,
          huffman_tree_index,
          stride_bits
        );
        return;
      }
      // Are we at a condition that we can write into our overlap?
      // That is, a valid huffman_tree_index for the offset?
      __builtin_assume(huffman_tree_index < HUFFMAN_TREES_COUNT);
      if (huffman_tree_index == LITLEN_HUFFMAN_INDEX)
      {
        if (bits_decoded + iteration_offset < MAX_OFFSET)
        {
          // If we're pruning then we stop after the first decode if we know
          // that we'll have the answer from somewhere else.
          set_copy(overlap, iteration_offset, bits_decoded, uncompressed_size, huffman_tree_index);
        }
        else
        {
          // This decode went beyond the first MAX_OFFSET bits.  Set it as
          // if it is an overlap but instead of storing how many bits are
          // beyond the stride, we'll store how many bits are beyond the
          // first MAX_OFFSET bits of the iteration.
          set_todo(overlap, iteration_offset, bits_decoded, uncompressed_size, huffman_tree_index);
        }
        return;
      }
      // We need to do more decoding to either find the end of the
      // block or a valid offset for recording in the overlaps array.
    }
  }
  else
  {
    while (true)
    {
      uint32_t bit_buffer =
        get_aligned_32_bits<BOUNDS_CHECK>(deflate_stream, stride_bit_offset + bits_decoded, 0, data_size);
      // Process enough symbols until the overlap is big enough or we
      // hit an error or end of block.
      uint16_t unused_match_distance;
      walk_huffman_tree(
        huffmans,
        huffman_tree_index,
        bit_buffer,
        bits_decoded,
        error_or_end_of_block,
        uncompressed_size,
        unused_match_distance
      );
      // We finished a single decode but might not yet have consumed
      // at least STRIDE_BYTES.
      __builtin_assume(huffman_tree_index < HUFFMAN_TREES_COUNT);
      if (error_or_end_of_block ||
          bits_decoded + iteration_offset >= stride_bits && huffman_tree_index == LITLEN_HUFFMAN_INDEX)
      {
        // Either we've consumed at least STRIDE_BITS bits or we hit an
        // error or end of block.  Let's store our result.
        set_overlap(
          overlap,
          error_or_end_of_block,
          iteration_offset,
          bits_decoded,
          uncompressed_size,
          huffman_tree_index,
          stride_bits
        );
        return; // This is enough decoding.
      }
    }
  }
}

// This stores an element of the output buffer, which is either the
// actual value to output or a length/distance pair from where to
// copy.
class UncompressedWord
{
public:
  // Returns true if a literal is stored here.
  inline __host__ __device__ bool is_literal() const { return data >= 0; }

  // Returns the literal stored within, only valid if is_literal
  // returns true.
  inline __host__ __device__ uint8_t get_literal() const { return data; }

  // Sets the literal.
  inline __host__ __device__ void set_literal(uint8_t literal) { data = literal; }

  // Get distance, only valid if is_literal is false.  The distance is
  // returned as a negative number.
  inline __host__ __device__ int32_t get_distance() const { return data; }

  // Sets a len/distance pair where length must be 1.  Distance should
  // be any negative number that fits in a 32-bit word.  Distance 0 is
  // not valid.
  inline __device__ void set_distance(int32_t distance) { data = distance; }

  inline __device__ void operator=(int32_t new_data) { data = new_data; }

  __device__ __host__ void print()
  {
    if (is_literal())
    {
      if (get_literal() > 0x1f && get_literal() < 0x7f)
      {
        printf("%c (%02x)\n", get_literal(), get_literal());
      }
      else
      {
        printf("  (%02x)\n", get_literal());
      }
    }
    else
    {
      int32_t distance = get_distance();
      printf("copy 1 byte from %d\n", distance);
    }
  }

private:
  int32_t data;
};

// Returns the number of words decompressed.  Each literal is
// decompressed into a 32-bit word.  Each LZ77 copy is decompressed
// into as many 32-bit words as the length of the match.
// deflate_stream is the bytes to decode.  The ring buffer is
// guaranteed to be big enough.  The stride_bit_offset is the offset
// into the deflate_stream for this specific stride.  The huffmans are
// the huffman tables to use for decoding.  huffman_tree_start_index
// indicates which huffman tree to start with.  iteration_offset is
// how many bits into the start of the stride we are, so
// stride_bit_offset - iteration_offset is actually the start of the
// stride.  write_index is a unique number that is used so that
// multiple threads can do the same decoding work.
// uncompressed_shared_words is the target into which to write the
// result.  current_uncompressed_offset is the start offset at which
// decompression should be written into uncompressed_shared_words.
// uncompressed_words_offset is the start bound for beginning writing.
// Any words before that value should not be written.  words_count is
// the total number of words that should be written.  Anything beyond
// uncompressed_words_offset+words_count should not be written out.
// Those last two are only used if RANGE_CHECK is true.
template <bool RANGE_CHECK, uint MAX_WRITE_INDEX, bool BOUNDS_CHECK = true, typename deflate_stream_t>
inline __device__ void decode_one(
  const deflate_stream_t &deflate_stream,
  const size_t stride_bit_offset,
  const HuffmanTable<MAX_SYMBOL_BITS, LITLEN_COUNT> *huffmans,
  const uint iteration_offset,
  const uint write_index,
  FixedArray<decode_huffman::UncompressedWord, UNCOMPRESSED_SHARED_WORDS_COUNT> &uncompressed_shared_words,
  const size_t current_uncompressed_offset,
  const size_t uncompressed_words_offset,
  const size_t words_count,
  const size_t stride_bits,
  const size_t data_size = 0
)
{
  uint huffman_tree_index = LITLEN_HUFFMAN_INDEX;
  // This is the overlap that we want to report.  We will subtract
  // 8*SRIDE_BYTES from it before we store it.  It counts up how
  // many bytes have been processed or skipped due to the
  // iteration_offset.
  unsigned int bits_decoded = 0;
  unsigned int uncompressed_size = 0; // This is in bytes.
  bool error_or_end_of_block = false;
  if (huffman_tree_index != 0)
  {
    // Skip the first one because it's a distance that will be done by
    // the stride before this one.
    uint16_t unused_match_distance;
    uint32_t bit_buffer =
      get_aligned_32_bits<BOUNDS_CHECK>(deflate_stream, stride_bit_offset + bits_decoded, 0, data_size);
    walk_huffman_tree(
      huffmans,
      huffman_tree_index,
      bit_buffer,
      bits_decoded,
      error_or_end_of_block,
      uncompressed_size,
      unused_match_distance
    );
  }
  uint16_t match_length;
  while (true)
  {
    // Process enough symbols until the overlap is big enough or we
    // hit an error.
    auto old_uncompressed_size = uncompressed_size;
    auto old_huffman_tree_index = huffman_tree_index;
    uint16_t match_distance;
    uint32_t bit_buffer =
      get_aligned_32_bits<BOUNDS_CHECK>(deflate_stream, stride_bit_offset + bits_decoded, 0, data_size);
    auto symbol = walk_huffman_tree(
      huffmans,
      huffman_tree_index,
      bit_buffer,
      bits_decoded,
      error_or_end_of_block,
      uncompressed_size,
      match_distance
    );
    // Write it into the correct place.
    if (old_huffman_tree_index == 0 && symbol < 256)
    {
      // This is a normal literal so just write it.
      if (!RANGE_CHECK ||
          (current_uncompressed_offset + old_uncompressed_size >= uncompressed_words_offset &&
           current_uncompressed_offset + old_uncompressed_size < uncompressed_words_offset + words_count))
      {
        uncompressed_shared_words[current_uncompressed_offset + old_uncompressed_size - uncompressed_words_offset]
          .set_literal(symbol);
      }
      // uncompressed size was already incremented by walk_huffman_tree.
    }
    else if (old_huffman_tree_index == 0 && symbol > 256)
    {
      // We got a length code so hold on to it until the next loop.
      match_length = uncompressed_size - old_uncompressed_size;
      // Undo the increment, we are not ready for it yet.
      uncompressed_size = old_uncompressed_size;
    }
    else if (old_huffman_tree_index != 0)
    {
      auto my_match_length = match_length;
      auto my_start = current_uncompressed_offset + old_uncompressed_size;
      if (RANGE_CHECK)
      {
        if (my_start < uncompressed_words_offset && my_start + my_match_length > uncompressed_words_offset)
        {
          // The start is before the start of the target memory so
          // advance it.
          my_match_length -= uncompressed_words_offset - my_start;
          my_start = uncompressed_words_offset;
        }
        if (my_start + my_match_length >= uncompressed_words_offset + words_count &&
            my_start < uncompressed_words_offset + words_count)
        {
          // The end is after the end of the target memory but that
          // start isn't so bring it back down.
          my_match_length = uncompressed_words_offset + words_count - my_start;
        }
      }

      for (auto i = 0; i < my_match_length / MAX_WRITE_INDEX; i += 1)
      {
        const auto offset = my_start + i * MAX_WRITE_INDEX + write_index;
        if (!RANGE_CHECK || (offset >= uncompressed_words_offset && offset < uncompressed_words_offset + words_count))
        {
          uncompressed_shared_words[offset - uncompressed_words_offset].set_distance(-match_distance);
        }
      }
      // Now do the runt.
      if (write_index < my_match_length % MAX_WRITE_INDEX)
      {
        const auto offset = my_start + my_match_length - my_match_length % MAX_WRITE_INDEX + write_index;
        if (!RANGE_CHECK || (offset >= uncompressed_words_offset && offset < uncompressed_words_offset + words_count))
        {
          uncompressed_shared_words[offset - uncompressed_words_offset].set_distance(-match_distance);
        }
      }

      // Now we can finally advance.
      uncompressed_size += match_length;
    }
    // We finished a single decode but might not yet have consumed
    // at least STRIDE_BYTES or we're in the middle of a match.
    if (error_or_end_of_block || (bits_decoded + iteration_offset >= stride_bits && huffman_tree_index == 0))
    {
      return; // Done decoding this stride.
    }
  }
}

/* Given a deflate stream, which is a stream of huffman encoded bytes
   as specified in RFC 1951, create an offset to overlap mapping for a
   specific 64-bit section of the stream.

   Each thread operates on a different 32-bit offset in the
   deflate_stream.  Given the Huffman tables for the literals/lengths
   and for the symbols, create a mapping from offset to overlap.  The
   offset is the number of bits into the 32-bits being examined that
   might be the correct start of a Huffman code.  The overlap is how
   many bits into the next 32-bits to start the next offset.  That is,
   the overlap is the number of bits consumed if we start decoding
   from the current offset such that the offset plus the overlap is
   the minimum number of bits that need to be decoded to be at least
   32.
*/
template <bool prune = false, bool BOUNDS_CHECK = true, typename deflate_stream_t, class O2OVector>
__device__ void decode_many(
  const deflate_stream_t &deflate_stream,
  const size_t stream_bit_offset,
  const HuffmanTable<MAX_SYMBOL_BITS, LITLEN_COUNT> *huffmans,
  O2OVector total_overlaps[O2O_VECTORS_PER_BLOCK],
  const size_t stride_bits,
  const size_t data_size
)
{
  if (threadIdx.x >= THREADS_PER_O2O_CREATION * O2O_VECTORS_PER_BLOCK)
  {
    return; // Nothing to do for this thread.
  }
  const unsigned int stride_id = threadIdx.x / THREADS_PER_O2O_CREATION;
  // start_length is 0 if we begin decoding literals/lengths.  If it's
  // one then we start by decoding distance.
  // TODO: stride or blockwise?
  for (uint offset_id = threadIdx.x % THREADS_PER_O2O_CREATION; offset_id < MAX_OFFSET;
       offset_id += THREADS_PER_O2O_CREATION)
  {
    const uint huffman_tree_start_index = offset_id / MAX_OFFSET;
    // Offset into this stride, which can be as large as the maximum
    // decode length, which is MAX_OFFSET.
    const unsigned int iteration_offset = offset_id % MAX_OFFSET;
    // Where in the deflate stream should we start looking?
    const auto start_bit_offset = stride_id * stride_bits + stream_bit_offset;
    const auto current_bit_offset = start_bit_offset + iteration_offset;
    decode_one<prune, BOUNDS_CHECK>(
      deflate_stream,
      current_bit_offset,
      huffmans,
      huffman_tree_start_index,
      iteration_offset,
      &total_overlaps[stride_id][offset_id],
      stride_bits,
      data_size
    );
  }
}

} // namespace decode_huffman

namespace lookback
{

template <>
inline __device__ DoWorkResult do_work<FixedRing<decode_huffman::UncompressedWord, UNCOMPRESSED_WORDS_COUNT>>(
  FixedRing<decode_huffman::UncompressedWord, UNCOMPRESSED_WORDS_COUNT> &elements,
  const size_t start,
  const size_t count,
  size_t index,
  decode_huffman::UncompressedWord *mine
)
{
  if (index >= count)
  {
    return NOT_FINISHED_NO_WRITEBACK; // Nothing to do but wait for other threads.
  }
  if (mine->is_literal())
  {
    return FINISHED_NO_WRITEBACK; // Finished here, go to the next element.
  }
  const int32_t mine_distance = mine->get_distance();
  const decode_huffman::UncompressedWord prev = elements[start + index + mine_distance];
  if (prev.is_literal())
  {
    mine->set_literal(prev.get_literal());
    return FINISHED_WRITEBACK;
  }
  // Otherwise, prev is also a distance, so coalesce.
  mine->set_distance(mine_distance + prev.get_distance());
  return NOT_FINISHED_WRITEBACK; // We need to do more work.
}

template <>
inline __device__ DoWorkResult do_work<FixedArray<decode_huffman::UncompressedWord, UNCOMPRESSED_SHARED_WORDS_COUNT>>(
  volatile FixedArray<decode_huffman::UncompressedWord, UNCOMPRESSED_SHARED_WORDS_COUNT> &elements,
  const size_t start,
  const size_t count,
  size_t max_lookback,
  size_t index,
  decode_huffman::UncompressedWord *mine
)
{
  if (index >= count)
  {
    return NOT_FINISHED_NO_WRITEBACK; // Nothing to do but wait for other threads.
  }
  if (mine->is_literal())
  {
    return FINISHED_NO_WRITEBACK; // Finished here, go to the next element.
  }
  const int32_t mine_distance = mine->get_distance();
  if (start + index < -mine_distance)
  {
    // For everything else we need prev and prev won't be valid so
    // we'll move on.
    return FINISHED_NO_WRITEBACK;
  }
  decode_huffman::UncompressedWord prev;
  atomic_copy(&prev, elements[start + index + mine_distance]);
  if (prev.is_literal())
  {
    mine->set_literal(prev.get_literal());
    return FINISHED_WRITEBACK;
  }
  // Otherwise, prev is also a distance.
  if (index + max_lookback < -(mine_distance + prev.get_distance()))
  {
    return FINISHED_NO_WRITEBACK; // This is looking backward too far
    // so don't process it further.
  }
  mine->set_distance(mine_distance + prev.get_distance());
  return NOT_FINISHED_WRITEBACK; // We need to do more work.
}

} // namespace lookback
