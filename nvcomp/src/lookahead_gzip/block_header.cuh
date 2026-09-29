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

#pragma once

// Decode a DEFLATE block header from the very start, creating the
// decoding structure that will be needed for decoding the data.
#include "common.cuh"
#include "constants.cuh"
#include "fixed_array.cuh"
#include "huffman_table.cuh"
#include "lookback.cuh"
#include "mailbox.cuh"
#include "prescan.cuh"
#include "ring_buffer.cuh"

struct symbol_length_t
{
  int16_t value;
};

namespace lookback
{
template <>
inline __device__ DoWorkResult do_work<FixedArray<symbol_length_t, LITLEN_COUNT + DIST_COUNT>>(
  volatile FixedArray<symbol_length_t, LITLEN_COUNT + DIST_COUNT> &elements,
  const size_t start,
  const size_t count,
  const size_t max_lookback,
  const size_t index,
  symbol_length_t *mine
)
{
  if (index >= count)
  {
    return NOT_FINISHED_NO_WRITEBACK;
  }
  if (mine->value >= 0)
  {
    return FINISHED_NO_WRITEBACK;
  }
  // mine->value is negative for back-references (RFC 1951 repeat code 16). A valid repeat always
  // points to an already-decoded earlier length, so the target is < index. On a short header the
  // speculatively over-decoding threads can synthesize a back-reference that points before the start
  // of the array; without this guard the chained resolution reads symbol_lengths at a wrapped
  // (negative) shared index -> out-of-bounds. Treat such a corrupt reference as an unused symbol.
  const size_t back_distance = static_cast<size_t>(-mine->value);
  if (back_distance > index)
  {
    mine->value = 0;
    return FINISHED_WRITEBACK;
  }
  symbol_length_t prev;
  atomic_copy(&prev, elements[index - back_distance]);
  if (prev.value >= 0)
  {
    mine->value = prev.value;
    return FINISHED_WRITEBACK;
  }
  // Otherwise, it was a pointer to a pointer.
  mine->value += prev.value;
  return NOT_FINISHED_WRITEBACK;
}
} // namespace lookback

namespace block_header
{
struct uint128_t
{
  uint128_t() = default;
  // Store the value shifted left by shift_left.
  inline __device__ uint128_t(uint32_t value, uint shift_left)
  {
    // left shift is saturating on the GPU so if the shift is more
    // than the width of uint, we'll get 0.  That's what we want.
    data[0] = value << (shift_left - 0 * (sizeof(*data) * 8));
    data[1] = value << (shift_left - 1 * (sizeof(*data) * 8));
    data[2] = value << (shift_left - 2 * (sizeof(*data) * 8));
    data[3] = value << (shift_left - 3 * (sizeof(*data) * 8));
  }
  inline __device__ uint32_t operator>>(const uint &rhs)
  {
    // "rhs" can be >= 96 when "SWARHistogram::get_element" calls it iwth a distance code length >= 13.
    return get_aligned_32_bits<true>(data, rhs, 0, sizeof(data));
  }
  inline __device__ uint128_t operator+(const uint128_t &rhs)
  {
    uint128_t ret;
    for (uint i = 0; i < 2; i++)
    {
      ret.data64[i] = this->data64[i] + rhs.data64[i];
    }
    return ret;
  }
  inline __device__ uint128_t &operator+=(const uint128_t &rhs)
  {
    for (uint i = 0; i < 2; i++)
    {
      this->data64[i] += rhs.data64[i];
    }
    return *this;
  }
  inline __device__ uint operator&(const uint &rhs) { return data[0] & rhs; }
  inline __device__ void print() { printf("%08x%08x%08x%08x\n", data[3], data[2], data[1], data[0]); }

  template <uint INDEX_COUNT>
  inline __device__ void additive_identity(uint index)
  {
    for (uint i = index; i < 2; i += INDEX_COUNT)
    {
      data64[i] = 0;
    }
  }

private:
  union
  {
    uint32_t data[4];
    uint64_t data64[2];
  };
};

// element 0 should be ignored.  Don't use it.  Shift by 32 ought to
// saturate but it seems like it only saturates sometimes.  Maybe it
// doesn't saturate if it's a constant?
// https://forums.developer.nvidia.com/t/problem-with-left-shift/20660/4
const __device__ uint clc_shifts[] = {31, 0, 2, 5, 9, 14, 19, 24, 29};
const __device__ uint litlen_shifts[] = {128, 0, 2, 5, 9, 14, 20, 32, 40, 49, 64, 73, 82, 96, 105, 114, 123};

// Return a new T that is the value left shifted by left_shift.
template <typename T>
inline __device__ T new_left_shift(const uint value, const uint left_shift)
{
  return value << left_shift;
}

// Return a new T that is the value left shifted by left_shift.
template <>
inline __device__ uint128_t new_left_shift(const uint value, const uint left_shift)
{
  return uint128_t(value, left_shift);
}

// Create a histogram of data using SWAR.  To use, first set each
// element to a value.  Then run prescan, which will convert all the
// elements into the sums of the elements prior.  Then use the getters
// to retrieve those sums.
template <const uint OFFSETS[], typename T, uint COUNT, uint BUCKETS>
class SWARHistogram
{
public:
  constexpr __device__ uint size() { return COUNT; }
  template <uint NEW_COUNT>
  inline __device__ SWARHistogram<OFFSETS, T, NEW_COUNT, BUCKETS> &slice()
  {
    return *reinterpret_cast<SWARHistogram<OFFSETS, T, NEW_COUNT, BUCKETS> *>(this);
  }
  // Set the element at the specified index to the value and all the rest to 0.
  inline __device__ void set_element(const uint index, const uint bucket, const uint value)
  {
    histogram[index] = new_left_shift<T>(value, OFFSETS[bucket]);
  }
  inline __device__ void prescan() { block_prescan<COUNT>(histogram, &this->total); }
  // Return the number inside the bucket of the element specified.
  // After a prescan, this will return the total of the buckets that
  // were originally set, from [0..index).
  inline __device__ uint get_element(uint index, uint bucket)
  {
    uint shift_right_amount = OFFSETS[bucket];
    uint bit_length = OFFSETS[bucket + 1] - shift_right_amount;
    uint mask = (1 << bit_length) - 1;
    return (histogram[index] >> shift_right_amount) & mask;
  }
  // Return the number inside
  inline __device__ uint get_total(uint bucket)
  {
    uint shift_right_amount = OFFSETS[bucket];
    uint bit_length = OFFSETS[bucket + 1] - shift_right_amount;
    uint mask = (1 << bit_length) - 1;
    return (total >> shift_right_amount) & mask;
  }
  __device__ void print()
  {
    for (uint i = 0; i < COUNT; i++)
    {
      printf("%3d: ", i);
      for (uint j = 0; j < BUCKETS; j++)
      {
        printf("%2d ", get_element(i, j));
      }
      printf("\n");
    }
  }

private:
  // The order below is important so that slice continues to work.
  T total;
  T histogram[COUNT];
};

constexpr uint BITS_PER_CODE_LENGTH_CODE = 3;
constexpr uint MAX_CODE_OFFSET = 14;

constexpr uint HLIT_OFFSET = 3; // Skip past BFINAL and the 2-bit block type.
constexpr uint HLIT_BITS = 5;
constexpr uint HDIST_OFFSET = HLIT_OFFSET + HLIT_BITS;
constexpr uint HDIST_BITS = 5;
constexpr uint HCLEN_OFFSET = HDIST_OFFSET + HDIST_BITS;
constexpr uint HCLEN_BITS = 4;
constexpr uint CODE_LENGTH_CODES_OFFSET = HCLEN_OFFSET + HCLEN_BITS;

using block_header_copy_type = uint32_t;

class SymbolsOverlap
{
public:
  // Copy some elements from in to out.
  template <uint INDEX_COUNT>
  inline __device__ void partial_copy(const uint i, const SymbolsOverlap &in)
  {
    static_assert(
      sizeof(*this) % sizeof(block_header_copy_type) == 0,
      "partial_copy requires that the data to copy is a whole "
      "number of block_header_copy_type words."
    );
    constexpr uint work_items_count = sizeof(*this) / sizeof(block_header_copy_type);
    for (unsigned int offset = 0; offset < work_items_count / INDEX_COUNT; offset += 1)
    {
      uint current_index = offset * INDEX_COUNT + i;
      reinterpret_cast<block_header_copy_type *>(this)[current_index] =
        (reinterpret_cast<const block_header_copy_type *>(&in))[current_index];
    }
    // Handle the runt.
    if (work_items_count % INDEX_COUNT != 0 && i < work_items_count % INDEX_COUNT)
    {
      uint current_index = work_items_count / INDEX_COUNT * INDEX_COUNT + i;
      reinterpret_cast<block_header_copy_type *>(this)[current_index] =
        (reinterpret_cast<const block_header_copy_type *>(&in))[current_index];
    }
  }

  template <uint INDEX_COUNT>
  inline __device__ void partial_add(const uint i, const SymbolsOverlap &left, const SymbolsOverlap &right)
  {
    for (unsigned int index = i; index < MAX_CODE_OFFSET; index += INDEX_COUNT)
    {
      if (MAX_CODE_OFFSET % INDEX_COUNT == 0)
      {
        __builtin_assume(index < MAX_CODE_OFFSET);
      }
      if (index < MAX_CODE_OFFSET)
      {
        const auto &this_overlap = left.overlap[index];
        overlap[index] = right.overlap[this_overlap];
        uncompressed_size[index] = left.uncompressed_size[index] + right.uncompressed_size[this_overlap];
      }
    }
  }
  template <uint INDEX_COUNT>
  inline __device__ void additive_identity(uint index)
  {
    for (uint offset = 0; offset < MAX_CODE_OFFSET / INDEX_COUNT; offset++)
    {
      uint i = offset * INDEX_COUNT + index;
      overlap[i] = i;
      uncompressed_size[i] = 0;
    }
    if (MAX_CODE_OFFSET % INDEX_COUNT != 0 && index < MAX_CODE_OFFSET % INDEX_COUNT)
    {
      uint i = MAX_CODE_OFFSET / INDEX_COUNT * INDEX_COUNT;
      overlap[i] = i;
      uncompressed_size[i] = 0;
    }
  }

  inline __device__ void set_overlap(uint offset, uint8_t overlap) { this->overlap[offset] = overlap; }
  inline __device__ uint8_t get_overlap(uint offset) const { return this->overlap[offset]; }
  inline __device__ void set_uncompressed_size(uint offset, uint16_t uncompressed_size)
  {
    this->uncompressed_size[offset] = uncompressed_size;
  }
  inline __device__ uint16_t get_uncompressed_size(uint offset) const { return this->uncompressed_size[offset]; }

private:
  // This will store the overlap into the next stride for each
  // possible offset, of which there are 15.
  uint8_t overlap[MAX_CODE_OFFSET];
  uint16_t uncompressed_size[MAX_CODE_OFFSET];
  uint16_t padding[1];
};

inline __device__ uint16_t walk_codes_huffman_tree2(
  const HuffmanTable<7, 19> &length_codes,
  uint64_t &bit_buffer,
  unsigned int &bits_decoded,
  uint &symbol,
  uint &bit_buffer_available
)
{
  uint32_t bit_buffer_reversed = __brev(bit_buffer);
  const auto bit_length = length_codes.bits_to_length(bit_buffer_reversed);
  const auto symbols_offset = length_codes.get_offset(bit_length, bit_buffer_reversed);

  // Speculative threads decoding from non-aligned bit offsets can produce
  // out-of-range symbol indices. These results are discarded by prescan.
  const bool invalid_offset = symbols_offset >= length_codes.get_symbols_count();
  symbol = invalid_offset ? 0 : length_codes.get_symbol(symbols_offset);
  const auto extra_bits = get_clc_extra_bits(symbol);
  uint uncompressed_size = get_clc_uncompressed_size(symbol);
  if (symbol >= 16)
  {
    uncompressed_size += (bit_buffer >> bit_length) & ((1 << extra_bits) - 1);
  }
  bits_decoded += bit_length + extra_bits;
  bit_buffer >>= bit_length + extra_bits;
  bit_buffer_available -= bit_length + extra_bits;
  return uncompressed_size;
}

// Given a histogram of lengths, make a Huffman decoder.  Each element
// in the histogram is a one-hot representation.  COUNT is the maximum
// possible size of the histogram but only `count` of them are valid.
template <uint THREADS, uint MAX_BITS, uint COUNT, const uint OFFSETS[], typename T>
inline __device__ void lengths_to_huffman(
  SWARHistogram<OFFSETS, T, COUNT, MAX_BITS + 1> *code_lengths_histogram,
  HuffmanTable<MAX_BITS, COUNT> *huffman_tree,
  uint symbol,
  uint length_of_symbol,
  uint count
)
{
  const unsigned int thread_id_in_block = threadIdx.x;
  code_lengths_histogram->prescan();
  sync_warp_or_threads<COUNT>();
  if (thread_id_in_block <= MAX_BITS)
  {
    // This isn't storing "symbols before length" right now, it's just
    // storing the symbols themselves.  We'll need to do a prescan on
    // the array inside the Huffman table to make it right.
    huffman_tree->set_symbols_before_length(thread_id_in_block, code_lengths_histogram->get_total(thread_id_in_block));
  }
  sync_warp_or_threads<COUNT>(); // Need for all threads to agree on
  // symbols_before_length array
  // contents.
  huffman_tree->prescan_symbols_before_length();
  // Now the symbols_before_length is valid and correct and we can use
  // the get_symbols_before_length function.
  cooperative_groups::thread_block::sync();
  __builtin_assume(count <= COUNT);
  constexpr uint threads_for_set_symbol = nvcomp::roundUpDiv(COUNT, WARP_SIZE_U) * WARP_SIZE_U;
  static_assert(
    threads_for_set_symbol >= COUNT,
    "Must have enough threads to set all the symbols at the same "
    "time without a loop."
  );
  if (thread_id_in_block < threads_for_set_symbol)
  {
    // Now we place all the symbols into the correct spot in the
    // symbols array.  Don't write the length 0 elements, they are
    // excluded from the sorted array.
    if (thread_id_in_block < count && length_of_symbol != 0)
    {
      // How many symbols are there with a length less than length_of_symbol?
      const auto number_of_symbols_before_length = huffman_tree->get_symbols_before_length(length_of_symbol);
      // How many symbols with the same length are there before this one?
      const auto symbols_with_same_length_before_symbol = code_lengths_histogram->get_element(symbol, length_of_symbol);
      const auto number_of_symbols_before_symbol = number_of_symbols_before_length +
                                                   symbols_with_same_length_before_symbol;
      // When the compressed input is short, threads decoding past the real
      // end of the block consume uninitialized bits and produce nonsense
      // code-lengths. The resulting histogram prescan can push
      // number_of_symbols_before_symbol past the end of the symbols[] array,
      // causing an out-of-bounds shared-memory write.
      if (number_of_symbols_before_symbol < COUNT)
      {
        huffman_tree->set_symbol(number_of_symbols_before_symbol, symbol);
      }
    }
  }
  else
  {
    static_assert(
      THREADS - threads_for_set_symbol >= MAX_BITS,
      "Number of threads beyond the set symbol threads should be "
      "at least enough for all the min code calculation."
    );
    const uint length_id = thread_id_in_block - threads_for_set_symbol;
    uint min_code = 0;
    if (length_id <= MAX_BITS)
    {
      if (huffman_tree->get_symbols_before_length(length_id) < huffman_tree->get_symbols_count())
      {
        // Setting the loop bounds to length_id is slower, weird!
        for (uint i = 0; i < MAX_BITS; i++)
        {
          if (i < length_id)
          {
            min_code = (min_code + code_lengths_histogram->get_total(i)) * 2;
          }
        }
        min_code <<= (sizeof(uint32_t) * 8 - length_id);
      }
      else
      {
        min_code = 0xffffffff;
      }
      huffman_tree->set_length_to_min_code(length_id, min_code);
    }
  }
}

const __device__ uint8_t code_length_codes_order[] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};

template <const uint CLC_SHIFTS[], const uint LITLEN_SHIFTS[]>
struct BlockHeaderShared
{
  // First there are three bits for final block and what kind of
  // block.  Then 5+5+4 for the lengths of the following sections.
  // Then up to 19 code length codes of 3 bits each.  Then up to 286
  // literal length codes, each one is a symbol with up to 7 bits.
  // Then up to 32 distance codes, again with up to 7 bits each.
  // 3+5+5+4+19*3+286*7+30*7 is 2286 bits, which is less than 72
  // uint32_t.
  SymbolsOverlap symbols_overlaps[72 + 1];
  uint32_t header[72];
  SWARHistogram<CLC_SHIFTS, uint32_t, sizeof(code_length_codes_order), 1 << BITS_PER_CODE_LENGTH_CODE>
    code_lengths_histogram;
  HuffmanTable<(1 << BITS_PER_CODE_LENGTH_CODE) - 1, sizeof(code_length_codes_order)> length_codes;
  //We allocate 1 more word than necessary because we don't want to
  // read off the end of the array where there is no check for it
  // below.  Adding the check would probably be slower than just
  // loading one more 32-bit word.
  uint32_t lengths_header[73];
  // 72 words is the max header plus 1 for the aggregate.
  SymbolsOverlap add_result[72 / 2];
  FixedArray<symbol_length_t, LITLEN_COUNT + DIST_COUNT> symbol_lengths;
  uint end_of_header;
  uint symbol_length_atomic;
  SWARHistogram<LITLEN_SHIFTS, uint128_t, LITLEN_COUNT, 16> litlen_histogram alignas(16);
  SWARHistogram<LITLEN_SHIFTS, uint128_t, DIST_COUNT, 16> dist_histogram alignas(16);
};

// header has the first 72 32-bit words of the block, aligned for
// convenience.  Return the bit offset to the start of the data in the
// block.
static __device__ uint decode_dynamic_huffman(
  const uint32_t *header,
  HuffmanTable<MAX_SYMBOL_BITS, const_max(LITLEN_COUNT, DIST_COUNT)> *huffmans,
  BlockHeaderShared<clc_shifts, litlen_shifts> *shmem
)
{
  const unsigned int thread_id_in_block = threadIdx.x;
  uint hlit = (header[0] >> HLIT_OFFSET & ((1 << HLIT_BITS) - 1)) + 257;
  uint hdist = (header[0] >> HDIST_OFFSET & ((1 << HDIST_BITS) - 1)) + 1;
  uint hclen = (header[0] >> HCLEN_OFFSET & ((1 << HCLEN_BITS) - 1)) + 4;
  SWARHistogram<clc_shifts, uint32_t, sizeof(code_length_codes_order), 1 << BITS_PER_CODE_LENGTH_CODE>
    &code_lengths_histogram = shmem->code_lengths_histogram;
  uint current_bit_position = CODE_LENGTH_CODES_OFFSET;
  uint hclen_position = current_bit_position + BITS_PER_CODE_LENGTH_CODE * thread_id_in_block;
  uint length_of_symbol = 0; // This is the length of the symbol that
  // this thread will work on.
  uint symbol = 0; // This is the symbol that this thread will work on.
  if (thread_id_in_block < sizeof(code_length_codes_order))
  {
    symbol = code_length_codes_order[thread_id_in_block];
    if (thread_id_in_block < hclen)
    {
      // Without checking bounds, as header is 72 bytes of shared memory.
      length_of_symbol = get_aligned_32_bits<false>(header, hclen_position);
      length_of_symbol &= (1 << BITS_PER_CODE_LENGTH_CODE) - 1;
    }
    code_lengths_histogram.set_element(symbol, length_of_symbol, 1);
  }
  // Now we have the lengths for each symbol in the SWARHistogram and
  // each symbol has a value 1 in the histogram bucket corresponding
  // to that symbol's length.
  __syncwarp(); // The histogram fits in 32 threads.
  HuffmanTable<7, 19> &length_codes = shmem->length_codes;
  constexpr uint threads_for_clc_huffman = 256;
  lengths_to_huffman<threads_for_clc_huffman>(&code_lengths_histogram, &length_codes, symbol, length_of_symbol, hclen);
  current_bit_position += hclen * BITS_PER_CODE_LENGTH_CODE;
  // Now we need to shift the header over because otherwise it'll be
  // extra reads when we do the decoding because the stride will
  // sometimes cross three dwords instead of two.
  uint32_t *lengths_header = shmem->lengths_header;
  if (thread_id_in_block >= threads_for_clc_huffman && thread_id_in_block < threads_for_clc_huffman + 74)
  {
    // thread_id_in_block is the header that we're trying to fill
    uint target_word = thread_id_in_block - threads_for_clc_huffman;
    // Without checking bounds, as header is 72 bytes of shared memory.
    lengths_header[target_word] = get_aligned_32_bits<false>(header, current_bit_position, target_word);
  }
  cooperative_groups::thread_block::sync();
  // Now decode the rest of the bits using the huffman tree.
  auto &symbols_overlaps = shmem->symbols_overlaps;
  for (uint i = 0; i < 72 * MAX_CODE_OFFSET; i += BLOCK_HEADER_THREADS_PER_BLOCK)
  {
    const uint work_id = thread_id_in_block + i;
    const uint stride_id = work_id / MAX_CODE_OFFSET;
    if (stride_id >= 72)
    {
      continue;
    }
    const uint start_bit_offset = stride_id * HEADER_STRIDE_BITS;
    const uint iteration_offset = work_id % MAX_CODE_OFFSET;
    uint current_bit_offset = start_bit_offset + iteration_offset;
    uint current_dword_id = current_bit_offset / HEADER_STRIDE_BITS;
    uint64_t bit_buffer = lengths_header[current_dword_id++];
    // We must shift because we might have an offset that doesn't
    // line up with the byte.
    bit_buffer >>= current_bit_offset % HEADER_STRIDE_BITS;
    // This will store the number of bits that are valid in the
    // bit_buffer.  The buffer always has at least 32 bits.
    unsigned int bit_buffer_available = sizeof(*lengths_header) * 8 - (current_bit_offset % HEADER_STRIDE_BITS);
    bit_buffer |= ((uint64_t)(lengths_header[current_dword_id++])) << bit_buffer_available;
    bit_buffer_available += sizeof(*lengths_header) * 8;
    // bit_buffer will now have at least 32 + (32-13) bits
    // available.  That's 51 bits.  The largest possible decode is
    // 14 bits so if we decode all the way to 31 used and then 14
    // more, which is the worst case, that would be just 45 bits,
    // which is comfortably below 51.  So we won't need to load more
    // bits into the bit_buffer.
    unsigned int bits_decoded = 0;
    unsigned int uncompressed_size = 0;
    while (true)
    {
      // Process enough symbols until the overlap is big enough or we
      // hit an error.
      uint symbol;
      uncompressed_size +=
        walk_codes_huffman_tree2(length_codes, bit_buffer, bits_decoded, symbol, bit_buffer_available);
      // We finished a single decode but might not yet have consumed
      // at least HEADER_STRIDE_BYTES.
      if (bits_decoded + iteration_offset >= HEADER_STRIDE_BITS)
      {
        break; // This is enough decoding.
      }
    }
    symbols_overlaps[stride_id].set_overlap(iteration_offset, iteration_offset + bits_decoded - HEADER_STRIDE_BITS);
    symbols_overlaps[stride_id].set_uncompressed_size(iteration_offset, uncompressed_size);
  }
  cooperative_groups::thread_block::sync();

  auto add_result = shmem->add_result;
  block_prescan_kernel<BLOCK_HEADER_THREADS_PER_BLOCK, 72, MAX_CODE_OFFSET / 2, MAX_CODE_OFFSET / 2>(
    symbols_overlaps,
    &symbols_overlaps[72],
    add_result
  );
  cooperative_groups::thread_block::sync();

  // Now we decode the symbols and write them into the right place.
  // We'll zero out the array so the codes 17 and 18 which require
  // writing zeros will be no-ops.  Only the symbols which require
  // copies are complicated.  For copies, we'll write a negative
  // number that indicates how far back to go in the array to find the
  // copy location.  Copies can be of length [3..6] so, at most, we
  // need to write 6 elements -1,-2,-3,-4,-5,-6.  To do this in
  // parallel, we'll use 6 times as many threads.  For fewer copies,
  // those threads will not write anything.  Also, no threads should
  // write beyond the number of lengths that need to be written, which
  // is hclen + hcdist.  We'll put everything into one array for now
  // and later split it into two.

  // This is the maximum number of symbols that can be decoded.
  auto &symbol_lengths = shmem->symbol_lengths;
  // This is the bit position of the first bit in header that we
  // didn't use.
  uint &end_of_header = shmem->end_of_header;
  if (thread_id_in_block < LITLEN_COUNT + DIST_COUNT)
  {
    // Set all to zero so that we don't need to deal with symbols 17
    // and 18, which are just writing zeros, RFC1951 section 3.2.7.
    symbol_lengths[thread_id_in_block].value = 0;
  }
  cooperative_groups::thread_block::sync();
  if (thread_id_in_block < 72 * 6)
  { // times 6 because of the copies
    const uint stride_id = thread_id_in_block / 6;
    const uint copy_offset = thread_id_in_block % 6;
    uint iteration_offset = symbols_overlaps[stride_id].get_overlap(0);
    const uint start_bit_offset = stride_id * HEADER_STRIDE_BITS + iteration_offset;
    uint current_bit_offset = start_bit_offset;
    uint current_dword_id = stride_id;
    uint64_t bit_buffer = lengths_header[current_dword_id++];
    // We must shift because we might have an offset that doesn't
    // line up with the byte.
    bit_buffer >>= current_bit_offset % HEADER_STRIDE_BITS;
    // This will store the number of bits that are valid in the
    // bit_buffer.  The buffer always has at least 32 bits.
    unsigned int bit_buffer_available = 32 - (current_bit_offset % 32);
    bit_buffer |= ((uint64_t)(lengths_header[current_dword_id++])) << bit_buffer_available;
    bit_buffer_available += 32;
    unsigned int bits_decoded = 0;
    // This is the first stride that is actually part of the data and
    // not just bits before the symbol lengths.
    uint current_symbol_offset =
      symbols_overlaps[stride_id].get_uncompressed_size(0); // Where to write the next symbol.
    while (current_symbol_offset < hlit + hdist)
    {
      // Process enough symbols until the overlap is big enough or we
      // hit an error.
      uint old_symbol_offset = current_symbol_offset;
      uint new_symbol;
      current_symbol_offset +=
        walk_codes_huffman_tree2(length_codes, bit_buffer, bits_decoded, new_symbol, bit_buffer_available);
      if (current_symbol_offset == hlit + hdist)
      {
        // We just decoded the very last code of the huffman trees.
        // Record this position so that we can return it later.  Only
        // one thread should ever get here so we ought to be okay.
        // Hopefully there is no weird huffman tree that uses a repeat
        // or zeros that stretch beyond hilt+hdist.
        end_of_header = start_bit_offset + bits_decoded + current_bit_position;
      }
      if (new_symbol < 16 && copy_offset == 0)
      {
        if (old_symbol_offset < hlit + hdist)
        {
          symbol_lengths[old_symbol_offset].value = new_symbol;
        }
      }
      else if (new_symbol == 16)
      {
        // These are copies.  We don't know the value but we can know
        // from where.  Write that index to the location from which to
        // copy.
        if (old_symbol_offset + copy_offset < hlit + hdist && // If it isn't off the end of the array.
            copy_offset < current_symbol_offset - old_symbol_offset)
        { // Number of copies to make.
          symbol_lengths[old_symbol_offset + copy_offset].value = -(copy_offset + 1);
        }
      }
      if (bits_decoded + iteration_offset >= HEADER_STRIDE_BITS)
      {
        break; // This is enough decoding.
      }
    }
  }
  auto &symbol_length_atomic = shmem->symbol_length_atomic;
  if (threadIdx.x == 0)
  {
    symbol_length_atomic = BLOCK_HEADER_THREADS_PER_BLOCK;
  }
  cooperative_groups::thread_block::sync();
  // Now resolve all the items that need copying.  In the beginning,
  // there can be elements that are copying from the one prior, -1.
  // If they point to a literal, we are done.  If they point to
  // another copy, at worst it could be -1.  So the -1 will be
  // converted to a -2.  Now the array will have no -1.  On the next
  // round, worst case, a -2 points to a -2, so we will get -4.  Etc.
  // Eventually the negative value will be bigger than 286+30, which
  // is the maximum length of the array so we must be done in
  // log2(286+30) steps.
  lookback::block_lookback_atomics<BLOCK_HEADER_THREADS_PER_BLOCK>(
    symbol_lengths,
    0,
    hlit + hdist,
    hlit + hdist,
    &symbol_length_atomic
  );
  cooperative_groups::thread_block::sync();
  auto &litlen_histogram = shmem->litlen_histogram;
  auto &dist_histogram = shmem->dist_histogram;

  const uint litlen_symbol_length = thread_id_in_block < hlit ? symbol_lengths[thread_id_in_block].value : 0;
  const uint dist_symbol_length = thread_id_in_block < hdist ? symbol_lengths[thread_id_in_block + hlit].value : 0;
  if (thread_id_in_block < litlen_histogram.size())
  {
    litlen_histogram.set_element(thread_id_in_block, litlen_symbol_length, 1);
  }
  if (thread_id_in_block < dist_histogram.size())
  {
    dist_histogram.set_element(thread_id_in_block, dist_symbol_length, 1);
  }
  cooperative_groups::thread_block::sync();
  // We could do the next two in parallel if we wanted.  The code gets
  // a little more complex but time is saved.  It's only worthwhile if
  // block header becomes the bottleneck.
  lengths_to_huffman<BLOCK_HEADER_THREADS_PER_BLOCK>(
    &litlen_histogram,
    &huffmans[LITLEN_HUFFMAN_INDEX],
    thread_id_in_block,
    litlen_symbol_length,
    hlit
  );

  cooperative_groups::thread_block::sync();
  // RFC 1951 defines only distance codes 0-29.
  if (hdist > DIST_COUNT)
  {
    return 0;
  }
  lengths_to_huffman<BLOCK_HEADER_THREADS_PER_BLOCK>(
    &dist_histogram,
    &(huffmans[DISTANCE_HUFFMAN_INDEX].template slice<30>()),
    thread_id_in_block,
    dist_symbol_length,
    hdist
  );
  return end_of_header;
}

const __device__ uint8_t FIXED_HUFFMANS[sizeof(HuffmanInfo().huffmans)] = {
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x30, 0x00, 0x00,
  0x00, 0xc8, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x18, 0x00, 0x00, 0x00, 0xae, 0x00, 0x00, 0x00, 0x1e, 0x01, 0x00, 0x00, 0x1e, 0x01, 0x00, 0x00, 0x1e, 0x01,
  0x00, 0x00, 0x1e, 0x01, 0x00, 0x00, 0x1e, 0x01, 0x00, 0x00, 0x1e, 0x01, 0x00, 0x00, 0x1e, 0x01, 0x00, 0x00, 0x00,
  0x01, 0x01, 0x01, 0x02, 0x01, 0x03, 0x01, 0x04, 0x01, 0x05, 0x01, 0x06, 0x01, 0x07, 0x01, 0x08, 0x01, 0x09, 0x01,
  0x0a, 0x01, 0x0b, 0x01, 0x0c, 0x01, 0x0d, 0x01, 0x0e, 0x01, 0x0f, 0x01, 0x10, 0x01, 0x11, 0x01, 0x12, 0x01, 0x13,
  0x01, 0x14, 0x01, 0x15, 0x01, 0x16, 0x01, 0x17, 0x01, 0x00, 0x00, 0x01, 0x00, 0x02, 0x00, 0x03, 0x00, 0x04, 0x00,
  0x05, 0x00, 0x06, 0x00, 0x07, 0x00, 0x08, 0x00, 0x09, 0x00, 0x0a, 0x00, 0x0b, 0x00, 0x0c, 0x00, 0x0d, 0x00, 0x0e,
  0x00, 0x0f, 0x00, 0x10, 0x00, 0x11, 0x00, 0x12, 0x00, 0x13, 0x00, 0x14, 0x00, 0x15, 0x00, 0x16, 0x00, 0x17, 0x00,
  0x18, 0x00, 0x19, 0x00, 0x1a, 0x00, 0x1b, 0x00, 0x1c, 0x00, 0x1d, 0x00, 0x1e, 0x00, 0x1f, 0x00, 0x20, 0x00, 0x21,
  0x00, 0x22, 0x00, 0x23, 0x00, 0x24, 0x00, 0x25, 0x00, 0x26, 0x00, 0x27, 0x00, 0x28, 0x00, 0x29, 0x00, 0x2a, 0x00,
  0x2b, 0x00, 0x2c, 0x00, 0x2d, 0x00, 0x2e, 0x00, 0x2f, 0x00, 0x30, 0x00, 0x31, 0x00, 0x32, 0x00, 0x33, 0x00, 0x34,
  0x00, 0x35, 0x00, 0x36, 0x00, 0x37, 0x00, 0x38, 0x00, 0x39, 0x00, 0x3a, 0x00, 0x3b, 0x00, 0x3c, 0x00, 0x3d, 0x00,
  0x3e, 0x00, 0x3f, 0x00, 0x40, 0x00, 0x41, 0x00, 0x42, 0x00, 0x43, 0x00, 0x44, 0x00, 0x45, 0x00, 0x46, 0x00, 0x47,
  0x00, 0x48, 0x00, 0x49, 0x00, 0x4a, 0x00, 0x4b, 0x00, 0x4c, 0x00, 0x4d, 0x00, 0x4e, 0x00, 0x4f, 0x00, 0x50, 0x00,
  0x51, 0x00, 0x52, 0x00, 0x53, 0x00, 0x54, 0x00, 0x55, 0x00, 0x56, 0x00, 0x57, 0x00, 0x58, 0x00, 0x59, 0x00, 0x5a,
  0x00, 0x5b, 0x00, 0x5c, 0x00, 0x5d, 0x00, 0x5e, 0x00, 0x5f, 0x00, 0x60, 0x00, 0x61, 0x00, 0x62, 0x00, 0x63, 0x00,
  0x64, 0x00, 0x65, 0x00, 0x66, 0x00, 0x67, 0x00, 0x68, 0x00, 0x69, 0x00, 0x6a, 0x00, 0x6b, 0x00, 0x6c, 0x00, 0x6d,
  0x00, 0x6e, 0x00, 0x6f, 0x00, 0x70, 0x00, 0x71, 0x00, 0x72, 0x00, 0x73, 0x00, 0x74, 0x00, 0x75, 0x00, 0x76, 0x00,
  0x77, 0x00, 0x78, 0x00, 0x79, 0x00, 0x7a, 0x00, 0x7b, 0x00, 0x7c, 0x00, 0x7d, 0x00, 0x7e, 0x00, 0x7f, 0x00, 0x80,
  0x00, 0x81, 0x00, 0x82, 0x00, 0x83, 0x00, 0x84, 0x00, 0x85, 0x00, 0x86, 0x00, 0x87, 0x00, 0x88, 0x00, 0x89, 0x00,
  0x8a, 0x00, 0x8b, 0x00, 0x8c, 0x00, 0x8d, 0x00, 0x8e, 0x00, 0x8f, 0x00, 0x18, 0x01, 0x19, 0x01, 0x1a, 0x01, 0x1b,
  0x01, 0x1c, 0x01, 0x1d, 0x01, 0x90, 0x00, 0x91, 0x00, 0x92, 0x00, 0x93, 0x00, 0x94, 0x00, 0x95, 0x00, 0x96, 0x00,
  0x97, 0x00, 0x98, 0x00, 0x99, 0x00, 0x9a, 0x00, 0x9b, 0x00, 0x9c, 0x00, 0x9d, 0x00, 0x9e, 0x00, 0x9f, 0x00, 0xa0,
  0x00, 0xa1, 0x00, 0xa2, 0x00, 0xa3, 0x00, 0xa4, 0x00, 0xa5, 0x00, 0xa6, 0x00, 0xa7, 0x00, 0xa8, 0x00, 0xa9, 0x00,
  0xaa, 0x00, 0xab, 0x00, 0xac, 0x00, 0xad, 0x00, 0xae, 0x00, 0xaf, 0x00, 0xb0, 0x00, 0xb1, 0x00, 0xb2, 0x00, 0xb3,
  0x00, 0xb4, 0x00, 0xb5, 0x00, 0xb6, 0x00, 0xb7, 0x00, 0xb8, 0x00, 0xb9, 0x00, 0xba, 0x00, 0xbb, 0x00, 0xbc, 0x00,
  0xbd, 0x00, 0xbe, 0x00, 0xbf, 0x00, 0xc0, 0x00, 0xc1, 0x00, 0xc2, 0x00, 0xc3, 0x00, 0xc4, 0x00, 0xc5, 0x00, 0xc6,
  0x00, 0xc7, 0x00, 0xc8, 0x00, 0xc9, 0x00, 0xca, 0x00, 0xcb, 0x00, 0xcc, 0x00, 0xcd, 0x00, 0xce, 0x00, 0xcf, 0x00,
  0xd0, 0x00, 0xd1, 0x00, 0xd2, 0x00, 0xd3, 0x00, 0xd4, 0x00, 0xd5, 0x00, 0xd6, 0x00, 0xd7, 0x00, 0xd8, 0x00, 0xd9,
  0x00, 0xda, 0x00, 0xdb, 0x00, 0xdc, 0x00, 0xdd, 0x00, 0xde, 0x00, 0xdf, 0x00, 0xe0, 0x00, 0xe1, 0x00, 0xe2, 0x00,
  0xe3, 0x00, 0xe4, 0x00, 0xe5, 0x00, 0xe6, 0x00, 0xe7, 0x00, 0xe8, 0x00, 0xe9, 0x00, 0xea, 0x00, 0xeb, 0x00, 0xec,
  0x00, 0xed, 0x00, 0xee, 0x00, 0xef, 0x00, 0xf0, 0x00, 0xf1, 0x00, 0xf2, 0x00, 0xf3, 0x00, 0xf4, 0x00, 0xf5, 0x00,
  0xf6, 0x00, 0xf7, 0x00, 0xf8, 0x00, 0xf9, 0x00, 0xfa, 0x00, 0xfb, 0x00, 0xfc, 0x00, 0xfd, 0x00, 0xfe, 0x00, 0xff,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1e, 0x00, 0x00, 0x00, 0x1e, 0x00,
  0x00, 0x00, 0x1e, 0x00, 0x00, 0x00, 0x1e, 0x00, 0x00, 0x00, 0x1e, 0x00, 0x00, 0x00, 0x1e, 0x00, 0x00, 0x00, 0x1e,
  0x00, 0x00, 0x00, 0x1e, 0x00, 0x00, 0x00, 0x1e, 0x00, 0x00, 0x00, 0x1e, 0x00, 0x00, 0x00, 0x1e, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x01, 0x00, 0x02, 0x00, 0x03, 0x00, 0x04, 0x00, 0x05, 0x00, 0x06, 0x00, 0x07, 0x00, 0x08, 0x00, 0x09,
  0x00, 0x0a, 0x00, 0x0b, 0x00, 0x0c, 0x00, 0x0d, 0x00, 0x0e, 0x00, 0x0f, 0x00, 0x10, 0x00, 0x11, 0x00, 0x12, 0x00,
  0x13, 0x00, 0x14, 0x00, 0x15, 0x00, 0x16, 0x00, 0x17, 0x00, 0x18, 0x00, 0x19, 0x00, 0x1a, 0x00, 0x1b, 0x00, 0x1c,
  0x00, 0x1d, 0x00
};

// Given a deflate stream and an offset into it, make the litlen and
// distance huffmans.  Return the number of bits in the DEFLATE block
// header, which is the size of the dynamic tables if there are
// dynamic tables, or just 3 if it's a static table, or it's 19-27 if
// it's non-compressed, depending on the alignment.  Zero is never a
// valid return so we will use 0 to indicate an error.
template <bool BOUNDS_CHECK = false, typename T>
__device__ uint decode_block_header(
  const T &deflate_stream,
  const size_t stream_bit_offset,
  const size_t stream_remaining_bits,
  HuffmanTable<MAX_SYMBOL_BITS, const_max(LITLEN_COUNT, DIST_COUNT)> *huffmans,
  BlockHeaderShared<clc_shifts, litlen_shifts> *shmem,
  bool &is_final
)
{
  const size_t data_size_bytes = nvcomp::roundUpDiv(stream_bit_offset + stream_remaining_bits, 8);
  // Copy the necessary bits of the deflate stream into shared memory.
  // This will save some access time later.  Also, we align all the
  // bits so that the rest of the accesses can have a known,
  // consistent alignment and offset.
  const auto first_block_bits =
    get_aligned_32_bits<BOUNDS_CHECK>(deflate_stream, stream_bit_offset, 0, data_size_bytes);
  is_final = (first_block_bits & BLOCK_FINAL_MASK) == BLOCK_FINAL_MASK;

  if ((first_block_bits & BLOCK_TYPE_MASK) == BLOCK_TYPE_DYNAMIC)
  {
    const size_t thread_id = threadIdx.x;
    uint32_t *header = shmem->header;
    // header is the first 72 32-bit words of the block, aligned.
    if (thread_id < min(72ull, (stream_remaining_bits + 31ull) / 32ull))
    {
      header[thread_id] =
        get_aligned_32_bits<BOUNDS_CHECK>(deflate_stream, stream_bit_offset, thread_id, data_size_bytes);
    }
    cooperative_groups::thread_block::sync();
    return decode_dynamic_huffman(header, huffmans, shmem);
  }
  else if ((first_block_bits & BLOCK_TYPE_MASK) == BLOCK_TYPE_NO_COMPRESSION)
  {
    const auto len_nlen_start = nvcomp::roundUpDiv(stream_bit_offset + 3, 8) * 8;
    if (threadIdx.x == 0)
    {
      // Non-compressed block.
      huffmans->set_non_compressed();
      // How many bits skipped to align the len and nlen?
      const auto len_nlen = get_aligned_32_bits<BOUNDS_CHECK>(deflate_stream, len_nlen_start, 0, data_size_bytes);
      huffmans->set_non_compressed_length(len_nlen & 0xffff);
      // TODO: Check that NLEN == ~LEN?
    }
    return len_nlen_start + sizeof(uint16_t) * 8 * 2 - // LEN + NLEN
           stream_bit_offset;
  }
  else if ((first_block_bits & BLOCK_TYPE_MASK) == BLOCK_TYPE_FIXED)
  {
    static_assert(
      sizeof(HuffmanInfo().huffmans) % sizeof(uint64_t) == 0,
      "The data to copy should be a multiple of 8 in size so that "
      "we can use 8 byte copies."
    );
    copy_with_runt<BLOCK_HEADER_THREADS_PER_BLOCK, sizeof(HuffmanInfo().huffmans) / sizeof(uint64_t)>(
      reinterpret_cast<uint64_t *&>(huffmans),
      0,
      reinterpret_cast<const uint64_t *>(&FIXED_HUFFMANS),
      0
    );
    return 3;
  }
  return 0;
}

// Note:
// This code does not try to explicitly sync with the host. Instead, it is the
// the deflate block decoder device functions that synchronize with the host, and
// when the new request arrives in this CTA, there's an implicit guarantee that
// the input buffer contains the right (amount of) data.
template <bool BOUNDS_CHECK = false, typename T>
inline __device__ void decode_block_header(
  const T &deflate_stream,
  MailboxD2D<HuffmanInfo> huffman_mailbox,
  const size_t compressed_size_bits /* or 11111..1100000 for streaming mode */
)
{
  // Response & request are bit offsets in the deflate stream
  // that indicate the start of each raw deflate block header.
  // The bit offsets are linear and do not wrap around.
  size_t response = 0; // Only valid for threadIdx.x == 0
  __shared__ size_t request;
  if (threadIdx.x == 0)
  {
    request = huffman_mailbox.get_request();
  }
  cooperative_groups::thread_block::sync();
  __shared__ BlockHeaderShared<clc_shifts, litlen_shifts> shmem;
  // This is for partial_copy of the SymbolsOverlap but we put the
  // assertion here so that it is run only once instead of in each
  // call to partial_copy, which would be too slow to do.
  expect_eq(
    size_t(reinterpret_cast<block_header_copy_type *>(&(shmem.symbols_overlaps[0]))) % sizeof(block_header_copy_type),
    0
  );
  while (true)
  {
    if (threadIdx.x == 0)
    {
      request = huffman_mailbox.wait_for_request_to_not_be(response);
    }
    cooperative_groups::thread_block::sync();
    if (request == SIZE_MAX)
    {
      return;
    }
    __shared__ uint32_t huffmans_data[nvcomp::roundUpDiv(sizeof(huffman_mailbox.data->huffmans), sizeof(uint32_t))];
    HuffmanTable<MAX_SYMBOL_BITS, const_max(LITLEN_COUNT, DIST_COUNT)> *huffmans =
      reinterpret_cast<HuffmanTable<MAX_SYMBOL_BITS, const_max(LITLEN_COUNT, DIST_COUNT)> *>(huffmans_data);
    bool is_final;
    auto header_bits_size = decode_block_header<BOUNDS_CHECK>(
      deflate_stream,
      request,
      (compressed_size_bits - request),
      huffmans,
      &shmem,
      is_final
    );
    cooperative_groups::thread_block::sync();
    if (threadIdx.x == 0)
    {
      huffman_mailbox.data->header_bits_size = header_bits_size;
      huffman_mailbox.data->is_final = is_final;
    }
    shmemcpy<BLOCK_HEADER_THREADS_PER_BLOCK, sizeof(huffman_mailbox.data->huffmans)>(
      &huffman_mailbox.data->huffmans,
      huffmans
    );
    cooperative_groups::thread_block::sync();
    // Let the requester have the answer.
    if (threadIdx.x == 0)
    {
      response = request;
      huffman_mailbox.set_response(response);
    }
  }
}
}; // namespace block_header
