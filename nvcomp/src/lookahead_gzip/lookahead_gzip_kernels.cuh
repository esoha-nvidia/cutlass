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

#include "block_header.cuh"
#include "constants.cuh"
#include "decode_huffman.cuh"
#include "huffman_table.cuh"
#include "include/types.h"
#include "inputfetcher.cuh"
#include "mailbox.cuh"
#include "nvcomp/shared_types.h"
#include "offset_to_overlap.cuh"
#include "outputstream.cuh"
#include "ring_buffer.cuh"
#include "timing_reporter.cuh"

namespace lookahead_gzip
{

constexpr uint THREADS_PER_ELEMENT = const_max(THREADS_PER_O2O_ADD, THREADS_PER_O2O_COPY);
constexpr uint PRESCAN_ELEMENTS_PER_BLOCK = THREADS_PER_BLOCK / THREADS_PER_ELEMENT;

template <typename O2OVector>
struct DecodeShared
{
  __host__ __device__ DecodeShared(char *data, uint block_count)
      : data(data)
      , block_count(block_count)
      , overlaps(calc_overlaps())
      , shmem_deflate_stream(calc_shmem_deflate_stream())
      , shmem_deflate_stream_count(calc_shmem_deflate_stream_count())
      , final(calc_final())
      , stride_bytes(calc_stride_bytes())
      , stride_bits(calc_stride_bits())
      , shared_temps(calc_shared_temps())
      , shared_data(calc_shared_data())
      , unfinished(calc_unfinished())
      , next_index(calc_next_index())
      , unfinished_lookback_atomic(calc_unfinished_lookback_atomic())
      , uncompressed_shared_words(calc_uncompressed_shared_words())
      , uncompressed_shared_words_atomic(calc_uncompressed_shared_words_atomic())
  {}

  size_t size_in_bytes()
  {
    char *biggest = 0;
    biggest = max(
      biggest,
      next_pointer<char>(calc_uncompressed_shared_words_atomic(), uncompressed_shared_words_atomic_count())
    );
    biggest = max(biggest, next_pointer<char>(calc_unfinished_lookback_atomic(), unfinished_lookback_atomic_count()));
    biggest = max(biggest, next_pointer<char>(calc_shared_data(), shared_data_count()));
    return biggest - data;
  }

  char *const data;
  const uint block_count;
  O2OVector *const overlaps;
  uint32_t *const shmem_deflate_stream;
  const size_t stride_bytes;
  const size_t stride_bits;
  const size_t shmem_deflate_stream_count;
  O2OVector *const final;
  O2OVector *const shared_temps;
  O2OVector *const shared_data;
  uint32_t *const unfinished;
  uint *const next_index;
  uint *const unfinished_lookback_atomic;
  FixedArray<decode_huffman::UncompressedWord, UNCOMPRESSED_SHARED_WORDS_COUNT> *uncompressed_shared_words;
  uint *const uncompressed_shared_words_atomic;

private:
  __host__ __device__ O2OVector *calc_overlaps() { return reinterpret_cast<O2OVector *>(data); }
  __host__ __device__ size_t overlaps_count() const { return O2O_VECTORS_PER_BLOCK; }
  __host__ __device__ uint32_t *calc_shmem_deflate_stream()
  {
    return next_pointer<uint32_t>(calc_overlaps(), overlaps_count());
  }
  // Size of each STRIDE.  Bigger will mean that each thread is doing
  // more work but there will be fewer threads, which also means fewer
  // results to later prescan.  So a bigger number is worsening
  // offer_to_overlap computation peformance in exchange for better
  // prescan performance.
  __host__ __device__ size_t calc_stride_bytes() const
  {
    auto ret = nvcomp::roundUpDiv(EXPECTED_BLOCK_COMPRESSED_SIZE, (block_count * O2O_VECTORS_PER_BLOCK));
    // Need to be able to fit the end into the overlap type.
    assert(ret * 8 + MAX_OFFSET < (1 << (sizeof(OVERLAP_BASE_TYPE) * 8 - 1)));
    // This must be at least big enough for one iteration.  And we
    // should have twice that so that we have enough room to write one
    // while the other is being read.
    assert(INPUT_RING_BUFFER_SIZE * sizeof(uint32_t) > block_count * O2O_VECTORS_PER_BLOCK * ret * 2);

    return ret;
  }
  __host__ __device__ size_t calc_stride_bits() const { return calc_stride_bytes() * 8; }
  __host__ __device__ size_t calc_shmem_deflate_stream_count() const
  {
    return nvcomp::roundUpDiv(
      (calc_stride_bits() * O2O_VECTORS_PER_BLOCK +
       MAX_OFFSET + // We might need a decode in overlap to get to a valid start Huffman index.
       64), // Because the bit buffer is loaded with funnelshift of 64-bits.
      32
    );
  }
  __host__ __device__ O2OVector *calc_final()
  {
    return next_pointer<O2OVector>(calc_shmem_deflate_stream(), calc_shmem_deflate_stream_count());
  }
  __host__ __device__ size_t final_count() const { return 1; }
  __host__ __device__ O2OVector *calc_shared_temps() { return next_pointer<O2OVector>(calc_final(), final_count()); }
  __host__ __device__ size_t shared_temps_count() const { return PRESCAN_ELEMENTS_PER_BLOCK; }
  __host__ __device__ O2OVector *calc_shared_data()
  {
    return next_pointer<O2OVector>(calc_shared_temps(), shared_temps_count());
  }
  __host__ __device__ size_t shared_data_count() const
  {
    return shared_data_size(block_count, PRESCAN_ELEMENTS_PER_BLOCK) * PRESCAN_ELEMENTS_PER_BLOCK + 1;
  }

  __host__ __device__ uint32_t *calc_unfinished()
  {
    return next_pointer<uint32_t>(calc_shmem_deflate_stream(), calc_shmem_deflate_stream_count());
  }
  __host__ __device__ size_t unfinished_count() const { return PRUNE ? O2O_VECTORS_PER_BLOCK * MAX_OFFSET : 1; }
  __host__ __device__ uint *calc_next_index() { return next_pointer<uint32_t>(calc_unfinished(), unfinished_count()); }
  __host__ __device__ size_t next_index_count() const { return 1; }
  __host__ __device__ uint *calc_unfinished_lookback_atomic()
  {
    return next_pointer<uint32_t>(calc_next_index(), next_index_count());
  }
  __host__ __device__ size_t unfinished_lookback_atomic_count() const { return 1; }

  __host__ __device__ FixedArray<decode_huffman::UncompressedWord, UNCOMPRESSED_SHARED_WORDS_COUNT> *
  calc_uncompressed_shared_words()
  {
    return reinterpret_cast<FixedArray<decode_huffman::UncompressedWord, UNCOMPRESSED_SHARED_WORDS_COUNT> *>(
      next_pointer<decode_huffman::UncompressedWord>(calc_shmem_deflate_stream(), calc_shmem_deflate_stream_count())
    );
  }
  __host__ __device__ size_t uncompressed_shared_words_count() const { return 1; }
  __host__ __device__ uint *calc_uncompressed_shared_words_atomic()
  {
    return next_pointer<uint>(calc_uncompressed_shared_words(), uncompressed_shared_words_count());
  }
  __host__ __device__ size_t uncompressed_shared_words_atomic_count() const { return 1; }

  template <typename T>
  static __host__ __device__ size_t get_size_in_bytes(T *start, size_t count)
  {
    return sizeof(*start) * count;
  }

  template <typename S, typename T>
  static __host__ __device__ S *next_pointer(T *start, size_t count)
  {
    auto ret = reinterpret_cast<uintptr_t>(start + count);
    ret = nvcomp::roundUpDiv(ret, sizeof(copy_type)) * sizeof(copy_type);
    return reinterpret_cast<S *>(ret);
  }
  template <typename T>
  static __host__ __device__ T max(T a, T b)
  {
    return a > b ? a : b;
  }
};

struct kernel_data_base_t
{
  O2OVector<OVERLAP_TYPE> *total_overlaps;
  size_t *block_bit_offset;
  MailboxD2D<HuffmanInfo> huffman_mailbox;
  FixedRing<decode_huffman::UncompressedWord, UNCOMPRESSED_WORDS_COUNT> *uncompressed_words;
  uint *uncompressed_words_atomic;
  int *gzip_header_offset; // in bits
  PartialGrid *grid;
#if TIMING
  TimingReporter *timing_reporter;
#endif // TIMING
};

template <bufferType_t BufferType>
struct kernel_data_t : kernel_data_base_t
{};

template <>
struct kernel_data_t<bufferType_t::RING> : kernel_data_base_t
{
  BufferH2D<uint32_t> deflate_stream;
  BufferD2H<char> uncompressed_stream;
};

template <>
struct kernel_data_t<bufferType_t::LINEAR> : kernel_data_base_t
{};

template <
  uint THREADS_PER_BLOCK,
  uint ELEMENTS_PER_BLOCK,
  uint THREADS_PER_ADD,
  uint THREADS_PER_COPY,
  typename O2OVector,
  typename grid_t>
inline __device__ void get_final(
  O2OVector elements[ELEMENTS_PER_BLOCK],
  O2OVector *aggregates,
  O2OVector *final,
  grid_t &grid,
  DecodeShared<O2OVector> &decode_shmem,
  const uint blocks
)
{
  grid_prescan_up<THREADS_PER_BLOCK, ELEMENTS_PER_BLOCK, THREADS_PER_ADD, THREADS_PER_COPY>(
    elements,
    aggregates,
    final,
    grid,
    decode_shmem.shared_temps,
    decode_shmem.shared_data,
    blocks
  );
}

// Returns the size of the block from the start of the O2OVectors
// until the end of the block.  This assumes that we did find the end
// of the block somewhere.  Also, it only counts the bits since the
// first O2OVector in this pass and doesn't count the bits that came
// before.
template <
  uint THREADS_PER_BLOCK,
  uint ELEMENTS_PER_BLOCK,
  uint THREADS_PER_ADD,
  uint THREADS_PER_COPY,
  typename O2OVector,
  typename grid_t>
inline __device__ size_t get_block_size_with_prescan(
  O2OVector elements[ELEMENTS_PER_BLOCK],
  O2OVector *aggregates,
  const O2OVector *final,
  grid_t &grid,
  DecodeShared<O2OVector> &decode_shmem,
  const uint current_raw_offset,
  const size_t current_data_start,
  size_t *out,
  const uint blocks
)
{
  grid_prescan_down<THREADS_PER_BLOCK, ELEMENTS_PER_BLOCK, THREADS_PER_O2O_ADD, THREADS_PER_O2O_COPY>(
    elements,
    aggregates,
    final,
    grid,
    decode_shmem.shared_temps,
    decode_shmem.shared_data,
    blocks
  );
  grid.sync();

  // Hooray, we are done.  But which one is the end of block?
  // There is only one o2o vector where the element at
  // current_raw_offset is end and the element before it isn't.
  // So report that one.

  if (threadIdx.x < ELEMENTS_PER_BLOCK)
  {
    const O2OVector &mine = elements[threadIdx.x];
    const O2OVector &next = (threadIdx.x + 1 < ELEMENTS_PER_BLOCK) ? elements[threadIdx.x + 1]
                            : (blockIdx.x + 1 < blocks)            ? aggregates[blockIdx.x + 1]
                                                                   : *final;
    if (!mine[current_raw_offset].is_end_of_block() && next[current_raw_offset].is_end_of_block())
    {
      *out = current_data_start + (blockIdx.x * ELEMENTS_PER_BLOCK + threadIdx.x) * decode_shmem.stride_bits +
             next[current_raw_offset].get_end_of_block_stride_bits();
    }
  }
  // We need all threads to know about this so re-read it from global memory.
  grid.sync();
  return *out;
}

template <
  uint THREADS_PER_BLOCK,
  uint ELEMENTS_PER_BLOCK,
  uint THREADS_PER_ADD,
  uint THREADS_PER_COPY,
  typename overlap_base_t,
  typename grid_t>
inline __device__ size_t get_block_size(
  O2OVector<UncompressedSizeOverlap<overlap_base_t>> elements[ELEMENTS_PER_BLOCK],
  O2OVector<UncompressedSizeOverlap<overlap_base_t>> *aggregates,
  const O2OVector<UncompressedSizeOverlap<overlap_base_t>> *final,
  grid_t &grid,
  DecodeShared<O2OVector<UncompressedSizeOverlap<overlap_base_t>>> &decode_shmem,
  const uint current_raw_offset,
  const size_t current_data_start,
  size_t *out,
  const uint blocks
)
{
  return get_block_size_with_prescan<THREADS_PER_BLOCK, ELEMENTS_PER_BLOCK, THREADS_PER_ADD, THREADS_PER_COPY>(
    elements,
    aggregates,
    final,
    grid,
    decode_shmem,
    current_raw_offset,
    current_data_start,
    out,
    blocks
  );
}

// Complete the prescan of the O2OVectors.
template <
  uint THREADS_PER_BLOCK,
  uint ELEMENTS_PER_BLOCK,
  uint THREADS_PER_ADD,
  uint THREADS_PER_COPY,
  typename O2OVector,
  typename grid_t>
inline __device__ void complete_prescan(
  O2OVector elements[ELEMENTS_PER_BLOCK],
  O2OVector *aggregates,
  const O2OVector *final,
  grid_t &grid,
  DecodeShared<O2OVector> &decode_shmem,
  const uint blocks
)
{
  grid_prescan_down<THREADS_PER_BLOCK, ELEMENTS_PER_BLOCK, THREADS_PER_O2O_ADD, THREADS_PER_O2O_COPY>(
    elements,
    aggregates,
    final,
    grid,
    decode_shmem.shared_temps,
    decode_shmem.shared_data,
    blocks
  );
}

template <uint THREADS_PER_BLOCK, uint ELEMENTS_PER_BLOCK, typename overlap_base_t>
inline __device__ uint get_uncompressed_size(const UncompressedSizeOverlap<overlap_base_t> &final_overlap)
{
  return final_overlap.get_uncompressed_size();
}

template <uint ELEMENTS_PER_BLOCK, typename overlap_base_t>
inline __device__ uint32_t get_uncompressed_stride_offset(
  const O2OVector<UncompressedSizeOverlap<overlap_base_t>> overlaps[ELEMENTS_PER_BLOCK],
  const uint thread_id,
  const uint current_raw_offset
)
{
  return overlaps[thread_id][current_raw_offset].get_uncompressed_size();
}

template <uint THREADS_PER_BLOCK, uint ELEMENTS_PER_BLOCK, typename overlap_base_t>
inline __device__ uint32_t get_block_uncompressed_end_offset(
  const uint current_raw_offset,
  const O2OVector<UncompressedSizeOverlap<overlap_base_t>> *aggregates,
  const uint32_t current_uncompressed_size,
  const uint blocks
)
{
  if (blockIdx.x + 1 < blocks)
  {
    return aggregates[blockIdx.x + 1][current_raw_offset].get_uncompressed_size();
  }
  else
  {
    return current_uncompressed_size;
  }
}

template <uint ELEMENTS_PER_BLOCK, uint MAX_WRITE_INDEX, bool RANGE_CHECK, typename deflate_stream_t, typename O2OVector>
inline __device__ void write_uncompressed_words(
  const deflate_stream_t &deflate_stream,
  const size_t current_data_start,
  const HuffmanTable<MAX_SYMBOL_BITS, LITLEN_COUNT> *huffmans,
  const uint current_raw_offset,
  O2OVector overlaps[ELEMENTS_PER_BLOCK],
  FixedArray<decode_huffman::UncompressedWord, UNCOMPRESSED_SHARED_WORDS_COUNT> &uncompressed_shared_words,
  const size_t uncompressed_words_offset,
  const size_t words_count,
  const size_t stride_bits,
  const size_t deflate_stream_size
)
{
  // Decode at the correct offset for each O2OVector and write into
  // uncompressed_words.  Only the words that are in the range
  // uncompressed_words_offset to
  // uncompressed_words_offset+words_count are written because the
  // rest were too big for the uncompressed_words.
  const uint stride_id = threadIdx.x / MAX_WRITE_INDEX;
  const uint write_index = threadIdx.x % MAX_WRITE_INDEX;
  if (stride_id >= ELEMENTS_PER_BLOCK)
  {
    return; // This is not a valid stride.
  }
  if (overlaps[stride_id][current_raw_offset].is_error_or_end_of_block())
  {
    return; // This is past the end of the block, no processing is necessary.
  }
  const uint iteration_offset = overlaps[stride_id][current_raw_offset].get_overlap();
  const auto start_bit_offset = (stride_id)*stride_bits + current_data_start;
  const auto current_bit_offset = start_bit_offset + iteration_offset;
  decode_huffman::decode_one<RANGE_CHECK, MAX_WRITE_INDEX>(
    deflate_stream,
    current_bit_offset,
    huffmans,
    iteration_offset,
    write_index,
    uncompressed_shared_words,
    get_uncompressed_stride_offset<ELEMENTS_PER_BLOCK>(overlaps, stride_id, current_raw_offset) -
      get_uncompressed_stride_offset<ELEMENTS_PER_BLOCK>(overlaps, 0, current_raw_offset),
    uncompressed_words_offset,
    words_count,
    stride_bits,
    deflate_stream_size
  );
}

template <typename O2OVector, typename deflate_stream_t, typename decode_shmem_t>
static inline __device__ void resolve_prune(
  O2OVector overlaps[O2O_VECTORS_PER_BLOCK],
  size_t current_data_start,
  const deflate_stream_t &deflate_stream,
  const HuffmanTable<MAX_SYMBOL_BITS, LITLEN_COUNT> *huffmans,
  decode_shmem_t &decode_shmem,
  const size_t deflate_stream_size
)
{
  // Everything is either an error, end of block, copy or an
  // unfinished decode.  Put the indices of the unfinished ones into
  // an array.
  auto &unfinished = decode_shmem.unfinished;
  auto &next_index = decode_shmem.next_index;
  if (threadIdx.x == 0)
  {
    next_index[0] = 0;
  }
  cooperative_groups::thread_block::sync();
  for (uint stride = 0; stride < O2O_VECTORS_PER_BLOCK * MAX_OFFSET / THREADS_PER_BLOCK; stride++)
  {
    uint i = stride * THREADS_PER_BLOCK + threadIdx.x;
    if (!overlaps[i / MAX_OFFSET][i % MAX_OFFSET].is_error_or_end_of_block() &&
        !overlaps[i / MAX_OFFSET][i % MAX_OFFSET].is_copy())
    {
      unfinished[atomicAdd(&next_index[0], 1)] = i;
    }
  }
  if ((O2O_VECTORS_PER_BLOCK * MAX_OFFSET) % THREADS_PER_BLOCK > 0 &&
      threadIdx.x < (O2O_VECTORS_PER_BLOCK * MAX_OFFSET) % THREADS_PER_BLOCK)
  {
    uint i = O2O_VECTORS_PER_BLOCK * MAX_OFFSET / THREADS_PER_BLOCK * THREADS_PER_BLOCK + threadIdx.x;
    if (!overlaps[i / MAX_OFFSET][i % MAX_OFFSET].is_error_or_end_of_block() &&
        !overlaps[i / MAX_OFFSET][i % MAX_OFFSET].is_copy())
    {
      unfinished[atomicAdd(&next_index[0], 1)] = i;
    }
  }
  cooperative_groups::thread_block::sync();
  for (uint stride = threadIdx.x; stride < next_index[0]; stride += THREADS_PER_BLOCK)
  {
    uint todo = unfinished[stride];
    // We need to convert the todo into the correct overlap and
    // offset that needs work.
    const uint o2o_vector_id = todo / MAX_OFFSET;
    // This is the row in the O2O that we need to decode.
    auto &to_update = overlaps[todo / MAX_OFFSET][todo % MAX_OFFSET];
    typename O2OVector::value_type the_rest;
    const uint huffman_tree_start_index = to_update.get_huffman_tree_index();
    const uint iteration_offset = to_update.get_overlap() + MAX_OFFSET;
    const auto start_bit_offset = o2o_vector_id * decode_shmem.stride_bits + current_data_start;
    const auto current_bit_offset = start_bit_offset + iteration_offset;
    if (iteration_offset < decode_shmem.stride_bits)
    {
      decode_huffman::decode_one<false>(
        deflate_stream,
        current_bit_offset,
        huffmans,
        huffman_tree_start_index,
        iteration_offset,
        &the_rest,
        decode_shmem.stride_bits,
        deflate_stream_size
      );
      to_update.add(to_update, the_rest);
    }
    else
    {
      // We already know that it isn't an error or end of block
      // because it's todo.  The overlap is currently written as bits
      // beyond MAX_OFFSET instead of bits beyond the STRIDE_BITS.
      // Fix that here.
      decode_huffman::todo_to_overlap(&to_update, decode_shmem.stride_bits);
    }
  }
  auto &unfinished_lookback_atomic = decode_shmem.unfinished_lookback_atomic;
  unfinished_lookback_atomic[0] = THREADS_PER_BLOCK;
  cooperative_groups::thread_block::sync();
  auto overlaps1d = reinterpret_cast<volatile OVERLAP_TYPE *>(&overlaps[0][0]);
  lookback::block_lookback_atomics<THREADS_PER_BLOCK>(
    overlaps1d,
    0,
    O2O_VECTORS_PER_BLOCK * MAX_OFFSET,
    0, // ignored
    &unfinished_lookback_atomic[0]
  );
  cooperative_groups::thread_block::sync();
}

template <typename O2OVector>
inline __device__ void ruin_shared_data(DecodeShared<O2OVector> &decode_shmem)
{
  if (threadIdx.x == 0)
  {
    decode_shmem.shared_data.ruin();
    decode_shmem.shared_temps.ruin();
    decode_shmem.final.ruin();
  }
}

template <typename O2OVector>
inline __device__ void ruin_unfinished_data(DecodeShared<O2OVector> &decode_shmem)
{
  if (threadIdx.x == 0)
  {
    decode_shmem.unfinished.ruin();
    decode_shmem.next_index.ruin();
  }
}

template <uint THREADS_PER_BLOCK, typename S, typename D>
inline __device__ void copy_with_runt(D &dst, const size_t dst_index, S &src, const size_t src_index, const size_t size)
{
  const size_t max_offset = src.get_size() / sizeof(src[0]);
  for (size_t index = threadIdx.x, src_offset = threadIdx.x + src_index; index < size;
       index += THREADS_PER_BLOCK, src_offset += THREADS_PER_BLOCK)
  {
    if (src.get_buffer_type() == bufferType_t::RING || src_offset < max_offset)
    {
      dst[dst_index + index] = src[src_offset];
    }
  }
}

class HuffmanHandler
{
public:
  inline __device__ HuffmanHandler(size_t stream_bit_offset, MailboxD2D<HuffmanInfo> &&huffman_mailbox)
      : huffman_mailbox(huffman_mailbox)
  {
    set_current_block_start(stream_bit_offset);
  }

  // Waits until mailbox response becomes current_block_start
  inline __device__ void wait_for_response()
  {
    timing_reporter.toggle(TimingReporter::Timer::WAIT_FOR_HUFFMANS);
    huffman_mailbox.wait_for_response_to_be(current_block_start);
    timing_reporter.toggle(TimingReporter::Timer::WAIT_FOR_HUFFMANS);
  }

  // One should issue a wait_for_response() before this call,
  // to make sure, that the data is ready
  inline __device__ void copy_huffmans(HuffmanTable<MAX_SYMBOL_BITS, LITLEN_COUNT> *huffmans, bool &is_final)
  {
    timing_reporter.toggle(TimingReporter::Timer::COPY_HUFFMANS);
    const auto &header_bits_size = huffman_mailbox.data->header_bits_size;
    is_final = huffman_mailbox.data->is_final;
    current_data_start = get_current_block_start() + header_bits_size;
    if (header_bits_size == 0)
    {
      // The header was invalid, we will indicate this by setting the
      // current_data_start to 0.
      current_data_start = 0;
    }
    shmemcpy<THREADS_PER_BLOCK, sizeof(HuffmanInfo().huffmans)>(huffmans, &huffman_mailbox.data->huffmans);
    timing_reporter.toggle(TimingReporter::Timer::COPY_HUFFMANS);
  }
  inline __device__ size_t get_current_block_start() const { return current_block_start; }
  inline __device__ bool is_error() const { return current_data_start == 0; }
  inline __device__ void set_current_block_start(size_t current_block_start)
  {
    this->current_block_start = current_block_start;
    // We always want to notify the Huffman creation kernel as soon as possible.
    send_current_block_start(current_block_start);
  }
  inline __device__ void set_done() { send_current_block_start(SIZE_MAX); }
  inline __device__ size_t get_current_data_start() { return current_data_start; }
  inline __device__ void increment_current_data_start(uint x) { current_data_start += x; }

private:
  inline __device__ void send_current_block_start(size_t current_block_start)
  {
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
      huffman_mailbox.set_request(current_block_start);
    }
  }
  size_t current_block_start;
  size_t current_data_start;
  MailboxD2D<HuffmanInfo> huffman_mailbox;
};

template <bool prune, typename shmem_deflate_stream_t, typename O2OVector, typename decode_shmem_t>
inline __device__ void make_o2o_vectors(
  const shmem_deflate_stream_t &shmem_deflate_stream,
  const size_t &bit_offset,
  const HuffmanTable<MAX_SYMBOL_BITS, LITLEN_COUNT> *huffmans,
  O2OVector overlaps[O2O_VECTORS_PER_BLOCK],
  decode_shmem_t &decode_shmem,
  const size_t deflate_stream_size
)
{
  timing_reporter.toggle(TimingReporter::Timer::DECODE_ONE);
  decode_huffman::decode_many<prune>(
    shmem_deflate_stream,
    bit_offset,
    huffmans,
    overlaps,
    decode_shmem.stride_bits,
    deflate_stream_size
  );
  cooperative_groups::thread_block::sync();
  timing_reporter.toggle(TimingReporter::Timer::DECODE_ONE);
  timing_reporter.toggle(TimingReporter::Timer::RESOLVE_PRUNE);
  if (PRUNE)
  {
    resolve_prune(overlaps, bit_offset, shmem_deflate_stream, huffmans, decode_shmem, deflate_stream_size);
  }
  timing_reporter.toggle(TimingReporter::Timer::RESOLVE_PRUNE);
}

template <uint ELEMENTS_PER_BLOCK, typename O2OVector, typename decode_shmem_t, typename grid_t>
inline __device__ size_t prescan_o2o_vectors(
  O2OVector elements[ELEMENTS_PER_BLOCK],
  O2OVector *aggregates,
  grid_t &grid,
  const uint current_raw_offset,
  const size_t current_data_start,
  size_t *block_bit_offset,
  decode_shmem_t &decode_shmem,
  const uint grid_dim_x
)
{
  timing_reporter.toggle(TimingReporter::Timer::GET_FINAL);
  auto &final = decode_shmem.final[0]; // The aggregate of all the O2O in all blocks.
  get_final<THREADS_PER_BLOCK, O2O_VECTORS_PER_BLOCK, THREADS_PER_O2O_ADD, THREADS_PER_O2O_COPY>(
    elements,
    aggregates,
    &final,
    grid,
    decode_shmem,
    grid_dim_x
  );
  cooperative_groups::thread_block::sync();
  timing_reporter.toggle(TimingReporter::Timer::GET_FINAL);
  timing_reporter.toggle(TimingReporter::Timer::COMPLETE_PRESCAN);
  // Because this is exclusive sum, it could be that we final in
  // the aggregate but not in the members before it.
  const auto &final_overlap = final[current_raw_offset];
  size_t new_block_start = 0;
  if (final_overlap.is_end_of_block())
  {
    new_block_start =
      get_block_size<THREADS_PER_BLOCK, O2O_VECTORS_PER_BLOCK, THREADS_PER_O2O_ADD, THREADS_PER_O2O_COPY>(
        elements,
        aggregates,
        &final,
        grid,
        decode_shmem,
        current_raw_offset,
        current_data_start,
        block_bit_offset,
        grid_dim_x
      );
  }
  else
  {
    complete_prescan<THREADS_PER_BLOCK, O2O_VECTORS_PER_BLOCK, THREADS_PER_O2O_ADD, THREADS_PER_O2O_COPY>(
      elements,
      aggregates,
      &final,
      grid,
      decode_shmem,
      grid_dim_x
    );
  }
  timing_reporter.toggle(TimingReporter::Timer::COMPLETE_PRESCAN);
  return new_block_start;
}

template <typename O2OVector, typename grid_t, bufferType_t BufferType>
inline __device__ void decompress(
  InputFetcher<BufferType> &&input_fetcher,
  O2OVector *aggregates,
  size_t *block_bit_offset,
  HuffmanHandler &&huffman_handler /* in registers, 1 per chunk */,
  FixedRing<decode_huffman::UncompressedWord, UNCOMPRESSED_WORDS_COUNT> *uncompressed_words,
  uint *uncompressed_words_atomic,
  OutputStream<BufferType> &&output_stream,
  grid_t &grid,
  nvcompStatus_t *device_status
)
{
  // Note:
  // HuffmanInfo contains 2 huffman trees (just like a dynamic huffman block)
  // - 1 tree for the literal + length symbols
  // - 1 tree for the distance symbols
  __shared__ uint32_t huffmans_data[nvcomp::roundUpDiv(sizeof(HuffmanInfo().huffmans), sizeof(uint32_t))];
  HuffmanTable<MAX_SYMBOL_BITS, LITLEN_COUNT> *huffmans =
    reinterpret_cast<HuffmanTable<MAX_SYMBOL_BITS, LITLEN_COUNT> *>(huffmans_data);
  extern __shared__ char shmem[];
  const auto grid_dim_x = grid.get_total();
  DecodeShared<O2OVector> decode_shmem(shmem, grid_dim_x);
  // This is for partial_copy of the O2OVector but we put the
  // assertion here so that it is run only once instead of in each
  // call to partial_copy, which would be too slow to do.
  expect_eq(size_t(reinterpret_cast<copy_type *>(&(decode_shmem.shared_data[0]))) % sizeof(copy_type), 0);
  size_t current_uncompressed_offset = 0;
  //uint decodes = 0;
  bool is_final;
  uint current_deflate_block_id = 0;
  // We only need the extra element for the prescan for the aggregate.
  // Having this array be a power of two size seems to help
  // performance, not sure why.  TODO: Replace this with a union
  // shared.
  auto &overlaps = decode_shmem.overlaps;
  size_t to_global_lz77_start = 0;
  size_t to_global_lz77_end = 0;

  do
  {
    // Each deflate block starts with 3 bits:
    // BFINAL (first bit): Indicates whether this is the last deflate block of the stream
    // BTYPE (next two bits): Indicates the content of the deflate block (see RFC 1951)

    // Note, that we might not be able to fit the entire deflate block into the ring buffer
    // so at some point the current deflate block's start might be invalid.
    // Note 2: huffman_handler stores block starts in [bits], but input_fetcher fetches in sizeof(uint32_t),
    //         hence the division by 32
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
      huffman_handler.wait_for_response();
    }
    grid.sync(); // Let everything wait for the Huffmans to be ready.

    // We copy into huffmans the huffman table that starts at huffman_handler.get_current_block_start() + header length
    // With this huffman table we can decode the entire deflate block
    huffman_handler.copy_huffmans(huffmans, is_final);

    if (huffman_handler.is_error())
    {
      if (blockIdx.x == 0 && threadIdx.x == 0)
      {
        // Signal the end of the output stream.
        output_stream.set_error();
        // Quit the decode_block_header_kernel.
        huffman_handler.set_done();
        if (device_status)
        {
          *device_status = nvcompErrorCannotDecompress;
        }
      }
      return;
    }
    // Now find the end of the block.
    uint current_raw_offset = 0;
    while (true)
    {
      if constexpr (BufferType == bufferType_t::RING)
      {
        if (blockIdx.x == 0 && threadIdx.x == 0)
        {
          // Note:
          // The deflate block header has been already read out. Here we need to make sure
          // that the part of the deflate block that we are about to read AND (very important)
          // a potential next block header can be parsed immediately without calling for another
          // `wait_for_input`. The worst-case deflate header size was determined, hence we
          // potentially over-request, but that is fine.
          input_fetcher.wait_for_input(huffman_handler.get_current_data_start());
        }
      }
      grid.sync();

      uint current_uncompressed_size;
      bool found_end_of_block;
      uint next_raw_offset;
      if (huffmans->is_non_compressed())
      {
        found_end_of_block = true;
        current_uncompressed_size = huffmans->get_non_compressed_length();
        const auto next_block_start = huffman_handler.get_current_data_start() + current_uncompressed_size * 8;
        huffman_handler.set_current_block_start(next_block_start);
        *block_bit_offset = next_block_start;
      }
      else
      {
        // This is a compressed block so we need to decode it.
        auto &shmem_deflate_stream = decode_shmem.shmem_deflate_stream;
        copy_with_runt<THREADS_PER_BLOCK>(
          shmem_deflate_stream,
          0,
          input_fetcher,
          (huffman_handler.get_current_data_start() + blockIdx.x * (O2O_VECTORS_PER_BLOCK * decode_shmem.stride_bits)) /
            (sizeof(shmem_deflate_stream[0]) * 8),
          decode_shmem.shmem_deflate_stream_count
        );
        cooperative_groups::thread_block::
          sync(); // sync only in thread block, but this makes sure that the shared mem is good for all threads in the CTA

        make_o2o_vectors<PRUNE>(
          shmem_deflate_stream,
          (huffman_handler.get_current_data_start() + blockIdx.x * (O2O_VECTORS_PER_BLOCK * decode_shmem.stride_bits)) %
            (sizeof(shmem_deflate_stream[0]) * 8),
          huffmans,
          overlaps,
          decode_shmem,
          decode_shmem.shmem_deflate_stream_count * sizeof(uint32_t)
        );
        size_t new_block_start = prescan_o2o_vectors<O2O_VECTORS_PER_BLOCK>(
          overlaps,
          aggregates,
          grid,
          current_raw_offset,
          huffman_handler.get_current_data_start(),
          block_bit_offset,
          decode_shmem,
          grid_dim_x
        );
        if (!is_final && new_block_start > 0)
        {
          huffman_handler.set_current_block_start(new_block_start);
        }

        timing_reporter.toggle(TimingReporter::Timer::GET_UNCOMPRESSED_SIZE);
        // Get the uncompressed offset for each stride.
        const auto &final_overlap = decode_shmem.final[0][current_raw_offset];
        found_end_of_block = final_overlap.is_end_of_block();
        next_raw_offset = final_overlap.get_raw_overlap();
        current_uncompressed_size = get_uncompressed_size<THREADS_PER_BLOCK, O2O_VECTORS_PER_BLOCK>(final_overlap);
        grid.sync();
        timing_reporter.toggle(TimingReporter::Timer::GET_UNCOMPRESSED_SIZE);
      }
      for (size_t uncompressed_words_offset = 0; uncompressed_words_offset < current_uncompressed_size;)
      {
        const size_t words_for_iteration = min(
          uncompressed_words->size() - MAX_LOOKBACK - (to_global_lz77_end - to_global_lz77_start),
          current_uncompressed_size - uncompressed_words_offset
        );
        if constexpr (BufferType == bufferType_t::RING)
        {
          // Make sure that we have room enough for the eventual output.
          if (blockIdx.x == 0 && threadIdx.x == 0)
          {
            output_stream.wait_for_room(to_global_lz77_start, to_global_lz77_end);
          }
        }
        else
        {
          if (!HAZARDS && !output_stream.can_fit(to_global_lz77_end))
          {
            if (blockIdx.x == 0 && threadIdx.x == 0)
            {
              // Signal the end of the output stream.
              output_stream.set_error();
              // Quit the decode_block_header_kernel.
              huffman_handler.set_done();
              if (device_status)
              {
                *device_status = nvcompErrorOutputBufferTooSmall;
              }
            }
            return;
          }
        }
        if (huffmans->is_non_compressed())
        {
          // Copy the bytes into the uncompressed_words.  We put it
          // there in addition to the output stream because a future
          // LZ77 might need it.  In parallel, we need to resolve the
          // global lz77 as usual.
          uint thread_id_in_grid = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
          for (uint offset = 0; offset < words_for_iteration / (grid_dim_x * THREADS_PER_BLOCK); offset++)
          {
            uint index = thread_id_in_grid + offset * (grid_dim_x * THREADS_PER_BLOCK);
            (*uncompressed_words)[current_uncompressed_offset + index].set_literal(
              input_fetcher[(huffman_handler.get_current_data_start() / 8 + index) / 4] >>
              ((huffman_handler.get_current_data_start() / 8 + index) % 4) * 8
            );
          }
          // Now handle the runt.
          if (thread_id_in_grid < words_for_iteration % (grid_dim_x * THREADS_PER_BLOCK))
          {
            uint index = thread_id_in_grid +
                         words_for_iteration / (grid_dim_x * THREADS_PER_BLOCK) * (grid_dim_x * THREADS_PER_BLOCK);
            (*uncompressed_words)[current_uncompressed_offset + index].set_literal(
              input_fetcher[(huffman_handler.get_current_data_start() / 8 + index) / 4] >>
              ((huffman_handler.get_current_data_start() / 8 + index) % 4) * 8
            );
          }
          if (blockIdx.x == 0 && threadIdx.x == 0)
          {
            // Prepare the atomic for the grid_lookback_atomics that is
            // coming up.
            *uncompressed_words_atomic = grid_dim_x * THREADS_PER_BLOCK;
          }
          grid.sync();
          timing_reporter.toggle(TimingReporter::Timer::GLOBAL_LZ77);
          lookback::grid_lookback_atomics(
            *uncompressed_words,
            to_global_lz77_start,
            blockIdx.x * THREADS_PER_BLOCK + threadIdx.x,
            to_global_lz77_end - to_global_lz77_start,
            output_stream,
            0,
            uncompressed_words_atomic
          );
          timing_reporter.toggle(TimingReporter::Timer::GLOBAL_LZ77);
          to_global_lz77_start = to_global_lz77_end;
          to_global_lz77_end = current_uncompressed_offset + uncompressed_words_offset + words_for_iteration;
        }
        else
        {
          // First we need to decode the bytes between
          // uncompressed_words_offset and uncompressed_words_offset +
          // words_for_iteration. Hopefully it'll be enough but we
          // can't decode more than that because maybe we weren't
          // provided with enough memory.
          constexpr uint threads_for_write_uncompressed =
            nvcomp::roundUpDiv(THREADS_PER_O2O_DECODE * O2O_VECTORS_PER_BLOCK, WARP_SIZE_U) * WARP_SIZE_U;
          static_assert(
            threads_for_write_uncompressed < THREADS_PER_BLOCK,
            "Make sure that we did try to use more threads than "
            "we have for the writing."
          );
          static_assert(
            THREADS_PER_O2O_DECODE > 0,
            "Make sure that there is at least one thread for "
            "decoding."
          );
          if (blockIdx.x == 0 && threadIdx.x == 0)
          {
            // Prepare the atomic for the grid_lookback_atomics that is
            // coming up.
            *uncompressed_words_atomic = grid_dim_x * (THREADS_PER_BLOCK - threads_for_write_uncompressed);
          }
          grid.sync(); // All threads wait for there to be room enough
          // for the rest.

          // Write the words into shmem.  Each might be a literal or a
          // lookback.  We decompress the positions that are from
          // uncompressed_words_offset to
          // uncompressed_words_offset+words_for_iteration, which is
          // sure to fit into the uncompressed_words FixedArray leaving
          // room enough for 32768 lookback as required by the RFC.
          // How much of the block do we want to actually decode?  It
          // might not be the whole thing because we don't need the
          // parts that aren't part of this words_for_iteration.

          // block_start_location is the offset that this block would
          // first decode.
          const auto block_start_location =
            get_uncompressed_stride_offset<O2O_VECTORS_PER_BLOCK>(overlaps, 0, current_raw_offset);
          // block_end_location is one beyond the offset that this block could decode.
          const auto block_end_location = get_block_uncompressed_end_offset<THREADS_PER_BLOCK, O2O_VECTORS_PER_BLOCK>(
            current_raw_offset,
            aggregates,
            current_uncompressed_size,
            grid_dim_x
          );

          // What do we actually want to decode?  Only the parts that
          // overlap uncompressed_words_offset to
          // uncompressed_words_offset+words_for_iteration.  So we
          // want block_words_start and block_words_end clamped to the
          // range.  If they both clamp to the same value then there
          // will be no block decoding done, which is what we want.
          const uint32_t block_words_start = uncompressed_words_offset > block_start_location
                                               ? uncompressed_words_offset - block_start_location
                                               : 0;
          const uint32_t block_words_end =
            block_start_location >= uncompressed_words_offset + words_for_iteration
              ? 0
              : min(size_t(block_end_location), uncompressed_words_offset + words_for_iteration) - block_start_location;
          auto &uncompressed_shared_words = *decode_shmem.uncompressed_shared_words;
          for (uint start_word = block_words_start;
               start_word < block_words_end || to_global_lz77_start != to_global_lz77_end;
               start_word += uncompressed_shared_words.size())
          {
            const auto block_shmem_length =
              start_word >= block_words_end
                ? 0
                : min(size_t(block_words_end - start_word), uncompressed_shared_words.size());
            timing_reporter.toggle(TimingReporter::Timer::UNCOMPRESSED_SHMEM);
            if (threadIdx.x < threads_for_write_uncompressed)
            {
              if (block_shmem_length > 0)
              {
                auto &shmem_deflate_stream = decode_shmem.shmem_deflate_stream;
                if (start_word == 0 && block_shmem_length == block_end_location - block_start_location)
                {
                  // Do the whole thing so no range checks are needed.
                  write_uncompressed_words<O2O_VECTORS_PER_BLOCK, THREADS_PER_O2O_DECODE, false>(
                    shmem_deflate_stream,
                    (huffman_handler.get_current_data_start() +
                     (blockIdx.x * O2O_VECTORS_PER_BLOCK * decode_shmem.stride_bits)) %
                      32,
                    huffmans,
                    current_raw_offset,
                    overlaps,
                    uncompressed_shared_words,
                    0,
                    0, // Ignored.
                    decode_shmem.stride_bits,
                    decode_shmem.shmem_deflate_stream_count * sizeof(uint32_t)
                  );
                }
                else
                {
                  write_uncompressed_words<O2O_VECTORS_PER_BLOCK, THREADS_PER_O2O_DECODE, true>(
                    shmem_deflate_stream,
                    (huffman_handler.get_current_data_start() +
                     (blockIdx.x * O2O_VECTORS_PER_BLOCK * decode_shmem.stride_bits)) %
                      32,
                    huffmans,
                    current_raw_offset,
                    overlaps,
                    uncompressed_shared_words,
                    start_word, // Decompress from start_word.
                    block_shmem_length,
                    decode_shmem.stride_bits,
                    decode_shmem.shmem_deflate_stream_count * sizeof(uint32_t)
                  );
                }
              }
            }
            else
            {
              // Use the remaining threads for global LZ77.
              timing_reporter.toggle(TimingReporter::Timer::GLOBAL_LZ77, 0, threads_for_write_uncompressed);
              lookback::grid_lookback_atomics(
                *uncompressed_words,
                to_global_lz77_start,
                blockIdx.x * (THREADS_PER_BLOCK - threads_for_write_uncompressed) + threadIdx.x -
                  threads_for_write_uncompressed,
                to_global_lz77_end - to_global_lz77_start,
                output_stream,
                0,
                uncompressed_words_atomic
              );
              timing_reporter.toggle(TimingReporter::Timer::GLOBAL_LZ77, 0, threads_for_write_uncompressed);
            }
            to_global_lz77_start = to_global_lz77_end;
            // lookback in the shmem as much as possible.
            // It's possible that some of the data decoded into shmem
            // has lookbacks in places where a previous loop has already
            // gotten around to resolving.  So don't resolve backward
            // before
            // current_uncompressed_offset+uncompressed_words_offset.
            // How far before the start of the shmem are we allowed to lookback?
            const size_t max_lookback = block_start_location + start_word > uncompressed_words_offset
                                          ? block_start_location + start_word - uncompressed_words_offset + MAX_LOOKBACK
                                          : MAX_LOOKBACK;
            if (threadIdx.x == 0)
            {
              // This is the next element to work on.
              decode_shmem.uncompressed_shared_words_atomic[0] = THREADS_PER_BLOCK;
            }
            cooperative_groups::thread_block::sync();
            timing_reporter.toggle(TimingReporter::Timer::UNCOMPRESSED_SHMEM);
            timing_reporter.toggle(TimingReporter::Timer::BLOCK_LZ77);
            lookback::block_lookback_atomics<THREADS_PER_BLOCK, true>(
              uncompressed_shared_words,
              0,
              block_shmem_length,
              max_lookback,
              *uncompressed_words,
              current_uncompressed_offset + block_start_location + start_word,
              &decode_shmem.uncompressed_shared_words_atomic[0]
            );
            cooperative_groups::thread_block::sync();
            timing_reporter.toggle(TimingReporter::Timer::BLOCK_LZ77);
          }
          grid.sync();
          // The end of where we need to to do global LZ77 in the
          // uncompressed_words ring buffer.
          to_global_lz77_end = current_uncompressed_offset + uncompressed_words_offset + words_for_iteration;
        }
        uncompressed_words_offset += words_for_iteration;
      }

      current_uncompressed_offset += current_uncompressed_size;
      if (found_end_of_block)
      {
        current_deflate_block_id++;
        break;
      }
      current_raw_offset = next_raw_offset;
      // We can't add the final.get_overlap() into here because it might
      // be a length.  That's okay, it just means that we need to start
      // at a different row next time.
      huffman_handler.increment_current_data_start(grid_dim_x * O2O_VECTORS_PER_BLOCK * decode_shmem.stride_bits);
    }
    //} while(!is_final && --decodes != 0);
  } while (!is_final);
  // Clean up any remaining LZ77.
  grid.sync();

  if (to_global_lz77_start != to_global_lz77_end)
  {
    timing_reporter.toggle(TimingReporter::Timer::GLOBAL_LZ77);
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
      *uncompressed_words_atomic = grid_dim_x * THREADS_PER_BLOCK;
    }
    if constexpr (BufferType == bufferType_t::RING)
    {
      // Make sure that we have room enough for the eventual output.
      if (blockIdx.x == 0 && threadIdx.x == 0)
      {
        output_stream.wait_for_room(to_global_lz77_start, to_global_lz77_end);
      }
    }
    else
    {
      if (!HAZARDS && !output_stream.can_fit(to_global_lz77_end))
      {
        if (blockIdx.x == 0 && threadIdx.x == 0)
        {
          // Signal the end of the output stream.
          output_stream.set_error();
          // Quit the decode_block_header_kernel.
          huffman_handler.set_done();
          if (device_status)
          {
            *device_status = nvcompErrorOutputBufferTooSmall;
          }
        }
        return;
      }
    }
    grid.sync();
    lookback::grid_lookback_atomics(
      *uncompressed_words,
      to_global_lz77_start,
      blockIdx.x * THREADS_PER_BLOCK + threadIdx.x,
      to_global_lz77_end - to_global_lz77_start,
      output_stream,
      0,
      uncompressed_words_atomic
    );
    grid.sync();
    timing_reporter.toggle(TimingReporter::Timer::GLOBAL_LZ77);
  }
  if (blockIdx.x == 0 && threadIdx.x == 0)
  {
    // Signal the end of the output stream.
    output_stream.set_done(to_global_lz77_end);
    // Quit the decode_block_header_kernel.
    huffman_handler.set_done();
    if (device_status)
    {
      *device_status = nvcompSuccess;
    }
  }
}

// Given a stream and an offset in that stream, return the next offset
// to the next block.
template <typename O2OVector>
__device__ void
lookahead_gzip_streaming_kernel(kernel_data_t<bufferType_t::RING> &kernel_data, nvcompStatus_t *device_status)
{
  timing_reporter.reset();
  timing_reporter.toggle(TimingReporter::Timer::TOTAL);
  const auto grid_dim_x = kernel_data.grid->get_total();

  // Note: the last sizeof(uint32_t) may be better templated
  auto min_input_bytes = 72 * sizeof(uint32_t) + grid_dim_x *
                                                   DecodeShared<O2OVector>(0, grid_dim_x).shmem_deflate_stream_count *
                                                   sizeof(uint32_t);

  InputFetcher<bufferType_t::RING> input_fetcher(std::move(kernel_data.deflate_stream), min_input_bytes);
  OutputStream<bufferType_t::RING> output_stream(std::move(kernel_data.uncompressed_stream));
  HuffmanHandler huffman_handler(*kernel_data.gzip_header_offset, std::move(kernel_data.huffman_mailbox));

  decompress<O2OVector>(
    std::move(input_fetcher),
    kernel_data.total_overlaps,
    kernel_data.block_bit_offset,
    std::move(huffman_handler),
    kernel_data.uncompressed_words,
    kernel_data.uncompressed_words_atomic,
    std::move(output_stream),
    *kernel_data.grid,
    device_status
  );
  timing_reporter.toggle(TimingReporter::Timer::TOTAL);
#if TIMING
  if (threadIdx.x == 0 && blockIdx.x == 0)
  {
    *kernel_data.timing_reporter = timing_reporter;
  }
#endif // TIMING
}

template <typename O2OVector>
__device__ void lookahead_gzip_oneshot_kernel(
  kernel_data_t<bufferType_t::LINEAR> &kernel_data,
  const uint32_t *device_compressed_ptr,
  const size_t &device_compressed_bytes,
  char *device_uncompressed_ptr,
  const size_t &device_uncompressed_buffer_bytes,
  size_t *device_uncompressed_chunk_bytes,
  nvcompStatus_t *device_status
)
{
  timing_reporter.reset();
  timing_reporter.toggle(TimingReporter::Timer::TOTAL);

  InputFetcher<bufferType_t::LINEAR> input_fetcher(device_compressed_ptr, device_compressed_bytes);
  OutputStream<bufferType_t::LINEAR>
    output_stream(device_uncompressed_ptr, device_uncompressed_buffer_bytes, device_uncompressed_chunk_bytes);
  HuffmanHandler huffman_handler(*kernel_data.gzip_header_offset, std::move(kernel_data.huffman_mailbox));

  decompress<O2OVector>(
    std::move(input_fetcher),
    kernel_data.total_overlaps,
    kernel_data.block_bit_offset,
    std::move(huffman_handler),
    kernel_data.uncompressed_words,
    kernel_data.uncompressed_words_atomic,
    std::move(output_stream),
    *kernel_data.grid,
    device_status
  );
  timing_reporter.toggle(TimingReporter::Timer::TOTAL);
#if TIMING
  if (threadIdx.x == 0 && blockIdx.x == 0)
  {
    *kernel_data.timing_reporter = timing_reporter;
  }
#endif // TIMING
}

template <bufferType_t T>
__global__ void get_header_length(
  kernel_data_t<T> *__restrict__ kernel_data,
  const uint32_t *const *__restrict__ device_compressed_ptrs,
  const int num_chunks,
  const int blocks_per_chunk
)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_chunks)
  {
    // Initialize PartialGrid on device
    kernel_data[idx].grid->init(blocks_per_chunk);

    auto deflate_stream = [&]() {
      if constexpr (T == bufferType_t::RING)
      {
        return kernel_data[idx].deflate_stream.buffer;
      }
      else
      {
        return device_compressed_ptrs[idx];
      }
    }();

    const auto *bytes = reinterpret_cast<const uint8_t *>(deflate_stream);

    // If there are no gzip magic numbers, we assume that the input is a raw deflate buffer.
    if (bytes[0] != 0x1F || bytes[1] != 0x8b)
    {
      *kernel_data[idx].gzip_header_offset = 0;
    }
    else
    {
      uint8_t flags = bytes[3]; // FLG byte is at offset 3
      int header_length = 10;
      if (flags & 4)
      {
        // FEXTRA is set. XLEN is a 16-bit little-endian value.
        size_t xlen = bytes[header_length] | (static_cast<size_t>(bytes[header_length + 1]) << 8);
        header_length += 2;
        header_length += xlen;
      }
      if (flags & 8)
      {
        // FNAME is set (null-terminated string).
        while (bytes[header_length] != 0)
        {
          header_length++;
        }
        header_length++;
      }
      if (flags & 16)
      {
        // FCOMMENT is set (null-terminated string).
        while (bytes[header_length] != 0)
        {
          header_length++;
        }
        header_length++;
      }
      if (flags & 2)
      {
        // FHCRC is set.
        header_length += 2;
      }
      *kernel_data[idx].gzip_header_offset = header_length * 8;
    }
  }
}

template <typename O2OVector>
__launch_bounds__(std::max(THREADS_PER_BLOCK, BLOCK_HEADER_THREADS_PER_BLOCK), 1) __global__
  void combined_streaming_kernel(
    kernel_data_t<bufferType_t::RING> *__restrict__ kernel_data,
    int chunk_offset,
    nvcompStatus_t *__restrict__ device_statuses
  )
{
  int current_chunk = blockIdx.y + chunk_offset;

  static_assert(
    THREADS_PER_BLOCK % WARP_SIZE_U == 0 && BLOCK_HEADER_THREADS_PER_BLOCK % WARP_SIZE_U == 0,
    "Prevent warp divergence"
  );

  if (blockIdx.x < gridDim.x - 1)
  {
    if (threadIdx.x >= THREADS_PER_BLOCK)
    {
      return;
    }
    lookahead_gzip_streaming_kernel<O2OVector>(
      kernel_data[current_chunk],
      device_statuses ? device_statuses + current_chunk : nullptr
    );
  }
  else
  {
    if (threadIdx.x >= BLOCK_HEADER_THREADS_PER_BLOCK)
    {
      return;
    }
    block_header::decode_block_header<
      false // Streaming mode ring buffer size is always divisible by 4 - no need for bounds checking
      >(kernel_data[current_chunk].deflate_stream, kernel_data[current_chunk].huffman_mailbox, ~31ull);
  }
}

template <typename O2OVector>
__launch_bounds__(std::max(THREADS_PER_BLOCK, BLOCK_HEADER_THREADS_PER_BLOCK), 1) __global__
  void combined_oneshot_kernel(
    kernel_data_t<bufferType_t::LINEAR> *__restrict__ kernel_data,
    const uint32_t *const *__restrict__ device_compressed_ptrs,
    const size_t *__restrict__ device_compressed_bytes,
    char *const *__restrict__ device_uncompressed_ptrs,
    const size_t *__restrict__ device_uncompressed_buffer_bytes,
    size_t *__restrict__ device_uncompressed_chunk_bytes,
    int chunk_offset,
    nvcompStatus_t *__restrict__ device_statuses
  )
{
  int current_chunk = blockIdx.y + chunk_offset;

  static_assert(
    THREADS_PER_BLOCK % WARP_SIZE_U == 0 && BLOCK_HEADER_THREADS_PER_BLOCK % WARP_SIZE_U == 0,
    "Prevent warp divergence"
  );

  if (blockIdx.x < gridDim.x - 1)
  {
    if (threadIdx.x >= THREADS_PER_BLOCK)
    {
      return;
    }
    lookahead_gzip_oneshot_kernel<O2OVector>(
      kernel_data[current_chunk],
      device_compressed_ptrs[current_chunk],
      device_compressed_bytes[current_chunk],
      device_uncompressed_ptrs[current_chunk],
      device_uncompressed_buffer_bytes[current_chunk],
      device_uncompressed_chunk_bytes ? device_uncompressed_chunk_bytes + current_chunk : nullptr,
      device_statuses ? device_statuses + current_chunk : nullptr
    );
  }
  else
  {
    if (threadIdx.x >= BLOCK_HEADER_THREADS_PER_BLOCK)
    {
      return;
    }
    block_header::decode_block_header<true>(
      device_compressed_ptrs[current_chunk],
      kernel_data[current_chunk].huffman_mailbox,
      device_compressed_bytes[current_chunk] * 8
    );
  }
}

} // namespace lookahead_gzip
