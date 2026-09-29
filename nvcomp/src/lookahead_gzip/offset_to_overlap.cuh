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

#include "common.cuh"
#include "constants.cuh"
#include "fixed_array.cuh"
#include "lookback.cuh"

#include <stdint.h>

template <typename overlap_t>
class UncompressedSizeOverlap
{
public:
  using value_type = overlap_t;
  inline __host__ __device__ bool is_error() const { return overlap == (1 << (sizeof(overlap_t) * 8 - 1)); }
  inline __host__ __device__ bool is_end_of_block() const { return overlap > (1 << (sizeof(overlap_t) * 8 - 1)); }
  inline __host__ __device__ bool is_error_or_end_of_block() const
  {
    return overlap >= (1 << (sizeof(overlap_t) * 8 - 1));
  }
  // Only call this if is_error_or_end_of_block() returns false.
  inline __host__ __device__ bool is_copy() const { return overlap >= MAX_OFFSET; }
  // Only call this if is_error_or_end_of_block() returns false and
  // is_copy() returns false.
  inline __host__ __device__ uint get_huffman_tree_index() const { return 0; }
  // Only call this if is_end_of_block() returns true.
  inline __host__ __device__ uint get_end_of_block_stride_bits() const
  {
    return overlap & ~(1 << (sizeof(overlap_t) * 8 - 1));
  }
  // Only call this if is_error_or_end_of_block() returns false and
  // is_copy() returns false.
  inline __host__ __device__ uint get_overlap() const { return overlap % MAX_OFFSET; }
  inline __host__ __device__ uint get_raw_overlap() const { return overlap; }
  // Only call this if is_copy() returns true.
  inline __host__ __device__ uint get_copy() const { return overlap - MAX_OFFSET; }
  inline __host__ __device__ void set_raw_overlap(overlap_t new_overlap) { overlap = new_overlap; }
  inline __host__ __device__ void set_error_or_end_of_block(uint iteration_offset, overlap_t stride_bits)
  {
    if (stride_bits == 0)
    {
      iteration_offset = 0;
    }
    overlap = (iteration_offset + stride_bits) | (1 << (sizeof(overlap_t) * 8 - 1));
  }
  inline __host__ __device__ void set_overlap(uint huffman_tree_index, overlap_t new_overlap)
  {
    overlap = huffman_tree_index * MAX_OFFSET + new_overlap;
  }
  inline __host__ __device__ void set_copy(uint huffman_tree_index, overlap_t new_overlap)
  {
    overlap = huffman_tree_index * MAX_OFFSET + new_overlap + MAX_OFFSET;
  }
  __host__ __device__ uint32_t get_uncompressed_size() const { return uncompressed_size; }
  __host__ __device__ void set_uncompressed_size(uint32_t new_uncompressed_size)
  {
    uncompressed_size = new_uncompressed_size;
  }
  __host__ __device__ void set_additive_identity(overlap_t new_overlap)
  {
    overlap = new_overlap;
    uncompressed_size = 0;
  }
  // Call this only when you already know that left is not an error
  // nor end of block.
  inline __device__ void
  add(const UncompressedSizeOverlap<overlap_t> &left, const UncompressedSizeOverlap<overlap_t> &right)
  {
    overlap = right.overlap;
    uncompressed_size = left.uncompressed_size + right.uncompressed_size;
  }
  __host__ __device__ void print() const
  {
    printf(
      "%-8s %5d uncompressed_size: %5d",
      is_error()                             ? "err"
      : is_end_of_block()                    ? "end"
      : is_copy() && get_copy() > MAX_OFFSET ? "copy len"
      : is_copy()                            ? "copy"
      : (get_huffman_tree_index() == 1)      ? "     len"
                                             : "",

      is_error()          ? 0
      : is_end_of_block() ? get_end_of_block_stride_bits()
      : is_copy()         ? get_copy() % MAX_OFFSET
                          : get_overlap(),
      get_uncompressed_size()
    );
  }

private:
  overlap_t overlap;
  uint32_t uncompressed_size;
};

// This stores the minimum number of bits that must be processed
// starting at a bit position in a stride in order to reach the next
// stride.  The value is usually in the range
// [STRIDE_BITS-start_bit..STRIDE_BITS-start_bit+MAX_OFFSET)
// where start_bit is the position within the stride where the search
// began.  So if the stride starts at bit position 100 and we're
// computing the 4th iteration then we'll start decoding symbols at
// position 104.  If the stride is 500 bits then the total overlap is
// at least 500-4 and up to 500-4+MAX_OFFSET.
//
// We need to know if there was an error or we found end-of-block.
// For those cases, the MSB is used.  Error will have the rest of the
// bits 0 and end-of-block will have the non-zero number of bits that
// were read to complete the end-of-block.  It's not possible for
// end-of-block to happen with zero symbols read so there is no
// overlap in the representations.
//
// Each stride has multiple overlaps computed, one for each
// MAX_OFFSET, so this will store the results for the entire stride.
template <class overlap_t>
class O2OVector
{
public:
  using value_type = overlap_t;
  template <uint INDEX_COUNT>
  inline __device__ void additive_identity(uint index)
  {
    for (uint i = index; i < MAX_OFFSET; i += INDEX_COUNT)
    {
      overlap[i].set_additive_identity(i);
    }
  }
  template <uint INDEX_COUNT>
  inline __device__ void partial_copy(const uint i, const O2OVector &in)
  {
    static_assert(
      sizeof(*this) % sizeof(copy_type) == 0,
      "Data to copy should be a multiple of the copy_type "
      "in size."
    );
    for (unsigned int offset = 0; offset < sizeof(*this) / sizeof(copy_type); offset += INDEX_COUNT)
    {
      uint current_index = offset + i;
      bool predicate = current_index < sizeof(*this) / sizeof(copy_type);
      if (sizeof(*this) / sizeof(copy_type) % INDEX_COUNT == 0)
      {
        __builtin_assume(predicate);
      }
      if (predicate)
      {
        reinterpret_cast<copy_type *>(this)[current_index] = (reinterpret_cast<const copy_type *>(&in))[current_index];
      }
    }
  }
  template <uint INDEX_COUNT>
  inline __device__ void partial_add(const uint i, const O2OVector &left, const O2OVector &right)
  {
    for (unsigned int index = i; index < MAX_OFFSET; index += INDEX_COUNT)
    {
      if (MAX_OFFSET % INDEX_COUNT == 0)
      {
        __builtin_assume(index < MAX_OFFSET);
      }
      if (index < MAX_OFFSET)
      {
        auto left_overlap = left.overlap[index];
        if (left_overlap.is_error_or_end_of_block())
        {
          overlap[index] = left_overlap;
        }
        else
        {
          overlap[index].add(left_overlap, right.overlap[left_overlap.get_raw_overlap()]);
        }
      }
    }
  }
  __host__ __device__ void print() const
  {
    for (unsigned int offset = 0; offset < MAX_OFFSET; offset++)
    {
      auto index = offset;
      printf("%3s %2d: ", "", offset);
      overlap[index].print();
      printf("\n");
    }
  }
  inline __device__ overlap_t &operator[](uint index) { return overlap[index]; }
  inline const __device__ overlap_t &operator[](uint index) const { return overlap[index]; }
  inline volatile __device__ overlap_t &operator[](uint index) volatile { return overlap[index]; }
  inline const volatile __device__ overlap_t &operator[](uint index) const volatile { return overlap[index]; }

protected:
  // If the MSB is not set, this is the overlap into the next stride +
  // MAX_OFFSET if this is is_length.
  //
  // If MSB is set then this the rest of the bits, if they are zero,
  // indicate an error.  If the rest of the bits are not 0, that means
  // that an end-of-block was met and the rest of the bits encode the
  // number of bits decoded plus the offset as an unsigned 7-bit int.
  //
  // If the MSB is not set then the value is simply an index into the
  // next O2OVector.
  overlap_t overlap[MAX_OFFSET];
};

namespace lookback
{

template <>
inline __device__ DoWorkResult do_work<volatile OVERLAP_TYPE *>(
  volatile OVERLAP_TYPE *volatile &elements,
  const size_t start,
  const size_t count,
  const size_t max_lookback,
  const size_t index,
  OVERLAP_TYPE *mine
)
{
  uint stride_id = index / MAX_OFFSET;
  if (index >= count)
  {
    return NOT_FINISHED_NO_WRITEBACK;
  }
  if (mine->is_error_or_end_of_block() || !mine->is_copy())
  {
    return FINISHED_NO_WRITEBACK;
  }
  // By this point, it must be a copy.
  OVERLAP_TYPE prev;
  atomic_copy(&prev, elements[stride_id * MAX_OFFSET + mine->get_copy()]);
  mine->add(*mine, prev);
  if (mine->is_error_or_end_of_block() || !mine->is_copy())
  {
    return FINISHED_WRITEBACK;
  }
  // Otherwise it was a pointer to a pointer.
  return NOT_FINISHED_WRITEBACK;
}

} // namespace lookback
