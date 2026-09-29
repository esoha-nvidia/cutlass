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

/*
  Handle lookbacks in arrays.

  An array can be written as a series of elements where some elements
  are values and some elements are pointers to other values in that
  array.  The pointers may form a chain but it's assumed that the
  structure is a "reverse tree" such that everntually every pointer,
  followed enough, will reach a value.

  The lookback algorithm resolves all these pointers in parallel using
  the disjoint-set union-find algorithm to resolve all elements to
  values.
*/
#pragma once

#include <type_traits>

#include <cooperative_groups.h> // Allows us to sync between blocks.
#include <stdint.h>
namespace cg = cooperative_groups;

#include "constants.cuh"

namespace lookback
{

enum DoWorkResultFlags
{
  WRITEBACK = 0x1,
  FINISHED = 0x2,
};

enum DoWorkResult
{
  NOT_FINISHED_NO_WRITEBACK = 0,
  NOT_FINISHED_WRITEBACK = WRITEBACK,
  FINISHED_NO_WRITEBACK = FINISHED,
  FINISHED_WRITEBACK = FINISHED | WRITEBACK,
};

// Returns true if this element is finished being processed.
template <typename element_array_t, typename element_t>
DoWorkResult do_work(element_array_t &elements, const size_t start, const size_t count, size_t index, element_t *mine);

template <typename element_array_t, typename destination_array_t>
inline __device__ void grid_lookback_atomics(
  element_array_t &elements,
  const size_t start,
  const size_t start_index,
  const size_t count,
  destination_array_t &dest,
  const uint dest_offset,
  uint *counter
)
{
  // This is the element that we'll start working on.
  uint current_index = start_index;
  // Do we have work to do?
  typename std::remove_reference<decltype(elements[0])>::type mine;
  if (current_index < count)
  {
    mine = elements[start + current_index];
  }
  while (current_index < count)
  {
    // We aren't off the end of the list yet.
    DoWorkResult result = do_work(elements, start, count, current_index, &mine);
    if (result & WRITEBACK)
    {
      // mine was updated, write it out.
      elements[start + current_index] = mine;
    }
    if (result & FINISHED)
    {
      dest[dest_offset + start + current_index] = mine.get_literal();
      // We finished the work, go to the next element.
      current_index = atomicAdd(counter, 1);
      if (current_index < count)
      {
        // Must reread.
        mine = elements[start + current_index];
      }
    }
  }
}

// Copy from src to dst with a single instruction but we don't care if
// other threads are also doing it.  Just so as all the bytes appear
// to be copied at the same time, to all threads.
template <typename D, typename S>
inline __device__ void atomic_copy(D *dst, const S &src)
{
  static_assert(
    std::is_same<typename std::remove_volatile<D>::type, typename std::remove_volatile<S>::type>::value,
    "The types for atomic_copy must match."
  );
  using atomic_type = typename std::conditional<
    sizeof(src) == sizeof(uint8_t),
    uint8_t,
    typename std::conditional<
      sizeof(src) == sizeof(uint16_t),
      uint16_t,
      typename std::conditional<
        sizeof(src) == sizeof(uint32_t),
        uint32_t,
        typename std::conditional<sizeof(src) == sizeof(uint64_t), uint64_t, nullptr_t>::type>::type>::type>::type;
  using atomic_src_type = typename std::
    conditional<std::is_volatile<S>::value, typename std::add_volatile<atomic_type>::type, atomic_type>::type;
  using atomic_dst_type = typename std::
    conditional<std::is_volatile<D>::value, typename std::add_volatile<atomic_type>::type, atomic_type>::type;
  static_assert(sizeof(src) == sizeof(atomic_type), "The sizes of the types being copied must match.");
  // Check that the input types are the same.
  *reinterpret_cast<atomic_dst_type *>(dst) = *reinterpret_cast<const atomic_src_type *>(&src);
}

// Returns true if this element is finished being processed.
template <typename element_array_t, typename element_t>
DoWorkResult do_work(
  volatile element_array_t &elements,
  const size_t start,
  const size_t count,
  const size_t max_lookback,
  size_t index,
  element_t *mine
);

// start is the first element in the array to process.  count is how
// many of them to process.  max_lookback is how far before start a
// lookback is allowed to be.  That is, element[start+i] cannot have a
// value that is looking back to element[start - max_lookback].  It
// must be element[start - max_lookback + 1] or later.  counter is a
// shared memory variable that is used with atomic operations.
template <uint THREADS_PER_BLOCK, bool WRITE_DEST, typename element_array_t, typename destination_array_t>
inline __device__ void block_lookback_atomics(
  volatile element_array_t &elements,
  const size_t start,
  const size_t count,
  const size_t max_lookback,
  destination_array_t &dest,
  uint dest_offset,
  uint *counter
)
{
  // This is the element that we'll start working on.
  uint current_index = threadIdx.x;
  // Do we have work to do?
  typename std::remove_volatile<typename std::remove_reference<decltype(elements[0])>::type>::type mine;
  if (current_index < count)
  {
    atomic_copy(&mine, elements[start + current_index]);
  }
  while (!HAZARDS && (*counter != count + THREADS_PER_BLOCK) || HAZARDS && (current_index < count))
  {
    // We aren't off the end of the list yet.
    DoWorkResult result = do_work(elements, start, count, max_lookback, current_index, &mine);
    if (!HAZARDS)
    {
      cooperative_groups::thread_block::sync();
    }
    if (result & WRITEBACK)
    {
      // mine was updated, write it out.
      atomic_copy(&elements[start + current_index], mine);
    }
    if (result & FINISHED)
    {
      if (WRITE_DEST)
      {
        atomic_copy(&dest[dest_offset + start + current_index], mine);
      }
      // We finished the work, go to the next element.
      current_index = atomicAdd(counter, 1);
    }
    if (!HAZARDS)
    {
      cooperative_groups::thread_block::sync();
    }
    if ((result & FINISHED) && current_index < count)
    {
      // Must reread.
      atomic_copy(&mine, elements[start + current_index]);
    }
  }
}

template <uint THREADS_PER_BLOCK, typename element_array_t>
inline __device__ void block_lookback_atomics(
  element_array_t &elements,
  const size_t start,
  const size_t count,
  const size_t max_lookback,
  uint *counter
)
{
  block_lookback_atomics<THREADS_PER_BLOCK, false>(elements, start, count, max_lookback, elements, 0, counter);
}

} // namespace lookback
