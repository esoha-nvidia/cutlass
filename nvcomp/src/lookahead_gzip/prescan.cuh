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

// Prescan routines from:
// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
#pragma once

#include <cooperative_groups.h> // Allows us to sync between blocks.
#include <stdint.h>
#include <stdio.h>
namespace cg = cooperative_groups;

#include "common.cuh"

// Set ret to the additive_identity for the prefix_sum using exactly
// INDEX_COUNT threads by calling this function with index from 0
// through INDEX_COUNT-1.  You must implement this function for the
// element type of the prefix sum.
template <uint INDEX_COUNT, typename T>
inline __device__ void additive_identity(uint index, T *ret)
{
  ret->additive_identity<INDEX_COUNT>(index);
}

// Copy src into dst using exactly INDEX_COUNT threads by calling this
// function with index from 0 through INDEX_COUNT-1.  You must
// implement this function for the element type of the prefix sum.
template <uint INDEX_COUNT, typename T>
inline __device__ void partial_copy(uint index, const T &src, T *dst)
{
  dst->partial_copy<INDEX_COUNT>(index, src);
}

// Perform the non-commutative addition of operand0 and operand1 and
// store into dst using exactly INDEX_COUNT threads by calling this
// function with index from 0 through INDEX_COUNT-1.  You must
// implement this function for the element type of the prefix sum.
template <uint INDEX_COUNT, typename T>
inline __device__ void partial_add(uint index, const T &operand0, const T &operand1, T *dst)
{
  dst->partial_add<INDEX_COUNT>(index, operand0, operand1);
}

template <>
inline __device__ void additive_identity<1, uint32_t>(uint index, uint32_t *ret)
{
  if (index == 0)
  {
    *ret = 0;
  }
}

template <>
inline __device__ void additive_identity<1, uint16_t>(uint index, uint16_t *ret)
{
  if (index == 0)
  {
    *ret = 0;
  }
}

template <>
inline __device__ void partial_copy<1, uint32_t>(uint index, const uint32_t &src, uint32_t *dst)
{
  if (index == 0)
  {
    *dst = src;
  }
}

template <>
inline __device__ void
partial_add<1, uint32_t>(uint index, const uint32_t &operand0, const uint32_t &operand1, uint32_t *dst)
{
  if (index == 0)
  {
    *dst = operand0 + operand1;
  }
}

template <uint THREADS>
inline __device__ void sync_warp_or_threads()
{
  if (THREADS < WARP_SIZE_U)
  {
    __syncwarp();
  }
  else
  {
    cooperative_groups::thread_block::sync();
  }
}

// In-place prescan where threads [0..COUNT) are in a single
// warp/block.  The data is prescanned in place with operator+.  The
// return value is the total sum.
//
// This prescan supports non-power-of-two sized arrays by using the
// theory here:
// https://gitlab-master.nvidia.com/GPUDB/compression/lookahead_gzip/-/issues/15#note_8141542
template <uint COUNT, typename T>
static __device__ void block_prescan(T *data, T *total)
{
  const uint thid = threadIdx.x;
  int offset = 1;
  for (int d = upper_power_of_two(COUNT) >> 1; d > 0; d >>= 1)
  {
    // build sum in place up the tree
    int ai = offset * (2 * thid + 1) - 1;
    int bi = offset * (2 * thid + 2) - 1;
    bi = min(bi, COUNT - 1);
    if (thid < d && ai < COUNT - 1)
    {
      data[bi] = data[ai] + data[bi];
    }
    sync_warp_or_threads<COUNT>();
    offset *= 2;
  }
  if (thid == 0)
  {
    *total = data[COUNT - 1];
  }
  sync_warp_or_threads<COUNT>();
  if (thid == 0)
  {
    additive_identity<1>(0, &data[COUNT - 1]); // clear the last element
  }
  for (int d = 1; d < upper_power_of_two(COUNT); d *= 2)
  {
    // traverse down tree & build scan
    offset >>= 1;
    sync_warp_or_threads<COUNT>();
    int ai = offset * (2 * thid + 1) - 1;
    int bi = offset * (2 * thid + 2) - 1;
    bi = min(bi, COUNT - 1);
    T t;
    if (thid < d && ai < COUNT - 1)
    {
      t = data[bi] + data[ai];
      data[ai] = data[bi];
      data[bi] = t;
    }
  }
}

// This block prescan allows you to specify how to divide up the work
// of the prescan.  The number of threads that will be devoted to each
// add is the ADD_INDEX_COUNT.  The number of threads for each COPY
// will be the COPY_INDEX_COUNT.  It's up to the type T to decide if
// those should be done striped or blockwise.  The aggregate is put
// into aggregate and it will handle for COUNT being less than
// SIZEOF_DATA.  aggregate should be a shared variable.  aggregate is
// only filled in if the COUNT == SIZEOF_DATA.  If it's not then you
// can just get your own aggregate from inside the data.
//
// This prescan supports non-power-of-two sized arrays by using the
// theory here:
// https://gitlab-master.nvidia.com/esoha/lookahead_gzip/-/issues/15#note_8141542
//
// add_result should have at least upper_power_of_two(COUNT)/2 elements.
template <
  uint THREADS_PER_BLOCK, // Total number of threads in this block.
  uint COUNT, // How many elements in data?
  uint ADD_INDEX_COUNT, // How many threads per element addition.
  uint COPY_INDEX_COUNT, // How many threads per element copy.
  typename T>
static __device__ void block_prescan_up_kernel(T data[COUNT], T *aggregate, T add_result[COUNT / 2])
{
  constexpr uint MAX_INDEX_COUNT = const_max(ADD_INDEX_COUNT, COPY_INDEX_COUNT);
  const uint thid = threadIdx.x / MAX_INDEX_COUNT;
  const uint index = threadIdx.x % MAX_INDEX_COUNT;
  static_assert(
    THREADS_PER_BLOCK >= upper_power_of_two(COUNT) / 2 * MAX_INDEX_COUNT,
    "Need to have enough threads to compute prefix sum."
  );
  int offset = 1;
  for (int d = upper_power_of_two(COUNT) >> 1; d > 0; d >>= 1)
  {
    // build sum in place up the tree
    int ai = offset * (2 * thid + 1) - 1;
    int bi = offset * (2 * thid + 2) - 1;
    bi = min(bi, COUNT - 1);
    if (thid < d && ai < COUNT - 1 && index < ADD_INDEX_COUNT)
    {
      partial_add<ADD_INDEX_COUNT>(index, data[ai], data[bi], &add_result[thid]);
    }
    cooperative_groups::thread_block::sync();
    if (thid < d && ai < COUNT - 1 && index < COPY_INDEX_COUNT)
    {
      partial_copy<COPY_INDEX_COUNT>(index, add_result[thid], &data[bi]);
    }
    cooperative_groups::thread_block::sync();
    offset *= 2;
  }
  if (thid == 0 && index < COPY_INDEX_COUNT)
  {
    partial_copy<COPY_INDEX_COUNT>(index, data[COUNT - 1], aggregate);
  }
}

template <
  uint THREADS_PER_BLOCK, // Total number of threads in this block.
  uint COUNT, // How many elements in data?
  uint ADD_INDEX_COUNT, // How many threads per element addition.
  uint COPY_INDEX_COUNT, // How many threads per element copy.
  typename T>
static __device__ void block_prescan_down_kernel(T data[COUNT], T *, T add_result[COUNT / 2])
{
  constexpr uint MAX_INDEX_COUNT = const_max(ADD_INDEX_COUNT, COPY_INDEX_COUNT);
  const uint thid = threadIdx.x / MAX_INDEX_COUNT;
  const uint index = threadIdx.x % MAX_INDEX_COUNT;
  if (thid == 0 && index < COPY_INDEX_COUNT)
  {
    additive_identity<COPY_INDEX_COUNT>(index, &data[COUNT - 1]); // clear the last element
  }
  int offset = upper_power_of_two(COUNT);
  for (int d = 1; d < upper_power_of_two(COUNT); d *= 2)
  {
    // traverse down tree & build scan
    offset >>= 1;
    int ai = offset * (2 * thid + 1) - 1;
    int bi = offset * (2 * thid + 2) - 1;
    bi = min(bi, COUNT - 1);
    cooperative_groups::thread_block::sync();
    if (thid < d && ai < COUNT - 1 && index < ADD_INDEX_COUNT)
    {
      partial_add<ADD_INDEX_COUNT>(index, data[bi], data[ai], &add_result[thid]);
    }
    cooperative_groups::thread_block::sync();
    if (thid < d && ai < COUNT - 1 && index < COPY_INDEX_COUNT)
    {
      partial_copy<COPY_INDEX_COUNT>(index, data[bi], &data[ai]);
      partial_copy<COPY_INDEX_COUNT>(index, add_result[thid], &data[bi]);
    }
  }
}

template <
  uint THREADS_PER_BLOCK, // Total number of threads in this block.
  uint COUNT, // How many elements in data?
  uint ADD_INDEX_COUNT, // How many threads per element addition.
  uint COPY_INDEX_COUNT, // How many threads per element copy.
  typename T>
static __device__ void block_prescan_kernel(T data[COUNT], T *aggregate, T add_result[COUNT / 2])
{
  block_prescan_up_kernel<THREADS_PER_BLOCK, COUNT, ADD_INDEX_COUNT, COPY_INDEX_COUNT>(data, aggregate, add_result);
  cooperative_groups::thread_block::sync();
  block_prescan_down_kernel<THREADS_PER_BLOCK, COUNT, ADD_INDEX_COUNT, COPY_INDEX_COUNT>(data, aggregate, add_result);
}

static constexpr __host__ __device__ uint shared_data_size(uint count, uint elements_per_block)
{
  // elements_per_block must be at least 2 and count must be at least
  // 1.
  count = nvcomp::roundUpDiv(count, elements_per_block);
  uint ret = 1;
  while (count != 1)
  {
    ret++;
    count = nvcomp::roundUpDiv(count, elements_per_block);
  }
  return ret;
}

// This prescans the data that is in aggregates.  aggregates is also
// the scratch place for doing work so it must be at least as big as
// the original size of aggregates.  aggregates should be in global
// memory and we'll use a grid.sync() to synchronize it.  The
// aggregate of the entire prescan will be written at the memory
// location aggregates[x] where x is the returned value.
template <
  uint THREADS_PER_BLOCK, // How many threads in the threadblock?
  uint ADD_INDEX_COUNT, // How many threads to use per add operation?
  uint COPY_INDEX_COUNT, // How many threads to use per copy operation?
  class T,
  typename grid_t>
__device__ uint iterative_prescan_up(
  T *aggregates,
  grid_t &grid,
  T *shared_data,
  T shared_temps[THREADS_PER_BLOCK / const_max(ADD_INDEX_COUNT, COPY_INDEX_COUNT)],
  const uint count
)
{ // How many elements in aggregates?
  // shared data ought to have size:
  //   shared_data_size(COUNT, THREADS_PER_BLOCK/const_max(ADD_INDEX_COUNT, COPY_INDEX_COUNT)) *
  //   (THREADS_PER_BLOCK/const_max(ADD_INDEX_COUNT, COPY_INDEX_COUNT)) + 1

  // First prescan the overlaps.  Each block will prescan a portion.
  // Without prescan:
  //T aggregate;
  constexpr uint THREADS_PER_ELEMENT = const_max(ADD_INDEX_COUNT, COPY_INDEX_COUNT);
  constexpr uint ELEMENTS_PER_BLOCK = THREADS_PER_BLOCK / THREADS_PER_ELEMENT;
  const uint element_id_in_block = threadIdx.x / THREADS_PER_ELEMENT;
  const uint index = threadIdx.x % THREADS_PER_ELEMENT;
  const uint element_id_in_grid = blockIdx.x * ELEMENTS_PER_BLOCK + element_id_in_block;

  uint current_count = count;
  uint ret = 0;
  uint shared_data_offset = 0;
  while (true)
  {
    // We need to clear out the unused elements because addition may not
    // be defined on garbage and it might crash.
    if (element_id_in_block < ELEMENTS_PER_BLOCK && index < COPY_INDEX_COUNT)
    {
      additive_identity<COPY_INDEX_COUNT>(index, &shared_data[shared_data_offset + element_id_in_block]);
    }
    cooperative_groups::thread_block::sync();
    // First copy the right amount of aggregates into shared_data.  If
    // there is a runt, make sure to fill the rest of the array with the
    // additive identity so that we don't try to add two invalid entries
    // together, which may crash.
    if (element_id_in_block < ELEMENTS_PER_BLOCK && element_id_in_grid < current_count && index < COPY_INDEX_COUNT)
    {
      partial_copy<COPY_INDEX_COUNT>(
        index,
        aggregates[element_id_in_grid],
        &shared_data[shared_data_offset + element_id_in_block]
      );
    }
    cooperative_groups::thread_block::sync();
    block_prescan_up_kernel<THREADS_PER_BLOCK, ELEMENTS_PER_BLOCK, ADD_INDEX_COUNT, COPY_INDEX_COUNT>(
      shared_data + shared_data_offset,
      &shared_data[shared_data_offset + ELEMENTS_PER_BLOCK],
      shared_temps
    );
    // How many aggregates did we produce?  It's
    // current_count/ELEMENTS_PER_BLOCK but rounded up.
    uint aggregates_count = (current_count + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
    cooperative_groups::thread_block::sync();

    // Now copy the aggregates into the array at the next locations.
    if (element_id_in_block == 0 && blockIdx.x < aggregates_count && index < COPY_INDEX_COUNT)
    {
      partial_copy<COPY_INDEX_COUNT>(
        index,
        shared_data[shared_data_offset + ELEMENTS_PER_BLOCK],
        &aggregates[current_count + blockIdx.x]
      );
    }
    if (aggregates_count == 1)
    {
      // We're done.
      return ret + current_count;
    }
    else
    {
      // Otherwise we need to iterate.
      grid.sync(); // Let all those aggregates be written.
      // The final_index is the index into aggregates+COUNT to find the
      // big total.
      aggregates += current_count;
      shared_data_offset += ELEMENTS_PER_BLOCK;
      ret += current_count;
      current_count = aggregates_count;
    }
  }
}

template <
  uint THREADS_PER_BLOCK, // How many threads in the threadblock?
  uint ADD_INDEX_COUNT, // How many threads to use per add operation?
  uint COPY_INDEX_COUNT, // How many threads to use per copy operation?
  class T,
  typename grid_t>
__device__ void iterative_prescan_down(
  T *aggregates,
  grid_t &grid,
  T *shared_data,
  T shared_temps[THREADS_PER_BLOCK / const_max(ADD_INDEX_COUNT, COPY_INDEX_COUNT)],
  const uint count
)
{
  // shared data ought to have size:
  //   shared_data_size(COUNT, THREADS_PER_BLOCK/const_max(ADD_INDEX_COUNT, COPY_INDEX_COUNT)) *
  //   (THREADS_PER_BLOCK/const_max(ADD_INDEX_COUNT, COPY_INDEX_COUNT)) + 1
  constexpr uint THREADS_PER_ELEMENT = const_max(ADD_INDEX_COUNT, COPY_INDEX_COUNT);
  constexpr uint ELEMENTS_PER_BLOCK = THREADS_PER_BLOCK / THREADS_PER_ELEMENT;

  const uint total_counts = shared_data_size(count, ELEMENTS_PER_BLOCK);
  ;
  uint target_counts_index = total_counts;
  while (target_counts_index > 0)
  {
    uint current_count = count;
    uint previous_count = 0;
    uint counts_index = 1;
    while (counts_index != target_counts_index)
    {
      uint delta = nvcomp::roundUpDiv((current_count - previous_count), ELEMENTS_PER_BLOCK);
      previous_count = current_count;
      current_count += delta;
      counts_index++;
    }

    const uint element_id_in_block = threadIdx.x / THREADS_PER_ELEMENT;
    const uint index = threadIdx.x % THREADS_PER_ELEMENT;
    const uint element_id_in_grid = blockIdx.x * ELEMENTS_PER_BLOCK + element_id_in_block + previous_count;
    // The data in shared memory was prescaned up for each block but it
    // wasn't down scanned.
    block_prescan_down_kernel<THREADS_PER_BLOCK, ELEMENTS_PER_BLOCK, ADD_INDEX_COUNT, COPY_INDEX_COUNT>(
      shared_data + ELEMENTS_PER_BLOCK * (counts_index - 1),
      &shared_data[ELEMENTS_PER_BLOCK * (counts_index - 1) + ELEMENTS_PER_BLOCK],
      shared_temps
    );
    cooperative_groups::thread_block::sync();
    if (counts_index == total_counts)
    {
      // We don't need to add in the result of the prescan of the
      // aggregate because there is only one so it would just be the
      // identity element and adding that is like adding 0, it does
      // nothing.  We still need to copy it into global memory.
      if (element_id_in_block < ELEMENTS_PER_BLOCK && element_id_in_grid < current_count && index < COPY_INDEX_COUNT)
      {
        partial_copy<COPY_INDEX_COUNT>(
          index,
          shared_data[element_id_in_block + ELEMENTS_PER_BLOCK * (counts_index - 1)],
          &aggregates[element_id_in_grid]
        );
      }
    }
    else
    {
      // We can assume at the aggregates at aggregates[counts[counts_index]]
      // ... aggregates[counts[counts_index]+aggregates_count] is all set.  We need to
      // add those in.
      if (element_id_in_block < ELEMENTS_PER_BLOCK && element_id_in_grid < current_count && index < ADD_INDEX_COUNT)
      {
        partial_add<ADD_INDEX_COUNT>(
          index,
          aggregates[current_count + blockIdx.x],
          shared_data[element_id_in_block + ELEMENTS_PER_BLOCK * (counts_index - 1)],
          &shared_temps[element_id_in_block]
        );
      }
      cooperative_groups::thread_block::sync(); // Wait for the addition to complete.
      if (element_id_in_block < ELEMENTS_PER_BLOCK && element_id_in_grid < current_count && index < COPY_INDEX_COUNT)
      {
        partial_copy<COPY_INDEX_COUNT>(index, shared_temps[element_id_in_block], &aggregates[element_id_in_grid]);
      }
    }
    grid.sync();
    target_counts_index--;
  }
}

template <
  uint THREADS_PER_BLOCK, // How many threads in the threadblock?
  uint COUNT, // How many elements in aggregates?
  uint ADD_INDEX_COUNT, // How many threads to use per add operation?
  uint COPY_INDEX_COUNT, // How many threads to use per copy operation?
  class T,
  typename grid_t>
__device__ uint iterative_prescan(
  T *aggregates,
  grid_t &grid,
  T shared_data
    [shared_data_size(COUNT, THREADS_PER_BLOCK / const_max(ADD_INDEX_COUNT, COPY_INDEX_COUNT)) *
       (THREADS_PER_BLOCK / const_max(ADD_INDEX_COUNT, COPY_INDEX_COUNT)) +
     1],
  T shared_temps[THREADS_PER_BLOCK / const_max(ADD_INDEX_COUNT, COPY_INDEX_COUNT)]
)
{
  uint final_index = iterative_prescan_up<THREADS_PER_BLOCK, COUNT, ADD_INDEX_COUNT, COPY_INDEX_COUNT>(
    aggregates,
    grid,
    shared_data,
    shared_temps,
    COUNT
  );
  grid.sync();
  iterative_prescan_down<THREADS_PER_BLOCK, COUNT, ADD_INDEX_COUNT, COPY_INDEX_COUNT>(
    aggregates,
    grid,
    shared_data,
    shared_temps
  );
  return final_index;
}

// Prescan all the shared memory in all the threadblocks as if they
// were part of one big array.  Only do the upscan, which is a
// reduction.  shared_data size is:
/*
  constexpr uint THREADS_PER_ELEMENT = const_max(THREADS_PER_O2O_ADD, THREADS_PER_O2O_COPY);
  constexpr uint PRESCAN_ELEMENTS_PER_BLOCK = THREADS_PER_BLOCK/THREADS_PER_ELEMENT;
  constexpr uint block_prescan_width = upper_power_of_two(PRESCAN_ELEMENTS_PER_BLOCK);
  __shared__ O2OVector shared_data[
      shared_data_size(GRID_DIM_X, PRESCAN_ELEMENTS_PER_BLOCK) *
      (block_prescan_width+(block_prescan_width == PRESCAN_ELEMENTS_PER_BLOCK))];
*/
template <
  uint THREADS_PER_BLOCK,
  uint ELEMENTS_PER_BLOCK,
  uint THREADS_PER_ADD,
  uint THREADS_PER_COPY,
  typename element_t,
  typename grid_t>
inline __device__ void grid_prescan_up(
  element_t elements[ELEMENTS_PER_BLOCK],
  element_t *aggregates,
  element_t *final,
  grid_t &grid,
  element_t shared_temps[THREADS_PER_BLOCK / const_max(THREADS_PER_ADD, THREADS_PER_COPY)],
  element_t *shared_data,
  const uint blocks
)
{
  // We expect that the size of shared_data is
  //    shared_data_size(BLOCKS, THREADS_PER_BLOCK/const_max(THREADS_PER_ADD, THREADS_PER_COPY)) *
  //    (THREADS_PER_BLOCK/const_max(THREADS_PER_ADD, THREADS_PER_COPY)) + 1
  // Shared memory reduction per CTA
  block_prescan_kernel<THREADS_PER_BLOCK, ELEMENTS_PER_BLOCK, THREADS_PER_ADD, THREADS_PER_COPY>(
    elements,
    &shared_data[0],
    shared_temps
  );

  // Now we have each block prescaned.  Write the aggregates into
  // the aggregates array.
  cooperative_groups::thread_block::sync();
  // Writing it out now to global mem, each CTA does this
  if (threadIdx.x < THREADS_PER_COPY)
  {
    aggregates[blockIdx.x].partial_copy<THREADS_PER_COPY>(threadIdx.x, shared_data[0]);
  }

  // All collaboring CTAs wrote out their reduced results,
  // now we are left with block-count number of results in global memory
  grid.sync();

  uint final_index = iterative_prescan_up<THREADS_PER_BLOCK, THREADS_PER_ADD, THREADS_PER_COPY>(
    aggregates,
    grid,
    shared_data,
    shared_temps,
    blocks
  );

  // We need to have the final go to every block.
  grid.sync();
  if (threadIdx.x < THREADS_PER_COPY)
  {
    final->partial_copy<THREADS_PER_COPY>(threadIdx.x, aggregates[final_index]);
  }
}

template <
  uint THREADS_PER_BLOCK,
  uint ELEMENTS_PER_BLOCK,
  uint THREADS_PER_ADD,
  uint THREADS_PER_COPY,
  typename element_t,
  typename grid_t>
inline __device__ void grid_prescan_down(
  element_t elements[ELEMENTS_PER_BLOCK],
  element_t *aggregates,
  const element_t *,
  grid_t &grid,
  element_t shared_temps[THREADS_PER_BLOCK / const_max(THREADS_PER_ADD, THREADS_PER_COPY)],
  element_t *shared_data,
  const uint blocks
)
{
  constexpr uint THREADS_PER_ELEMENT = const_max(THREADS_PER_ADD, THREADS_PER_COPY);
  const uint element_id_in_block = threadIdx.x / THREADS_PER_ELEMENT;
  const uint index = threadIdx.x % THREADS_PER_ELEMENT;
  static_assert(
    THREADS_PER_BLOCK >= ELEMENTS_PER_BLOCK * THREADS_PER_ELEMENT,
    "Need to have enough threads to compute prefix sum."
  );

  iterative_prescan_down<THREADS_PER_BLOCK, THREADS_PER_ADD, THREADS_PER_COPY>(
    aggregates,
    grid,
    shared_data,
    shared_temps,
    blocks
  );
  grid.sync();
  if (index < THREADS_PER_ADD && element_id_in_block < ELEMENTS_PER_BLOCK)
  {
    shared_temps[element_id_in_block]
      .partial_add<THREADS_PER_ADD>(index, aggregates[blockIdx.x], elements[element_id_in_block]);
  }
  cooperative_groups::thread_block::sync();
  if (index < THREADS_PER_COPY && element_id_in_block < ELEMENTS_PER_BLOCK)
  {
    elements[element_id_in_block].partial_copy<THREADS_PER_COPY>(index, shared_temps[element_id_in_block]);
  }
}
