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

#include <cuda/atomic>

#include <type_traits>

#include "cascaded_hash.cuh"

namespace cascaded
{

// Threads cannot insert a value and assign an index in the same atomic operation
// The first thread to insert a value into the set also handles index assignment
// Threads with duplicate values may have to spin when waiting for index assignments
constexpr int STATIC_SET_SHORT_SLEEP_NS = 100; // used when waiting for index assignments

constexpr int STATIC_SET_TARGET_INVERSE_LOAD_FACTOR = 2; // Used when allocating space for the static set

// TODO: this could be all turned into a class with member functions, but block/warp usage would have to be clear.
template <typename data_t, typename index_t>
struct static_set
{
  // Prevent confusing atomicCAS errors when trying to use unsupported types.
  static_assert(
    std::is_same<data_t, unsigned int>::value || std::is_same<data_t, unsigned long long>::value,
    "static_set: data_t must be `unsigned int` (uint32_t) or `unsigned long long` "
    "to match CUDA's atomicCAS overloads."
  );

  // Ensure index_t type is one that has been tested (uint8_t, uint16_t, uint32_t, unsigned long long)
  static_assert(
    std::is_same<index_t, uint8_t>::value || std::is_same<index_t, uint16_t>::value ||
      std::is_same<index_t, uint32_t>::value || std::is_same<index_t, unsigned long long>::value,
    "static_set: index_t must be `uint8_t`, `uint16_t`, or `uint32_t`. If you want to use a different type, please add "
    "static_assert support and test it in test_cascaded_static_set.cu."
  );

  data_t *bucket_to_data_map; // used for checking if a value is present in the set
  index_t *bucket_to_index_map; // used for retrieving the unique index once the data is found or inserted
  uint32_t *shared_member_counter; // counter used to atomically assign unique indices
  uint32_t num_buckets;
  data_t sentinel_value; // the value assigned to index 0. Used to mark when a bucket is empty
};

/**
 *  @return the alignment requirement for the static set 
 */
inline __host__ __device__ uint32_t block_static_set_get_scratch_alignment_req()
{
  // in the future we may want to benefit from cache line locality, that will require alignment
  return 8;
}

/**
 *  @return the number of bytes required to make a static set for the given cardinality
 */
template <typename data_t, typename index_t>
inline __host__ __device__ size_t block_static_set_get_scratch_req(uint32_t estimated_num_unique_values)
{
  // if we change the probing strategy, then we may need to round to cache line and change alignment req
  return (sizeof(data_t) + sizeof(index_t)) * static_cast<size_t>(estimated_num_unique_values) *
         static_cast<size_t>(STATIC_SET_TARGET_INVERSE_LOAD_FACTOR);
}

/** 
 *  The entire CTA collaborates to initialize a static set. The set initially contains the sentinel value assigned to index 0
 *
 *  @param static_set convenience structure for wrapping all static set variables
 *  @param scratch_ptr pointer to the scratch space used to store the static set. Must be 8B aligned
 *  @param scratch_size size in bytes of the scratch space
 *  @param shared_counter used to atomically count the number of unique values
 *  @param estimated_num_unique_values the estimated number of unique values
 *  @param sentinel_value the value that will be assigned index 0. Used to denote empty sets.
 */
template <typename data_t, typename index_t>
__device__ void block_static_set_init(
  static_set<data_t, index_t> &static_set,
  uint64_t *scratch_ptr,
  uint32_t scratch_size,
  uint32_t *shared_counter,
  const uint32_t estimated_num_unique_values,
  const data_t sentinel_value
)
{
  // The static set is scaled to fit the scratch space provided
  uint32_t bytes_required_per_set = sizeof(data_t) + sizeof(index_t);
  uint32_t num_buckets = scratch_size / bytes_required_per_set;

  assert(num_buckets >= STATIC_SET_TARGET_INVERSE_LOAD_FACTOR * estimated_num_unique_values);
  assert(reinterpret_cast<uintptr_t>(scratch_ptr) % block_static_set_get_scratch_alignment_req() == 0);
  assert((scratch_size >= block_static_set_get_scratch_req<data_t, index_t>(estimated_num_unique_values)));

  static_set.num_buckets = num_buckets;

  // We divide the given scratch space into the set-to-data and set-to-index maps
  // We handle the larger type first to guarantee alignment
  // I do not want to waste bytes rounding up to alignment
  if (sizeof(data_t) >= sizeof(index_t))
  {
    static_set.bucket_to_data_map = reinterpret_cast<data_t *>(scratch_ptr);
    static_set.bucket_to_index_map = reinterpret_cast<index_t *>(static_set.bucket_to_data_map + num_buckets);
  }
  else
  {
    static_set.bucket_to_index_map = reinterpret_cast<index_t *>(scratch_ptr);
    static_set.bucket_to_data_map = reinterpret_cast<data_t *>(static_set.bucket_to_index_map + num_buckets);
  }

  // The shared counter is initialized to "1" because the sentinel value is assigned to index 0
  static_set.shared_member_counter = shared_counter;
  if (threadIdx.x == 0)
  {
    shared_counter[0] = 1;
  }

  // The sentinel value will be used to denote empty sets.
  // This is typically the first value in the input array.
  // This value is assigned to index 0
  static_set.sentinel_value = sentinel_value;

  // Initialize all sets to sentinel value
  // this loop is inefficient, probably not a perf concern most of the time
  // TODO: unroll this. It will be a performance concern on some large datasets with high cardinality
  for (uint32_t ix = threadIdx.x; ix < num_buckets; ix += blockDim.x)
  {
    static_set.bucket_to_data_map[ix] =
      sentinel_value; // This value will bypass all map read/writes and simply be assigned index 0
    static_set.bucket_to_index_map[ix] =
      0; // 0 is used as index sentinel because sentinel values never read/write either map
  }

  // Caller must syncthreads before this static set is used.
}

/**
 * Tries to insert a value into a static set. The static set guarantees each value in the set is unique IFF enough space is allocated.
 * The set assigns each unique value an index. Indices are guaranteed to be in-order with no holes. 
 * When the set is full, all "new" values are treated as unique. This allows static set users (e.g. dictionary encoding) to complete with valid indices.
 *
 * The shared counter in the static set can be used to detect if the static set is full or if index_t wrapped.
 *
 * When inserting a value into the set: 
 *    if the value equals the sentinel, return "0"
 *    if the value is found in the set, return the previously assigned index
 *    if the set is not full, insert the value and assign a new index. The set guarantees* that this index will be returned if this value is inserted by any other thread.
 *    if the set is full, assign the value a new index. This index is guaranteed* NOT to be returned by any other insertion call.
 *    
 * (*) Indices will wrap if the cardinality of the dataset exceeds the range of the index type. The above guarantees only hold if index_t does not wrap.
 *
 * All threads in the warp must participate in function call.
 *
 * @return true if this value is unique and was inserted. false if this value was found in the static set
 *
 * @param value the value to be inserted
 * @param set the static set to insert to
 * @param index output parameter - the unique index assigned to this value
 */
template <typename data_t, typename index_t>
__device__ bool warp_static_set_try_insert(
  const data_t my_value,
  static_set<data_t, index_t> &set,
  index_t &my_index,
  bool active = true
)
{
  bool is_unique = false;
  uint32_t my_ix_bucket = 0;
  my_index = 0; // Safe default for inactive threads

  if (active)
  {
    // The sentinel value is never entered into the static set
    // The sentinel value is always assigned an index of 0
    if (my_value != set.sentinel_value)
    {
      uint32_t my_hash = hash32(my_value);
      my_ix_bucket = my_hash % set.num_buckets;

      // TODO: We may want to limit the number of attempts to achieve better performance.
      uint32_t attempts_remaining = set.num_buckets;

      while (true)
      {
        // non-atomic check if value is already present
        data_t found = set.bucket_to_data_map[my_ix_bucket];

        if (found == set.sentinel_value)
        {
          // If set is empty, try to atomic insert my value
          found = atomicCAS(&set.bucket_to_data_map[my_ix_bucket], set.sentinel_value, my_value);
          is_unique = (found == set.sentinel_value);
        }

        if (found == my_value || is_unique)
        {
          break;
        }
        else
        {
          // linear probing
          my_ix_bucket = (my_ix_bucket + 1) % set.num_buckets;
        }

        attempts_remaining--;

        // Handle extreme cases where num_buckets < num_unqiue_values
        // In the future this case may be used to improve perf at the cost of ratio
        if (attempts_remaining == 0)
        {
          is_unique = true;
          my_ix_bucket = set.num_buckets;
          break;
        }
      }

      // Threads with unique values must increment a shared counter to be assigned a unique index
      if (is_unique)
      {
        my_index = atomicAdd(set.shared_member_counter, 1);

        // Note: assignment narrows uint32_t -> index_t. If cardinality exceeds
        // index_t's range, indices wrap and the dictionary is corrupt. The guard
        // below prevents the wrap from causing a kernel deadlock.

        // Anti-deadlock guard: a wrapped index of 0 collides with the
        // "unassigned" sentinel, which would hang duplicate-value threads
        // in the spin-wait below. Index assignments are already corrupted.
        if (my_index == 0)
        {
          my_index = 1;
        }

        if (my_ix_bucket < set.num_buckets)
        {
          cuda::atomic_ref<index_t, cuda::thread_scope_block> ref(set.bucket_to_index_map[my_ix_bucket]);

          // using atomic store to guarantee atomicity and visibility to the spin wait below
          ref.store(my_index, cuda::memory_order_relaxed);
        }
        else
        {
          // Table is full. Any value not found in the table must be treated as unique.
        }
      }
    }
  }

  __syncwarp(); // prevent deadlock if multiple thds in a warp have the same value

  // Threads with duplicate values know their bucket but not yet their unique
  // index. The unique index may not have been written yet if the inserting
  // thread is in a different warp. Spin until it appears.
  //
  // The inserting thread MUST be in a different warp from the spinner, or
  // this deadlocks. Same-warp duplicates are handled by the __syncwarp
  // above, which lets the inserting lane reach the bucket-write before the
  // spinning lane gets here.
  if (active && !is_unique && my_value != set.sentinel_value)
  {
    cuda::atomic_ref<index_t, cuda::thread_scope_block> ref(set.bucket_to_index_map[my_ix_bucket]);

    while (true)
    {
      index_t this_read = ref.load(cuda::memory_order_relaxed);
      if (this_read != 0)
      {
        my_index = this_read;
        break;
      }
      __nanosleep(STATIC_SET_SHORT_SLEEP_NS);
    }
  }

  return is_unique;
}

} // namespace cascaded
