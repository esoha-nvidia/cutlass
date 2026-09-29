/*
* Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
*  * Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
*  * Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and/or other materials provided with the distribution.
*  * Neither the name of NVIDIA CORPORATION nor the names of its
*    contributors may be used to endorse or promote products derived
*    from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
* PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
* CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
* EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
* PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
* PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
* OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/**

COMM-VIA-RING: Facilitates communication between a producer & consumer.
        Used to protect a ring buffer where producer can produce many types
        without synchronizing with consumer.

*/

#pragma once

#define IX_CONSUMER 1
#define IX_PRODUCER 0

#define COMM_VIA_RING_INITIAL_ATOMIC_VALUE 1
#define COMM_VIA_RING_NUM_ATOMICS 2

#define comm_atomic_t cuda::atomic<int, cuda::thread_scope_block>

namespace snappy
{

/**
 *  @brief Wait helper function that spins until func_wait_until_false(cached,target) is false
 *  @param counter The shared counter that this warp might need to wait for
 *  @param cached Stores the cached value of the shared counter IFF the counter is read
 *  @param target The second parameter to be passed to the lambda function
 *  @param func_need_wait Returns true IFF this warp needs to wait for the shared_counter
 *  @param need_threadfence True if threadfence must be called after waiting
 */
template <uint32_t SLEEP_DURATION, bool need_threadfence, typename Lambda>
inline __device__ void
comm_via_ring_wait_helper(comm_atomic_t &counter, int &cached, const int &target, Lambda func_need_wait)
{
  // "cached" is the last observed counter value stored in a register

  if (func_need_wait(cached, target))
  {
    if (!thread_warp_ix())
    {
      cached = counter.load(cuda::std::memory_order_relaxed);

      while (func_need_wait(cached, target))
      {
        __nanosleep(SLEEP_DURATION);
        cached = counter.load(cuda::std::memory_order_relaxed);
      }

      if (need_threadfence)
      {
        cuda::atomic_thread_fence(cuda::std::memory_order_acquire, cuda::thread_scope_block);
      }
    }

    // thread 0 broadcasts the updated cached value
    cached = __shfl_sync(WARP_ALL, cached, 0);
  }
}

inline __device__ void comm_via_ring_init_shmem(comm_atomic_t (&shared_counters)[COMM_VIA_RING_NUM_ATOMICS])
{
  if (threadIdx.x < COMM_VIA_RING_NUM_ATOMICS)
  {
    // A negative message count means the consumer needs to exit
    // Counters initialized to 1 so producer can signal exit without producing
    shared_counters[threadIdx.x] = COMM_VIA_RING_INITIAL_ATOMIC_VALUE;
  }
}

inline __device__ void comm_via_ring_init_cached_values(int &cached_num_produced, int &cached_num_consumed)
{
  // When num_produced < 0 it is time for the consumer to exit
  // Using '1' here instead of '0' so the consumer will still
  // know to exit even if no symbols are produced.
  cached_num_consumed = COMM_VIA_RING_INITIAL_ATOMIC_VALUE;
  cached_num_produced = COMM_VIA_RING_INITIAL_ATOMIC_VALUE;
}

/**
 *  @brief Producer spins until "space_needed" slots are available in the comm ring buffer.
 */
template <uint32_t SLEEP_DURATION, uint32_t RING_SIZE>
inline __device__ void comm_via_ring_producer_wait_for_space(
  comm_atomic_t (&shared_counters)[COMM_VIA_RING_NUM_ATOMICS],
  int &cached_num_produced,
  int &cached_num_consumed,
  const int space_needed
)
{
  // Make sure consumer is not lagging too far behind
  // must spin while consumer is less than or equal to this min_num_consumed threshold
  const int min_num_consumed = cached_num_produced + space_needed - RING_SIZE;

  auto func_need_wait = [=] __device__(int cached, int target) { return cached < target; };
  comm_via_ring_wait_helper<SLEEP_DURATION, false>(
    shared_counters[IX_CONSUMER],
    cached_num_consumed /* cached */,
    min_num_consumed /* target */,
    func_need_wait
  );
}

/**
 *  @brief Create num_active entries in the ring buffer. ASSUMES space is available.
 */
inline __device__ void comm_via_ring_produce(
  comm_atomic_t (&shared_counters)[COMM_VIA_RING_NUM_ATOMICS],
  int &cached_num_produced,
  int &cached_num_consumed,
  int num_active
)
{
  cached_num_produced += num_active;
  if (!thread_warp_ix())
  {
    shared_counters[IX_PRODUCER].store(cached_num_produced, cuda::std::memory_order_release);
  }
}

/**
  *  @brief Sends the exit signal to the consumer.
  */
inline __device__ void comm_via_ring_producer_exit(
  comm_atomic_t (&shared_counters)[COMM_VIA_RING_NUM_ATOMICS],
  int &cached_num_produced,
  int &cached_num_consumed
)
{
  if (!thread_warp_ix())
  {
    shared_counters[IX_PRODUCER].store(-1 * cached_num_produced, cuda::std::memory_order_release);
  }
}

/**
 *  @brief Consumer spins until entries are found in the comm ring buffer.
 */
template <uint32_t SLEEP_DURATION>
inline __device__ void comm_via_ring_consumer_wait_for_work(
  comm_atomic_t (&shared_counters)[COMM_VIA_RING_NUM_ATOMICS],
  int &cached_num_produced,
  int &cached_num_consumed,
  uint32_t &num_ready
)
{
  // Consumer must wait for Producer Count > Consumer Count
  // Consumer only has to stop if num_produced = num_consumed
  // This also handles case where num_produced is negative! (which is the producer exit signal)
  auto func_need_wait = [=] __device__(int cached, int target) { return cached == target; };
  comm_via_ring_wait_helper<SLEEP_DURATION, true>(
    shared_counters[IX_PRODUCER],
    cached_num_produced /* cached */,
    cached_num_consumed /* target */,
    func_need_wait
  );

  num_ready = abs(cached_num_produced) - cached_num_consumed;
}

/**
 *  @brief Read values from the comm ring buffer
 */
inline __device__ void comm_via_ring_consume(
  comm_atomic_t (&shared_counters)[COMM_VIA_RING_NUM_ATOMICS],
  int &cached_num_produced,
  int &cached_num_consumed,
  const uint32_t num_active
)
{
  // This function must ALWAYS be proceeded by a syncwarp to guarantee

  cached_num_consumed += num_active;

  if (!thread_warp_ix())
  {
    // memory_order_release NOT necessary here because consumer does not write any other values visible to producer
    shared_counters[IX_CONSUMER].store(cached_num_consumed, cuda::std::memory_order_relaxed);
  }
}

} // namespace snappy