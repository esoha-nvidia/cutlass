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

WARP_SNAPPY_FINDER: Finds symbols in a Snappy compressed stream. Extracts
        the minimum number of bytes so the symbols can be fully 
        decoded later.

        Snappy Specific.

Performs the following operations in this order:

    PREFETCH
    FIND
    EXTRACT
    COMM (Produce for WARP_SNAPPY_MAPPER)

*/

#include "comm_via_ring.cuh"
#include "CorrectnessChecks.cuh"
#include "extract.cuh"
#include "find.cuh"
#include "prefetch.cuh"

namespace snappy
{

/**
 *  @brief Simple Finder implementaiton used for small chunks.
 *  Only used for compressed chunks <= 512B
 */
inline __device__ void unsnap_warp_snappy_finder_small_chunk(
  comm_atomic_t (&shared_counters)[COMM_VIA_RING_NUM_ATOMICS],
  uint32_t (&shared_symbol_ring)[2 * SYMBOL_RING_SIZE],
  const uint8_t *input_buffer,
  const uint32_t input_size,
  uint8_t (&prefetch_data)[PREFETCH_SCRATCH_SIZE]
)
{
  for (uint32_t ix_byte = thread_warp_ix(); ix_byte < input_size; ix_byte += WARP_SIZE_U)
  {
    prefetch_data[ix_byte] = input_buffer[ix_byte];
  }

  // These values do not wrap around, they linearly increase until the chunk is decoded.
  // cached_num_produced % SYMBOL_RING_SIZE is the next producer write location
  // cached_num_consumed % SYMBOL_RING_SIZE is the next consumer read location
  int cached_num_produced, cached_num_consumed;
  comm_via_ring_init_cached_values(cached_num_produced, cached_num_consumed);

  __syncwarp();

  // Do not need to wait for space if SYMBOL_RING_SIZE >= 256.
  // Small chunks are <= 512B
  // Snappy symbols are >= 2B in length
  // Worst case num symbols = 256
  static_assert(SYMBOL_RING_SIZE >= 256);
  assert(input_size <= 512);

  uint32_t ix_input = 0;
  while (ix_input < input_size)
  {
    uint32_t header = 0;
    uint32_t tail = 0;
    unsnap_extract_safe(prefetch_data, ix_input, header, tail, input_size, 0);

    int ix_symbol = (cached_num_produced) % SYMBOL_RING_SIZE;
    shared_symbol_ring[ix_symbol] = header;
    shared_symbol_ring[SYMBOL_RING_SIZE + ix_symbol] = tail;
    cached_num_produced++;

    uint32_t input_shift = unsnap_find_next(prefetch_data, ix_input);
    // ensure forward progress even if input_shift is 0 or large enough to overflow ix_input (invalid input)
    ix_input = max(ix_input + input_shift, ix_input + 1);
  }
  __syncwarp();
  comm_via_ring_producer_exit(shared_counters, cached_num_produced, cached_num_consumed);
}

// ==================================================================================================================
// ==================================================================================================================   Real Implementation
// ==================================================================================================================

/**
 *  Finds & Extracts up to 128 symbols from a prefetch sector of 320B.
 *  64B are kept in reserve so that there are always 32 candidates that can 
 *  be decoded in parallel.
 */
inline __device__ void unsnap_warp_snappy_finder_find_some_fast(
  unsnap_prefetch_state &p,
  uint8_t (&prefetch_data)[PREFETCH_SCRATCH_SIZE],
  uint32_t &X,
  comm_atomic_t (&shared_counters)[COMM_VIA_RING_NUM_ATOMICS],
  int &cached_num_produced,
  int &cached_num_consumed,
  uint32_t (&shared_symbol_ring)[2 * SYMBOL_RING_SIZE]
)
{
  // reserve worst case space 256/2=up to 128 symbols
  comm_via_ring_producer_wait_for_space<SLEEP_DURATION_FINDER, SYMBOL_RING_SIZE>(
    shared_counters,
    cached_num_produced,
    cached_num_consumed,
    INPUT_WINDOW_SIZE / 2
  );

  // p.ix_prefetch is used to toggle betweeen two prefetch sectors

  // The actual prefetch sector size is (256B + 64B)
  // X stops at 256 to guarantee 36B are always available
  // guaranteed that 36B are always available per find32 call
  while (X < INPUT_WINDOW_SIZE)
  {
    uint32_t mask = 0;
    uint32_t this_input_shift = 0;

    // partially decode 32 candidate positions
    // MUST guarantee that 36B are available per find32 call
    unsnap_find32_reduce(prefetch_data, p.ix_prefetch + X, mask, this_input_shift);

    // up to 16 bits will be non-zero in the mask
    // can find at most 16 symbols in 32 candidate positions
    unsnap_extract32(prefetch_data, p.ix_prefetch + X, mask, cached_num_produced, shared_symbol_ring);

    // ensure forward progress even if this_input_shift is 0 or large enough to overflow X (invalid input)
    X = max(X + this_input_shift, X + 1);
  }

  X -= INPUT_WINDOW_SIZE;
}

/**
 *  Extracts ALL symbols from a given prefetch sector. Unoptimized. 
 */
inline __device__ void unsnap_warp_snappy_finder_find_all_slow(
  unsnap_prefetch_state &p,
  uint8_t (&prefetch_data)[PREFETCH_SCRATCH_SIZE],
  uint32_t &X,
  comm_atomic_t (&shared_counters)[COMM_VIA_RING_NUM_ATOMICS],
  int &cached_num_produced,
  int &cached_num_consumed,
  const uint32_t &prefetch_size,
  uint32_t (&shared_symbol_ring)[2 * SYMBOL_RING_SIZE]
)
{
  while (X < prefetch_size)
  {
    comm_via_ring_producer_wait_for_space<SLEEP_DURATION_FINDER, SYMBOL_RING_SIZE>(
      shared_counters,
      cached_num_produced,
      cached_num_consumed,
      1
    );
    __syncwarp();
    uint32_t this_input_shift = unsnap_find_next(prefetch_data, p.ix_prefetch + X);

    uint32_t header = 0;
    uint32_t tail = 0;
    unsnap_extract_safe(prefetch_data, X, header, tail, prefetch_size, p.ix_prefetch);

    uint32_t ix_symbol = (cached_num_produced) % SYMBOL_RING_SIZE;
    shared_symbol_ring[ix_symbol] = header;
    shared_symbol_ring[SYMBOL_RING_SIZE + ix_symbol] = tail;
    __syncwarp();
    comm_via_ring_produce(shared_counters, cached_num_produced, cached_num_consumed, 1);

    // ensure forward progress even if this_input_shift is 0 or large enough to overflow X (invalid input)
    X = max(X + this_input_shift, X + 1);
  }
}

/**
 *  Finds symbols in the compressed input stream, extracts 5B, and sends them to the Mapper Warp.
 */
template <bool CORRECTNESS_CHECK>
__device__ void unsnap_warp_snappy_finder(
  comm_atomic_t (&shared_counters)[COMM_VIA_RING_NUM_ATOMICS],
  uint32_t (&shared_symbol_ring)[2 * SYMBOL_RING_SIZE],
  const uint8_t *input_buffer,
  const uint32_t input_size,
  SnappyCorrectnessChecker<CORRECTNESS_CHECK> *correctness_checker
)
{
  // Each decode needs 5 total bytes
  // Batched decode needs 32 + 4 total bytes
  // Must preserve 36 bytes between iterations
  // round up to 32B alignment
  __align__(32) __shared__ uint8_t prefetch_data[PREFETCH_SCRATCH_SIZE];

  if (input_size <= 2 * INPUT_WINDOW_SIZE)
  {
    // OOB access guarded against
    unsnap_warp_snappy_finder_small_chunk(shared_counters, shared_symbol_ring, input_buffer, input_size, prefetch_data);
    SnappyCorrectnessChecker<CORRECTNESS_CHECK>::advanceFinder(INT32_MAX, correctness_checker);
    return;
  }

  // These values do not wrap around, they linearly increase until the chunk is decoded.
  // cached_num_produced % SYMBOL_RING_SIZE is the next producer write location
  // cached_num_consumed % SYMBOL_RING_SIZE is the next consumer read location
  int cached_num_produced, cached_num_consumed;
  comm_via_ring_init_cached_values(cached_num_produced, cached_num_consumed);

  unsnap_prefetch_state p;

  // ==========================================================================
  // ========================================================================== Warmup
  // ==========================================================================
  uint32_t X, input_alignment_fix;
  unsnap_prefetch_init(p, prefetch_data, X, input_alignment_fix, input_buffer, input_size);

  __syncwarp();

  unsnap_warp_snappy_finder_find_some_fast(
    p,
    prefetch_data,
    X,
    shared_counters,
    cached_num_produced,
    cached_num_consumed,
    shared_symbol_ring
  );
  unsnap_prefetch_advance(p, prefetch_data, input_alignment_fix);

  __syncwarp(); // must wait for prefetch_data to be filled
  SnappyCorrectnessChecker<CORRECTNESS_CHECK>::advanceFinder(cached_num_produced, correctness_checker);

  // Produce
  if (!thread_warp_ix())
  {
    shared_counters[IX_PRODUCER].store(cached_num_produced, cuda::std::memory_order_release);
  }

  // ==========================================================================
  // ========================================================================== Steady State
  // ==========================================================================
  while (p.num_remaining_bytes >= INPUT_WINDOW_SIZE)
  {
    if (SnappyCorrectnessChecker<CORRECTNESS_CHECK>::finderWaitForMapper(correctness_checker))
    {
      return; // mapper has exited, so we exit as well
    }

    unsnap_warp_snappy_finder_find_some_fast(
      p,
      prefetch_data,
      X,
      shared_counters,
      cached_num_produced,
      cached_num_consumed,
      shared_symbol_ring
    );
    unsnap_prefetch_advance(p, prefetch_data, INPUT_WINDOW_SIZE);

    __syncwarp(); // this sync protects BOTH prefetch_data (for next iter) and the previously extracted symbols

    SnappyCorrectnessChecker<CORRECTNESS_CHECK>::advanceFinder(cached_num_produced, correctness_checker);
    // Produce
    if (!thread_warp_ix())
    {
      shared_counters[IX_PRODUCER].store(cached_num_produced, cuda::std::memory_order_release);
    }
  }

  unsnap_prefetch_load_remainder(p, prefetch_data);

  __syncwarp(); // must wait for prefetch_data to be filled

  // ==========================================================================
  // ========================================================================== Tail
  // ==========================================================================
  uint32_t num_readable_bytes = p.num_remaining_bytes + PREFETCH_SKIN_SIZE;
  if (SnappyCorrectnessChecker<CORRECTNESS_CHECK>::finderWaitForMapper(correctness_checker))
  {
    return; // mapper has exited, so we exit as well
  }

  unsnap_warp_snappy_finder_find_all_slow(
    p,
    prefetch_data,
    X,
    shared_counters,
    cached_num_produced,
    cached_num_consumed,
    num_readable_bytes,
    shared_symbol_ring
  );

  __syncwarp(); // must wait for all symbols to be extracted before producing
  // INT32_MAX so that the mapper can surely process all remaining symbols. No danger of the finder deadlocking anymore.
  SnappyCorrectnessChecker<CORRECTNESS_CHECK>::advanceFinder(INT32_MAX, correctness_checker);

  comm_via_ring_producer_exit(shared_counters, cached_num_produced, cached_num_consumed);
}

} // namespace snappy