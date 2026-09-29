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

WARP_SNAPPY_MAPPER: Decodes batches of up to 32 symbols and maps them to
        output windows in Four-Bytes-per-Byte (4BpB) format (defined below).

        Snappy Specific.

4BpB Format: "Four Bytes per Byte" format stores a 4B index for 
             each byte of output to be written. 4BpB is processed
             as follows:

             output_buffer[ix] = 4BpB[ix] > BIG_OFFSET? 
                output_buffer[4BpB[ix]-BIG_OFFSET]          // match
            : 
                input_buffer[4BpB[ix]];                     // literal

             BIG_OFFSET is an arbitrary large number 

        0 to (BIG_OFFSET-1)    : literal
        BIG_OFFSET to UINT_MAX : match

        guarantee that all literal 4BpB < all match 4BpB
            
Performs the following operations in this order:

    COMM (Consume from WARP_SNAPPY_FINDER)
    DECODE
    MAP
    COMM (Produce for WARP_4BpB_PROCESSOR)

*/

#include "comm_via_swap.cuh"
#include "CorrectnessChecks.cuh"
#include "decode.cuh"
#include "map.cuh"
#include "smash.cuh"

namespace snappy
{

/**
 *  Decodes and Maps symbols received from the Finder warp. 
 *  Sends Windows in unsmashed 4BpB format to the Processor Warp.
 */
template <bool CORRECTNESS_CHECK>
__device__ void unsnap_warp_snappy_mapper(
  comm_atomic_t (&shared_message_counters_in)[COMM_VIA_RING_NUM_ATOMICS],
  comm_barrier_t (&shared_barriers)[COMM_VIA_SWAP_NUM_ATOMICS],
  uint32_t *shared_symbol_ring,
  uint32_t *shared_window_ring,
  uint32_t &decode_ix_input,
  uint32_t &decode_ix_output,
  bool &invalid_stream,
  const uint32_t output_size,
  SnappyCorrectnessChecker<CORRECTNESS_CHECK> *correctness_checker
)
{
  int cached_num_produced_in, cached_num_consumed_in;
  comm_via_ring_init_cached_values(cached_num_produced_in, cached_num_consumed_in);

  bool need_symbol = true;

  // 12B output token
  uint32_t my_cursor; // starting output index
  uint32_t my_end; // ending output index
  uint32_t my_source; // 4BpB reference (determines if literal or match)

  decode_ix_input = 0;
  decode_ix_output = 0;

  uint32_t write_ix_output = 0;

  comm_via_swap_produce_start(shared_barriers, 0);

  while (true)
  {
    if (need_symbol)
    {
      need_symbol = false;
      uint32_t num_ready = 0;

      //wait until work available
      SnappyCorrectnessChecker<CORRECTNESS_CHECK>::advanceMapper(cached_num_consumed_in, correctness_checker);
      SnappyCorrectnessChecker<CORRECTNESS_CHECK>::mapperWaitForFinder(correctness_checker);

      comm_via_ring_consumer_wait_for_work<SLEEP_DURATION_MAPPER_NO_SYMBOL>(
        shared_message_counters_in,
        cached_num_produced_in,
        cached_num_consumed_in,
        num_ready
      );
      if (num_ready == 0)
      {
        break;
      }

      __syncwarp(); // commute thread fence

      int ix_symbol = (cached_num_consumed_in + thread_warp_ix()) % SYMBOL_RING_SIZE;
      uint32_t header = shared_symbol_ring[ix_symbol];
      uint32_t tail = shared_symbol_ring[SYMBOL_RING_SIZE + ix_symbol];

      __syncwarp(); // make sure all thds finished read

      comm_via_ring_consume(
        shared_message_counters_in,
        cached_num_produced_in,
        cached_num_consumed_in,
        min(num_ready, WARP_SIZE_U)
      );
      bool active = thread_warp_ix() < num_ready;

      // updates total number of bytes used from input buffer
      // updates total number of bytes written to output buffer
      unsnap_decode_token32<CORRECTNESS_CHECK>(
        decode_ix_input,
        decode_ix_output,
        header,
        tail,
        my_cursor,
        my_end,
        my_source,
        correctness_checker,
        active,
        invalid_stream
      );
      if (SnappyCorrectnessChecker<CORRECTNESS_CHECK>::hasError(correctness_checker))
      {
        SnappyCorrectnessChecker<CORRECTNESS_CHECK>::mapperExit(correctness_checker);
        comm_via_swap_produce_end(shared_barriers, write_ix_output >> OUTPUT_WINDOW_SIZE_PWR);
        return;
      }

      // Convert cursor from global index to local index
      my_cursor -= write_ix_output;
      my_end -= write_ix_output;

      if (!active)
      {
        my_cursor = BIG_OFFSET;
        my_end = BIG_OFFSET;
      }
    }

    uint32_t ix_window = (write_ix_output >> OUTPUT_WINDOW_SIZE_PWR) & 1;
    unsnap_map32(my_cursor, my_end, my_source, shared_window_ring + ix_window * OUTPUT_WINDOW_SIZE);

    // decode_ix_output: total number of writable output bytes found so far
    // write_ix_output: total number of WRITTEN bytes so far
    // write_ix_output + OUTPUT_WINDOW_SIZE: number of WRITTEN bytes after this window is filled
    if (decode_ix_output <= write_ix_output + OUTPUT_WINDOW_SIZE)
    {
      // Last symbol finished in the current output window
      need_symbol = true;
    }

    if (decode_ix_output >= write_ix_output + OUTPUT_WINDOW_SIZE)
    {
      // Current output window is filled
      __syncwarp(); // make sure all window bytes are written before release

      // Release 4BpB window for the processor
      SnappyCorrectnessChecker<CORRECTNESS_CHECK>::mapperFreeWindow(correctness_checker);
      comm_via_swap_produce_end(shared_barriers, write_ix_output >> OUTPUT_WINDOW_SIZE_PWR);

      write_ix_output += OUTPUT_WINDOW_SIZE;
      my_cursor -= OUTPUT_WINDOW_SIZE;
      my_end -= OUTPUT_WINDOW_SIZE;

      // Wait for processor to release the other buffer
      SnappyCorrectnessChecker<CORRECTNESS_CHECK>::mapperWaitForProcessor(correctness_checker);
      comm_via_swap_produce_start(shared_barriers, write_ix_output >> OUTPUT_WINDOW_SIZE_PWR);
    }
    if (decode_ix_output > output_size)
    {
      break;
    }
  }
  const uint32_t num_full_windows = output_size >> OUTPUT_WINDOW_SIZE_PWR;
  const uint32_t windows_written = write_ix_output >> OUTPUT_WINDOW_SIZE_PWR;

  // Symbols ran out before all expected full windows were produced.
  // Pad with zero-filled full windows so the processor does not hang.
  for (uint32_t i = windows_written; i < num_full_windows; i++)
  {
    uint32_t buffer_half = i & 1;
    for (uint32_t j = thread_warp_ix(); j < OUTPUT_WINDOW_SIZE; j += WARP_SIZE_U)
    {
      shared_window_ring[buffer_half * OUTPUT_WINDOW_SIZE + j] = 0;
    }
    __syncwarp();

    SnappyCorrectnessChecker<CORRECTNESS_CHECK>::mapperFreeWindow(correctness_checker);
    comm_via_swap_produce_end(shared_barriers, i);

    SnappyCorrectnessChecker<CORRECTNESS_CHECK>::mapperWaitForProcessor(correctness_checker);
    comm_via_swap_produce_start(shared_barriers, i + 1);

    write_ix_output += OUTPUT_WINDOW_SIZE;
    invalid_stream = true;
  }

  // Last output window might be partially filled.
  // Must mask off uninitialized bytes
  __syncwarp();

  // Mask off all unused bytes of the last output window
  // for when UNCOMPRESSED_SIZE % OUTPUT_WINDOW_SIZE > 0
  // i.e. at the end of the uncompressed buffer
  uint32_t ix_window = (write_ix_output >> OUTPUT_WINDOW_SIZE_PWR) & 1;
  for (uint32_t i = decode_ix_output % OUTPUT_WINDOW_SIZE + thread_warp_ix(); i < OUTPUT_WINDOW_SIZE; i += WARP_SIZE_U)
  {
    shared_window_ring[ix_window * OUTPUT_WINDOW_SIZE + i] = 0;
  }

  __syncwarp(); // make sure all window bytes are written before release

  comm_via_swap_produce_end(shared_barriers, write_ix_output >> OUTPUT_WINDOW_SIZE_PWR);

  // Drain remaining symbols from the ring so the Finder doesn't hang
  // on producer_wait_for_space after we've stopped consuming.
  // For valid inputs the Finder has already called producer_exit,
  // so this loop exits immediately (num_ready == 0).
  while (true)
  {
    uint32_t drain_ready = 0;
    comm_via_ring_consumer_wait_for_work<SLEEP_DURATION_MAPPER_NO_SYMBOL>(
      shared_message_counters_in,
      cached_num_produced_in,
      cached_num_consumed_in,
      drain_ready
    );
    if (drain_ready == 0)
    {
      break;
    }

    // If we reach this point, the input must be invalid
    invalid_stream = true;
    comm_via_ring_consume(shared_message_counters_in, cached_num_produced_in, cached_num_consumed_in, drain_ready);
  }
}

} // namespace snappy