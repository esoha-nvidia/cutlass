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

WARP_C: Processes output windows in the Four-Bytes-per-Byte (4BpB) format.

        Compression format agnostic!

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

    COMM (Consume from WARP_SNAPPY_MAPPER)
    SMASH
    GATHER
    WRITE

*/

#include "comm_via_ring.cuh"
#include "comm_via_swap.cuh"
#include "CorrectnessChecks.cuh"
#include "gather_write.cuh"
#include "smash.cuh"

namespace snappy
{

/**
 *  Smashes, Gathers, and Writes output windows in 4BpB format received from the Mapper Warp.
 */
template <bool CORRECTNESS_CHECK>
__device__ void unsnap_warp_4BpB_processor(
  comm_barrier_t (&shared_barriers)[COMM_VIA_SWAP_NUM_ATOMICS],
  uint32_t *shared_window_ring,
  const uint8_t *input_buffer,
  uint8_t *output_buffer,
  const uint32_t output_size,
  SnappyCorrectnessChecker<CORRECTNESS_CHECK> *correctness_checker
)
{
  comm_via_swap_init_consumer(shared_barriers);

  uint32_t write_ix_output = 0;
  const uint32_t num_full_windows = output_size >> OUTPUT_WINDOW_SIZE_PWR;
  for (uint32_t ix_consume = 0; ix_consume < num_full_windows; ix_consume++)
  {
    if (SnappyCorrectnessChecker<CORRECTNESS_CHECK>::processorWaitForMapper(correctness_checker))
    {
      return;
    }
    comm_via_swap_consume_start(shared_barriers, ix_consume);

    __syncwarp(); // commute thread fence

    uint32_t ix_window = ix_consume & 1;

    unsnap_smash(write_ix_output, shared_window_ring + ix_window * OUTPUT_WINDOW_SIZE);

    // syncwarp not needed because bytes are smashed, gathered, & written by the same thread

    unsnap_gather_and_write(
      input_buffer,
      output_buffer,
      write_ix_output,
      shared_window_ring + ix_window * OUTPUT_WINDOW_SIZE
    );

    __syncwarp(); // make sure all thds finished read
    SnappyCorrectnessChecker<CORRECTNESS_CHECK>::processorConsumeWindow(correctness_checker);

    comm_via_swap_consume_end(shared_barriers, ix_consume);
    write_ix_output += OUTPUT_WINDOW_SIZE;
  }

  if (SnappyCorrectnessChecker<CORRECTNESS_CHECK>::processorWaitForMapper(correctness_checker))
  {
    return;
  }

  comm_via_swap_consume_start(shared_barriers, num_full_windows);

  __syncwarp(); // commute thread fence

  // Always exit main loop with one window left
  // Last window must always use masked G&W
  uint32_t ix_window = (num_full_windows) & 1;

  unsnap_smash(write_ix_output, shared_window_ring + ix_window * OUTPUT_WINDOW_SIZE);

  // syncwarp not needed because bytes are smashed, gathered, & written by the same thread

  unsnap_gather_and_write_masked(
    input_buffer,
    output_buffer,
    write_ix_output,
    shared_window_ring + ix_window * OUTPUT_WINDOW_SIZE
  );
}

} // namespace snappy