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

MAP: Create Four-Bytes-per-Byte (4BpB) format for a given output window. Maps
    LZ77 tokens to their corresponding output bytes.

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

*/

#pragma once

namespace snappy
{

/**
 *  @brief Threads collaborate on batches of up to 32 tokens. Threads are evenly distributed between active symbols.
 *  @param my_cursor first byte that DOES map to the current token.
 *  @param my_end first byte that does NOT map to the current token.
 *  @param my_source based gather source for the given token.
 *  @param shared_gather_sources gather sources in 4BpB format.
 *  @param active true if the thread has been asigned a symbol, false otherwise.
 *
 *  TODO: This function is a major bottleneck for some datasets. This function has NOT
 *         yet been optimized for the current 256B window case. 
 *
 *  @warning This function initializes data used by snappy::unsnap_smash. comm_via_swap prevents a race condition.
 *  @warning Compute Sanitizer will erroneously report a race 
 */
inline __device__ void
unsnap_map32(uint32_t &my_cursor, uint32_t &my_end, uint32_t &my_source, uint32_t *shared_gather_sources)
{
  // Threads are in one of the following states:
  // Complete
  // Active & inside the current window
  // Active, but outside the current window
  // Inactive / Not assigned work
  // These states are always in this order
  // Threads are always completed in order
  // All threads of each state are consecutive

  // Check if my symbol is in the current window
  uint32_t ix_start = min(OUTPUT_WINDOW_SIZE, my_cursor); // less than 16 bits
  uint32_t ix_end = min(OUTPUT_WINDOW_SIZE, my_end); // less than 16 bits
  bool is_active = ix_end > ix_start;

  // Pack end and start so they can be broadcast in a single shuffle
  uint32_t metadata = (ix_end << 16) | (ix_start);

  // Count the number of active symbols
  // Threads are divided evenly between active symbols
  // Active threads are guaranteed to be CONTIGUOUS in the above mask.
  uint32_t active_mask = __ballot_sync(WARP_ALL, is_active);
  uint32_t num_active_symbols = __popc(active_mask); // popc counts number of 1s
  assert(num_active_symbols > 0); // At least one thread must be active or we should not be in this function

  // Index of the first active symbol in the current batch
  uint32_t first_active = WARP_SIZE_U - __clz(active_mask) - num_active_symbols;

  // Evenly distribute inactive threads between active symbols
  // Each thread is assigned to a "group" and receives a (ix_start, ix_end, source) triplet
  // from the group leader that owns the active symbol.
  uint32_t my_group_id = thread_warp_ix() % num_active_symbols;
  uint32_t my_group_size = (WARP_SIZE_U - my_group_id + num_active_symbols - 1) /
                           num_active_symbols; // the last symbol gets fewer threads
  uint32_t my_group_owner = (my_group_id) + first_active;

  uint32_t my_group_metadata = __shfl_sync(WARP_ALL, metadata, my_group_owner);
  uint32_t my_group_end = (my_group_metadata >> 16) & TWO_BYTE_MASK;

  uint32_t my_lane_in_group = thread_warp_ix() / num_active_symbols;
  uint32_t my_group_cursor = (my_group_metadata & TWO_BYTE_MASK) + my_lane_in_group;
  uint32_t my_group_value = __shfl_sync(WARP_ALL, my_source, my_group_owner) + my_lane_in_group; // 4BpB gather source

  // "groups" of threads collaborate to map their assigned symbol.
  // group members initialize two bytes per iteration
  // TODO: This is a performance opportunity. Try to unroll this loop.
  // This loop is a hotspot and is difficult to unroll
  /*
        // Example of 4BpB initialization using 12B token

        while(my_cursor < my_end){
            4BpB_format[my_cursor++] = my_source++;
        }
    */
  while (my_group_cursor + my_group_size < my_group_end)
  {
    // Initialize 4BpB format for 2 output postitions at a time
    shared_gather_sources[my_group_cursor] = my_group_value;
    shared_gather_sources[my_group_cursor + my_group_size] = my_group_value + my_group_size;

    my_group_cursor += 2 * my_group_size;
    my_group_value += 2 * my_group_size;
  }

  // Guarantee that all members of my group will have at most one byte left to initialize
  // Handle tail if my symbol length was not divisible by my group size
  if (my_group_cursor < my_group_end)
  {
    shared_gather_sources[my_group_cursor] = my_group_value;
  }

  // Update my 12B token (cursor, source, end) depending on how many of my bytes were initialized
  uint32_t length = (ix_start < ix_end) ? ix_end - ix_start : 0;
  my_cursor += length;
  my_source += length;
}

} // namespace snappy