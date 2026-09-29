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

GATHER: Load all values for a given output window in Four-Bytes-per-Bytes (4BpB) format
WRITE: Write all values for a given output window 


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
 *  @brief Fastest implementation of 4BpB gather and write to global memory. Assumes all gather sources are non-zero
 */
inline __device__ void unsnap_gather_and_write(
  const uint8_t *input_buffer,
  uint8_t *output_buffer,
  uint32_t ix_output,
  uint32_t *shared_gather_sources
)
{
  constexpr uint32_t BYTES_PER_THREAD = OUTPUT_WINDOW_SIZE / WARP_SIZE_U;
  uint8_t my_vals[BYTES_PER_THREAD];

// gather
#pragma unroll
  for (uint32_t ix_byte = 0; ix_byte < BYTES_PER_THREAD; ix_byte++)
  {
    uint32_t my_ix_output = WARP_SIZE_U * ix_byte + thread_warp_ix();
    uint32_t source_index = shared_gather_sources[my_ix_output]; // 4BpB input

    const uint8_t *src = (source_index < BIG_OFFSET) ? input_buffer : output_buffer;
    src += (source_index & (BIG_OFFSET - 1));
    my_vals[ix_byte] = *src;
  }

// write
#pragma unroll
  for (uint32_t ix_byte = 0; ix_byte < BYTES_PER_THREAD; ix_byte++)
  {
    uint32_t my_ix_output = WARP_SIZE_U * ix_byte + thread_warp_ix();
    output_buffer[ix_output + my_ix_output] = my_vals[ix_byte];
  }
}

/**
 *  @brief Slower implementation of 4BpB gather and write to global memory. Assumes some gather sources are zero.
 *         Gather sources of 0 will not be loaded or written.
 *         For use at the unaligned end of a compressed output buffer.
 */
inline __device__ void unsnap_gather_and_write_masked(
  const uint8_t *input_buffer,
  uint8_t *output_buffer,
  uint32_t ix_output,
  uint32_t *shared_gather_sources
)
{
  constexpr uint32_t BYTES_PER_THREAD = OUTPUT_WINDOW_SIZE / WARP_SIZE_U;
  uint8_t my_vals[BYTES_PER_THREAD];

// Gather
#pragma unroll
  for (uint32_t ix_byte = 0; ix_byte < BYTES_PER_THREAD; ix_byte++)
  {
    uint32_t my_ix_output = WARP_SIZE_U * ix_byte + thread_warp_ix();
    uint32_t source_index = shared_gather_sources[my_ix_output];

    const uint8_t *src = (source_index < BIG_OFFSET) ? input_buffer : output_buffer;
    src += (source_index & (BIG_OFFSET - 1));

    if (source_index != 0)
    {
      my_vals[ix_byte] = *src;
    }
  }

// Write
#pragma unroll
  for (uint32_t ix_byte = 0; ix_byte < BYTES_PER_THREAD; ix_byte++)
  {
    uint32_t my_ix_output = WARP_SIZE_U * ix_byte + thread_warp_ix();
    uint32_t source_index = shared_gather_sources[my_ix_output];

    if (source_index != 0)
    {
      output_buffer[ix_output + my_ix_output] = my_vals[ix_byte];
    }
  }
}

} // namespace snappy