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

SMASH: Flattens all match-copy chains in a given output
       window. Input and output are Four-Bytes-per-Byte (4BpB) format.

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
 * @brief Resolves all internal matches for a given output window in 4BpB format.
 * @param ix_output The number of bytes already written to the output buffer.
 * @param shared_gather_sources The source indices for each byte of this output window in 4BpB format.
 */
inline __device__ void unsnap_smash(uint32_t ix_output, uint32_t *shared_gather_sources)
{
  // 4BpB indices above this threshold are matches that copy from a byte inside the current output window
  const uint32_t minimum_internal_match_index = BIG_OFFSET + ix_output;

  for (uint32_t ix_base = 0; ix_base < OUTPUT_WINDOW_SIZE; ix_base += WARP_SIZE_U)
  {
    uint32_t my_ix_output = ix_base + thread_warp_ix();
    uint32_t source = shared_gather_sources[my_ix_output];

    // literals and external matches require no action
    if (source >= minimum_internal_match_index)
    {
      while (source >= minimum_internal_match_index)
      {
        // Perform one pointer chase. This cannot be unrolled as is
        uint32_t parent = source - minimum_internal_match_index;

        // Stale values are SAFE here. Stale values just means I have to repeat work
        // Literals & external matches are GUARANTEED to have values that are unchanging
        source = shared_gather_sources[parent];
      }

      // future internal matches will not have to repeat this work
      shared_gather_sources[my_ix_output] = source;
    }
  }
}

} // namespace snappy