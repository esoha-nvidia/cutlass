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

DECODE: Decodes a compressed symbol (up to 5B) into my 12B token format.
        Also advances the ix_input and ix_output indices that will be used
        for future symbol decodes.

Token Format: this 12B token format is used for writing Four-Byte-per-Byte (4BpB) format
    uint32_t my_cursor: the output index to be written by this symbol
    uint32_t my_end   : the first output index to be written by the NEXT symbol
    uint32_t my_source: index of value to be copied to output_buffer[my_cursor]. If my_source > BIG_OFFSET the value is a match, else literal.

Terminology
    ix_input = index of first byte of symbol being decoded. Symbol header located at compressed_input_buffer[ix_input]
    ix_output = index of first output byte corresponding to current symbol
    input_shift = compressed length of a symbol
    output_shift = uncompressed length of a symbol
    match_distance = self explanatory
*/

#pragma once

#include "CorrectnessChecks.cuh"

namespace snappy
{

/**
 *  @brief Decodes a batch of up to 32 symbols into 12B token format.
 */
// TODO: reduce divergence here (see find_next which does this for input_shift)
template <bool CORRECTNESS_CHECK>
inline __device__ void unsnap_decode_token32(
  uint32_t &ix_input,
  uint32_t &ix_output,
  uint32_t header,
  uint32_t tail, // 8B extracted symbol format
  uint32_t &my_cursor,
  uint32_t &my_end,
  uint32_t &my_source, // 12B token format
  SnappyCorrectnessChecker<CORRECTNESS_CHECK> *correctness_checker,
  uint32_t active,
  bool &invalid_stream
)
{
  if (!active)
  {
    header = tail = 0;
  }

  const uint32_t tag = header & TAG_BIT_MASK;
  const uint32_t nontag = header >> NUM_TAG_BITS;
  uint32_t input_shift = 0;
  uint32_t output_shift = 0;

  /*
        Snappy Format Recap:
            00: Literal
            01: Copy with 1-byte offset
            10: Copy with 2-byte offset
            11: Copy with 4-byte offset
    */
  if (tag)
  {
    // MATCHES
    uint32_t match_distance = 0;

    constexpr uint32_t TAG_FOR_1B_MATCH = 1;

    if (tag == TAG_FOR_1B_MATCH)
    {
      // Close Match
      constexpr uint32_t BASE_OUTPUT_SHIFT = 3;
      constexpr uint32_t OUTPUT_SHIFT_BIT_EXTRACT_MASK = 7;
      output_shift = (nontag & OUTPUT_SHIFT_BIT_EXTRACT_MASK) + BASE_OUTPUT_SHIFT;

      constexpr uint32_t MATCH_DISTANCE_BIT_EXTRACT_MASK = 0xe0;
      constexpr uint32_t MATCH_DISTANCE_BIT_EXTRACT_POWER = 3;
      match_distance = ((header & MATCH_DISTANCE_BIT_EXTRACT_MASK) << MATCH_DISTANCE_BIT_EXTRACT_POWER);
      match_distance |= (tail & ONE_BYTE_MASK);
    }
    else
    {
      // Long Distance Match (2B or 4B distance)
      output_shift = nontag;
      match_distance = tail & (0xFFFFFFFF >> (32 - 8 * (1 << (tag - 1))));
    }
    // Avoid having match_distance of 0 or larger than BIG_OFFSET, which would cause an infinite loop in "unsnap_smash".
    if (match_distance == 0 || match_distance > BIG_OFFSET)
    {
      invalid_stream = true;
      match_distance = 1;
    }

    input_shift = (1 << (tag - 1)) - 1;
    my_source = BIG_OFFSET - match_distance;
  }
  else
  {
    // LITERALS
    uint32_t length_bytes = 0;
    output_shift = nontag;

    constexpr uint32_t MAX_SINGLE_BYTE_LITERAL_LENGTH = 59;
    constexpr uint32_t BITS_PER_BYTE = 8;
    if (nontag > MAX_SINGLE_BYTE_LITERAL_LENGTH)
    {
      length_bytes = nontag - MAX_SINGLE_BYTE_LITERAL_LENGTH;
      output_shift = tail & ((1 << (BITS_PER_BYTE * length_bytes)) - 1);
    }

    my_source = 1 + length_bytes;
    input_shift = (length_bytes + output_shift);
  }

  // Make sure inactive threads have in/out shifts of 0
  output_shift += active;
  input_shift += active;
  input_shift += active;

  uint32_t local_ix_input = 0;
  uint32_t local_ix_output = 0;

  typedef nvcomp::cub::WarpScan<uint32_t, WARP_SIZE_U> MyScan;
  __shared__ typename MyScan::TempStorage scan_storage;
  MyScan(scan_storage).ExclusiveSum(input_shift, local_ix_input);
  MyScan(scan_storage).ExclusiveSum(output_shift, local_ix_output);

  // Create the 12B token format
  my_cursor = ix_output + local_ix_output; // index where this symbol starts writing output
  my_end = my_cursor + output_shift; // index where this symbol STOPS writing output (exclusive)
  my_source += (tag ? my_cursor : ix_input + local_ix_input); // 4BpB gather index for my first byte

  SnappyCorrectnessChecker<CORRECTNESS_CHECK>::checkSymbols(
    my_source,
    my_cursor,
    my_end,
    ix_input + local_ix_input,
    output_shift,
    active,
    tag,
    __LINE__,
    __func__,
    correctness_checker
  );

  // EXAMPLE USAGE of 4BpB Format:
  //
  // while(my_cursor < my_end){
  //      4BpB_format[my_cursor++] = my_source++;
  // }

  constexpr uint32_t IX_LAST_THREAD_IN_WARP = 31;
  ix_input += __shfl_sync(WARP_ALL, local_ix_input + input_shift, IX_LAST_THREAD_IN_WARP);
  ix_output += __shfl_sync(WARP_ALL, local_ix_output + output_shift, IX_LAST_THREAD_IN_WARP);

  // Do not need to zero-out cursor, end, source for inactive threads.
  // Inactive threads will have (my_cusor == my_end) which will be treated as "completed"
}

} // namespace snappy