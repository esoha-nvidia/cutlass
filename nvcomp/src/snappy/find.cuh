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

FIND: Decode the MINIMUM amount of information required to
      find the next symbol in the compressed stream.

*/

#pragma once

namespace snappy
{

/** 
 *  @return The input shift of a symbol who's header is stored at input_buffer[ix_input];
 *  
 *  Number of bytes read depends on the symbol type. 
 */
inline __device__ uint32_t unsnap_find_next(const uint8_t *input_buffer, uint32_t ix_input)
{
  // Partial Decode
  const uint32_t header = input_buffer[ix_input];
  const uint32_t tag = header & 3;
  const uint32_t nontag = header >> 2;

  const uint32_t tag_msb = (tag >> 1) & 1;
  const uint32_t tag_lsb = (tag) & 1;
  const uint32_t is_literal = (tag == 0);

  // We can use uint32_t for input_shift because we set the maximum allowed chunk size to be < INT_MAX
  // but the Snappy format allows sizes of up to UINT_MAX, which could theoretically overflow input_shift.
  static_assert(
    nvcompSnappyDecompressionMaxAllowedChunkSize < std::numeric_limits<uint32_t>::max() - 4
  ); // 4 for the max literal token size
  uint32_t input_shift = (3 * tag_msb) + (2 * tag_lsb) + ((nontag + 2) * is_literal);

  constexpr uint32_t MAX_SINGLE_BYTE_LITERAL_LENGTH = 59;

  if (!tag && nontag > MAX_SINGLE_BYTE_LITERAL_LENGTH)
  {
    // Long literal
    const uint32_t length_bytes = nontag - MAX_SINGLE_BYTE_LITERAL_LENGTH;
    input_shift = (input_buffer[ix_input + 1]);

    if (length_bytes > 1)
    {
      input_shift |= (input_buffer[ix_input + 2] << 8);
    }
    if (length_bytes > 2)
    {
      input_shift |= (input_buffer[ix_input + 3] << 16);
    }
    if (length_bytes > 3)
    {
      input_shift |= (input_buffer[ix_input + 4] << 24);
    }

    input_shift++; // Literal length is 1 more than the value stored in the length bytes
    input_shift = 1 + length_bytes + input_shift; // total symbol length = header(1) + length_bytes + literal length
  }

  return input_shift;
}

/** 
 *  Find the location of up to 16 symbols in a 32B window using a parallel reduction.
 *
 *  @param input_buffer compressed input buffer.
 *  @param ix_next_symbol index of the header of the first un-decoded symbol in input buffer.
 *  @param mask A 32B mask where at bit is 1 iff this byte is the header of symbol.
 *  @param total_input_shift The total length of all symbols with headers in this 32B window.
 */
inline __device__ void
unsnap_find32_reduce(uint8_t *input_buffer, uint32_t ix_next_symbol, uint32_t &mask, uint32_t &total_input_shift)
{

  // This function can only be called when it is GUARANTEED that 36B are available starting at ix_next_symbol
  // Each thread decodes a candidate starting location
  // Decoding can use up to 5B

  // my_input_shift is the compressed length of the symbol starting at my candidate byte
  uint32_t my_input_shift = unsnap_find_next(input_buffer, ix_next_symbol + thread_warp_ix());

  // This code tries to find a sequence of symbols in this batch of 32 starting locations
  // We are guaranteed that byte 0 is the start of a symbol
  const uint32_t t = thread_warp_ix();
  uint32_t my_end = t + my_input_shift;
  uint32_t my_length = (my_end) >= WARP_SIZE_U ? 0
                                               : my_input_shift; // Check if this symbol must be the end of the sequence

  // Each thread is producing a 32b mask where 1 indicates that a symbol starts at this byte
  // Each thread's mask initialy has one 1
  uint32_t my_mask = (1 << t);

  // The second symbol in the sequence can be found without a shuffle.
  my_mask |= (1 << (t + my_length));

// !IMPORTANT!
// This reduction greatly improves LZ77 symbol-finding throughput.
// Instead of finding one symbol at a time sequentially, we are instead
// doing a parallel reduction where symbol SEQUENCES are merged together.
// Symbol sequences are represented as a 32b mask where 1 indicates the
// start of a symbol. This reduction can find a sequence of 16 symbols
// in only 4 iterations.
// Warning: This function is only faster when the average number of symbols
// per 32B is >4. This is "usually" true but not always.
#pragma unroll
  for (int i = 0; i < 4; i++)
  {
    // t + my_length is the starting index of the next symbol in my sequence

    // Load the symbol sequence starting at my next symbol
    // if t+my_length is > 32 then shuffle_down_sync returns my_mask
    uint32_t your_mask = __shfl_down_sync(WARP_ALL, my_mask, my_length);

    // Merge my sequence with the next symbol sequence
    // This conerges when my sequence length is outside the 32B window.
    my_mask |= your_mask;

    // Calculate the total length of my symbol sequence
    my_length = WARP_SIZE_U - __clz(my_mask) - t - 1;
  }

  // All threads must know the symbol sequence starting at byte 0
  mask = __shfl_sync(WARP_ALL, my_mask, 0);

  // All threads must know the total compressed size of the batch
  uint32_t ix_last = WARP_SIZE_U - 1 - __clz(mask); // index of the last symbol in the batch
  total_input_shift = ix_last + __shfl_sync(WARP_ALL, my_input_shift, ix_last);
}

} // namespace snappy