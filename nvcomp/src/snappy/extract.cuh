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

EXTRACT: Extract 5B starting at a known symbol location so 
         that the symbol can be decoded later.

*/

#pragma once
namespace snappy
{

/**
 *  @brief Extracts up to 5B from the compressed stream for later decoding. For use near end of compressed stream. 
 *  @param prefetch_data prefetched 256B of Compressed stream
 *  @param ix_input the index of the first byte of the symbol to be extracted
 *  @param header The first byte of the symbol
 *  @param tail The last four bytes of the symbol
 *  @param input_size The number of bytes in the compressed stream.
 *  @param read_offset The number of bytes to skip when reading from prefetch_data.
 */
inline __device__ void unsnap_extract_safe(
  uint8_t (&prefetch_data)[PREFETCH_SCRATCH_SIZE],
  const uint32_t ix_input,
  uint32_t &header,
  uint32_t &tail,
  uint32_t input_size,
  uint32_t read_offset
)
{
  header = prefetch_data[read_offset + ix_input];
  tail = 0;

  if (ix_input + 1 < input_size)
  {
    tail |= prefetch_data[read_offset + ix_input + 1];
  }
  if (ix_input + 2 < input_size)
  {
    tail |= prefetch_data[read_offset + ix_input + 2] << 8;
  }
  if (ix_input + 3 < input_size)
  {
    tail |= prefetch_data[read_offset + ix_input + 3] << 16;
  }
  if (ix_input + 4 < input_size)
  {
    tail |= prefetch_data[read_offset + ix_input + 4] << 24;
  }
}

/**
 *  @brief Extracts 5B from the compressed stream so that a given symbol can be decoded later. 
 *  @param prefetch_data prefetched 256B of Compressed stream
 *  @param ix_input the index of the first byte of the symbol to be extracted
 *  @param header The first byte of the symbol
 *  @param tail The last four bytes of the symbol
 */
inline __device__ void unsnap_extract(
  uint8_t (&prefetch_data)[PREFETCH_SCRATCH_SIZE],
  const uint32_t ix_input,
  uint32_t &header,
  uint32_t &tail
)
{
  header = prefetch_data[ix_input];
  tail = (prefetch_data[ix_input + 4] << 24) | (prefetch_data[ix_input + 3] << 16) |
         (prefetch_data[ix_input + 2] << 8) | (prefetch_data[ix_input + 1]);
}

/**
 *  @brief Extract 5B starting at input_buffer[ix_input]
 *  @param prefetch_data prefetched 256B of Compressed stream
 *  @param ix_input the index of the FIRST symbol to be extracted
 *  @param mask locations of all symbols to be extracted. 1 indicates the start of a symbol.
 *  @param num_symbols_found The total number of symbols added to the symbol ring so far.
 *  @param shared_symbol_ring Two consecutive arrays of size max_num_symbols. The first stores symbol 4B heads, the second stores symbol 4B tails.
 *  @param max_num_symbols The number of symbols that fit in the symbol ring
 */
inline __device__ void unsnap_extract32(
  uint8_t (&prefetch_data)[PREFETCH_SCRATCH_SIZE],
  const uint32_t ix_input,
  uint32_t &mask,
  int &num_symbols_found,
  uint32_t (&shared_symbol_ring)[2 * SYMBOL_RING_SIZE]
)
{
  const uint32_t ME = (1 << thread_warp_ix());
  if (mask & ME)
  {
    uint32_t header, tail;
    unsnap_extract(prefetch_data, ix_input + thread_warp_ix(), header, tail);
    uint32_t ix_symbol = (num_symbols_found + __popc(mask & (ME - 1))) % SYMBOL_RING_SIZE;
    shared_symbol_ring[ix_symbol] = header;
    shared_symbol_ring[SYMBOL_RING_SIZE + ix_symbol] = tail;
  }

  num_symbols_found += __popc(mask);
}

} // namespace snappy