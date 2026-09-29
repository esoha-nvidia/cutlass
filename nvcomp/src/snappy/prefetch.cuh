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

PREFETCH: Loads the entire compressed buffer in 256B chunks.

    A total of 320B are available after each prefecth.185
    64B are preserved after each iteration. 

    Uses two alternating sectors where uint8_t* next = current^320;

*/

#pragma once

namespace snappy
{

struct unsnap_prefetch_state
{
  const uint32_t *aligned_input_buffer;
  uint32_t ix_prefetch;
  uint32_t num_remaining_bytes;
};

/**
 *  @return The number of bytes to skip in the first prefetched sector
 */
inline __device__ void unsnap_prefetch_init(
  unsnap_prefetch_state &p,
  uint8_t (&prefetch_data)[PREFETCH_SCRATCH_SIZE],
  uint32_t &X,
  uint32_t &input_alignment_fix,
  const uint8_t *input_buffer,
  uint32_t input_size
)
{
  // chunks smaller than 2*INPUT_WINDOW_SIZE are considered "small" and should not use prefetch
  assert(input_size >= (2 * INPUT_WINDOW_SIZE));

  // Alignment check
  constexpr uint32_t LINE_SIZE = 128;
  const uint32_t input_alignment_offset = (uintptr_t)input_buffer & (LINE_SIZE - 1);
  //Note: the input alignment fix must be > 0 so that something always gets prefetched
  input_alignment_fix = (LINE_SIZE - input_alignment_offset);
  assert(input_alignment_fix > 0);
  X = PREFETCH_SKIN_SIZE + LINE_SIZE + input_alignment_offset;

  p.num_remaining_bytes = input_size;
  p.ix_prefetch = 0;

  for (uint32_t i = thread_warp_ix(); i < input_alignment_fix; i += WARP_SIZE_U)
  {
    prefetch_data[p.ix_prefetch + X + i] = input_buffer[i];
  }

  p.aligned_input_buffer = reinterpret_cast<const uint32_t *>(input_buffer + input_alignment_fix);
}

inline __device__ void unsnap_prefetch_advance(
  unsnap_prefetch_state &p,
  uint8_t (&prefetch_data)[PREFETCH_SCRATCH_SIZE],
  const uint32_t num_bytes_used
)
{
  p.num_remaining_bytes -= num_bytes_used;

  // Preserve 64B
  uint32_t ix_next_prefetch = p.ix_prefetch ^ PREFETCH_SECTOR_SIZE;
  prefetch_data[ix_next_prefetch + thread_warp_ix()] =
    prefetch_data[p.ix_prefetch + INPUT_WINDOW_SIZE + thread_warp_ix()];
  prefetch_data[ix_next_prefetch + WARP_SIZE_U + thread_warp_ix()] =
    prefetch_data[p.ix_prefetch + INPUT_WINDOW_SIZE + WARP_SIZE_U + thread_warp_ix()];

  // Manually load the next input window
  p.ix_prefetch = ix_next_prefetch;

  if (p.num_remaining_bytes >= INPUT_WINDOW_SIZE)
  {
    uint32_t *dst = reinterpret_cast<uint32_t *>(prefetch_data + p.ix_prefetch + PREFETCH_SKIN_SIZE);

    // Memcpy async did not improve performance. Possibly due to register pressure.
    // each thread loads 2*4B = 256B total
    dst[thread_warp_ix()] = p.aligned_input_buffer[thread_warp_ix()];
    dst[WARP_SIZE_U + thread_warp_ix()] = p.aligned_input_buffer[WARP_SIZE_U + thread_warp_ix()];

    p.aligned_input_buffer += (INPUT_WINDOW_SIZE / 4);
  }
}

inline __device__ void
unsnap_prefetch_load_remainder(unsnap_prefetch_state &p, uint8_t (&prefetch_data)[PREFETCH_SCRATCH_SIZE])
{
  const uint8_t *unaligned_input_buffer = reinterpret_cast<const uint8_t *>(p.aligned_input_buffer);

  for (uint32_t i = 0; i < p.num_remaining_bytes; i += WARP_SIZE_U)
  {
    uint32_t ix_byte = i + thread_warp_ix();
    if (ix_byte < p.num_remaining_bytes)
    {
      prefetch_data[p.ix_prefetch + PREFETCH_SKIN_SIZE + ix_byte] = unaligned_input_buffer[ix_byte];
    }
  }
}

} // namespace snappy