/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
#pragma once

#include <cassert>
#include <ciso646>

#include "CudaConstants.h"

inline __device__ int thread_warp_ix() { return threadIdx.x % WARP_SIZE; }

template <int N, typename data_t>
inline __device__ void lower_bound(int &ix_result, data_t search_val, const data_t *search_array, int array_size)
{
  // Adapted from binary_search_leftmost algorithm from here:
  // https://en.wikipedia.org/wiki/Binary_search_algorithm
  int &left = ix_result;
  left = 0;
  int right = array_size;

#pragma unroll
  for (int iter = 0; iter < N; ++iter)
  {
    int ix_mid = (left + right) / 2;

    if (search_array[ix_mid] <= search_val)
    {
      left = ix_mid;
    }
    else
    {
      right = ix_mid;
    }
  }
}

inline __device__ int do_per_thread_input_aligned_copy(const uint8_t *src, uint8_t *dst, int copy_length)
{
  if (copy_length > 0)
  {
    // Do a 4-byte copy
    // Note: This 4-byte load might cause uninitalized false alarms
    //       in compute-sanitizer initcheck (depending on the dataset),
    //       even if we are not using all the 4 bytes. I replaced
    //       this block with a byte-based read, and the errors went away.
    //       Therefore, once compute-sanitizer has proper suppression support,
    //       this 4-byte load needs to be suppressed.
    char4 input = *reinterpret_cast<const char4 *>(src);
    dst[0] = input.x;
    if (copy_length >= 4)
    {
      dst[1] = input.y;
      dst[2] = input.z;
      dst[3] = input.w;
      return 4;
    }
    else
    {
      if (copy_length > 1)
      {
        dst[1] = input.y;
      }
      if (copy_length > 2)
      {
        dst[2] = input.z;
      }
      return copy_length;
    }
  }
  else
  {
    return 0;
  }
}

inline __device__ void warp_input_4byte_aligned_copy(const uint8_t *src, uint8_t *dst, int copy_length)
{
  // Then we can finish the copy, doing reading 4 bytes at a time
  // Note: intentionally using unsigned WARP_SIZE_U to avoid unnecessary loop unrollings
  for (int ix_base = 0; ix_base < copy_length; ix_base += WARP_SIZE_U * 4)
  {
    const int thread_ix = ix_base + 4 * thread_warp_ix();
    int bytes_remaining = copy_length - thread_ix;
    do_per_thread_input_aligned_copy(&src[thread_ix], &dst[thread_ix], bytes_remaining);
  }
}

inline __device__ void warp_input_nonaligned_copy(const uint8_t *src, uint8_t *dst, int copy_length)
{
  // Note: intentionally using unsigned WARP_SIZE_U to avoid unnecessary loop unrollings
  for (int i = thread_warp_ix(); i < copy_length; i += WARP_SIZE_U)
  {
    dst[i] = src[i];
  }
}

/*
  * This function handles the special case where we're doing a long copy of repeated bytes, 
  * and we only need to read a collection of bytes into the warp once
  */
inline __device__ void match_copy_repeated_output(
  const int match_length,
  const uint8_t *input_buffer,
  const int offset,
  uint8_t *const output_buffer
)
{
  unsigned shared_load = 0;
  if (thread_warp_ix() < offset)
  {
    shared_load = input_buffer[thread_warp_ix()];
  }

  uint8_t thread_val = static_cast<uint8_t>(__shfl_sync(WARP_ALL, shared_load, thread_warp_ix() % offset));
  if (WARP_SIZE % offset == 0)
  {
    // In this case, the byte read by each thread will be the same on every iteration. Can make use of this optimization
    for (unsigned ix_byte = thread_warp_ix(); ix_byte < match_length; ix_byte += WARP_SIZE_U)
    {
      output_buffer[ix_byte] = thread_val;
    }
  }
  else
  {
    // The base case where we have to do a shuffle for each iteration
    for (unsigned ix_base_byte = 0; ix_base_byte < match_length; ix_base_byte += WARP_SIZE_U)
    {
      unsigned ix_byte = ix_base_byte + thread_warp_ix();
      uint8_t shfl_val = static_cast<uint8_t>(__shfl_sync(WARP_ALL, shared_load, ix_byte % offset));
      if (ix_byte < match_length)
      {
        output_buffer[ix_byte] = shfl_val;
      }
    }
  }
}

inline __device__ void do_per_thread_match_copies(
  int &match_length,
  const int offset,
  int *ix_output,
  uint8_t *output_buffer,
  const int num_active,
  const bool is_last,
  int &this_ix_output
)
{
  int num_copy;
  int ix_base = this_ix_output - offset;

  const uint8_t *input_ptr = &output_buffer[ix_base];
  unsigned align_val = is_last ? 0 : min(4u - static_cast<unsigned>((uintptr_t)input_ptr % 4), match_length);
  if (match_length < 16)
  {
    num_copy = match_length;
  }
  else
  {
    num_copy = 12 + align_val;
  }

  // Determine whether we're before the first, or within the literal region of another
  int copy_start = this_ix_output - offset;
  int copy_end = copy_start + num_copy;
  bool do_copy = copy_end < ix_output[0];
  int ix_loc = 0;
  if (__any_sync(WARP_ALL, not do_copy))
  {
    if (not do_copy)
    {
      // This checks whether we're in the literal region of another by searching for the output index of the
      // copy thread that starts just start index
      // of the copy source.
      const int num_search_iters = 6; // num_search_iters = log2(32) + 1
      lower_bound<num_search_iters>(ix_loc, copy_start, ix_output, num_active);
    }
    int shfl_match_length = __shfl_sync(WARP_ALL, match_length, ix_loc);
    if (not do_copy)
    {
      if (ix_output[ix_loc] + shfl_match_length < copy_start)
      {
        if (ix_loc + 1 == thread_warp_ix() or ix_output[ix_loc + 1] >= copy_end)
        {
          do_copy = true;
        }
      }
    }
  }

  if (do_copy)
  {
    match_length -= num_copy;
    if (is_last)
    {
      // Perform byte-wise copies
      for (int ix_byte = 0; ix_byte < num_copy; ++ix_byte)
      {
        output_buffer[this_ix_output++] = output_buffer[ix_base + ix_byte];
      }
    }
    else
    {
      int ix_byte = 0;
      // Handling the case of:
      // <literal: "xab"> <copy: offset=2 length=4> which results in "xababab"
      if (copy_end > this_ix_output)
      {
        align_val = num_copy; // in this case, simplify the logic by doing non-aligned copies
      }

      for (; ix_byte < align_val; ++ix_byte)
      {
        output_buffer[this_ix_output++] = output_buffer[copy_start + ix_byte];
      }

      // Then do 4-byte copies as possible
      int bytes_remaining = num_copy - ix_byte;
      while (bytes_remaining > 0)
      {
        // Note:
        // do_per_thread_input_aligned_copy might access uninitialized data as it loads using a 4-byte load
        // unconditionally. This might be flagged via `compute-sanitizer --tool initcheck`. However,
        // given we are executing this branch with `!is_last`, we can be certain that we are not accessing
        // out-of-bounds memory even with a 4-byte load.
        assert((ix_base + ix_byte + min(bytes_remaining, 4)) <= this_ix_output);
        int copy_length = do_per_thread_input_aligned_copy(
          &output_buffer[ix_base + ix_byte],
          &output_buffer[this_ix_output],
          bytes_remaining
        );
        bytes_remaining -= copy_length;
        this_ix_output += copy_length;
        ix_byte += copy_length;
      }
    }
  }
}

template <bool is_snappy>
inline __device__ void do_cooperative_match_copies(
  int &match_length,
  const int offset,
  int *ix_output,
  uint8_t *output_buffer,
  const int num_active,
  const bool is_last,
  int my_ix_output
)
{
  const bool active = match_length != 0;
  unsigned vote = __ballot_sync(WARP_ALL, active);
  while (vote)
  {
    // Loop through remaining copies that weren't completed by the per_thread copy logic
    // First, find the next incomplete copy and shuffle its appropriate values to all the threads
    int loc = __ffs(vote) - 1;
    int this_match_length = __shfl_sync(WARP_ALL, match_length, loc);
    int this_ix_output = __shfl_sync(WARP_ALL, my_ix_output, loc);
    int this_offset = __shfl_sync(WARP_ALL, offset, loc);

    int base_ix = this_ix_output - this_offset;
    if (this_offset >= this_match_length)
    {
      if constexpr (is_snappy)
      {
        // Match copy optimization only for snappy, because the maximum match length is 64.
        if (thread_warp_ix() < this_match_length)
        {
          output_buffer[this_ix_output + thread_warp_ix()] = output_buffer[base_ix + thread_warp_ix()];
        }
        if (WARP_SIZE + thread_warp_ix() < this_match_length)
        {
          output_buffer[this_ix_output + WARP_SIZE + thread_warp_ix()] =
            output_buffer[base_ix + WARP_SIZE + thread_warp_ix()];
        }
      }
      else if (is_last)
      {
        // Perform byte-wise copying cooperatively
        warp_input_nonaligned_copy(output_buffer + base_ix, output_buffer + this_ix_output, this_match_length);
      }
      else
      {
        // Case for copying the full match copy from the copy buffer
        // First, check whether 4-byte input alignment is still required (because the per-thread copy wasn't performed)
        const uint8_t *input_ptr = &output_buffer[base_ix];
        int align_copy_bytes = WARP_SIZE - static_cast<int>((uintptr_t)input_ptr % 4);
        align_copy_bytes = align_copy_bytes == WARP_SIZE ? 0 : align_copy_bytes;

        if (thread_warp_ix() < align_copy_bytes and thread_warp_ix() < this_match_length)
        {
          output_buffer[this_ix_output + thread_warp_ix()] = output_buffer[base_ix + thread_warp_ix()];
        }
        this_match_length -= align_copy_bytes;
        base_ix += align_copy_bytes;
        this_ix_output += align_copy_bytes;

        // Then we can finish the copy, doing reading 4 bytes at a time
        warp_input_4byte_aligned_copy(&output_buffer[base_ix], &output_buffer[this_ix_output], this_match_length);
      }
    }
    else if (this_offset <= WARP_SIZE and this_match_length > WARP_SIZE)
    {
      // Repeating N < 32 bytes over and over. Can optimize by loading each byte into registers
      match_copy_repeated_output(
        this_match_length,
        &output_buffer[base_ix],
        this_offset,
        &output_buffer[this_ix_output]
      );
    }
    else
    {
      // repeat offset bytes. Can't do the earlier optimization.
      for (unsigned ix_byte = thread_warp_ix(); ix_byte < this_match_length; ix_byte += WARP_SIZE_U)
      {
        output_buffer[this_ix_output + ix_byte] = output_buffer[base_ix + static_cast<int>(ix_byte % this_offset)];
      }
    }

    // Update vote -- toggle / clear the bit we just updated
    vote ^= 1U << loc;
    if (thread_warp_ix() == loc)
    {
      ix_output[loc] += match_length;
    }
  }
}

template <bool is_snappy>
inline __device__ void do_match_copies(
  int match_length,
  const int offset,
  bool &active,
  int *ix_output,
  uint8_t *output_buffer,
  const int num_active,
  const bool is_last
)
{
  assert(active || match_length == 0);

  int this_ix_output = ix_output[thread_warp_ix()];

  do_per_thread_match_copies(match_length, offset, ix_output, output_buffer, num_active, is_last, this_ix_output);
  __syncwarp(); // Fix racecheck FA. do_cooperative_match_copies starts with a ballot_sync, so the syncwarp() is redundant.

  // part 2 for per-sequence copy logic
  do_cooperative_match_copies<
    is_snappy>(match_length, offset, ix_output, output_buffer, num_active, is_last, this_ix_output);

  // Note: when using this version of `do_match_copies` in LZ4/Snappy/Deflate,
  //       we don't use `ix_output` after `do_match_copies()`. Therefore, unless we start using `ix_output`
  //       again, or `do_match_copies()` preceds `do_literal_copies()`, this part could be removed
  //       or templated away. TODO
  __syncwarp(); // The last thread might not have finished before now

  if (active)
  {
    ix_output[thread_warp_ix()] = ix_output[num_active - 1];
  }
}

inline __device__ void per_thread_literal_copies(
  unsigned &literal_length,
  uint8_t *output_buffer,
  int &ix_output,
  const uint8_t *literal_buffer,
  unsigned &ix_literal,
  bool is_last
)
{
  // Copy up to the alignment boundary.
  const uint8_t *this_input = literal_buffer + ix_literal;

  int num_copy;
  if (literal_length < 16)
  {
    num_copy = literal_length;
  }
  else
  {
    // Get close to 16, but set up aligned 4-byte copies
    int align_val = is_last ? 16 : (16 - (uintptr_t)this_input % 4);
    num_copy = min(align_val, literal_length);
  }

  for (int ix = 0; ix < num_copy; ++ix)
  {
    output_buffer[ix_output++] = this_input[ix];
  }

  ix_literal += num_copy;
  literal_length -= num_copy;
}

inline __device__ void cooperative_literal_copies(
  int &ix_output,
  const uint8_t *literal_buffer,
  uint8_t *output_buffer,
  const int literal_length,
  unsigned &ix_literal,
  bool is_last
)
{
  const bool active = literal_length != 0;

  // Perform cooperative copying:
  // - if we are NOT at the end of the chunk, do 4-byte copies
  // - otherwise do bytewise copies
  unsigned vote = __ballot_sync(WARP_ALL, active);
  while (vote)
  {
    // Get the first location
    int loc = __ffs(vote) - 1;
    unsigned this_literal_length = __shfl_sync(WARP_ALL, literal_length, loc);
    unsigned this_ix_literal = __shfl_sync(WARP_ALL, ix_literal, loc);
    int this_ix_output = __shfl_sync(WARP_ALL, ix_output, loc);

    if (is_last)
    {
      warp_input_nonaligned_copy(&literal_buffer[this_ix_literal], &output_buffer[this_ix_output], this_literal_length);
    }
    else
    {
      warp_input_4byte_aligned_copy(
        &literal_buffer[this_ix_literal],
        &output_buffer[this_ix_output],
        this_literal_length
      );
    }

    if (thread_warp_ix() == loc)
    {
      ix_output += this_literal_length;
      ix_literal += this_literal_length;
    }

    // Update vote -- toggle / clear the bit we just updated
    vote ^= 1U << loc;
  }
}

inline __device__ void do_literal_copies(
  const uint8_t *literals,
  unsigned &ix_literal,
  unsigned literal_length,
  const bool active,
  int &ix_output,
  uint8_t *output_buffer,
  bool is_last
)
{
  if (active)
  {
    per_thread_literal_copies(literal_length, output_buffer, ix_output, literals, ix_literal, is_last);
  }

  cooperative_literal_copies(ix_output, literals, output_buffer, literal_length, ix_literal, is_last);
  __syncwarp(); // Need to ensure all writes have finished before continuing
}

inline __device__ void
copy_remaining_literals(const int num_rem_literals, const uint8_t rle_byte, uint8_t *output_buffer, const int ix_output)
{
  for (int ix = thread_warp_ix(); ix < num_rem_literals; ix += WARP_SIZE)
  {
    output_buffer[ix_output + ix] = rle_byte;
  }
}

inline __device__ void copy_remaining_literals(
  const int num_rem_literals,
  const uint8_t *literal_buffer,
  const int ix_literal,
  uint8_t *output_buffer,
  const int ix_output
)
{
  for (int ix = thread_warp_ix(); ix < num_rem_literals; ix += WARP_SIZE)
  {
    output_buffer[ix_output + ix] = literal_buffer[ix_literal + ix];
  }
}
