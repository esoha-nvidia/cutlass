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

#include "LZ77_decomp.cuh"
#include "types.cuh"
#include "utils.cuh"

// #define LZ_DECOMP_LOGGING 1

// TODO: Go through this code and switch many things from unsigned -> size_t, unsigned-> int, etc.
// Currently a number of things won't work if the frame is bigger than ~2 GB

// Note: Multiple functions are very similar to LZ77_decomp.cuh. Planned to merge them with LZ77_decomp.cuh when reworking ZSTD.
namespace zstd
{

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

#ifndef ANS_PRECOMPUTE_OFFSET

/*
 *  This function implements the repeat offsets logic described here:
 *  https://github.com/facebook/zstd/blob/dev/doc/zstd_compression_format.md#repeat-offsets
 *  
 *  This function starts at start_ix within the warp and sequentially updates
 *  the repeat codes one sequence at a time.
 * 
 *  Input conditions: 
 *    - each thread has data associated with a particular sequence
 *  Output conditions:
 *    - the repeat offsets (in shared memory) are in the correct state given all the sequences
 *      the warp is currently processing
 *    - The offset for each thread with a "thread_warp_ix()" >= the start_ix has been updated
 *  
 */
inline __device__ void sequentially_update_repeat_codes(
  int32_t *repeat_offset,
  int &offset,
  const int literal_length,
  const unsigned start_ix,
  const int num_active
)
{
  for (int ix_val = start_ix; ix_val < num_active; ++ix_val)
  {
    if (thread_warp_ix() == ix_val)
    {
      unsigned val_offset = offset;

      if (val_offset <= 3)
      {
        // This is a repeat offset. Depending on the value of ix_offset, update the repeat codes appropriately
        unsigned ix_offset = val_offset - 1;

        if (literal_length == 0)
        {
          ++ix_offset;
        }

        if (ix_offset == 0)
        {
          val_offset = repeat_offset[0];
        }
        else
        {
          val_offset = ix_offset < 3 ? repeat_offset[ix_offset] : repeat_offset[0] - 1;
          if (ix_offset > 1)
          {
            repeat_offset[2] = repeat_offset[1];
          }
          repeat_offset[1] = repeat_offset[0];
          repeat_offset[0] = val_offset;
        }
      }
      else
      {
        // This isn't a repeated offset. Insert the new value at the front of the repeat offsets and
        // shift the first two offsets back 1 place.
        val_offset -= 3;

        repeat_offset[2] = repeat_offset[1];
        repeat_offset[1] = repeat_offset[0];
        repeat_offset[0] = val_offset;
      }
      offset = val_offset;
    }

    __syncwarp(); // This is necessary for correctness since we need to process one thread at a time
  }
}

inline __device__ void
compute_offset(const bool active, int &offset, const int literal_length, int32_t *repeat_offset, const int num_active)
{
  const unsigned do_repeat_offset = active && offset <= 3;
  const unsigned any_repeated_offset = __ballot_sync(WARP_ALL, do_repeat_offset);
  // Manually update the repeated offsets for the first 3 threads.
  const unsigned active_mask = (1 << num_active) - 1;
  unsigned this_offset = offset - 3;
  if (any_repeated_offset)
  {
    unsigned ix_offset = offset - 1;

    if (literal_length == 0)
    {
      ++ix_offset;
    }
    unsigned do_update_offset = do_repeat_offset && ix_offset > 0;
    unsigned any_update_offset = __ballot_sync(WARP_ALL, do_update_offset);
    if (not any_update_offset)
    {
      // The only repeat values don't change the offset codes.
      bool do_find_prev_offset = false;
      if (offset > 3)
      {
        offset = this_offset;
      }
      else
      {
        do_find_prev_offset = true;
      }

      int ix_shfl = thread_warp_ix();
      if (active and do_find_prev_offset)
      {
        assert(ix_offset == 0);
        unsigned dont_repeat_offset = ~any_repeated_offset;
        unsigned mask = (1 << thread_warp_ix()) - 1;
        unsigned prev_bits = dont_repeat_offset & mask;

        int ix_first = 31 - __clz(prev_bits);
        if (ix_first == -1)
        {
          // In this case we'll repeat the offset from the previous iteration
          offset = repeat_offset[0];
        }
        else
        {
          assert(ix_first >= 0);
          ix_shfl = ix_first;
        }
      }

      __syncwarp(WARP_ALL); // To silence racecheck, but this is redundant with the below shuffle.
      offset = __shfl_sync(WARP_ALL, offset, ix_shfl);

      // Update the repeat_offsets
      unsigned mask_all_after_this = ~((1 << (thread_warp_ix() + 1)) - 1);
      unsigned mask_active_after_this = active_mask & mask_all_after_this;
      unsigned mask_unrepeated_offset_after_this = ~any_repeated_offset & mask_active_after_this;
      int index = __popc(mask_unrepeated_offset_after_this); // The index is, how many are active after this one?
      unsigned update_count =
        min(__popc(~any_repeated_offset & active_mask), 3); // How many updates are there? No more than 3.
      unsigned shuffle_count = 3 - update_count;

      // First shuffle if there are fewer than 3 updates. I.e. if there is 1 update,
      // 1 becomes 0 and 2 becomes 1.
      //
      if (thread_warp_ix() < shuffle_count)
      {
        const unsigned shuffle_mask = (1 << shuffle_count) - 1;
        const unsigned shuffle_val = repeat_offset[thread_warp_ix()];
        __syncwarp(shuffle_mask);
        repeat_offset[thread_warp_ix() + update_count] = shuffle_val;
      }

      __syncwarp(WARP_ALL); // Necessary because the value being set is used above
      if (active and index < 3)
      {
        repeat_offset[index] = offset;
      }
    }
    else
    {
      // Get the first value, loop
      int start_ix = __ffs(any_repeated_offset) - 1;
      if (start_ix >= 3)
      {
        // If the first repeated offset is at least in slot 3, then we can just initialize
        // using the previous three offsets
        if (thread_warp_ix() < start_ix and start_ix - thread_warp_ix() <= 3)
        {
          repeat_offset[abs(start_ix - 1 - thread_warp_ix())] = this_offset;
        }
      }
      else
      {
        start_ix = 0; // Just do this sequentially for all values
      }

      if (thread_warp_ix() < start_ix)
      {
        offset = this_offset;
      }
      __syncwarp(); // Need to have finished the above before sequential update
      sequentially_update_repeat_codes(repeat_offset, offset, literal_length, start_ix, num_active);
    }
  }
  else if (num_active >= 3)
  {
    // If no repeated offsets, the repeat codes are the last three
    int shuffle_lane = num_active - 1 - thread_warp_ix();
    if (num_active > thread_warp_ix() and num_active - thread_warp_ix() <= 3)
    {
      repeat_offset[num_active - 1 - thread_warp_ix()] = this_offset;
    }
    offset = this_offset;
  }
  else
  {
    sequentially_update_repeat_codes(repeat_offset, offset, literal_length, 0, num_active);
  }
}

#else

__device__ __noinline__ void
compute_offset(const bool active, int &offset, const int literal_length, int32_t *repeat_offset, const int /*num active*/)
{
  // If encounter -1, -2, -3, then just use the value from repeat offset
  if (active)
  {
    if (offset < 0)
    {
      offset = repeat_offset[abs(offset) - 1]; // -1 to zero-index
    }
  }
}

#endif

__device__ inline void reinit_repeat_codes(int32_t *repeat_offset)
{
  if (thread_warp_ix() == 0)
  {
    repeat_offset[0] = ZSTD_INITIAL_REPEAT_OFFSETS[0];
    repeat_offset[1] = ZSTD_INITIAL_REPEAT_OFFSETS[1];
    repeat_offset[2] = ZSTD_INITIAL_REPEAT_OFFSETS[2];
  }
  __syncwarp(); // necessary so all threads have the result
}

template <bool is_last>
inline __device__ void do_per_thread_match_copies(
  const int literal_length,
  int &match_length,
  const int offset,
  int &ix_output,
  uint8_t *output_buffer,
  const int num_active,
  const bool active
)
{
  int num_copy;
  int ix_base = ix_output - offset;

  const uint8_t *input_ptr = &output_buffer[ix_base];
  unsigned align_val = min(4u - static_cast<unsigned>((uintptr_t)input_ptr % 4), match_length);
  if (match_length < 16)
  {
    num_copy = match_length;
  }
  else
  {
    num_copy = 12 + align_val;
  }

  // Determine whether we're before the first, or within the literal region of another
  int copy_start = ix_output - offset;
  int copy_end = copy_start + num_copy;
  int first_ix_output = __shfl_sync(WARP_ALL, ix_output, 0);
  bool do_copy = active and copy_end < first_ix_output;
  int ix_loc = 0;
  if (__any_sync(WARP_ALL, not do_copy))
  {
    // This checks whether we're in the literal region of another by searching for the output index of the
    // copy thread that starts just start index
    // of the copy source.
    const int num_search_iters = 5; // num_search_iters = log2(32)
    lower_bound<num_search_iters>(ix_loc, copy_start, ix_output);

    int ix_shfl = __shfl_sync(WARP_ALL, match_length + ix_output, ix_loc);
    int next_ix_out = __shfl_sync(WARP_ALL, ix_output, ix_loc + 1);
    if (not do_copy)
    {
      if (ix_shfl < copy_start)
      {
        if (ix_loc + 1 == thread_warp_ix() or next_ix_out >= copy_end)
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
        // output_buffer[this_ix_output++] = output_buffer[ix_base + ix_byte];
        output_buffer[ix_output++] = output_buffer[copy_start + ix_byte];
      }
    }
    else
    {
      // Perform alignment, then vectorized copies
      int ix_byte = 0;
      // Handling the case of:
      // <literal: "xab"> <copy: offset=2 length=4> which results in "xababab"
      if (copy_end > ix_output)
      {
        align_val = num_copy; // in this case, simplify the logic by doing non-aligned copies
      }

      for (; ix_byte < align_val; ++ix_byte)
      {
        output_buffer[ix_output++] = output_buffer[copy_start + ix_byte];
      }

      // Then do 4-byte copies as possible
      int bytes_remaining = num_copy - ix_byte;
      while (bytes_remaining > 0)
      {
        int copy_length = do_per_thread_input_aligned_copy(
          &output_buffer[ix_base + ix_byte],
          &output_buffer[ix_output],
          bytes_remaining
        );
        bytes_remaining -= copy_length;
        ix_output += copy_length;
        ix_byte += copy_length;
      }
    }
  }
}

template <bool is_last>
inline __device__ void do_cooperative_match_copies(
  const int literal_length,
  int &match_length,
  const int offset,
  int &ix_output,
  uint8_t *output_buffer,
  const int num_active
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
    int this_ix_output = __shfl_sync(WARP_ALL, ix_output, loc);
    int this_offset = __shfl_sync(WARP_ALL, offset, loc);

    int base_ix = this_ix_output - this_offset;
    if (this_offset >= this_match_length)
    {
      if (is_last)
      {
        // Perform bytewise copying
        warp_input_nonaligned_copy(&output_buffer[base_ix], &output_buffer[this_ix_output], this_match_length);
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
      for (unsigned ix_byte = thread_warp_ix(); ix_byte < this_match_length; ix_byte += 32)
      {
        assert(this_ix_output + ix_byte < ZSTD_BLOCK_SIZE_MAX);
        output_buffer[this_ix_output + ix_byte] = output_buffer[base_ix + static_cast<int>(ix_byte % this_offset)];
      }
    }

    // Update vote -- toggle / clear the bit we just updated
    vote ^= 1U << loc;
    if (thread_warp_ix() == loc)
    {
      ix_output += match_length;
    }
  }
}

template <bool is_last>
inline __device__ void do_match_copies(
  const int literal_length,
  int match_length,
  const int offset,
  bool &active,
  int &ix_output,
  uint8_t *output_buffer,
  const int num_active
)
{
  do_per_thread_match_copies<is_last>(literal_length, match_length, offset, ix_output, output_buffer, num_active, active);
  __syncwarp(); // Fix racecheck FA. do_cooperative_match_copies starts with a ballot_sync, so the syncwarp() is redundant.

  // part 2 for per-sequence copy logic
  do_cooperative_match_copies<is_last>(literal_length, match_length, offset, ix_output, output_buffer, num_active);

  __syncwarp(); // The last thread might not have finished before now
  ix_output = __shfl_sync(WARP_ALL, ix_output, num_active - 1);
}

inline __device__ void
do_literal_copies(const uint8_t rle_byte, int &ix_output, int literal_length, uint8_t *output_buffer)
{
  // Copy up to the alignment boundary.
  for (int ix = 0; ix < literal_length; ++ix)
  {
    output_buffer[ix_output++] = rle_byte;
  }
}

inline __device__ void
do_literal_copies(const uint8_t *literals, int &ix_literal, int &ix_output, int literal_length, uint8_t *output_buffer)
{
  unsigned align_val = min(4u - static_cast<unsigned>((uintptr_t)(&literals[ix_literal]) % 4), literal_length);
#pragma unroll 1
  for (int ix = 0; ix < align_val; ++ix)
  {
    output_buffer[ix_output++] = literals[ix_literal++];
  }

  literal_length -= align_val;
#pragma unroll 1
  while (literal_length > 0)
  {
    uchar4 reg = *reinterpret_cast<const uchar4 *>(&literals[ix_literal]);

    output_buffer[ix_output++] = reg.x;
    if (literal_length > 1)
    {
      output_buffer[ix_output++] = reg.y;
    }
    if (literal_length > 2)
    {
      output_buffer[ix_output++] = reg.z;
    }
    if (literal_length > 3)
    {
      output_buffer[ix_output++] = reg.w;
    }

    ix_literal += min(literal_length, 4);
    literal_length -= 4;
  }
}

// lit_only and seq_only are used to allow a preprocessing pass:
// If lit_only, then a warp is doing independent work (literal copies without match copies) on a late block
// If seq_only, then a warp is doing the match copies after the literal copies have already been completed by another warp
// Otherwise, both are false
inline __device__ unsigned decompress_block_lz4(
  uint8_t *output_buffer,
  DeviceBlockShare &block_share,
  const uint8_t *literals,
  const uint8_t rle_literal_byte,
  int32_t *repeat_offset,
  bool lit_only = false,
  bool seq_only = false,
  bool is_rle_literals = false
)
{
  assert(not(lit_only and seq_only));
  // Iterate through the sequences. Can do this 32 at a time along the same lines as GDeflate
  __shared__ IntWarpScan::TempStorage temp_scan_storage[NUM_WARPS_PER_CTA];
  IntWarpScan warp_scan{temp_scan_storage[ix_warp()]};

  int num_sequences = block_share.num_sequences;
  sequence *seq_buffer = block_share.sequence_buffer;

  int ix_literal = 0;
  int ix_output = 0;
  int final_ix_literal = 0;

  auto &decode_seq_count = block_share.decode_seq_count;
  auto &decode_lit_count = block_share.decode_lit_count;

  int current_decoded_seq_count = 0;
  if (not seq_only)
  {
    // In the seq only case, the literal copy method already waited for all the sequences to complete
    // Thus we can avoid loading this value.
    // relaxed because it synchronizes with the below lit count load
    current_decoded_seq_count = decode_seq_count.load(cuda::std::memory_order_relaxed);
  }
  int current_decoded_lit_count = decode_lit_count.load(cuda::std::memory_order_acquire);

  for (int base_ix_seq = 0; base_ix_seq < num_sequences; base_ix_seq += WARP_SIZE)
  {
    // Check for the block share
    int literal_length = 0;
    int offset = 0;
    int match_length = 0;
    int ix_top_seq = min(num_sequences, base_ix_seq + WARP_SIZE - 1);

    const int ix_seq = base_ix_seq + thread_warp_ix();
    bool active = ix_seq < num_sequences;

    int num_active = min(num_sequences - base_ix_seq, WARP_SIZE);

    if (not seq_only)
    {
// In the seq only case, the literal copy method already waited for all the sequences to complete
// Thus we can avoid loading this value
#ifdef STAGE_LOGGING
      if (decode_seq_count < ix_top_seq)
      {
        print0(
          "bid %d warp %d waiting for seq %d %d frame %d block %d clock %lu\n",
          blockIdx.x,
          ix_warp(),
          decode_seq_count.load(cuda::std::memory_order_relaxed),
          ix_top_seq,
          block_share.ix_frame,
          block_share.ix_block,
          cuda::std::chrono::system_clock::now()
        );
      }
#endif
      wait_for_atomic(decode_seq_count, ix_top_seq, current_decoded_seq_count, ZSTD_LONG_SLEEP_NS);
    }

    if (active)
    {
      const auto &sequence = seq_buffer[ix_seq];
      literal_length = sequence[0];
      offset = sequence[1];
      match_length = sequence[2];
    }

    int scan_result;
    if (not seq_only)
    {
      warp_scan.ExclusiveSum(literal_length, scan_result);
      ix_literal += scan_result;
      final_ix_literal = __shfl_sync(WARP_ALL, ix_literal + literal_length, WARP_SIZE - 1);
    }

    warp_scan.ExclusiveSum(literal_length + match_length, scan_result);
    ix_output += scan_result;

    __syncwarp(WARP_ALL);

    if (not seq_only)
    {
      // Wait on the appropriate # of literals
      __syncwarp();

      if (is_rle_literals)
      {
        do_literal_copies(rle_literal_byte, ix_output, literal_length, output_buffer);
      }
      else
      {
        const int req_literals = final_ix_literal;
        assert(req_literals <= 128 * 1024);

#ifdef STAGE_LOGGING
        // bool waiting = false;
        if (decode_lit_count < req_literals)
        {
          // waiting = true;
          print0(
            "bid %d warp %d waiting decode lit count %d req literals %d total literals %d total seq %d frame %d block "
            "%d clock %lu\n",
            blockIdx.x,
            ix_warp(),
            decode_lit_count.load(cuda::std::memory_order_relaxed),
            req_literals,
            block_share.num_literals,
            block_share.num_sequences,
            block_share.ix_frame,
            block_share.ix_block,
            cuda::std::chrono::system_clock::now()
          );
        }
#endif
        wait_for_atomic(decode_lit_count, req_literals, current_decoded_lit_count, ZSTD_LONG_SLEEP_NS);
        do_literal_copies(literals, ix_literal, ix_output, literal_length, output_buffer);
      }
      __syncwarp();

      if (lit_only)
      {
        ix_output = __shfl_sync(WARP_ALL, ix_output, WARP_SIZE - 1) +
                    __shfl_sync(WARP_ALL, match_length, WARP_SIZE - 1);
      }
      ix_literal = final_ix_literal;
    }
    else
    {
      ix_output += literal_length;
    }

    if (not lit_only)
    {
      compute_offset(active, offset, literal_length, repeat_offset, num_active);

#ifdef LZ_DECOMP_LOGGING
      if (block_share.ix_frame == 2030 and block_share.ix_block == 0)
      {
        printf(
          "frame %d block %d tid %d seq %d match len %d lit len %d off %d ix literal %d ix out %d num lit %d num seq "
          "%d\n",
          block_share.ix_frame,
          block_share.ix_block,
          threadIdx.x,
          ix_seq,
          match_length,
          literal_length,
          offset,
          ix_literal,
          ix_output,
          block_share.num_literals,
          block_share.num_sequences
        );
      }
#endif

      assert(not active or offset > 0);

      __syncwarp(WARP_ALL); // Need previous shared / global writes to be visible

      // Note, that each thread handles a separate sequence.
      // The compressed stream can end on a match copy,
      // hence we need to pay special attention not to over-read the
      // input buffer.
      bool is_last = (base_ix_seq + WARP_SIZE) >= num_sequences;
      if (is_last)
      {
        do_match_copies<true>(literal_length, match_length, offset, active, ix_output, output_buffer, num_active);
      }
      else
      {
        do_match_copies<false>(literal_length, match_length, offset, active, ix_output, output_buffer, num_active);
      }
    }
  }

  int num_rem_literals = 0;
  int final_ix_output = ix_output;

  // Don't return until all sequences are decoded
  // We need the "total seq bytes" variable to be filled in
  wait_for_atomic(decode_seq_count, num_sequences, current_decoded_seq_count, ZSTD_LONG_SLEEP_NS);

  if (not seq_only)
  {
    num_rem_literals = block_share.num_literals - final_ix_literal;
    // Copy remaining literals
    if (not is_rle_literals)
    {

#ifdef STAGE_LOGGING
      if (decode_lit_count < block_share.num_literals)
      {
        print0(
          "bid %d warp %d decode lit count %d req literals %d total literals %d decode seq %d total seq %d frame %d "
          "block %d clock %lu\n",
          blockIdx.x,
          ix_warp(),
          decode_lit_count.load(cuda::std::memory_order_relaxed),
          block_share.num_literals,
          block_share.num_literals,
          block_share.decode_seq_count.load(cuda::std::memory_order_relaxed),
          block_share.num_sequences,
          block_share.ix_frame,
          block_share.ix_block,
          cuda::std::chrono::system_clock::now()
        );
      }
#endif
      wait_for_atomic(decode_lit_count, block_share.num_literals, current_decoded_lit_count, ZSTD_SHORT_SLEEP_NS);
      copy_remaining_literals(num_rem_literals, literals, final_ix_literal, output_buffer, final_ix_output);
    }
    else
    {
      copy_remaining_literals(num_rem_literals, rle_literal_byte, output_buffer, final_ix_output);
    }

    assert(final_ix_output + num_rem_literals <= ZSTD_BLOCK_SIZE_MAX);
  }

  return final_ix_output + num_rem_literals;
}

} // namespace zstd