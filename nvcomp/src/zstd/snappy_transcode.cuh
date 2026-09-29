#pragma once

#include "common.h"
#include "LZ77_decomp.cuh"

namespace zstd
{

inline __device__ void
write_match_tag(uint8_t *dst, int &ix_output, const int offset_len, const int iter_match_len, const int offset)
{
  uint8_t tag_byte = (offset_len == 2 ? 0x2 : 0x3) | ((iter_match_len - 1) << 2);
  dst[ix_output++] = tag_byte;
  write_bytes_le_unaligned(dst, ix_output, offset, offset_len);
}

inline __device__ void
write_literal_tag(uint8_t *dst, int &ix_output, const int literal_length, const int num_literal_tag_bytes)
{
  // How many bytes?
  uint8_t tag_byte = num_literal_tag_bytes == 0 ? literal_length - 1 : num_literal_tag_bytes + 59;
  tag_byte <<= 2;
  dst[ix_output++] = tag_byte;
  write_bytes_le_unaligned(dst, ix_output, literal_length - 1, num_literal_tag_bytes);
}

inline __device__ int compute_literal_tag_byte_count(const int literal_length)
{
  return literal_length < 60 ? 0
                             : nvcomp::roundUpDiv(
                                 32 - __clz(literal_length),
                                 8
                               ); // 0,1,2,3, or 4 bytes used for literal length little endian
}

inline __device__ int transcode_snappy_lz(
  uint8_t *output_buffer,
  DeviceBlockShare &block_share,
  const uint8_t *literals,
  const uint8_t rle_literal_byte,
  const bool is_rle_literals,
  int32_t *repeat_offset,
  int *ix_output,
  int *ix_literal
)
{
  // Iterate through the sequences. Can do this 32 at a time along the same lines as GDeflate
  IntWarpScan::TempStorage temp_scan_storage;
  IntWarpScan warp_scan{temp_scan_storage};

  int num_sequences = block_share.num_sequences;
  sequence *seq_buffer = block_share.sequence_buffer;
  int &this_ix_output = ix_output[thread_warp_ix()];
  int &this_ix_literal = ix_literal[thread_warp_ix()];
  this_ix_literal = 0;
  this_ix_output = 0;
  auto &decode_seq_count = block_share.decode_seq_count;
  auto &decode_lit_count = block_share.decode_lit_count;

  int current_decoded_seq_count = decode_seq_count.load(cuda::std::memory_order_relaxed);
  int current_decoded_lit_count = decode_lit_count.load(cuda::std::memory_order_acquire);

  for (int base_ix_seq = 0; base_ix_seq < num_sequences; base_ix_seq += WARP_SIZE)
  {
    // Check for the block share
    int literal_length = 0;
    int offset = 0;
    int match_length = 0;
    int ix_top_seq = min(num_sequences, base_ix_seq + WARP_SIZE);

    const int ix_seq = base_ix_seq + thread_warp_ix();
    bool active = ix_seq < num_sequences;

    int num_active = min(num_sequences - base_ix_seq, WARP_SIZE);

    wait_for_atomic(decode_seq_count, ix_top_seq, current_decoded_seq_count, ZSTD_LONG_SLEEP_NS);

    if (active)
    {
      const auto &sequence = seq_buffer[ix_seq];
      literal_length = sequence[0];
      offset = sequence[1];
      match_length = sequence[2];
    }

    // How big should the scan be?
    // Tag bytes -- 1 for literals, ceilDiv(match_length / 64) for matches
    // Offsets -- usually 2 byte, except 4 byte for >= 64 KB offset. Unfortunately this section repeats.

    // Match len --
    // Snappy token:
    // Ouch, we don't know how many bytes to use
    // Can just use 4 byte offsets until we have repeat offsets. Phew.
    compute_offset(active, offset, literal_length, repeat_offset, num_active);
    int offset_len = 2;
    constexpr int MIN_FOUR_BYTE_OFFSET = 64 << 10;
    if (offset >= MIN_FOUR_BYTE_OFFSET)
    {
      offset_len = 4;
    }

    const int match_tag_count = ceilDiv(match_length, 64);
    const int num_match_tag_bytes = match_tag_count * (1 + offset_len);
    const int num_literal_tag_bytes = literal_length > 0 ? 1 + compute_literal_tag_byte_count(literal_length) : 0;
    const int total_seq_bytes = literal_length + num_match_tag_bytes + num_literal_tag_bytes;

    int scan_result;
    warp_scan.ExclusiveSum(literal_length, scan_result);
    this_ix_literal += scan_result;
    if (thread_warp_ix() == num_active - 1)
    {
      ix_literal[WARP_SIZE] = this_ix_literal + literal_length;
    }

    warp_scan.ExclusiveSum(total_seq_bytes, scan_result);
    this_ix_output += scan_result;

#ifdef SNAPPY_TRANSCODE_LOGGING
    printf(
      "tid %d ix seq %d lit len %d offset %d raw offset %d mat len %d num seq %d this ix output %d total seq bytes %d "
      "num match tag bytes %d literal tag bytes %d\n",
      threadIdx.x,
      ix_seq,
      literal_length,
      offset,
      raw_offset,
      match_length,
      num_sequences,
      this_ix_output,
      total_seq_bytes,
      num_match_tag_bytes,
      num_literal_tag_bytes
    );
#endif

    __syncwarp(WARP_ALL);

    // Write the tags first, then we'll wait on literals.
    // Write literal tag first
    if (literal_length > 0)
    {
      write_literal_tag(output_buffer, this_ix_output, literal_length, num_literal_tag_bytes - 1);
    }

    // Wait on the appropriate # of literals
    if (is_rle_literals)
    {
      do_literal_copies(rle_literal_byte, ix_output[thread_warp_ix()], literal_length, output_buffer);
    }
    else
    {
      const int req_literals = ix_literal[WARP_SIZE];
      wait_for_atomic(decode_lit_count, req_literals, current_decoded_lit_count, ZSTD_LONG_SLEEP_NS);
      // TODO: switch transcoding to use regs rather than shmem, too
      do_literal_copies(
        literals,
        ix_literal[thread_warp_ix()],
        ix_output[thread_warp_ix()],
        literal_length,
        output_buffer
      );
    }

    __syncwarp();
    ix_literal[thread_warp_ix()] = ix_literal[WARP_SIZE];

    int iter_match_len = match_length;
    while (iter_match_len > 0)
    {
      int this_match_len = min(iter_match_len, 64);
      iter_match_len -= 64;
      write_match_tag(output_buffer, this_ix_output, offset_len, this_match_len, offset);
    }

    __syncwarp();
    ix_output[thread_warp_ix()] = ix_output[WARP_SIZE - 1];
  }

  int num_rem_literals = block_share.num_literals - ix_literal[WARP_SIZE];

  // Copy remaining literals
  wait_for_atomic(decode_lit_count, block_share.num_literals, current_decoded_lit_count, ZSTD_LONG_SLEEP_NS);

  // Write a tag then write the remaining literals.
  if (thread_warp_ix() == 0 and num_rem_literals > 0)
  {
    const int num_literal_tag_bytes = compute_literal_tag_byte_count(num_rem_literals);
    write_literal_tag(output_buffer, ix_output[0], num_rem_literals, num_literal_tag_bytes);
  }

  __syncwarp();
  int final_ix_output = ix_output[0];

  if (is_rle_literals)
  {
    copy_remaining_literals(num_rem_literals, rle_literal_byte, output_buffer, final_ix_output);
  }
  else
  {
    copy_remaining_literals(num_rem_literals, literals, ix_literal[WARP_SIZE], output_buffer, final_ix_output);
  }
  assert(final_ix_output + num_rem_literals <= ZSTD_BLOCK_SIZE_MAX);

  return final_ix_output + num_rem_literals;
}

} // end namespace zstd
