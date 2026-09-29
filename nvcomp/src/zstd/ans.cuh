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

#include <cstdint>

#include "constants.cuh"
#include "EntropyTables.cuh"
#include "io.cuh"
#include "Reduction.cuh"
#include "types.cuh"
#include "utils.cuh"

// #define HUFF_ANS_WEIGHT_LOGGING 1
// #define FSE_LOGGING 1

namespace zstd
{

// Table creation methods
template <typename FSETable_t>
__device__ __noinline__ void init_fse_table(
  FSETable_t &fse_table,
  const uint8_t num_symbols,
  const int16_t *const frequencies,
  ANSTableConstructionBuffers &ans_buffers,
  Uint8WarpScan &uint8_warp_scan,
  Uint16WarpScan &uint16_warp_scan,
  WarpReduceUint16 &uint16_warp_reduce
)
{
  const int table_size = 1 << fse_table.accuracy_log;
  assert(num_symbols <= ANS_MAX_SYMBOLS);
#ifndef NDEBUG
  uint16_t freq_sum = 0;
  for (unsigned ix = thread_warp_ix(); ix < num_symbols; ix += WARP_SIZE_U)
  {
    freq_sum += abs(frequencies[ix]);
  }

  // Reduce
  freq_sum = uint16_warp_reduce.Reduce(freq_sum, cub_sum());
  freq_sum = __shfl_sync(WARP_ALL, freq_sum, 0);

  // This below shouldn't be needed, but this isn't tight loop code
  // Planning to file an nvBug. Without this racecheck triggers despite the
  // above shfl_sync
  __syncwarp(WARP_ALL);

  assert(freq_sum == table_size);
#endif

  uint32_t *state_desc = ans_buffers.state_desc;

  int remaining_states = table_size;
  for (int warp_ix_symbol = 0; warp_ix_symbol < num_symbols; warp_ix_symbol += WARP_SIZE)
  {
    const int ix_symbol = warp_ix_symbol + thread_warp_ix();

    uint8_t active = ((ix_symbol < num_symbols) and frequencies[ix_symbol] == -1);
    uint32_t active_mask = __ballot_sync(WARP_ALL, active);
    const uint32_t my_index = remaining_states - __popc(active_mask & ((1 << (thread_warp_ix() + 1)) - 1));
    if (active)
    {
      fse_table.set_symbol(my_index, ix_symbol);
      state_desc[ix_symbol] = 1;
    }

    remaining_states -= __popc(active_mask);
  }

  const uint16_t step = (table_size >> 1) + (table_size >> 3) + 3;
  const uint16_t mask = table_size - 1;

  uint16_t *valid_spots = ans_buffers.valid_spots;

  uint16_t num_valid_spots = 0;
  uint16_t ix_step = 0;
  while (num_valid_spots < remaining_states)
  {
    uint16_t pos = (ix_step + thread_warp_ix()) * step;
    uint16_t this_pos = pos & mask;
    uint16_t sum_result;
    uint16_t active = this_pos < remaining_states;
    uint16_warp_scan.ExclusiveSum(active, sum_result);
    uint16_t this_ix = num_valid_spots + sum_result;
    if (active and this_ix < remaining_states)
    {
      valid_spots[num_valid_spots + sum_result] = this_pos;
    }

    uint16_t total_active = __shfl_sync(WARP_ALL, sum_result + active, 31);
    num_valid_spots += total_active;
    ix_step += WARP_SIZE_U;
  }

  __syncwarp(); // Needed to ensure valid_spots is finished writing before continuing

  uint16_t ix_valid_spot = 0;
  // Use a warp-level ix_symbol so that every thread can participate in the CUB exclusive sum

  for (int warp_ix_symbol = 0; warp_ix_symbol < num_symbols; warp_ix_symbol += WARP_SIZE)
  {
    const int ix_symbol = warp_ix_symbol + thread_warp_ix();
    uint16_t active_frequencies = ((ix_symbol < num_symbols) and frequencies[ix_symbol] > 0);
    if (active_frequencies)
    {
      active_frequencies = frequencies[ix_symbol];
    }

    // Get the total number of spots.
    uint16_t exc_sum_res = 0;
    uint16_warp_scan.ExclusiveSum(active_frequencies, exc_sum_res);

    ix_valid_spot += exc_sum_res;

    if (active_frequencies)
    {
      state_desc[ix_symbol] = active_frequencies;
      uint16_t stop = ix_valid_spot + active_frequencies;
      for (; ix_valid_spot < stop; ++ix_valid_spot)
      {
        assert(ix_valid_spot < num_valid_spots);
        fse_table.set_symbol(valid_spots[ix_valid_spot], ix_symbol);
      }
    }
    ix_valid_spot = __shfl_sync(WARP_ALL, ix_valid_spot, WARP_SIZE - 1);
  }

  __syncwarp(); // ensure state_desc is fully written

  // Fill baseline and num bits.
  for (int ix_state_base = 0; ix_state_base < table_size; ix_state_base += WARP_SIZE)
  {
    int ix_state = ix_state_base + thread_warp_ix();
    const int num_active = min(table_size - ix_state_base, WARP_SIZE);
    const unsigned active_mask = WARP_ALL >> (WARP_SIZE - num_active);
    if (thread_warp_ix() < num_active)
    {
      uint8_t symbol = fse_table.get_symbol(ix_state);
      unsigned matches = __match_any_sync(active_mask, symbol);
      unsigned total_matches = __popc(matches);
      unsigned prev_matches = __popc(matches & ((1 << thread_warp_ix()) - 1));

      uint16_t next_state_desc = state_desc[symbol] + prev_matches;
      __syncwarp(active_mask);
      if (total_matches == prev_matches + 1)
      {
        state_desc[symbol] += total_matches;
      }

      fse_table.set_n_bits(ix_state, fse_table.accuracy_log - highest_set_bit(next_state_desc));
      // Note:
      // Moving `__syncwarp(active_mask)` to between `set_n_bits` and `get_n_bits` to avoid spurious races,
      // due to reading entire 4 bytes from the table in these functions where some of the bits could have
      // been written by neighboring threads in the warp.
      __syncwarp(active_mask);
      fse_table.set_baseline(ix_state, (next_state_desc << fse_table.get_n_bits(ix_state)) - table_size);
    }
  }
  __syncwarp(); // ensure the fse table is fully written
}

inline __device__ void init_compressed_fse_table(
  FSETable &fse_table,
  const uint8_t num_symbols,
  const int16_t *const frequencies,
  const uint8_t seq_type,
  ANSTableConstructionBuffers &ans_buffers,
  Uint8WarpScan &uint8_warp_scan,
  Uint16WarpScan &uint16_warp_scan,
  WarpReduceUint16 &uint16_warp_reduce
)
{
  fse_table.zero_table(seq_type);
  fse_table.n_bit_wrapper.set_symbol_bits(SEQ_MAX_BITS[seq_type][0]);
  fse_table.n_bit_wrapper.set_baseline_bits(SEQ_MAX_BITS[seq_type][1]);
  fse_table.n_bit_wrapper.set_nbits_bits(SEQ_MAX_BITS[seq_type][2]);
  fse_table.n_bit_wrapper.set_cell_bits(
    SEQ_MAX_BITS[seq_type][0] + SEQ_MAX_BITS[seq_type][1] + SEQ_MAX_BITS[seq_type][2]
  );
  __syncwarp(); // Avoid race between table config (above) and table fill (below)

  init_fse_table(fse_table, num_symbols, frequencies, ans_buffers, uint8_warp_scan, uint16_warp_scan, uint16_warp_reduce);
}

inline __device__ int
do_decode_fse_header(BitReader &bit_reader, int16_t *frequencies, const int max_accuracy_log, int &num_symbols)
{
  int accuracy_log = 5 + bit_reader.read_bits(4);
  assert(accuracy_log <= max_accuracy_log);

  int remaining = 1 << accuracy_log;
  int ix_symbol = 0;
  while (remaining > 0 && ix_symbol < ANS_MAX_SYMBOLS)
  {
    int num_bits = highest_set_bit(remaining + 1) + 1;
    uint16_t val = bit_reader.read_bits(num_bits);

    const uint16_t lower_mask = ((uint16_t)1 << (num_bits - 1)) - 1;
    const uint16_t threshold = ((uint16_t)1 << num_bits) - 1 - (remaining + 1);

    if ((val & lower_mask) < threshold)
    {
      bit_reader.rewind_bits(1);
      val = val & lower_mask;
      --num_bits;
    }
    else if (val > lower_mask)
    {
      val = val - threshold;
    }

    int16_t probability = (int16_t)val - 1;

#ifdef FSE_LOGGING
    print0(
      "acc log %d num symb %d ix symbol %d prob %d num bits %d remaining %d threshold %u\n",
      accuracy_log,
      num_symbols,
      ix_symbol,
      probability,
      num_bits,
      remaining,
      threshold
    );
#endif

    remaining -= probability < 0 ? -probability : probability;
    frequencies[ix_symbol] = probability;
    ++ix_symbol;

    if (probability == 0)
    {
      uint8_t repeat = bit_reader.read_bits(2);
      uint8_t total_zeros = repeat;

      // TODO: could make this more efficient by finding the next clear bit (all bits will be set until the last repeat 2-bit value)
      while (repeat == 3)
      {
        repeat = bit_reader.read_bits(2);
        total_zeros += repeat;
      }

      // Could potentially add a *ton* of zeros
      for (int zero_symbol_ix = thread_warp_ix(); zero_symbol_ix < total_zeros; zero_symbol_ix += WARP_SIZE)
      {
        const int thread_symbol_ix = ix_symbol + zero_symbol_ix;
        frequencies[thread_symbol_ix] = 0;
      }

      ix_symbol += total_zeros;
      assert(ix_symbol < ANS_MAX_SYMBOLS); // corruption
    }
  }

  assert(remaining == 0);
  num_symbols = ix_symbol;

  bit_reader.align_stream();

  __syncwarp(); // Make sure frequencies is written before continuing

  return accuracy_log;
}

inline __device__ void decode_fse_header(
  FSETable &fse_table,
  BitReader &bit_reader,
  int16_t *frequencies,
  const int max_accuracy_log,
  const int max_symbols,
  uint8_t seq_type,
  Uint8WarpScan &uint8_warp_scan,
  Uint16WarpScan &uint16_warp_scan,
  WarpReduceUint16 &uint16_warp_reduce,
  ANSTableConstructionBuffers &ans_buffers,
  const bool build_table
)
{
  int num_symbols;
  int accuracy_log = do_decode_fse_header(bit_reader, frequencies, max_accuracy_log, num_symbols);

  if (not build_table)
  {
    return;
  }

  // Only set the table log if we're building a new table
  fse_table.accuracy_log = accuracy_log;

  __syncwarp();

  init_compressed_fse_table(
    fse_table,
    num_symbols,
    frequencies,
    seq_type,
    ans_buffers,
    uint8_warp_scan,
    uint16_warp_scan,
    uint16_warp_reduce
  );
}

inline __device__ void decode_fse_header(
  RawFSETable &fse_table,
  BitReader &bit_reader,
  int16_t *frequencies,
  const int max_accuracy_log,
  const int max_symbols,
  uint8_t seq_type,
  Uint8WarpScan &uint8_warp_scan,
  Uint16WarpScan &uint16_warp_scan,
  WarpReduceUint16 &uint16_warp_reduce,
  ANSTableConstructionBuffers &ans_buffers,
  const bool build_table
)
{
  int num_symbols;
  int accuracy_log = do_decode_fse_header(bit_reader, frequencies, max_accuracy_log, num_symbols);

  if (not build_table)
  {
    return;
  }

  // Only set the table log if we're building a new table
  fse_table.accuracy_log = accuracy_log;

  __syncwarp();
  init_fse_table(fse_table, num_symbols, frequencies, ans_buffers, uint8_warp_scan, uint16_warp_scan, uint16_warp_reduce);
}

inline __device__ size_t fse_decompress_interleaved2(const RawFSETable &fse_table, uint8_t *out, BitReader &bit_reader)
{
  const unsigned buffer_size = bit_reader.rem_bytes();
  assert(buffer_size > 0); // corruption

  const uint8_t *src = bit_reader.get_read_pointer(buffer_size);

  const int padding = 8 - highest_set_bit(src[buffer_size - 1]);
  int32_t offset = buffer_size * 8 - padding;

  if (thread_warp_ix() >= 2)
  {
    return 0;
  }

  const uint32_t active_mask = 0x3;

  // For each, decode the state
  uint16_t prev_state =
    read_stream_bits_shared_shuffles<2>(src, fse_table.accuracy_log, offset, thread_warp_ix(), active_mask, buffer_size);

#ifdef HUFF_ANS_WEIGHT_LOGGING
  printf(
    "thread %d init state %u num bits %d read from src %p byte %u %u\n",
    thread_warp_ix(),
    prev_state,
    fse_table.accuracy_log,
    src + offset / 8,
    src[offset / 8],
    src[offset / 8 + 1]
  );
#endif

  uint16_t symbols_written = 0;
  while (true)
  {
    uint8_t symbol = fse_table.symbols[prev_state];

    int32_t second_offset = __shfl_sync(active_mask, offset, 1);

    // Write out one or both from previous iteration
    if (second_offset < 0) // Overflow
    {
      if (offset >= 0) // First thread didn't overflow
      {
#ifdef HUFF_ANS_WEIGHT_LOGGING
        printf(
          "thread %d symbol ix %d state %u symbol %u bit offset %d n bits %u\n",
          thread_warp_ix(),
          symbols_written + thread_warp_ix(),
          prev_state,
          symbol,
          offset,
          fse_table.get_n_bits(prev_state)
        );
#endif

        out[symbols_written++] = symbol; // Write one
      }
      break;
    }
    else
    {
      offset = second_offset;

#ifdef HUFF_ANS_WEIGHT_LOGGING
      printf(
        "thread %d symbol ix %d state %u symbol %u bit offset %d n bits %u\n",
        thread_warp_ix(),
        symbols_written + thread_warp_ix(),
        prev_state,
        symbol,
        offset,
        fse_table.get_n_bits(prev_state)
      );
#endif

      // Write both
      out[thread_warp_ix() + symbols_written] = symbol;

      symbols_written += 2;
    }

    // Update to the next state
    const uint16_t rest = read_stream_bits_shared_shuffles<2>(
      src,
      fse_table.next_bits[prev_state],
      offset,
      thread_warp_ix(),
      active_mask,
      buffer_size
    );

    prev_state = fse_table.baselines[prev_state] + rest;

#ifdef HUFF_ANS_WEIGHT_LOGGING
    printf("thread %d bits %d rest %u prev state %u\n", thread_warp_ix(), bits, rest, prev_state);
#endif
  }

  return symbols_written;
}

inline __device__ void decode_seq_table(
  BitReader &bit_reader,
  ANSTableConstructionBuffers &ans_buffers,
  uint8_t sequence_mode,
  const SequenceType seq_type,
  FSETable &fse_table,
  const FSETable *prev_fse_table,
  Uint8WarpScan &uint8_warp_scan,
  Uint16WarpScan &uint16_warp_scan,
  WarpReduceUint16 &uint16_warp_reduce,
  bool build_table
)
{
  int seq_val = static_cast<int>(seq_type);

  switch (sequence_mode)
  {
    case 0: {

      if (not build_table)
      {
        return;
      }
      const int16_t *frequencies = default_distributions[seq_val];
      fse_table.accuracy_log = default_accuracies[seq_val];
      __syncwarp(
        WARP_ALL
      ); // accuracy log write. Not a true race because all values are set the same, but synchronize to avoid FAs in racecheck

      init_compressed_fse_table(
        fse_table,
        SEQ_MAX_SYMBOLS[seq_val],
        frequencies,
        seq_val,
        ans_buffers,
        uint8_warp_scan,
        uint16_warp_scan,
        uint16_warp_reduce
      );
      break;
    }
    case 1: {
      // RLE
      const uint8_t symbol = bit_reader.read_bits(8);

      if (not build_table)
      {
        return;
      }

      fse_table.n_bit_wrapper.set_symbol_bits(SEQ_MAX_BITS[seq_val][0]);
      fse_table.n_bit_wrapper.set_nbits_bits(SEQ_MAX_BITS[seq_val][1]);
      fse_table.n_bit_wrapper.set_baseline_bits(SEQ_MAX_BITS[seq_val][2]);
      fse_table.n_bit_wrapper.set_cell_bits(
        SEQ_MAX_BITS[seq_val][0] + SEQ_MAX_BITS[seq_val][1] + SEQ_MAX_BITS[seq_val][2]
      );

      __syncwarp(WARP_ALL); // Make sure we've set the above before continuing

      fse_table.accuracy_log = 0;
      fse_table.zero_table(seq_val);
      fse_table.set_symbol(0, symbol);
      fse_table.set_n_bits(0, 0);
      fse_table.set_baseline(0, 0);

      break;
    }
    case 2: {
      int16_t *frequencies = ans_buffers.frequencies;
      decode_fse_header(
        fse_table,
        bit_reader,
        frequencies,
        max_accuracies[seq_val],
        SEQ_MAX_SYMBOLS[seq_val],
        seq_val,
        uint8_warp_scan,
        uint16_warp_scan,
        uint16_warp_reduce,
        ans_buffers,
        build_table
      );

      break;
    }
    case 3: {

      if (not build_table)
      {
        return;
      }
      assert(prev_fse_table); // ensure prev_fse_table is not null

      // Do the copy
      fse_table.deepcopy(*prev_fse_table, seq_val);
      break;
    }
  }
}

inline __device__ int decode_num_sequences(BitReader &bit_reader)
{
  int num_sequences;
  uint8_t header = bit_reader.read_bits(8);
  uint8_t num_bytes = 1;
  constexpr int MAX_TWO_BYTE_SEQUENCE_COUNT = 0x7f00;

  if (header < 128)
  {
    num_sequences = header;
  }
  else if (header < 255)
  {
    num_bytes += 1;
    num_sequences = static_cast<int32_t>((uint32_t{header - 128u} << 8u) + bit_reader.read_bits(8));
  }
  else
  { // num sequences >= MAX_TWO_BYTE_SEQUENCE_COUNT
    num_bytes += 2;
    num_sequences = static_cast<int32_t>(bit_reader.read_bits(16) + MAX_TWO_BYTE_SEQUENCE_COUNT);
  }
  return num_sequences;
}

inline __device__ void decode_sequence_tables(
  const uint8_t *input,
  const int comp_block_size,
  ANSTableConstructionBuffers &ans_buffers,
  BlockHeader &block_header,
  const FSETables *prev_fse_tables,
  FSETables &new_fse_tables,
  Uint8WarpScan &uint8_warp_scan,
  Uint16WarpScan &uint16_warp_scan,
  WarpReduceUint16 &uint16_warp_reduce
)
{
  const int offset = block_header.seq_section_start;
  BitReader bit_reader{input + offset, comp_block_size - offset};
  uint8_t compression_modes = bit_reader.read_bits(8);
  uint8_t literal_length_mode = (compression_modes >> 6) & 3;
  uint8_t offset_mode = (compression_modes >> 4) & 3;
  uint8_t match_length_mode = (compression_modes >> 2) & 3;

  // Decode the three tables
  if (prev_fse_tables)
  {
    decode_seq_table(
      bit_reader,
      ans_buffers,
      literal_length_mode,
      SequenceType::LiteralLength,
      new_fse_tables.ll_fse_table,
      &prev_fse_tables->ll_fse_table,
      uint8_warp_scan,
      uint16_warp_scan,
      uint16_warp_reduce,
      true /*build_table*/
    );
    decode_seq_table(
      bit_reader,
      ans_buffers,
      offset_mode,
      SequenceType::Offset,
      new_fse_tables.of_fse_table,
      &prev_fse_tables->of_fse_table,
      uint8_warp_scan,
      uint16_warp_scan,
      uint16_warp_reduce,
      true /*build_table*/
    );
    decode_seq_table(
      bit_reader,
      ans_buffers,
      match_length_mode,
      SequenceType::MatchLength,
      new_fse_tables.ml_fse_table,
      &prev_fse_tables->ml_fse_table,
      uint8_warp_scan,
      uint16_warp_scan,
      uint16_warp_reduce,
      true /*build_table*/
    );
  }
  else
  {
    decode_seq_table(
      bit_reader,
      ans_buffers,
      literal_length_mode,
      SequenceType::LiteralLength,
      new_fse_tables.ll_fse_table,
      nullptr,
      uint8_warp_scan,
      uint16_warp_scan,
      uint16_warp_reduce,
      true /*build_table*/
    );
    decode_seq_table(
      bit_reader,
      ans_buffers,
      offset_mode,
      SequenceType::Offset,
      new_fse_tables.of_fse_table,
      nullptr,
      uint8_warp_scan,
      uint16_warp_scan,
      uint16_warp_reduce,
      true /*build_table*/
    );
    decode_seq_table(
      bit_reader,
      ans_buffers,
      match_length_mode,
      SequenceType::MatchLength,
      new_fse_tables.ml_fse_table,
      nullptr,
      uint8_warp_scan,
      uint16_warp_scan,
      uint16_warp_reduce,
      true /*build_table*/
    );
  }

  const unsigned final_rem_bytes = bit_reader.rem_bytes();
  block_header.block_seq_table_size = comp_block_size - offset - final_rem_bytes;

  __syncwarp(WARP_ALL); // Ensure threads don't continue before tables filled in
}

// TODO: merge all of this into ANSSequenceDecoder
inline __device__ void do_decode_sequences_for_frame_sizes(
  int num_sequences,
  const uint8_t *next_bits,
  const uint32_t *baseline_32,
  int input_ix_table,
  uint8_t ix_in_zstd_block,
  int ix_base_thread,
  int ix_thread_seq,
  const FSETable *fse_table,
  int32_t &num_match_copy_bytes,
  const int buffer_size,
  DeviceBlockShare &block_share,
  const unsigned block_mask,
  ANSCache &ans_cache,
  unsigned *shared_words,
  uint8_t *shared_reg
)
{
  // Now, we need to use these 6 threads to do their thing.
  // First three threads do decoding of the most recent sequence
  // Next three threads load the next state.
  // Order:
  // 0: OF Decode
  // 1: ML Decode
  // 2: LL decode
  // 3: LL State
  // 4: ML State
  // 5: OF State

  int ix_table = input_ix_table;

  num_match_copy_bytes = 0;

  const int sourceLane = ix_base_thread + 5 - ix_in_zstd_block;
  unsigned this_bits;

  // This is the same message as in ANSSequenceDecoder.cuh:
  // Set default value for Coverity, which doesn't like that nothing is set in threads with ix_in_zstd_block < 3.
  // Threads with ix_in_zstd_block < 3 get the value of this_symbol with __shfl_sync from matching thread with ix_in_zstd_block >= 3
  // At the same time they passed their this_symbol (which is now set to 0) to threads with ix_in_zstd_block >= 3, which ignore the returned value.
  unsigned this_symbol = 0;
  for (int ix_sequence = 0; ix_sequence < num_sequences; ++ix_sequence)
  {
    uint32_t new_baseline;

    if (ix_in_zstd_block >= 3)
    {
      uint8_t packed_symbol;
      uint8_t packed_bits;
      fse_table->get_all(ix_table, packed_symbol, new_baseline, packed_bits);
      this_symbol = static_cast<unsigned>(packed_symbol);
      this_bits = static_cast<unsigned>(packed_bits);
    }

    __syncwarp(block_mask); // necessary to prevent the below code from advancing before it's ready

    unsigned shfl_symbol = __shfl_sync(block_mask, this_symbol, sourceLane);
    if (ix_in_zstd_block < 3)
    {
      ix_table = shfl_symbol;
      this_bits = next_bits[ix_table];
      new_baseline = baseline_32[ix_table];
    }

    uint32_t new_val = 0;
    if (ans_cache.total_bits_remaining > 0) [[likely]]
    {
      // Note:
      // this_bits is set for all 6 threads, and hence the `shared_reg` is set correctly
      *shared_reg = this_bits;
      __syncwarp(block_mask); // shared_reg write needs to be visible in the cache read
      unsigned reg1 = shared_words[0];
      unsigned reg2 = shared_words[1];
      new_val = ans_cache.read(reg1, reg2, this_bits, ix_in_zstd_block);
    }

    ix_table = new_baseline + new_val;
    __syncwarp(block_mask);

    if (ix_in_zstd_block == 1)
    {
      num_match_copy_bytes += ix_table;
    }
  }

  assert((ans_cache.total_bits_remaining - static_cast<int>(ans_cache.num_bits_of_front_padding)) <= 0);

  // Need to synchronize before return so that all threads have
  // written the appropriate values before informing the outside world that the block is done
  __syncwarp(block_mask);
}

inline __device__ void decode_sequences_for_frame_sizes(
  BitReader &bit_reader,
  int num_sequences,
  FSETables &fse_tables,
  const BlockHeader &block_header,
  DeviceBlockShare &block_share,
  int ix_in_zstd_block,
  int ix_zstd_block,
  int32_t &num_match_copy_bytes,
  ANSCache &ans_cache,
  unsigned *shared_words,
  uint8_t *shared_reg
)
{
  int ix_base_thread = ix_zstd_block * NUM_THREADS_PER_ANS_BLOCK;
  const unsigned block_mask = ((1 << NUM_THREADS_PER_ANS_BLOCK) - 1) << ix_base_thread;

  // Skip past table spec
  bit_reader.get_read_pointer(block_header.block_seq_table_size);

  // Now decode the sequences. Do this using six threads.
  assert(ix_in_zstd_block < NUM_THREADS_PER_ANS_BLOCK);

  uint8_t ix_seq = ix_in_zstd_block % 3;
  const FSETable *fse_table = &fse_tables.get_table(ix_seq);

  const int buffer_size = bit_reader.rem_bytes();
  const uint8_t *const src = bit_reader.get_read_pointer(buffer_size);
  int padding = 8 - highest_set_bit(src[buffer_size - 1]);

  ans_cache.init_cache(src, buffer_size, ix_in_zstd_block);

  uint16_t prev_state;

  const uint8_t *next_bits;
  const uint32_t *baseline_32;

  int num_bits = ix_in_zstd_block < 3 ? fse_table->accuracy_log : 0;
  *shared_reg = static_cast<uint8_t>(num_bits);
  __syncwarp(block_mask); // shared reg write must be visible
  unsigned reg1 = shared_words[0];
  unsigned reg2 = shared_words[1];
  uint16_t read_val = ans_cache.read(reg1, reg2, num_bits, ix_in_zstd_block);

  // Strangely, the zstd format flips the ordering of LL / ML / Offset again and again...
  if (ix_in_zstd_block == 0 or ix_in_zstd_block == 5)
  {
    ix_seq = 1; // also serves as shuffle lane
  }
  else if (ix_in_zstd_block == 1 or ix_in_zstd_block == 4)
  {
    ix_seq = 2;
  }
  else
  { // ix_in_zstd_block == 2 or == 3
    ix_seq = 0;
  }

  prev_state = __shfl_sync(block_mask, read_val, ix_seq + ix_base_thread);
  fse_table = &fse_tables.get_table(ix_seq);

  if (ix_in_zstd_block < 3)
  {
    next_bits = seq_extra_bits[ix_seq];
    baseline_32 = seq_baselines[ix_seq];
  }
  else
  {
    next_bits = NULL;
  }

  do_decode_sequences_for_frame_sizes(
    num_sequences,
    next_bits,
    baseline_32,
    prev_state,
    ix_in_zstd_block,
    ix_base_thread,
    ix_seq,
    fse_table,
    num_match_copy_bytes,
    buffer_size,
    block_share,
    block_mask,
    ans_cache,
    shared_words,
    shared_reg
  );
}

} // namespace zstd
