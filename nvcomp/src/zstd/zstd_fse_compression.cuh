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

#include <cuda_runtime.h>

#include <cstdint>

#include "ans/ans_tools.cuh"
#include "bit_writer.cuh"
#include "EntropyTables.cuh"
#include "io.cuh"
#include "types.cuh"
#include "utils.cuh"

// #define FSE_TABLE_LOGGING 1

namespace zstd
{

inline __device__ unsigned determine_min_tablelog(const unsigned src_size, unsigned max_symbol_value)
{
  unsigned min_bits_src = highest_set_bit(src_size) + 1;
  unsigned min_bits_symbols = highest_set_bit(max_symbol_value) + 2;
  unsigned min_bits = min(min_bits_src, min_bits_symbols);
  return min_bits;
}

// This emulates the table log selection used by zstd
inline __device__ int
determine_tableLog(const unsigned max_table_log, const unsigned src_size, const unsigned max_symbol_value)
{
  assert(src_size > 1); /* Not supported, RLE should be used instead */

  // The below code, which this is replicating, comes from ZSTD CPU hash:
  // e9d6fc867ab00b13e4f20dca94812464f9adbe39

  // U32 maxBitsSrc = ZSTD_highbit32((U32)(srcSize - 1)) - minus;
  // U32 tableLog = maxTableLog;
  // U32 minBits = FSE_minTableLog(srcSize, maxSymbolValue);
  // assert(srcSize > 1); /* Not supported, RLE should be used instead */
  // if (tableLog==0) tableLog = FSE_DEFAULT_TABLELOG;
  // if (maxBitsSrc < tableLog) tableLog = maxBitsSrc;   /* Accuracy can be reduced */
  // if (minBits > tableLog) tableLog = minBits;   /* Need a minimum to safely represent all symbol values */
  // if (tableLog < FSE_MIN_TABLELOG) tableLog = FSE_MIN_TABLELOG;
  // if (tableLog > FSE_MAX_TABLELOG) tableLog = FSE_MAX_TABLELOG;
  // return tableLog;

  const int MAGIC_MINUS_VALUE = 2;

  const unsigned max_bits_src = highest_set_bit(src_size - 1) - MAGIC_MINUS_VALUE;
  unsigned table_log = max_table_log;
  unsigned min_bits = determine_min_tablelog(src_size, max_symbol_value);

  if (max_bits_src < table_log)
  {
    table_log = max_bits_src;
  }
  if (min_bits > table_log)
  {
    table_log = min_bits;
  }
  table_log = max(table_log, ZSTD_FSE_MIN_TABLELOG);
  table_log = min(table_log, max_table_log);
  // print0("table log %u min bits %u max bits %u src size %u man compute %u\n", table_log, min_bits, max_bits_src, src_size, highest_set_bit(src_size));
  return table_log;
}

inline __device__ void tANS_construct_encoding_table(
  SymbolEncoder &symbol_encoder,
  const int16_t *norm_counts,
  uint16_t *starts,
  uint8_t *dst_table,
  uint8_t max_symbol_value,
  uint8_t tablelog,
  Uint16WarpScan &uint16_warp_scan
)
{
  if (threadIdx.x == 0)
  {
    symbol_encoder.tablelog = tablelog;
  }
  __syncwarp();
  uint16_t table_size = 1 << tablelog;

  /* Place low-probability symbols. Low-probability symbols have probabilities
     * less than 1/table_size. These symbols are placed at the end of the decoding
     * table in an effort to skew their effective probabilities downwards, as the
     * actual number of states designated for these symbols in the table cannot 
     * be less than 1.
     * 
     * At the same time, determine symbol start offsts
     */
  uint32_t lp_thresh = table_size - 1;
  uint16_t acc = 0;
  for (uint16_t base_symbol = 0; base_symbol <= max_symbol_value; base_symbol += WARP_SIZE_U)
  {
    uint16_t symbol = base_symbol + threadIdx.x;
    bool active = symbol <= max_symbol_value;
    uint16_t abs_norm = 0;
    if (symbol <= max_symbol_value)
    {
      abs_norm = abs(norm_counts[symbol]);
    }

    uint16_t scan_res;
    uint16_warp_scan.ExclusiveSum(abs_norm, scan_res);
    uint32_t symbol_start = acc + scan_res;
    if (symbol <= max_symbol_value)
    {
      starts[symbol] = symbol_start;
    }

    acc = __shfl_sync(WARP_ALL, symbol_start + abs_norm, WARP_SIZE - 1);

    bool negative = false;
    if (active)
    {
      negative = norm_counts[symbol] == -1;
    }
    uint32_t negatives = __ballot_sync(WARP_ALL, negative);
    uint32_t earlier_negatives = negatives & ((1 << threadIdx.x) - 1);
    uint8_t idx = __popc(earlier_negatives);

    if (negative)
    {
      dst_table[lp_thresh - idx] = symbol;
      // printf("filling idx %u with symbol %u\n", lp_thresh - idx, dst_table[lp_thresh - idx]);
    }

    lp_thresh -= __popc(negatives);
  }

  // place regular probability symbols

  /* Places regular probability symbols in the table. By choosing a step size
     * equal to (5/8) * table_size + 3, we are guaranteed to hit each cell of the
     * table once before ever seeing the same cell again due to the fact that
     * gcd(5, 8) = 1 and gcd(3, 8) = 1. In practice, this technique distributes 
     * symbols with good uniformity.
     */
  uint32_t step = (table_size >> 1) + (table_size >> 3) + 3;
  uint32_t position = (step * threadIdx.x) & (table_size - 1);
  for (uint16_t symbol = 0; symbol <= max_symbol_value; ++symbol)
  {
    int16_t nc = norm_counts[symbol];
    for (int warp_base_pos = 0; warp_base_pos < nc; warp_base_pos += WARP_SIZE)
    {
      int nOcc = warp_base_pos + threadIdx.x;
      bool thread_active = nOcc < nc;
      uint32_t active = __ballot_sync(WARP_ALL, thread_active);

      uint32_t matches = active & __ballot_sync(WARP_ALL, position > lp_thresh);
      int8_t first_match_tid = __ffs(matches) - 1;

      /* In case there is a subset of threads whose distribution table 
             * positions end up in the region designated for low-probability
             * symbols, step each of those threads until none of them end up
             * in the low-probability region.
             */
      while (matches)
      {
        active = (active >> first_match_tid) << first_match_tid;
        if ((active >> threadIdx.x) & 1)
        {
          position = (position + step) & (table_size - 1);
        }
        matches = __ballot_sync(WARP_ALL, ((active >> threadIdx.x) & 1) && (position > lp_thresh));
        first_match_tid = __ffs(matches) - 1;
      }

      if (thread_active)
      {
        dst_table[position] = symbol;
      }
      position = __shfl_sync(WARP_ALL, position, WARP_SIZE - __clz(active) - 1);

      // compute a prefix scan to determine each thread's writing offset into the
      // distribution table
      position = (position + (threadIdx.x + 1) * step) & (table_size - 1);
    }
  }
  __syncwarp(); // Finish the above before using the table

  // for (int ix = threadIdx.x; ix < table_size; ix += 32) {
  //     printf("table[%d]=%u\n", ix, dst_table[ix]);
  // }

  // build encoding table -- assumed power of 2 so we don't need to worry about partial warp operations
  assert(table_size % WARP_SIZE_U == 0);
  for (uint16_t i = threadIdx.x; i < table_size; i += WARP_SIZE_U)
  {
    uint8_t symbol = dst_table[i];
    uint32_t matches = __match_any_sync(WARP_ALL, symbol);
    uint32_t earlier_matches = matches & ((1 << threadIdx.x) - 1);
    uint8_t idx = __popc(earlier_matches);

    uint16_t pos = starts[symbol] + idx;
    __syncwarp(); // ensure starts is read before continuing

    if (threadIdx.x == __ffs(matches) - 1)
    {
      starts[symbol] += __popc(matches);
    }

    // printf("ix %u symbol %u thread %d pos %u table size %u norm count %d\n",
    //     i, symbol, threadIdx.x, pos, table_size, norm_counts[symbol]);
    symbol_encoder.encoding_table[pos] = table_size + i;
    __syncwarp(); // ensure starts is written before proceeding to the next iteration
  }

  /* create symbol translation table containing each symbol's offset into encoding table
     * and number of bits to be read out to the stream
     */
  for (uint16_t i = threadIdx.x; i <= max_symbol_value; i += WARP_SIZE_U)
  {
    uint32_t max_bits;
    uint32_t min_state_max_bits;
    switch (norm_counts[i])
    {
      case 0:
        // fill for compatibility while encoding first round of symbols
        symbol_encoder.stt[i].n_bits = 0;
        symbol_encoder.stt[i].state_offset = 0;
        break;
      case -1:
      // fall through
      case 1:
        /* every state requires reading out the max number of bits, which is
                * tablelog + 1
                */
        symbol_encoder.stt[i].n_bits = (tablelog << (ANS_ALGORITHM_MAX_TABLELOG + 1)) - (1 << tablelog);
        symbol_encoder.stt[i].state_offset = starts[i] - 2;
        break;
      default:
        /* The maximum number of bits that would need to be read out to the stream is calculated
                * as ceil(log_2(max_state / max_L_s)). The below expression is ceil(log_2(min_state / min_L_s)),
                * which is equivalent to the above for integer-valued max_state and max_L_s.
                **/
        max_bits = tablelog - (31 - __clz(norm_counts[i] - 1));

        /* This is the minimum state that requires reading in the max number of bits. */
        min_state_max_bits = norm_counts[i] << max_bits;

        /* By shifting and subtracting min_state_max_bits, we make it such that right shifting
                * by ANS_ALGORITHM_MAX_TABLELOG + 1 will recover max_num_bits - 1. If we add a state greater than or
                * equal to min_state_max_bits before right shifting, we recover max_num_bits, as 
                * desired.
                * */
        symbol_encoder.stt[i].n_bits = (max_bits << (ANS_ALGORITHM_MAX_TABLELOG + 1)) - min_state_max_bits;

        /* subtract norm_counts[i] here to compensate for adding a total of norm_counts[i]
                 * while keeping track of the symbol offset during encoding table construction.
                 * Subtract another norm_counts[i] because it would be subtracted anyway every time
                 * we encode a symbol.
                 */
        symbol_encoder.stt[i].state_offset = starts[i] - 2 * norm_counts[i];
        // printf("symbol %u start %u norm count %d state offset %d max bits %u n bits %u\n",
        //         i, starts[i], norm_counts[i], stt[i].state_offset, max_bits, stt[i].n_bits);
    }
  }
  __syncwarp();
}

inline __device__ void
output_fse_header(BitWriter &bit_writer, int accuracy_log, const int16_t *norm_counts, const int num_counts)
{
  // Write out accuracy log in 4 bits
  if (threadIdx.x == 0)
  {
    bit_writer.write_bits(accuracy_log - 5, 4);
  }
  __syncwarp();

  int remaining = 1 << accuracy_log;

  int ix_symbol = 0;
  while (remaining > 0 and ix_symbol < num_counts)
  {
    int16_t prob = norm_counts[ix_symbol];
    int num_bits = highest_set_bit(remaining + 1) + 1;
    const uint16_t threshold = ((uint16_t)1 << num_bits) - 1 - (remaining + 1);
    uint16_t write_val = prob + 1;
    unsigned lower_mask = ((uint16_t)1 << (num_bits - 1)) - 1;
    if (write_val < threshold)
    {
      write_val = write_val & lower_mask;
      --num_bits;
    }
    else if (write_val >= 1 << (num_bits - 1))
    {
      write_val = write_val + threshold;
    }

    if (threadIdx.x == 0)
    {
      bit_writer.write_bits(write_val, num_bits);
    }
#ifdef FSE_TABLE_LOGGING
    print0(
      "num counts %d ix symbol %d remaining %d write val %d prob %d num bits %d thresh %u\n",
      num_counts,
      ix_symbol,
      remaining,
      write_val,
      prob,
      num_bits,
      threshold
    );
#endif

    remaining -= abs(prob);
    if (norm_counts[ix_symbol] == 0)
    {
      // Find the length of the run
      int num_repeat_zeroes = 0;
      for (int ix_warp_symbol = ix_symbol + 1; ix_warp_symbol < num_counts; ix_warp_symbol += WARP_SIZE)
      {
        int ix_thread_symbol = ix_warp_symbol + threadIdx.x;
        int first_nonzero_symbol =
          __ffs(__ballot_sync(WARP_ALL, ix_thread_symbol >= num_counts or norm_counts[ix_thread_symbol] != 0));
        num_repeat_zeroes += first_nonzero_symbol - 1;
        if (first_nonzero_symbol > 0)
        {
          break;
        }
        else
        {
          num_repeat_zeroes += WARP_SIZE;
        }
      }

      if (threadIdx.x == 0)
      {
        bool finished = false;
        for (int ix = 0; ix < num_repeat_zeroes; ix += 48)
        {
          unsigned write_val = min(48, num_repeat_zeroes - ix);
          // Note: in testing this is always false
          if (write_val == 48)
          {
            bit_writer.write_bits(0xffffffffu, WARP_SIZE);
          }
          else
          {
            int num_triples = write_val / 3;
            unsigned this_write = (1 << (num_triples * 2)) - 1;
            bit_writer.write_bits(this_write, num_triples * 2);
            int rem_zeroes = write_val % 3;
            bit_writer.write_bits(rem_zeroes, 2);
            finished = true;
          }
        }

        if (not finished)
        {
          bit_writer.write_bits(0, 2);
        }
      }
      __syncwarp();
      ix_symbol += num_repeat_zeroes;
    }
    ++ix_symbol;
  }

  if (threadIdx.x == 0)
  {
    bit_writer.align_bytewise(); // Round up to the next whole byte as specified by the format
  }
}

// returns -1 if created a table, otherwise returns the RLE symbol value
template <int cexpr_max_symbol_value>
inline __device__ int init_zstd_fse_encoder(
  int &accuracy_log,
  const uint32_t *frequencies,
  SequenceCompressBuffers &seq_buffers,
  const int symbol_count,
  SymbolEncoder &symbol_encoder,
  const bool use_default,
  const int ix_seq_type
)
{
  const int16_t *norm_frequencies;
  unsigned max_symbol_value = 0;
  if (use_default)
  {
    norm_frequencies = default_distributions[ix_seq_type];
    accuracy_log = default_accuracies[ix_seq_type];
    max_symbol_value = SEQ_MAX_SYMBOLS[ix_seq_type] - 1;
  }
  else
  {
    unsigned num_nonzero_symbols = 0;
    for (int ix = threadIdx.x; ix < cexpr_max_symbol_value; ix += WARP_SIZE)
    {
      if (frequencies[ix] != 0)
      {
        // printf("freq %u ix %u\n", frequencies[ix], ix);
        max_symbol_value = max(max_symbol_value, ix);
        ++num_nonzero_symbols;
      }
    }

    unsigned nonzero_mask = __ballot_sync(WARP_ALL, num_nonzero_symbols > 0);
    if (__popc(nonzero_mask) == 1)
    {

      unsigned ix_nonzero = __ffs(nonzero_mask) - 1;
      unsigned nonzero_count = __shfl_sync(WARP_ALL, num_nonzero_symbols, ix_nonzero);
      // Use the max symbol value of the nonzero thread regardless of whether we'll do rle or not.
      max_symbol_value = __shfl_sync(WARP_ALL, max_symbol_value, ix_nonzero);
      if (nonzero_count == 1)
      {
        return max_symbol_value;
      }
    }
    else
    {
      max_symbol_value = warpReduceMax(max_symbol_value, WARP_ALL);
    }

    accuracy_log = determine_tableLog(accuracy_log, symbol_count, max_symbol_value);

    nvcomp::ans_shared::normalize_frequencies(
      seq_buffers.norm_weight_freqs,
      accuracy_log,
      frequencies,
      symbol_count,
      max_symbol_value
    );
    __syncwarp();

    int sum_freq = 0;
    for (int ix = threadIdx.x; ix <= max_symbol_value; ix += WARP_SIZE)
    {
      sum_freq += abs(seq_buffers.norm_weight_freqs[ix]);
    }

    sum_freq = warpReduceSum(sum_freq, WARP_ALL);
#ifndef NDEBUG
    const int table_size = 1 << accuracy_log;
    // printf("sum freq %d table size %d\n", sum_freq, table_size);
    assert(sum_freq == table_size);
#endif
    norm_frequencies = seq_buffers.norm_weight_freqs;
  }

  __shared__ Uint16WarpScan::TempStorage temp_storage;
  Uint16WarpScan uint16_warp_scan{temp_storage};
  tANS_construct_encoding_table(
    symbol_encoder,
    norm_frequencies,
    seq_buffers.starts,
    seq_buffers.dst_table,
    max_symbol_value,
    accuracy_log,
    uint16_warp_scan
  );

  return -1;
}

// In the initial pass, separate the tokens into symbols and compute the histogram
inline __device__ void perform_initial_sequence_pass(
  const int num_sequences,
  const CompressSequenceBuffer &sequences,
  uint8_t *symbol_buffer,
  uint32_t frequencies[ZSTD_SEQ_ANS_STREAMS][SEQ_ANS_MAX_SYMBOLS],
  const SharedConstantTables &constant_tables
)
{
  uint8_t *symbol_arrays[ZSTD_SEQ_ANS_STREAMS];

#pragma unroll
  for (int ix = 0; ix < ZSTD_SEQ_ANS_STREAMS; ++ix)
  {
    symbol_arrays[ix] = symbol_buffer + num_sequences * ix;
  }

  for (int ix_seq = 0; ix_seq < ZSTD_SEQ_ANS_STREAMS; ++ix_seq)
  {
    for (int ix_freq = threadIdx.x; ix_freq < SEQ_ANS_MAX_SYMBOLS; ix_freq += WARP_SIZE)
    {
      frequencies[ix_seq][ix_freq] = 0;
    }
  }
  __syncwarp();

  int num_active = WARP_SIZE;
  unsigned mask = WARP_ALL;
  for (int base_ix_seq = 0; base_ix_seq < num_sequences; base_ix_seq += WARP_SIZE)
  {

    int ix_seq = base_ix_seq + threadIdx.x;
    if (base_ix_seq + WARP_SIZE >= num_sequences)
    {
      num_active = num_sequences - base_ix_seq;
      mask = WARP_ALL >> (WARP_SIZE - num_active); // num_active is > 0 because base_ix_seq < num_sequences
    }

    if (ix_seq < num_sequences)
    {
#pragma unroll
      for (int ix_seq_val = 0; ix_seq_val < ZSTD_SEQ_ANS_STREAMS; ++ix_seq_val)
      {
        const uint32_t val = sequences(ix_seq, ix_seq_val);
        int ix_symbol;
        // Do a binary search to find the symbol.
        if (ix_seq_val == 0)
        {
          flexible_lower_bound<6>(ix_symbol, val, constant_tables.ll_baselines, SEQ_MAX_SYMBOLS[ix_seq_val]);
        }
        else if (ix_seq_val == 1)
        {
          flexible_lower_bound<5>(ix_symbol, val, constant_tables.of_baselines, SEQ_MAX_SYMBOLS[ix_seq_val]);
        }
        else
        {
          flexible_lower_bound<6>(ix_symbol, val, constant_tables.ml_baselines, SEQ_MAX_SYMBOLS[ix_seq_val]);
        }

        symbol_arrays[ix_seq_val][ix_seq] = ix_symbol;
        warpMatchAdd(ix_symbol, mask, frequencies[ix_seq_val]);
      }
    }
  }
  __syncwarp();
}

/* @brief: This function stages small "shared_write_count" writes
 *         into registers at a warp level, buffering up the writes until we can do a 
 *         32-way prefix sum to determine output locations, and have each thread in the warp
 *         perform write operations 
 * inputs:
 *   write_val: The register write value for a thread. If not filled yet, uninitialized
 *   num_bits: The register number of bits for a thread. If not filled yet, uninitialized
 *   shared_write_count: The number of new writes provided in this call.
 *   bit_writer: Used to actually write the values to the stream when appropriate
 *   staged_write_count: The number of writes currently staged in registers
 *   shared_write_val: Shared memory buffer of write values
 *   shared_num_bits: Shared memory buffer of # of bits to write
 *   force_flush_to_bitwriter: Called for the last write before the various temporaries will go out of scope  
 * 
 * Detail:
 *   This function takes "shared_write_count" writes from 
 *   the shared_write_val / shared_num_bits arrays.
 *   These writes are shuffled into the appropriate registers, such that 
 *   Thread 0 holds the oldest write and thread 31 holds the newest write
 *   The number of registers with a value is stored in staged_write_count.
 *
 *   When all threads are full (staged_write_count >= 32), all 32 threads prefix sum / write to the
 *   bitstream through the bit_writer
 *
 *   The last write is called with "force_flush_to_bitwriter" set true. This fully flushes
 *   all staged writes to the bit writer.
 * 
 * Postconditions:
 *   staged_write_count is < 32. 
 *   Bit writer may have been updated. 
 *   write_val / num_bits values may be updated for each thread
*/
inline __device__ void staged_output_write(
  uint32_t &write_val,
  int &num_bits,
  const int shared_write_count,
  BitWriter &bit_writer,
  int &staged_write_count,
  uint32_t *shared_write_val,
  uint8_t *shared_num_bits,
  const bool force_flush_to_bitwriter
)
{
  // Determine which threads should store the new writes in registers
  const int ix_shared = threadIdx.x - staged_write_count;

  if (ix_shared >= 0 and ix_shared < shared_write_count)
  {
    write_val = shared_write_val[ix_shared];
    num_bits = shared_num_bits[ix_shared];
  }

  staged_write_count += shared_write_count;
  if (force_flush_to_bitwriter or staged_write_count >= WARP_SIZE)
  {
    // Now flush to the output stream through the bit writer
    __shared__ IntWarpScan::TempStorage temp_storage;
    IntWarpScan int_warp_scan{temp_storage};
    int scan_offset;
    int_warp_scan.ExclusiveSum(num_bits, scan_offset);
    bit_writer.write_bits(write_val, num_bits, threadIdx.x, scan_offset, WARP_ALL, WARP_SIZE);

    // Reinitialize
    write_val = 0;
    num_bits = 0;

    // If we had more than 32 writes, there are some leftover writes in the shared memory buffers.
    // Stage these now in registers.
    staged_write_count -= WARP_SIZE;
    __syncwarp();
    if (static_cast<int>(threadIdx.x) < staged_write_count)
    {
      // In this case we have a remainder that resides in the shared buffer.
      const int ix_shared = threadIdx.x + shared_write_count - staged_write_count;

      write_val = shared_write_val[ix_shared];
      num_bits = shared_num_bits[ix_shared];
    }

    // Finally, handle the corner case where we are forcing all writes to the bitwriter and we need multiple
    // "write_bits" calls to achieve this.
    if (force_flush_to_bitwriter and staged_write_count > 0)
    {
      int scan_offset = custom_exclusive_scan<32>(num_bits, WARP_ALL, threadIdx.x);
      bit_writer.write_bits(write_val, num_bits, threadIdx.x, scan_offset, WARP_ALL, WARP_SIZE);
    }
  }
}

inline __device__ void do_compress_sequences(
  SymbolEncoder *encoders,
  BitWriter &bit_writer,
  const uint8_t *symbols,
  const uint32_t *baseline,
  const uint8_t *table_extra_bits,
  const CompressSequenceBuffer &sequences,
  const int num_sequences,
  const int ix_seq_type
)
{
  // Uses 6 threads for compression operations - reverse of the decompression
  // 0: OF state renormalization bits
  // 1: ML state renormalization bits
  // 2: LL state renormalization bits
  // 3: LL extra bits
  // 4: ML extra bits
  // 5: OF extra bits

  // Additionally, uses all 32 threads in the warp to perform writes

  // Producing at least 6 write values at a time
  int seq_load_count = num_sequences % WARP_SIZE;
  seq_load_count = seq_load_count == 0 ? WARP_SIZE : seq_load_count;
  int thread_match_len = -1;
  int thread_lit_len = -1;
  int thread_offset = -1;
  if (threadIdx.x < seq_load_count)
  {
    thread_match_len = sequences.get_match_length(num_sequences - seq_load_count + threadIdx.x);
    thread_lit_len = sequences.get_literal_length(num_sequences - seq_load_count + threadIdx.x);
    thread_offset = sequences.get_offset(num_sequences - seq_load_count + threadIdx.x);
  }
  int next_load = num_sequences - seq_load_count - 1;

  __shared__ uint32_t shared_write_val[2 * ZSTD_SEQ_ANS_STREAMS];
  __shared__ uint8_t shared_num_bits[2 * ZSTD_SEQ_ANS_STREAMS];
  int staged_write_count = 0; // caching values until ready to do the prefix sum
  // __shared__ sequence shared_sequences[WARP_SIZE];

  auto &encoder = encoders[ix_seq_type];

  // For the first sequence, need to write out the extra bits and initialize the ANS state
  int ix_seq = num_sequences - 1;
  uint8_t this_symbol = symbols[ix_seq];
  const bool not_rle_encoder = encoder.tablelog > 0;

  if (threadIdx.x < ZSTD_SEQ_ANS_STREAMS and not_rle_encoder)
  {
    encoder.encode_first_symbol(this_symbol);
  }

  int num_bits = 0;
  uint32_t write_val = 0;
  int this_match_len = __shfl_sync(WARP_ALL, thread_match_len, seq_load_count - 1);
  int this_lit_len = __shfl_sync(WARP_ALL, thread_lit_len, seq_load_count - 1);
  int this_offset = __shfl_sync(WARP_ALL, thread_offset, seq_load_count - 1);
  if (threadIdx.x >= ZSTD_SEQ_ANS_STREAMS and threadIdx.x < 2 * ZSTD_SEQ_ANS_STREAMS)
  {
    shared_num_bits[threadIdx.x - ZSTD_SEQ_ANS_STREAMS] = table_extra_bits[this_symbol];
    uint32_t seq_val;
    if (ix_seq_type == 0)
    {
      seq_val = this_lit_len;
    }
    else if (ix_seq_type == 1)
    {
      seq_val = this_offset;
    }
    else
    {
      seq_val = this_match_len;
    }
    shared_write_val[threadIdx.x - ZSTD_SEQ_ANS_STREAMS] = seq_val - baseline[this_symbol];
  }
  __syncwarp(); // Capture shared write

  // Stage the output writes in various thread registers
  staged_output_write(
    write_val,
    num_bits,
    ZSTD_SEQ_ANS_STREAMS,
    bit_writer,
    staged_write_count,
    shared_write_val,
    shared_num_bits,
    false /*force_flush_to_bitwriter*/
  );
  __syncwarp();

  --ix_seq;

  for (; ix_seq >= 0; --ix_seq)
  {
    this_symbol = symbols[ix_seq];
    if (ix_seq == next_load)
    {
      // Read the match tokens into registers, per thread
      thread_match_len = sequences.get_match_length(next_load - WARP_SIZE + threadIdx.x + 1);
      thread_lit_len = sequences.get_literal_length(next_load - WARP_SIZE + threadIdx.x + 1);
      thread_offset = sequences.get_offset(next_load - WARP_SIZE + threadIdx.x + 1);
      next_load -= WARP_SIZE;
    }
    const int src_ix = ix_seq % WARP_SIZE;
    int this_match_len = __shfl_sync(WARP_ALL, thread_match_len, src_ix);
    int this_lit_len = __shfl_sync(WARP_ALL, thread_lit_len, src_ix);
    int this_offset = __shfl_sync(WARP_ALL, thread_offset, src_ix);
    // Use 6 threads
    if (threadIdx.x < ZSTD_SEQ_ANS_STREAMS)
    {
      EncodeResult res{};
      if (not_rle_encoder)
      {
        res = encoder.encode_symbol(this_symbol);
      }
      shared_num_bits[threadIdx.x] = res.num_bits;
      shared_write_val[threadIdx.x] = res.prev_state;
    }
    else if (threadIdx.x < 2 * ZSTD_SEQ_ANS_STREAMS)
    {
      uint32_t seq_val;
      if (ix_seq_type == 0)
      {
        seq_val = this_lit_len;
      }
      else if (ix_seq_type == 1)
      {
        seq_val = this_offset;
      }
      else
      {
        seq_val = this_match_len;
      }
      shared_num_bits[threadIdx.x] = table_extra_bits[this_symbol];
      shared_write_val[threadIdx.x] = seq_val - baseline[this_symbol];

      // printf("ix seq %d val %u thread %d tablelog %d\n", ix_seq, seq_val, threadIdx.x, encoder.tablelog);
    }
    __syncwarp(); // Capture shared write
    // if (threadIdx.x < 6) {
    //     printf("tid %d writing %d num bits %d to ix %d\n", threadIdx.x, shared_write_val[threadIdx.x], shared_num_bits[threadIdx.x], staged_write_count + threadIdx.x);
    // }

    constexpr int shared_write_count = 2 * ZSTD_SEQ_ANS_STREAMS;
    staged_output_write(
      write_val,
      num_bits,
      shared_write_count,
      bit_writer,
      staged_write_count,
      shared_write_val,
      shared_num_bits,
      false /*force_flush_to_bitwriter*/
    );

    __syncwarp();
  }

  // Write out the last state. This is strangely not in the same order.
  // Needs to be in the order: ML, OF, LL
  if (threadIdx.x < ZSTD_SEQ_ANS_STREAMS)
  {
    const int this_ix_seq_type = 2 - threadIdx.x;
    auto &this_encoder = encoders[this_ix_seq_type];

    shared_write_val[threadIdx.x] = this_encoder.state;
    shared_num_bits[threadIdx.x] = this_encoder.tablelog;

    // printf("writing state %u num bits %d bit offset %d pointer %p base pointer %p\n",
    //     this_encoder.state & ((1 << num_bits) - 1), num_bits, bit_writer.get_offset(),
    //     reinterpret_cast<uint8_t*>(bit_writer.output) + ((bit_writer.bit_offset + 7) / 8), bit_writer.output);
  }

  __syncwarp(); // capture shared write

  staged_output_write(
    write_val,
    num_bits,
    ZSTD_SEQ_ANS_STREAMS,
    bit_writer,
    staged_write_count,
    shared_write_val,
    shared_num_bits,
    true /*force_flush_to_bitwriter*/
  );
  __syncwarp(); // Ensure sync before writing final byte

  if (threadIdx.x == 0)
  {
    bit_writer.write_bits(1, 1); // single bit, then finish the byte
    bit_writer.align_bytewise();
  }
  __syncwarp();
}

inline __device__ void write_sequence_section_header(BitWriter &bit_writer, const int num_sequences)
{
  // Contains the number of sequences and the symbol compression modes
  if (num_sequences < 128)
  {
    bit_writer.write_bits(num_sequences, 8);
  }
  else if (num_sequences < 0x7f00)
  {
    bit_writer.write_bits((num_sequences >> 8) + 128, 8);
    bit_writer.write_bits(num_sequences & 0xff, 8);
    // Note: currently, since "COMPRESS_NOMINAL_BLOCK_SIZE" is 64KB, we will never hit this case.
  }
  else
  {
    bit_writer.write_bits(0xff, 8);
    bit_writer.write_bits(num_sequences - 0x7f00, 16);
  }
}

inline __device__ void compress_sequences(
  const int num_sequences,
  const CompressSequenceBuffer &sequences,
  uint8_t *compressed_buffer,
  int *comp_buffer_size,
  SequenceCompressBuffers &seq_buffers,
  SymbolEncoder *encoders,
  uint8_t *symbol_array,
  const int max_output_size,
  const SharedConstantTables &constant_tables
)
{
  __shared__ BitWriter bit_writer;
  bit_writer.init(compressed_buffer, max_output_size);

  __syncwarp();

  // Write the header needed to decompress the sequences
  if (threadIdx.x == 0)
  {
    write_sequence_section_header(bit_writer, num_sequences);
  }
  __syncwarp();

  if (num_sequences == 0)
  {
    bit_writer.flush(threadIdx.x, WARP_ALL, WARP_SIZE);
    __syncwarp();
    *comp_buffer_size = bit_writer.compute_byte_offset(compressed_buffer);
    return;
  }

  // For now, use FSE_Compressed_Mode for all

  // First build the arrays from the sequences

  perform_initial_sequence_pass(num_sequences, sequences, symbol_array, seq_buffers.weight_freqs, constant_tables);

  // for (int ix_seq = threadIdx.x; ix_seq < num_sequences; ix_seq += 32) {
  //     printf("ix seq %d LL %u extra %u symbol %u baseline %u n bits %u\n",
  //             ix_seq, sequences[ix_seq][0], extra_bits[ix_seq], symbols[ix_seq],
  //             SEQ_LITERAL_LENGTH_BASELINES[symbols[ix_seq]], SEQ_LITERAL_LENGTH_EXTRA_BITS[symbols[ix_seq]]);
  // }

  uint8_t mode_byte = 0;
  unsigned shift_bits = 6;
  uint8_t *write_mode_byte;
  if (threadIdx.x == 0)
  {
    write_mode_byte = bit_writer.check_out_byte();
  }
  __syncwarp();
  for (int ix_seq_type = 0; ix_seq_type < ZSTD_SEQ_ANS_STREAMS; ++ix_seq_type)
  {
    // Construct the symbol encoders and write out the tables
    int accuracy_log = max_accuracies[ix_seq_type];

    bool use_default = num_sequences < ZSTD_MAX_DEFAULT_SEQUENCES;
    int rle_val = init_zstd_fse_encoder<SEQ_ANS_MAX_SYMBOLS>(
      accuracy_log,
      seq_buffers.weight_freqs[ix_seq_type],
      seq_buffers,
      num_sequences,
      encoders[ix_seq_type],
      use_default,
      ix_seq_type
    );
    __syncwarp();

    if (rle_val == -1)
    {
      if (not use_default)
      {
        // Not rle and not default
        output_fse_header(bit_writer, accuracy_log, seq_buffers.norm_weight_freqs, SEQ_ANS_MAX_SYMBOLS);
        mode_byte |= 2 << shift_bits;
      }
    }
    else
    {
      // Output the rle header
      assert(not use_default);
      if (threadIdx.x == 0)
      {
        bit_writer.write_bits(rle_val, 8);
        mode_byte |= 1 << shift_bits;
        encoders[ix_seq_type].tablelog = 0;
      }
      __syncwarp();
    }
    shift_bits -= 2;
  }

  if (threadIdx.x == 0)
  {
    bit_writer.check_in_byte(write_mode_byte, mode_byte);
  }

  // The ANS structures are stored as "LL, OF, ML".
  // The compress threads for ANS are "OF, ML, LL".
  // Use seq ix to do the appropriate mapping

  // Dispatch symbols appropriately.

  // Get the seq ix of the thread
  int ix_seq_type;
  const uint32_t *baselines;
  const uint8_t *extra_bits;
  if (threadIdx.x == 0 || threadIdx.x == 5)
  {
    // Offset
    baselines = constant_tables.of_baselines;
    extra_bits = constant_tables.of_extra_bits;
    ix_seq_type = 1;
  }
  else if (threadIdx.x == 1 || threadIdx.x == 4)
  {
    // Match len
    baselines = constant_tables.ml_baselines;
    extra_bits = constant_tables.ml_extra_bits;
    ix_seq_type = 2;
  }
  else
  {
    // Lit len
    baselines = constant_tables.ll_baselines;
    extra_bits = constant_tables.ll_extra_bits;
    ix_seq_type = 0;
  }

  __syncwarp();
  bit_writer.flush(threadIdx.x, WARP_ALL, WARP_SIZE);
  do_compress_sequences(
    encoders,
    bit_writer,
    &symbol_array[ix_seq_type * num_sequences],
    baselines,
    extra_bits,
    sequences,
    num_sequences,
    ix_seq_type
  );

  __syncwarp();

  bit_writer.flush(threadIdx.x, WARP_ALL, WARP_SIZE);
  *comp_buffer_size = bit_writer.compute_byte_offset(compressed_buffer);

  __syncwarp(); // Finish write before possibly proceeding to the next iteration.
}

} // namespace zstd
