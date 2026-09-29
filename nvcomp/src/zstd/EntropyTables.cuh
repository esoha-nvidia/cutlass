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

#include "types.cuh"
#include "utils.cuh"

namespace zstd
{

// Used for Huffman compression
struct CompressHuffmanTable
{
  uint16_t code[HUF_MAX_SYMBOLS];
  uint8_t bits[HUF_MAX_SYMBOLS];
  uint8_t max_bits;
  int num_symbols;
};

// This is where the Huffman table is stored for table construction
// (init_huff_tables)
// It's stored in the block share, then copied into the "MetaHuffmanTable" during decompression
struct HuffmanTable
{
  uint8_t symbols[HUF_MAX_SYMBOLS];
  uint16_t basecodes[HUF_MAX_OFFSET_SIZE];
  uint8_t offsets[HUF_MAX_OFFSET_SIZE];
  int max_bits;

  inline __device__ void deepcopy(const HuffmanTable &other_table)
  {
    auto in_warp_idx = thread_warp_ix();
    for (int ix = in_warp_idx; ix < HUF_MAX_SYMBOLS; ix += WARP_SIZE)
    {
      symbols[ix] = other_table.symbols[ix];
    }

    if (in_warp_idx < HUF_MAX_OFFSET_SIZE)
    {
      basecodes[in_warp_idx] = other_table.basecodes[in_warp_idx];
      offsets[in_warp_idx] = other_table.offsets[in_warp_idx];
    }

    max_bits = other_table.max_bits;
  }
};

// Helper struct for MetaHuffmanTable -- each Huffman table corresponds
// to one of these
struct ActiveHuffmanTable
{
  uint8_t symbols[HUF_MAX_SYMBOLS];
  int max_bits;
};

// Used for compact SHMEM storage of Huffman tables for decompression.
// Static allocation of this allows the compiler to know
// Shared, so LDS instructions are used.
struct MetaHuffmanTable
{
  ActiveHuffmanTable huff_tables[MAX_HUFF_TABLES];
  uint8_t offsets[HUF_MAX_OFFSET_SIZE * MAX_HUFF_TABLES]; // By having this be contiguous, can remove bank conflicts

  inline __device__ void deepcopy(const HuffmanTable &other_table, int ix_table)
  {
    auto in_warp_idx = thread_warp_ix();
    for (int ix = in_warp_idx; ix < HUF_MAX_SYMBOLS; ix += WARP_SIZE)
    {
      huff_tables[ix_table].symbols[ix] = other_table.symbols[ix];
    }
    if (in_warp_idx < HUF_MAX_OFFSET_SIZE)
    {
      offsets[HUF_MAX_OFFSET_SIZE * ix_table + in_warp_idx] = other_table.offsets[in_warp_idx];
    }
    huff_tables[ix_table].max_bits = other_table.max_bits;
  }

  inline const __device__ uint8_t &get_offset(const int ix_table, const int ix_offset) const
  {
    return offsets[HUF_MAX_OFFSET_SIZE * ix_table + ix_offset];
  }
};

// Raw FSE Tables are computationally cheaper but take more space.
// The table init for Huffman uses these.
struct RawFSETable
{
  int accuracy_log;
  uint8_t *symbols;
  uint16_t *baselines;
  uint8_t *next_bits;
  __device__ void set_symbol(uint16_t state, uint8_t symbol) { symbols[state] = symbol; }

  __device__ void set_baseline(uint16_t state, uint16_t baseline) { baselines[state] = baseline; }

  __device__ void set_n_bits(uint16_t state, uint8_t n_bits) { next_bits[state] = n_bits; }

  __device__ uint8_t get_symbol(uint16_t state) { return symbols[state]; }
  __device__ uint16_t get_baseline(uint16_t state) { return baselines[state]; }

  __device__ uint8_t get_n_bits(uint16_t state) { return next_bits[state]; }
};

struct NBitsWrapper
{
  uchar4 n_bits;
  inline __device__ uint8_t get_symbol_bits() const { return n_bits.w; }

  inline __device__ uint8_t get_baseline_bits() const { return n_bits.x; }

  inline __device__ uint8_t get_nbits_bits() const { return n_bits.y; }

  inline __device__ uint8_t get_cell_bits() const { return n_bits.z; }

  inline __device__ void set_symbol_bits(uint8_t val) { n_bits.w = val; }

  inline __device__ void set_baseline_bits(uint8_t val) { n_bits.x = val; }

  inline __device__ void set_nbits_bits(uint8_t val) { n_bits.y = val; }

  inline __device__ void set_cell_bits(uint8_t val) { n_bits.z = val; }
};

// The main FSE table type for decompressing one of the
// 3 alphabets within a single block
struct FSETable
{
  uint32_t *tbl;
  int accuracy_log; // TODO: we could combine the accuracy log / n bits registers, because the accuracy log
  // is never used after the copy and the other values aren't used during the copy. The bit values can be
  // uniquely determined using the seg type, which is also available during the copy.
  // So a union of "accuracy log / NBitsWrapper" would be sufficient
  NBitsWrapper n_bit_wrapper;

  inline __device__ int compute_table_size_words(uint8_t seq_type)
  {
    const int table_size = 1 << accuracy_log;
    const int cell_size_bits = SEQ_MAX_BITS[seq_type][0] + SEQ_MAX_BITS[seq_type][1] + SEQ_MAX_BITS[seq_type][2];
    return (cell_size_bits * table_size + 31) / 32;
  }

  inline __device__ void deepcopy(const FSETable &other_table, uint8_t seq_type)
  {
    accuracy_log = other_table.accuracy_log;
    n_bit_wrapper = other_table.n_bit_wrapper;

    const int tbl_size_words = compute_table_size_words(seq_type);
    for (int ix = thread_warp_ix(); ix < tbl_size_words; ix += WARP_SIZE)
    {
      tbl[ix] = other_table.tbl[ix];
    }

    __syncwarp(WARP_ALL);
  }

  inline __device__ void zero_table(uint8_t seq_type)
  {
    const int tbl_size_words = compute_table_size_words(seq_type);
    for (int ix = thread_warp_ix(); ix < tbl_size_words; ix += WARP_SIZE)
    {
      tbl[ix] = 0;
    }
    __syncwarp(WARP_ALL);
  }

  /* Using these getters and setters is very inefficient. Calling each individually
   * will result in 2-3 times as many shared memory accesses as are necessary
   */
  __device__ void set(uint16_t state, uint8_t cumul_bits, uint16_t val, uint8_t n_bits)
  {
    uint16_t bit_offset = n_bit_wrapper.get_cell_bits() * state + cumul_bits;
    uint8_t bo_mod = bit_offset & 31;
    uint16_t word1_idx = bit_offset >> 5;

    uint64_t bit_mask = ~(((1ULL << n_bits) - 1) << bo_mod);
    uint64_t set_val = ((uint64_t)val) << bo_mod;

    uint32_t lo = set_val;
    uint32_t lo_mask = bit_mask;
    uint32_t hi = set_val >> 32;
    uint32_t hi_mask = bit_mask >> 32;

    // clear out bits we want to set
    atomicAnd(&tbl[word1_idx], lo_mask);
    atomicOr(&tbl[word1_idx], lo);
    if (bo_mod + n_bits > 32)
    {
      atomicAnd(&tbl[word1_idx + 1], hi_mask);
      atomicOr(&tbl[word1_idx + 1], hi);
    }
  }

  __device__ uint16_t get(uint16_t state, uint8_t cumul_bits, uint8_t n_bits) const
  {
    uint16_t bit_offset = n_bit_wrapper.get_cell_bits() * state + cumul_bits;

    uint16_t word1_idx = bit_offset >> 5;
    uint32_t word1 = tbl[word1_idx];

    // only read the second word if necessary
    uint32_t word2 = ((bit_offset & 31) + n_bits > 32) ? tbl[word1_idx + 1] : 0;
    return __funnelshift_r(word1, word2, bit_offset) & ((1 << n_bits) - 1);
  }

  __device__ void get_all(uint16_t state, uint8_t &symbol, uint32_t &baseline, uint8_t &n_bits) const
  {
    uint16_t bit_offset = n_bit_wrapper.get_cell_bits() * state;

    uint16_t word1_idx = bit_offset >> 5;
    uint32_t word1 = tbl[word1_idx];
    // It's safe to always read the next word because we allocate 1 extra word of SHMEM to avoid OOB accesses
    uint32_t word2 = tbl[word1_idx + 1];
    uint32_t tot = __funnelshift_r(word1, word2, bit_offset);

    symbol = tot & ((1 << n_bit_wrapper.get_symbol_bits()) - 1);
    baseline = (tot >> n_bit_wrapper.get_symbol_bits()) & ((1 << n_bit_wrapper.get_baseline_bits()) - 1);
    n_bits = (tot >> (n_bit_wrapper.get_symbol_bits() + n_bit_wrapper.get_baseline_bits())) &
             ((1 << n_bit_wrapper.get_nbits_bits()) - 1);
  }

  __device__ void set_symbol(uint16_t state, uint8_t symbol) { set(state, 0, symbol, n_bit_wrapper.get_symbol_bits()); }

  __device__ uint8_t get_symbol(uint16_t state) const { return get(state, 0, n_bit_wrapper.get_symbol_bits()); }

  __device__ void set_baseline(uint16_t state, uint16_t baseline)
  {
    set(state, n_bit_wrapper.get_symbol_bits(), baseline, n_bit_wrapper.get_baseline_bits());
  }
  __device__ uint16_t get_baseline(uint16_t state) const
  {
    return get(state, n_bit_wrapper.get_symbol_bits(), n_bit_wrapper.get_baseline_bits());
  }

  __device__ void set_n_bits(uint16_t state, uint8_t n_bits)
  {
    set(
      state,
      n_bit_wrapper.get_symbol_bits() + n_bit_wrapper.get_baseline_bits(),
      n_bits,
      n_bit_wrapper.get_nbits_bits()
    );
  }

  __device__ uint8_t get_n_bits(uint16_t state) const
  {
    return get(
      state,
      n_bit_wrapper.get_symbol_bits() + n_bit_wrapper.get_baseline_bits(),
      n_bit_wrapper.get_nbits_bits()
    );
  }
};

// The main FSETables struct that used to store information for ANS decompressing a single block
// Now, it's a construction helper for the individual tables.
struct FSETables
{
  FSETable ll_fse_table;
  FSETable of_fse_table;
  FSETable ml_fse_table;

  FSETables() = default;
  // This constructor takes in a global fsetables object and a shmem buffer,
  // and fills the appropriate amount
  __device__ FSETables(FSETables &other, uint32_t *buffer)
  {
    int ix_buffer = 0;
    ll_fse_table.tbl = buffer;
    ll_fse_table.deepcopy(other.ll_fse_table, static_cast<uint8_t>(SequenceType::LiteralLength));
    ix_buffer = max_ll_words;
    of_fse_table.tbl = &buffer[ix_buffer];
    of_fse_table.deepcopy(other.of_fse_table, static_cast<uint8_t>(SequenceType::Offset));
    ix_buffer += max_of_words;
    ml_fse_table.tbl = &buffer[ix_buffer];
    ml_fse_table.deepcopy(other.ml_fse_table, static_cast<uint8_t>(SequenceType::MatchLength));
  }

  inline __device__ void deepcopy(FSETables &other)
  {
    ll_fse_table.deepcopy(other.ll_fse_table, static_cast<uint8_t>(SequenceType::LiteralLength));
    of_fse_table.deepcopy(other.of_fse_table, static_cast<uint8_t>(SequenceType::Offset));
    ml_fse_table.deepcopy(other.ml_fse_table, static_cast<uint8_t>(SequenceType::MatchLength));
  }

  __device__ FSETable &get_table(uint8_t index)
  {
    switch (index)
    {
      case 0:
        return ll_fse_table;
      case 1:
        return of_fse_table;
      case 2:
        return ml_fse_table;
      default:
        assert(false);
        return ml_fse_table;
    }
  }

  const __device__ FSETable &get_table(uint8_t index) const
  {
    switch (index)
    {
      case 0:
        return ll_fse_table;
      case 1:
        return of_fse_table;
      case 2:
        return ml_fse_table;
      default:
        assert(false);
        return ml_fse_table;
    }
  }
};

template <typename MemoryManagerT>
inline __device__ void allocate_fse_table(RawFSETable &fse_table, MemoryManagerT &memory_manager, uint8_t max_accuracy)
{
  const size_t table_size = 1 << max_accuracy;

  fse_table.symbols = memory_manager.template allocate<uint8_t>(table_size);
  fse_table.baselines = memory_manager.template allocate<uint16_t>(table_size);
  fse_table.next_bits = memory_manager.template allocate<uint8_t>(table_size);
}

inline __device__ size_t fse_table_size_words(uint8_t max_accuracy, uint8_t seq_type)
{
  const size_t table_size = 1 << max_accuracy;
  const size_t cell_size_bits = SEQ_MAX_BITS[seq_type][0] + SEQ_MAX_BITS[seq_type][1] + SEQ_MAX_BITS[seq_type][2];
  return (cell_size_bits * table_size + 31) / 32;
}

template <typename MemoryManagerT>
inline __device__ void
allocate_fse_table(FSETable &fse_table, MemoryManagerT &memory_manager, uint8_t max_accuracy, uint8_t seq_type)
{
  fse_table.tbl = memory_manager.template allocate<uint32_t>(fse_table_size_words(max_accuracy, seq_type));
}

// Used for table construction
struct ANSTableConstructionBuffers
{
  uint32_t state_desc[SEQ_ANS_MAX_SYMBOLS];
  uint16_t valid_spots[MAX_ANS_TABLE_SIZE];
  int16_t frequencies[SEQ_ANS_MAX_SYMBOLS];
};

struct HuffmanTableConstructionBuffers
{
  RawFSETable fse_table; // Will allocate maximum sized table
  ANSTableConstructionBuffers ans_buffers;
  uint32_t rank_counts[HUF_MAX_OFFSET_SIZE];
  int16_t frequencies[HUF_MAX_SYMBOLS];
  uint8_t bits[HUF_MAX_SYMBOLS];
  uint8_t weights[HUF_MAX_SYMBOLS];
};

struct EncodeResult
{
  uint16_t prev_state;
  uint8_t num_bits;
};

// Used to do the ans encoding for compression
struct SymbolEncoder
{
  uint16_t *encoding_table;
  st_cell *stt;
  int tablelog;
  uint16_t state;

  SymbolEncoder() = default;

  inline __device__ void encode_first_symbol(uint8_t symbol)
  {
    // Reverse the calculation we did during symbol table construction to recover
    // abs(norm_counts[symbol]) without doing an additional gmem access
    st_cell cell = stt[symbol];
    uint32_t max_n_bits = (cell.n_bits + (1 << (ANS_ALGORITHM_MAX_TABLELOG + 1))) >> (ANS_ALGORITHM_MAX_TABLELOG + 1);
    state = (max_n_bits << (ANS_ALGORITHM_MAX_TABLELOG + 1)) - cell.n_bits;

    int norm_count = state >> max_n_bits;
    int init_loc = norm_count + cell.state_offset;

    state = encoding_table[init_loc];
  }

  inline __device__ EncodeResult encode_symbol(uint8_t symbol)
  {
    EncodeResult res;
    st_cell cell = stt[symbol];
    res.prev_state = state;
    res.num_bits = (cell.n_bits + state) >> (ANS_ALGORITHM_MAX_TABLELOG + 1);
    const uint16_t enc_idx = (state >> res.num_bits);
    state = encoding_table[cell.state_offset + enc_idx];
    // printf("thread %d encoding symbol %u state %u prev state %u bits %u cell.num bits %u state offset %d enc_idx %u\n",
    //     thread_warp_ix(), symbol, state, res.prev_state, res.num_bits, cell.n_bits, cell.state_offset, enc_idx);
    return res;
  }
};

// The buffers required for ANS compression
template <int max_table_size, int max_symbols>
struct ANSCompressTableBuffers
{
  uint16_t starts[max_symbols];
  uint8_t dst_table[max_table_size];
  uint32_t weight_freqs[3][max_symbols];
  int16_t norm_weight_freqs[max_symbols];
};

typedef ANSCompressTableBuffers<MAX_ANS_TABLE_SIZE, SEQ_ANS_MAX_SYMBOLS> SequenceCompressBuffers;

// The buffers required for Huffman compression
struct CompressHuffmanBuffers
{
  ANSCompressTableBuffers<HUF_FSE_MAX_TABLE_SIZE, HUF_MAX_OFFSET_SIZE> ans_buffers;
  SymbolEncoder encoder;
  uint32_t rank_counts[HUF_MAX_OFFSET_SIZE];
  uint16_t basecodes[HUF_MAX_OFFSET_SIZE];
};

inline __device__ void alloc_comp_huff_ans_encoder(SymbolEncoder &encoder)
{
  __shared__ st_cell stt[HUF_MAX_OFFSET_SIZE];
  __shared__ uint16_t encoding_table[HUF_FSE_MAX_TABLE_SIZE];

  encoder.stt = stt;
  encoder.encoding_table = encoding_table;
  encoder.tablelog = HUF_FSE_WEIGHT_MAX_ACCURACY_LOG;
}

constexpr size_t compute_huff_weights_fse_table_size() { return 1 << HUF_FSE_WEIGHT_MAX_ACCURACY_LOG; }

constexpr size_t compute_fse_table_alloc() { return 2 * sizeof(ANSTableConstructionBuffers) - 1; }

constexpr size_t compute_huff_table_alloc()
{
  return 2 * sizeof(HuffmanTableConstructionBuffers) - 1 + compute_huff_weights_fse_table_size();
}

inline __device__ void
allocate_fse_tables(MemoryManager &shmem_manager, FSETables *fse_tables, const int num_ans_tables = MAX_ANS_TABLES)
{
  for (int ix = 0; ix < num_ans_tables; ++ix)
  {
    allocate_fse_table(fse_tables[ix].ll_fse_table, shmem_manager, LITERAL_LENGTH_MAX_ACCURACY, 0);
    allocate_fse_table(fse_tables[ix].of_fse_table, shmem_manager, OFFSET_MAX_ACCURACY, 1);
    allocate_fse_table(fse_tables[ix].ml_fse_table, shmem_manager, MATCH_LENGTH_MAX_ACCURACY, 2);
  }
}

inline __device__ void allocate_comp_fse_tables(SymbolEncoder *encoders)
{
  __shared__ st_cell stt_ll[LITERAL_LENGTH_MAX_SYMBOLS];
  __shared__ uint16_t encoding_table_ll[1 << LITERAL_LENGTH_MAX_ACCURACY];

  __shared__ st_cell stt_of[OFFSET_MAX_SYMBOLS];
  __shared__ uint16_t encoding_table_of[1 << OFFSET_MAX_ACCURACY];

  __shared__ st_cell stt_ml[MATCH_LENGTH_MAX_SYMBOLS];
  __shared__ uint16_t encoding_table_ml[1 << MATCH_LENGTH_MAX_ACCURACY];

  encoders[0].tablelog = LITERAL_LENGTH_MAX_ACCURACY;
  encoders[0].stt = stt_ll;
  encoders[0].encoding_table = encoding_table_ll;

  encoders[1].tablelog = OFFSET_MAX_ACCURACY;
  encoders[1].stt = stt_of;
  encoders[1].encoding_table = encoding_table_of;

  encoders[2].tablelog = MATCH_LENGTH_MAX_ACCURACY;
  encoders[2].stt = stt_ml;
  encoders[2].encoding_table = encoding_table_ml;
}

struct SharedExtraBitTable
{
  static constexpr int max_symbols =
    std::max(std::max(OFFSET_MAX_SYMBOLS, MATCH_LENGTH_MAX_SYMBOLS), LITERAL_LENGTH_MAX_SYMBOLS);
  uint8_t extra_bits[max_symbols];
  uint32_t baselines[max_symbols];
};

// These "extra bits" tables could be stored in __constant__ memory,
// but it's faster to load in shared memory.
struct SharedExtraBitTables
{
  SharedExtraBitTable ll_table;
  SharedExtraBitTable of_table;
  SharedExtraBitTable ml_table;

  inline __device__ void init()
  {
    for (int ix = threadIdx.x; ix < LITERAL_LENGTH_MAX_SYMBOLS; ix += blockDim.x)
    {
      ll_table.extra_bits[ix] = SEQ_LITERAL_LENGTH_EXTRA_BITS[ix];
      ll_table.baselines[ix] = SEQ_LITERAL_LENGTH_BASELINES[ix];
    }

    for (int ix = threadIdx.x; ix < OFFSET_MAX_SYMBOLS; ix += blockDim.x)
    {
      of_table.extra_bits[ix] = SEQ_OFFSET_EXTRA_BITS[ix];
      of_table.baselines[ix] = SEQ_OFFSET_BASELINES[ix];
    }

    for (int ix = threadIdx.x; ix < MATCH_LENGTH_MAX_SYMBOLS; ix += blockDim.x)
    {
      ml_table.extra_bits[ix] = SEQ_MATCH_LENGTH_EXTRA_BITS[ix];
      ml_table.baselines[ix] = SEQ_MATCH_LENGTH_BASELINES[ix];
    }
  }

  __device__ SharedExtraBitTable *operator[](int ix)
  {
    switch (static_cast<SequenceType>(ix))
    {
      case SequenceType::LiteralLength:
        return &ll_table;
      case SequenceType::Offset:
        return &of_table;
      case SequenceType::MatchLength:
        return &ml_table;
      default:
        assert(false); // invalid input
        return &ll_table;
    }
  }
};

// The aggregation of all the shared constant tables.
// This is used by compression and decompression
struct SharedConstantTables
{
  uint32_t ll_baselines[LITERAL_LENGTH_MAX_SYMBOLS];
  uint8_t ll_extra_bits[LITERAL_LENGTH_MAX_SYMBOLS];
  uint32_t of_baselines[OFFSET_MAX_SYMBOLS];
  uint8_t of_extra_bits[OFFSET_MAX_SYMBOLS];
  uint32_t ml_baselines[MATCH_LENGTH_MAX_SYMBOLS];
  uint8_t ml_extra_bits[MATCH_LENGTH_MAX_SYMBOLS];

  inline __device__ void init()
  {
    for (int ix = thread_warp_ix(); ix < LITERAL_LENGTH_MAX_SYMBOLS; ix += WARP_SIZE)
    {
      ll_baselines[ix] = SEQ_LITERAL_LENGTH_BASELINES[ix];
      ll_extra_bits[ix] = SEQ_LITERAL_LENGTH_EXTRA_BITS[ix];
    }

    for (int ix = thread_warp_ix(); ix < OFFSET_MAX_SYMBOLS; ix += WARP_SIZE)
    {
      of_baselines[ix] = SEQ_OFFSET_BASELINES[ix];
      of_extra_bits[ix] = SEQ_OFFSET_EXTRA_BITS[ix];
    }

    for (int ix = thread_warp_ix(); ix < MATCH_LENGTH_MAX_SYMBOLS; ix += WARP_SIZE)
    {
      ml_baselines[ix] = SEQ_MATCH_LENGTH_BASELINES[ix];
      ml_extra_bits[ix] = SEQ_MATCH_LENGTH_EXTRA_BITS[ix];
    }
    __syncwarp();
  }
};

} // namespace zstd