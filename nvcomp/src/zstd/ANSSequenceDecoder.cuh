#pragma once

#include "ans.cuh"

// #define ZSTD_FSE_LOGGING 1

namespace zstd
{

constexpr int ANS_OUTPUT_INTERVAL = 256;

struct ANSSequenceDecoder
{
  ANSCache &ans_cache;
  sequence *sequence_buffer;
  FSETable fse_table;
  int num_sequences;
  int ix_sequence;
  const int ix_base_thread;
  int ix_table;
  unsigned ix_seq;
  const int ix_in_zstd_block;
  const unsigned block_mask;
  cg::thread_block_tile<1, void> group;
  SharedExtraBitTable *shared_eb_table;
  unsigned *shared_bit_ptr;
  int total_seq_bytes;
  DeviceBlockShare *this_block_share;

  // Eventually link to shared baselines / basecodes
  __device__ ANSSequenceDecoder(
    ANSCache &ans_cache,
    const int ix_in_zstd_block,
    const int ix_base_thread,
    SharedExtraBitTables &extra_tables,
    const int ix_zstd_block,
    unsigned *shared_bits
  )
      : ans_cache(ans_cache)
      , sequence_buffer(nullptr)
      , num_sequences(0)
      , ix_sequence(0)
      , ix_base_thread(ix_base_thread)
      , ix_table(0)
      , ix_seq(0)
      , ix_in_zstd_block(ix_in_zstd_block)
      , block_mask(((1 << NUM_THREADS_PER_ANS_BLOCK) - 1) << ix_base_thread)
      , group(cg::this_thread())
      , shared_eb_table()
      , shared_bit_ptr(&shared_bits[ix_zstd_block * 2])
      , total_seq_bytes(0)
      , this_block_share(nullptr)
  {
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
    shared_eb_table = extra_tables[ix_seq];
  }

  __device__ void init(const FSETables &fse_tables)
  {
#ifdef STAGE_LOGGING
    if (ix_in_zstd_block == 0)
    {
      printf(
        "bid %d warp %d starting FSE on frame %d block %d clock %lu\n",
        blockIdx.x,
        ix_warp(),
        this_block_share->ix_frame,
        this_block_share->ix_block,
        cuda::std::chrono::system_clock::now()
      );
    }
#endif
    BitReader bit_reader{this_block_share->compressed_buffer, this_block_share->comp_block_size};
    bit_reader.get_read_pointer(this_block_share->block_header.seq_section_start);

    num_sequences = this_block_share->num_sequences;
    sequence_buffer = this_block_share->sequence_buffer;
    // Skip past table spec
    bit_reader.get_read_pointer(this_block_share->block_header.block_seq_table_size);

    // Now decode the sequences. Do this using six threads.
    assert(ix_in_zstd_block < NUM_THREADS_PER_ANS_BLOCK);

    ix_seq = ix_in_zstd_block % 3;

    const uint8_t accuracy_log = fse_tables.get_table(ix_seq).accuracy_log;

    const int buffer_size = bit_reader.rem_bytes();
    const uint8_t *const src = bit_reader.get_read_pointer(buffer_size);
    int padding = 8 - highest_set_bit(src[buffer_size - 1]);

    ans_cache.init_cache(src, buffer_size, ix_in_zstd_block);

    int num_bits = ix_in_zstd_block < 3 ? accuracy_log : 0;
    uint8_t *this_bit_ptr = reinterpret_cast<uint8_t *>(shared_bit_ptr) + ix_in_zstd_block;
    *this_bit_ptr = num_bits;
    __syncwarp(block_mask); // share bits read must be visible
    unsigned reg1 = shared_bit_ptr[0];
    unsigned reg2 = shared_bit_ptr[1];
    uint16_t read_val = ans_cache.read(reg1, reg2, num_bits, ix_in_zstd_block);

#ifdef ZSTD_FSE_LOGGING
    printf(
      "initial read val %u pointer %p num bits %d buffer size %d bit read buff size %d\n",
      read_val,
      &src[buffer_size - 1],
      num_bits,
      buffer_size,
      bit_reader.buffer_size
    );
#endif

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

    uint16_t state = __shfl_sync(block_mask, read_val, ix_seq + ix_base_thread);
    fse_table = fse_tables.get_table(ix_seq);

    ix_table = state;

    ix_sequence = 0;
    total_seq_bytes = 0;
  }

  inline __device__ void decode_sequence(bool active)
  {
    uint32_t other_baseline = 0;
    uint32_t new_baseline = 0;
    const int sourceLane = min(ix_base_thread + 5 - ix_in_zstd_block, WARP_SIZE - 1);

    uint8_t this_bits;
    if (active and ix_in_zstd_block >= 3)
    {
      uint8_t this_symbol;
      fse_table.get_all(ix_table, this_symbol, new_baseline, this_bits);

#ifdef ZSTD_FSE_LOGGING
      printf(
        "ix sequence %d state %u bits %u symbol %u baseline %u\n",
        ix_sequence,
        ix_table,
        this_bits,
        this_symbol,
        new_baseline
      );
#endif
      uint8_t other_bits = shared_eb_table->extra_bits[this_symbol];
      other_baseline = shared_eb_table->baselines[this_symbol];
      uint8_t *other_bit_ptr = reinterpret_cast<uint8_t *>(shared_bit_ptr) + 5 - ix_in_zstd_block;
      uint8_t *this_bit_ptr = reinterpret_cast<uint8_t *>(shared_bit_ptr) + ix_in_zstd_block;
      *this_bit_ptr = this_bits;
      *other_bit_ptr = other_bits;
    }

    __syncwarp();
    other_baseline = __shfl_sync(WARP_ALL, other_baseline, sourceLane);

    unsigned reg1, reg2;
    if (active)
    {
      reg1 = shared_bit_ptr[0];
      reg2 = shared_bit_ptr[1];
    }

    if (active and ix_in_zstd_block < 3)
    {
      this_bits = (reg1 >> (ix_in_zstd_block * 8)) & 0xff;
    }
    uint32_t new_val = 0;

    if (active and ans_cache.total_bits_remaining > 0)
    {
      new_val = ans_cache.read(reg1, reg2, this_bits, ix_in_zstd_block);
#ifdef ZSTD_FSE_LOGGING
      printf("read val %u ix table %d\n", new_val, ix_table);
#endif
    }

    if (ix_in_zstd_block < 3)
    {
      new_baseline = other_baseline;
    }

    ix_table = new_baseline + new_val;
    total_seq_bytes += ix_table; // We do this for all threads for perf, but it will only be used by a few threads

    if (active and ix_in_zstd_block < 3)
    {
#ifdef ZSTD_FSE_LOGGING
      printf("write ix seq %d val %u\n", ix_sequence, ix_table);
#endif
      sequence_buffer[ix_sequence][ix_seq] = ix_table;
    }

    __syncwarp();
  }

  inline __device__ void decode_sequences(int iter_sequence_count, bool active)
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

    // Set default value for Coverity, which doesn't like that nothing is set in threads with ix_in_zstd_block < 3.
    // Threads with ix_in_zstd_block < 3 get the value of this_symbol with __shfl_sync from matching thread with ix_in_zstd_block >= 3
    // At the same time they passed their this_symbol (which is now set to 0) to threads with ix_in_zstd_block >= 3, which ignore the returned value.
    for (int ix_local_iter = 0; ix_local_iter < iter_sequence_count;)
    {
      int inner_size = min(iter_sequence_count, ix_local_iter + ANS_OUTPUT_INTERVAL);
      // Do more inner loops
      static_assert(ANS_OUTPUT_INTERVAL % 2 == 0);
      for (; ix_local_iter < inner_size; ++ix_local_iter, ++ix_sequence)
      {
        decode_sequence(active);
      }

      if (active and ix_in_zstd_block == 0)
      {
        // decode seq count is interpreted as "th7e last ix_sequence that was decoded"
        // We also reuse this variable and specify that when all sequences are decoded, we store "num_sequences"
        this_block_share->decode_seq_count.store(ix_sequence - 1, cuda::std::memory_order_release);
      }
    }
  }

  inline __device__ void finish_block()
  {
    if (ix_in_zstd_block == 1)
    {
#ifdef STAGE_LOGGING
      printf(
        "finishing fse frame %d block %d clock %lu\n",
        this_block_share->ix_frame,
        this_block_share->ix_block,
        cuda::std::chrono::system_clock::now()
      );
#endif

      this_block_share->total_seq_bytes = total_seq_bytes;
      this_block_share->decode_seq_count.store(num_sequences, cuda::std::memory_order_release);
    }
  }
};

} // namespace zstd
