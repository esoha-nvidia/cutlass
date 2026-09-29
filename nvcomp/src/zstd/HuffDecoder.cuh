#pragma once

#include "huffman.cuh"

namespace zstd
{

struct HuffDecoder
{
private:
  uint8_t *literal_buffer;
  const uint8_t *src;
  const uint8_t *symbols;
  DeviceBlockShare *block_share_ptr;
  int compressed_size;
  int num_literals;
  int full_num_literals;
  int ix_lit;
  const int ix_in_zstd_block;
  bool active;
  int num_streams;
  unsigned block_mask;
  HuffCache huff_cache;
  uint16_t state;
  unsigned bc2;
  unsigned bc3;
  unsigned bc5;
  unsigned bc6;
  unsigned bc8;
  unsigned bc9;
  unsigned bc11;
  unsigned of0;
  unsigned of25;
  unsigned of811;
  int max_bits;
  int ix_cache;
  unsigned cache_val;
  int align_bytes;

public:
  __device__ HuffDecoder(const int ix_in_zstd_block, unsigned &shared_val)
      : ix_in_zstd_block(ix_in_zstd_block)
      , active(false)
      , huff_cache(shared_val)
  {}

  inline __device__ void init(DeviceBlockShare &block_share, const MetaHuffmanTable &meta_table, const int ix_block)
  {

    const HuffmanTable &huff_table = *block_share.global_huff_table;
    auto &block_header = block_share.block_header;
    const unsigned literal_block_start = 3;
    unsigned huf_start_offset = literal_block_start + block_header.literal_header_size +
                                block_header.literal_header.table_desc_size;
    compressed_size = block_share.block_header.literal_header.compressed_size -
                      block_header.literal_header.table_desc_size;
    num_literals = block_share.block_header.literal_header.regenerated_size;
    full_num_literals = num_literals;

    ix_lit = 0;
    src = block_share.compressed_buffer + huf_start_offset;
    literal_buffer = block_share.literal_buffer;
    num_streams = block_share.block_header.literal_header.num_streams;
#ifdef STAGE_LOGGING
    if (thread_warp_ix() % NUM_THREADS_PER_HUFF_BLOCK == 0)
    {
      printf(
        "bid %d warp %d starting huffman on frame %d block %d num streams %d clock %lu\n",
        blockIdx.x,
        ix_warp(),
        block_share.ix_frame,
        block_share.ix_block,
        num_streams,
        cuda::std::chrono::system_clock::now()
      );
    }
#endif
    if (num_streams == 4)
    {
      block_mask = ((1 << NUM_THREADS_PER_HUFF_BLOCK) - 1)
                   << thread_warp_ix() / NUM_THREADS_PER_HUFF_BLOCK * NUM_THREADS_PER_HUFF_BLOCK;
      update_huffman_4way_pointers(src, literal_buffer, num_literals, ix_in_zstd_block, compressed_size, block_mask);
    }
    else if (ix_in_zstd_block > 0)
    {
      return;
    }
    else
    {
      block_mask = 1 << thread_warp_ix();
    }
    active = true;

    const int padding = 8 - highest_set_bit(src[compressed_size - 1]);
    int bit_offset = compressed_size * 8 - padding;

    huff_cache.init(src, bit_offset);
    state = huff_cache.read(huff_table.max_bits, block_mask);

    bc2 = huff_table.basecodes[2] | (huff_table.basecodes[1] << 16);
    bc3 = huff_table.basecodes[3];
    bc5 = huff_table.basecodes[5] | (huff_table.basecodes[4] << 16);
    bc6 = huff_table.basecodes[6];
    bc8 = huff_table.basecodes[8] | (huff_table.basecodes[7] << 16);
    bc9 = huff_table.basecodes[9];
    bc11 = huff_table.basecodes[11] | (huff_table.basecodes[10] << 16);
    const uint8_t *offsets = &meta_table.get_offset(ix_block, 0);
    of0 = (offsets[11]) + (offsets[5] << 8) + (offsets[8] << 16) + (offsets[2] << 24);
    of25 = (offsets[4]) + (offsets[3] << 8) + (offsets[1] << 16) + (offsets[0] << 24);
    of811 = (offsets[10]) + (offsets[9] << 8) + (offsets[7] << 16) + (offsets[6] << 24);

    symbols = meta_table.huff_tables[ix_block].symbols;
    max_bits = huff_table.max_bits;
    align_bytes = (4 - ((uintptr_t)literal_buffer % 4)) & 0x3;
    ix_cache = 0;
    cache_val = 0;
    block_share_ptr = &block_share;
  }

  inline __device__ void do_decode_literals(const int num_decode, const unsigned full_mask)
  {
    assert(active); //ensure the object is properly initialized
    const int ix_stop = ix_lit + num_decode;

    // The below loop is so complicated to avoid a global write on every iteration. Now we write words in the likely case.
    for (; ix_lit < ix_stop; ++ix_lit)
    {
      const bool valid = ix_lit < num_literals;
      uint8_t lit = decode_huffman_symbol(
        state,
        src,
        full_mask,
        symbols,
        max_bits,
        huff_cache,
        bc2,
        bc3,
        bc5,
        bc6,
        bc8,
        bc9,
        bc11,
        of0,
        of25,
        of811,
        ix_lit < num_literals
      );

      // This is so we can do the reads but then skip ahead
      if (not valid)
      {
        continue;
      }

      if (align_bytes > 0)
      {
        literal_buffer[ix_lit] = lit;
        --align_bytes;
      }
      else
      {
        cache_val |= lit << (ix_cache * 8);
        ++ix_cache;
        if (ix_cache == 4)
        {
          *reinterpret_cast<uint32_t *>(&literal_buffer[ix_lit - 3]) = cache_val;
          cache_val = 0;
          ix_cache = 0;

#if (__CUDA_ARCH__ == 890)
          if (ix_in_zstd_block == 0 and ix_lit % 128 < 4)
          {
            // Allow some incremental progress. Not great because a single thread has to drive
            // this part for some time.
            block_share_ptr->decode_lit_count.store(ix_lit, cuda::std::memory_order_release);
          }
#endif // __CUDA_ARCH__
        }
      }
    }

    ix_lit = min(ix_lit, num_literals);

    if (ix_lit == num_literals and num_literals > 0)
    {
      for (int ix = 0; ix < ix_cache; ++ix)
      {
        literal_buffer[ix_lit - ix_cache + ix] = (cache_val >> (ix * 8)) & 0xff;
      }
    }
  }

  inline __device__ void decode_literals(const int num_decode, const unsigned full_mask)
  {
    do_decode_literals(num_decode, full_mask);
  }

  inline __device__ int num_rem_literals() { return num_literals - ix_lit; }

  inline __device__ void finish_block()
  {
    __syncwarp(block_mask);

    if (ix_in_zstd_block == 0)
    {
#ifdef STAGE_LOGGING
      printf(
        "finishing huff frame %d block %d clock %lu bid %d warp %d\n",
        block_share_ptr->ix_frame,
        block_share_ptr->ix_block,
        cuda::std::chrono::system_clock::now(),
        blockIdx.x,
        ix_warp()
      );
#endif

      block_share_ptr->decode_lit_count.store(full_num_literals, cuda::std::memory_order_release);
    }
    active = false;
  }

  inline __device__ bool is_active() { return active; }
};

} // namespace zstd
