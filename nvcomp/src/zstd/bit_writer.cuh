// For each write, write into the shmem cache.
// When the cache is full, write the shmem buffer to global
// For now, for simplicity, 1-byte writes to global are performed
// 128-byte shmem allocation allows a warp to write 32 4-byte values at once
#pragma once

#include "io.cuh"
#include "utils.cuh"

namespace zstd
{

struct BitWriter
{

private:
  uint8_t *output;
  int global_byte_offset;
  int shmem_bit_offset;
  int max_size;
  static const int SHMEM_ARRAY_SIZE = 128;
  uint8_t __align__(4) shmem_buffer[SHMEM_ARRAY_SIZE];

public:
  BitWriter() = default;

  inline __device__ void init(uint8_t *output_, int max_size_)
  {
    global_byte_offset = 0;
    shmem_bit_offset = 0;
    max_size = max_size_;
    output = output_;

    // Zero all the values
    uint32_t *u32_shmem_buffer = reinterpret_cast<uint32_t *>(shmem_buffer);
    static_assert(SHMEM_ARRAY_SIZE % sizeof(uint32_t) == 0);
    for (int ix = thread_warp_ix(); ix < SHMEM_ARRAY_SIZE / sizeof(uint32_t); ix += WARP_SIZE)
    {
      u32_shmem_buffer[ix] = 0;
    }
    __syncwarp();
  }

  // Used for writing manually.
  inline __device__ uint8_t *get_output_ptr()
  {
    const int total_bit_offset = global_byte_offset * 8 + shmem_bit_offset;
    assert(total_bit_offset % 8 == 0);
    return output + total_bit_offset / 8;
  }

  inline __device__ uint8_t *check_out_byte()
  {
    const int total_bit_offset = global_byte_offset * 8 + shmem_bit_offset;
    assert(total_bit_offset % 8 == 0);
    uint8_t *res = output + total_bit_offset / 8;
    shmem_bit_offset += 8;
    return res;
  }

  inline __device__ void check_in_byte(uint8_t *loc, uint8_t val)
  {
    // Check whether this loc still in the shmem buffer
    const int loc_byte_offset = (uintptr_t)(loc - output);
    const int this_shmem_offset = loc_byte_offset - global_byte_offset;
    if (this_shmem_offset < 0)
    {
      // write the byte -- the bitwriter has already continued
      *loc = val;
    }
    else
    {
      // Otherwise, just fill it into the shmem array. It'll be added later
      shmem_buffer[this_shmem_offset] = val;
    }
  }

  // This one assumes one thread is active for the given BitWriter
  inline __device__ void write_bits(uint32_t write_val, int num_bits)
  {
    write_bits_le(reinterpret_cast<uint32_t *>(shmem_buffer), shmem_bit_offset, write_val, num_bits);
  }

  inline __device__ void write_bits(
    uint32_t write_val,
    int num_bits,
    const int ix_thread,
    const int scan_offset,
    const unsigned mask,
    int n_threads,
    bool shuffle_end = true
  )
  {
    assert(__popc(__match_any_sync(mask, n_threads)) == n_threads); // Make sure n_threads is the same for all threads
    int this_shmem_offset = scan_offset + shmem_bit_offset;
    write_bits_le(reinterpret_cast<uint32_t *>(shmem_buffer), this_shmem_offset, write_val, num_bits);

    int ix_shuffle = shuffle_end ? n_threads - 1 : 0;
    __syncwarp(mask);
    if (ix_thread == ix_shuffle)
    {
      shmem_bit_offset += scan_offset + num_bits;
    }
    __syncwarp(mask);
    if (shmem_bit_offset / 8 >= 2 * n_threads)
    {
      flush(ix_thread, mask, n_threads);
    }
  }

  inline __device__ void flush(const int ix_thread, const unsigned mask, int n_threads)
  {
    // Flush up to byte boundary
    int num_bytes = shmem_bit_offset / 8;
    if (global_byte_offset + num_bytes > max_size) [[unlikely]]
    {
      int global_byte_offset_local = global_byte_offset;
      __syncwarp(mask);
      global_byte_offset = global_byte_offset_local + num_bytes;
      shmem_bit_offset = 0;
      __syncwarp(mask);
      return;
    }

    uint8_t *global_ptr = output + global_byte_offset;

    // Write out to global
    for (int ix = ix_thread; ix < num_bytes; ix += n_threads)
    {
      global_ptr[ix] = shmem_buffer[ix];
    }
    __syncwarp(mask);

    int fill_first_byte = shmem_bit_offset % 8 != 0;
    // Fill in the first byte with the remaining bits
    if (fill_first_byte and ix_thread == 0)
    {
      shmem_buffer[0] = shmem_buffer[num_bytes];
    }
    __syncwarp(mask);

    // Zero out the previously-set bytes
    for (int ix = ix_thread + fill_first_byte; ix <= num_bytes; ix += n_threads)
    {
      shmem_buffer[ix] = 0;
    }

    if (ix_thread == 0)
    {
      global_byte_offset += num_bytes;
      shmem_bit_offset -= num_bytes * 8;
    }

    __syncwarp(mask);
  }

  inline __device__ bool check_fully_flushed() { return shmem_bit_offset == 0; }

  inline __device__ void increment_bytes(int num_bytes)
  {
    assert(shmem_bit_offset == 0);
    global_byte_offset += num_bytes;
  }

  inline __device__ void align_bytewise()
  {
    unsigned rem_bits = shmem_bit_offset % 8;
    shmem_bit_offset += rem_bits ? (8 - rem_bits) : 0;
  }

  inline __device__ unsigned compute_byte_offset(const uint8_t *compare) const
  {
    assert(shmem_bit_offset == 0);
    uint8_t *my_ptr = reinterpret_cast<uint8_t *>(output) + global_byte_offset;
    return (uintptr_t)(my_ptr - compare);
  }
};

} // end namespace zstd
