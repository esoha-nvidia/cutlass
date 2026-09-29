/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <algorithm>
#include <chrono>
#include <istream>
#include <thread>

#include "atomics.cuh"
#include "constants.cuh"
#include "exception.hpp"
#include "include/types.h"

#include <stdint.h>

using namespace lookahead_gzip;

// Ring buffer classes, only used in streaming mode

template <typename T>
class BufferH2D
{
public:
  void init(
    T *const *buffer_ptr,
    cuda::atomic<size_t, cuda::thread_scope_system> *read_position,
    cuda::atomic<size_t, cuda::thread_scope_system> *write_position,
    cudaStream_t prefetch_stream
  )
  {
    this->buffer = *buffer_ptr;
    this->read_position = read_position;
    this->write_position = write_position;
    this->prefetch_stream = prefetch_stream;
    host_read_position = 0;
    host_write_position = 0;
  }

  constexpr __host__ __device__ size_t get_size() const { return INPUT_RING_BUFFER_SIZE; }

  constexpr __host__ __device__ T base() const { return T(); }

  inline __device__ T operator[](size_t index) const { return buffer[index % INPUT_RING_BUFFER_SIZE]; }

  inline __host__ __device__ T at(size_t index) const { return buffer[index % INPUT_RING_BUFFER_SIZE]; }

  // Sync the host thread with the prefetch stream
  inline __host__ void sync_with_stream() const { CUDA_CHECK(cudaStreamSynchronize(prefetch_stream)); }

  // Try not too call this one too much because it's slow.
  inline __device__ void set_read_position(size_t new_read_position)
  {
    read_position->store(new_read_position, cuda::std::memory_order_release);
    // To wake up all cuda::atomic::wait.  However, cuda notify_all is
    // currently a no-op and cuda wait is spin-waiting with backoff.
    // So we don't use either of them.

    // read_position->notify_all();
  }

  // Try not too call this one too much because it's slow.
  inline __device__ size_t get_write_position() const { return write_position->load(cuda::std::memory_order_acquire); }

  // Try not too call this one too much because it's slow.
  inline __host__ void set_write_position(size_t new_write_position) const
  {
    write_position->store(new_write_position, cuda::std::memory_order_release);
  }

  // Called when the host finishes writing to the buffer.
  // Sets the most significatnt byte to 1 causing the write position to be always bigger than buffer itself.
  inline __host__ void set_finished_writing_flag()
  {
    const size_t numBits = sizeof(host_write_position) * 8;
    const size_t msbMask = static_cast<size_t>(1) << (numBits - 1);

    size_t new_host_write_position = host_write_position | msbMask;
    write_position->store(new_host_write_position, cuda::std::memory_order_release);
  }

  // Read bytes from the input stream.
  // Note: write_position always satisfies (write_position % size/2) == 0.
  template <bool WAIT_FOR_SPACE>
  inline __host__ size_t append(std::istream &in_stream, int device_id, std::atomic<bool> &notification)
  {
    // Note:
    // Only wait if in_stream still contains data
    // Not doing so, we might end up waiting infinitely,
    // hanging the binary.
    if (!WAIT_FOR_SPACE || in_stream.peek() != std::ifstream::traits_type::eof())
    {
      update_read_position<WAIT_FOR_SPACE>(notification);
    }
    auto size = INPUT_RING_BUFFER_SIZE;
    size_t to_write = size - (host_write_position - get_read_position());
    //                       ^ amount of data unprocessed by GPU
    assert(size % 2 == 0);
    if (to_write < size / 2)
    {
      // Note: we are always writing size/2, irrespective of how much
      //       data is available
      return 0;
    }
    cudaMemLocation cpu_location{cudaMemLocationTypeHost, 0};
    CUDA_CHECK(
      cudaMemPrefetchAsync(&buffer[host_write_position % size], size / 2 * sizeof(T), cpu_location, 0, prefetch_stream)
    );
    in_stream.read(reinterpret_cast<char *>(&buffer[host_write_position % size]), size / 2 * sizeof(T));
    auto written = in_stream.gcount();
    // A zero-length prefetch is rejected by the CUDA runtime, so only prefetch to the GPU when we read data.
    if (written > 0)
    {
      cudaMemLocation gpu_location{cudaMemLocationTypeDevice, device_id};
      CUDA_CHECK(cudaMemPrefetchAsync(&buffer[host_write_position % size], written, gpu_location, 0, prefetch_stream));
    }

    // Note: before updating atomics, we must make sure
    //       that the actual prefetching finished, as atomics
    //       are not stream-ordered
    // TODO(bnagy): add the stream ordering host callback here (cudaLaunchHostFunc)
    sync_with_stream();

    if (in_stream.eof())
    {
      // We're going to pretend that everything is valid when we hit
      // the end so that the reader can fearlessly read off the end of
      // the file.
      host_write_position = get_read_position() + size;
      set_finished_writing_flag();
    }
    else
    {
      host_write_position += written / sizeof(T);
      set_write_position(host_write_position);
    }

    return written;
  }

  inline __host__ void reset()
  {
    read_position->store(0, cuda::std::memory_order_release);
    host_read_position = 0;
    host_write_position = 0;
    set_write_position(0);
  }

  T *buffer;

private:
  // Try not to call this one too much because it's slow.
  template <bool WAIT_FOR_SPACE>
  inline __host__ void update_read_position(std::atomic<bool> &notification)
  {
    if constexpr (WAIT_FOR_SPACE)
    {
      host_read_position =
        atomic_wait(read_position, host_read_position, cuda::std::memory_order_acquire, 10U, notification, 1000);
    }
    else
    {
      host_read_position = read_position->load(cuda::std::memory_order_acquire);
    }
  }

  inline __host__ size_t get_read_position() const { return host_read_position; }

  // CUDA stream used to manipulate read/write buffers
  cudaStream_t prefetch_stream;

  // A local copy of the write_position below.  Reading this may be
  // faster than accessing the one below so we keep a local copy.
  // Note: this is a linear position, does not wrap around
  size_t host_write_position;

  // When write_position and read_position are equal, that indicates
  // there there buffer is empty, not full.
  // Note: this is a linear position, does not wrap around
  cuda::atomic<size_t, cuda::thread_scope_system> *write_position;

  // A local copy of the read_position below.  Reading this may be
  // faster than accessing the one below so we keep a local copy.
  // Note: this is a linear position, dores not wrap around
  size_t host_read_position;

  // Where the next read will go.  This must be less than size.
  // Note: this is a linear position, does not wrap around
  cuda::atomic<size_t, cuda::thread_scope_system> *read_position;
};

template <typename T>
class BufferD2H
{
public:
  constexpr __host__ __device__ size_t get_size() const { return OUTPUT_RING_BUFFER_SIZE; }

  constexpr __host__ __device__ T base() const { return T(); }

  void init(
    T *const *buffer_ptr,
    cuda::atomic<size_t, cuda::thread_scope_system> *read_position,
    cuda::atomic<size_t, cuda::thread_scope_system> *write_position,
    cudaStream_t prefetch_stream
  )
  {
    this->buffer = *buffer_ptr;
    this->read_position = read_position;
    this->write_position = write_position;
    this->prefetch_stream = prefetch_stream;
    host_write_position = 0;
  }

  // Get one element from the buffer.
  inline __host__ __device__ T &operator[](size_t index) { return buffer[index % OUTPUT_RING_BUFFER_SIZE]; }

  // Get one element from the buffer.
  inline const __host__ __device__ T &operator[](size_t index) const { return buffer[index % OUTPUT_RING_BUFFER_SIZE]; }

  // Get one element from the buffer.
  inline __host__ __device__ T &at(size_t index) { return buffer[index % OUTPUT_RING_BUFFER_SIZE]; }

  // Get one element from the buffer.
  inline const __host__ __device__ T &at(size_t index) const { return buffer[index % OUTPUT_RING_BUFFER_SIZE]; }

  // Sync the host thread with the prefetch stream
  inline __host__ void sync_with_stream() const { CUDA_CHECK(cudaStreamSynchronize(prefetch_stream)); }

  // Try not too call this one too much because it's slow.
  inline __host__ void set_read_position(size_t new_read_position)
  {
    read_position->store(new_read_position, cuda::std::memory_order_release);
  }

  // Try not to call this one too much because it's slow.
  inline __device__ size_t get_read_position() const { return read_position->load(cuda::std::memory_order_acquire); }

  // Try not too call this one too much because it's slow.
  template <bool WAIT_FOR_SPACE>
  inline __host__ void update_write_position()
  {
    if constexpr (WAIT_FOR_SPACE)
    {
      host_write_position = atomic_wait(write_position, host_write_position, cuda::std::memory_order_acquire, 10U);
    }
    else
    {
      host_write_position = write_position->load(cuda::std::memory_order_acquire);
    }
  }

  inline __host__ size_t get_write_position() const { return host_write_position; }

  // Try not too call this one too much because it's slow.
  inline __device__ void set_write_position(size_t new_write_position) const
  {
    write_position->store(new_write_position, cuda::std::memory_order_release);
    // To wake up all cuda::atomic::wait.  However, cuda notify_all is
    // currently a no-op and cuda wait is spin-waiting with backoff.
    // So we don't use either of them.

    // write_position->notify_all();
  }

  inline __host__ void prefetch_to_cpu(size_t start, size_t count) const
  {
    cudaMemLocation cpu_location{cudaMemLocationTypeHost, 0};
    while (count > 0)
    {
      size_t elements_until_end = OUTPUT_RING_BUFFER_SIZE - (start % OUTPUT_RING_BUFFER_SIZE);
      size_t elements_to_prefetch = std::min(count, elements_until_end);
      CUDA_CHECK(cudaMemPrefetchAsync(
        &buffer[start % OUTPUT_RING_BUFFER_SIZE],
        elements_to_prefetch * sizeof(T),
        cpu_location,
        0,
        prefetch_stream
      ));
      count -= elements_to_prefetch;
      start += elements_to_prefetch;
    }
  }

  inline __host__ void prefetch_to_gpu(size_t start, size_t count, int device_id) const
  {
    cudaMemLocation gpu_location{cudaMemLocationTypeDevice, device_id};
    while (count > 0)
    {
      size_t elements_until_end = OUTPUT_RING_BUFFER_SIZE - (start % OUTPUT_RING_BUFFER_SIZE);
      size_t elements_to_prefetch = std::min(count, elements_until_end);
      CUDA_CHECK(cudaMemPrefetchAsync(
        &buffer[start % OUTPUT_RING_BUFFER_SIZE],
        elements_to_prefetch * sizeof(T),
        gpu_location,
        0,
        prefetch_stream
      ));
      count -= elements_to_prefetch;
      start += elements_to_prefetch;
    }
  }

  inline __device__ void set_done(size_t done_size) const { set_write_position(0xf000000000000000ULL | done_size); }

  inline __host__ bool is_done() const { return (get_write_position() & 0xf000000000000000ULL) != 0; }

  inline __host__ size_t get_done_position() const { return get_write_position() & ~0xf000000000000000ULL; }

  inline __device__ void set_error() const { set_write_position(0xffffffffffffffffULL); }

  inline __host__ bool is_error() const { return get_write_position() == 0xffffffffffffffffULL; }

  inline __host__ void reset()
  {
    write_position->store(0, cuda::std::memory_order_relaxed);
    host_write_position = 0;
    set_read_position(0);
  }

  T *buffer;

private:
  cudaStream_t prefetch_stream;
  // A local copy of the write_position below.  Reading this may be
  // faster than accessing the one below so we keep a local copy.
  size_t host_write_position;

  // When write_position and read_position are equal, that indicates
  // there there buffer is empty, not full.
  cuda::atomic<size_t, cuda::thread_scope_system> *write_position;

  // Where the next read will go.  This must be less than size.
  cuda::atomic<size_t, cuda::thread_scope_system> *read_position;
};
