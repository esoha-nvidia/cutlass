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

#include "ring_buffer.cuh"

// Class that handles the uncompressed stream write-out
template <bufferType_t BufferType>
class OutputStream
{};

// OutputStream specialization for the streaming use case
// (space is not readily available)
template <>
class OutputStream<bufferType_t::RING>
{
public:
  inline __device__ OutputStream(BufferD2H<char> &&uncompressed_stream)
      : uncompressed_stream(uncompressed_stream)
  {
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
      // We try to call this function as infrequently as possible
      // because it's slow.
      uncompressed_stream_read_position = uncompressed_stream.get_read_position();
    }
  }

  // new_write_position is the first byte that the GPU might write.
  // It will never write bytes before it.  end_write_position is the
  // byte beyond the last byte that the GPU might write.  It may need
  // to write bytes up to that position.  It will never write bytes
  // beyond it.
  inline __device__ void wait_for_room(size_t new_write_position, size_t end_write_position)
  {
    timing_reporter.toggle(TimingReporter::Timer::WAIT_FOR_ROOM);
    // We only want to write the new position when it matters,
    // which is when the buffer is at least half full because
    // the CPU will only empty it if it's at least half full.
    if (new_write_position >= uncompressed_stream_read_position + uncompressed_stream.get_size() / 2 &&
        !already_set_write_position)
    {
      uncompressed_stream.set_write_position(new_write_position);
      already_set_write_position = true;
    }
    // Now we check if there is room enough in the output stream
    // to write all the uncompressed bytes from the uncompressed
    // words.  If not, we'll spin wait.
    while (uncompressed_stream.get_size() + uncompressed_stream_read_position < end_write_position)
    {
      uncompressed_stream_read_position = uncompressed_stream.get_read_position();
      already_set_write_position = false;
    }
    timing_reporter.toggle(TimingReporter::Timer::WAIT_FOR_ROOM);
  }

  inline __device__ void set_done(size_t end_position) const { uncompressed_stream.set_done(end_position); }

  inline __device__ void set_error() const { uncompressed_stream.set_error(); }

  inline __device__ char &operator[](size_t index) { return uncompressed_stream[index]; }

private:
  size_t uncompressed_stream_read_position = 0;
  bool already_set_write_position = false;
  BufferD2H<char> uncompressed_stream;
};

// OutputStream specialization for the one-shot use case
// (space IS readily available)
template <>
class OutputStream<bufferType_t::LINEAR>
{
public:
  inline __device__ OutputStream(
    char *device_uncompressed_ptr,
    const size_t &device_uncompressed_buffer_bytes,
    size_t *device_uncompressed_chunk_bytes
  )
      : device_uncompressed_ptr(device_uncompressed_ptr)
      , device_uncompressed_buffer_bytes(device_uncompressed_buffer_bytes)
      , device_uncompressed_chunk_bytes(device_uncompressed_chunk_bytes)
  {}

  inline __device__ char &operator[](size_t index) { return device_uncompressed_ptr[index]; }

  inline __device__ void set_done(size_t end_position)
  {
    if (device_uncompressed_chunk_bytes)
    {
      *device_uncompressed_chunk_bytes = end_position;
    }
  }

  inline __device__ void set_error()
  {
    if (device_uncompressed_chunk_bytes)
    {
      // Following suit with BufferD2H<>, although there is no restriction
      *device_uncompressed_chunk_bytes = 0xffffffffffffffffULL;
    }
  }

  // Note:
  // last_byte denotes the index 1 after the last index to be written
  inline __device__ bool can_fit(size_t last_byte) { return last_byte <= device_uncompressed_buffer_bytes; }

private:
  char *device_uncompressed_ptr;
  const size_t &device_uncompressed_buffer_bytes;
  size_t *device_uncompressed_chunk_bytes;
};
