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
#include "timing_reporter.cuh"

// Class that makes sure that the data in the deflate stream
// is indeed available before accessing it on the device
template <bufferType_t BufferType>
class InputFetcher
{};

// InputFetcher specialization for the streaming use case
// (data IS NOT readily available)
template <>
class InputFetcher<bufferType_t::RING>
{
public:
  inline __device__ InputFetcher(BufferH2D<uint32_t> &&deflate_stream, uint min_input_bytes)
      : deflate_stream(deflate_stream)
      , min_input_bytes(min_input_bytes)
  {
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
      // We try to call this function as infrequently as possible
      // because it's slow.
      deflate_stream_write_position = deflate_stream.get_write_position();
    }
  }

  inline __device__ void wait_for_input(size_t current_offset_bits)
  {
    timing_reporter.toggle(TimingReporter::Timer::WAIT_FOR_INPUT);
    // First, do we want to update the CPU about the read position
    // in the input deflate stream.  We'll only do it when we know
    // that it matters, that is, it causes there to be at least
    // half the buffer empty.
    size_t new_read_position = current_offset_bits / (8 * sizeof(deflate_stream.base()));

    if (deflate_stream_write_position <= deflate_stream.get_size() / 2 + new_read_position &&
        !already_set_read_position)
    {
      // Note: this is the only place where the read position is updated
      deflate_stream.set_read_position(new_read_position);
      already_set_read_position = true;
    }

    // Now we check if there are so few input bytes available that
    // we can't even process anything.  We'll spin wait if we must.
    // When host writer finishes, write position will have its most
    // significant bit set to 1, causing this loop to terminate.
    const uint read_margin = (min_input_bytes + sizeof(deflate_stream.base()) - 1) / sizeof(deflate_stream.base());
    while (deflate_stream_write_position < new_read_position + read_margin)
    {
      // Fetch the latest and try again.  We try to call this
      // function as infrequently as possible because it's slow.
      deflate_stream_write_position = deflate_stream.get_write_position();
      already_set_read_position = false;
    }
    timing_reporter.toggle(TimingReporter::Timer::WAIT_FOR_INPUT);
  }

  // Note: the index can be linear, even if we are accessing a
  //       ring buffer under the hood.
  inline __device__ uint32_t operator[](size_t index) const { return deflate_stream[index]; }

  inline __device__ size_t get_size() const { return deflate_stream.get_size(); }

  constexpr __device__ bufferType_t get_buffer_type() const { return bufferType_t::RING; }

private:
  // Until where the host wrote data (exclusive, in elements)
  // Note: valid only on (blockIdx.x, threadIdx.x) = (0, 0)
  size_t deflate_stream_write_position;

  bool already_set_read_position = false;
  BufferH2D<uint32_t> deflate_stream;
  uint min_input_bytes;
};

// InputFetcher specialization for the one-shot use case
// (data IS readily available)
template <>
class InputFetcher<bufferType_t::LINEAR>
{
public:
  inline __device__ InputFetcher(const uint32_t *device_compressed_ptr, const size_t &device_compressed_bytes)
      : device_compressed_ptr(device_compressed_ptr)
      , device_compressed_bytes(device_compressed_bytes)
  {}

  inline __device__ uint32_t operator[](size_t index) const { return device_compressed_ptr[index]; }

  inline __device__ size_t get_size() const { return device_compressed_bytes; }

  constexpr __device__ bufferType_t get_buffer_type() const { return bufferType_t::LINEAR; }

private:
  const uint32_t *device_compressed_ptr;
  const size_t device_compressed_bytes;
};
