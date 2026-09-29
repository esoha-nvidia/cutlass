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

#include <cuda.h>
#include <cuda/atomic>

#include <iostream>
#include <new>
#include <stdexcept>
#include <string>

#include "exception.hpp"

// Class meant to hunt down data races in device code
// between host and device that are otherwise not easily detectable
class RaceHelper
{
public:
  __host__ RaceHelper(const char *identifier, int device_id)
      : m_last_position(0)
      , m_ring_buffer(nullptr)
      , m_position(nullptr)
      , d_reduction_buffer(nullptr)
      , m_identifier(identifier)
  {

    // Making sure what we are about to do is supported
    int concurrent_managed_access;
    CUDA_CHECK(cudaDeviceGetAttribute(&concurrent_managed_access, cudaDevAttrConcurrentManagedAccess, device_id));
    if (concurrent_managed_access != 1)
    {
      throw std::runtime_error("Concurrent managed access between host and device is not supported");
    }

    // Note: The host should be able to pick up elements before a wrap around happens
    //       on the device side.
    CUDA_CHECK(cudaMallocManaged(&m_position, sizeof(cuda::atomic<size_t, cuda::thread_scope_system>) * 1));
    CUDA_CHECK(cudaMallocManaged(&m_ring_buffer, sizeof(size_t) * N));

    m_position->store(0, cuda::std::memory_order_release);

    // Additional buffer space for on-device debugging
    CUDA_CHECK(cudaMalloc(&d_reduction_buffer, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_reduction_buffer, 0x00, sizeof(uint32_t)));
  }

  __host__ ~RaceHelper(void) noexcept
  {
    try
    {
      CUDA_CHECK(cudaFree(m_position));
      CUDA_CHECK(cudaFree(m_ring_buffer));
      CUDA_CHECK(cudaFree(d_reduction_buffer));
    }
    catch (std::runtime_error &e)
    {
      std::cerr << "ERROR: Unsuccessful device memory free-up within RaceHelper(" << m_identifier << ")" << std::endl;
      std::cerr << e.what() << std::endl;
    }
  }

  RaceHelper(const RaceHelper &other) noexcept = delete;
  RaceHelper(RaceHelper &&other) noexcept = delete;
  RaceHelper &operator=(const RaceHelper &) noexcept = delete;

  __host__ void print_data_if_changed(void) noexcept
  {
    size_t last_device_position = m_position->load(cuda::std::memory_order_acquire);
    if (last_device_position != m_last_position)
    {
      // we can print up to but excluding tmp
      std::cerr << m_identifier << ": ";
      while (m_last_position != last_device_position)
      {
        std::cerr << m_ring_buffer[m_last_position] << " ";
        m_last_position = (m_last_position + 1) % N;
      }
      std::cerr << std::endl;
    }
  }

  __device__ uint32_t *get_reduction_buffer(void) noexcept { return d_reduction_buffer; }

  // May be called from a single thread
  __device__ void add(size_t data)
  {
    size_t pos = m_position->load(cuda::std::memory_order_relaxed);
    m_ring_buffer[pos] = data;
    m_position->store((pos + 1) % N, cuda::std::memory_order_release);
  }

#if defined(__CUDACC__)
  // May be called from the grid
  __device__ void add_block0_thread0(size_t data)
  {
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
      add(data);
    }
  }
#endif
private:
  // Configuration
  // Note: the ring buffer MUST be large enough, so that the host can read out
  //       all elements before the buffer wraps around on the device side.
  static constexpr int N = 1024 * 1024;

  // Host-only read cursor: the next position where the host will read
  size_t m_last_position;

  // Shared counter for synchronization between host and device
  size_t *m_ring_buffer;
  cuda::atomic<size_t, cuda::thread_scope_system> *m_position;

  // Various global buffers for debugging
  // Buffer for sum reduction
  uint32_t *d_reduction_buffer;

  // In case there are multiple instances of RaceHelper
  std::string m_identifier;
};
