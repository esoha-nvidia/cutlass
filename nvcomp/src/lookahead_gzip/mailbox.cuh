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

#include <cuda/atomic>

#include "common.cuh"
#include "exception.hpp"

// Class used to exchange data between concurrently running thread blocks:
// - Thread blocks that decode a raw deflate block in parallel
// - Thread block that decodes a raw deflate block header
template <typename T>
class MailboxD2D
{
public:
  constexpr __host__ __device__ T base() const { return T(); }

  void init(
    cuda::atomic<size_t, cuda::thread_scope::thread_scope_device> *request_ptr,
    cuda::atomic<size_t, cuda::thread_scope::thread_scope_device> *response_ptr,
    T *data_ptr
  )
  {
    request = request_ptr;
    response = response_ptr;
    data = data_ptr;
  }

  void reset(cudaStream_t stream)
  {
    CUDA_CHECK(cudaMemsetAsync(request, 0x00, sizeof(*request), stream));
    CUDA_CHECK(cudaMemsetAsync(response, 0x00, sizeof(*response), stream));
  }

  // Don't call this too much, it can be slow.
  __device__ void set_request(size_t value) { request->store(value, cuda::std::memory_order_release); }

  // Don't call this too much, it can be slow.
  __device__ void set_response(size_t value) { response->store(value, cuda::std::memory_order_release); }

  // Don't call this too much, it can be slow.  Don't call it in a
  // loop, use wait_for_request_to_not_be.
  __device__ size_t get_request() { return request->load(cuda::std::memory_order_acquire); }

  // Don't call this too much, it can be slow.
  __device__ size_t get_response() { return response->load(cuda::std::memory_order_acquire); }

  // Don't call this too much, it can be slow.
  __device__ size_t wait_for_request_to_not_be(size_t value)
  {
    size_t new_value;
    do
    {
      new_value = get_request();
    } while (new_value == value);
    return new_value;
  }

  // Don't call this too much, it can be slow.
  __device__ size_t wait_for_response_to_be(size_t value)
  {
    size_t new_value;
    do
    {
      new_value = get_response();
    } while (new_value != value);
    return new_value;
  }

  // The data that will be sent back and forth.
  T *data;

private:
  // This is written by the requester to request new data.
  cuda::atomic<size_t, cuda::thread_scope::thread_scope_device> *request;
  // The request is copied into here by the responder to indicate that
  // the data is ready.
  cuda::atomic<size_t, cuda::thread_scope::thread_scope_device> *response;
};
