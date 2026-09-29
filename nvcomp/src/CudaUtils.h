/*
 * Copyright (c) 2019-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>
#include <type_traits>

// Green context (SM partitioning) runtime APIs are available from CTK 13.1.
// Use this macro to guard any code that creates or queries green context streams.
#define NVCOMP_GREEN_CONTEXT_MINIMUM_CRT_VERSION 13010
#define NVCOMP_HAS_GREEN_CONTEXT (CUDART_VERSION >= NVCOMP_GREEN_CONTEXT_MINIMUM_CRT_VERSION)

namespace nvcomp
{

// TODO(mpayrits): Decide on whether to use utilities in this file more frequently
// or prune it down. Functions like copy and copy_async aren't used anywhere. For
// starters, the CopyDirection enum seems unnecessary.

enum CopyDirection
{
  HOST_TO_DEVICE = cudaMemcpyHostToDevice,
  DEVICE_TO_HOST = cudaMemcpyDeviceToHost,
  DEVICE_TO_DEVICE = cudaMemcpyDeviceToDevice
};

class CudaUtils
{
public:
  static void sync(cudaStream_t stream);

  /**
   * @brief Perform checked asynchronous memcpy.
   *
   * @tparam T The data type.
   * @param dst The destination address.
   * @param src The source address.
   * @param count The number of elements to copy.
   * @param kind The direction of the copy.
   * @param stream THe stream to operate on.
   */
  template <typename T>
  static void
  copy_async(T *const dst, const T *const src, const size_t count, const CopyDirection kind, cudaStream_t stream)
  {
    check(
      cudaMemcpyAsync(dst, src, sizeof(T) * count, static_cast<cudaMemcpyKind>(kind), stream),
      "CudaUtils::copy_async(dst, src, count, kind, stream)"
    );
  }

  /**
   * @brief Perform a synchronous memcpy.
   *
   * @tparam T The data type.
   * @param dst The destination address.
   * @param src The source address.
   * @param count The number of elements to copy.
   * @param kind The direction of the copy.
   */
  template <typename T>
  static void copy(T *const dst, const T *const src, const size_t count, const CopyDirection kind)
  {
    check(
      cudaMemcpy(dst, src, sizeof(T) * count, static_cast<cudaMemcpyKind>(kind)),
      "CudaUtils::copy(dst, src, count, kind)"
    );
  }

  static bool is_host_pointer(const void *ptr);

  static bool is_device_pointer(const void *ptr);

  // This throws NVCompException on error
  static bool is_stream_for_device(cudaStream_t stream, int device_id);

  // This throws NVCompException on error
  static CUdevice get_stream_device(cudaStream_t stream);

  // Check if the device (and current driver) associated with the given stream
  // can use asynchronous memory allocations and deallocations
  static bool can_use_async_mem_ops(cudaStream_t stream);

  // returns the SM count of the specified device id
  static int get_sm_count(int device_id);

  // returns the SM count of the specified device stream
  // this call does not alter the currently set device and requires that stream
  // is associated with the currently active device context
  static int get_sm_count(cudaStream_t stream);
};

template <typename FnT>
cudaError_t cudaLaunchHostLambda(cudaStream_t stream, FnT &&lambda, bool force_sync)
{
  // FnT&& is a forwarding reference: FnT deduces to T& for lvalues, T for rvalues.
  // We require an rvalue so the lambda won't be accidentally copied.
  static_assert(std::is_rvalue_reference_v<decltype(lambda)>, "Must be called with an rvalue (use std::move)");

  // The callback wrapper can only recover the callable via void*, so the callable
  // must be invocable with no arguments (all arguments should be passed via captures).
  static_assert(std::is_invocable_v<std::decay_t<FnT>>, "Lambda must be callable with no arguments");

  if (force_sync)
  {
    auto status = cudaStreamSynchronize(stream);
    if (status != cudaSuccess)
    {
      return status;
    }
    lambda();
    return cudaSuccess;
  }

  // Move-construct the lambda onto the heap so it outlives this scope.
  // cudaLaunchHostFunc will invoke the callback asynchronously on the host.
  auto *fn = new std::decay_t<FnT>(std::forward<FnT>(lambda));

  auto status = cudaLaunchHostFunc(
    stream,
    [](void *userData) {
      // Recover the typed lambda from the type-erased pointer
      auto *f = static_cast<std::decay_t<FnT> *>(userData);
      (*f)();
      delete f;
    },
    fn
  );

  if (status != cudaSuccess)
  {
    delete fn;
  }
  return status;
}

} // namespace nvcomp
