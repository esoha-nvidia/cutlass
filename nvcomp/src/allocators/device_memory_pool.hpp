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

#include <cuda_runtime.h>

#include <vector>

// Requires CudaDriver.h, device_guard.h, and exception.hpp to be included
// by the consumer before this header. This is necessary because python/ and
// src/ maintain separate copies of CudaDriver.h and device_guard.h in
// different namespaces (nvcomp::python vs nvcomp).
#include "exception.hpp"

namespace nvcomp
{

#if defined(_WIN32)
//  Returns non-zero once the OS has begun process-exit DLL teardown
//  (LdrShutdownProcess set its internal flag).
//  Used to skip cleanup paths that would call into already-detached DLLs.
extern "C" __declspec(dllimport) unsigned char __stdcall RtlDllShutdownInProgress();
#endif

/**
 * @brief Per-device CUDA memory pool that is hardware decompression engine
 * compatible when the device supports it.
 */
struct DeviceMemoryPool
{
  DeviceMemoryPool()
      : pools_()
  {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    pools_.resize(device_count);

    for (int device_id = 0; device_id < device_count; ++device_id)
    {
      cudaMemPoolProps props = {};
      props.location.type = cudaMemLocationTypeDevice;
      props.location.id = device_id;
      props.allocType = cudaMemAllocationTypePinned;
      props.usage = 0;

      int decompress_support_mask = 0;
      auto result = CudaDriver::cuDeviceGetAttribute(
        &decompress_support_mask,
        CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_ALGORITHM_MASK,
        device_id
      );
      if (result == CUDA_SUCCESS && decompress_support_mask)
      {
        props.usage = cudaMemPoolCreateUsageHwDecompress;
      }

      CUDA_CHECK(cudaMemPoolCreate(&pools_[device_id], &props));
    }
  }

  ~DeviceMemoryPool()
  {
#if defined(_WIN32)
    if (RtlDllShutdownInProgress())
    {
      return;
    }
#endif
    for (auto &pool : pools_)
    {
      CUDA_CHECK_LOG(cudaMemPoolDestroy(pool));
    }
  }

  DeviceMemoryPool(const DeviceMemoryPool &) = delete;
  DeviceMemoryPool &operator=(const DeviceMemoryPool &) = delete;

  void *allocate(size_t size, cudaStream_t stream) const
  {
    DeviceGuard device_guard{stream};
    int device_id;
    CUDA_CHECK(cudaGetDevice(&device_id));

    void *ptr;
    CUDA_CHECK(cudaMallocFromPoolAsync(&ptr, size, pools_[device_id], stream));
    return ptr;
  }

private:
  std::vector<cudaMemPool_t> pools_;
};

/**
 * @brief Returns the singleton DeviceMemoryPool instance.
 */
inline DeviceMemoryPool &get_device_memory_pool()
{
  static DeviceMemoryPool pool;
  return pool;
}

} // namespace nvcomp
