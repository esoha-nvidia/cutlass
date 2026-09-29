/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

#include <cuda/memory_resource>

#include <cstddef>
#include <iostream>
#include <type_traits>

#ifdef USE_NVTX
#include <nvtx3/nvtx3.hpp>
#endif

#include "nvcomp.h"

namespace nvcomp
{

using host_device_async_resource_ref =
  typename cuda::mr::resource_ref<cuda::mr::host_accessible, cuda::mr::device_accessible>;

/**
 * @brief Get the resource being used for pinned memory allocations.
 *
 * @return The resource used for pinned allocations
 */
host_device_async_resource_ref get_pinned_memory_resource();

/**
 * @brief RAII class for maintaining host pinned allocations in a scope
 */
class HostPinnedGuard
{
public:
  HostPinnedGuard(size_t size, size_t alignment, cudaStream_t stream)
      : size(size)
      , alignment(alignment)
      , stream(stream)
      , ptr(nullptr)
  {
    auto mr = get_pinned_memory_resource();
    {
#ifdef USE_NVTX
      nvtx3::scoped_range pinned_alloc{"Pinned async allocation"};
#endif // USE_NVTX
      ptr = mr.allocate(stream, size, alignment);
    }
  }

  ~HostPinnedGuard()
  {
    try
    {
      get_pinned_memory_resource().deallocate(stream, ptr, size, alignment);
    }
    catch (const std::exception &e)
    {
      std::cerr << "HostPinnedGuard: failed to deallocate: " << e.what() << std::endl;
    }
    catch (...)
    {
      // catch-all for coverity
      std::cerr << "HostPinnedGuard: failed to deallocate: unknown exception" << std::endl;
    }
  }

  void *get_ptr() const { return ptr; }

  // Deleted methods
  HostPinnedGuard(const HostPinnedGuard &) = delete;
  HostPinnedGuard(HostPinnedGuard &&) = delete;
  HostPinnedGuard &operator=(const HostPinnedGuard &) = delete;
  HostPinnedGuard &operator=(HostPinnedGuard &&) = delete;

private:
  size_t size;
  size_t alignment;
  cudaStream_t stream;
  void *ptr;
};

} // namespace nvcomp
