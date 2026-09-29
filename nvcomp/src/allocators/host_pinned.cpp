/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "host_pinned.hpp"

#include <cuda/memory_resource>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cassert>
#include <mutex>
#include <optional>
#include <string>

#include "CudaUtils.h"
#include "Environment.hpp"
#include "exception.hpp"
#include "Logging.h"
#include "numa_utils.hpp"

namespace nvcomp
{

namespace
{

constexpr size_t MAX_DEFAULT_PINNED_POOL_SIZE = 10 * 1024 * 1024;

// cuda::pinned_memory_pool relies on cuMemPoolCreate with
// CU_MEM_LOCATION_TYPE_HOST_NUMA, which was introduced in CUDA 12.6 and its
// corresponding drivers. Older drivers fall back to cudaHostAlloc.
constexpr int MIN_PINNED_POOL_DRIVER_VERSION = 12060;

// Fallback for when async pinned memory pools are not supported.
class BasicHostPinnedResource
{
public:
  void *allocate([[maybe_unused]] cuda::stream_ref stream, size_t bytes, size_t alignment)
  {
    return allocate_sync(bytes, alignment);
  }

  void deallocate(cuda::stream_ref stream, void *ptr, size_t bytes, size_t alignment) noexcept
  {
    // cudaFreeHost does not respect stream ordering.
    CUDA_CHECK_LOG(cudaStreamSynchronize(stream.get()));
    deallocate_sync(ptr, bytes, alignment);
  }

  void *allocate_sync(size_t bytes, [[maybe_unused]] size_t alignment)
  {
    assert(alignment <= 256);
    void *allocation = nullptr;
    CUDA_CHECK(cudaHostAlloc(&allocation, bytes, cudaHostAllocDefault));
    return allocation;
  }

  void deallocate_sync(void *ptr, size_t /*bytes*/, size_t /*alignment*/) noexcept
  {
    CUDA_CHECK_LOG(cudaFreeHost(ptr));
  }

  bool operator==(const BasicHostPinnedResource &) const { return true; }

  bool operator!=(const BasicHostPinnedResource &other) const { return !(*this == other); }

  friend constexpr void get_property(const BasicHostPinnedResource &, cuda::mr::device_accessible) noexcept {}

  friend constexpr void get_property(const BasicHostPinnedResource &, cuda::mr::host_accessible) noexcept {}
};

static_assert(cuda::mr::resource<BasicHostPinnedResource>);
static_assert(cuda::mr::resource_with<BasicHostPinnedResource, cuda::mr::device_accessible, cuda::mr::host_accessible>);

bool driver_supports_pinned_memory_pools()
{
  int driver_version = 0;
  const cudaError_t error = cudaDriverGetVersion(&driver_version);
  if (error != cudaSuccess)
  {
    LOG_INFO(
      "Unable to query the CUDA driver version: {}. Host pinned memory pools unsupported.",
      cudaGetErrorString(error)
    );
    (void)cudaGetLastError();
    return false;
  }

  if (driver_version < MIN_PINNED_POOL_DRIVER_VERSION)
  {
    LOG_INFO(
      "The installed CUDA driver supports up to CUDA {}, but host pinned memory pools require CUDA {}. "
      "Host pinned memory pools unsupported.",
      driver_version,
      MIN_PINNED_POOL_DRIVER_VERSION
    );
    return false;
  }

  return true;
}

host_device_async_resource_ref make_basic_pinned_mr()
{
  static cuda::mr::shared_resource<BasicHostPinnedResource> mr{cuda::std::in_place_type<BasicHostPinnedResource>};
  return host_device_async_resource_ref{mr};
}

size_t get_pinned_pool_size()
{
  const auto env_var_value = nvcomp::getenv(PINNED_POOL_SIZE_ENV);
  if (!env_var_value.empty())
  {
    try
    {
      return static_cast<size_t>(std::stoull(env_var_value));
    }
    catch (const std::exception &e)
    {
      LOG_ERROR(
        "Failed to parse environment variable {}='{}': {}. Falling back to default pinned pool size.",
        PINNED_POOL_SIZE_ENV,
        env_var_value,
        e.what()
      );
    }
  }

  size_t free_memory = 0;
  size_t total_memory = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_memory, &total_memory));

  // 0.5% of the total device memory, capped at 10 MiB.
  return std::min(total_memory / 200, MAX_DEFAULT_PINNED_POOL_SIZE);
}

cuda::memory_pool_properties get_pinned_pool_properties()
{
  const size_t pinned_pool_size = get_pinned_pool_size();
  cuda::memory_pool_properties properties{};
  properties.initial_pool_size = pinned_pool_size;
  properties.release_threshold = 2 * pinned_pool_size;
  properties.max_pool_size = 0; // No explicit limit.
  return properties;
}

} // namespace

host_device_async_resource_ref make_default_pinned_mr()
{
  do
  {
    // CCCL already does extensive checks to see whether async memory allocations and the like are supported.
    // Hence only checking driver support here.
    if (!driver_supports_pinned_memory_pools())
    {
      break;
    }

    try
    {
      // Gather pool properties
      const auto properties = get_pinned_pool_properties();
      // Query current NUMA node
      int numa_node = get_current_numa_node();

      // The singleton pool is placed on the NUMA node local to the thread that
      // first initializes the default pinned-memory resource.
      static cuda::mr::shared_resource<cuda::pinned_memory_pool>
        mr{cuda::std::in_place_type<cuda::pinned_memory_pool>, numa_node, properties};
      return host_device_async_resource_ref{mr};
    }
    catch (const std::exception &e)
    {
      LOG_ERROR(
        "Failed to create default pinned memory resource: {}. Falling back to basic pinned memory resource.",
        e.what()
      );
    }
  } while (0);
  return make_basic_pinned_mr();
}

host_device_async_resource_ref host_mr()
{
  static std::mutex mr_lock;
  std::lock_guard lock{mr_lock};

  static std::optional<host_device_async_resource_ref> mr_ref;
  if (not mr_ref.has_value())
  {
    mr_ref = make_default_pinned_mr();
  }
  return *mr_ref;
}

host_device_async_resource_ref get_pinned_memory_resource() { return host_mr(); }

} // namespace nvcomp
