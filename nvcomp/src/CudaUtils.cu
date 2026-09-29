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

#include <cuda.h>

#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "CudaDriver.h"
#include "CudaUtils.h"
#include "exception.hpp"
#include "Logging.h"
#include "nvcomp.hpp"

namespace nvcomp
{

namespace
{
[[maybe_unused]] std::string to_string(const void *const ptr)
{
  std::ostringstream oss;
  oss << ptr;
  return oss.str();
}
} // namespace

// Note:
// is_host_pointer is expected to return true for paged, pinned, and managed allocations.
bool CudaUtils::is_host_pointer(const void *const ptr) { return !is_device_pointer(ptr); }

bool CudaUtils::is_device_pointer(const void *const ptr)
{
  cudaPointerAttributes attr;
  CUDA_CHECK(cudaPointerGetAttributes(&attr, ptr));
  return attr.type == cudaMemoryTypeDevice;
}

CUdevice CudaUtils::get_stream_device(cudaStream_t stream)
{
  CUcontext context;
  CUresult result = CudaDriver::cuStreamGetCtx_portable(stream, &context);

  if (result != CUDA_SUCCESS)
  {
    throw NVCompException(
      nvcompErrorCudaError,
      std::string("nvCOMP error: Unable to get context for stream ") +
        std::to_string(reinterpret_cast<uintptr_t>(stream))
    );
  }
  result = CudaDriver::cuCtxPushCurrent(context);
  if (result != CUDA_SUCCESS)
  {
    throw NVCompException(
      nvcompErrorCudaError,
      std::string("nvCOMP error: Unable to push context ") + std::to_string(reinterpret_cast<uintptr_t>(context)) +
        std::string(" for stream ") + std::to_string(reinterpret_cast<uintptr_t>(stream))
    );
  }

  CUdevice stream_device_handle;
  result = CudaDriver::cuCtxGetDevice(&stream_device_handle);
  if (result != CUDA_SUCCESS)
  {
    throw NVCompException(
      nvcompErrorCudaError,
      std::string("nvCOMP error: Unable to get device from context ") +
        std::to_string(reinterpret_cast<uintptr_t>(context)) + std::string(" for stream ") +
        std::to_string(reinterpret_cast<uintptr_t>(stream))
    );
  }
  result = CudaDriver::cuCtxPopCurrent(&context);
  if (result != CUDA_SUCCESS)
  {
    throw NVCompException(
      nvcompErrorCudaError,
      std::string("nvCOMP error: Unable to pop context ") + std::to_string(reinterpret_cast<uintptr_t>(context)) +
        std::string(" for stream ") + std::to_string(reinterpret_cast<uintptr_t>(stream))
    );
  }
  return stream_device_handle;
}

bool CudaUtils::can_use_async_mem_ops(cudaStream_t stream)
{
  CUdevice stream_device_handle = CudaUtils::get_stream_device(stream);
  int attribute_res_val;
  CUresult result = CudaDriver::cuDeviceGetAttribute(
    &attribute_res_val,
    CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED,
    stream_device_handle
  );
  if (result == CUDA_SUCCESS && attribute_res_val == 1)
  {
    return true;
  }
  return false;
}

int CudaUtils::get_sm_count(const int device_id)
{
  int num_sms;
  CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device_id));
  return num_sms;
}

int CudaUtils::get_sm_count([[maybe_unused]] cudaStream_t stream)
{
#if NVCOMP_HAS_GREEN_CONTEXT
  // cudaStreamGetDevResource returns the SM count visible to the stream —
  // can be smaller than the device total for green context partitions.
  if (CudaDriver::get_latest_supported_cuda_version() >= NVCOMP_GREEN_CONTEXT_MINIMUM_CRT_VERSION)
  {
    cudaDevResource res{};
    CUDA_CHECK(cudaStreamGetDevResource(stream, &res, cudaDevResourceTypeSm));
    return res.sm.smCount;
  }
#endif
  // cudaStreamGetDevResource is not valid pre 13.1
  // Use cudaGetDevice rather than get_stream_device to
  // avoid cuCtxPushCurrent/cuCtxPopCurrent corrupting an outer DeviceGuard's
  // context when called from within a DeviceGuard(stream) scope.
  int device_id;
  CUDA_CHECK(cudaGetDevice(&device_id));
  return get_sm_count(device_id);
}

bool CudaUtils::is_stream_for_device(cudaStream_t stream, int device_id)
{
  CUdevice stream_device_handle = get_stream_device(stream);
  CUdevice device_handle;
  CUresult result = CudaDriver::cuDeviceGet(&device_handle, device_id);
  if (result != CUDA_SUCCESS)
  {
    throw NVCompException(
      nvcompErrorCudaError,
      std::string("nvCOMP error: Unable to get device handle for device #") + std::to_string(device_id)
    );
  }
  return stream_device_handle == device_handle;
}

} // namespace nvcomp
