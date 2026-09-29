/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuda.h>

#include "CudaDriver.h"
#include "exception.hpp"

namespace nvcomp
{

// TODO: Combine python/device_guard.h with this in an nvcomp_utils
//       intermediate (object) library to mitigate code duplication

/**
 * Simple RAII device handling:
 * Switch to new device on construction, back to old
 * device on destruction
 */
class DeviceGuard
{
public:
  /// @brief Saves current device id and restores it upon object destruction
  DeviceGuard()
      : old_context_(NULL)
      , old_device_(-1)
  {
    CU_CHECK(CudaDriver::cuCtxGetCurrent(&old_context_));
    CUDA_CHECK(cudaGetDevice(&old_device_));
  }

  /// @brief Saves current device id, sets a new one from the given integer and switches back
  ///        to the original device upon object destruction.
  ///
  /// @note For device id < 0, it is no-op
  explicit DeviceGuard(int new_device)
      : old_context_(NULL)
      , old_device_(-1)
  {
    if (new_device >= 0)
    {
      CU_CHECK(CudaDriver::cuCtxGetCurrent(&old_context_));
      CUDA_CHECK(cudaGetDevice(&old_device_));
      CUDA_CHECK(cudaSetDevice(new_device));
    }
  }

  /// @brief Saves current device id, sets a new one from user stream and switches back
  ///        to the original device upon object destruction.
  explicit DeviceGuard(cudaStream_t stream)
      : old_context_(NULL)
      , old_device_(-1)
  {
    CU_CHECK(CudaDriver::cuCtxGetCurrent(&old_context_));
    CUDA_CHECK(cudaGetDevice(&old_device_));
    CUcontext stream_ctx{};
    CU_CHECK(CudaDriver::cuStreamGetCtx_portable(stream, &stream_ctx));
    CU_CHECK(CudaDriver::cuCtxSetCurrent(stream_ctx));
  }

  DeviceGuard(const DeviceGuard &) = delete;
  DeviceGuard &operator=(const DeviceGuard &) = delete;
  DeviceGuard(DeviceGuard &&) = delete;
  DeviceGuard &operator=(DeviceGuard &&) = delete;

  ~DeviceGuard()
  {
    try
    {
      if (old_device_ >= 0)
      {
        // Restore device
        // Note: this might have the side-effect that it creates a primary context for
        //       the original device
        cudaError_t err = cudaSetDevice(old_device_);
        if (err != cudaSuccess)
        {
          std::cerr << "Failed to recover the runtime API state via cudaSetDevice(): " << err << std::endl;
        }

        // Restore context
        // Note: NULL is accepted by cuCtxSetCurrent
        CUresult err2 = CudaDriver::cuCtxSetCurrent(old_context_);
        if (err2 != CUDA_SUCCESS)
        {
          std::cerr << "Failed to recover previous context via cuCtxSetCurrent(): " << err2 << std::endl;
        }
      }
    }
    catch (const std::runtime_error &err)
    {
      std::cerr << "Fatal error in DeviceGuard destructor:" << std::endl << err.what() << std::endl;
    }
  }

private:
  CUcontext old_context_;
  int old_device_;
};

} // namespace nvcomp
