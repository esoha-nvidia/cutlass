/*
* Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
*  * Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
*  * Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and/or other materials provided with the distribution.
*  * Neither the name of NVIDIA CORPORATION nor the names of its
*    contributors may be used to endorse or promote products derived
*    from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
* PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
* CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
* EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
* PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
* PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
* OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once

#include <cassert>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

#ifdef __has_include
#if __has_include(<cuda_runtime.h>) && __has_include(<cuda.h>)
#include <cuda.h>
#include <cuda_runtime.h>
#endif
#endif

#include "nvcomp.hpp"

// TODO: NVCompException could also print in its constructor to `std::cerr`,
//       as the default Windows console behavior is to not print the
//       exception message.

namespace nvcomp
{

std::string get_driver_api_error_string(CUresult e);

namespace error
{

template <bool Throw, typename ErrT, bool Debug>
ErrT check_cuda_error(
  ErrT e,
  [[maybe_unused]] const char *call_desc,
  [[maybe_unused]] const char *file,
  [[maybe_unused]] int line
)
{
  if constexpr (Debug)
  {
    assert(call_desc != nullptr && file != nullptr && line >= 0);
  }
  if constexpr (std::is_same_v<ErrT, ::cudaError_t>)
  {
    if (e == ::cudaSuccess)
    {
      return e;
    }
  }
  else
  {
    static_assert(std::is_same_v<ErrT, CUresult>);
    if (e == CUDA_SUCCESS)
    {
      return e;
    }
  }

  auto [context, err_msg] = [e]() {
    if constexpr (std::is_same_v<ErrT, ::cudaError_t>)
    {
      return std::pair("CUDA Runtime API", ::cudaGetErrorString(e));
    }
    else
    {
      return std::pair("CUDA Driver API", get_driver_api_error_string(e));
    }
  }();

  std::stringstream error_ss;
  error_ss << context << " failure: ";
  if constexpr (Debug)
  {
    error_ss << "call \"" << call_desc << "\" failed with message \"";
  }
  error_ss << err_msg;
  if constexpr (Debug)
  {
    error_ss << "\" at " << file << ":" << line;
  }

  if constexpr (Throw)
  {
    throw NVCompException(nvcompErrorCudaError, error_ss.str());
  }
  else
  {
    std::cerr << error_ss.str() << std::endl;
  }

  return e;
}

#ifndef NDEBUG
#define NVCOMP_GENERIC_CHECK_CUDA(call, ErrT, Throw)                                                                   \
  ::nvcomp::error::check_cuda_error<Throw, ErrT, true>((call), #call, __FILE__, __LINE__)
#else
#define NVCOMP_GENERIC_CHECK_CUDA(call, ErrT, Throw)                                                                   \
  ::nvcomp::error::check_cuda_error<Throw, ErrT, false>((call), nullptr, nullptr, 0)
#endif

#ifndef CUDA_CHECK
#define CUDA_CHECK(call) NVCOMP_GENERIC_CHECK_CUDA(call, cudaError_t, /*Throw=*/true)
#endif
#ifndef CUDA_CHECK_LOG
#define CUDA_CHECK_LOG(call) NVCOMP_GENERIC_CHECK_CUDA(call, cudaError_t, /*Throw=*/false)
#endif
#ifndef CU_CHECK
#define CU_CHECK(call) NVCOMP_GENERIC_CHECK_CUDA(call, CUresult, /*Throw=*/true)
#endif
#ifndef CU_CHECK_LOG
#define CU_CHECK_LOG(call) NVCOMP_GENERIC_CHECK_CUDA(call, CUresult, /*Throw=*/false)
#endif

inline void check_cuda_buffer(const void *ptr)
{
  if (ptr == nullptr)
  {
    throw NVCompException(nvcompErrorInvalidValue, "NULL CUDA buffer not accepted");
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    std::string error_str(
      "Encountered Cuda Error from previous asynchronous launches: " + std::to_string(err) + ": '" +
      std::string(cudaGetErrorString(err)) + "'"
    );
    error_str += ".";

    throw NVCompException(nvcompErrorCudaError, error_str);
  }

  cudaPointerAttributes attrs = {};
  err = cudaPointerGetAttributes(&attrs, ptr);
  (void)cudaGetLastError(); // reset the cuda error (if any)
  if (err != cudaSuccess || attrs.type == cudaMemoryTypeUnregistered)
  {
    throw NVCompException(nvcompErrorCudaError, "Buffer is not CUDA-accessible");
  }
}

} // namespace error
} // namespace nvcomp
