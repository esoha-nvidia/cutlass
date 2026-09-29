/*
 * Copyright (c) 2018-2026, NVIDIA CORPORATION. All rights reserved.
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

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "CudaConstants.h"
#include "exception.hpp"
#include "nvcomp.hpp"
#include "nvcomp/shared_types.h"
#include "nvcomp/utils.hpp"

#if defined(_WIN32)
#include <time.h>
using ssize_t = ptrdiff_t;
#endif

namespace nvcomp
{

template <typename T>
T *align(T *const ptr, const size_t alignment)
{
  const size_t bits = reinterpret_cast<size_t>(ptr);
  const size_t mask = alignment - 1;

  return reinterpret_cast<T *>(((bits - 1) | mask) + 1);
}

template <typename T>
size_t relativeEndOffset(const void *start, const T *subsection, const size_t length)
{
  std::ptrdiff_t diff = reinterpret_cast<const char *>(subsection) - static_cast<const char *>(start);
  return static_cast<size_t>(diff) + length * sizeof(T);
}

template <typename T = size_t>
T relativeEndOffset(const void *start, const void *subsection)
{
  std::ptrdiff_t diff = reinterpret_cast<const char *>(subsection) - static_cast<const char *>(start);
  return static_cast<T>(diff);
}

template <typename T>
constexpr NVCOMP_HOST_DEVICE_FUNCTION bool isAligned(const T *const ptr, const size_t align)
{
  return (reinterpret_cast<uintptr_t>(ptr) % align) == 0;
}

inline __host__ __device__ bool isValidNvcompType(nvcompType_t type)
{
  switch (type)
  {
    case NVCOMP_TYPE_BITS:
    case NVCOMP_TYPE_CHAR:
    case NVCOMP_TYPE_UCHAR:
    case NVCOMP_TYPE_SHORT:
    case NVCOMP_TYPE_USHORT:
    case NVCOMP_TYPE_FLOAT16:
    case NVCOMP_TYPE_FLOAT32:
    case NVCOMP_TYPE_FLOAT64:
    case NVCOMP_TYPE_INT:
    case NVCOMP_TYPE_UINT:
    case NVCOMP_TYPE_LONGLONG:
    case NVCOMP_TYPE_ULONGLONG:
    case NVCOMP_TYPE_FLOAT8_E4M3:
      return true;
    default:
      return false;
  }
}

/**
 * @brief Cast to dim3, with debug-only range check, for CUDA kernel launch grid
 * or block dimensions
 */
template <typename S0, typename S1, typename S2>
dim3 cuda_dim_cast(const S0 i, const S1 j, const S2 k)
{
  static_assert(
    std::numeric_limits<S0>::is_integer && std::numeric_limits<S1>::is_integer && std::numeric_limits<S2>::is_integer,
    "Types for cuda_dim_cast must be integer"
  );

  assert(is_cast_valid<unsigned int>(i) && is_cast_valid<unsigned int>(j) && is_cast_valid<unsigned int>(k));

  return dim3(static_cast<unsigned int>(i), static_cast<unsigned int>(j), static_cast<unsigned int>(k));
}

template <typename S0, typename S1, typename S2>
dim3 cuda_dim_cast(S0 i, S1 j)
{
  return cuda_dim_cast(i, j, 1);
}

template <bool nothrow = false>
std::conditional_t<nothrow, nvcompStatus_t, void>
try_clear_device_statuses(size_t batch_size, nvcompStatus_t *device_statuses, cudaStream_t stream)
{
  if (device_statuses)
  {
    static_assert(nvcompSuccess == 0);
    auto result = cudaMemsetAsync(device_statuses, 0, batch_size * sizeof(nvcompStatus_t), stream);
    if constexpr (nothrow)
    {
      if (result != cudaSuccess)
      {
        return nvcompErrorCudaError;
      }
    }
    else
    {
      CUDA_CHECK(result);
    }
  }
  if constexpr (nothrow)
  {
    return nvcompSuccess;
  }
}

} // namespace nvcomp
