/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include <iostream>
#include <stdexcept>
#include <string>

#include "Logging.h"
#include "nvcomp.hpp"

namespace nvcomp
{

class Check
{
public:
  // NOTE: there is no C++11/C++14 standard way to get the function name.
  // In the future we could try to handle major compilers, and get the
  // name that way, as well as use the c++20 method.
  static nvcompStatus_t exception_to_error(const std::exception &e, const std::string &function_name);
};

#define NVCOMP_WRAP_CHECK_FUNC(func, ...)                                                                              \
  do                                                                                                                   \
  {                                                                                                                    \
    ::nvcompStatus_t status = func(__VA_ARGS__);                                                                       \
    if (status != ::nvcompStatus_t::nvcompSuccess)                                                                     \
    {                                                                                                                  \
      return status;                                                                                                   \
    }                                                                                                                  \
  } while (false)

inline constexpr __host__ __device__ bool is_power_of_two(std::size_t n) noexcept
{
  return n > 0 && (n & (n - 1)) == 0;
}

template <typename T>
inline nvcompStatus_t check_specific_alignment(const char *desc, T *p, std::size_t alignment)
{
  assert(is_power_of_two(alignment));
  auto p_uint = reinterpret_cast<std::uintptr_t>(p);
  if ((p_uint % alignment) != 0)
  {
    LOG_ERROR("input {:s} (value {:#x}) must be aligned to {:d} bytes", desc, p_uint, alignment);
    return nvcompStatus_t::nvcompErrorAlignment;
  }
  else
  {
    return nvcompStatus_t::nvcompSuccess;
  }
}

#define NVCOMP_CHECK_SPECIFIC_ALIGNMENT(desc, p, alignment)                                                            \
  NVCOMP_WRAP_CHECK_FUNC(::nvcomp::check_specific_alignment, desc, p, alignment)

template <typename T>
inline nvcompStatus_t check_alignment(const char *desc, T *p)
{
  static_assert(!std::is_void_v<T>, "Single-argument check_alignment cannot be used with pointers-to-void.");

  NVCOMP_CHECK_SPECIFIC_ALIGNMENT(desc, p, alignof(T));

  return nvcompStatus_t::nvcompSuccess;
}

template <typename T>
inline nvcompStatus_t check_alignment(const char *desc, T *p, std::size_t specific_alignment)
{
  const std::size_t max_alignment = [&]() {
    if constexpr (std::is_void_v<T>)
    {
      return specific_alignment;
    }
    else
    {
      return std::max(sizeof(T), specific_alignment);
    }
  }();

  NVCOMP_CHECK_SPECIFIC_ALIGNMENT(desc, p, max_alignment);

  return nvcompStatus_t::nvcompSuccess;
}

#undef NVCOMP_CHECK_SPECIFIC_ALIGNMENT

inline nvcompStatus_t check_not_null(const char *desc, const void *p)
{
  if (p == nullptr)
  {
    LOG_ERROR("input {:s} must not be null", desc);
    return nvcompStatus_t::nvcompErrorInvalidValue;
  }
  else
  {
    return nvcompStatus_t::nvcompSuccess;
  }
}

} // namespace nvcomp

// Necessary to ensure that we can always pass non-empty variadic packs to
// macros below. Before C++20's __VA_OPT__, doing otherwise quickly results in
// ill-formed macro expansions like "some_func(first_arg, )".
// When passing __VA_ARGS__ from another macro, they must be passed as
// NVCOMP_STRINGIFY_FIRST(__VA_ARGS__, dummy) to ensure the ... of this macro
// receives at least one argument. This would otherwise not be true if
// __VA_ARGS__ contained a single argument.
#define NVCOMP_STRINGIFY_FIRST(a, ...) #a

// Check pointer alignment.
// Can be called with either a single or two arguments.
// The first argument is always a pointer to some type. In the single-argument
// variant, it is checked that the pointer is aligned to the size of the
// pointed-to type. In the two argument variant, the pointer alignment is
// checked against the maximum of the pointed-to type size and the second
// argument.
// If the pointed-to type is a void type, only the two-argument variant is
// allowed. The pointer alignment is directly checked against the second
// argument.
#define NVCOMP_CHECK_ALIGNMENT(...)                                                                                    \
  NVCOMP_WRAP_CHECK_FUNC(::nvcomp::check_alignment, NVCOMP_STRINGIFY_FIRST(__VA_ARGS__, dummy), __VA_ARGS__)

#define NVCOMP_CHECK_NOT_NULL(p) NVCOMP_WRAP_CHECK_FUNC(::nvcomp::check_not_null, #p, p)

#define NVCOMP_CHECK_CHUNK_SIZE(chunk_size, max_allowed_chunk_size)                                                    \
  do                                                                                                                   \
  {                                                                                                                    \
    if (chunk_size > max_allowed_chunk_size)                                                                           \
    {                                                                                                                  \
      LOG_ERROR("Chunk size {} must not exceed {} bytes", chunk_size, max_allowed_chunk_size);                         \
      return nvcompStatus_t::nvcompErrorChunkSizeTooLarge;                                                             \
    }                                                                                                                  \
  } while (false)

#define NVCOMP_CHECK_BATCH_SIZE(batch_size, max_allowed_batch_size)                                                    \
  do                                                                                                                   \
  {                                                                                                                    \
    if (batch_size > static_cast<size_t>(max_allowed_batch_size))                                                      \
    {                                                                                                                  \
      LOG_ERROR("Batch size {} must not exceed {}", batch_size, max_allowed_batch_size);                               \
      return nvcompStatus_t::nvcompErrorBatchSizeTooLarge;                                                             \
    }                                                                                                                  \
  } while (false)
