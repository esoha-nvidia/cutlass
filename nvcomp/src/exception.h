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

#if defined __cplusplus && !defined(ALLOW_C_CUDA_CHECK)
#error "In C++ translation units the exception.hpp header should be included instead."
#endif // __cplusplus

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(func)                                                                                               \
  do                                                                                                                   \
  {                                                                                                                    \
    cudaError_t rt = (func);                                                                                           \
    if (rt != cudaSuccess)                                                                                             \
    {                                                                                                                  \
      fprintf(                                                                                                         \
        stderr,                                                                                                        \
        "Runtime API call failure \"" #func "\" with %d (%s) at " __FILE__ ":%d\n",                                    \
        (int)rt,                                                                                                       \
        cudaGetErrorString(rt),                                                                                        \
        __LINE__                                                                                                       \
      );                                                                                                               \
      exit(EXIT_FAILURE);                                                                                              \
    }                                                                                                                  \
  } while (0)

#define CU_CHECK(func)                                                                                                 \
  do                                                                                                                   \
  {                                                                                                                    \
    CUresult rt = (func);                                                                                              \
    if (rt != CUDA_SUCCESS)                                                                                            \
    {                                                                                                                  \
      char const *msg;                                                                                                 \
      cuGetErrorString(rt, &msg);                                                                                      \
      fprintf(                                                                                                         \
        stderr,                                                                                                        \
        "Driver API call failure \"" #func "\" with %d (%s) at " __FILE__ ":%d\n",                                     \
        (int)rt,                                                                                                       \
        msg,                                                                                                           \
        __LINE__                                                                                                       \
      );                                                                                                               \
      exit(EXIT_FAILURE);                                                                                              \
    }                                                                                                                  \
  } while (0)

#define CUDA_CHECK_RETURN(func, error_return_val)                                                                      \
  do                                                                                                                   \
  {                                                                                                                    \
    cudaError_t rt = (func);                                                                                           \
    if (rt != cudaSuccess)                                                                                             \
    {                                                                                                                  \
      fprintf(                                                                                                         \
        stderr,                                                                                                        \
        "Runtime API call failure \"" #func "\" with %d (%s) at " __FILE__ ":%d\n",                                    \
        (int)rt,                                                                                                       \
        cudaGetErrorString(rt),                                                                                        \
        __LINE__                                                                                                       \
      );                                                                                                               \
      return error_return_val;                                                                                         \
    }                                                                                                                  \
  } while (0)

#define CU_CHECK_RETURN(func, error_return_val)                                                                        \
  do                                                                                                                   \
  {                                                                                                                    \
    CUresult rt = (func);                                                                                              \
    if (rt != CUDA_SUCCESS)                                                                                            \
    {                                                                                                                  \
      char const *msg;                                                                                                 \
      cuGetErrorString(rt, &msg);                                                                                      \
      fprintf(                                                                                                         \
        stderr,                                                                                                        \
        "Driver API call failure \"" #func "\" with %d (%s) at " __FILE__ ":%d\n",                                     \
        (int)rt,                                                                                                       \
        msg,                                                                                                           \
        __LINE__                                                                                                       \
      );                                                                                                               \
      return error_return_val;                                                                                         \
    }                                                                                                                  \
  } while (0)
