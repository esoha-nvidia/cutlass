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

#include <cuda_runtime_api.h>

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
template <typename T>
static cudaError_t cudaMallocSafe(T **devPtr, size_t size)
#else
cudaError_t cudaMallocSafe(void **devPtr, size_t size)
#endif // __cplusplus
{
  cudaError_t err = cudaMalloc((void **)devPtr, size);
  if (err == cudaErrorMemoryAllocation)
  {
    // Attempt to get memory information
    size_t gpu_bytes_free, gpu_bytes_total;
    cudaError_t err_meminfo = cudaMemGetInfo(&gpu_bytes_free, &gpu_bytes_total);
    if (err_meminfo != cudaSuccess)
    {
      return err_meminfo;
    }

    if (gpu_bytes_free < size)
    {
      fprintf(
        stderr,
        "WARNING: Cannot fit data in GPU memory. Bytes requested: %zu"
        " > bytes available: %zu.\n",
        size,
        gpu_bytes_free
      );
      exit(3);
    }
  }
  return err;
}
