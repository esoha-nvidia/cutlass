/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

#include <cstdint>

#include "nvcomp/utils.hpp"
#include "zstd/constants.cuh"

#pragma once

namespace zstd
{

// Template allows for use of a smaller underlying type in use cases where we don't need so much dynamic range.
struct sequence
{
  // Use an array here so we can access it in a warp-smart way.
  // 0: Lit len
  // 1: Offset
  // 2: Match len
  int vals[3];

  inline __device__ int &operator[](int ix) { return vals[ix]; }

  inline const __device__ int &operator[](int ix) const { return vals[ix]; }
};

// This is here because this file can be included by host or device code, whereas the
// place where CompressSequenceBuffer is defined is only included by device code.
inline size_t __host__ __device__ get_compacted_seq_buff_size(size_t num_sequences)
{
  // Num sequences must be even, see definition of CompressSequenceBuffer
  num_sequences = nvcomp::roundUpTo(num_sequences, 2);
  size_t result = num_sequences * 2 * sizeof(uint16_t); // class structures and mat/lit lengths
  result += nvcomp::roundUpDiv(num_sequences * OFFSET_BIT_COUNT, 32) * sizeof(uint32_t);
  return result;
}

} // namespace zstd
