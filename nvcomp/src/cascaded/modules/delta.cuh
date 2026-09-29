/*
 * Copyright (c) 2021-2026, NVIDIA CORPORATION. All rights reserved.
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

#include <cub/cub.cuh>

#include "common.h"

using nvcomp::roundUpDiv;

namespace modules
{

/**
 * Perform delta compression on a single threadblock.
 *
 * This function calculate the adjacent differences between consecutive elements
 * of the input buffer.
 *
 * @param[in] input_buffer Array of size \p input_size of the input elements.
 * @param[out] output_buffer Array of size (\p input_size - 1) of the adjacent
 * differences.
 */
template <typename data_type, typename size_type>
__device__ void block_delta_compress(const data_type *input_buffer, size_type input_size, data_type *output_buffer)
{
  for (size_type element_idx = threadIdx.x; element_idx + 1 < input_size; element_idx += blockDim.x)
  {
    output_buffer[element_idx] = input_buffer[element_idx + 1] - input_buffer[element_idx];
  }
}

/**
 * Perform delta decompression on a single threadblock.
 *
 * This function calculate the prefix sum of the input elements, i.e. the output
 * sequence should be `initial_value`, `initial_value + input_buffer[0]`,
 * `initial_value + input_buffer[0] + input_buffer[1]`, etc.
 *
 * @param[in] input_buffer Array of size \p input_num_elements of the input
 * elements.
 * @param[in] initial_value The first element of the uncompressed buffer.
 * @param[out] output_buffer Array of size (\p input_num_elements + 1) of the
 * prefix sum output.
 */
template <typename data_type, typename size_type, int threadblock_size>
__device__ void block_delta_decompress(
  const data_type *input_buffer,
  data_type initial_value,
  size_type input_num_elements,
  data_type *output_buffer
)
{
  typedef nvcomp::cub::BlockScan<data_type, threadblock_size> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;

  const int num_rounds = roundUpDiv(input_num_elements, threadblock_size);

  for (int round = 0; round < num_rounds; round++)
  {
    const size_type idx = round * threadblock_size + threadIdx.x;

    data_type input_val = 0;
    if (idx < input_num_elements)
    {
      input_val = input_buffer[idx];
    }

    data_type output_val;
    data_type aggregate;
    BlockScan(temp_storage).ExclusiveScan(input_val, output_val, initial_value, cub_sum(), aggregate);
    initial_value += aggregate;

    if (idx < input_num_elements)
    {
      output_buffer[idx] = output_val;
    }

    __syncthreads();
  }

  if (threadIdx.x == 0)
  {
    output_buffer[input_num_elements] = initial_value;
  }
}

} // namespace modules