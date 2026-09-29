/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

#include "cascaded/common/cascaded_static_set.cuh"
#include "cascaded/common/cascaded_utils.cuh"
#include "dict_warp_dx.cuh"

namespace dictionary
{

/**
 *  @return the number of bytes required to build a dictionary for the given cardinality
 */
template <typename data_t, typename index_t>
__host__ __device__ size_t block_dictionary_get_scratch_req(int est_num_unique_inputs)
{
  // In the future, this block dictionary impl may require additional scratch
  return cascaded::block_static_set_get_scratch_req<data_t, index_t>(est_num_unique_inputs);
}

/**
 *  Uses a block of threads to dictionary encode the given input, using the provided static set
 *
 *  @param input The input array to be dictionary encoded
 *  @param num_inputs The number of elements in the input array
 *  @param est_num_unique_inputs Estimated number of unique values (for sizing the static set)
 *  @param dict_values The unique values of the input array in index order
 *  @param dict_indices The unique-value-index for each value in the input array in input order
 *  @param num_unique_values Output: the actual number of unique values found
 *  @param scratch_ptr The scratch space that will be used to store the static set
 *  @param scratch_size_bytes The number of bytes available in the scratch space.
 */
template <typename data_t, typename index_t>
__device__ void block_dictionary_encode(
  const data_t *input,
  const uint32_t num_inputs,
  const uint32_t est_num_unique_inputs,

  data_t *dict_values,
  index_t *dict_indices,
  index_t &num_unique_values,

  uint64_t *scratch_ptr,
  const uint32_t scratch_size_bytes
)
{
  assert(num_inputs > 0);
  assert(blockDim.x % WARP_SIZE == 0);

  cascaded::static_set<data_t, index_t> my_set;
  __shared__ uint32_t shared_counter;

  // Initialize the static set
  cascaded::block_static_set_init(
    my_set,
    scratch_ptr,
    scratch_size_bytes,
    &shared_counter,
    est_num_unique_inputs,
    input[0] // sentinel value
  );

  // The sentinel value must be manually added to the dictionary
  if (threadIdx.x == 0)
  {
    dict_values[0] = input[0];
  }

  __syncthreads(); // ensure scratch space is initialized

  // Each warp processes WARP_SIZE elements at a time
  const int warp_id = threadIdx.x / WARP_SIZE;

  for (int ix_base = warp_id * WARP_SIZE; ix_base < num_inputs; ix_base += blockDim.x)
  {
    int ix_input = ix_base + cascaded::thread_warp_ix();

    // Load value (use sentinel for out-of-bounds to avoid invalid reads)
    data_t my_value = (ix_input < num_inputs) ? input[ix_input] : input[0];
    bool active = (ix_input < num_inputs);

    warp_dictionary_encode(
      my_set,
      dict_values,
      dict_indices + ix_base, // Advance pointer for each warp batch
      my_value,
      active
    );
  }

  __syncthreads(); // all threads must finish to get the true num_unique_values

  num_unique_values = shared_counter;
}

/** 
 *  Uses a block of threads to decode the given dictionary
 *  Each warp independently processes batches of elements
 *
 *  @param dict_values unique values in the dictionary
 *  @param dict_indices the index assigned to each element of the input array
 *  @param num_indices the number of values in the original input array
 *  @param output the recovered input array
 */
template <typename data_t, typename index_t, uint32_t BLOCK_SIZE, uint32_t UNROLL_COUNT>
__device__ void
block_dictionary_decode(const data_t *dict_values, const index_t *dict_indices, const int num_indices, data_t *output)
{
  assert(blockDim.x == BLOCK_SIZE);

  const int warp_id = threadIdx.x / WARP_SIZE;
  constexpr int num_warps = BLOCK_SIZE / WARP_SIZE;
  constexpr int elts_per_batch = WARP_SIZE * UNROLL_COUNT;

  // Each warp processes its own batches
  int batch_id = warp_id;
  while (batch_id * elts_per_batch < num_indices)
  {
    int batch_offset = batch_id * elts_per_batch;
    int remaining = num_indices - batch_offset;

    if (remaining >= elts_per_batch)
    {
      // Full batch - all threads active
      warp_dictionary_decode<data_t, index_t, UNROLL_COUNT>(
        dict_values,
        dict_indices + batch_offset,
        output + batch_offset,
        true // all threads active
      );
    }
    else
    {
      // Partial batch - use UNROLL_COUNT=1 for tail elements
      for (int tail_offset = 0; tail_offset < remaining; tail_offset += WARP_SIZE)
      {
        int elem_idx = batch_offset + tail_offset + cascaded::thread_warp_ix();
        bool active = (elem_idx < num_indices);

        warp_dictionary_decode<data_t, index_t, 1>(
          dict_values,
          dict_indices + batch_offset + tail_offset,
          output + batch_offset + tail_offset,
          active
        );
      }
    }

    batch_id += num_warps;
  }
}

} // namespace dictionary
