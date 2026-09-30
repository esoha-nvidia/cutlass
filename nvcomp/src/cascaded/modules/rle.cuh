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

namespace nvcomp::cascaded::modules
{

/**
 * Perform RLE compression on a single threadblock.
 *
 * Note: \p num_outputs is written by the last thread of the threadblock, and
 * must be a shared memory location so that the change can be propagated to
 * other threads in the threadblock.
 *
 * @param[in] input_buffer Uncompressed input buffer.
 * @param[in] num_inputs Number of elements in \p input_buffer.
 * @param[out] val_buffer Value array after the RLE compression.
 * @param[out] count_buffer Count array after the RLE compression.
 * @param[out] num_outputs Number of output elements. This is the array size of
 * \p val_buffer or \p count_buffer. The buffer must locate in shared memory.
 * @param[in] tmp_buffer Temporary buffer that needs to hold at least
 * \p num_inputs elements.
 */
template <typename data_type, typename size_type, typename run_type, int threadblock_size>
__device__ void block_rle_compress(
  const data_type *input_buffer,
  const size_type num_inputs,
  data_type *val_buffer,
  run_type *count_buffer,
  size_type *num_outputs,
  run_type *tmp_buffer
)
{
  // In this kernel, we assign `num_inputs_per_thread` consecutive elements to a
  // thread. Then, if an input element is the last element of a run, the thread
  // stores the value and the count to the output buffer. Note that an input
  // element is the last element of a run if and only if
  //   a) The value of this element is different from the value of the next
  //   element.
  //   or b) This element is the last element.
  //
  // The algorithm consists of the following steps.
  //   1. Each thread counts the number of last elements.
  //   2. Use prefix sum on the counts to calculate the output location of each
  //   thread.
  //   3. For each last element, store the value in `val_buffer` and the input
  //   index in `tmp_buffer`.
  //   4. Calculate the adjacent differences of input indices in `tmp_buffer` to
  //   get the counts, and store into `count_buffer`.
  //
  // Note: `tmp_buffer` is used because we cannot calculate adjacent differences
  // in place.

  typedef nvcomp::cub::BlockScan<size_type, threadblock_size> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;

  const size_type num_inputs_per_thread = roundUpDiv(num_inputs, threadblock_size);

  // Step 1: Count the number of last elements of the current thread

  size_type num_outputs_current_thread = 0;

  data_type val = input_buffer[threadIdx.x * num_inputs_per_thread];
  data_type next_val;

  for (int ielement = 0; ielement < num_inputs_per_thread; ielement++)
  {
    const int idx = threadIdx.x * num_inputs_per_thread + ielement;
    if (idx >= num_inputs)
    {
      break;
    }

    if (idx + 1 == num_inputs)
    {
      num_outputs_current_thread++;
      break;
    }

    next_val = input_buffer[idx + 1];
    num_outputs_current_thread += next_val != val;
    val = next_val;
  }

  __syncthreads();

  // Step 2: Use prefix sum to get the output location

  size_type output_idx;
  BlockScan(temp_storage).ExclusiveSum(num_outputs_current_thread, output_idx);

  __syncthreads();

  // Step 3: For each last element, store the index and the value

  val = input_buffer[threadIdx.x * num_inputs_per_thread];

  for (int ielement = 0; ielement < num_inputs_per_thread; ielement++)
  {
    const int idx = threadIdx.x * num_inputs_per_thread + ielement;
    if (idx >= num_inputs)
    {
      break;
    }

    if (idx + 1 == num_inputs)
    {
      tmp_buffer[output_idx] = idx + 1;
      val_buffer[output_idx] = input_buffer[idx];
      output_idx++;
      break;
    }

    data_type next_val = input_buffer[idx + 1];
    if (next_val != val)
    {
      tmp_buffer[output_idx] = idx + 1;
      val_buffer[output_idx] = input_buffer[idx];
      output_idx++;
    }
    val = next_val;
  }

  if (threadIdx.x == threadblock_size - 1)
  {
    // After step 2, `output_idx` of the last thread is the sum of the number of
    // last elements in all threads except itself (since we use ExclusiveSum).
    // During step 3, the number of last elements of the current thread is added
    // to `num_outputs`. Therefore, `output_idx` here is the number of runs
    // in total.
    *num_outputs = output_idx;
  }

  // syncthreads is necessary here to make `num_outputs` avaiable on all threads
  // in the current threadblock.
  __syncthreads();

  // Step 4: Calculate the adjacent differences between indices, which is the
  // counts of every runs.

  for (int ioutput = 1 + threadIdx.x; ioutput < *num_outputs; ioutput += threadblock_size)
  {
    count_buffer[ioutput] = tmp_buffer[ioutput] - tmp_buffer[ioutput - 1];
  }

  if (threadIdx.x == 0)
  {
    count_buffer[0] = tmp_buffer[0];
  }
}

/**
 * Perform RLE decompression on a single threadblock.
 *
 * @param[in] val_buffer Values of the runs.
 * @param[in] count_buffer Counts of the runs.
 * @param[in] num_runs Number of runs. This is also the size in terms of number
 * of elements for \p val_buffer and \p count_buffer.
 * @param[out] output_buffer Pointer to the output uncompressed buffer.
 * @param[out] output_num_elements Number of uncompressed elements. This
 * argument should be unique per thread. The output should be the sum of
 * \p count_buffer.
 */
template <typename data_type, typename size_type, typename run_type, int threadblock_size>
__device__ void block_rle_decompress(
  const data_type *val_buffer,
  const run_type *count_buffer,
  size_type num_runs,
  data_type *output_buffer,
  size_type *output_num_elements
)
{
  // In this kernel, we assign runs to threads in a round-robin fashion. The
  // algorithm is divided into rounds, where each round handles
  // `threadblock_size` runs, with one run per thread. During each round, we
  // first use prefix sum to calculate the output offsets, and then each thread
  // stores the value of the run into the output locations.

  typedef nvcomp::cub::BlockScan<run_type, threadblock_size> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;

  *output_num_elements = 0;

  for (int round = 0; round < roundUpDiv(num_runs, threadblock_size); round++)
  {
    const int idx = round * threadblock_size + threadIdx.x;

    run_type current_count = 0;
    if (idx < num_runs)
    {
      current_count = count_buffer[idx];
    }

    run_type output_offset;
    run_type aggregate;
    BlockScan(temp_storage).ExclusiveSum(current_count, output_offset, aggregate);

    if (idx < num_runs)
    {
      const auto current_val = val_buffer[idx];
      for (int element_idx = 0; element_idx < current_count; element_idx++)
      {
        output_buffer[*output_num_elements + output_offset + element_idx] = current_val;
      }
    }
    *output_num_elements += aggregate;

    // syncthreads is necessary to make sure temporary storage is not
    // overwritten in the next iteration until all threads finish for the
    // current iteration.
    __syncthreads();
  }
}

} // namespace nvcomp::cascaded::modules
