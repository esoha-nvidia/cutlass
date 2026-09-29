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

#include "comm_via_ring.cuh"
#include "comm_via_swap.cuh"
#include "constants.cuh"
#include "CorrectnessChecks.cuh"
#include "nvcomp/snappy.h"
#include "util.cuh"
#include "warp_4BpB_processor.cuh"
#include "warp_snappy_finder.cuh"
#include "warp_snappy_mapper.cuh"

namespace snappy
{

/**
 * @brief Snappy decompression device function
 **/
template <bool CORRECTNESS_CHECK>
inline __device__ void do_unsnap(
  const uint8_t *const __restrict__ device_in_ptr,
  const uint64_t device_in_bytes,
  uint8_t *const __restrict__ device_out_ptr,
  const uint64_t device_out_available_bytes,
  nvcompStatus_t *const __restrict__ nvcomp_statuses,
  SnappyCorrectnessChecker<CORRECTNESS_CHECK> *__restrict__ correctness_checker,
  uint64_t *__restrict__ device_out_bytes
)
{
  if constexpr (CORRECTNESS_CHECK)
  {
    if (threadIdx.x < WARP_SIZE)
    {
      SnappyCorrectnessChecker<CORRECTNESS_CHECK>::Initialize(
        correctness_checker,
        device_in_ptr,
        device_in_bytes,
        device_out_available_bytes
      );
      SnappyCorrectnessChecker<CORRECTNESS_CHECK>::preliminaryChecks(0, 0, __LINE__, __func__, correctness_checker);
    }
    __syncthreads();
    if (SnappyCorrectnessChecker<CORRECTNESS_CHECK>::hasError(correctness_checker))
    {
      if (threadIdx.x == 0)
      {
        *nvcomp_statuses = nvcompErrorCannotDecompress;
      }
      return;
    }
  }

  uint32_t header_size_bytes;
  int32_t error;
  uint32_t output_size =
    get_uncompressed_size(reinterpret_cast<const uint8_t *>(device_in_ptr), device_in_bytes, header_size_bytes, error);

  const uint8_t *input_buffer = reinterpret_cast<const uint8_t *>(device_in_ptr) + header_size_bytes;
  uint8_t *output_buffer = reinterpret_cast<uint8_t *>(device_out_ptr);
  uint64_t input_size_long = device_in_bytes - header_size_bytes;

  // If EITHER the input OR the output buffers are larger than the MAX_STREAM_SIZE then the 4BpB format breaks.
  bool valid = error >= 0 && input_size_long <= nvcompSnappyDecompressionMaxAllowedChunkSize &&
               output_size <= nvcompSnappyDecompressionMaxAllowedChunkSize;

  uint32_t input_size = static_cast<uint32_t>(input_size_long);

  if (!threadIdx.x)
  {
    // Initialize to erred state in case decompression crashses
    // If decompression is successful, this error will be retracted
    if (nvcomp_statuses)
    {
      *nvcomp_statuses = nvcompErrorCannotDecompress;
    }

    if (valid)
    {
      // Trying to avoid using a register by storing output size to gmem
      if (device_out_bytes)
      {
        *device_out_bytes = output_size;
      }
    }
    else
    {
      // Metadata was invalid. Cannot decompress
      if (device_out_bytes)
      {
        *device_out_bytes = 0;
      }
    }
  }

  if (!valid)
  {
    return;
  }

#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ comm_atomic_t shared_counters_finder_to_mapper[2];
  __shared__ comm_barrier_t shared_barriers_mapper_to_processor[4];

  comm_via_ring_init_shmem(shared_counters_finder_to_mapper);
  comm_via_swap_init_shmem(shared_barriers_mapper_to_processor);

  // Allocated 8B per symbol ring slot even though only 5B is needed.
  // Testing showed that this over-allocation improved performance via alignment
  __shared__ uint32_t shared_symbol_ring[2 * SYMBOL_RING_SIZE];
  __shared__ uint32_t shared_window_ring[WINDOW_RING_SIZE * OUTPUT_WINDOW_SIZE];

  __syncthreads();

  if (threadIdx.x < WARP_SIZE_U)
  {
    // PREFETCH, FIND, EXTRACT
    unsnap_warp_snappy_finder<CORRECTNESS_CHECK>(
      shared_counters_finder_to_mapper,
      shared_symbol_ring,
      input_buffer,
      input_size,
      correctness_checker
    );
  }
  else if (threadIdx.x < 2 * WARP_SIZE_U)
  {
    // DECODE, MAP
    uint32_t decode_ix_input, decode_ix_output;
    bool invalid_stream = false;
    unsnap_warp_snappy_mapper<CORRECTNESS_CHECK>(
      shared_counters_finder_to_mapper,
      shared_barriers_mapper_to_processor,
      shared_symbol_ring,
      shared_window_ring,
      decode_ix_input,
      decode_ix_output,
      invalid_stream,
      output_size,
      correctness_checker
    );

    invalid_stream = __ballot_sync(~0U, invalid_stream);

    if (SnappyCorrectnessChecker<CORRECTNESS_CHECK>::hasError(correctness_checker))
    {
      return;
    }
    // This Mapper warp is the only warp that has the actual compressed & uncompressed sizes
    snappy_unsnap_err_check(
      decode_ix_input,
      decode_ix_output,
      input_size,
      invalid_stream,
      device_out_bytes,
      nvcomp_statuses
    );
    SnappyCorrectnessChecker<CORRECTNESS_CHECK>::checkInputOutputSizes(
      decode_ix_input + header_size_bytes,
      decode_ix_output,
      __LINE__,
      __func__,
      nvcomp_statuses,
      correctness_checker
    );
  }
  else
  {
    // SMASH, GATHER & WRITE
    unsnap_warp_4BpB_processor<CORRECTNESS_CHECK>(
      shared_barriers_mapper_to_processor,
      shared_window_ring,
      input_buffer,
      output_buffer,
      output_size,
      correctness_checker
    );
  }
}

} // namespace snappy