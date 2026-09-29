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

/**

COMM-VIA-SWAP: Facilitates communication between a producer & consumer.
        Used to protect a PAIR of buffers where producer can produce exactly
        once before waiting for consumer.
        
This design copied from 7.26.5. Spatial Partitioning (also known as Warp Specialization) of programming guide
*/

#pragma once

#define COMM_VIA_SWAP_NUM_ATOMICS 4
#define COMM_VIA_SWAP_NUM_BUFFERS 2

#include <cuda/barrier>
#define comm_barrier_t cuda::barrier<cuda::thread_scope_block>
#define IX_READY 0
#define IX_FILLED 2
#define NUM_SWAPPING_THREADS 64

namespace snappy
{

/**
 *  @brief: must be called BEFORE a syncthreads. Initializes atomics
 */
inline __device__ void comm_via_swap_init_shmem(comm_barrier_t (&shared_barriers)[COMM_VIA_SWAP_NUM_ATOMICS])
{
  if (threadIdx.x < COMM_VIA_SWAP_NUM_ATOMICS)
  {
    // When computer sanitizer complains: the barriers are initialized right here
    init(shared_barriers + threadIdx.x, NUM_SWAPPING_THREADS);
  }
}

/**
 *  @brief: must be called AFTER a syncthreads by consumer only. Allows producer to produce. 
 */
inline __device__ void comm_via_swap_init_consumer(comm_barrier_t (&shared_barriers)[COMM_VIA_SWAP_NUM_ATOMICS])
{
  // consumer must let producer know that both buffers are ready to be filled
  for (uint32_t ix_buffer = 0; ix_buffer < COMM_VIA_SWAP_NUM_BUFFERS; ix_buffer++)
  {
    comm_barrier_t::arrival_token token = shared_barriers[ix_buffer].arrive();
  }
}

inline __device__ void
comm_via_swap_produce_start(comm_barrier_t (&shared_barriers)[COMM_VIA_SWAP_NUM_ATOMICS], const uint32_t ix_produce)
{
  shared_barriers[IX_READY + (ix_produce % COMM_VIA_SWAP_NUM_BUFFERS)].arrive_and_wait();
}

inline __device__ void
comm_via_swap_produce_end(comm_barrier_t (&shared_barriers)[COMM_VIA_SWAP_NUM_ATOMICS], const uint32_t ix_produce)
{
  comm_barrier_t::arrival_token token = shared_barriers[IX_FILLED + (ix_produce % COMM_VIA_SWAP_NUM_BUFFERS)].arrive();
}

inline __device__ void
comm_via_swap_consume_start(comm_barrier_t (&shared_barriers)[COMM_VIA_SWAP_NUM_ATOMICS], const uint32_t ix_consume)
{
  shared_barriers[IX_FILLED + (ix_consume % COMM_VIA_SWAP_NUM_BUFFERS)].arrive_and_wait();
}

inline __device__ void
comm_via_swap_consume_end(comm_barrier_t (&shared_barriers)[COMM_VIA_SWAP_NUM_ATOMICS], const uint32_t ix_consume)
{
  comm_barrier_t::arrival_token token = shared_barriers[IX_READY + (ix_consume % COMM_VIA_SWAP_NUM_BUFFERS)].arrive();
}

} // namespace snappy