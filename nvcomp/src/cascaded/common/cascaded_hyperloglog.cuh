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

#include <cub/block/block_reduce.cuh>

#include <cuda_runtime.h>

#include <cassert>
#include <climits>
#include <cmath>
#include <cstdint>

#include "cascaded_hash.cuh"
#include "cascaded_utils.cuh"

namespace nvcomp::cascaded
{

// HLL sized to fill a shared-memory budget. Registers are uint32_t so we can
// use atomicMax in shared memory. num_buckets must be a power of two.
template <uint32_t hyperloglog_size_bytes>
struct hyperloglog_config
{
  static_assert(
    (hyperloglog_size_bytes & (hyperloglog_size_bytes - 1)) == 0,
    "hyperloglog_size_bytes must be power of 2"
  );
  static_assert(hyperloglog_size_bytes >= 64, "hyperloglog_size_bytes too small");

  static constexpr uint32_t num_buckets = hyperloglog_size_bytes / sizeof(uint32_t);
  static_assert(num_buckets > 0, "num_buckets must be positive");

  static constexpr uint32_t bucket_bits = log2_pow2(num_buckets);
  static constexpr uint32_t bucket_mask = num_buckets - 1;

  // Bits left in the 32-bit hash after peeling off the bucket index.
  static constexpr uint32_t hash_bits = sizeof(uint32_t) * CHAR_BIT - bucket_bits;
};

// Bias-correction constant alpha_m from Flajolet et al.
// Needed because the raw harmonic-mean estimator is biased upward.
// Templated so the branches collapse at compile time.
template <uint32_t m>
constexpr __host__ __device__ double hll_get_alpha()
{
  if constexpr (m == 16)
  {
    return 0.673;
  }
  else if constexpr (m == 32)
  {
    return 0.697;
  }
  else if constexpr (m == 64)
  {
    return 0.709;
  }
  else
  {
    return 0.7213 / (1.0 + 1.079 / static_cast<double>(m));
  }
}

template <uint32_t hyperloglog_size_bytes, uint32_t num_threads_per_block>
inline __device__ void block_hll_init(uint32_t *hll_registers)
{
  using HLL = hyperloglog_config<hyperloglog_size_bytes>;

  assert(blockDim.x == num_threads_per_block);

#pragma unroll
  for (uint32_t ix_base = 0; ix_base < HLL::num_buckets; ix_base += num_threads_per_block)
  {
    uint32_t ix_reg = ix_base + threadIdx.x;
    if (ix_reg < HLL::num_buckets)
    {
      hll_registers[ix_reg] = 0;
    }
  }
}

template <uint32_t hyperloglog_size_bytes, typename data_t>
inline __device__ void thread_hll_insert(uint32_t *hll_registers, data_t my_value)
{
  using HLL = hyperloglog_config<hyperloglog_size_bytes>;

  uint32_t hash = hash32(my_value);
  uint32_t bucket = hash & HLL::bucket_mask;
  uint32_t remaining = hash >> HLL::bucket_bits; // hash_bits wide

  // rho = 1 + position of leading 1-bit in `remaining` (counting from MSB of
  // the hash_bits-wide field). If remaining == 0, all hash_bits are zero and
  // rho = hash_bits + 1 (the theoretical max).
  uint32_t rho;
  if (remaining == 0)
  {
    rho = HLL::hash_bits + 1;
  }
  else
  {
    // __clz operates on 32-bit values; `remaining` lives in the low
    // hash_bits bits, so subtract the padding zeros above it.
    rho = __clz(remaining) - (32 - HLL::hash_bits) + 1;
  }

  // Must be atomic: multiple threads in the warp can hit the same bucket,
  // and we need max-semantics to preserve the HLL invariant.
  atomicMax(&hll_registers[bucket], rho);
}

template <uint32_t hyperloglog_size_bytes, uint32_t num_threads_per_block, typename data_t>
inline __device__ void block_hll_insert(uint32_t *hll_registers, const data_t *values, uint32_t num_values)
{
  assert(blockDim.x == num_threads_per_block);

  for (uint32_t i = threadIdx.x; i < num_values; i += num_threads_per_block)
  {
    thread_hll_insert<hyperloglog_size_bytes, data_t>(hll_registers, values[i]);
  }
}

template <uint32_t hyperloglog_size_bytes, uint32_t num_threads_per_block>
inline __device__ uint32_t block_hll_estimate_cardinality(const uint32_t *hll_registers)
{
  using HLL = hyperloglog_config<hyperloglog_size_bytes>;
  constexpr uint32_t m = HLL::num_buckets;

  // Small-range correction threshold from Flajolet et al.
  constexpr double small_range_threshold = 2.5;

  assert(blockDim.x == num_threads_per_block);

  using BlockReduceDouble = nvcomp::cub::BlockReduce<double, num_threads_per_block>;
  using BlockReduceUint32 = nvcomp::cub::BlockReduce<uint32_t, num_threads_per_block>;

  // Reuse a single shmem allocation for both reductions (they run serially).
  __shared__ union
  {
    typename BlockReduceDouble::TempStorage d;
    typename BlockReduceUint32::TempStorage u;
  } temp_storage;

  __shared__ uint32_t s_estimate;

  double sum = 0.0;
  uint32_t zeros = 0;

  for (uint32_t i = threadIdx.x; i < m; i += num_threads_per_block)
  {
    uint32_t r = hll_registers[i];
    // ldexp(1.0, -r) == 2^-r exactly, and is a bit-manipulation rather
    // than a transcendental call.
    sum += ldexp(1.0, -static_cast<int>(r));
    zeros += (r == 0);
  }

  // First reduction: harmonic-mean denominator.
  double sum_total = BlockReduceDouble(temp_storage.d).Sum(sum);

  // Must sync before reusing the union for the second reduction — CUB may
  // still be reading from temp_storage when other threads fall through.
  __syncthreads();

  uint32_t zeros_total = BlockReduceUint32(temp_storage.u).Sum(zeros);

  if (threadIdx.x == 0)
  {
    constexpr double alpha = hll_get_alpha<m>();
    double raw_estimate = alpha * static_cast<double>(m) * m / sum_total;

    // Small-range correction: linear counting when many buckets are empty.
    if (raw_estimate <= small_range_threshold * m && zeros_total != 0)
    {
      raw_estimate = static_cast<double>(m) * log(static_cast<double>(m) / static_cast<double>(zeros_total));
    }

    // Large-range correction is omitted: the return type is uint32_t, so
    // cardinalities near 2^32 are outside the representable range anyway.
    // Clamp on overflow to avoid UB from the double->uint32_t cast.
    constexpr double uint32_max_d = static_cast<double>(UINT32_MAX);
    if (!(raw_estimate >= 0.0))
    { // handles NaN too
      s_estimate = 0;
    }
    else if (raw_estimate >= uint32_max_d)
    {
      s_estimate = UINT32_MAX;
    }
    else
    {
      s_estimate = static_cast<uint32_t>(raw_estimate + 0.5);
    }
  }

  __syncthreads();

  return max(s_estimate, 1); // cardinality cannot be less than 1
}

} // namespace nvcomp::cascaded
