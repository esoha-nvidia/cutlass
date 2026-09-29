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

#include <type_traits>

#include "cascaded/common/cascaded_static_set.cuh"
#include "cascaded/common/cascaded_utils.cuh"

/*
 * These warp functions are intended to be used when dictionary is a layer
 * in a multi-stage compression pipeline.
 *
 * They are especially intended to be used when the indices (and/or values)
 * are batched in shared memory and then bitpacked before being written
 * to global memory.
 *
 * In some cases a dictionary can fit entirely in shared memory. This file
 * is written such that the following arrays may live in shmem or gmem:
 *   - dict_values     (only when cardinality is known to be extremely low)
 *   - dict_indices    (especially when bitpacking batches of indices)
 *   - recovered_input (when dictionary decoding is a middle decoding layer,
 *                      e.g., before a delta-decode pass)
 */

namespace dictionary
{

// Restrict data_t and index_t to the type set the dictionary is built and
// tested against: `uint32_t` or `unsigned long long`. data_t in
// particular must match the CUDA atomicCAS overloads used by the underlying
// static set; both are spelled as the built-in types (not the <cstdint>
// aliases) to match those overloads exactly. These are exactly the combinations
// exercised in test_cascaded_dictionary_dx.cu (see DICT_DX_TYPE_MATRIX).
template <typename T>
inline constexpr bool is_supported_dict_dx_type = std::is_same<T, uint32_t>::value ||
                                                  std::is_same<T, unsigned long long>::value;

/**
 * @brief Warp-collectively dictionary-encode WARP_SIZE values held in registers.
 *
 * All 32 lanes must call together. Active lane L writes its dictionary index
 * to dict_indices[L]; the lane that first inserts a unique value writes it
 * to dict_values at the index returned by the static set.
 *
 * @note Writes to dict_values are not fenced. Caller must sync before any
 *       thread reads dict_values across warps.
 *
 * @param my_set       Initialized static set, shared across all participating warps.
 * @param dict_values  Output array of unique values. Shared across warps; do
 *                     not offset per-warp.
 * @param dict_indices Output slice for this warp's batch; writes dict_indices[0..WARP_SIZE).
 *                     Caller must offset to a unique slice per (warp, call).
 * @param my_value     Value this lane wants to insert (passed by value so
 *                     callers can feed computed values directly).
 * @param active       Whether this lane participates. Inactive lanes still
 *                     call the function but produce no output.
 */
template <typename data_t, typename index_t>
__device__ void warp_dictionary_encode(
  cascaded::static_set<data_t, index_t> &my_set,
  data_t *dict_values,
  index_t *dict_indices,
  data_t my_value,
  bool active = true
)
{
  static_assert(
    is_supported_dict_dx_type<data_t>,
    "warp_dictionary_encode: data_t must be `uint32_t` or `unsigned long long`."
  );
  static_assert(
    is_supported_dict_dx_type<index_t>,
    "warp_dictionary_encode: index_t must be `uint32_t` or `unsigned long long`."
  );

  index_t my_index;
  bool is_unique = cascaded::warp_static_set_try_insert(my_value, my_set, my_index, active);

  if (active)
  {
    if (is_unique)
    {
      dict_values[my_index] = my_value;
    }

    dict_indices[cascaded::thread_warp_ix()] = my_index;
  }
}

/**
 * @brief Warp-collectively dictionary-decode WARP_SIZE * ELEMS_PER_THREAD values.
 *
 * All 32 lanes must call together. Across the unrolled loop, lane L reads
 * dict_indices[WARP_SIZE * i + L] and writes recovered_input[WARP_SIZE * i + L]
 * for i in [0, ELEMS_PER_THREAD).
 *
 * @note When `active` varies within a warp, it must be true for a contiguous
 *       prefix of lanes; sparse masks leave gaps in recovered_input.
 *
 * @tparam ELEMS_PER_THREAD Values decoded per lane per call.
 *
 * @param dict_values    Dictionary values. Shared across warps; do not offset per-warp.
 * @param dict_indices   Input slice for this warp's batch; reads dict_indices[0..WARP_SIZE * ELEMS_PER_THREAD).
 *                       Caller must offset to a unique slice per (warp, call).
 * @param recovered_input Output slice for this warp's decoded values; same layout as dict_indices.
 * @param active         Whether this lane participates. Inactive lanes perform no loads or stores.
 */
template <typename data_t, typename index_t, uint32_t ELEMS_PER_THREAD>
__device__ void warp_dictionary_decode(
  const data_t *dict_values,
  const index_t *dict_indices,
  data_t *recovered_input,
  bool active = true
)
{
  static_assert(
    is_supported_dict_dx_type<data_t>,
    "warp_dictionary_decode: data_t must be `uint32_t` or `unsigned long long`."
  );
  static_assert(
    is_supported_dict_dx_type<index_t>,
    "warp_dictionary_decode: index_t must be `uint32_t` or `unsigned long long`."
  );

  // TODO: Can reduce register usage by storing my_values on top of my_indices

  if (active)
  {
    index_t my_indices[ELEMS_PER_THREAD];

#pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++)
    {
      my_indices[i] = dict_indices[WARP_SIZE * i + cascaded::thread_warp_ix()];
    }

    data_t my_values[ELEMS_PER_THREAD];

#pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++)
    {
      my_values[i] = dict_values[my_indices[i]]; // Scattered reads
    }

#pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++)
    {
      recovered_input[WARP_SIZE * i + cascaded::thread_warp_ix()] = my_values[i];
    }
  }
}

} // namespace dictionary
