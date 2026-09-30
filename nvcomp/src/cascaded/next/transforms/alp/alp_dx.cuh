/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <cuda_runtime.h>

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "cascaded/next/transforms/alp/alp_utils.cuh"
#include "CudaConstants.h"

/*
  Transforms floating-point values to integers (i.e. 1.23 -> 123) by multiplying by some
  10^exponent, with one exponent shared per-batch. For data with low decimal precision, this
  greatly improves the compression ratio achieved by later compression stages like bitpack.

  This is inspired by ALP (Adaptive Lossless floating-Point), reference implementation:
  https://github.com/cwida/ALP. Three major changes from ALP for performance:

  - Only one exponent is used for transforming values, rather than ALP's exponent/factor pair.
    Transforming from integer to floating-point uses a higher precision multiplication to recover
    the value exactly while only using one exponent
  - If a value in a batch cannot be transformed to an integer, the transformation is skipped
    for the whole batch rather than transforming the successful values and storing the
    exceptions separately. This makes the encode/decode simpler and faster
  - ALP finds the best exponent/factor pair per-batch by sampling and estimating compressed size;
    this implementation instead finds the best exponent per-batch in a single pass over its
    values, which is possible due to only using one exponent and not dealing with exceptions

  Data Requirements
  - words holds floating-point bit patterns, WordsPerThread per lane
  - each value occupies sizeof(T) / sizeof(uint32_t) consecutive words, least significant first

  API Requirements
  - all functions must be called collectively by every lane in a converged warp
  - num_values must be warp uniform, and so is the returned AlpPlan
  - decode_words_in_place must receive the plan that encode_words_in_place returned

  This is a device-side dispatcher for the typed implementations in alp_dx_float32.cuh and
  alp_dx_float64.cuh
*/

namespace nvcomp::cascaded::next::transforms::alp
{

/**
 * Per-type ALP implementation. Every specialization provides the encode_words_in_place and
 * decode_words_in_place members that the wrappers below forward to.
 */
template <typename T>
struct AlpTransform;

/**
 * Finds one decimal exponent that represents every value exactly and replaces each value with
 * its encoded integer. Leaves the words unchanged when no exponent works for the whole warp.
 *
 * @tparam T Floating-point type of the values held in words
 * @tparam WordsPerThread Number of 32-bit words each lane holds
 * @param words Floating-point bit patterns in registers, replaced by encoded integers
 * @param num_values Number of logical values in the warp's batch
 * @return The selected plan, which decode_words_in_place needs to invert the transform
 */
template <typename T, size_t WordsPerThread>
inline __device__ AlpPlan encode_words_in_place(uint32_t *words, const uint32_t num_values)
{
  static_assert((WordsPerThread * sizeof(uint32_t)) % sizeof(T) == 0u);
  assert(num_values <= WARP_SIZE * WordsPerThread * sizeof(uint32_t) / sizeof(T));
  return AlpTransform<T>::template encode_words_in_place<WordsPerThread>(words, num_values);
}

/**
 * Reconstructs the original floating-point bit patterns in the same word array. A plan that was
 * not applied leaves the words unchanged.
 *
 * @tparam T Floating-point type to reconstruct
 * @tparam WordsPerThread Number of 32-bit words each lane holds
 * @param words Encoded integers in registers, replaced by floating-point bit patterns
 * @param plan AlpPlan returned by encode_words_in_place
 * @param num_values Number of logical values in the warp's batch
 */
template <typename T, size_t WordsPerThread>
inline __device__ void decode_words_in_place(uint32_t *words, const AlpPlan plan, const uint32_t num_values)
{
  static_assert((WordsPerThread * sizeof(uint32_t)) % sizeof(T) == 0u);
  assert(num_values <= WARP_SIZE * WordsPerThread * sizeof(uint32_t) / sizeof(T));
  AlpTransform<T>::template decode_words_in_place<WordsPerThread>(words, plan, num_values);
}

} // namespace nvcomp::cascaded::next::transforms::alp
