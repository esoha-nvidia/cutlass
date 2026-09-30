/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or its
 * affiliates is strictly prohibited.
 */

#pragma once

#include <cstdint>
#include <type_traits>

#include "cascaded/next/primitives/kibi_cascader/kibi_cascader_constants.cuh"

namespace nvcomp::cascaded::next::kibi_cascader
{

/**
  @tparam T Data type of the input vector
  @tparam SearchSpace Mask indicating which encodings may be considered
  @return The fixed number of metadata words generated for T and SearchSpace
*/
template <typename T, kibi_cascader_optimization_search_space SearchSpace>
inline constexpr uint32_t get_num_metadata_words()
{
  static_assert(std::is_arithmetic_v<T>);
  static_assert(sizeof(T) == sizeof(uint32_t) || sizeof(T) == sizeof(uint64_t));

  // A value is bitpacked as 32-bit planes, each contributing a change mask and a base value
  if constexpr (std::is_floating_point_v<T>)
  {

    constexpr uint32_t WORDS_PER_PLANE = 2u;
    constexpr uint32_t NUM_PLANES = sizeof(T) / sizeof(uint32_t);
    constexpr bool HAS_ALP_METADATA = std::is_floating_point_v<T> && (SearchSpace & SEARCH_ALP) != 0u;
    return NUM_PLANES * WORDS_PER_PLANE + static_cast<uint32_t>(HAS_ALP_METADATA);
  }
  else
  {
    static_assert(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>);
    constexpr bool SEARCHES_DELTA = (SearchSpace & SEARCH_DELTA) != 0u;
    constexpr bool SEARCHES_INTEGER_TRANSFORMS = SEARCHES_DELTA || (SearchSpace & SEARCH_FOR) != 0u;
    constexpr bool SEARCHES_RLE = (SearchSpace & SEARCH_RLE) != 0u;
    if constexpr (std::is_same_v<T, uint32_t>)
    {
      return 2u + static_cast<uint32_t>(SEARCHES_INTEGER_TRANSFORMS || SEARCHES_RLE) +
             static_cast<uint32_t>(SEARCHES_DELTA) + static_cast<uint32_t>(SEARCHES_RLE);
    }
    else
    {
      return 4u + static_cast<uint32_t>(SEARCHES_INTEGER_TRANSFORMS || SEARCHES_RLE) +
             2u * static_cast<uint32_t>(SEARCHES_DELTA) + static_cast<uint32_t>(SEARCHES_RLE);
    }
  }
}

} // namespace nvcomp::cascaded::next::kibi_cascader
