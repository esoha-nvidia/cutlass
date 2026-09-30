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

#include <cuda_runtime.h>

#include <cassert>
#include <cstdint>
#include <type_traits>

#include "cascaded/next/primitives/kibi_cascader/kibi_cascader_constants.cuh"
#include "cascaded/next/primitives/kibi_cascader/kibi_cascader_encoding.cuh"
#include "cascaded/next/primitives/kibi_cascader/kibi_cascader_utils.cuh"
#include "cascaded/next/primitives/kibi_cascader/transforms/kibi_transform_delta.cuh"
#include "cascaded/next/primitives/kibi_cascader/transforms/kibi_transform_frame_of_reference.cuh"
#include "cascaded/next/primitives/kibi_cascader/transforms/kibi_transform_signed_delta.cuh"
#include "cascaded/next/primitives/kibi_cascader/transforms/kibi_transform_zigzag.cuh"
#include "CudaConstants.h"

namespace nvcomp::cascaded::next::kibi_cascader::warp_transforms
{

// Applies the selected transformation collectively to interleaved register values.
template <typename T>
inline __device__ void warp_apply_transform(
  uint32_t values[KC_WORDS_PER_THREAD],
  const PreBitpackTransform transform,
  const uint32_t input_size_bytes,
  const T parameter,
  const T secondary = T{0}
)
{
  static_assert(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>);
  switch (transform)
  {
    case PreBitpackTransform::None:
      thread_normalize_padding<T>(values, input_size_bytes, warp_shuffle(thread_value<T>(values, 0u), 0u));
      break;
    case PreBitpackTransform::FrameOfReference:
      thread_transform_frame_of_reference<T>(values, parameter);
      thread_normalize_padding<T>(values, input_size_bytes, warp_shuffle(thread_value<T>(values, 0u), 0u));
      break;
    case PreBitpackTransform::Delta:
      warp_transform_delta<T>(values);
      thread_normalize_padding<T>(values, input_size_bytes, T{0});
      break;
    case PreBitpackTransform::DeltaZigzag:
      warp_transform_delta<T>(values);
      thread_transform_zigzag<T>(values);
      thread_normalize_padding<T>(values, input_size_bytes, T{0});
      break;
    case PreBitpackTransform::DeltaFrameOfReference:
      warp_transform_delta<T>(values);
      warp_transform_delta_frame_of_reference<T>(values, input_size_bytes, secondary);
      break;
    default:
      assert(false);
      break;
  }
}

// Inverts a transformation allowed by SearchSpace collectively on interleaved register values.
template <typename T, kibi_cascader_optimization_search_space SearchSpace>
inline __device__ void warp_inverse_transform(
  uint32_t values[KC_WORDS_PER_THREAD],
  const PreBitpackTransform transform,
  const T parameter,
  const T secondary
)
{
  static_assert(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>);
  switch (transform)
  {
    case PreBitpackTransform::None:
      break;
    case PreBitpackTransform::FrameOfReference:
      if constexpr ((SearchSpace & SEARCH_FOR) != 0u)
      {
        thread_inverse_transform_frame_of_reference<T>(values, parameter);
      }
      else
      {
        assert(false);
      }
      break;
    case PreBitpackTransform::Delta:
      if constexpr ((SearchSpace & SEARCH_DELTA) != 0u)
      {
        warp_inverse_transform_delta<T>(values, parameter);
      }
      else
      {
        assert(false);
      }
      break;
    case PreBitpackTransform::DeltaZigzag:
      if constexpr ((SearchSpace & SEARCH_DELTA) != 0u)
      {
        warp_inverse_transform_signed_delta<T>(values, parameter);
      }
      else
      {
        assert(false);
      }
      break;
    case PreBitpackTransform::DeltaFrameOfReference:
      if constexpr ((SearchSpace & SEARCH_DELTA) != 0u)
      {
        thread_inverse_transform_frame_of_reference<T>(values, secondary);
        if (threadIdx.x % WARP_SIZE == 0u)
        {
          store_thread_value<T>(values, 0u, T{0});
        }
        warp_inverse_transform_delta<T>(values, parameter);
      }
      else
      {
        assert(false);
      }
      break;
    default:
      assert(false);
      break;
  }
}

} // namespace nvcomp::cascaded::next::kibi_cascader::warp_transforms
