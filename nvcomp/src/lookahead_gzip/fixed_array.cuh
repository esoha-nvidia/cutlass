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

// Like std::array but a little more flexible.
#pragma once

#include <cassert>

#include "constants.cuh"

template <typename T, size_t COUNT>
class FixedSlice
{
public:
  using value_type = T;
  constexpr __device__ size_t size() { return COUNT; }
  inline __device__ T &operator[](size_t index)
  {
    expect_lt(index, COUNT);
    return data[index];
  }
  inline const __device__ T &operator[](size_t index) const
  {
    expect_lt(index, COUNT);
    return data[index];
  }
  inline volatile __device__ T &operator[](size_t index) volatile
  {
    expect_lt(index, COUNT);
    return data[index];
  }
  inline const volatile __device__ T &operator[](size_t index) const volatile
  {
    expect_lt(index, COUNT);
    return data[index];
  }
  // Return a FixedSlice that starts at start and has length size.
  // Check at compile time that it doesn't go out of the bounds of the
  // input.
  template <size_t size = COUNT>
  __device__ FixedSlice<T, size> slice(size_t start = 0)
  {
    static_assert(size <= COUNT, "The new slice should be smaller than the current slice.");
    expect_le(start + size, COUNT);
    return FixedSlice<T, size>(data + start);
  }

private:
  template <typename T0, size_t COUNT0>
  friend class FixedArray;
  template <typename T0, size_t COUNT0>
  friend class FixedSlice;
  __device__ FixedSlice(T *data) { this->data = data; }
  T *data;
};

template <typename T, size_t COUNT>
class FixedArray
{
public:
  using value_type = T;
  constexpr __device__ size_t size() const { return COUNT; }
  // Return a FixedSlice that starts at start and has length size.
  // Check at compile time that it doesn't go out of the bounds of the
  // input.
  template <size_t size = COUNT>
  __device__ FixedSlice<T, size> slice(size_t start = 0)
  {
    static_assert(size <= COUNT, "The new slice should be smaller than the array.");
    expect_le(start + size, COUNT);
    return FixedSlice<T, size>(data + start);
  }
  inline __device__ T &operator[](size_t index)
  {
    expect_lt(index, COUNT);
    return data[index];
  }
  inline const __device__ T &operator[](size_t index) const
  {
    expect_lt(index, COUNT);
    return data[index];
  }
  inline volatile __device__ T &operator[](size_t index) volatile
  {
    expect_lt(index, COUNT);
    return data[index];
  }
  inline const volatile __device__ T &operator[](size_t index) const volatile
  {
    expect_lt(index, COUNT);
    return data[index];
  }
  inline __device__ void ruin()
  {
    for (size_t i = 0; i < sizeof(data); i++)
    {
      ((uint8_t *)data)[i] = 32;
    }
  }
  //__device__ inline T* get_raw_data() {
  //  return data;
  //}
  FixedArray() = default;
  FixedArray(const FixedArray &) = delete; // non construction-copyable
  FixedArray &operator=(const FixedArray &) = delete; // non copyable

private:
  T data[COUNT];
};

template <typename T, size_t COUNT>
class FixedRing
{
public:
  constexpr __device__ size_t size() { return COUNT; }
  inline __device__ T &operator[](size_t index) { return data[index % COUNT]; }
  inline const __device__ T &operator[](size_t index) const { return data[index % COUNT]; }
  inline __device__ T &at(size_t index) { return data[index % COUNT]; }
  inline const __device__ T &at(size_t index) const { return data[index % COUNT]; }
  FixedRing() = default;
  FixedRing(const FixedRing &) = delete; // non construction-copyable
  FixedRing &operator=(const FixedRing &) = delete; // non copyable

private:
  T data[COUNT];
};
