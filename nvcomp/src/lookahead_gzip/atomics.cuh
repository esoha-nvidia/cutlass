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

#include <cuda/atomic>

#include <atomic>
#include <chrono>
#include <thread>

template <typename atomic_t, typename T>
__host__ T atomic_wait(atomic_t *atomic, T old_value, cuda::std::memory_order order, uint nanoseconds)
{
  T new_value;
  do
  {
    std::this_thread::sleep_for(std::chrono::nanoseconds(nanoseconds));
    new_value = atomic->load(order);
  } while (new_value == old_value);
  return new_value;
}

template <typename atomic_t, typename T>
__host__ T atomic_wait(
  atomic_t *atomic,
  T old_value,
  cuda::std::memory_order order,
  uint nanoseconds,
  std::atomic<bool> &notification,
  uint loads_before_notification
)
{
  T new_value;
  uint counter = 0;
  do
  {
    std::this_thread::sleep_for(std::chrono::nanoseconds(nanoseconds));
    new_value = atomic->load(order);
    counter = (counter + 1) % loads_before_notification;
    // Note:
    // Check notification, once counter reaches 0.
    // If notification returns true, we need to break
    // out of the loop.
  } while (new_value == old_value && (counter || !notification.load(std::memory_order_acquire)));
  return new_value;
}
