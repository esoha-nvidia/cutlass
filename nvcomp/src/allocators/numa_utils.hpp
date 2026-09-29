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

#if defined(_WIN32)
#include <Windows.h>
#elif defined(__linux__)
#include <cerrno>
#include <cstring>

#include <sys/syscall.h>
#include <unistd.h>
#else
#error "Determining the current NUMA node is not implemented on this platform."
#endif

#include "nvcomp.hpp"

namespace nvcomp
{

static int get_current_numa_node()
{
#if defined(_WIN32)
  PROCESSOR_NUMBER processor{};
  GetCurrentProcessorNumberEx(&processor);

  USHORT numa_node = 0;
  if (GetNumaProcessorNodeEx(&processor, &numa_node) == 0)
  {
    throw NVCompException(
      nvcompErrorInternal,
      "Failed to determine the current NUMA node: " + std::to_string(GetLastError())
    );
  }
  return static_cast<int>(numa_node);
#else
  unsigned int numa_node = 0;
  if (syscall(SYS_getcpu, nullptr, &numa_node, nullptr) == -1)
  {
    throw NVCompException(
      nvcompErrorInternal,
      "Failed to determine the current NUMA node: " + std::string(strerror(errno))
    );
  }
  return static_cast<int>(numa_node);
#endif
}

} // namespace nvcomp