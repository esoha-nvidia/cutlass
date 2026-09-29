/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

#include <cstdlib>
#include <string>

// Recognized environment variables
#define USE_HW_DECOMPRESSION_ENV "NVCOMP_HW_DECOMPRESSION"
#define PINNED_POOL_SIZE_ENV "NVCOMP_PINNED_POOL_SIZE"
#define CHECK_LZ4_CORRECTNESS_ENV "NVCOMP_CHECK_LZ4_CORRECTNESS"
#define LZ4_CORRECTNESS_LOG_OUTPUT_ENV "NVCOMP_LZ4_CORRECTNESS_LOG_OUTPUT"
#define CHECK_SNAPPY_CORRECTNESS_ENV "NVCOMP_CHECK_SNAPPY_CORRECTNESS"
#define SNAPPY_CORRECTNESS_LOG_OUTPUT_ENV "NVCOMP_SNAPPY_CORRECTNESS_LOG_OUTPUT"
#define CHECK_DEFLATE_CORRECTNESS_ENV "NVCOMP_CHECK_DEFLATE_CORRECTNESS"
#define DEFLATE_CORRECTNESS_LOG_OUTPUT_ENV "NVCOMP_DEFLATE_CORRECTNESS_LOG_OUTPUT"
#define ZSTD_USE_SINGLE_CTA_FOR_LZ_COMPRESS_ENV "NVCOMP_ZSTD_USE_SINGLE_CTA_FOR_LZ_COMPRESS"
#define DATA_LOGGING_DIRECTORY_ENV "NVCOMP_DATA_LOGGING_DIRECTORY"

namespace nvcomp
{

[[maybe_unused]] static std::string getenv(const char *name)
{
#if defined(_MSC_VER)
  // Note:
  // We need to use the safe version on Windows, otherwise we are presented
  // with a warning (that is converted to an error).
  size_t len = 0;
  char buf[128];
  bool success = ::getenv_s(&len, buf, sizeof(buf), name) == 0;
  return success ? buf : std::string{};
#else
  char *buf = ::getenv(name);
  return buf ? buf : std::string{};
#endif // defined(_MSC_VER)
}

[[maybe_unused]] static int setenv(const char *name, const char *value)
{
#if defined(_MSC_VER)
  // Note:
  // On Windows, we use _putenv_s which always overwrites if the variable exists.
  // Windows doesn't have a direct equivalent to the POSIX setenv overwrite behavior.
  return ::_putenv_s(name, value);
#else
  return ::setenv(name, value, 1);
#endif // defined(_MSC_VER)
}

// Class analogous to DeviceGuard that sets and resets an
// environment variable in a RAII fashion.
class EnvironmentGuard
{
public:
  EnvironmentGuard(const char *name, const char *set_value, const char *reset_value) noexcept
      : name_(name)
      , reset_value_(reset_value)
  {
    ::nvcomp::setenv(name, set_value);
  }

  EnvironmentGuard(const EnvironmentGuard &other) noexcept = delete;
  EnvironmentGuard(EnvironmentGuard &&other) noexcept = delete;

  EnvironmentGuard &operator=(const EnvironmentGuard &other) noexcept = delete;
  EnvironmentGuard &operator=(EnvironmentGuard &&other) noexcept = delete;

  ~EnvironmentGuard() noexcept { ::nvcomp::setenv(name_, reset_value_); }

private:
  const char *name_;
  const char *reset_value_;
};

} // namespace nvcomp