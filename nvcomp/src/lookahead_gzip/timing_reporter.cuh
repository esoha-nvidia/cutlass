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

#include <cstdio>

namespace lookahead_gzip
{

class NullTimingReporter
{
public:
  enum class Timer
  {
    TOTAL,
    WAIT_FOR_HUFFMANS,
    COPY_HUFFMANS,
    WAIT_FOR_INPUT,
    DECODE_ONE,
    RESOLVE_PRUNE,
    GET_FINAL,
    COMPLETE_PRESCAN,
    GET_UNCOMPRESSED_SIZE,
    WAIT_FOR_ROOM,
    UNCOMPRESSED_SHMEM,
    BLOCK_LZ77,
    GLOBAL_LZ77,
    NUM_TIMERS,
  };

  __device__ void report_timing()
  {
    // Do nothing.
  }

  __device__ void toggle(Timer, uint = 0, uint = 0)
  {
    // Do nothing.
  }

  __device__ void reset(uint = 0, uint = 0)
  {
    // Do nothing.
  }
};

class TimingReporter : public NullTimingReporter
{
public:
  void report_timing()
  {
    fprintf(
      stderr,
      "total: %zd\n"
      "\twait for huffmans: %zd\n"
      "\tcopy huffmans: %zd\n"
      "\twait for input: %zd\n"
      "\tdecode one: %zd\n"
      "\tresolve prune: %zd\n"
      "\tget final: %zd\n"
      "\tcomplete prescan: %zd\n"
      "\tget uncompressed size: %zd\n"
      "\twait for room: %zd\n"
      "\tuncompressed shmem: %zd\n"
      "\tblock lz77: %zd\n"
      "\tglobal lz77: %zd\n",
      get(Timer::TOTAL) / 1000000,
      get(Timer::WAIT_FOR_HUFFMANS) / 1000000,
      get(Timer::COPY_HUFFMANS) / 1000000,
      get(Timer::WAIT_FOR_INPUT) / 1000000,
      get(Timer::DECODE_ONE) / 1000000,
      get(Timer::RESOLVE_PRUNE) / 1000000,
      get(Timer::GET_FINAL) / 1000000,
      get(Timer::COMPLETE_PRESCAN) / 1000000,
      get(Timer::GET_UNCOMPRESSED_SIZE) / 1000000,
      get(Timer::WAIT_FOR_ROOM) / 1000000,
      get(Timer::UNCOMPRESSED_SHMEM) / 1000000,
      get(Timer::BLOCK_LZ77) / 1000000,
      get(Timer::GLOBAL_LZ77) / 1000000
    );
  }

  __host__ __device__ size_t get(Timer t) { return timers[static_cast<unsigned long>(t)]; }

  __device__ void toggle(Timer t, uint block = 0, uint thread = 0)
  {
    if (threadIdx.x == thread && blockIdx.x == block)
    {
      timers[static_cast<unsigned long>(t)] = clock64() - get(t);
    }
  }

  __device__ void reset(uint block = 0, uint thread = 0)
  {
    if (threadIdx.x == thread && blockIdx.x == block)
    {
      for (uint i = 0; i < sizeof(timers) / sizeof(timers[0]); i++)
      {
        timers[i] = 0;
      }
    }
  }

private:
  uint64_t timers[static_cast<unsigned long>(Timer::NUM_TIMERS)];
};

#if TIMING
__device__ TimingReporter timing_reporter;
#else
__device__ NullTimingReporter timing_reporter;
#endif

} // namespace lookahead_gzip
