/*
* Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

#include "CudaConstants.h"
#include "nvcomp/snappy.h"

namespace snappy
{

// This BIG_OFFSET is used to distinguish between literal and match copies
// in the Four Bytes-per-Byte (4BpB) format.
#define BIG_OFFSET (nvcompSnappyDecompressionMaxAllowedChunkSize + 1)

// 4BpB format requires using one bit to distinguish literals from matches
// This is why we cannot support the Snappy format max chunk size of 4GB - 1
static_assert(BIG_OFFSET > nvcompSnappyDecompressionMaxAllowedChunkSize);
static_assert(BIG_OFFSET + nvcompSnappyDecompressionMaxAllowedChunkSize <= ((1ull << 32) - 1));

// Duration of HW sleep when waiting for other warp
// These do not seem to matter at all
constexpr uint32_t LONG_SLEEP_DURATION = 100;
constexpr uint32_t SHORT_SLEEP_DURATION = 50;

constexpr uint32_t SLEEP_DURATION_FINDER = LONG_SLEEP_DURATION; // waits when mapper has 128 symbols to map
constexpr uint32_t SLEEP_DURATION_MAPPER_NO_SYMBOL = LONG_SLEEP_DURATION; // waits when 0 symbols ready to map
constexpr uint32_t SLEEP_DURATION_MAPPER_NO_WINDOW = SHORT_SLEEP_DURATION; // waits when processor has 2 windows to g&w
constexpr uint32_t SLEEP_DURATION_PROCESSOR = LONG_SLEEP_DURATION; // waits when 0 maps ready

// Compressed Buffer is loaded in INPUT_WINDOW_SIZE chunks
constexpr uint32_t INPUT_WINDOW_SIZE = 256;

// 64B are preserved after each input window
// This allows 32 candidate symbol locations to be decoded
// Each requiring up to 5B. (needs 36B total)
constexpr uint32_t PREFETCH_SKIN_SIZE = 64;
constexpr uint32_t PREFETCH_SECTOR_SIZE = INPUT_WINDOW_SIZE + PREFETCH_SKIN_SIZE; // 64 + 256
constexpr uint32_t PREFETCH_SCRATCH_SIZE = 2 * PREFETCH_SECTOR_SIZE;

// SYMBOL_RING used for communication from Warp Finder to Warp Mapper
constexpr uint32_t SYMBOL_RING_SIZE = 256; // 8B per slot

// 4BpB format is initialized for 128B slices of Warp Mapper to Warp Processor
// decompressed output.
constexpr uint32_t OUTPUT_WINDOW_SIZE_PWR = 8;
constexpr uint32_t OUTPUT_WINDOW_SIZE = (1 << OUTPUT_WINDOW_SIZE_PWR);
static_assert(OUTPUT_WINDOW_SIZE % WARP_SIZE_U == 0);
static_assert(OUTPUT_WINDOW_SIZE < (1 << 16)); // current map strategy requires all cursors be 16b
static_assert(BIG_OFFSET > OUTPUT_WINDOW_SIZE);

// WINDOW_RING used for communication from Warp Mapper to Warp Processor
// If I had more shmem budget this is where I would put it
constexpr uint32_t WINDOW_RING_SIZE = 2;

// Snappy symbols have a 2-bit tag
#define TAG_BIT_MASK 3
#define NUM_TAG_BITS 2

// Masks for integers of various sizes
#define ONE_BYTE_MASK 255
#define TWO_BYTE_MASK 65535

} // namespace snappy