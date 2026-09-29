/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
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
#include "LZ4Types.cuh"

namespace nvcomp
{

// Detail type helpers -- used to reference 4 / 8-byte values
using word_type = uint32_t;
using double_word_type = uint64_t;

/**
 * @brief The number of threads to use per chunk in compression.
 */
const int LZ4_COMP_THREADS_PER_CHUNK = WARP_SIZE;

/**
 * @brief The number of bytes to bitshuffle in a single thread using vectorized operations.
 */
const int LZ4_BITSHUFFLE_BYTES_PER_THREAD = 32;
const int LZ4_INVERSE_BITSHUFFLE_BYTES_PER_THREAD = 32;

/**
 * @brief The number of bytes to process per thread per bitplane.
 */
const int LZ4_BITSHUFFLE_BYTES_PER_THREAD_PER_BITPLANE = 4;

/**
 * @brief Each bitshuffeable block should be a multiple of this value.
 */
const int LZ4_BITSHUFFLE_BLOCK_ELEMENTS_MULTIPLIER = 8;

/**
 * @brief The maximum number of bytes to bitshuffle in a single full block.
 */
const int LZ4_BITSHUFFLE_MAX_BLOCK_SIZE = 8192;

/**
 * @brief The number of threads to use per bitshuffle single full block.
 */
const int LZ4_BITSHUFFLE_THREADS_PER_BLOCK = 256;

/**
 * @brief The size of the shared memory buffer (in uint4 units) to use per bitshuffle block.
 */
const int LZ4_BITSHUFFLE_SHARED_MEMORY_SIZE_IN_UINT4 = 512;

/**
 * @brief The number of chunks to compression concurrently per threadblock.
 */
constexpr int LZ4_COMP_CHUNKS_PER_BLOCK = 1;

/**
 * @brief The number of threads to use per chunk in decompression.
 */
const int LZ4_DECOMP_THREADS_PER_CHUNK = WARP_SIZE;

/**
 * @brief The number of chunks to decompress concurrently per threadblock.
 */
constexpr int LZ4_DECOMP_CHUNKS_PER_BLOCK = 2;

/**
 * @brief The size of the shared memory buffer to use per decompression stream.
 */
constexpr const lz4::position_type DECOMP_INPUT_BUFFER_SIZE = lz4::LZ4DecompressWarpMemory::BUFFER_SIZE;

/**
 * @brief The threshold of reading from the buffer during decompression, that
 * more data will be loaded into the buffer and its contents shifted.
 */
constexpr const lz4::position_type DECOMP_BUFFER_PREFETCH_DIST = DECOMP_INPUT_BUFFER_SIZE / 2;

/**
 * @brief The value used to explicitly represent an invalid offset. This
 * denotes an empty slot in the hashtable.
 */
constexpr const lz4::offset_type NULL_OFFSET = static_cast<lz4::offset_type>(-1);

/**
 * @brief The maximum size of a valid offset.
 */
constexpr const lz4::position_type MAX_OFFSET = (1U << 16) - 1;

/**
 * @brief The last 5 bytes of input are always literals.
 * @brief The last match must start at least 12 bytes before the end of block.
 */
constexpr const uint8_t MIN_ENDING_LITERALS_BYTES = 5;
constexpr const uint8_t LAST_VALID_MATCH_BYTES = 12;

} // namespace nvcomp