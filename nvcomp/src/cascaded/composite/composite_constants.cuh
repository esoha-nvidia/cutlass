/*
 * Copyright (c) 2021-2026, NVIDIA CORPORATION. All rights reserved.
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

#include "nvcomp/utils.hpp"

namespace composite
{

constexpr int default_chunk_size = 4096;

constexpr int max_chunk_metadata_size = 64;

constexpr int max_num_rle_layers = 7;

// Resulting function from rearranging the formula in "get_chunk_metadata_size"
// to solve for the number of delta layers, given the max chunk metadata size and the number of RLE layers.
constexpr int get_max_num_delta_layers(int num_RLEs, int max_chunk_metadata_size)
{
  return static_cast<int>(
    (max_chunk_metadata_size - nvcomp::roundUpTo(static_cast<size_t>(4 + 4 * (num_RLEs + 1)), sizeof(size_t))) /
    sizeof(size_t)
  );
}

// Since max number of RLE layers is 7, and max_chunk_metadata_size is 64,
// the function get_max_num_delta_layers constrains the max number of delta layers to be 3, assuming the data has a maximum size of 8B.
constexpr int max_num_delta_layers = get_max_num_delta_layers(max_num_rle_layers, max_chunk_metadata_size);

constexpr int composite_compress_threadblock_size = 128;
constexpr int composite_decompress_threadblock_size = 128;

} // namespace composite