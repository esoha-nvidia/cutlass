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

#include "common.h"
#include "modules_types.cuh"
#include "Reduction.cuh"

using nvcomp::roundUpDiv;
using nvcomp::roundUpTo;
using nvcomp::roundUpToAlignment;

namespace modules
{

/**
 * Helper function to calculate the frame of reference and bitwidth in
 * bitpacking layer.
 *
 * @param[in] input Array of size \p num_elements of the input elements.
 * @param[out] frame_of_reference Frame of reference in the bitpacking layer.
 * Currently this is the smallest element of \p input. This argument will be set
 * by thread 0 so the memory location should be accessible by thread 0 (e.g., in
 * shared memory).
 * @param[out] bitwidth_ptr The highest 16 bits of this field store the number
 * of bits needed in the bitpacked buffer to represent a single input element.
 * The lowest 16 bits store the number of elements. This argument will be set
 * by thread 0 so the memory location should be accessible by thread 0 (e.g., in
 * shared memory).
 */
template <typename data_type, typename size_type, int threadblock_size>
__device__ void
get_for_bitwidth(const data_type *input, size_type num_elements, data_type *frame_of_reference, uint32_t *bitwidth_ptr)
{
  // First, we calculate the maximum and the minimum of the input elements. We
  // process input elements in rounds, where each round processes
  // `threadblock_size` elements, with one element per thread.

  typedef nvcomp::cub::BlockReduce<data_type, threadblock_size> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  // Default value for coverity.
  // Real value is set in the if statement below.
  // Only variables from valid threads are used in BlockReduce below.
  data_type thread_data = 0;
  int num_valid = min(num_elements, static_cast<size_type>(threadblock_size));
  if (threadIdx.x < num_elements)
  {
    thread_data = input[threadIdx.x];
  }

  data_type minimum = BlockReduce(temp_storage).Reduce(thread_data, cub_minimum(), num_valid);
  __syncthreads();
  data_type maximum = BlockReduce(temp_storage).Reduce(thread_data, cub_maximum(), num_valid);
  __syncthreads();

  const int num_rounds = roundUpDiv(num_elements, threadblock_size);

  for (int round = 1; round < num_rounds; round++)
  {
    num_valid = min(num_elements - round * threadblock_size, static_cast<size_type>(threadblock_size));
    if (threadIdx.x < num_valid)
    {
      thread_data = input[threadIdx.x + round * threadblock_size];
    }

    const data_type local_min = BlockReduce(temp_storage).Reduce(thread_data, cub_minimum(), num_valid);
    __syncthreads();
    const data_type local_max = BlockReduce(temp_storage).Reduce(thread_data, cub_maximum(), num_valid);
    __syncthreads();

    if (threadIdx.x == 0 && local_min < minimum)
    {
      minimum = local_min;
    }
    if (threadIdx.x == 0 && local_max > maximum)
    {
      maximum = local_max;
    }
  }

  // Next, we store the frame of reference, the bitwidth and the number of
  // elements into the desired location.

  if (threadIdx.x == 0)
  {
    *frame_of_reference = minimum;

    uint32_t bitwidth;
    // calculate bit-width
    if (sizeof(data_type) > sizeof(int))
    {
      const long long int range = static_cast<uint64_t>(maximum) - static_cast<uint64_t>(minimum);
      // need 64 bit clz
      bitwidth = sizeof(long long int) * num_bits_per_byte - __clzll(range);
    }
    else
    {
      const int range = static_cast<uint32_t>(maximum) - static_cast<uint32_t>(minimum);
      // can use 32 bit clz
      bitwidth = sizeof(int) * num_bits_per_byte - __clz(range);
    }
    *bitwidth_ptr = (bitwidth << 16) | static_cast<uint32_t>(num_elements);
  }
}

/**
 * Perform bitpacking on a single threadblock.
 *
 * @param[in] input Uncompressed input buffer.
 * @param[in] num_elements Number of input elements in \p input.
 * @param[in] data_is_deltas Whether the data pointed to by input is deltas.
 * @param[out] output Bitpacked data including metadata.
 * @param[out] out_bytes Size of the bitpacked data in bytes. This argument
 * should be unique to each thread.
 */
template <typename data_type, typename size_type, int threadblock_size>
__device__ void block_bitpack(
  const data_type *input,
  size_type num_elements,
  bool data_is_deltas,
  uint32_t *output,
  size_type *out_bytes
)
{
  // First, we need to use unsigned type during bitpacking because bit-shift on
  // negative values has undefined behavior. Next, we need to consider two
  // cases. If the bitwidth is larger than 32 bits, we need to use the unsigned
  // version of the `data_type` to make sure the difference between the data and
  // FOR can fit. If bitwidth is smaller than 32 bits, we need to use `uint32_t`
  // instead of the `data_type` to avoid left-shift beyond `data_type` limit.
  using unsigned_data_type = std::make_unsigned_t<data_type>;
  using padded_data_type = larger_t<unsigned_data_type, uint32_t>;

  auto for_ptr = reinterpret_cast<data_type *>(output);
  uint32_t *current_ptr = roundUpToAlignment<uint32_t>(for_ptr + 1);

  // Use signed data type if input could store negative values, e.g. the
  // output of delta layer. Although the signed type and the unsigned type have
  // the same raw bits, the interpretation of the smallest element is different
  // for negative values. If the data type is already signed, it can be checked
  // at compile time, so we can avoid the runtime check.
  const bool use_signed = std::is_signed<data_type>() || data_is_deltas;
  if (use_signed)
  {
    using signed_data_type = std::make_signed_t<data_type>;
    get_for_bitwidth<signed_data_type, size_type, threadblock_size>(
      reinterpret_cast<const signed_data_type *>(input),
      num_elements,
      reinterpret_cast<signed_data_type *>(for_ptr),
      current_ptr
    );
  }
  else
  {
    get_for_bitwidth<data_type, size_type, threadblock_size>(input, num_elements, for_ptr, current_ptr);
  }

  __syncthreads();

  const data_type frame_of_reference = *for_ptr;
  const uint32_t bitwidth = (*current_ptr & 0xFFFF0000) >> 16;
  current_ptr = reinterpret_cast<uint32_t *>(roundUpToAlignment<data_type>(current_ptr + 1));

  const int num_output_elements = roundUpDiv(num_elements * bitwidth, sizeof(uint32_t) * num_bits_per_byte);
  if (out_bytes != nullptr)
  {
    // The bitpacking metadata consists of FOR (data_type size) and bitwidth and
    // the number of elements (4B). The start of the bitpacking data needs to be
    // both 4B and data_type aligned.
    *out_bytes = roundUpTo(sizeof(data_type) + 4, max(static_cast<size_t>(4), sizeof(data_type))) +
                 num_output_elements * sizeof(uint32_t);
  }

  for (int out_idx = threadIdx.x; out_idx < num_output_elements; out_idx += threadblock_size)
  {
    // The kernel works by assigning each 4B element of the bitpacked data to a
    // thread. The kernel then iterates over chunks of input, filling the bits
    // for each element, and then writing the stored bits to the output.
    const int out_bit_start = out_idx * sizeof(uint32_t) * num_bits_per_byte;
    const int out_bit_end = out_bit_start + sizeof(uint32_t) * num_bits_per_byte;
    const int input_idx_start = out_bit_start / bitwidth;
    const int input_idx_end = roundUpDiv(out_bit_end, bitwidth);

    uint32_t output_val = 0;
    for (int input_idx = input_idx_start; input_idx < input_idx_end; input_idx++)
    {
      unsigned_data_type input_val = 0;
      if (input_idx < num_elements)
      {
        input_val = static_cast<unsigned_data_type>(input[input_idx] - frame_of_reference);
      }

      auto padded_val = static_cast<padded_data_type>(input_val);
      const int offset = input_idx * bitwidth - out_bit_start;
      if (offset > 0)
      {
        padded_val <<= offset;
      }
      else
      {
        padded_val >>= -offset;
      }
      output_val |= static_cast<uint32_t>(padded_val);
    }
    current_ptr[out_idx] = output_val;
  }
}

/**
 * Perform bitunpacking on a single threadblock.
 *
 * @param[in] input Bitpacked input data.
 * @param[out] output Unpacked data.
 * @param[out] out_num_elements Number of `data_type` elements in the unpacked
 * data. This argument should be unique to each thread.
 */
template <typename data_type, typename size_type>
__device__ void block_bitunpack(const uint32_t *input, data_type *output, size_type *out_num_elements)
{
  const data_type *for_ptr = reinterpret_cast<const data_type *>(input);
  const uint32_t *current_ptr = roundUpToAlignment<uint32_t>(for_ptr + 1);

  const data_type frame_of_reference = *for_ptr;
  const uint32_t bitwidth = (*current_ptr & 0xFFFF0000) >> 16;
  const uint32_t num_elements = *current_ptr & 0x0000FFFF;

  if (out_num_elements != nullptr)
  {
    *out_num_elements = static_cast<size_type>(num_elements);
  }

  // Casting to unsigned type since bit shifting on negative numbers is
  // undefined.
  typedef typename std::make_unsigned<data_type>::type unsigned_type;
  const unsigned_type *data_ptr = roundUpToAlignment<unsigned_type>(current_ptr + 1);

  // Assign each output `data_type` element to a thread.
  for (int out_idx = threadIdx.x; out_idx < num_elements; out_idx += blockDim.x)
  {
    if (bitwidth == 0)
    {
      output[out_idx] = frame_of_reference;
    }
    else
    {
      // Shifting by width of the type is UB
      const unsigned_type mask = bitwidth < sizeof(unsigned_type) * num_bits_per_byte
                                   ? (static_cast<unsigned_type>(1) << bitwidth) - 1
                                   : static_cast<unsigned_type>(-1);

      // The current output element needs bits from at most two `data_type`
      // input elements, with indices `low_idx` and `high_idx`.
      constexpr size_t num_bits_data_type = sizeof(data_type) * num_bits_per_byte;
      const int low_idx = (out_idx * bitwidth) / num_bits_data_type;
      const int high_idx = (out_idx + 1) * bitwidth / num_bits_data_type;
      const int offset = out_idx * bitwidth - low_idx * num_bits_data_type;

      // Load and shift bits from `low_idx` input element
      unsigned_type base_value = data_ptr[low_idx] >> offset;

      if (low_idx < high_idx && offset != 0)
      {
        // If the current output element crosses input element boundary (i.e. it
        // corresponds to two input elements), we load and shift from `high_idx
        // as well.
        base_value += data_ptr[high_idx] << (num_bits_data_type - offset);
      }

      base_value &= mask;

      output[out_idx] = static_cast<data_type>(base_value) + frame_of_reference;
    }
  }
}

} // namespace modules