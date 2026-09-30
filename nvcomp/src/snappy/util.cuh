/*
* Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
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

#include "nvcomp/shared_types.h"
#ifdef __CUDACC__
#include "LZ77_decomp.cuh"
#endif

namespace snappy
{

/**
 *  @brief Extracts the uncompressed size from the compressed buffer.
 *  @param input_buffer Pointer to the compressed input buffer.
 *  @param input_size_bytes Size of compressed input buffer.
 *  @param header_size_bytes Number of compressed input bytes used to store the uncompressed size.
 *  @param error non-zero if an error was detected.
 */
inline __host__ __device__ uint32_t get_uncompressed_size(
  const uint8_t *input_buffer,
  const size_t input_size_bytes,
  uint32_t &header_size_bytes,
  int32_t &error
)
{
  header_size_bytes = 0;
  error = 0;
  uint32_t uncompressed_size = input_buffer[header_size_bytes++];

  if (uncompressed_size > 0x7f)
  {
    uint32_t c = (header_size_bytes < input_size_bytes) ? input_buffer[header_size_bytes++] : 0;
    uncompressed_size = (uncompressed_size & 0x7f) | (c << 7);

    if (uncompressed_size >= (0x80 << 7))
    {
      c = (header_size_bytes < input_size_bytes) ? input_buffer[header_size_bytes++] : 0;
      uncompressed_size = (uncompressed_size & ((0x7f << 7) | 0x7f)) | (c << 14);
      if (uncompressed_size >= (0x80 << 14))
      {
        c = (header_size_bytes < input_size_bytes) ? input_buffer[header_size_bytes++] : 0;
        uncompressed_size = (uncompressed_size & ((0x7f << 14) | (0x7f << 7) | 0x7f)) | (c << 21);
        if (uncompressed_size >= (0x80 << 21))
        {
          c = (header_size_bytes < input_size_bytes) ? input_buffer[header_size_bytes++] : 0;
          if (c < 0x8)
          {
            uncompressed_size = (uncompressed_size & ((0x7f << 21) | (0x7f << 14) | (0x7f << 7) | 0x7f)) | (c << 28);
          }
          else
          {
            error = -1;
          }
        }
      }
    }
  }

  return uncompressed_size;
}

#ifdef __CUDACC__
/**
*  @brief Checks if actual decoded & written sizes match expected
*  @param actual_input_size number of bytes decoded
*  @param actual_output_size number of bytes written to output buffer
*  @param expected_input_size size of compressed buffer that was given as parameter
*  @param device_out_bytes pointer to the expected output size, updated with the actual size on mismatch
*  @param nvcomp_status pointer to nvCOMP status, currently indicates failure
*/
inline __device__ void snappy_unsnap_err_check(
  uint32_t actual_input_size,
  uint32_t actual_output_size,
  uint32_t expected_input_size,
  const bool invalid_stream,
  uint64_t *device_out_bytes,
  nvcompStatus_t *const nvcomp_status
)
{
  if (!thread_warp_ix())
  {
    // If we found a zero offset, we know for sure that the compressed stream is invalid, so we can already set success to false in that case.
    bool success = !invalid_stream;

    // Make sure we decoded all bytes in the compressed stream
    if (expected_input_size != actual_input_size)
    {
      success = false;
    }

    // Make sure the total uncompressed size matches what was expected
    if (*device_out_bytes != actual_output_size)
    {
      *device_out_bytes = actual_output_size;
      success = false;
    }

    if (success)
    {
      *nvcomp_status = nvcompSuccess;
    }
    else
    {
      // nvcomp_status is already set to nvcompErrorCannotDecompress
    }
  }
}
#endif // __CUDACC__

} // namespace snappy
