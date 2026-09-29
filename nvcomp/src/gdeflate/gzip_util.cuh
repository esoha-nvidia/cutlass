/*
* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

#include <cstddef>
#include <cstdint>

// GZIP header flags (RFC 1952)
static constexpr uint32_t GZ_FLG_FTEXT = 0x01; // ASCII text hint
static constexpr uint32_t GZ_FLG_FHCRC = 0x02; // Header CRC present
static constexpr uint32_t GZ_FLG_FEXTRA = 0x04; // Extra fields present
static constexpr uint32_t GZ_FLG_FNAME = 0x08; // Original file name present
static constexpr uint32_t GZ_FLG_FCOMMENT = 0x10; // Comment present

namespace deflate
{

/**
  * @brief Parse GZIP header (RFC 1952).
  * @return Header length in bytes, or -1 on error/invalid header.
  */
inline __host__ __device__ int parse_gzip_header(const uint8_t *src, size_t src_size)
{
  int hdr_len = -1;
  if (src_size >= 18)
  {
    uint32_t sig = (src[0] << 16) | (src[1] << 8) | src[2];
    if (sig == 0x1f8b08) // 24-bit GZIP inflate signature {0x1f, 0x8b, 0x08}
    {
      uint32_t flags = src[3];
      hdr_len = 10;
      if (flags & GZ_FLG_FEXTRA)
      {
        int xlen = src[hdr_len] | (src[hdr_len + 1] << 8);
        hdr_len += 2 + xlen;
        if (static_cast<size_t>(hdr_len) >= src_size)
        {
          return -1;
        }
      }
      if (flags & GZ_FLG_FNAME)
      {
        do
        {
          if (static_cast<size_t>(hdr_len) >= src_size)
          {
            return -1;
          }
        } while (src[hdr_len++] != 0);
      }
      if (flags & GZ_FLG_FCOMMENT)
      {
        do
        {
          if (static_cast<size_t>(hdr_len) >= src_size)
          {
            return -1;
          }
        } while (src[hdr_len++] != 0);
      }
      if (flags & GZ_FLG_FHCRC)
      {
        hdr_len += 2;
      }
      if (static_cast<size_t>(hdr_len) + 8 > src_size)
      {
        hdr_len = -1;
      }
      if ((flags & 0xE0) != 0)
      { // Reserved flags must be 0
        hdr_len = -1;
      }
    }
  }
  return hdr_len;
}

} // namespace deflate