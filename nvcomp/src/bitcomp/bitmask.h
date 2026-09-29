#pragma once

namespace bitcomp
{

namespace bitmask
{
// *************************************************************************************************
// Build bitmask from lower bit of each byte using dot product
// Dot product factors (1, 2, 4, 8 : 16, 32, 64, 128)
// Assumes the input bytes are either 0 or 1 (single bit set),
// like afert a __vsetxx4() call
inline __device__ uint bitmask_dp4(uint &lo, uint &hi)
{
  uint tmp;
  // DP4 instructions only available on sm_62+
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 620
  asm("dp4a.u32.u32 %0, %1, 0x08040201, 0;" : "=r"(tmp) : "r"(lo));
  asm("dp4a.u32.u32 %0, %1, 0x80402010, %2;" : "=r"(tmp) : "r"(hi), "r"(tmp));
#else
  tmp = (lo + (lo >> 7) + (lo >> 14) + (lo >> 21) + ((hi + (hi >> 7) + (hi >> 14) + (hi >> 21)) << 4)) & 0xff;
#endif
  return tmp;
}

// *************************************************************************************************
// Returns an 8-bit mask if the 8 bytes of (lo,hi) are non-zero
inline __device__ uint nzbmask(const uint &lo, const uint &hi)
{
  // Set the lowest bit of each byte to 1 if the byte is nonzero
  uint lo_nz = __vsetne4(lo, 0);
  uint hi_nz = __vsetne4(hi, 0);
  return bitmask_dp4(lo_nz, hi_nz);
}

// *************************************************************************************************
// Returns an 8-bit mask if the 8 bytes of (lo,hi) are zero
inline __device__ uint zbmask(const uint &lo, const uint &hi)
{
  // Set the lowest bit of each byte to 1 if the byte is nonzero
  uint lo_nz = __vseteq4(lo, 0);
  uint hi_nz = __vseteq4(hi, 0);
  return bitmask_dp4(lo_nz, hi_nz);
}

// *************************************************************************************************
// Returns an 8-bit mask if the 8 bytes of (lo,hi) are 0xff
inline __device__ uint ffbmask(const uint &lo, const uint &hi)
{
  // Set the lowest bit of each byte to 1 if the byte is nonzero
  uint lo_nz = __vseteq4(lo, 0xffffffff);
  uint hi_nz = __vseteq4(hi, 0xffffffff);
  return bitmask_dp4(lo_nz, hi_nz);
}

// *************************************************************************************************
} // namespace bitmask
} // namespace bitcomp