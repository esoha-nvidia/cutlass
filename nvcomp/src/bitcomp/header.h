#include <cassert>

#include "bitcomp_private.h"

#pragma once

namespace bitcomp
{
namespace header
{

// ****************************************************************************
//
// The compressed data starts with the header, which contains both global
// and block-specific information.
//
// First 256 bits : Flags and global compressed info
// [0   :  63] : Flags & compressed method details, starting with "NVZ" magic number
//             : [ 0: 7] = "N"
//             : [ 8:15] = "V"
//             : [16:23] = "Z"
//             : [24:31] = data type (from bitcompDataType_t)
//             : [32:35] = Compressed mode (from bitcompMode_t)
//             : [36:39] = Algorithm used (from bitcompAlgorithm_t, RLE, ZBM)
//             : [40:43] = Integer format (from bitcompIntFormat_t)
//             : [44:64] = Reserved for later.
// [64  : 127] : (uint64) Compressed size (not counting the header and offset table)
// [128 : 191] : (uint 64) Original uncompressed size
// [192 : 255] : (fp16, fp32 or fp64) Scaling delta
//
// The offset table is stored right after the first 256 bits, with one 64-bit entry for
// every 8KB block (even incomplete) of uncompressed data.
// Each 64-bit offset entry (uint64) contains:
//   Bits [0:50] = block offset (in 4-byte words), counting from the end of the header.
//   Bits [51:62] = compressed block size, in 4-byte words
//   Bit 63 : Overflow/incompressible flag
//
// The header must be aligned on at least 64-bit boundaries.

// ****************************************************************************
// Various sections of the header

const int globalHeaderValues = 4; // Number of 64-bit global values in the header

template <typename T>
__forceinline__ __host__ __device__ T *getFlagsAddress(T *header)
{
  return header;
}
template <typename T>
__forceinline__ __host__ __device__ T *getCompressedLengthAddress(T *header)
{
  return reinterpret_cast<T *>((uint64 *)header + 1);
}
template <typename T>
__forceinline__ __host__ __device__ T *getUncompressedLengthAddress(T *header)
{
  return reinterpret_cast<T *>((uint64 *)header + 2);
}
template <typename T>
__forceinline__ __host__ __device__ T *getScalarAddress(T *header)
{
  return reinterpret_cast<T *>((uint64 *)header + 3);
}

// ****************************************************************************
// Compressed, uncompressed, header sizes

// Number of blocks, from the uncompressed size
__forceinline__ __host__ __device__ uint64 computeNumBlocks(uint64 uncompressedSize)
{
  return ((uncompressedSize + 8191) >> 13);
}

// Number of blocks, from the uncompressed size in the header
__forceinline__ __host__ __device__ uint64 getNumBlocks(const void *header)
{
  const uint64 *hdr64 = reinterpret_cast<const uint64 *>(getUncompressedLengthAddress(header));
  return computeNumBlocks(*hdr64);
}

// Write the uncompressed size in the header
__forceinline__ __host__ __device__ void setUncompressedSize(void *header, uint64 size)
{
  uint64 *hdr64 = reinterpret_cast<uint64 *>(getUncompressedLengthAddress(header));
  *hdr64 = size;
}

// Read the uncompressed size in the header
__forceinline__ __host__ __device__ uint64 getUncompressedSize(const void *header)
{
  const uint64 *hdr64 = reinterpret_cast<const uint64 *>(getUncompressedLengthAddress(header));
  return (*hdr64);
}

// Total header length in words, from the uncompressed size
__forceinline__ __host__ __device__ uint64 computeHeaderLengthInWords(uint64 uncompressedSize)
{
  return ((globalHeaderValues + computeNumBlocks(uncompressedSize)) * 2);
}

// Total header length in bytes, from the uncompressed size
__forceinline__ __host__ __device__ uint64 computeHeaderLength(uint64 uncompressedSize)
{
  return ((globalHeaderValues + computeNumBlocks(uncompressedSize)) * sizeof(uint64));
}

// Total header length in bytes, from the uncompressed size in the header
__forceinline__ __host__ __device__ uint64 getHeaderLengthInWords(const void *header)
{
  const uint64 *hdr64 = reinterpret_cast<const uint64 *>(getUncompressedLengthAddress(header));
  return (computeHeaderLengthInWords(*hdr64));
}

// Total header length in bytes, from the uncompressed size in the header
__forceinline__ __host__ __device__ uint64 getHeaderLength(const void *header)
{
  const uint64 *hdr64 = reinterpret_cast<const uint64 *>(getUncompressedLengthAddress(header));
  return (computeHeaderLength(*hdr64));
}

// Total compressed size, in bytes
__forceinline__ __host__ __device__ uint64 getTotalCompressedSize(const uint64 nblocks, const uint64 compsize)
{
  // 4 x 64-bit header + nblocks x 64-bit offsets + compressed data (in bytes instead of words)
  return ((globalHeaderValues + nblocks) * sizeof(uint64) + compsize * 4);
}
__forceinline__ __host__ __device__ uint64 getTotalCompressedSize(const void *header)
{
  uint64 nblocks = getNumBlocks(header);
  const uint64 *hdr64 = reinterpret_cast<const uint64 *>(getCompressedLengthAddress(header));
  return getTotalCompressedSize(nblocks, hdr64[0]);
}
// ****************************************************************************
// Compression flags

__forceinline__ __host__ __device__ uint getMagicNumber(const void *header)
{
  const uint *flags = reinterpret_cast<const uint *>(getFlagsAddress(header));
  return (flags[0] & 0xffffff);
}

__forceinline__ __host__ __device__ bitcompDataType_t getDataType(const void *header)
{
  const uint *flags = reinterpret_cast<const uint *>(getFlagsAddress(header));
  return (bitcompDataType_t)(flags[0] >> 24);
}

__forceinline__ __host__ __device__ bitcompMode_t getCompMode(const void *header)
{
  const uint *flags = reinterpret_cast<const uint *>(getFlagsAddress(header));
  return (bitcompMode_t)(flags[1] & 0xf);
}

__forceinline__ __host__ __device__ bitcompAlgorithm_t getAlgorithm(const void *header)
{
  const uint *flags = reinterpret_cast<const uint *>(getFlagsAddress(header));
  return (bitcompAlgorithm_t)((flags[1] >> 4) & 0xf);
}

__forceinline__ __host__ __device__ bitcompIntFormat_t getIntFormat(const void *header)
{
  const uint *flags = reinterpret_cast<const uint *>(getFlagsAddress(header));
  return (bitcompIntFormat_t)((flags[1] >> 8) & 0xf);
}

template <bitcompDataType_t dataType, bitcompMode_t compMode, bitcompAlgorithm_t algo, bitcompIntFormat_t ifmt>
__forceinline__ __host__ __device__ void computeFlags(uint *flags)
{
  flags[0] = 0x5a564e + // will read "NVZ"
             (dataType << 24);
  flags[1] = (compMode & 0xf) + ((algo << 4) & 0xf0) + ((ifmt << 8) & 0xf00);
}

__forceinline__ __host__ __device__ void computeFlags(
  uint *flags,
  bitcompDataType_t dataType,
  bitcompMode_t compMode,
  bitcompAlgorithm_t algo,
  bitcompIntFormat_t ifmt
)
{
  flags[0] = 0x5a564e + // will read "NVZ"
             (dataType << 24);
  flags[1] = (compMode & 0xf) + ((algo << 4) & 0xf0) + ((ifmt << 8) & 0xf00);
}

template <bitcompDataType_t dataType, bitcompMode_t compMode, bitcompAlgorithm_t algo, bitcompIntFormat_t ifmt>
__forceinline__ __host__ __device__ void setFlags(void *header)
{
  uint *flags = reinterpret_cast<uint *>(getFlagsAddress(header));
  computeFlags<dataType, compMode, algo, ifmt>(flags);
}

__forceinline__ __host__ __device__ void setFlags(
  void *header,
  bitcompDataType_t dataType,
  bitcompMode_t compMode,
  bitcompAlgorithm_t algo,
  bitcompIntFormat_t ifmt
)
{
  uint *flags = reinterpret_cast<uint *>(getFlagsAddress(header));
  computeFlags(flags, dataType, compMode, algo, ifmt);
}

// Verify header flags
template <bitcompDataType_t dataType, bitcompMode_t compMode, bitcompAlgorithm_t algo, bitcompIntFormat_t ifmt>
__forceinline__ __host__ __device__ bool hasValidFlags(const void *header)
{
  const uint *flags = reinterpret_cast<const uint *>(getFlagsAddress(header));
  uint tmp[2];
  computeFlags<dataType, compMode, algo, ifmt>(tmp);
  return (tmp[0] == flags[0] && tmp[1] == flags[1]);
}
__forceinline__ __host__ __device__ bool hasValidFlags(
  const void *header,
  bitcompDataType_t dataType,
  bitcompMode_t compMode,
  bitcompAlgorithm_t algo,
  bitcompIntFormat_t ifmt
)
{
  const uint *flags = reinterpret_cast<const uint *>(getFlagsAddress(header));
  uint tmp[2];
  computeFlags(tmp, dataType, compMode, algo, ifmt);
  return (tmp[0] == flags[0] && tmp[1] == flags[1]);
}

__forceinline__ __host__ __device__ bool hasValidMagicNumber(const void *header)
{
  const uint *flags = reinterpret_cast<const uint *>(getFlagsAddress(header));
  uint res = getMagicNumber(flags);
  return (res == 0x5a564e);
}

// ****************************************************************************
// Floating point scaling

template <typename T>
__forceinline__ __host__ __device__ T getScalingDelta(const void *header)
{
  const T *scaleptr = reinterpret_cast<const T *>(getScalarAddress(header));
  return (*scaleptr);
}

// Write the scaling delta
template <typename T>
__forceinline__ __host__ __device__ void setScalingDelta(void *header, T scale)
{
  // Note:
  // The largest scaling type is a 64-bit double, hence
  // we are constructing a 64-bit wide type for proper initialization for all T
  uint64_t tmp = 0;
  memcpy(&tmp, &scale, sizeof(T));
  // Always writing out 64 bits, irrespective of type T
  auto scaleptr = getScalarAddress(header);
  assert((((uintptr_t)scaleptr) & 0x7) == 0);
  *reinterpret_cast<uint64_t *>(scaleptr) = tmp;
}

// ****************************************************************************
// Block information (offset, size, overflow/incompressible flag)

// Write the block info, no overflow
__forceinline__ __host__ __device__ void setBlockInfo(void *hdr, uint iblock, uint64 &offset, int lcomp)
{
  uint64 *hdr64 = reinterpret_cast<uint64 *>(hdr);
  // 1-bit overflow flag + 12 bits lcomp + 51 bits offset
  hdr64[globalHeaderValues + iblock] = offset | (static_cast<uint64>(lcomp) << 51);
}

// Write the block info, with overflow
__forceinline__ __host__ __device__ void setBlockInfoOverflow(void *hdr, uint iblock, uint64 &offset, uint lcomp)
{
  uint64 *hdr64 = reinterpret_cast<uint64 *>(hdr);
  // 1-bit overflow flag + 12 bits lcomp + 51 bits offset
  hdr64[globalHeaderValues + iblock] = offset | (static_cast<uint64>(lcomp) << 51) | (1ULL << 63);
}

// Read the block offset, size and overflow flag from the header
__forceinline__ __host__ __device__ void
getBlockInfo(const void *hdr, int iblock, uint64 &offset, uint &lcomp, bool &overflow)
{
  const uint64 *hdr64 = reinterpret_cast<const uint64 *>(hdr);
  // 1-bit overflow flag + 12 bits lcomp + 51 bits offset
  offset = hdr64[globalHeaderValues + iblock];
  overflow = offset & (1ULL << 63);
  lcomp = static_cast<uint>(offset >> 51) & 0xfff;
  offset &= 0x7ffffffffffff;
}

} // namespace header

} // namespace bitcomp
