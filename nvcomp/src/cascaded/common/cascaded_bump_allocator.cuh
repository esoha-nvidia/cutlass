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

#include <cassert>
#include <cstdint>

namespace cascaded
{

// TODO: remove this, use constant created by other MR
constexpr uint32_t WARP_SIZE = 32;

// =============================================================================
// Bump allocator for variable-length writes with a contiguous header table.
//
// Intended use:
//    - data is only written during compression, never read
//    - data is only read during decompression, never written
//    - reservations cannot be "de-allocated"
//
// Format:
//   [ header_table : 2B * num_slots ]   (caller-owned, contiguous)
//   [ payload      : variable        ]   (caller-owned, bump-allocated)
//
//
// Per-slot header (16 bits, little-endian):
//   - size_minus_one (BUMP_SIZE_BITS): encoded size, real size = stored + 1
//   - tag (BUMP_TAG_BITS): caller-defined
//   - owner (BUMP_OWNER_BITS): writing warp id
//
// Capacity:
//   - num_slots: total slot count (==N), known to both writer and reader
//   - max payload size per allocation: 1024 bytes (10-bit bias-by-1)
//   - max owner id: 7 (3 bits, supports up to 8 writing warps)
//
// Sync model:
//   - block_bump_allocator_init: block-collective. Caller must __syncthreads()
//     before any reservation occurs.
//   - warp_bump_reserve: warp-collective. All 32 lanes must call together.
//   - block_bump_reader_init: block-collective.
//   - warp_bump_pop: warp-collective. All 32 lanes must call together.
//   - Between writes and reads: caller must __syncthreads() to make the bump
//     state and header table writes visible block-wide.
//
// Constraints:
//   - blockDim.x must be a multiple of WARP_SIZE.
//   - The same warps_per_block must be used for compression and decompression.
//   - num_slots must be the exact count of warp_bump_reserve calls; the reader
//     terminates when scan_slot >= num_slots.
//   - Reservations must have size in [1, BUMP_MAX_SIZE]. Size 0 is not
//     encodable; callers wanting an empty allocation should reserve a
//     1-byte placeholder.
//
// Owner semantics:
//   - owner is implicit: derived from threadIdx.x / WARP_SIZE on every call.
//
// =============================================================================

// Header bit layout
constexpr uint32_t BUMP_HEADER_BITS = 16; // hard-coded to allow for 1KB uncompressed subchunks
constexpr uint32_t BUMP_SIZE_BITS = 10; // hard-coded to match bitpacker 1KB batch design
constexpr uint32_t BUMP_OWNER_BITS = 3; // hard-coded to allow for CTA's of <= 8 warps
constexpr uint32_t BUMP_TAG_BITS =
  BUMP_HEADER_BITS - BUMP_SIZE_BITS -
  BUMP_OWNER_BITS; // remaining bits are not used by allocator, but can be used by caller

constexpr uint32_t BUMP_MAX_SIZE = 1u << BUMP_SIZE_BITS; // 1024
constexpr uint32_t BUMP_MAX_TAG = (1u << BUMP_TAG_BITS) - 1u; // 7
constexpr uint32_t BUMP_MAX_OWNER = (1u << BUMP_OWNER_BITS) - 1u; // 7

constexpr uint32_t BUMP_SIZE_MASK = (1u << BUMP_SIZE_BITS) - 1u;
constexpr uint32_t BUMP_TAG_MASK = (1u << BUMP_TAG_BITS) - 1u;
constexpr uint32_t BUMP_OWNER_MASK = (1u << BUMP_OWNER_BITS) - 1u;

constexpr uint32_t BUMP_TAG_SHIFT = BUMP_SIZE_BITS;
constexpr uint32_t BUMP_OWNER_SHIFT = BUMP_SIZE_BITS + BUMP_TAG_BITS;

// =============================================================================
// Common Header read/write free functions
// =============================================================================

__host__ __device__ __forceinline__ uint16_t bump_header_pack(uint32_t owner, uint32_t tag, uint32_t size)
{
  // I am choosing to instead store the range (1,1024) (using a bias of +1) because the likelihood of a 1024B
  // buffer (compression failed) is much greater than a 0B buffer (compression yields only metadata, no bits per value)

  // Size 0 must be bumped to 1
  if (size == 0u)
  {
    size = 1u;
  }

  return static_cast<uint16_t>(
    ((size - 1u) & BUMP_SIZE_MASK) | ((tag & BUMP_TAG_MASK) << BUMP_TAG_SHIFT) |
    ((owner & BUMP_OWNER_MASK) << BUMP_OWNER_SHIFT)
  );
}

__host__ __device__ __forceinline__ void bump_header_unpack(uint16_t h, uint32_t &size, uint32_t &owner, uint32_t &tag)
{
  size = (h & BUMP_SIZE_MASK) + 1u;
  owner = (h >> BUMP_OWNER_SHIFT) & BUMP_OWNER_MASK;
  tag = (h >> BUMP_TAG_SHIFT) & BUMP_TAG_MASK;
}

// =============================================================================
// Producer / Allocator / Write Side
// =============================================================================

struct BumpAllocator
{
  uint16_t *header_table; // size >= num_slots * 2 bytes
  uint8_t *payload_base; // contiguous payload region
  unsigned long long *bump_state; // packed atomic: low 32 bits = slot count, high 32 bits = byte count
};

/** Block-collective. Initializes shared state (clears the bump state). The
 * header table is not zeroed; slots are written before they are read.
 *
 * Caller must __syncthreads() after this call before any reservation begins.
 */
__device__ __forceinline__ void block_bump_allocator_init(
  BumpAllocator &alloc,
  uint16_t *header_table,
  uint8_t *payload_base,
  unsigned long long *bump_state
)
{
  alloc.header_table = header_table;
  alloc.payload_base = payload_base;
  alloc.bump_state = bump_state;

  if (threadIdx.x == 0)
  {
    *bump_state = 0ULL;
  }
}

/**
 * All 32 lanes must call. Lane 0 issues the atomic, broadcasts the (slot,
 * offset) pair to the warp, and writes the header. All lanes return the same
 * payload pointer.
 *
 * Owner is implicit: threadIdx.x / WARP_SIZE.
 *
 * Caller is responsible for:
 *   - size in [1, BUMP_MAX_SIZE]
 *   - tag in [0, BUMP_MAX_TAG]
 *   - the resulting slot index < num_slots (caller knows the partition)
 *   - the resulting payload offset + size <= payload region capacity
 *
 * All four are debug-asserted; release builds trust the caller.
*/
__device__ __forceinline__ uint8_t *warp_bump_reserve(BumpAllocator &alloc, uint32_t tag, uint32_t size)
{
  assert(size > 0 && size <= BUMP_MAX_SIZE);
  assert(tag <= BUMP_MAX_TAG);

  const uint32_t lane_id = threadIdx.x & (WARP_SIZE - 1);
  const uint32_t owner = threadIdx.x / WARP_SIZE;
  assert(owner <= BUMP_MAX_OWNER);

  // Pack (1 slot, size bytes) into a single 64-bit add: low 32 = slot bump, high 32 = byte bump.
  unsigned long long packed_request = 1ULL | (static_cast<unsigned long long>(size) << 32);
  unsigned long long got = 0;
  if (lane_id == 0)
  {
    got = atomicAdd(alloc.bump_state, packed_request);
  }
  // Broadcast both 32-bit halves to all lanes.
  uint32_t slot = static_cast<uint32_t>(got);
  uint32_t offset = static_cast<uint32_t>(got >> 32);
  offset = __shfl_sync(0xFFFFFFFFu, offset, 0);

  // Lane 0 writes the header. Other lanes can compute and return the payload pointer.
  if (lane_id == 0)
  {
    alloc.header_table[slot] = bump_header_pack(owner, tag, size);
  }

  return alloc.payload_base + offset;
}

// =============================================================================
// Consumer / Reader
// =============================================================================

/**
 * Carries num_slots so the reader is self-contained. The pop loop terminates
 * when scan_slot >= num_slots.
 */
struct BumpReader
{
  const uint16_t *header_table;
  const uint8_t *payload_base;
  uint32_t num_slots;
  uint32_t scan_slot;
  uint32_t scan_offset;
};

struct BumpPopResult
{
  const uint8_t *payload;
  uint32_t size;
  uint32_t tag;
};

/**
 * Block-collective in the sense that all threads must build a consistent
 * view, but stateless aside from local fields. No syncthreads required.
 */
__device__ __forceinline__ void
block_bump_reader_init(BumpReader &reader, const uint16_t *header_table, const uint8_t *payload_base, uint32_t num_slots)
{
  reader.header_table = header_table;
  reader.payload_base = payload_base;
  reader.num_slots = num_slots;
  reader.scan_slot = 0;
  reader.scan_offset = 0;
}

/** 
 * All 32 lanes must call. Each lane scans the header table redundantly from
 * reader.scan_slot; since all lanes start from identical state and read the
 * same memory, they reach the same conclusion deterministically. No shuffles
 * or broadcasts required.
 *
 * Owner is implicit: threadIdx.x / WARP_SIZE.
 *
 * Returns true if an owned slot was found; the result struct is populated.
 * Returns false if the reader is exhausted; result is unspecified.
*/
__device__ __forceinline__ bool warp_bump_pop(BumpReader &reader, BumpPopResult &result)
{
  // TODO: Should we re-factor this function so that we use "work stealing" instead of
  //       fixed "each decomp warp must reverse all work created by the corresponding comp warp" ?
  //       This will require:
  //         1 - each warp will have to count the number of allocations for each producer. This is ~8 separate counters. Must be warp-private
  //         2 - each warp increments a global work counter, does the corresponding decompression
  //         3 - each warp must decode all jobs processed by other warps so that read position is known

  const uint32_t my_owner = threadIdx.x / WARP_SIZE;

  uint32_t found_offset = 0;
  uint32_t found_size = 0;
  uint32_t found_tag = 0;
  uint32_t found = 0; // 0 or 1

  while (reader.scan_slot < reader.num_slots)
  {
    uint16_t h = reader.header_table[reader.scan_slot];
    uint32_t size, owner, tag;
    bump_header_unpack(h, size, owner, tag);

    if (owner == my_owner)
    {
      found_offset = reader.scan_offset;
      found_size = size;
      found_tag = tag;
      found = 1;

      // Advance past the consumed slot for the next call.
      reader.scan_slot += 1;
      reader.scan_offset += size;
      break;
    }
    else
    {
      // Skip this slot; advance cursors and continue.
      reader.scan_slot += 1;
      reader.scan_offset += size;
    }
  }

  if (found)
  {
    result.payload = reader.payload_base + found_offset;
    result.size = found_size;
    result.tag = found_tag;
    return true;
  }

  return false;
}

} // namespace cascaded
