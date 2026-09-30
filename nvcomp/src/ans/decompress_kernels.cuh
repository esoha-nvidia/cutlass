/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include <cuda/std/bit>

#include "common.h"

#include <ans/ans_utils.cuh>
#include <ans/DecodePolicy.hpp>
#include <ans/symbol_decoder.cuh>
#include <ans/types.cuh>
#include <nvcomp/shared_types.h>
#include <nvcomp/utils.hpp>

namespace ans_gpu_lib
{
namespace detail
{

template <bool BOUNDS_CHECK>
__device__ void construct_decoding_table(
  uint32_t *table,
  TableBuildScratch &scratch,
  const void *comp_chunk,
  const uint32_t comp_chunk_size
)
{
  constexpr int SYMBOLS_PER_THREAD = NV_SYMBOL_COUNT / NUM_DECOMP_THREADS_PER_CTA;
  static_assert(
    SYMBOLS_PER_THREAD * NUM_DECOMP_THREADS_PER_CTA == NV_SYMBOL_COUNT,
    "construct_decoding_table assumes NUM_DECOMP_THREADS_PER_CTA evenly divides the symbol count"
  );

  uint32_t lane_id = get_lane_id();
  uint32_t wid = threadIdx.x / WARP_SIZE_U;

  const uint8_t *chunk_start = static_cast<const uint8_t *>(comp_chunk);
  const uint8_t *chunk_end = chunk_start + comp_chunk_size;

  // Table-build scalars live in the fixed header at chunk offset 0, so they load in
  // one hop. Counts sit after the sizes array and the derived mantissas.
  const ANS_fixed_header *fixed_header = reinterpret_cast<const ANS_fixed_header *>(chunk_start);
  const int min_symbol_value = safe_guard_generic<BOUNDS_CHECK>(&fixed_header->min_symbol_, chunk_start, chunk_end);
  const int max_symbol_value = safe_guard_generic<BOUNDS_CHECK>(&fixed_header->max_symbol_, chunk_start, chunk_end);
  const uint32_t tablelog = safe_guard_generic<BOUNDS_CHECK>(&fixed_header->tablelog_, chunk_start, chunk_end);
  assert(tablelog >= MIN_MODE_TABLELOG && tablelog <= MAX_TABLELOG);
  const uint32_t table_size = 1u << tablelog;

  const int16_t *norm_counts = fixed_header->get_norm_counts();

  uint2 *pdfs_and_cdfs = scratch.pdfs_and_cdfs;

  // Populate pdfs with unconditional LDGs + stores, so the norm_counts loads are
  // in flight while the dependent (higher-latency to resolve) min/max symbol
  // loads are still outstanding. Guarding the store on the window test instead
  // would let the compiler sink/predicate the load behind that comparison and
  // serialize the global loads -- the stall we're avoiding here.
  //
  // norm_counts is stored WINDOWED: entry i is symbol (min_symbol_value + i). Threads
  // whose symbol falls outside [min, max] clamp their load to entry 0 (and discard the
  // value) so the load still issues unconditionally. Each thread covers
  // SYMBOLS_PER_THREAD symbols (t, t + NUM_DECOMP_THREADS_PER_CTA, ...), so
  // SYMBOLS_PER_THREAD * NUM_DECOMP_THREADS_PER_CTA == NV_SYMBOL_COUNT slots are
  // populated with no index guard. Out-of-window symbols get
  // pdf 0: they then contribute 0 to the exclusive prefix sum, so in-window cdfs are
  // unaffected, and they're independently masked by `active` in the fill below.
  uint32_t pdf_items[SYMBOLS_PER_THREAD];
#pragma unroll
  for (int ix_symbol = 0; ix_symbol < SYMBOLS_PER_THREAD; ++ix_symbol)
  {
    const int symbol = threadIdx.x + ix_symbol * NUM_DECOMP_THREADS_PER_CTA;
    const bool in_window = symbol >= min_symbol_value && symbol <= max_symbol_value;
    const int entry = in_window ? (symbol - min_symbol_value) : 0;
    const int16_t raw = safe_guard_generic<BOUNDS_CHECK>(&norm_counts[entry], chunk_start, chunk_end);
    pdf_items[ix_symbol] = in_window ? static_cast<uint32_t>(abs(raw)) : 0u;
  }

// 256-element exclusive prefix sum over NUM_DECOMP_THREADS_PER_CTA threads
// (SYMBOLS_PER_THREAD items/thread).
#pragma unroll
  for (int j = 0; j < SYMBOLS_PER_THREAD; ++j)
  {
    pdfs_and_cdfs[threadIdx.x + j * NUM_DECOMP_THREADS_PER_CTA].x = pdf_items[j];
  }
  __syncthreads();

  using Scan = TableBuildScan;

  uint32_t scan_in[SYMBOLS_PER_THREAD];
#pragma unroll
  for (int j = 0; j < SYMBOLS_PER_THREAD; ++j)
  {
    scan_in[j] = pdfs_and_cdfs[threadIdx.x * SYMBOLS_PER_THREAD + j].x;
  }

  uint32_t scan_out[SYMBOLS_PER_THREAD];
  Scan(scratch.scan_smem).ExclusiveSum(scan_in, scan_out);

#pragma unroll
  for (int j = 0; j < SYMBOLS_PER_THREAD; ++j)
  {
    pdfs_and_cdfs[threadIdx.x * SYMBOLS_PER_THREAD + j].y = scan_out[j];
  }
  __syncthreads();

// The whole symbol range is covered in SYMBOLS_PER_THREAD warp-uniform passes:
// each pass handles NUM_DECOMP_WARPS_PER_CTA * WARP_SIZE = NUM_DECOMP_THREADS_PER_CTA
// symbols, so SYMBOLS_PER_THREAD passes cover all NV_SYMBOL_COUNT symbols.
// SYMBOLS_PER_THREAD passes cover all NV_SYMBOL_COUNT symbols.
//
// Most symbols have a short slot run: a symbol with pdf in
// [1, ANS_TABLE_SELF_FILL_MAX] is written directly by its owning lane, so the 32
// lanes of a warp fill their own runs in parallel (pdf 0 writes nothing). Only the
// rarer "heavy" symbols above that threshold are balloted out and processed one at
// a time, with all 32 lanes cooperating to fill that one symbol's slots.
//
// Symbols are interleaved across warps (warp w owns w, w+NUM_DECOMP_WARPS_PER_CTA,
// w+2*NUM_DECOMP_WARPS_PER_CTA, ...) rather than handed out in contiguous blocks. Heavy
// symbols tend to cluster (e.g. a run of large pdfs), so a contiguous block
// assignment would dump the whole cluster onto one warp while the others
// idle; the round-robin spread keeps the heavy work balanced across warps.
#pragma unroll
  for (int ix_symbol = 0; ix_symbol < SYMBOLS_PER_THREAD; ++ix_symbol)
  {
    const int symbol = (lane_id * NUM_DECOMP_WARPS_PER_CTA + wid) + ix_symbol * NUM_DECOMP_THREADS_PER_CTA;
    const bool active = symbol >= min_symbol_value && symbol <= max_symbol_value;
    const uint32_t pdf = active ? pdfs_and_cdfs[symbol].x : 0;
    const uint32_t cdf = active ? pdfs_and_cdfs[symbol].y : 0;

    // Common case: a short run is written by its owning lane, all 32 lanes at once.
    // pdf 0 writes nothing and a run longer than the threshold is left to the
    // cooperative path below, both by falling out of this predicate.
    const uint32_t self_pdf = (pdf <= ANS_TABLE_SELF_FILL_MAX) ? pdf : 0u;
#pragma unroll
    for (uint32_t ix_pdf = 0; ix_pdf < ANS_TABLE_SELF_FILL_MAX; ++ix_pdf)
    {
      if (ix_pdf < self_pdf)
      {
        uint32_t slot = safe_guard_generic<BOUNDS_CHECK>(cdf + ix_pdf, table_size);
        table[slot] = (ix_pdf << 20) | (pdf << 8) | static_cast<uint32_t>(symbol);
      }
    }

    // Heavy symbols (pdf > ANS_TABLE_SELF_FILL_MAX): cooperatively fill, one heavy lane
    // at a time. The ballot/shuffle are warp-uniform since every lane reaches them together.
    uint32_t heavy = __ballot_sync(WARP_ALL, pdf > ANS_TABLE_SELF_FILL_MAX);
    while (heavy != 0)
    {
      const int lane = __ffs(heavy) - 1;
      heavy &= heavy - 1;

      const uint32_t h_symbol = __shfl_sync(WARP_ALL, static_cast<uint32_t>(symbol), lane);
      const uint32_t h_pdf = __shfl_sync(WARP_ALL, pdf, lane);
      const uint32_t h_cdf = __shfl_sync(WARP_ALL, cdf, lane);

      for (uint32_t ix_pdf = lane_id; ix_pdf < h_pdf; ix_pdf += WARP_SIZE)
      {
        uint32_t slot = safe_guard_generic<BOUNDS_CHECK>(h_cdf + ix_pdf, table_size);
        table[slot] = (ix_pdf << 20) | (h_pdf << 8) | h_symbol;
      }
    }
  }
}

template <bool BOUNDS_CHECK, typename DecoderPolicy>
__device__ __forceinline__ void decode_sub_chunk(
  const uint32_t *table,
  const uint8_t *comp_sub_chunk,
  uint32_t comp_sub_chunk_size,
  void *uncomp_sub_chunk,
  const uint8_t *sub_chunk_mantissas,
  uint16_t *warp_renorm_buf,
  int num_decodes,
  nvcompStatus_t *status = nullptr
)
{
  typename DecoderPolicy::template Decoder<BOUNDS_CHECK> sd(table, comp_sub_chunk, comp_sub_chunk_size, warp_renorm_buf);
  DecoderPolicy::template decode_body<BOUNDS_CHECK>(sd, uncomp_sub_chunk, sub_chunk_mantissas, num_decodes);
  if constexpr (BOUNDS_CHECK)
  {
    write_decode_status(sd.getError(), status);
  }
}

// Compile-time decode mode. Generic branches at runtime on the bitstream when the host
// does not know the data type, or when a named type leaves the state count at 0.
// Char/Fp16/Fp8/Fp32 fix the type at compile time so that specialization contains
// ONLY that decode path and
// is register-allocated for it alone -- a type the host specified via the
// decompress opts does not pay the other types' register footprint (which would
// otherwise raise the shared kernel's ceiling and spill to local memory / LDL).
// STATES_PER_LANE is 0 when the count comes from the bitstream (Generic), else 1 or 2.
enum class DecodeMode
{
  Generic,
  Char,
  Fp16,
  Fp8,
  Fp32
};

// Generic reads the bitstream type. Typed kernels pin it so symbol counts fold
// at compile time; validate_chunk still rejects a disagreeing header.
template <DecodeMode MODE>
inline constexpr __host__ __device__ AnsStreamType decode_symbol_type(AnsStreamType stream_type = AnsStreamType::Char)
{
  if constexpr (MODE == DecodeMode::Generic)
  {
    return stream_type;
  }
  else if constexpr (MODE == DecodeMode::Fp8)
  {
    return AnsStreamType::Fp8;
  }
  else if constexpr (MODE == DecodeMode::Fp32)
  {
    return AnsStreamType::Fp32;
  }
  else if constexpr (MODE == DecodeMode::Char)
  {
    return AnsStreamType::Char;
  }
  else if constexpr (MODE == DecodeMode::Fp16)
  {
    return AnsStreamType::Fp16;
  }
}

template <DecodeMode MODE>
inline constexpr uint32_t decode_table_capacity_log()
{
  if constexpr (MODE == DecodeMode::Generic)
  {
    return MAX_TABLELOG;
  }
  else
  {
    return ans_tablelog(decode_symbol_type<MODE>());
  }
}

// All threads in the CTA must call this and take the same branch.
template <DecodeMode MODE, uint8_t STATES_PER_LANE, bool BOUNDS_CHECK>
__device__ bool validate_chunk(
  const ANS_fixed_header *fixed_header,
  uint32_t comp_chunk_size,
  AnsStreamType stream_type,
  uint8_t stream_states_per_lane,
  uint8_t stream_tablelog,
  uint32_t uncomp_chunk_size,
  uint32_t max_sub_chunk_size,
  uint32_t uncomp_chunk_buf_size,
  const void *uncomp_chunk,
  uint8_t max_sub_chunk_count,
  nvcompType_t expected_data_type,
  uint8_t expected_states_per_lane,
  int tid,
  nvcompStatus_t *status
)
{
  auto fail = [&](nvcompStatus_t err) {
    if (tid == 0)
    {
      *status = err;
    }
    return false;
  };

  if (comp_chunk_size < sizeof(ANS_fixed_header))
  {
    return fail(nvcompErrorCannotDecompress);
  }

  if (!ans_known_stream_type(stream_type) || (stream_states_per_lane != 1 && stream_states_per_lane != 2))
  {
    return fail(nvcompErrorCannotDecompress);
  }

  const AnsStreamType checked_type = decode_symbol_type<MODE>(stream_type);
  bool type_matches_request = true;
  if constexpr (MODE != DecodeMode::Generic)
  {
    // A pinned kernel only decodes the type it was specialized for.
    type_matches_request = stream_type == checked_type;
  }
  else if (expected_data_type != NVCOMP_TYPE_BITS)
  {
    // BITS is the "unknown" request, so anything decodes; every other type must match.
    const bool wants_char = expected_data_type == NVCOMP_TYPE_CHAR || expected_data_type == NVCOMP_TYPE_UCHAR;
    type_matches_request = (wants_char && stream_type == AnsStreamType::Char) ||
                           (expected_data_type == NVCOMP_TYPE_FLOAT16 && stream_type == AnsStreamType::Fp16) ||
                           (expected_data_type == NVCOMP_TYPE_FLOAT8_E4M3 && stream_type == AnsStreamType::Fp8) ||
                           (expected_data_type == NVCOMP_TYPE_FLOAT32 && stream_type == AnsStreamType::Fp32);
  }
  const bool states_match_request = STATES_PER_LANE == 0 ? (expected_states_per_lane == 0 ||
                                                            stream_states_per_lane == expected_states_per_lane)
                                                         : stream_states_per_lane == STATES_PER_LANE;
  if (!type_matches_request || !states_match_request || stream_tablelog != ans_tablelog(checked_type))
  {
    return fail(nvcompErrorCannotDecompress);
  }

  if (!cuda::std::has_single_bit(max_sub_chunk_size))
  {
    return fail(nvcompErrorCannotDecompress);
  }
  const uint32_t bytes_per_symbol = ans_bytes_per_symbol(checked_type);
  // Every sub-chunk boundary must fall between complete typed values. FP8 may
  // have one raw trailing byte at the end of the chunk, but its sub-chunks still
  // contain whole adjacent pairs.
  if (max_sub_chunk_size % bytes_per_symbol != 0 ||
      (checked_type != AnsStreamType::Fp8 && uncomp_chunk_size % bytes_per_symbol != 0))
  {
    return fail(nvcompErrorCannotDecompress);
  }

  const uint32_t actual_sub_chunk_count = ans_derived_num_sub_chunks(uncomp_chunk_size, max_sub_chunk_size);
  if (actual_sub_chunk_count > MAX_SUB_CHUNKS_PER_CHUNK)
  {
    return fail(nvcompErrorSubChunkCountTooLarge);
  }
  if (max_sub_chunk_count != 0 && actual_sub_chunk_count > max_sub_chunk_count)
  {
    return fail(nvcompErrorSubChunkCountTooSmall);
  }
  if (fixed_header->min_symbol_ > fixed_header->max_symbol_ || fixed_header->sub_chunk_0_offset() > comp_chunk_size)
  {
    // sub_chunk_0_offset is after the size array, mantissas, normalized counts,
    // and alignment padding. This guarantees all side-band reads are in-bounds.
    return fail(nvcompErrorCannotDecompress);
  }
  if (uncomp_chunk_size > uncomp_chunk_buf_size)
  {
    return fail(nvcompErrorOutputBufferTooSmall);
  }
  if ((reinterpret_cast<uintptr_t>(uncomp_chunk) % ANS_SUB_CHUNK_SLOT_ALIGN) != 0)
  {
    return fail(nvcompErrorOutputBufferAlignmentTooSmall);
  }
  return true;
}

template <DecodeMode MODE, uint8_t STATES_PER_LANE, bool BOUNDS_CHECK, bool ONE_SUBCHUNK_PER_WARP>
__device__ __forceinline__ void decompress_chunk(
  const void *const *comp_chunks,
  const size_t *comp_chunk_sizes,
  void *const *uncomp_chunks,
  const size_t *uncomp_chunk_sizes,
  size_t *actual_uncomp_chunk_sizes,
  nvcompStatus_t *block_statuses,
  uint8_t max_sub_chunk_count,
  nvcompType_t expected_data_type,
  uint8_t expected_states_per_lane,
  uint8_t skip_validate,
  const volatile uint32_t &chunk_idx
)
{
  static_assert(STATES_PER_LANE <= 2, "STATES_PER_LANE is 0 (bitstream), 1, or 2");
  static_assert(MODE != DecodeMode::Generic || STATES_PER_LANE == 0);
  static_assert(MODE != DecodeMode::Fp8 || STATES_PER_LANE == 1 || STATES_PER_LANE == 2);
  static_assert(MODE != DecodeMode::Fp32 || STATES_PER_LANE == 1 || STATES_PER_LANE == 2);
  static_assert(MODE != DecodeMode::Fp16 || STATES_PER_LANE == 1 || STATES_PER_LANE == 2);
  static_assert(MODE != DecodeMode::Char || STATES_PER_LANE == 1 || STATES_PER_LANE == 2);
  assert(blockDim.y == 1 || !ONE_SUBCHUNK_PER_WARP);

  // Per-CTA shared decoding table. Live across
  // both the build and decode phases, so it stays out of the build/decode union.
  __shared__ __align__(sizeof(uint4)) DecodeTable<decode_table_capacity_log<MODE>()> table_smem;
  __shared__ uint32_t padded_sub_chunk_offsets[MAX_SUB_CHUNKS_PER_CHUNK];
  // Table-build scratch and decode buffers overlap in a union; never live at the
  // same time. Declared here so the decode phase can name smem.decode after the
  // setup locals have gone out of scope.
  __shared__ __align__(sizeof(uint4)) SetupAndDecodeSmem smem;
  uint32_t *table = table_smem.slots;

  // sub_chunk_0_off is a single u32 derived from several header fields; cheaper to
  // keep than to rematerialize. Everything else used by decode is reloaded after
  // table-build so those SSA values are not CSE'd across the barrier.
  uint32_t sub_chunk_0_off = 0;
  const int tid = ONE_SUBCHUNK_PER_WARP ? static_cast<int>(threadIdx.x)
                                        : static_cast<int>(blockDim.x * blockIdx.y + threadIdx.x);

  {
    // ---------- (1) Pointer setup + chunk-metadata dependency chain ----------
    // Chunk index from shared memory (not blockIdx.x) so the compiler does not
    // keep that special register live. Array bases are the kernel's grid_constant
    // parameters; inlining this function keeps those loads on the param bank.
    const void *comp_chunk = comp_chunks[chunk_idx];
    const uint32_t comp_chunk_size = static_cast<uint32_t>(comp_chunk_sizes[chunk_idx]);
    uint8_t *uncomp_chunk = reinterpret_cast<uint8_t *>(uncomp_chunks[chunk_idx]);
    const uint32_t uncomp_chunk_buf_size = static_cast<uint32_t>(uncomp_chunk_sizes[chunk_idx]);
    size_t *actual_uncomp_chunk_size = &actual_uncomp_chunk_sizes[chunk_idx];
    nvcompStatus_t *status = &block_statuses[chunk_idx];

    const uint8_t *comp_chunk_start = reinterpret_cast<const uint8_t *>(comp_chunk);
    const uint8_t *comp_chunk_end = comp_chunk_start + comp_chunk_size;
    const ANS_fixed_header *fixed_header = reinterpret_cast<const ANS_fixed_header *>(comp_chunk_start);
    const uint8_t type_byte = safe_guard_generic<BOUNDS_CHECK>(&fixed_header->type_, comp_chunk_start, comp_chunk_end);
    const AnsStreamType stream_type = static_cast<AnsStreamType>(type_byte);
    const uint8_t stream_states_per_lane =
      safe_guard_generic<BOUNDS_CHECK>(&fixed_header->states_per_lane_, comp_chunk_start, comp_chunk_end);
    const uint8_t stream_tablelog =
      safe_guard_generic<BOUNDS_CHECK>(&fixed_header->tablelog_, comp_chunk_start, comp_chunk_end);
    constexpr bool compiletime_is_fp8 = (MODE == DecodeMode::Fp8);
    const bool runtime_is_fp8 = (MODE == DecodeMode::Generic) ? (stream_type == AnsStreamType::Fp8)
                                                              : compiletime_is_fp8;

    nvcompStatus_t err = nvcompSuccess;
    uint32_t uncomp_chunk_size =
      safe_guard_generic<BOUNDS_CHECK>(&fixed_header->uncomp_bytes_, comp_chunk_start, comp_chunk_end, &err);
    const uint32_t max_sub_chunk_size =
      safe_guard_generic<BOUNDS_CHECK>(&fixed_header->max_sub_chunk_size_, comp_chunk_start, comp_chunk_end, &err);

    if (skip_validate == 0)
    {
      const bool valid = validate_chunk<MODE, STATES_PER_LANE, BOUNDS_CHECK>(
        fixed_header,
        comp_chunk_size,
        stream_type,
        stream_states_per_lane,
        stream_tablelog,
        uncomp_chunk_size,
        max_sub_chunk_size,
        uncomp_chunk_buf_size,
        uncomp_chunk,
        max_sub_chunk_count,
        expected_data_type,
        expected_states_per_lane,
        tid,
        status
      );
      if (!valid)
      {
        if (tid == 0)
        {
          *actual_uncomp_chunk_size = 0;
        }
        return;
      }
    }

    if (uncomp_chunk_size == 0)
    {
      if (tid == 0)
      {
        if (!BOUNDS_CHECK || err != nvcompSuccess)
        {
          *status = err;
        }
        *actual_uncomp_chunk_size = 0;
      }
      return;
    }

    // Typed kernels pin the type so symbol counts fold. Do not select on skip_validate:
    // that parameter is runtime and would keep a symbol-type register live across table-build.
    const uint32_t actual_sub_chunk_count = ans_derived_num_sub_chunks(uncomp_chunk_size, max_sub_chunk_size);

    // FP8 odd N: the unpaired trailing byte was stored raw at mantissas[chunk_symbols]
    // and is not a decoded ANS symbol. Emit it here (before per-warp early returns).
    // Also report actual output size once up front, including chunks that never
    // reach a decode (e.g. fp8 N == 1).
    if (blockIdx.y == 0 && threadIdx.x == 0)
    {
      if (runtime_is_fp8 && (uncomp_chunk_size & 1))
      {
        const uint32_t chunk_symbols = [&] {
          if constexpr (MODE == DecodeMode::Generic)
          {
            return ans_num_symbols(stream_type, uncomp_chunk_size);
          }
          else
          {
            return ans_num_symbols(decode_symbol_type<MODE>(), uncomp_chunk_size);
          }
        }();
        // mantissas_offset() is header-derived, so a malformed header can push this load
        // past the chunk.
        const uint8_t *const tail_byte = comp_chunk_start + fixed_header->mantissas_offset() + chunk_symbols;
        uncomp_chunk[uncomp_chunk_size - 1] =
          safe_guard_generic<BOUNDS_CHECK>(tail_byte, comp_chunk_start, comp_chunk_end);
      }
      *actual_uncomp_chunk_size = uncomp_chunk_size;
      if constexpr (!BOUNDS_CHECK)
      {
        *status = nvcompSuccess;
      }
    }

    // Extra Y-CTAs have no sub-chunk. ONE_SUBCHUNK_PER_WARP launches one CTA.
    if constexpr (!ONE_SUBCHUNK_PER_WARP)
    {
      if (blockIdx.y * NUM_DECOMP_WARPS_PER_CTA >= actual_sub_chunk_count)
      {
        return;
      }
    }

    const uint32_t *const sizes = fixed_header->get_sub_chunk_sizes();
    sub_chunk_0_off = fixed_header->sub_chunk_0_offset();
    const uint32_t padded_bytes =
      (static_cast<uint32_t>(threadIdx.x) < actual_sub_chunk_count)
        ? nvcomp::roundUpTo(
            safe_guard_generic<BOUNDS_CHECK>(&sizes[threadIdx.x], comp_chunk_start, comp_chunk_end),
            ANS_SUB_CHUNK_SLOT_ALIGN
          )
        : 0u;
    cta_inclusive_prefix_sum<NUM_DECOMP_THREADS_PER_CTA>(padded_sub_chunk_offsets, padded_bytes, actual_sub_chunk_count);

    // ---------- (2) Build the decoding table into shared memory ----------
    construct_decoding_table<BOUNDS_CHECK>(table, smem.build, comp_chunk, comp_chunk_size);
  }

  // ---------- CTA-wide barrier: make the freshly-built shared table visible ----------
  __syncthreads();

  {
    // Reload chunk bases and header scalars with expressions that do not CSE with
    // the setup-phase safe_guard u32 loads, so those registers can die in table-build.
    const uint8_t *comp_chunk_start = reinterpret_cast<const uint8_t *>(comp_chunks[chunk_idx]);
    uint8_t *uncomp_chunk = reinterpret_cast<uint8_t *>(uncomp_chunks[chunk_idx]);
    // Unguarded even under BOUNDS_CHECK: setup read both words through safe_guard, which
    // yields 0 when out of bounds, and a zero uncomp_bytes_ already returned above. So
    // reaching here means the first 8 header bytes are inside the chunk, and the buffer
    // is 16 B-aligned.
    static_assert(offsetof(ANS_fixed_header, uncomp_bytes_) == 0);
    static_assert(offsetof(ANS_fixed_header, max_sub_chunk_size_) == sizeof(uint32_t));
    const uint2 uncomp_and_max = *reinterpret_cast<const uint2 *>(comp_chunk_start);
    const uint32_t uncomp_chunk_size = uncomp_and_max.x;
    const uint32_t max_sub_chunk_size = uncomp_and_max.y;
    const uint32_t actual_sub_chunk_count = ans_derived_num_sub_chunks(uncomp_chunk_size, max_sub_chunk_size);
    const int sub_chunk_idx = (ONE_SUBCHUNK_PER_WARP ? static_cast<int>(threadIdx.x)
                                                     : static_cast<int>(blockDim.x * blockIdx.y + threadIdx.x)) /
                              WARP_SIZE;

    auto run = [&](auto policy) {
      using Policy = decltype(policy);
      auto decode_one = [&](int sc) {
        // stream_type does not survive the barrier; re-read it. Pinned modes ignore the
        // argument, so the load folds away for them.
        const AnsStreamType type =
          decode_symbol_type<MODE>(reinterpret_cast<const ANS_fixed_header *>(comp_chunk_start)->type());
        const uint32_t sub_chunk_start_bytes = max_sub_chunk_size * static_cast<uint32_t>(sc);
        uint8_t *uncomp_sub_chunk = uncomp_chunk + sub_chunk_start_bytes;
        const uint32_t remaining_bytes = (sub_chunk_start_bytes < uncomp_chunk_size)
                                           ? uncomp_chunk_size - sub_chunk_start_bytes
                                           : 0u;
        const uint32_t uncomp_sub_chunk_symbols = ans_num_symbols(type, min(max_sub_chunk_size, remaining_bytes));
        const uint32_t max_sub_chunk_symbols = ans_num_symbols(type, max_sub_chunk_size);
        const uint32_t max_sub_chunk_mantissa_bytes = max_sub_chunk_symbols * ans_mantissa_bytes_per_symbol(type);
        const uint8_t *sub_chunk_mantissas = comp_chunk_start + ans_mantissas_offset(type, actual_sub_chunk_count) +
                                             static_cast<uint32_t>(sc) * max_sub_chunk_mantissa_bytes;
        const uint32_t *sc_sizes = reinterpret_cast<const uint32_t *>(comp_chunk_start + ans_sub_chunk_sizes_offset());
        const uint8_t *comp_chunk_end = comp_chunk_start + static_cast<uint32_t>(comp_chunk_sizes[chunk_idx]);
        const uint32_t comp_sub_chunk_size =
          safe_guard_generic<BOUNDS_CHECK>(&sc_sizes[sc], comp_chunk_start, comp_chunk_end);
        const uint32_t packed_start_bytes = (sc == 0) ? 0u : padded_sub_chunk_offsets[static_cast<uint32_t>(sc) - 1];
        const uint8_t *comp_sub_chunk = comp_chunk_start + sub_chunk_0_off + packed_start_bytes;
        uint16_t *warp_renorm_buf = smem.decode.warp[threadIdx.x / WARP_SIZE].renorm_buf;

        nvcompStatus_t *status = nullptr;
        if constexpr (BOUNDS_CHECK)
        {
          status = &block_statuses[chunk_idx];
        }

        decode_sub_chunk<BOUNDS_CHECK, Policy>(
          table,
          comp_sub_chunk,
          comp_sub_chunk_size,
          uncomp_sub_chunk,
          sub_chunk_mantissas,
          warp_renorm_buf,
          uncomp_sub_chunk_symbols,
          status
        );

        __syncwarp();
      };

      if constexpr (ONE_SUBCHUNK_PER_WARP)
      {
        if (static_cast<uint32_t>(sub_chunk_idx) < actual_sub_chunk_count)
        {
          decode_one(sub_chunk_idx);
        }
      }
      else
      {
        const int warp_stride = static_cast<int>(gridDim.y * NUM_DECOMP_WARPS_PER_CTA);
        for (int sc = sub_chunk_idx; sc < static_cast<int>(actual_sub_chunk_count); sc += warp_stride)
        {
          decode_one(sc);
        }
      }
    };

    // if constexpr on MODE / STATES_PER_LANE so each specialization compiles ONLY
    // its decode path. Generic branches on the bitstream.
    if constexpr (MODE == DecodeMode::Fp8 && STATES_PER_LANE == 2)
    {
      run(FP8DecodePolicy<FP8X2Impl>{});
    }
    else if constexpr (MODE == DecodeMode::Fp8 && STATES_PER_LANE == 1)
    {
      run(FP8DecodePolicy<FP8X1Impl>{});
    }
    else if constexpr (MODE == DecodeMode::Fp32 && STATES_PER_LANE == 2)
    {
      run(FP32DecodePolicy<FP32X2Impl>{});
    }
    else if constexpr (MODE == DecodeMode::Fp32 && STATES_PER_LANE == 1)
    {
      run(FP32DecodePolicy<FP32X1Impl>{});
    }
    else if constexpr (MODE == DecodeMode::Fp16 && STATES_PER_LANE == 2)
    {
      run(FP16DecodePolicy<FP16X2Impl>{});
    }
    else if constexpr (MODE == DecodeMode::Fp16 && STATES_PER_LANE == 1)
    {
      run(FP16DecodePolicy<FP16X1Impl>{});
    }
    else if constexpr (MODE == DecodeMode::Char && STATES_PER_LANE == 2)
    {
      run(CharDecodePolicy<CharX2Impl>{});
    }
    else if constexpr (MODE == DecodeMode::Char && STATES_PER_LANE == 1)
    {
      run(CharDecodePolicy<CharX1Impl>{});
    }
    else // DecodeMode::Generic
    {
      const ANS_fixed_header *hdr = reinterpret_cast<const ANS_fixed_header *>(comp_chunk_start);
      assert(hdr->states_per_lane() == 1 || hdr->states_per_lane() == 2);
      const AnsStreamType stream_type = hdr->type();
      const bool runtime_is_char = stream_type == AnsStreamType::Char;
      const bool runtime_is_fp8 = stream_type == AnsStreamType::Fp8;
      const bool runtime_is_fp32 = stream_type == AnsStreamType::Fp32;
      const bool one_state = hdr->states_per_lane() == 1;
      if (runtime_is_char)
      {
        if (one_state)
        {
          run(CharDecodePolicy<CharX1Impl>{});
        }
        else
        {
          run(CharDecodePolicy<CharX2Impl>{});
        }
      }
      else if (runtime_is_fp8)
      {
        if (one_state)
        {
          run(FP8DecodePolicy<FP8X1Impl>{});
        }
        else
        {
          run(FP8DecodePolicy<FP8X2Impl>{});
        }
      }
      else if (runtime_is_fp32)
      {
        if (one_state)
        {
          run(FP32DecodePolicy<FP32X1Impl>{});
        }
        else
        {
          run(FP32DecodePolicy<FP32X2Impl>{});
        }
      }
      else if (one_state)
      {
        run(FP16DecodePolicy<FP16X1Impl>{});
      }
      else
      {
        run(FP16DecodePolicy<FP16X2Impl>{});
      }
    }
  }
}

__global__ void decompress_get_sizes_kernel(
  const void *const *comp_chunks,
  const size_t *comp_chunk_sizes,
  // need the other arg here for the comp block sizes to ensure safety
  size_t *actual_uncomp_chunk_sizes,
  const size_t num_uncomp_chunks
)
{
  auto chunk_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (chunk_id < num_uncomp_chunks)
  {
    actual_uncomp_chunk_sizes[chunk_id] =
      read_metadata_uncomp_block_size(comp_chunks[chunk_id], static_cast<uint32_t>(comp_chunk_sizes[chunk_id]));
  }
}

} // namespace detail
} //namespace ans_gpu_lib
