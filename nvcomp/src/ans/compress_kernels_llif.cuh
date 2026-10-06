/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <ans/ans_arch_profile.cuh>
#include <ans/compress_kernels.cuh>
#include <ans/histogram.cuh>
#include <ans/normalize_counts_common.cuh>
#include <ans/simple_defrag.cuh>
#include <ans/symbol_encoder.cuh>

namespace ans_gpu_lib
{
namespace detail
{

// Single templated shared-memory block for compress_kernel<EncodePolicy>. The histogram
// cp.async window + norm-counts/max-symbol are live together (a struct); the histogram
// counts, the packed encode table, and the defrag scratch never co-exist (a union). Each
// instantiation reserves exactly its type's layout (mirrors SetupAndDecodeSmem on decode).
// The histogram count buffer is a single CTA-shared row (NV_SYMBOL_COUNT bins) for every
// type.
template <typename EncodePolicy>
struct CompressSmem
{
  // Per-warp histogram cp.async window: DEPTH rows x 32 lanes x sizeof(LoadT).
  static constexpr int HIST_STAGE_BYTES_PER_WARP = EncodePolicy::HIST_PREFETCH_DEPTH * WARP_SIZE *
                                                   static_cast<int>(sizeof(typename EncodePolicy::LoadT));

  struct Scratch
  {
    __align__(16) uint8_t hist_stage[NUM_COMP_WARPS_PER_CTA][HIST_STAGE_BYTES_PER_WARP];
    int shared_norm_counts[NV_SYMBOL_COUNT];
    uint32_t shared_min_symbol_value;
    uint32_t shared_max_symbol_value;
    // Set when a warp encodes a symbol the sampled model does not cover (sampling only).
    uint32_t shared_uncovered;
  } scratch;
  union HistTable
  {
    uint32_t shared_cta_counts[NV_SYMBOL_COUNT];
    uint2 shared_table[NV_SYMBOL_COUNT];
  } ht;
  // Parked chunk bases and per-warp 32-bit offsets from comp_chunk. Encode rematerializes
  // pointers from these so 64-bit addresses do not stay live across the hot loop.
  EncodeChunkSmem enc;
};

// Tail of a zero-ANS-symbol chunk: a header with one sub-chunk of size 0, then
// defrag for the final compressed size. The caller has already written any fp8
// trailing raw byte via the policy hook.
template <typename EncodePolicy>
inline __device__ void finish_zero_ans_symbol_chunk(
  void *comp_chunk,
  IndexT uncomp_bytes,
  int max_sub_chunk_size,
  uint32_t &comp_chunk_size_out,
  uint32_t subchunk_comp_buffer_size
)
{
  ANS_fixed_header *fixed_header = static_cast<ANS_fixed_header *>(comp_chunk);
  if (threadIdx.x == 0)
  {
    // Degenerate [0, 0] symbol range: one norm_counts entry, which stays 0.
    fixed_header->init(
      uncomp_bytes,
      static_cast<uint32_t>(max_sub_chunk_size) * EncodePolicy::INPUT_BYTES_PER_SYMBOL,
      /*min_symbol=*/0,
      /*max_symbol=*/0,
      static_cast<uint8_t>(EncodePolicy::TABLELOG),
      EncodePolicy::STREAM_TYPE,
      static_cast<uint8_t>(EncodePolicy::STATES_PER_LANE)
    );
    fixed_header->get_norm_counts()[0] = 0;
    fixed_header->get_sub_chunk_sizes()[0] = 0;
  }
  __syncthreads();
  // One sub-chunk, so nothing moves; this only publishes the final size.
  simple_defrag_chunk_cta<NUM_COMP_THREADS_PER_CTA>(fixed_header, subchunk_comp_buffer_size, comp_chunk_size_out);
}

// CTA-wide sum of the NV_SYMBOL_COUNT histogram bins. Ends CTA-synchronized.
inline __device__ uint32_t cta_sum_histogram_counts(const uint32_t *shared_cta_counts)
{
  constexpr int BLOCK_DIM_X = NUM_COMP_WARPS_PER_CTA * WARP_SIZE;
  __shared__ uint32_t shared_count_total;
  if (threadIdx.x == 0)
  {
    shared_count_total = 0;
  }
  __syncthreads();
  uint32_t thread_total = 0;
  for (uint32_t symbol = threadIdx.x; symbol < NV_SYMBOL_COUNT; symbol += BLOCK_DIM_X)
  {
    thread_total += shared_cta_counts[symbol];
  }
  atomicAdd(&shared_count_total, thread_total);
  __syncthreads();
  return shared_count_total;
}

// Span floor / dilation (sampled path only). Flooring a missed symbol to count >= 1 buys
// it a table slot for a few counts out of the table's total. Runs before normalize so the
// floors normalize like any other count; the min/max scan is scratch, as normalize and the
// table build recompute both. Ends CTA-synchronized.
inline __device__ void apply_span_floor(
  uint32_t *shared_cta_counts, // smem [NV_SYMBOL_COUNT], floored in place
  uint32_t *shared_min_symbol_value, // smem scratch
  uint32_t *shared_max_symbol_value, // smem scratch
  HistFloor floor, // what the sampled model must cover; None when this chunk was exact
  int dilate_neg, // ObservedBand: floor this many symbols below the observed min
  int dilate_pos // ObservedBand: floor this many symbols above the observed max
)
{
  constexpr int BLOCK_DIM_X = NUM_COMP_WARPS_PER_CTA * WARP_SIZE;

  if (floor == HistFloor::WholeAlphabet)
  {
    for (uint32_t symbol = threadIdx.x; symbol < NV_SYMBOL_COUNT; symbol += BLOCK_DIM_X)
    {
      if (shared_cta_counts[symbol] == 0)
      {
        shared_cta_counts[symbol] = 1;
      }
    }
    __syncthreads();
  }
  else if (floor == HistFloor::ObservedBand)
  {
    if (threadIdx.x == 0)
    {
      *shared_min_symbol_value = NV_SYMBOL_COUNT - 1;
      *shared_max_symbol_value = 0;
    }
    __syncthreads();
    for (uint32_t symbol = threadIdx.x; symbol < NV_SYMBOL_COUNT; symbol += BLOCK_DIM_X)
    {
      if (shared_cta_counts[symbol] != 0)
      {
        atomicMin(shared_min_symbol_value, symbol);
        atomicMax(shared_max_symbol_value, symbol);
      }
    }
    __syncthreads();

    const int floor_lo = max(0, static_cast<int>(*shared_min_symbol_value) - dilate_neg);
    const int floor_hi =
      min(static_cast<int>(NV_SYMBOL_COUNT) - 1, static_cast<int>(*shared_max_symbol_value) + dilate_pos);
    for (uint32_t symbol = threadIdx.x; symbol < NV_SYMBOL_COUNT; symbol += BLOCK_DIM_X)
    {
      if (static_cast<int>(symbol) >= floor_lo && static_cast<int>(symbol) <= floor_hi &&
          shared_cta_counts[symbol] == 0)
      {
        shared_cta_counts[symbol] = 1;
      }
    }
    __syncthreads();
  }
}

// Shared normalize + table build. Consumes the flat CTA histogram in
// shared_cta_counts[0..NV_SYMBOL_COUNT), writes normalized counts and the observed
// symbol window, then builds the packed {cdf | div_scale | shift, magic} encoding table
// into shared_table. Requires num_symbols > 0. Ends CTA-synchronized.
template <uint32_t TABLELOG>
inline __device__ void normalize_and_build_table(
  uint32_t *shared_cta_counts, // smem [NV_SYMBOL_COUNT] flat histogram (consumed by normalize)
  int *shared_norm_counts, // smem [NV_SYMBOL_COUNT] out
  uint2 *shared_table, // smem [NV_SYMBOL_COUNT] out (may alias shared_cta_counts)
  uint32_t *shared_min_symbol_value, // smem out
  uint32_t *shared_max_symbol_value, // smem out
  HistFloor floor, // what the sampled model must cover; None when this chunk was exact
  int dilate_neg, // ObservedBand: floor this many symbols below the observed min
  int dilate_pos // ObservedBand: floor this many symbols above the observed max
)
{
  constexpr int BLOCK_DIM_X = NUM_COMP_WARPS_PER_CTA * WARP_SIZE;

  // ---- Phase 1: span floor / dilation (sampled path only) ----
  apply_span_floor(shared_cta_counts, shared_min_symbol_value, shared_max_symbol_value, floor, dilate_neg, dilate_pos);

  // ---- Phase 2: normalize (parallel, all BLOCK_DIM_X threads) into smem ----
  // Normalize against the total the counts ACTUALLY hold, summed here rather than taken
  // from the caller. For an exact histogram the two agree by construction, but a sampled
  // one does not: each warp truncates its own range independently, and the span floor adds
  // counts afterwards. Feeding normalize a total larger than the counts sum to would leave
  // part of the table unallocated, and a decode table with unfilled slots mis-decodes.
  if (threadIdx.x == 0)
  {
    *shared_max_symbol_value = 0;
    *shared_min_symbol_value = NV_SYMBOL_COUNT - 1;
  }
  __syncthreads();
  const IndexT count_total = static_cast<IndexT>(cta_sum_histogram_counts(shared_cta_counts));
  assert(count_total > 0);

  normalize_counts_chunk_parallel(
    threadIdx.x,
    shared_cta_counts,
    shared_norm_counts,
    shared_max_symbol_value,
    count_total,
    TABLELOG
  );
  __syncthreads();

  // ---- Phase 3: build the packed {cdf | div_scale | shift, magic} table in smem ----
  // The stored model only covers [min, max], so the lower bound is picked up here, off
  // the same pass: normalize maps a symbol to a nonzero count exactly when it occurred,
  // so the first nonzero entry is the window's lower bound. (normalize already reports
  // the upper bound, which its own fallback loops need.) A chunk with symbols always has
  // at least one nonzero entry, so min <= max.
  static_assert(BLOCK_DIM_X == NV_SYMBOL_COUNT, "the table build assumes one symbol per thread");
  {
    const uint32_t symbol = threadIdx.x;
    const uint32_t abs_norm = static_cast<uint32_t>(abs(shared_norm_counts[symbol]));
    if (abs_norm != 0)
    {
      atomicMin(shared_min_symbol_value, symbol);
    }
    // One symbol per thread, so the symbol's cdf is just the exclusive prefix.
    const uint32_t cdf = block_excl_prefix_sum<BLOCK_DIM_X>(abs_norm);

    uint32_t packed = 0;
    uint32_t magic = 0;
    if (abs_norm != 0)
    {
      constexpr uint32_t num_bits_in_uint32_t = 32;
      const uint32_t shift = num_bits_in_uint32_t - __clz(abs_norm - 1);
      const uint64_t m = ((1ULL << num_bits_in_uint32_t) * ((1ULL << shift) - abs_norm)) / abs_norm + 1;
      packed = pack_cdf_div_scale_shift<TABLELOG>(abs_norm, cdf, shift);
      magic = static_cast<uint32_t>(m);
    }
    // Uncovered symbol (count 0): magic == 0, which a covered symbol can never produce
    // (its magic is always >= 1). That makes a symbol the model does not cover
    // unambiguously detectable at encode time -- distinct from a lone dominant symbol,
    // whose div_scale also packs to 0 -- and it avoids the div-by-zero the magic formula
    // would otherwise hit here.
    shared_table[symbol] = make_uint2(packed, magic);
  }
  __syncthreads();
}

// Fused prepare: Phase 1 (histogram + optional mantissas) + Phase 2 (normalize) +
// Phase 3 (build table). Requires EncodePolicy::num_symbols(bytes) > 0.
template <typename EncodePolicy>
inline __device__ void prepare_encoding_table(
  const void *uncomp_chunk,
  IndexT uncomp_chunk_size_bytes, // BYTES (raw input size)
  void *comp_chunk, // raw comp chunk; chunk-header destination
  uint8_t num_sub_chunks,
  uint2 *shared_table, // smem [NV_SYMBOL_COUNT]
  int *shared_norm_counts, // smem [NV_SYMBOL_COUNT]
  uint32_t *shared_min_symbol_value, // smem
  uint32_t *shared_max_symbol_value, // smem
  uint32_t *shared_cta_counts, // smem [NV_SYMBOL_COUNT]
  uint8_t *hist_stage, // smem [NUM_COMP_WARPS_PER_CTA][stage_bytes_per_warp]
  int stage_bytes_per_warp,
  uint32_t sample_shift // 0 = exact; else keep 1/2^shift of each warp slice
)
{
  constexpr int BLOCK_DIM_X = NUM_COMP_WARPS_PER_CTA * WARP_SIZE;
  assert(EncodePolicy::num_symbols(uncomp_chunk_size_bytes) > 0);
  static_assert(
    EncodePolicy::HIST_FLOOR == HistFloor::ObservedBand ||
      (EncodePolicy::HIST_DILATE_NEG == 0 && EncodePolicy::HIST_DILATE_POS == 0),
    "the dilation radii are only read for HistFloor::ObservedBand, so nonzero radii on any other floor "
    "would be silently ignored"
  );

  compute_histogram<EncodePolicy, BLOCK_DIM_X>(
    uncomp_chunk,
    static_cast<uint8_t *>(comp_chunk),
    uncomp_chunk_size_bytes,
    num_sub_chunks,
    shared_cta_counts,
    sample_shift,
    hist_stage,
    stage_bytes_per_warp
  );

  // An exact histogram needs no floor, whatever the policy would ask for when sampling.
  const HistFloor floor = sample_shift > 0 ? EncodePolicy::HIST_FLOOR : HistFloor::None;

  // shared_table may alias shared_cta_counts (same union member); safe because the
  // counts are consumed by normalize into shared_norm_counts before the table build
  // writes them.
  normalize_and_build_table<EncodePolicy::TABLELOG>(
    shared_cta_counts,
    shared_norm_counts,
    shared_table,
    shared_min_symbol_value,
    shared_max_symbol_value,
    floor,
    static_cast<int>(EncodePolicy::HIST_DILATE_NEG),
    static_cast<int>(EncodePolicy::HIST_DILATE_POS)
  );
}

// SampledHist picks the model (sampled vs. exact); Detect instruments the encode loop to
// report a symbol the model missed. Separate: WholeAlphabet samples without needing it.
template <typename EncodePolicy, bool Detect, bool SampledHist>
__device__ __forceinline__ bool prepare_and_encode_chunk(CompressSmem<EncodePolicy> &smem)
{
  static_assert(SampledHist || !Detect, "an exact histogram cannot leave a symbol uncovered");
  const auto wid = threadIdx.x / WARP_SIZE_U;
  const IndexT bytes = smem.enc.bytes;
  const IndexT symbols = smem.enc.symbols;
  const int max_sub_chunk_size = smem.enc.max_sub_chunk_size;
  const int num_sub_chunks_per_chunk = static_cast<int>(smem.enc.num_sub_chunks);
  const uint32_t sample_shift = SampledHist ? smem.enc.sample_shift : 0u;

  if constexpr (Detect)
  {
    if (threadIdx.x == 0)
    {
      smem.scratch.shared_uncovered = 0u;
    }
    __syncthreads();
  }

  const uint8_t nsc = static_cast<uint8_t>(num_sub_chunks_per_chunk);
  prepare_encoding_table<EncodePolicy>(
    smem.enc.uncomp_chunk,
    bytes,
    smem.enc.comp_chunk,
    nsc,
    smem.ht.shared_table,
    smem.scratch.shared_norm_counts,
    &smem.scratch.shared_min_symbol_value,
    &smem.scratch.shared_max_symbol_value,
    smem.ht.shared_cta_counts,
    &smem.scratch.hist_stage[0][0],
    CompressSmem<EncodePolicy>::HIST_STAGE_BYTES_PER_WARP,
    sample_shift
  );

  // Symbol window for the stored norm_counts: exactly the symbols the model covers, so
  // norm_counts[i] is symbol (min_sym + i). For the fp16 rotate the occupied symbols are
  // one contiguous exponent band, which is what makes this window tight.
  const uint32_t min_sym = smem.scratch.shared_min_symbol_value;
  const uint32_t max_sym = smem.scratch.shared_max_symbol_value;
  const uint32_t norm_entries = max_sym - min_sym + 1;

  ANS_fixed_header *fixed_header = static_cast<ANS_fixed_header *>(smem.enc.comp_chunk);
  if (threadIdx.x == 0)
  {
    fixed_header->init(
      bytes,
      static_cast<uint32_t>(max_sub_chunk_size) * EncodePolicy::INPUT_BYTES_PER_SYMBOL,
      static_cast<uint8_t>(min_sym),
      static_cast<uint8_t>(max_sym),
      static_cast<uint8_t>(EncodePolicy::TABLELOG),
      EncodePolicy::STREAM_TYPE,
      static_cast<uint8_t>(EncodePolicy::STATES_PER_LANE)
    );

    // Policies that emit the mantissas while encoding need the region offset; the others
    // already filled it during the histogram pass and leave this 0.
    smem.enc.mantissas_chunk_offset = EncodePolicy::ENCODE_WRITES_MANTISSAS
                                        ? ans_mantissas_offset(EncodePolicy::STREAM_TYPE, nsc)
                                        : 0u;
  }
  // Counts offset is a function of type, bytes, and nsc (all CTA-uniform here), so this
  // write does not wait on init. Encode reads the header after the syncthreads below.
  int16_t *const norm_counts =
    reinterpret_cast<int16_t *>(smem.enc.comp_at(ans_norm_counts_offset(EncodePolicy::STREAM_TYPE, bytes, nsc)));
  for (uint32_t i = threadIdx.x; i < norm_entries; i += blockDim.x)
  {
    norm_counts[i] = smem.scratch.shared_norm_counts[min_sym + i];
  }
  __syncthreads();

  [[maybe_unused]] bool warp_uncovered = false;
  for (int sc = wid; sc < num_sub_chunks_per_chunk; sc += NUM_COMP_WARPS_PER_CTA)
  {
    warp_uncovered |=
      encode_sub_chunk<EncodePolicy, Detect>(sc, symbols, max_sub_chunk_size, smem.ht.shared_table, smem.enc);
  }

  if constexpr (Detect)
  {
    // One shared write per warp that saw a miss, not per symbol.
    if (warp_uncovered && get_lane_id() == 0)
    {
      atomicOr(&smem.scratch.shared_uncovered, 1u);
    }
  }
  __syncthreads();

  if constexpr (Detect)
  {
    return smem.scratch.shared_uncovered != 0u;
  }
  else
  {
    return false;
  }
}

// Unified fused compress kernel. One CTA per chunk; host dispatch selects the
// EncodePolicy instantiation by data type, and Sampled by the option given to compress.
// Batch pointer args are __grid_constant__: grid-uniform and immutable, so they
// stay in the parameter bank instead of the live register file.
template <typename EncodePolicy, bool Sampled>
__global__
__launch_bounds__(NUM_COMP_WARPS_PER_CTA *WARP_SIZE, EncodePolicy::COMP_MIN_BLOCKS_PER_SM) void compress_kernel(
  __grid_constant__ void *const *const comp_chunks, // raw comp chunks (chunk-header + mantissa dest)
  const __grid_constant__ void *const *const uncomp_chunks, // raw source
  const __grid_constant__ size_t *const uncomp_chunk_sizes, // BYTES (raw input size)
  const __grid_constant__ int max_sub_chunk_size, // symbols
  __grid_constant__ size_t *const comp_chunk_sizes, // out: final per-chunk compressed size
  __grid_constant__ nvcompStatus_t *const device_statuses, // optional per-chunk status
  const __grid_constant__ uint32_t subchunk_comp_buffer_size,
  const __grid_constant__ uint32_t histogram_reduction_log2, // 0 = exact; else keep 1/2^shift of each warp slice
  // Optional col-major matrix pack (CUTLASS C). Null pack_C skips packing.
  // FLOAT16 copies 16-bit elements (fp16/bf16); other types keep a float tile.
  const __grid_constant__ void *const pack_C,
  const __grid_constant__ int pack_ldc,
  const __grid_constant__ int pack_M,
  const __grid_constant__ int pack_N,
  const __grid_constant__ int pack_tile_m,
  const __grid_constant__ int pack_tile_n,
  const __grid_constant__ int pack_mn_swapped
)
{
  __shared__ CompressSmem<EncodePolicy> smem;
  __shared__ uint32_t packed_chunk_size_bytes;
  // One CTA handles one chunk. 2-D grid when packing from a matrix (tiles_m x tiles_n).
  volatile __shared__ uint32_t chunk_idx_handoff;

  if (threadIdx.x == 0)
  {
    chunk_idx_handoff = pack_C != nullptr ? blockIdx.x + blockIdx.y * gridDim.x : blockIdx.x;
  }
  __syncthreads();
  const uint32_t bid = chunk_idx_handoff;

  if (pack_C != nullptr)
  {
    const int tile_m = pack_mn_swapped ? static_cast<int>(blockIdx.y) : static_cast<int>(blockIdx.x);
    const int tile_n = pack_mn_swapped ? static_cast<int>(blockIdx.x) : static_cast<int>(blockIdx.y);
    const int m0 = tile_m * pack_tile_m;
    const int n0 = tile_n * pack_tile_n;
    const int remain_m = pack_M - m0;
    const int remain_n = pack_N - n0;
    const int rows = remain_m < pack_tile_m ? remain_m : pack_tile_m;
    const int cols = remain_n < pack_tile_n ? remain_n : pack_tile_n;
    const int tile_elems = pack_tile_m * pack_tile_n;
    for (int i = static_cast<int>(threadIdx.x); i < tile_elems; i += static_cast<int>(blockDim.x))
    {
      const int row = i % pack_tile_m;
      const int col = i / pack_tile_m;
      if constexpr (EncodePolicy::STREAM_TYPE == AnsStreamType::Fp16)
      {
        uint16_t *packed = const_cast<uint16_t *>(static_cast<const uint16_t *>(uncomp_chunks[bid]));
        uint16_t packed_value = 0;
        if (row < rows && col < cols)
        {
          packed_value =
            reinterpret_cast<uint16_t const *>(pack_C)[(m0 + row) + (n0 + col) * pack_ldc];
        }
        packed[row + col * pack_tile_m] = packed_value;
      }
      else
      {
        float *packed = const_cast<float *>(static_cast<const float *>(uncomp_chunks[bid]));
        float value = 0.f;
        if (row < rows && col < cols)
        {
          value = reinterpret_cast<float const *>(pack_C)[(m0 + row) + (n0 + col) * pack_ldc];
        }
        packed[row + col * pack_tile_m] = value;
      }
    }
    __syncthreads();
  }

  // Narrow once here: the API caps chunk size far below IndexT's range, so
  // everything downstream indexes with IndexT.
  static_assert(
    nvcompANSCompressionMaxAllowedChunkSize <= std::numeric_limits<IndexT>::max(),
    "ANS chunk size exceeds IndexT range"
  );
  const IndexT bytes = static_cast<IndexT>(uncomp_chunk_sizes[bid]);
  constexpr IndexT INPUT_BYTES_PER_VALUE = EncodePolicy::STREAM_TYPE == AnsStreamType::Fp16   ? sizeof(uint16_t)
                                           : EncodePolicy::STREAM_TYPE == AnsStreamType::Fp32 ? sizeof(uint32_t)
                                                                                              : sizeof(uint8_t);
  if (bytes % INPUT_BYTES_PER_VALUE != 0)
  {
    if (threadIdx.x == 0)
    {
      comp_chunk_sizes[bid] = 0;
      if (device_statuses != nullptr)
      {
        device_statuses[bid] = nvcompErrorCannotCompress;
      }
    }
    return;
  }

  const IndexT symbols = EncodePolicy::num_symbols(bytes);

  // Zero-ANS-symbol chunk: header plus any fp8 trailing raw byte, then defrag.
  // The tail hook persists the fp8 N == 1 raw byte; this skips the histogram.
  if (symbols == 0)
  {
    if (threadIdx.x == 0)
    {
      EncodePolicy::write_chunk_tail(
        static_cast<uint8_t *>(comp_chunks[bid]),
        uncomp_chunks[bid],
        bytes,
        /*num_sub_chunks=*/1
      );
    }
    __syncthreads();
    finish_zero_ans_symbol_chunk<EncodePolicy>(
      comp_chunks[bid],
      bytes,
      max_sub_chunk_size,
      packed_chunk_size_bytes,
      subchunk_comp_buffer_size
    );
    if (threadIdx.x == 0)
    {
      comp_chunk_sizes[bid] = packed_chunk_size_bytes;
    }
    return;
  }

  const uint32_t max_sub_chunk_size_bytes = static_cast<uint32_t>(max_sub_chunk_size) *
                                            EncodePolicy::INPUT_BYTES_PER_SYMBOL;
  const int num_sub_chunks_per_chunk = static_cast<int>(ans_derived_num_sub_chunks(bytes, max_sub_chunk_size_bytes));

  // Only a band floor can leave a symbol without a slot, so it alone instantiates detect /
  // redo; WholeAlphabet and an exact histogram both cover the chunk by construction.
  constexpr bool SAMPLED_HIST = Sampled && EncodePolicy::HIST_FLOOR != HistFloor::None;
  constexpr bool DETECT = SAMPLED_HIST && EncodePolicy::HIST_FLOOR == HistFloor::ObservedBand;
  const bool can_sample = Sampled && symbols >= MIN_SAMPLED_HIST_SYMBOLS;

  if (threadIdx.x == 0)
  {
    smem.enc.uncomp_chunk = uncomp_chunks[bid];
    smem.enc.comp_chunk = comp_chunks[bid];
    smem.enc.bytes = bytes;
    smem.enc.symbols = symbols;
    smem.enc.num_sub_chunks = static_cast<uint32_t>(num_sub_chunks_per_chunk);
    smem.enc.sample_shift = can_sample ? histogram_reduction_log2 : 0u;
    smem.enc.max_sub_chunk_size = max_sub_chunk_size;
  }
  __syncthreads();

  if constexpr (DETECT)
  {
    // Redo from an exact histogram. Rare (dilation covers
    // near-miss symbols), and it overwrites the abandoned first attempt in place.
    // Scalars come from EncodeChunkSmem, not live kernel registers.
    if (prepare_and_encode_chunk<EncodePolicy, true, /*SampledHist=*/true>(smem))
    {
      prepare_and_encode_chunk<EncodePolicy, /*Detect=*/false, /*SampledHist=*/false>(smem);
    }
  }
  else
  {
    prepare_and_encode_chunk<EncodePolicy, false, SAMPLED_HIST>(smem);
  }

  ANS_fixed_header *const defrag_header = static_cast<ANS_fixed_header *>(smem.enc.comp_chunk);

  simple_defrag_chunk_cta<NUM_COMP_THREADS_PER_CTA>(defrag_header, subchunk_comp_buffer_size, packed_chunk_size_bytes);
  if (threadIdx.x == 0)
  {
    comp_chunk_sizes[chunk_idx_handoff] = packed_chunk_size_bytes;
  }
}

} // namespace detail
} // namespace ans_gpu_lib
