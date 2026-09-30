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
#include <ans/defrag.cuh>
#include <ans/histogram.cuh>
#include <ans/normalize_counts_common.cuh>
#include <ans/symbol_encoder.cuh>
#include <cooperative_groups.h>
#include <limits>
#include <nvcomp/ans.h>

namespace cg = cooperative_groups;

namespace ans_gpu_lib
{
namespace detail
{

// Single templated shared-memory block for compress_chunk / compress_kernel.
// NumWarps defaults to the host LLIF launch (4). The in-kernel device API used
// by a 256-thread GEMM CTA instantiates NumWarps = 8.
template <typename EncodePolicy, int NumWarps = NUM_COMP_WARPS_PER_CTA>
struct CompressSmem
{
  static constexpr int kThreads = NumWarps * WARP_SIZE;
  static_assert(NumWarps > 0, "CompressSmem needs at least one warp");
  static_assert(kThreads >= MAX_SUB_CHUNKS_PER_CHUNK, "defrag needs THREADS >= MAX_SUB_CHUNKS_PER_CHUNK");
  static_assert(NV_SYMBOL_COUNT % kThreads == 0, "BLOCK_DIM_X must divide NV_SYMBOL_COUNT");

  uint8_t *ans_comp_chunk;
  struct Scratch
  {
    __align__(16) uint8_t input_buf[NumWarps][EncodePolicy::ENC_BUF_BYTES];
    int shared_norm_counts[NV_SYMBOL_COUNT];
    uint32_t shared_max_symbol_value;
  } scratch;
  union HistTable
  {
    uint32_t shared_cta_counts[NV_SYMBOL_COUNT];
    uint2 shared_table[NV_SYMBOL_COUNT];
    DefragSmem<kThreads> defrag;
  } ht;
};

// Tail of a zero-ANS-symbol chunk: a header with one sub-chunk of size 0, then
// defrag for the final compressed size. The caller has already written the size
// prefix (and any fp8 trailing raw byte) via the policy hooks.
// header_uncomp_size comes from EncodePolicy::header_uncomp_size.
template <int BlockThreads>
inline __device__ void finish_zero_ans_symbol_chunk(
  void *comp_chunk,
  uint8_t *ans_comp_chunk,
  IndexT header_uncomp_size,
  int max_sub_chunk_size,
  size_t &comp_chunk_size_out,
  uint32_t slot_words,
  DefragSmem<BlockThreads> &defrag_smem
)
{
  // ans_comp_chunk is already 8-byte aligned: output buffer is 8-aligned and
  // histogram_write_chunk_prefix advances by a multiple of sizeof(size_t).
  assert(reinterpret_cast<uintptr_t>(ans_comp_chunk) % alignof(ANS_sub_chunk_header) == 0);
  ANS_sub_chunk_header *header = reinterpret_cast<ANS_sub_chunk_header *>(ans_comp_chunk);
  if (threadIdx.x == 0)
  {
    header->init(/*num_sub_chunks=*/1, header_uncomp_size, max_sub_chunk_size, /*max_sym=*/0, DEFAULT_TABLELOG);
    header->get_sub_chunk_sizes()[0] = 0;
  }
  __syncthreads();
  defrag_chunk_cta<BlockThreads, ANS_FUSED_DEFRAG_UINT4S_PER_THREAD>(
    comp_chunk,
    slot_words,
    comp_chunk_size_out,
    defrag_smem
  );
}

// Shared normalize + table build. Consumes the flat CTA histogram in
// shared_cta_counts[0..NV_SYMBOL_COUNT), writes normalized counts and max symbol,
// then builds the packed {cdf | div_scale | shift, magic} encoding table into
// shared_table. Requires num_symbols > 0. Ends CTA-synchronized.
template <int BlockThreads>
inline __device__ void normalize_and_build_table(
  uint32_t *shared_cta_counts, // smem [NV_SYMBOL_COUNT] flat histogram (consumed by normalize)
  int *shared_norm_counts, // smem [NV_SYMBOL_COUNT] out
  uint2 *shared_table, // smem [NV_SYMBOL_COUNT] out (may alias shared_cta_counts)
  uint32_t *shared_max_symbol_value, // smem out
  IndexT num_symbols
)
{
  assert(num_symbols > 0);

  // ---- Phase 2: normalize (parallel, all BlockThreads threads) into smem ----
  if (threadIdx.x == 0)
  {
    *shared_max_symbol_value = 0;
  }
  __syncthreads();
  normalize_counts_chunk_parallel<BlockThreads, int>(
    threadIdx.x,
    shared_cta_counts,
    shared_norm_counts,
    shared_max_symbol_value,
    num_symbols,
    DEFAULT_TABLELOG
  );
  __syncthreads();

  // ---- Phase 3: build the packed {cdf | div_scale | shift, magic} table in smem ----
  uint32_t acc = 0;
  for (uint32_t symbol = threadIdx.x; symbol < NV_SYMBOL_COUNT; symbol += BlockThreads)
  {
    uint32_t abs_norm = abs(shared_norm_counts[symbol]);
    uint32_t total;
    uint32_t prefix = block_excl_prefix_sum<BlockThreads>(abs_norm, total);

    constexpr uint32_t num_bits_in_uint32_t = 32;
    uint32_t shift = num_bits_in_uint32_t - __clz(abs_norm - 1);
    uint64_t magic = ((1ULL << num_bits_in_uint32_t) * ((1ULL << shift) - abs_norm)) / abs_norm + 1;

    uint32_t cdf = acc + prefix;
    shared_table[symbol] = make_uint2(pack_cdf_div_scale_shift(abs_norm, cdf, shift), static_cast<uint32_t>(magic));

    acc += total;
  }
  __syncthreads();
}

// Fused prepare: Phase 1 (histogram + optional side-band) + Phase 2 (normalize) +
// Phase 3 (build table). Requires EncodePolicy::num_symbols(bytes) > 0.
template <typename EncodePolicy, int BlockThreads>
inline __device__ void prepare_encoding_table(
  const void *uncomp_chunk,
  IndexT uncomp_chunk_size_bytes, // BYTES (raw input size)
  void *comp_chunk, // raw comp chunk; chunk-header destination
  uint8_t *&ans_comp_chunk_out, // CTA-local out (Phase 1)
  uint2 *shared_table, // smem [NV_SYMBOL_COUNT]
  int *shared_norm_counts, // smem [NV_SYMBOL_COUNT]
  uint32_t *shared_max_symbol_value, // smem
  uint32_t *shared_cta_counts // smem [NV_SYMBOL_COUNT]
)
{
  assert(EncodePolicy::num_symbols(uncomp_chunk_size_bytes) > 0);

  auto group = cg::this_thread_block();
  compute_histogram<EncodePolicy, BlockThreads>(
    uncomp_chunk,
    static_cast<uint8_t *>(comp_chunk),
    uncomp_chunk_size_bytes,
    ans_comp_chunk_out,
    shared_cta_counts,
    group
  );

  // shared_table may alias shared_cta_counts (same union member); safe because the
  // counts are consumed by normalize into shared_norm_counts before the table build
  // writes them.
  normalize_and_build_table<BlockThreads>(
    shared_cta_counts,
    shared_norm_counts,
    shared_table,
    shared_max_symbol_value,
    EncodePolicy::num_symbols(uncomp_chunk_size_bytes)
  );
}

// One-chunk ANS compress. The calling kernel's blockDim.x must equal BlockThreads
// (CUB BlockScan / histogram / defrag all __syncthreads() the full CTA).
template <typename EncodePolicy, int BlockThreads>
inline __device__ void compress_chunk(
  void *comp_chunk,
  const void *uncomp_chunk,
  IndexT bytes,
  size_t &comp_chunk_size_out,
  int max_sub_chunk_size,
  uint32_t slot_words,
  CompressSmem<EncodePolicy, BlockThreads / WARP_SIZE> &smem
)
{
  constexpr int NumWarps = BlockThreads / WARP_SIZE;
  static_assert(BlockThreads % WARP_SIZE == 0, "compress_chunk BlockThreads must be a warp multiple");
  assert(static_cast<int>(blockDim.x) == BlockThreads);

  const auto wid = threadIdx.x / WARP_SIZE_U;
  assert(wid < static_cast<unsigned>(NumWarps));

  static_assert(
    nvcompANSCompressionMaxAllowedChunkSize <= std::numeric_limits<IndexT>::max(),
    "ANS chunk size exceeds IndexT range"
  );

  const IndexT symbols = EncodePolicy::num_symbols(bytes);
  const IndexT header_uncomp_size = EncodePolicy::header_uncomp_size(bytes);

  if (symbols == 0)
  {
    if (threadIdx.x == 0)
    {
      uint8_t *const comp = static_cast<uint8_t *>(comp_chunk);
      smem.ans_comp_chunk = EncodePolicy::histogram_write_chunk_prefix(comp, bytes);
      EncodePolicy::write_chunk_tail(comp, uncomp_chunk, bytes);
    }
    __syncthreads();
    finish_zero_ans_symbol_chunk<BlockThreads>(
      comp_chunk,
      smem.ans_comp_chunk,
      header_uncomp_size,
      max_sub_chunk_size,
      comp_chunk_size_out,
      slot_words,
      smem.ht.defrag
    );
    return;
  }

  prepare_encoding_table<EncodePolicy, BlockThreads>(
    uncomp_chunk,
    bytes,
    comp_chunk,
    smem.ans_comp_chunk,
    smem.ht.shared_table,
    smem.scratch.shared_norm_counts,
    &smem.scratch.shared_max_symbol_value,
    smem.ht.shared_cta_counts
  );

  const uint8_t max_sym = static_cast<uint8_t>(smem.scratch.shared_max_symbol_value);
  const int num_sub_chunks_per_chunk = static_cast<int>(nvcomp::roundUpDiv(symbols, max_sub_chunk_size));

  assert(reinterpret_cast<uintptr_t>(smem.ans_comp_chunk) % alignof(ANS_sub_chunk_header) == 0);
  ANS_sub_chunk_header *sub_chunk_header = reinterpret_cast<ANS_sub_chunk_header *>(smem.ans_comp_chunk);
  if (threadIdx.x == 0)
  {
    sub_chunk_header->init(num_sub_chunks_per_chunk, header_uncomp_size, max_sub_chunk_size, max_sym, DEFAULT_TABLELOG);
  }
  for (int i = threadIdx.x; i <= max_sym; i += BlockThreads)
  {
    sub_chunk_header->get_norm_counts()[i] = smem.scratch.shared_norm_counts[i];
  }
  __syncthreads();

  for (int sc = static_cast<int>(wid); sc < num_sub_chunks_per_chunk; sc += NumWarps)
  {
    // WAR guard (cross-sub-chunk): this warp reuses input_buf[wid] across sub-chunks, so
    // the previous sub-chunk's last staged read must finish before the next stage.
    __syncwarp();
    encode_sub_chunk<EncodePolicy>(
      sc,
      uncomp_chunk,
      symbols,
      smem.ans_comp_chunk,
      max_sub_chunk_size,
      smem.ht.shared_table,
      smem.scratch.input_buf[wid]
    );
  }

  __syncthreads();
  defrag_chunk_cta<BlockThreads, ANS_FUSED_DEFRAG_UINT4S_PER_THREAD>(
    comp_chunk,
    slot_words,
    comp_chunk_size_out,
    smem.ht.defrag
  );
}

// Unified fused compress kernel. One CTA per chunk; host dispatch selects the
// EncodePolicy instantiation by data type.
template <typename EncodePolicy>
__global__ __launch_bounds__(NUM_COMP_WARPS_PER_CTA *WARP_SIZE, ANS_COMP_MIN_BLOCKS_PER_SM) void compress_kernel(
  void *const *comp_chunks, // raw comp chunks (chunk-header + side-band dest)
  const void *const *uncomp_chunks, // raw source
  const size_t *uncomp_chunk_sizes, // BYTES (raw input size)
  int max_sub_chunk_size, // symbols
  size_t *comp_chunk_sizes, // out: final per-chunk compressed size (written by fused defrag)
  uint32_t slot_words // worst-case sub-chunk slot stride, in uint32 words
)
{
  __shared__ CompressSmem<EncodePolicy> smem;
  const auto bid = blockIdx.x;
  const IndexT bytes = static_cast<IndexT>(uncomp_chunk_sizes[bid]);
  compress_chunk<EncodePolicy, NUM_COMP_THREADS_PER_CTA>(
    comp_chunks[bid],
    uncomp_chunks[bid],
    bytes,
    comp_chunk_sizes[bid],
    max_sub_chunk_size,
    slot_words,
    smem
  );
}

} // namespace detail
} // namespace ans_gpu_lib
