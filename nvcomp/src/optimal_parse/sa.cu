/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#include <cub/cub.cuh>

#include <thrust/host_vector.h>
#include <thrust/scan.h>

#include <ciso646>

#include "common.h"
#include "exception.hpp"
#include "nvcomp.hpp"
#include "nvcomp/utils.hpp"
#include "sa.cuh"

using namespace nvcomp;

namespace sa
{

template <typename offset_type>
union rank_t
{
  using type = std::conditional_t<std::is_same_v<offset_type, U16>, U32, U64>;

  type rank;
  offset_type rank_pair[2];
  U8 rank_quad[sizeof(type)];
};

template <typename SizeT, typename Function>
__launch_bounds__(256) __global__ void for_each_chunk_kernel(
  const SizeT *__restrict chunk_starts,
  const SizeT *__restrict chunk_stops,
  Function func,
  size_t ctas_per_chunk
)
{
  const U32 chunk = blockIdx.x / ctas_per_chunk;
  const U32 sub_chunk = blockIdx.x - (chunk * ctas_per_chunk);
  const U32 chunk_start = chunk_starts[chunk];
  const U32 chunk_size = chunk_stops[chunk] - chunk_start;
  const U32 idx = sub_chunk * blockDim.x + threadIdx.x;
  if (idx < chunk_size)
  {
    func(chunk, idx, chunk_start, chunk_size);
  }
}

template <typename SizeT, typename DeviceFunc>
void for_each_chunk(
  cudaStream_t stream,
  const SizeT *chunk_starts,
  const SizeT *chunk_stops,
  size_t num_chunks,
  size_t max_chunk_size,
  DeviceFunc func
)
{
  const dim3 block(256);
  auto ctas_per_chunk = roundUpDiv(max_chunk_size, static_cast<size_t>(block.x));
  auto grid = cuda_dim_cast(num_chunks * ctas_per_chunk);
  for_each_chunk_kernel<<<grid, block, 0, stream>>>(chunk_starts, chunk_stops, func, ctas_per_chunk);
  CUDA_CHECK(cudaGetLastError());
}

constexpr int cta_thread_count = 512;
template <typename offset_type>
__global__ void optimized_rank_iter_kernel(
  offset_type *current_sa,
  uint32_t *device_scan,
  offset_type *device_sa_scan,
  rank_t<offset_type> *ranks,
  const size_t *device_starts,
  const size_t *device_stops,
  int gap
)
{
  // The ops are:
  /**
   * 1) is rank less than the next rank? fill in an array with the result
   * 2) do the per-chunk scans using this info
   * 3) reorder "sa scan" based on the scan
   * 4) compute ranks -- this is going to be easier
  */
  const int ix_chunk = blockIdx.x;
  const uint32_t chunk_start = device_starts[ix_chunk];
  const uint32_t chunk_size = device_stops[ix_chunk] - chunk_start;

  rank_t<offset_type> *chunk_ranks = &ranks[chunk_start];

  uint32_t *chunk_scan = &device_scan[chunk_start];
  offset_type *chunk_sa = &current_sa[chunk_start];

  constexpr int NUM_WARPS_PER_CTA = cta_thread_count / WARP_SIZE;
  const int ix_warp = threadIdx.x / WARP_SIZE_U;
  const int ix_warp_thread = threadIdx.x % WARP_SIZE_U;
  assert(ix_warp < NUM_WARPS_PER_CTA);

  __shared__ uint8_t iter_res[NUM_WARPS_PER_CTA];
  __shared__ uint32_t global_offset;
  const bool global_offset_thread = threadIdx.x == (NUM_WARPS_PER_CTA - 1) * WARP_SIZE;
  if (threadIdx.x == global_offset_thread)
  {
    global_offset = 0;
  }

  for (uint32_t ix_base = 0; ix_base < chunk_size; ix_base += blockDim.x)
  {

    uint32_t iter_warp_inc = 0;
    uint32_t ix_byte = ix_base + threadIdx.x;
    uint32_t sub_scan_val = ix_byte + 1 < chunk_size ? chunk_ranks[ix_byte].rank < chunk_ranks[ix_byte + 1].rank : 0;

    unsigned iter_mask = __ballot_sync(WARP_ALL, sub_scan_val);
    if (ix_warp_thread == 0)
    {
      iter_warp_inc = __popc(iter_mask);
      iter_res[ix_warp] = iter_warp_inc;
    }

    // Get my own output value. This is the number of values leading up to myself
    unsigned sub_mask = iter_mask & ((1 << ix_warp_thread) - 1);
    uint32_t warp_inc = __popc(sub_mask);

    offset_type this_current_sa;
    if (ix_byte < chunk_size)
    {
      this_current_sa = chunk_sa[ix_byte];
    }

    __syncthreads();

    // Need an ix_warp-way sum to get the local offset.
    // Use shuffles to achieve this.
    // Global offset can come from the 2 syncthreads we already have
    unsigned offset_val = 0; // default value for coverity, real one is computed in the if below
    if (ix_warp_thread < ix_warp)
    {
      unsigned shfl_mask = (1 << ix_warp) - 1;
      offset_val = iter_res[ix_warp_thread];
      for (int ix = 1; ix < ix_warp; ix <<= 1)
      {
        unsigned inc_val = __shfl_up_sync(shfl_mask, offset_val, ix);
        if (ix_warp_thread >= ix)
        {
          offset_val += inc_val;
        }
      }
    }

    uint32_t local_offset = global_offset;
    if (ix_warp > 0)
    {
      local_offset += __shfl_sync(WARP_ALL, offset_val, ix_warp - 1);
    }

    if (ix_byte < chunk_size)
    {
      chunk_scan[ix_byte] = local_offset + warp_inc;

      chunk_ranks[ix_byte].rank_pair[1] = local_offset + warp_inc;
      chunk_ranks[ix_byte].rank_pair[0] =
        this_current_sa; // avoid an extra memory load later -- can share L1 cache load time later
    }

    __syncthreads();

    if (global_offset_thread)
    {
      global_offset = local_offset + iter_warp_inc;
    }
  }
}

template <typename offset_type>
inline __device__ void init_ranks(
  U32 chunk,
  U32 idx,
  U32 chunk_start,
  U32 chunk_size,
  offset_type *sa,
  const U8 *const *data_ptrs,
  rank_t<offset_type> *ranks
)
{
  // Init SA
  sa[chunk_start + idx] = idx;
  // Init Ranks
  auto S = data_ptrs[chunk];
  rank_t<offset_type> rank;
  constexpr size_t rank_size = 2 * sizeof(offset_type);
  rank.rank_quad[rank_size - 1] = idx < chunk_size ? S[idx] : 0;
  rank.rank_quad[rank_size - 2] = idx + 1 < chunk_size ? S[idx + 1] : 0;
  rank.rank_quad[rank_size - 3] = idx + 2 < chunk_size ? S[idx + 2] : 0;
  rank.rank_quad[rank_size - 4] = idx + 3 < chunk_size ? S[idx + 3] : 0;
  if constexpr (std::is_same_v<offset_type, U32>)
  {
    rank.rank_quad[3] = idx + 4 < chunk_size ? S[idx + 4] : 0;
    rank.rank_quad[2] = idx + 5 < chunk_size ? S[idx + 5] : 0;
    rank.rank_quad[1] = idx + 6 < chunk_size ? S[idx + 6] : 0;
    rank.rank_quad[0] = idx + 7 < chunk_size ? S[idx + 7] : 0;
  }

  ranks[chunk_start + idx] = rank;
}

template <typename offset_type>
inline __device__ void compute_ranks(
  U32 chunk,
  U32 idx,
  U32 chunk_start,
  U32 chunk_size,
  offset_type *sa,
  offset_type *sa_scan,
  rank_t<offset_type> *ranks,
  int gap
)
{
  auto suffix_id = ranks[chunk_start + idx].rank_pair[0];
  ranks[chunk_start + idx].rank_pair[0] = suffix_id + gap < chunk_size ? sa_scan[chunk_start + suffix_id + gap] : 0;
}

/**
 * @brief Compute the Longest Common Prefix between consecutive suffixes
 * @param chunk The chunk we are in
 * @param idx The index within the chunk
 * @param chunk_start The starting offset of the chunk in the contiguous buffer
 * @param chunk_size The size of the chunk
 * @param sa The suffix arrays for each chunk in a contiguous buffer
 * @param data_ptrs The data for each chunk
 */
template <int maxMatchLength, typename LCP_t, typename offset_type>
inline __device__ LCP_t
compute_lcp(U32 chunk, U32 idx, U32 chunk_start, U32 chunk_size, offset_type *sa, const U8 *const *data_ptrs)
{
  LCP_t len = 0;
  if (idx + 1 < chunk_size)
  {
    const U8 *S = data_ptrs[chunk];
    offset_type suf_a = sa[chunk_start + idx];
    offset_type suf_b = sa[chunk_start + idx + 1];
    for (; len < min(maxMatchLength, chunk_size); ++len)
    {
      if (suf_a + len >= chunk_size or suf_b + len >= chunk_size or S[suf_a + len] != S[suf_b + len])
      {
        break;
      }
    }
  }
  return len;
}

template <typename offset_type>
size_t getSegmentedSortTempStorageSpace(size_t max_chunk_size, size_t batch_size)
{
  size_t temp_bytes = 0;
  typedef U32 *OffsetTypePtr;
  nvcomp::cub::DoubleBuffer<U32> keys(nullptr, nullptr);
  nvcomp::cub::DoubleBuffer<offset_type> values(nullptr, nullptr);

  // TODO: pre cuda 13 cub only supports int, newer versions support int64_t
  // should we add #if CUDART_VERSION >= 13000 and check int64t in that case?
  nvcomp::cub::DeviceSegmentedRadixSort::SortPairs(
    NULL,
    temp_bytes,
    keys,
    values,
    narrow_cast<int>(max_chunk_size * batch_size),
    narrow_cast<int>(batch_size),
    OffsetTypePtr{},
    OffsetTypePtr{}
  );
  return temp_bytes;
}

template <typename offset_type>
size_t saGetTempStorageSize(size_t max_chunk_size, size_t batch_size)
{
  const size_t total_max_bytes = max_chunk_size * batch_size;
  if (total_max_bytes > static_cast<size_t>(std::numeric_limits<int>::max()))
  {
    // Note:
    // currently DeviceSegmentedRadixSort::SortPairs(...) does not support
    // sorting beyond INT_MAX items. One workaround could be to split up the
    // original data into multiple parts (maximum INT_MAX items per part),
    // and shift the data pointers (`db_ranks`, `db_sa`) and the offsets
    // (`device_start` and `device_stops`) offsets, respectively.
    // These multiple parts could be sorted serially while respecting the maximum
    // item limit.
    // Note update: since CUDA 13 SortPairs uses int64_t parameters,
    // should we add #if CUDART_VERSION >= 13000 and check int64t in that case?
    std::string err_msg = "Too many items to sort: " + std::to_string(total_max_bytes) +
                          ", max sortable: " + std::to_string(std::numeric_limits<int>::max());
    std::cerr << err_msg << std::endl;
    throw NVCompException(nvcompErrorNotSupported, err_msg);
  }

  size_t temp_bytes = 0;

  // Temp for sort
  temp_bytes += roundUpTo(getSegmentedSortTempStorageSpace<offset_type>(max_chunk_size, batch_size), sizeof(void *));

  // Double buffered SA scratch -- a double buffer in scratch allows array of structs
  // rather than struct of arrays for the resulting tuples (sa, inv sa, lcp)
  temp_bytes += roundUpTo(total_max_bytes * sizeof(offset_type), sizeof(void *)) * 2;

  // Double buffered ranks
  temp_bytes += roundUpTo(total_max_bytes * sizeof(rank_t<offset_type>), sizeof(void *)) * 2;

  temp_bytes += roundUpTo(batch_size * sizeof(size_t) * 2, sizeof(void *)); // device starts, device stops
  return temp_bytes;
}

__global__ void fill_output_indices(
  const size_t *decomp_data_sizes,
  size_t *device_starts,
  size_t *device_stops,
  const size_t max_chunk_size,
  const size_t num_chunks
)
{
  const int ix_chunk = blockIdx.x * blockDim.x + threadIdx.x;
  if (ix_chunk >= num_chunks)
  {
    return;
  }
  device_starts[ix_chunk] = max_chunk_size * ix_chunk;
  device_stops[ix_chunk] = device_starts[ix_chunk] + decomp_data_sizes[ix_chunk];
}

template <int maxMatchLength, typename offset_type>
nvcomp::cub::DoubleBuffer<offset_type> saBuildImpl(
  U8 *device_temp,
  const U8 *const *device_data_ptrs,
  const size_t *device_data_sizes,
  size_t max_chunk_size,
  size_t batch_size,
  size_t *device_starts,
  size_t *device_stops,
  size_t used_temp_space,
  cudaStream_t stream
)
{
  size_t total_max_items = max_chunk_size * batch_size;
  if (total_max_items > static_cast<size_t>(std::numeric_limits<int>::max()))
  {
    // Note:
    // currently DeviceSegmentedRadixSort::SortPairs(...) does not support
    // sorting beyond INT_MAX items. One workaround could be to split up the
    // original data into multiple parts (maximum INT_MAX items per part),
    // and shift the data pointers (`db_ranks`, `db_sa`) and the offsets
    // (`device_start` and `device_stops`) offsets, respectively.
    // These multiple parts could be sorted serially while respecting the maximum
    // item limit.
    // Note update: since CUDA 13 SortPairs uses int64_t parameters,
    // should we add #if CUDART_VERSION >= 13000 and check int64t in that case?
    std::string err_msg = "Too many items to sort: " + std::to_string(total_max_items) +
                          ", max sortable: " + std::to_string(std::numeric_limits<int>::max());
    std::cerr << err_msg << std::endl;
    throw NVCompException(nvcompErrorNotSupported, err_msg);
  }

  size_t sort_temp_bytes = getSegmentedSortTempStorageSpace<offset_type>(max_chunk_size, batch_size);

  // This should come at the beginning, as we've already computed the offsets
  auto sort_temp = reinterpret_cast<U32 *>(device_temp + used_temp_space);
  used_temp_space += roundUpTo(sort_temp_bytes, sizeof(void *));

  auto device_sa_alt1 = reinterpret_cast<offset_type *>(device_temp + used_temp_space);
  used_temp_space += roundUpTo(total_max_items * sizeof(offset_type), sizeof(void *));

  auto device_sa_alt2 = reinterpret_cast<offset_type *>(device_temp + used_temp_space);
  used_temp_space += roundUpTo(total_max_items * sizeof(offset_type), sizeof(void *));

  auto device_ranks = reinterpret_cast<rank_t<offset_type> *>(device_temp + used_temp_space);
  used_temp_space += roundUpTo(total_max_items * sizeof(rank_t<offset_type>), sizeof(void *));

  auto device_ranks_alt = reinterpret_cast<rank_t<offset_type> *>(device_temp + used_temp_space);
  used_temp_space += roundUpTo(total_max_items * sizeof(rank_t<offset_type>), sizeof(void *));

  assert(used_temp_space <= saGetTempStorageSize<offset_type>(max_chunk_size, batch_size));

  // compute chunk offsets
  // gdeflate
  CUDA_CHECK(cudaMemsetAsync(device_starts, 0x0, sizeof(size_t), stream));
  nvcomp::thrust::inclusive_scan(
    nvcomp::thrust::cuda::par_nosync.on(stream),
    device_data_sizes,
    device_data_sizes + batch_size,
    device_stops
  );
  CUDA_CHECK(
    cudaMemcpyAsync(device_starts + 1, device_stops, sizeof(size_t) * (batch_size - 1), cudaMemcpyDeviceToDevice, stream)
  );

  nvcomp::cub::DoubleBuffer<offset_type> db_sa(device_sa_alt1, device_sa_alt2);
  using rank_int_type = typename rank_t<offset_type>::type;
  nvcomp::cub::DoubleBuffer<rank_int_type> db_ranks(
    reinterpret_cast<rank_int_type *>(device_ranks),
    reinterpret_cast<rank_int_type *>(device_ranks_alt)
  );

  constexpr int initial_gap = sizeof(offset_type);
  static_assert(maxMatchLength <= std::numeric_limits<int>::max() / 2);
  int loop_limit = maxMatchLength * 2;
  if (max_chunk_size < loop_limit)
  {
    loop_limit = static_cast<int>(max_chunk_size);
  }

  // At least one iteration is required to initialize SA
  // and run the sort (e.g., when max_chunk_size == 1).
  // Note: There may be faster alternatives, but this is a simple
  //       solution for cases when `max_chunk_Size` is actually
  //       less than or equal to `initial_gap`, i.e.,
  //       when `max_chunk_size` = {1, 2, 3, 4}.
  if (loop_limit <= initial_gap)
  {
    loop_limit = initial_gap + 1;
  }

  for (int gap = initial_gap; gap < loop_limit; gap *= 2)
  {
    // Sorting is not done in place so we have to double buffer
    offset_type *current_sa = db_sa.Current();
    rank_t<offset_type> *device_ranks = reinterpret_cast<rank_t<offset_type> *>(db_ranks.Current());
    U32 *device_scan = reinterpret_cast<U32 *>(db_ranks.Alternate()); // Reuse memory
    offset_type *device_sa_scan = db_sa.Alternate(); // Reuse memory

    // Compute Ranks
    if (gap == initial_gap)
    {

      for_each_chunk(
        stream,
        device_starts,
        device_stops,
        batch_size,
        max_chunk_size,
        [=] __device__(U32 chunk, U32 idx, U32 chunk_start, U32 chunk_size) {
          init_ranks(chunk, idx, chunk_start, chunk_size, current_sa, device_data_ptrs, device_ranks);
        }
      );
    }
    else
    {
      optimized_rank_iter_kernel<<<cuda_dim_cast(batch_size), cta_thread_count, 0, stream>>>(
        current_sa,
        device_scan,
        device_sa_scan,
        device_ranks,
        device_starts,
        device_stops,
        gap
      );
      CUDA_CHECK(cudaGetLastError());

      for_each_chunk(
        stream,
        device_starts,
        device_stops,
        batch_size,
        max_chunk_size,
        [=] __device__(U32 chunk, U32 idx, U32 chunk_start, U32 chunk_size) {
          device_sa_scan[chunk_start + current_sa[chunk_start + idx]] = device_scan[chunk_start + idx] -
                                                                        device_scan[chunk_start];
        }
      );

      // Compute Ranks using previous ranks and ranks for new prefix
      for_each_chunk(
        stream,
        device_starts,
        device_stops,
        batch_size,
        max_chunk_size,
        [=] __device__(U32 chunk, U32 idx, U32 chunk_start, U32 chunk_size) {
          compute_ranks(chunk, idx, chunk_start, chunk_size, current_sa, device_sa_scan, device_ranks, gap);
        }
      );
    }

    // pre cuda 13 cub only supports int, newer versions support int64_t
    // should we add #if CUDART_VERSION >= 13000 and check int64t in that case?
    assert(total_max_items <= std::numeric_limits<int>::max());
    assert(batch_size <= std::numeric_limits<int>::max());
    nvcomp::cub::DeviceSegmentedRadixSort::SortPairs(
      sort_temp,
      sort_temp_bytes,
      db_ranks,
      db_sa,
      static_cast<int>(total_max_items),
      static_cast<int>(batch_size),
      device_starts,
      device_stops,
      0,
      sizeof(rank_int_type) * 8,
      stream
    );
  }

  return db_sa;
}

template <int saMaxMatchLength, typename LCP_t, typename offset_type>
void saBuildSubBatch(
  U8 *device_temp,
  const U8 *const *device_data_ptrs,
  const size_t *device_data_sizes,
  size_t max_chunk_size,
  size_t batch_size,
  offset_type **device_sa_out,
  offset_type **device_inv_sa_out,
  LCP_t **device_lcp_out,
  cudaStream_t stream
)
{
  if (max_chunk_size == 0)
  {
    return;
  }

  size_t used_temp_space = 0;

  size_t *device_starts = reinterpret_cast<size_t *>(device_temp + used_temp_space);
  used_temp_space += roundUpTo(batch_size * sizeof(size_t), sizeof(void *));

  size_t *device_stops = reinterpret_cast<size_t *>(device_temp + used_temp_space);
  used_temp_space += roundUpTo(batch_size * sizeof(size_t), sizeof(void *));

  auto db_sa = saBuildImpl<saMaxMatchLength, offset_type>(
    device_temp,
    device_data_ptrs,
    device_data_sizes,
    max_chunk_size,
    batch_size,
    device_starts,
    device_stops,
    used_temp_space,
    stream
  );

  auto result_sa = db_sa.Current();

  // Compute LCP
  for_each_chunk(
    stream,
    device_starts,
    device_stops,
    batch_size,
    max_chunk_size,
    [=] __device__(U32 chunk, U32 idx, U32 chunk_start, U32 chunk_size) {
      LCP_t len = compute_lcp<saMaxMatchLength, LCP_t, offset_type>(
        chunk,
        idx,
        chunk_start,
        chunk_size,
        result_sa,
        device_data_ptrs
      );
      auto sa_offset = result_sa[chunk_start + idx];
      assert(sa_offset < max_chunk_size);

      device_sa_out[chunk][idx] = sa_offset;
      device_lcp_out[chunk][idx] = len;
      device_inv_sa_out[chunk][sa_offset] = idx;
    }
  );
}

template <int saMaxMatchLength, typename LCP_t, typename offset_type>
void saBuild(
  U8 *device_temp,
  const U8 *const *device_data_ptrs,
  const size_t *device_data_sizes,
  size_t max_chunk_size,
  size_t batch_size,
  offset_type **device_sa_out,
  offset_type **device_inv_sa_out,
  LCP_t **device_lcp_out,
  cudaStream_t stream
)
{
  // Making sure that the total element count remains below INT_MAX
  // as currently DeviceSegmentedRadixSort(...) does not support total element counts
  // above INT_MAX
  size_t sub_batch_max_size = batch_size;
  if (max_chunk_size * batch_size > static_cast<size_t>(std::numeric_limits<int>::max()))
  {
    // Sub-batching is necessary
    // Note: Trying to schedule as many chunks as possible into a sub-batch,
    //       minimizing the total iteration count needed
    sub_batch_max_size = static_cast<size_t>(std::numeric_limits<int>::max()) / max_chunk_size;
  }

  size_t sub_batch_count = roundUpDiv(batch_size, sub_batch_max_size);
  for (size_t sub_batch_ix = 0; sub_batch_ix < sub_batch_count; ++sub_batch_ix)
  {
    size_t sub_batch_offset = sub_batch_ix * sub_batch_max_size;
    size_t sub_batch_size = std::min(batch_size - sub_batch_offset, sub_batch_max_size);

    saBuildSubBatch<saMaxMatchLength, LCP_t>(
      device_temp,
      device_data_ptrs + sub_batch_offset,
      device_data_sizes + sub_batch_offset,
      max_chunk_size,
      sub_batch_size,
      device_sa_out + sub_batch_offset,
      device_inv_sa_out + sub_batch_offset,
      device_lcp_out + sub_batch_offset,
      stream
    );
  }
}

// Explicit instantiations
template size_t saGetTempStorageSize<U16>(size_t max_chunk_size, size_t batch_size);
template size_t saGetTempStorageSize<U32>(size_t max_chunk_size, size_t batch_size);

template void saBuild<gdeflate::deflateMaxMatchLength, U16, U16>(
  U8 *device_temp,
  const U8 *const *device_data_ptrs,
  const size_t *device_data_sizes,
  size_t max_chunk_size,
  size_t batch_size,
  U16 **device_sa_out,
  U16 **device_inv_sa_out,
  U16 **device_lcp_out,
  cudaStream_t stream
);
template void saBuild<gdeflate::deflateL6MaxMatchLength, U8, U16>(
  U8 *device_temp,
  const U8 *const *device_data_ptrs,
  const size_t *device_data_sizes,
  size_t max_chunk_size,
  size_t batch_size,
  U16 **device_sa_out,
  U16 **device_inv_sa_out,
  U8 **device_lcp_out,
  cudaStream_t stream
);

template void saBuild<gdeflate::deflateMaxMatchLength, U16, U32>(
  U8 *device_temp,
  const U8 *const *device_data_ptrs,
  const size_t *device_data_sizes,
  size_t max_chunk_size,
  size_t batch_size,
  U32 **device_sa_out,
  U32 **device_inv_sa_out,
  U16 **device_lcp_out,
  cudaStream_t stream
);
template void saBuild<gdeflate::deflateL6MaxMatchLength, U8, U32>(
  U8 *device_temp,
  const U8 *const *device_data_ptrs,
  const size_t *device_data_sizes,
  size_t max_chunk_size,
  size_t batch_size,
  U32 **device_sa_out,
  U32 **device_inv_sa_out,
  U8 **device_lcp_out,
  cudaStream_t stream
);

} // namespace sa
