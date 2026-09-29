#include <cub/cub.cuh>

#include "common.h"
#include "exception.hpp"
#include "LZ77_decomp.cuh"
#include "ManagerUtils.hpp"
#include "nvcomp/version.h"
#include "nvcomp_common_deps/hlif_shared.hpp"
#include "Reduction.cuh"

#include <assert.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

namespace nvcomp
{

inline __host__ __device__ uintptr_t round_up_align_address(uintptr_t address, size_t alignment)
{
  return (address + (alignment - 1)) & ~(alignment - 1);
}

__global__ void setup_decomp_llif_buffers(
  const CommonHeader *common_header,
  uint8_t *decomp_buffer,
  const uint8_t *comp_buffer,
  uint8_t **uncomp_buffers,
  const uint8_t **comp_buffers,
  const size_t *comp_offsets,
  size_t *uncomp_sizes,
  const size_t *provided_uncomp_chunk_offsets,
  const size_t *provided_uncomp_sizes,
  const bool include_uncomp_offsets_and_sizes
)
{
  const size_t num_chunks = common_header->num_chunks;
  const size_t chunk_id = threadIdx.x + blockDim.x * blockIdx.x;
  if (chunk_id >= num_chunks)
  {
    return;
  }

  comp_buffers[chunk_id] = comp_buffer + comp_offsets[chunk_id];

  if (include_uncomp_offsets_and_sizes)
  {
    // uncomp sizes are provided
    uncomp_buffers[chunk_id] = decomp_buffer + provided_uncomp_chunk_offsets[chunk_id];
    uncomp_sizes[chunk_id] = provided_uncomp_sizes[chunk_id];
  }
  else
  {
    uncomp_buffers[chunk_id] = decomp_buffer + chunk_id * common_header->uncomp_chunk_size;

    uncomp_sizes[chunk_id] = chunk_id == num_chunks - 1
                               ? common_header->decomp_data_size - chunk_id * common_header->uncomp_chunk_size
                               : common_header->uncomp_chunk_size;
  }
}

__global__ void setup_batched_decomp_llif_buffers(
  size_t *batch_chunks_exclusive_sum,
  uint8_t **device_input_comp_buffers,
  uint8_t **device_output_decomp_buffers,
  int header_size,
  uint8_t **uncomp_chunk_buffers,
  const uint8_t **comp_chunk_buffers,
  size_t *uncomp_sizes,
  size_t *comp_chunk_sizes,
  size_t min_alignment
)
{
  size_t batch_id = blockIdx.y;
  size_t this_batch_chunk_id = blockDim.x * blockIdx.x + threadIdx.x;

  const uint8_t *this_batch_input_comp_buffer = device_input_comp_buffers[batch_id];
  uint8_t *this_batch_output_decomp_buffer = device_output_decomp_buffers[batch_id];

  CommonHeader *this_batch_common_header = reinterpret_cast<CommonHeader *>(device_input_comp_buffers[batch_id]);
  this_batch_input_comp_buffer += header_size;

  const size_t batch_num_chunks = this_batch_common_header->num_chunks;

  if (this_batch_chunk_id >= batch_num_chunks)
  {
    return;
  }

  uintptr_t aligned_address =
    round_up_align_address(reinterpret_cast<uintptr_t>(this_batch_input_comp_buffer), alignof(size_t));
  const size_t *this_batch_comp_chunk_offsets = reinterpret_cast<size_t *>(aligned_address);
  const size_t *this_batch_comp_sizes = this_batch_comp_chunk_offsets + batch_num_chunks;

  this_batch_input_comp_buffer = reinterpret_cast<const uint8_t *>(
    round_up_align_address(reinterpret_cast<uintptr_t>(this_batch_comp_sizes + batch_num_chunks), min_alignment)
  );

  const size_t global_chunk_id = batch_chunks_exclusive_sum[batch_id] + this_batch_chunk_id;

  //Prepare pointers for (reading) compressed chunks, and (writing) uncompressed chunks
  comp_chunk_buffers[global_chunk_id] = this_batch_input_comp_buffer +
                                        this_batch_comp_chunk_offsets[this_batch_chunk_id];
  uncomp_chunk_buffers[global_chunk_id] = this_batch_output_decomp_buffer +
                                          (this_batch_chunk_id * this_batch_common_header->uncomp_chunk_size);

  //Store Pointers to chunk sizes in shared memory
  comp_chunk_sizes[global_chunk_id] = this_batch_comp_sizes[this_batch_chunk_id];

  uncomp_sizes[global_chunk_id] = this_batch_chunk_id == batch_num_chunks - 1
                                    ? this_batch_common_header->decomp_data_size -
                                        (this_batch_chunk_id * this_batch_common_header->uncomp_chunk_size)
                                    : this_batch_common_header->uncomp_chunk_size;
}

void setup_batched_decomp_llif_buffers_host(
  uint8_t **device_input_comp_buffers,
  const uint8_t *const *host_input_comp_buffers,
  uint8_t **output_decomp_buffers,
  int header_size,
  uint8_t **uncomp_chunk_buffers,
  const uint8_t **comp_chunk_buffers,
  const uint8_t **host_comp_chunk_buffers,
  size_t *uncomp_sizes,
  size_t *comp_chunk_sizes,
  size_t batch_count,
  size_t min_alignment
)
{
  size_t running_offset = 0;
  for (size_t batch_id = 0; batch_id < batch_count; ++batch_id)
  {
    uint8_t *output_decomp_buffer = output_decomp_buffers[batch_id];
    const CommonHeader *const common_header = reinterpret_cast<const CommonHeader *>(host_input_comp_buffers[batch_id]);
    const size_t batch_num_chunks = common_header->num_chunks;
    const size_t *const device_comp_chunk_offsets =
      roundUpToAlignment<size_t>(device_input_comp_buffers[batch_id] + header_size);

    const size_t *const host_comp_chunk_offsets =
      roundUpToAlignment<const size_t>(host_input_comp_buffers[batch_id] + header_size);
    const size_t *const host_comp_sizes = host_comp_chunk_offsets + batch_num_chunks;

    // skip over offsets and sizes, then align the compressed-data region to
    // min_alignment to match the compress side (no-op when min_alignment == 8).
    const uint8_t *device_input_comp_buffer = reinterpret_cast<const uint8_t *>(round_up_align_address(
      reinterpret_cast<uintptr_t>(device_comp_chunk_offsets + 2 * batch_num_chunks),
      min_alignment
    ));
    const uint8_t *host_input_comp_buffer = reinterpret_cast<const uint8_t *>(
      round_up_align_address(reinterpret_cast<uintptr_t>(host_comp_chunk_offsets + 2 * batch_num_chunks), min_alignment)
    );

    size_t global_chunk_id = 0;
    for (size_t chunk_id = 0; chunk_id < batch_num_chunks; ++chunk_id)
    {
      global_chunk_id = running_offset + chunk_id;

      comp_chunk_buffers[global_chunk_id] = device_input_comp_buffer + host_comp_chunk_offsets[chunk_id];
      host_comp_chunk_buffers[global_chunk_id] = host_input_comp_buffer + host_comp_chunk_offsets[chunk_id];
      uncomp_chunk_buffers[global_chunk_id] = output_decomp_buffer + (chunk_id * common_header->uncomp_chunk_size);
      comp_chunk_sizes[global_chunk_id] = host_comp_sizes[chunk_id];
      uncomp_sizes[global_chunk_id] = common_header->uncomp_chunk_size;
    }
    //For the last chunk, we need to set the uncomp_size to the remaining data size
    uncomp_sizes[global_chunk_id] = common_header->decomp_data_size -
                                    ((batch_num_chunks - 1) * common_header->uncomp_chunk_size);

    running_offset += batch_num_chunks;
  }
}

__global__ void
round_up_alignment_kernel(const size_t *in_array, size_t *out_array, const size_t n_elems, const size_t alignment)
{
  const size_t ix_thread = threadIdx.x + blockDim.x * blockIdx.x;
  if (ix_thread >= n_elems)
  {
    return;
  }
  out_array[ix_thread] = roundUpTo(in_array[ix_thread], alignment);
}

__global__ void compact_comp_buffers_and_header_output(
  uint8_t *__restrict__ comp_buffer,
  const uint8_t *const *__restrict__ comp_buffers,
  const size_t *__restrict__ comp_sizes,
  CommonHeader *__restrict__ header,
  size_t *__restrict__ chunk_offset_buffer,
  const size_t num_chunks,
  const size_t decomp_buffer_size,
  const size_t uncomp_chunk_size,
  const nvcompFormatType_t format_type,
  size_t *__restrict__ comp_size,
  const size_t alignment
)
{
  const size_t ix_chunk = blockIdx.x + (num_chunks == 1 ? 0 : 1);
  assert(ix_chunk < num_chunks);

  const size_t local_block_idx = blockIdx.y;
  const size_t chunk_size = comp_sizes[ix_chunk];

  if (ix_chunk > 0)
  {
    const uint64_t *this_input = reinterpret_cast<const uint64_t *>(comp_buffers[ix_chunk]);
    const size_t chunk_offset = chunk_offset_buffer[ix_chunk];

    // Note: chunk_offset_buffer contains the exclusive-sum-scanned offsets
    uint64_t *output = reinterpret_cast<uint64_t *>(comp_buffer + chunk_offset);

    // Now everything is 8-byte aligned
    assert(alignment >= 8);
    const size_t dword_count = chunk_size / 8; // for the entire gridDim.y
    const size_t rem = chunk_size % 8;

    // Divide the work among the threads in the block
    const size_t dwords_per_block = (dword_count + gridDim.y - 1) / gridDim.y;
    const size_t start_dword = local_block_idx * dwords_per_block;
    const size_t end_dword = min(start_dword + dwords_per_block, dword_count);

    // Copy complete dword-s
    for (size_t ix_dword = start_dword + threadIdx.x; ix_dword < end_dword; ix_dword += blockDim.x)
    {
      output[ix_dword] = this_input[ix_dword];
    }

    // Note:
    // last bytes that do not form a complete dword are copied by the last local block
    // in the chunk
    if (local_block_idx == (gridDim.y - 1))
    {
      const uint8_t *this_input_byte = reinterpret_cast<const uint8_t *>(this_input + end_dword);
      uint8_t *output_byte = reinterpret_cast<uint8_t *>(output + end_dword);
      if (threadIdx.x < rem)
      {
        output_byte[threadIdx.x] = this_input_byte[threadIdx.x];
      }
    }

    // Note:
    // Using the first local block to fill up the alignment bytes with zeros.
    // Note, however, that it might happen that there is a single local block / chunk.
    if (local_block_idx == 0)
    {
      const size_t previous_chunk_size = comp_sizes[ix_chunk - 1];
      const size_t previous_rem = roundUpTo(previous_chunk_size, alignment) - previous_chunk_size;
      if (threadIdx.x < previous_rem)
      {
        comp_buffer[chunk_offset - previous_rem + threadIdx.x] = 0x00;
      }
    }
  }

  // Additionally fill out the required header values.
  if ((threadIdx.x == 0) and (local_block_idx == 0))
  {
    if (ix_chunk == num_chunks - 1)
    {
      init_common_header(*header);
      header->format = format_type;
      header->decomp_data_size = decomp_buffer_size;
      header->num_chunks = num_chunks;
      header->include_chunk_starts = true;
      header->full_comp_buffer_checksum = false;
      header->decomp_buffer_checksum = false;
      header->include_per_chunk_comp_buffer_checksums = false;
      header->include_per_chunk_decomp_buffer_checksums = false;
      header->uncomp_chunk_size = uncomp_chunk_size;
      header->comp_data_offset = (uintptr_t)comp_buffer - (uintptr_t)header;
      header->comp_data_size = chunk_offset_buffer[ix_chunk] + chunk_size;
      if (comp_size != nullptr)
      {
        *comp_size = header->comp_data_size + header->comp_data_offset;
      }
    }
  }
}

__global__ void batched_compact_comp_buffers_and_header_output(
  size_t batch_count,
  size_t total_num_chunks,
  uint8_t **output_comp_buffers,
  CompressionConfig *compression_configs,
  const uint8_t *const *comp_chunk_buffers,
  const size_t *comp_sizes,
  size_t *chunk_offset_buffer,
  const size_t uncomp_chunk_size,
  const nvcompFormatType_t format_type,
  size_t *comp_size,
  int header_size,
  size_t min_alignment
)
{
  const size_t global_chunk_id = blockIdx.x;

  if (global_chunk_id >= total_num_chunks)
  {
    return;
  }

  //Determine the global chunk this thread belongs to, and it's corresponding batch
  int batch_idx = 0;
  size_t this_batch_chunk_id = 0;
  size_t this_batch_num_chunks = 0;
  size_t batch_global_chunk_offset = 0;

  for (int i = 0; i < batch_count; ++i)
  {
    this_batch_chunk_id = global_chunk_id - batch_global_chunk_offset;
    this_batch_num_chunks = compression_configs[i].num_chunks;
    batch_global_chunk_offset += this_batch_num_chunks;
    if (global_chunk_id < batch_global_chunk_offset)
    {
      batch_idx = i;
      break;
    }
  }
  const size_t local_block_idx = blockIdx.y;

  const size_t uncomp_buffer_size = compression_configs[batch_idx].uncompressed_buffer_size;

  uint8_t *this_batch_output_comp_buffer = output_comp_buffers[batch_idx];

  //Prepare Comp Buffer for each batch
  CommonHeader *this_batch_common_header = reinterpret_cast<CommonHeader *>(this_batch_output_comp_buffer);

  this_batch_output_comp_buffer += header_size;

  size_t *this_batch_output_comp_chunk_offsets = roundUpToAlignment<size_t>(this_batch_output_comp_buffer);
  size_t *this_batch_output_comp_sizes = this_batch_output_comp_chunk_offsets + this_batch_num_chunks;
  this_batch_output_comp_buffer = reinterpret_cast<uint8_t *>(this_batch_output_comp_sizes + this_batch_num_chunks);
  // Align the compressed-data region to the format's strongest required alignment
  // (chunks are compacted here at min_alignment-aligned offsets). Must match the
  // single-buffer and decompress paths. No-op when min_alignment == sizeof(size_t).
  this_batch_output_comp_buffer = reinterpret_cast<uint8_t *>(
    round_up_align_address(reinterpret_cast<uintptr_t>(this_batch_output_comp_buffer), min_alignment)
  );

  const size_t chunk_size = comp_sizes[global_chunk_id];
  size_t this_chunk_offset = chunk_offset_buffer[global_chunk_id];

  this_chunk_offset -= (chunk_offset_buffer[(batch_global_chunk_offset - this_batch_num_chunks)]);

  if (this_batch_chunk_id > 0)
  {
    const uint64_t *this_input = reinterpret_cast<const uint64_t *>(comp_chunk_buffers[global_chunk_id]);
    uint64_t *output = reinterpret_cast<uint64_t *>(this_batch_output_comp_buffer + this_chunk_offset);

    // Now everything is 8-byte aligned
    const size_t dword_count = (chunk_size + 7) / 8;

    // Divide the work among the threads in the block
    const size_t dwords_per_block = (dword_count + gridDim.y - 1) / gridDim.y;
    const size_t start_dword = local_block_idx * dwords_per_block;
    const size_t end_dword = min(start_dword + dwords_per_block, dword_count);

    for (size_t ix_dword = start_dword + threadIdx.x; ix_dword < end_dword; ix_dword += blockDim.x)
    {
      output[ix_dword] = this_input[ix_dword];
    }
  }

  if ((threadIdx.x == 0) and (local_block_idx == 0))
  {
    auto &chunk_offset = this_batch_output_comp_chunk_offsets[this_batch_chunk_id];
    auto &chunk_comp_size = this_batch_output_comp_sizes[this_batch_chunk_id];

    chunk_offset = this_chunk_offset;
    chunk_comp_size = comp_sizes[global_chunk_id];

    // Additionally fill out the required header values.
    if ((this_batch_chunk_id == (this_batch_num_chunks - 1)))
    {
      init_common_header(*this_batch_common_header);
      this_batch_common_header->format = format_type;
      this_batch_common_header->decomp_data_size = uncomp_buffer_size;
      this_batch_common_header->num_chunks = this_batch_num_chunks;
      this_batch_common_header->include_chunk_starts = true;
      this_batch_common_header->full_comp_buffer_checksum = false;
      this_batch_common_header->decomp_buffer_checksum = false;
      this_batch_common_header->include_per_chunk_comp_buffer_checksums = false;
      this_batch_common_header->include_per_chunk_decomp_buffer_checksums = false;
      this_batch_common_header->uncomp_chunk_size = uncomp_chunk_size;
      this_batch_common_header->comp_data_offset = (uintptr_t)this_batch_output_comp_buffer -
                                                   (uintptr_t)this_batch_common_header;
      this_batch_common_header->comp_data_size = this_chunk_offset + chunk_size;
      if (comp_size != nullptr)
      {
        comp_size[batch_idx] = this_batch_common_header->comp_data_size + this_batch_common_header->comp_data_offset;
      }
    }
  }
}

template <unsigned int block_size>
__global__ void max_reduce_device_status_kernel(int *statuses, int *max_status, const unsigned int num_chunks)
{
  using BlockReduce = cub::BlockReduce<int, block_size>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  auto chunk_id = blockIdx.x * blockDim.x + threadIdx.x;
  int val = chunk_id < num_chunks ? statuses[chunk_id] : static_cast<int>(nvcompSuccess);

  // Compute block-wide max
  int block_max = BlockReduce(temp_storage).Reduce(val, cub_maximum{});

  if (threadIdx.x == 0)
  {
    atomicMax(max_status, block_max);
  }
}

template <unsigned int block_size>
__global__ void
max_device_status_kernel_batch(int *statuses, const size_t *num_chunk_offsets, const size_t total_num_chunks)
{
  using BlockReduce = cub::BlockReduce<int, block_size>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  auto batch_id = blockIdx.y;
  auto chunk_id = blockIdx.x * blockDim.x + threadIdx.x;

  size_t offset = num_chunk_offsets[batch_id];
  size_t num_chunks = (batch_id == (gridDim.y - 1) ? total_num_chunks : num_chunk_offsets[batch_id + 1]) - offset;
  if (num_chunks == 1)
  {
    // If the batch has a single chunk, there's nothing to do
    return;
  }

  int *max_status = statuses + offset + num_chunks - 1;
  int val = chunk_id < (num_chunks - 1) ? statuses[offset + chunk_id] : static_cast<int>(nvcompSuccess);

  // Compute block-wide max
  int block_max = BlockReduce(temp_storage).Reduce(val, cub_maximum{});

  if (threadIdx.x == 0)
  {
    atomicMax(max_status, block_max);
  }
}

__global__ void batched_device_status_writeout_kernel(
  nvcompStatus_t *d_statuses,
  nvcompStatus_t *h_statuses,
  const size_t *d_num_chunk_offsets,
  const size_t total_num_chunks,
  const size_t batch_count
)
{
  auto batch_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch_id >= batch_count)
  {
    return;
  }

  size_t offset = (batch_id == (batch_count - 1) ? total_num_chunks : d_num_chunk_offsets[batch_id + 1]) - 1;
  h_statuses[batch_id] = d_statuses[offset];
}

void cub_exclusive_sum_scratch_compute_size_t(size_t &scratch_buffer_req, const size_t num_items, cudaStream_t stream)
{
  // Size the temporary storage. input/output are dummy values, not used in scratch compute
  size_t *input{};
  size_t *output{};
  nvcomp::cub::DeviceScan::ExclusiveSum(nullptr, scratch_buffer_req, input, output, narrow_cast<int>(num_items), stream);
}

void cub_exclusive_sum(
  const size_t *input,
  size_t *output,
  size_t num_items,
  uint8_t *scratch_buffer,
  const size_t scratch_buffer_size,
  cudaStream_t stream
)
{
  // Run the prefix sum (exclusive). The copied scratch buffer is necessary because we don't want
  // exclusive sum to receive a non-const ref.
  size_t scratch_buffer_copy = scratch_buffer_size;
  nvcomp::cub::DeviceScan::ExclusiveSum(
    (void *)scratch_buffer,
    scratch_buffer_copy,
    input,
    output,
    narrow_cast<int>(num_items),
    stream
  );
}

__global__ void increase_array(size_t *array, size_t num_elements, size_t increment)
{
  size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < num_elements)
  {
    array[idx] += increment;
  }
}

void max_reduce_device_status_single(
  nvcompStatus_t *d_statuses,
  nvcompStatus_t *h_status,
  const size_t num_chunks,
  cudaStream_t stream
)
{
  // Note:
  // Instead of performing max reduction to a separate bin,
  // we write the end result to the last bin, thus saving us from one extra memory zeroing,
  // and allocating one extra bin.
  static_assert(sizeof(nvcompStatus_t) == sizeof(int));
  int *d_max_status = reinterpret_cast<int *>(d_statuses) + num_chunks - 1;

  if (num_chunks > 1)
  {
    constexpr unsigned int block_size = 256;
    const auto grid_size = cuda_dim_cast((num_chunks - 1 + block_size - 1) / block_size);

    // Note:
    // Throughout the code, there's an implicit assumption that the number of chunks will not exceed
    // std::numeric_limits<unsigned int>::max(). If that is violated, many parts of the codebase needs
    // to change. Hence the safe assumption here that we can just cast down from `size_t` to
    // `unsigned int`.
    max_reduce_device_status_kernel<block_size><<<grid_size, block_size, 0, stream>>>(
      reinterpret_cast<int *>(d_statuses),
      d_max_status,
      narrow_cast<unsigned int>(num_chunks - 1)
    );
    CUDA_CHECK(cudaGetLastError());
  }

  CUDA_CHECK(cudaMemcpyAsync(h_status, d_max_status, sizeof(nvcompStatus_t), cudaMemcpyDeviceToHost, stream));
}

void max_reduce_host_status_batched(
  nvcompStatus_t *pinned_statuses,
  const std::vector<DecompressionConfig> &decomp_configs,
  size_t batch_count
)
{
  size_t running_offset = 0;
  // Per-batch max reduction
  static_assert(sizeof(nvcompStatus_t) == sizeof(int));
  for (size_t batch_id = 0; batch_id < batch_count; ++batch_id)
  {
    const size_t start = running_offset;
    const size_t this_batch_num_chunks = decomp_configs[batch_id].num_chunks;
    const size_t end = start + this_batch_num_chunks;
    int max_status = static_cast<int>(nvcompSuccess);
    for (size_t chunk_id = start; chunk_id < end; ++chunk_id)
    {
      int this_chunk_status = static_cast<int>(pinned_statuses[chunk_id]);
      if (this_chunk_status > max_status)
      {
        max_status = this_chunk_status;
      }
    }
    *decomp_configs[batch_id].get_status() = static_cast<nvcompStatus_t>(max_status);
    running_offset += this_batch_num_chunks;
  }
}

void max_reduce_device_status_batched(
  nvcompStatus_t *d_statuses,
  nvcompStatus_t *h_statuses,
  const size_t *d_num_chunk_offsets,
  const size_t total_num_chunks,
  const size_t max_num_chunks,
  const size_t batch_count,
  cudaStream_t stream
)
{
  // Note:
  // The idea here is similar as for the non-batched variant.
  // We store the maximums in the last item, so we can avoid the global
  // zeroing of the output, and reduce the reduction count.

  constexpr unsigned int block_size = 256;
  dim3 grid_size{cuda_dim_cast((max_num_chunks + block_size - 1) / block_size), cuda_dim_cast(batch_count), 1};
  max_device_status_kernel_batch<block_size>
    <<<grid_size, block_size, 0, stream>>>(reinterpret_cast<int *>(d_statuses), d_num_chunk_offsets, total_num_chunks);
  CUDA_CHECK(cudaGetLastError());

  // Now we have the batch maximums at each individual batch's last status
  // Write them back to the host pinned memory
  // TODO: maybe this could be fused with another kernel following this
  batched_device_status_writeout_kernel<<<cuda_dim_cast((batch_count + block_size - 1) / block_size), block_size, 0, stream>>>(
    d_statuses,
    h_statuses,
    d_num_chunk_offsets,
    total_num_chunks,
    batch_count
  );
  CUDA_CHECK(cudaGetLastError());
}

void increase_array_by(size_t *input, size_t num_elements, size_t increment, cudaStream_t stream)
{
  size_t num_blocks = (num_elements + WARP_SIZE - 1) / WARP_SIZE;
  increase_array<<<cuda_dim_cast(num_blocks), WARP_SIZE, 0, stream>>>(input, num_elements, increment);
  CUDA_CHECK(cudaGetLastError());
}

__global__ void decrease_array(size_t *array, size_t num_elements, size_t decrement)
{
  size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < num_elements)
  {
    array[idx] -= decrement;
  }
}

void decrease_array_by(size_t *input, size_t num_elements, size_t decrement, cudaStream_t stream)
{
  size_t num_blocks = (num_elements + WARP_SIZE - 1) / WARP_SIZE;
  decrease_array<<<cuda_dim_cast(num_blocks), WARP_SIZE, 0, stream>>>(input, num_elements, decrement);
  CUDA_CHECK(cudaGetLastError());
}

} // namespace nvcomp
