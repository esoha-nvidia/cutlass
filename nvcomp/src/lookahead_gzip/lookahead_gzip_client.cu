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

#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>

#include "common.cuh"
#include "constants.cuh"
#include "CudaUtils.h"
#include "device_guard.h"
#include "include/lookahead_gzip_client.h"
#include "include/types.h"
#include "lookahead_gzip_kernels.cuh"
#include "nvcomp/gzip.h"
#include "nvcomp/shared_types.h"
#include "nvcomp/utils.hpp"

namespace lookahead_gzip
{

using cudaAtomicSystem = cuda::atomic<size_t, cuda::thread_scope::thread_scope_system>;
using cudaAtomicDevice = cuda::atomic<size_t, cuda::thread_scope::thread_scope_device>;
using uncompressedWords = FixedRing<decode_huffman::UncompressedWord, UNCOMPRESSED_WORDS_COUNT>;

// Explicit instantiations
template class LookaheadGzipBaseClient<bufferType::LINEAR>;
template class LookaheadGzipBaseClient<bufferType::RING>;

template <bufferType_t T>
LookaheadGzipBaseClient<T>::LookaheadGzipBaseClient(
  lookaheadGzipConfig_t *config,
  size_t temp_bytes,
  uint8_t *device_temp_ptr
) noexcept
    : device_temp_offset(0)
    , num_chunks(config->num_chunks)
    , SM_count(config->SM_count)
{
  static_assert(
    sizeof(size_t) == 8,
    "We expect size_t to be big enough to store "
    "the number of bits in a very big file"
  );

  lhgzip_blocks_per_chunk =
    std::max(static_cast<int>(SM_count / num_chunks), static_cast<int>(LOOKAHEAD_GZIP_MIN_CTA_AMOUNT)) - 1;

  for (uint32_t i = 0; i < num_chunks; i++)
  {
    batch_data.emplace_back(kernel_data_t<T>());
    auto &kernel_data = batch_data.back();

    kernel_data.total_overlaps = reinterpret_cast<O2OVector<OVERLAP_TYPE> *>(device_temp_ptr + device_temp_offset);
    device_temp_offset += nvcomp::roundUpTo(
      sizeof(O2OVector<OVERLAP_TYPE>) * lhgzip_blocks_per_chunk * O2O_VECTORS_PER_BLOCK * 2,
      alignof(size_t)
    );

    kernel_data.block_bit_offset = reinterpret_cast<size_t *>(device_temp_ptr + device_temp_offset);
    device_temp_offset += nvcomp::roundUpTo(sizeof(size_t), alignof(HuffmanInfo));

    auto mailbox_data = reinterpret_cast<HuffmanInfo *>(device_temp_ptr + device_temp_offset);
    device_temp_offset += nvcomp::roundUpTo(sizeof(HuffmanInfo), alignof(cudaAtomicDevice));

    auto mailbox_request = reinterpret_cast<cudaAtomicDevice *>(device_temp_ptr + device_temp_offset);
    device_temp_offset += nvcomp::roundUpTo(sizeof(cudaAtomicDevice), alignof(cudaAtomicDevice));

    auto mailbox_response = reinterpret_cast<cudaAtomicDevice *>(device_temp_ptr + device_temp_offset);
    device_temp_offset += nvcomp::roundUpTo(sizeof(cudaAtomicDevice), alignof(uncompressedWords));

    kernel_data.huffman_mailbox.init(mailbox_request, mailbox_response, mailbox_data);

    kernel_data.uncompressed_words = reinterpret_cast<uncompressedWords *>(device_temp_ptr + device_temp_offset);
    device_temp_offset += nvcomp::roundUpTo(sizeof(uncompressedWords), alignof(uint));

    kernel_data.uncompressed_words_atomic = reinterpret_cast<uint *>(device_temp_ptr + device_temp_offset);
    device_temp_offset += nvcomp::roundUpTo(sizeof(uint), alignof(int));

    kernel_data.gzip_header_offset = reinterpret_cast<int *>(device_temp_ptr + device_temp_offset);
    device_temp_offset += nvcomp::roundUpTo(sizeof(int), alignof(PartialGrid));

    kernel_data.grid = reinterpret_cast<PartialGrid *>(device_temp_ptr + device_temp_offset);
#if TIMING
    device_temp_offset += nvcomp::roundUpTo(sizeof(PartialGrid), alignof(TimingReporter));
    kernel_data.timing_reporter = reinterpret_cast<TimingReporter *>(device_temp_ptr + device_temp_offset);
    device_temp_offset +=
      nvcomp::roundUpTo(sizeof(TimingReporter), std::max(alignof(O2OVector<OVERLAP_TYPE>), alignof(kernel_data_t<T>)));
#else
    device_temp_offset +=
      nvcomp::roundUpTo(sizeof(PartialGrid), std::max(alignof(O2OVector<OVERLAP_TYPE>), alignof(kernel_data_t<T>)));
#endif // TIMING
  }
}

template <bufferType_t T>
LookaheadGzipBaseClient<T>::~LookaheadGzipBaseClient() noexcept = default;

template <bufferType_t T>
void LookaheadGzipBaseClient<T>::decompressGetTempSize(lookaheadGzipConfig_t *config, size_t *temp_bytes)
{
  size_t lhgzip_blocks_per_chunk = std::max(config->SM_count / config->num_chunks, (int)LOOKAHEAD_GZIP_MIN_CTA_AMOUNT) -
                                   1;

  size_t o2o_vectors_size = sizeof(O2OVector<OVERLAP_TYPE>) * lhgzip_blocks_per_chunk * O2O_VECTORS_PER_BLOCK * 2;
  size_t block_bit_offset_size = sizeof(size_t);
  size_t mailbox_data_size = sizeof(HuffmanInfo);
  size_t mailbox_atomics_size = sizeof(cudaAtomicDevice) * 2;
  size_t uncompressed_words_size = sizeof(uncompressedWords);
  size_t uncompressed_words_atomic_size = sizeof(uint);
  size_t gzip_header_offset_size = sizeof(int);
  size_t partial_grid_size = sizeof(PartialGrid);
#if TIMING
  size_t timing_reporter_size = sizeof(TimingReporter);
#endif // TIMING
  size_t device_kernel_data = sizeof(kernel_data_t<T>);

  // Note: the order is important to make sure we are getting the right amount of temp space
  //       with the right alignment
  size_t required_scratch_space = nvcomp::roundUpTo(o2o_vectors_size, alignof(size_t));
  required_scratch_space += nvcomp::roundUpTo(block_bit_offset_size, alignof(HuffmanInfo));
  required_scratch_space += nvcomp::roundUpTo(mailbox_data_size, alignof(cudaAtomicDevice));
  required_scratch_space += nvcomp::roundUpTo(mailbox_atomics_size, alignof(uncompressedWords));
  required_scratch_space += nvcomp::roundUpTo(uncompressed_words_size, alignof(uint));
  required_scratch_space += nvcomp::roundUpTo(uncompressed_words_atomic_size, alignof(int));
  required_scratch_space += nvcomp::roundUpTo(gzip_header_offset_size, alignof(PartialGrid));
#if TIMING
  required_scratch_space += nvcomp::roundUpTo(partial_grid_size, alignof(TimingReporter));
  required_scratch_space +=
    nvcomp::roundUpTo(timing_reporter_size, std::max(alignof(O2OVector<OVERLAP_TYPE>), alignof(kernel_data_t<T>)));
#else
  required_scratch_space +=
    nvcomp::roundUpTo(partial_grid_size, std::max(alignof(O2OVector<OVERLAP_TYPE>), alignof(kernel_data_t<T>)));
#endif // TIMING
  required_scratch_space += nvcomp::roundUpTo(device_kernel_data, alignof(kernel_data_t<T>));
  required_scratch_space *= config->num_chunks;

  cudaDeviceProp deviceProp;
  CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, config->device_id));
  if (deviceProp.totalGlobalMem < required_scratch_space)
  {
    throw std::runtime_error("Cannot allocate temp memory on your device. Please try bigger chunk size.");
  }

  *temp_bytes = required_scratch_space;
}

template <bufferType_t T>
void LookaheadGzipBaseClient<T>::decompress(
  [[maybe_unused]] const uint32_t *const *device_compressed_ptrs,
  [[maybe_unused]] const size_t *device_compressed_bytes,
  [[maybe_unused]] char *const *device_uncompressed_ptrs,
  [[maybe_unused]] const size_t *device_uncompressed_buffer_bytes,
  [[maybe_unused]] size_t *device_uncompressed_chunk_bytes,
  uint8_t *device_temp_ptr,
  nvcompStatus_t *decomp_statuses,
  cudaStream_t stream
)
{
  // Note:
  // The alignment via device_temp_offset was prepared such that
  // device_kernel_data is naturally aligned for an array of kernel_data_t<T> objects.
  assert(device_temp_offset % alignof(kernel_data_t<T>) == 0);
  auto device_kernel_data = reinterpret_cast<kernel_data_t<T> *>(device_temp_ptr + device_temp_offset);
  CUDA_CHECK(cudaMemcpyAsync(
    device_kernel_data,
    batch_data.data(),
    sizeof(kernel_data_t<T>) * batch_data.size(),
    cudaMemcpyHostToDevice,
    stream
  ));

  // Measured a lot of other values, this is the fastest one.
  constexpr uint32_t GZIP_HEADER_LENGTH_THREADS_PER_BLOCK = 1;

  dim3 headerGridDim(nvcomp::roundUpDiv(num_chunks, GZIP_HEADER_LENGTH_THREADS_PER_BLOCK), 1, 1);
  dim3 headerBlockDim(GZIP_HEADER_LENGTH_THREADS_PER_BLOCK, 1, 1);

  get_header_length<T><<<headerGridDim, headerBlockDim, 0, stream>>>(
    device_kernel_data,
    device_compressed_ptrs,
    static_cast<int>(batch_data.size()),
    lhgzip_blocks_per_chunk
  );
  CUDA_CHECK(cudaGetLastError());

  auto blocks_per_chunk = lhgzip_blocks_per_chunk + 1;

  dim3 blockDim(std::max(THREADS_PER_BLOCK, BLOCK_HEADER_THREADS_PER_BLOCK), 1, 1);
  uint32_t chunk_offset = 0;
  while (chunk_offset < num_chunks)
  {
    dim3 gridDim(blocks_per_chunk, std::min(SM_count / blocks_per_chunk, num_chunks - chunk_offset), 1);
    if constexpr (T == bufferType_t::RING)
    {
      void *lookahead_gzip_kernel_args[] = {&device_kernel_data, &chunk_offset, &decomp_statuses};
      CUDA_CHECK(cudaLaunchCooperativeKernel(
        reinterpret_cast<void *>(combined_streaming_kernel<O2OVector<OVERLAP_TYPE>>),
        gridDim,
        blockDim,
        lookahead_gzip_kernel_args,
        DecodeShared<O2OVector<OVERLAP_TYPE>>(0, lhgzip_blocks_per_chunk).size_in_bytes(),
        stream
      ));
    }
    else
    {
      void *lookahead_gzip_kernel_args[] = {
        &device_kernel_data,
        &device_compressed_ptrs,
        &device_compressed_bytes,
        &device_uncompressed_ptrs,
        &device_uncompressed_buffer_bytes,
        &device_uncompressed_chunk_bytes,
        &chunk_offset,
        &decomp_statuses
      };
      CUDA_CHECK(cudaLaunchCooperativeKernel(
        reinterpret_cast<void *>(combined_oneshot_kernel<O2OVector<OVERLAP_TYPE>>),
        gridDim,
        blockDim,
        lookahead_gzip_kernel_args,
        DecodeShared<O2OVector<OVERLAP_TYPE>>(0, lhgzip_blocks_per_chunk).size_in_bytes(),
        stream
      ));
    }
    CUDA_CHECK(cudaGetLastError());
    chunk_offset += gridDim.y;
  }
}

/*
 * ONESHOT mode function implementation ---------------------------------------
 */
LookaheadGzipOneshotClient::LookaheadGzipOneshotClient(
  lookaheadGzipConfig_t *config,
  size_t temp_bytes,
  uint8_t *device_temp_ptr
)
    : LookaheadGzipBaseClient<bufferType::LINEAR>(config, temp_bytes, device_temp_ptr)
{

  for (uint32_t i = 0; i < num_chunks; i++)
  {
    batch_data[i].huffman_mailbox.reset(config->prefetch_input_stream);
  }
}

void LookaheadGzipOneshotClient::createConfig(size_t num_chunks, cudaStream_t stream, lookaheadGzipConfig_t *config)
{
  config->prefetch_input_stream = stream;
  config->prefetch_output_stream = stream;
  config->num_chunks = static_cast<int>(num_chunks);
  {
    nvcomp::DeviceGuard guard(stream);
    CUDA_CHECK(cudaGetDevice(&config->device_id));
    config->SM_count = nvcomp::CudaUtils::get_sm_count(stream);
  }
}

__global__ void getGzipSizeKernel(
  const uint8_t *const *compressed_ptrs,
  const size_t *compressed_bytes,
  size_t *uncompressed_chunk_bytes,
  nvcompStatus_t *statuses,
  size_t num_chunks
)
{
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_chunks)
  {
    return;
  }

  const uint8_t *ptr = compressed_ptrs[i];
  // Validate gzip magic number
  if (ptr[0] != 0x1f || ptr[1] != 0x8b)
  {
    uncompressed_chunk_bytes[i] = 0;
    if (statuses)
    {
      statuses[i] = nvcompErrorCannotDecompress;
    }
    return;
  }

  // ISIZE: last 4 bytes of the gzip stream (little-endian uint32)
  const uint8_t *trailer = ptr + compressed_bytes[i] - 4;
  uint32_t isize = trailer[0] | (trailer[1] << 8) | (trailer[2] << 16) | (trailer[3] << 24);
  uncompressed_chunk_bytes[i] = isize;
  if (statuses)
  {
    statuses[i] = nvcompSuccess;
  }
}

nvcompStatus_t LookaheadGzipOneshotClient::getGzipSize(
  const uint8_t *const *device_compressed_ptrs,
  const size_t *device_compressed_bytes,
  size_t *device_uncompressed_chunk_bytes,
  const size_t num_chunks,
  nvcompStatus_t *device_statuses,
  cudaStream_t stream
)
{

  const int threads = 256;
  const int blocks = (static_cast<int>(num_chunks) + threads - 1) / threads;
  getGzipSizeKernel<<<blocks, threads, 0, stream>>>(
    device_compressed_ptrs,
    device_compressed_bytes,
    device_uncompressed_chunk_bytes,
    device_statuses,
    num_chunks
  );
  CUDA_CHECK(cudaGetLastError());

  return nvcompSuccess;
}

size_t LookaheadGzipOneshotClient::getMinLookaheadGzipCTAAmount() noexcept { return LOOKAHEAD_GZIP_MIN_CTA_AMOUNT; }

size_t LookaheadGzipOneshotClient::getMinLookaheadGzipChunkSize() noexcept { return LOOKAHEAD_GZIP_MIN_CHUNK_SIZE; }

void LookaheadGzipOneshotClient::decompress(
  const uint32_t *const *device_compressed_ptrs,
  const size_t *device_compressed_bytes,
  char *const *device_uncompressed_ptrs,
  const size_t *device_uncompressed_buffer_bytes,
  size_t *device_uncompressed_chunk_bytes,
  uint8_t *device_temp_ptr,
  nvcompStatus_t *decomp_statuses,
  cudaStream_t stream
)
{
  LookaheadGzipBaseClient<bufferType_t::LINEAR>::decompress(
    device_compressed_ptrs,
    device_compressed_bytes,
    device_uncompressed_ptrs,
    device_uncompressed_buffer_bytes,
    device_uncompressed_chunk_bytes,
    device_temp_ptr,
    decomp_statuses,
    stream
  );
}

/*
 * STREAMING mode function implementation -------------------------------------
 */
LookaheadGzipStreamingClient::LookaheadGzipStreamingClient(
  lookaheadGzipConfig_t *config,
  size_t temp_bytes,
  uint8_t *device_temp_ptr
)
    : LookaheadGzipBaseClient<bufferType::RING>(config, temp_bytes, device_temp_ptr)
    , uncompressed_read_position(0)
    , done_notification(false)
    , done(false)
    , device_id(config->device_id)
    , common_buffer(nullptr)
    , input_ring_buffer(nullptr)
    , output_ring_buffer(nullptr)
    , ring_buffer_pointers(nullptr)
{

  // Allocate managed memory for the input/output ring buffers and the pointers
  size_t buffer_size = nvcomp::roundUpTo(INPUT_RING_BUFFER_SIZE * sizeof(uint32_t), 256) +
                       nvcomp::roundUpTo(OUTPUT_RING_BUFFER_SIZE * sizeof(char), 256) + 4 * sizeof(cudaAtomicSystem);
  CUDA_CHECK(cudaMallocManaged(&common_buffer, buffer_size));
  input_ring_buffer = common_buffer;
  output_ring_buffer = reinterpret_cast<char *>(common_buffer) +
                       nvcomp::roundUpTo(INPUT_RING_BUFFER_SIZE * sizeof(uint32_t), 256);
  ring_buffer_pointers = reinterpret_cast<char *>(output_ring_buffer) +
                         nvcomp::roundUpTo(OUTPUT_RING_BUFFER_SIZE * sizeof(char), 256);

  auto input_read_atomic = reinterpret_cast<cudaAtomicSystem *>(ring_buffer_pointers);
  auto input_write_atomic = reinterpret_cast<cudaAtomicSystem *>(input_read_atomic + 1);
  auto output_read_atomic = reinterpret_cast<cudaAtomicSystem *>(input_write_atomic + 1);
  auto output_write_atomic = reinterpret_cast<cudaAtomicSystem *>(output_read_atomic + 1);

  auto &kernel_data = batch_data.front();

  // H2D stream
  // read and write positions are in managed memory
  kernel_data.deflate_stream.init(
    reinterpret_cast<uint32_t *const *>(&input_ring_buffer),
    input_read_atomic,
    input_write_atomic,
    config->prefetch_input_stream
  );

  // D2H stream
  // read and write positions are in managed memory
  kernel_data.uncompressed_stream.init(
    reinterpret_cast<char *const *>(&output_ring_buffer),
    output_read_atomic,
    output_write_atomic,
    config->prefetch_output_stream
  );
}

LookaheadGzipStreamingClient::~LookaheadGzipStreamingClient() noexcept
{
  try
  {
    // Signal the input thread to stop and wait for it to finish
    done_notification.store(true, std::memory_order_release);
    if (send_input_thread.joinable())
    {
      send_input_thread.join();
    }
    CUDA_CHECK(cudaFree(common_buffer));
  }
  catch (const std::exception &e)
  {
    std::cerr << "Fatal error during cleanup:\n" << e.what() << std::endl;
  }
}

void LookaheadGzipStreamingClient::createConfig(lookaheadGzipConfig_t *config, cudaStream_t stream)
{
  // STREAMING mode supports only single chunk processing.
  config->num_chunks = 1;

  {
    nvcomp::DeviceGuard guard(stream);
    CUDA_CHECK(cudaGetDevice(&config->device_id));

    // Make sure the current architecture with driver can support
    // concurrent access to shared buffers:
    // - concurrentManagedAccess, or
    // - hostNativeAtomicSupported
    // https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/atomic.html
    //
    // Note:
    // The implementation uses managed buffers for concurrent host/device data
    // exchange, hence we only check for that property.
    int concurrentManagedAccess;
    CUDA_CHECK(cudaDeviceGetAttribute(&concurrentManagedAccess, cudaDevAttrConcurrentManagedAccess, config->device_id));
    if (concurrentManagedAccess != 1)
    {
      throw std::runtime_error(
        "Concurrent managed buffer access is not supported on your platform and/or by your driver."
      );
    }

    config->SM_count = nvcomp::CudaUtils::get_sm_count(stream);
    CUDA_CHECK(cudaStreamCreateWithFlags(&config->prefetch_input_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&config->prefetch_output_stream, cudaStreamNonBlocking));
  }
}

void LookaheadGzipStreamingClient::freeConfig(lookaheadGzipConfig_t *config)
{
  CUDA_CHECK(cudaStreamDestroy(config->prefetch_input_stream));
  CUDA_CHECK(cudaStreamDestroy(config->prefetch_output_stream));
}

// TODO(bnagy): this function is a bit awkward, can we get rid of this?
void LookaheadGzipStreamingClient::reset(cudaStream_t stream)
{
  auto &kernel_data = batch_data.front();
  kernel_data.deflate_stream.reset();
  kernel_data.huffman_mailbox.reset(stream);
  kernel_data.uncompressed_stream.reset();

  uncompressed_read_position = 0;
  done_notification = false;
  done = false;
}

void LookaheadGzipStreamingClient::decompress(
  uint8_t *device_temp_ptr,
  nvcompStatus_t *decomp_statuses,
  cudaStream_t stream
)
{
  LookaheadGzipBaseClient<bufferType::RING>::decompress(
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    device_temp_ptr,
    decomp_statuses,
    stream
  );
}

nvcompStatus_t LookaheadGzipStreamingClient::read(std::ostream &output_stream, size_t &count)
{
  auto &kernel_data = batch_data.front();

  // Update GPU write position
  kernel_data.uncompressed_stream.template update_write_position<true>();
  // Note:
  // when error is true, done is also true
  bool error = kernel_data.uncompressed_stream.is_error();
  done = kernel_data.uncompressed_stream.is_done();

  // Don't read until the GPU doesn't have at least ring buffer/2 uncompressed data available
  if (!done && (kernel_data.uncompressed_stream.get_write_position() <
                kernel_data.uncompressed_stream.get_size() / 2 + uncompressed_read_position))
  {
    return nvcompSuccess;
  }
  else if (!error)
  {
    count = done ? kernel_data.uncompressed_stream.get_done_position() - uncompressed_read_position
                 : kernel_data.uncompressed_stream.get_size() / 2;
    kernel_data.uncompressed_stream.prefetch_to_cpu(uncompressed_read_position, count);
  }

  // Write out the received data in parts if the count wraps
  // around the ring-buffer boundary
  size_t count_remaining = count;
  size_t local_read_position = uncompressed_read_position;
  size_t iterations = 0;
  while (count_remaining > 0)
  {
    size_t count_until_end = kernel_data.uncompressed_stream.get_size() -
                             (local_read_position % kernel_data.uncompressed_stream.get_size());
    size_t count_to_write = std::min(count_remaining, count_until_end);
    const char *data = &kernel_data.uncompressed_stream.at(local_read_position);
    output_stream.write(data, count_to_write);

    count_remaining -= count_to_write;
    local_read_position += count_to_write;
    ++iterations;
  }
  assert(iterations <= 2);

  if (done)
  {
    done_notification.store(true, std::memory_order_release);
    send_input_thread.join();
#if TIMING
    int clock_rate;
    TimingReporter timing_reporter;
    CUDA_CHECK(
      cudaMemcpy(&timing_reporter, kernel_data.timing_reporter, sizeof(TimingReporter), cudaMemcpyDeviceToHost)
    );
    CUDA_CHECK(cudaDeviceGetAttribute(&clock_rate, cudaDevAttrClockRate, device_id));
    fprintf(stderr, "clock is %d\n", clock_rate);
    timing_reporter.report_timing();
#endif // TIMING
  }
  return error ? nvcompErrorCannotDecompress : nvcompSuccess;
}

size_t LookaheadGzipStreamingClient::writeStart(std::istream &input_stream)
{
  auto &kernel_data = batch_data.front();
  size_t total_written_bytes = 0;
  while (true)
  {
    auto currently_written_bytes =
      kernel_data.deflate_stream.template append<false>(input_stream, device_id, done_notification);
    if (currently_written_bytes == 0)
    {
      break;
    }
    total_written_bytes += currently_written_bytes;
  }
  return total_written_bytes;
}

void LookaheadGzipStreamingClient::writeRemainder(std::istream &input_stream)
{
  auto send_input = [&]() {
    auto &kernel_data = batch_data.front();
    // Note: when the device signals either done or error
    //       done will turn `true`, and we can exit the "feeding"
    //       loop
    while (!input_stream.fail() && !input_stream.eof() && !done)
    {
      kernel_data.deflate_stream.append<true>(input_stream, device_id, done_notification);
    }
  };
  send_input_thread = std::thread(send_input);
}

void LookaheadGzipStreamingClient::advance(size_t count)
{
  auto &kernel_data = batch_data.front();

  kernel_data.uncompressed_stream.prefetch_to_gpu(uncompressed_read_position, count, device_id);
  uncompressed_read_position += count;

  // Note:
  // Make sure, that the pages were migrated to the host
  // before we notify the GPU about the new position
  kernel_data.uncompressed_stream.sync_with_stream();

  kernel_data.uncompressed_stream.set_read_position(uncompressed_read_position);
}

bool LookaheadGzipStreamingClient::is_done() const { return done; }

} // namespace lookahead_gzip
