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

#include "GzipStreamingUtils.h"

#include <cuda_runtime_api.h>

#include <array>
#include <condition_variable>
#include <exception>
#include <istream>
#include <mutex>
#include <ostream>
#include <utility>

#include "nvcomp/utils.hpp"

namespace nvcomp
{

void write_or_throw(std::ostream &output, const void *data, size_t bytes)
{
  output.write(static_cast<const char *>(data), static_cast<std::streamsize>(bytes));
  if (!output)
  {
    throw NVCompException(nvcompErrorInternal, "gzip streaming: failed to write compressed output");
  }
}

StreamGzipCompressionContext::StreamGzipCompressionContext(
  size_t max_input_size,
  cudaStream_t compute_stream,
  unsigned int *crc_buf
)
{
  this->compute_stream = compute_stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&read_stream, cudaStreamNonBlocking));
  try
  {
    CUDA_CHECK(cudaStreamCreateWithFlags(&write_stream, cudaStreamNonBlocking));
  }
  catch (...)
  {
    CUDA_CHECK_LOG(cudaStreamDestroy(read_stream));
    throw;
  }
  crc_context.crcBuf = crc_buf; // caller-owned, carved from device temp
  this->max_input_size = max_input_size;
}

void StreamGzipCompressionContext::init_crc(size_t *d_input_size, cudaStream_t stream)
{
  crc_context.spec = GZIP_CRC32_SPEC;

  const int conf_status = cuCRC32ConfHeur(
    &crc_context,
    1u,
    reinterpret_cast<const unsigned long long *>(d_input_size),
    &crc_config,
    static_cast<unsigned long long>(max_input_size),
    stream
  );
  if (conf_status != CUCRC32_SUCCESS)
  {
    throw NVCompException(nvcompErrorInternal, "cuCRC32ConfHeur failed");
  }

  if (cuCRC32Beg(&crc_context, 1u, stream) != CUCRC32_SUCCESS)
  {
    throw NVCompException(nvcompErrorInternal, "cuCRC32Beg failed");
  }
}

void StreamGzipCompressionContext::update_crc(
  const uint8_t **input_buffers,
  const size_t *d_input_size,
  cudaEvent_t h2d_done,
  cudaStream_t stream
)
{
  CUDA_CHECK(cudaStreamWaitEvent(stream, h2d_done, 0));
  const int add_status = cuCRC32Add(
    &crc_context,
    crc_config,
    1u,
    reinterpret_cast<const unsigned long long *>(d_input_size),
    reinterpret_cast<const unsigned char *const *>(input_buffers),
    stream
  );
  if (add_status != CUCRC32_SUCCESS)
  {
    throw NVCompException(nvcompErrorInternal, "cuCRC32Add failed");
  }
}

uint32_t StreamGzipCompressionContext::finalize_crc(cudaStream_t stream)
{
  if (cuCRC32End(&crc_context, 1u, stream) != CUCRC32_SUCCESS)
  {
    throw NVCompException(nvcompErrorInternal, "cuCRC32End failed");
  }
  uint32_t host_crc = 0;
  CUDA_CHECK(cudaMemcpyAsync(&host_crc, crc_context.crcBuf, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return host_crc;
}

void StreamGzipCompressionContext::update_isize(size_t partial_isize) { isize += partial_isize; }

StreamGzipCompressionContext::~StreamGzipCompressionContext() noexcept
{
  // compute_stream and crc_context.crcBuf are caller-owned, do not destroy them here.
  CUDA_CHECK_LOG(cudaStreamDestroy(read_stream));
  CUDA_CHECK_LOG(cudaStreamDestroy(write_stream));
}

// This function returns PER WINDOW scratch requirement.
size_t StreamGzipCompressionWindow::device_scratch_bytes_requirement(
  size_t input_buffer_size,
  size_t output_buffer_size,
  size_t temp_buffer_bytes
)
{
  // Sum of the per-window device segments. Must mirror the carve in the constructor.
  return roundUpTo(input_buffer_size, sizeof(size_t)) // input
         + roundUpTo(output_buffer_size, sizeof(size_t)) // output
         + roundUpTo(temp_buffer_bytes, sizeof(size_t)) // scratch
         + 2 * roundUpTo(sizeof(uint8_t *), sizeof(size_t)) // input + output pointer cells
         + 2 * sizeof(size_t) // input + output size cells
         + roundUpTo(sizeof(nvcompStatus_t), sizeof(size_t)); // status cell
}

StreamGzipCompressionWindow::StreamGzipCompressionWindow(
  const size_t input_buffer_size,
  const size_t output_buffer_size,
  const size_t temp_buffer_bytes,
  uint8_t *device_base
)
{
  this->input_buffer_size = input_buffer_size;
  this->temp_bytes = temp_buffer_bytes;

  const size_t in_seg = roundUpTo(input_buffer_size, sizeof(size_t)); // device/host input
  const size_t out_seg = roundUpTo(output_buffer_size, sizeof(size_t)); // device/host output
  const size_t temp_seg = roundUpTo(temp_buffer_bytes, sizeof(size_t)); // device temp
  const size_t ptr_seg = roundUpTo(sizeof(uint8_t *), sizeof(size_t)); // device pointer slot
  const size_t size_seg = sizeof(size_t); // device size slot

  const size_t host_size = in_seg + out_seg;
  CUDA_CHECK(cudaMallocHost(&host_memory_handle_, host_size));

  // Carve this window's device_scratch_bytes_requirement-sized slice of the caller-provided device temp.
  uint8_t *d_base = device_base;
  size_t d_off = 0;
  d_in = d_base + d_off;
  d_off += in_seg;
  d_out = d_base + d_off;
  d_off += out_seg;
  d_temp = d_base + d_off;
  d_off += temp_seg;
  d_in_ptr = reinterpret_cast<const uint8_t **>(d_base + d_off);
  d_off += ptr_seg;
  d_out_ptr = reinterpret_cast<uint8_t **>(d_base + d_off);
  d_off += ptr_seg;
  d_in_size = reinterpret_cast<size_t *>(d_base + d_off);
  d_off += size_seg;
  d_out_size = reinterpret_cast<size_t *>(d_base + d_off);
  d_off += size_seg;
  d_status = reinterpret_cast<nvcompStatus_t *>(d_base + d_off);

  // Carve the pinned-host blob into the staging sub-buffers.
  uint8_t *h_base = static_cast<uint8_t *>(host_memory_handle_);
  h_in = h_base;
  h_out = h_base + in_seg;

  // The batch_size == 1 pointer arrays are constant for this window; publish them once. If any of
  // these CUDA calls throw, the destructor won't run, so release what we already own and rethrow.
  try
  {
    CUDA_CHECK(cudaMemcpy(d_in_ptr, &d_in, sizeof(d_in), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_out_ptr, &d_out, sizeof(d_out), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventCreateWithFlags(&h2d_done, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&compress_done, cudaEventDisableTiming));
  }
  catch (...)
  {
    if (h2d_done)
    {
      CUDA_CHECK_LOG(cudaEventDestroy(h2d_done));
    }
    if (compress_done)
    {
      CUDA_CHECK_LOG(cudaEventDestroy(compress_done));
    }
    CUDA_CHECK_LOG(cudaFreeHost(host_memory_handle_));
    throw;
  }
}

StreamGzipCompressionWindow::~StreamGzipCompressionWindow() noexcept
{
  if (h2d_done)
  {
    CUDA_CHECK_LOG(cudaEventDestroy(h2d_done));
  }
  if (compress_done)
  {
    CUDA_CHECK_LOG(cudaEventDestroy(compress_done));
  }
  // Device memory is owned by the caller-provided temp; only the pinned host is ours.
  CUDA_CHECK_LOG(cudaFreeHost(host_memory_handle_));
}

size_t StreamGzipCompressionWindow::read(std::istream &input, cudaStream_t stream)
{
  input.read(reinterpret_cast<char *>(h_in), static_cast<std::streamsize>(input_buffer_size));
  h_in_size = static_cast<size_t>(input.gcount());
  if (input.bad())
  {
    throw NVCompException(nvcompErrorInternal, "gzip streaming: failed to read input");
  }
  if (h_in_size == 0)
  {
    return 0;
  }
  CUDA_CHECK(cudaMemcpyAsync(d_in, h_in, h_in_size, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_in_size, &h_in_size, sizeof(size_t), cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaEventRecord(h2d_done, stream));
  return h_in_size;
}

size_t StreamGzipCompressionWindow::write(std::ostream &output, cudaStream_t stream)
{
  CUDA_CHECK(cudaEventSynchronize(compress_done));
  // Compression for this window is done; pull back its status alongside its size, and refuse to emit
  // the payload if the GPU reported a compression error.
  nvcompStatus_t status = nvcompSuccess;
  CUDA_CHECK(cudaMemcpyAsync(&status, d_status, sizeof(status), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(&h_out_size, d_out_size, sizeof(size_t), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream)); // need status + h_out_size before copying the payload
  if (status != nvcompSuccess)
  {
    throw NVCompException(status, "gzip streaming: window compression failed");
  }
  CUDA_CHECK(cudaMemcpyAsync(h_out, d_out, h_out_size, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  write_or_throw(output, h_out, h_out_size);
  return h_out_size;
}

// PipelineChannel

void PipelineChannel::push(int value)
{
  {
    std::lock_guard<std::mutex> lk(mutex_);
    queue_.push_back(value);
  }
  cv_.notify_one();
}

bool PipelineChannel::pop(int &out)
{
  std::unique_lock<std::mutex> lk(mutex_);
  cv_.wait(lk, [&] { return !queue_.empty() || closed_; });
  if (queue_.empty())
  {
    return false;
  }
  out = queue_.front();
  queue_.pop_front();
  return true;
}

void PipelineChannel::close()
{
  {
    std::lock_guard<std::mutex> lk(mutex_);
    closed_ = true;
  }
  cv_.notify_all();
}

// StreamGzipPipeline

namespace
{
// Construct the fixed pool of windows in place.
template <std::size_t... I>
std::array<StreamGzipCompressionWindow, sizeof...(I)> make_windows(
  size_t output_buffer_size,
  size_t temp_bytes,
  uint8_t *device_base,
  size_t window_device_bytes,
  std::index_sequence<I...>
)
{
  return {StreamGzipCompressionWindow(
    GZIP_STREAMING_WINDOW_SIZE,
    output_buffer_size,
    temp_bytes,
    device_base + I * window_device_bytes
  )...};
}
} // namespace

StreamGzipPipeline::StreamGzipPipeline(
  StreamGzipCompressionContext &context,
  std::istream &input,
  std::ostream &output,
  size_t output_buffer_size,
  size_t temp_bytes,
  gdeflate::gdeflate_compression_algo algorithm,
  void *device_temp
)
    : context_(context)
    , input_(input)
    , output_(output)
    , algorithm_(algorithm)
    , windows_(make_windows(
        output_buffer_size,
        temp_bytes,
        static_cast<uint8_t *>(device_temp),
        StreamGzipCompressionWindow::device_scratch_bytes_requirement(
          GZIP_STREAMING_WINDOW_SIZE,
          output_buffer_size,
          temp_bytes
        ),
        std::make_index_sequence<NUM_WINDOWS>{}
      ))
{
  // Every window starts free to fill.
  for (int i = 0; i < NUM_WINDOWS; ++i)
  {
    free_queue_.push(i);
  }

  // The running CRC is accumulated on the compute stream, in window order.
  context_.init_crc(windows_[0].d_in_size, context_.compute_stream);
}

void StreamGzipPipeline::fail()
{
  {
    std::lock_guard<std::mutex> lk(error_mutex_);
    if (!first_error_)
    {
      first_error_ = std::current_exception();
    }
  }
  // Unblock every stage so all threads can drain and exit.
  free_queue_.close();
  compress_queue_.close();
  write_queue_.close();
}

void StreamGzipPipeline::run_reader()
{
  try
  {
    int slot = -1;
    while (free_queue_.pop(slot))
    {
      const size_t read_bytes = windows_[slot].read(input_, context_.read_stream);
      if (read_bytes == 0)
      {
        free_queue_.push(slot); // clean EOF; return the unused slot
        break;
      }
      context_.update_isize(read_bytes);
      compress_queue_.push(slot);
    }
    compress_queue_.close();
  }
  catch (...)
  {
    fail();
  }
}

void StreamGzipPipeline::run_compress()
{
  try
  {
    int slot = -1;
    while (compress_queue_.pop(slot))
    {
      context_
        .update_crc(windows_[slot].d_in_ptr, windows_[slot].d_in_size, windows_[slot].h2d_done, context_.compute_stream);
      windows_[slot].compress(context_.compute_stream, algorithm_);
      write_queue_.push(slot);
    }
    write_queue_.close();
  }
  catch (...)
  {
    fail();
  }
}

void StreamGzipPipeline::run_writer()
{
  try
  {
    int slot = -1;
    while (write_queue_.pop(slot))
    {
      windows_[slot].write(output_, context_.write_stream);
      free_queue_.push(slot);
    }
  }
  catch (...)
  {
    fail();
  }
}

void StreamGzipPipeline::rethrow_first_error()
{
  if (first_error_)
  {
    std::rethrow_exception(first_error_);
  }
}

} // namespace nvcomp
