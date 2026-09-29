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

#include <array>
#include <condition_variable>
#include <deque>
#include <exception>
#include <iosfwd>
#include <mutex>

#include "crc/cuCRC32.h"
#include "exception.hpp"
#include "gdeflate/gdeflate.h"
#include "GzipConstants.cuh"
#include "nvcomp/shared_types.h"

namespace nvcomp
{

// Running state shared by the whole stream
struct StreamGzipCompressionContext
{
  size_t isize = 0; // running uncompressed size
  crcCtx_t crc_context{}; // cuCRC32 running state; crc_context.crcBuf is caller-owned (carved from device temp)
  unsigned int crc_config = 0;
  size_t max_input_size = 0;

  cudaStream_t read_stream = nullptr;
  cudaStream_t compute_stream = nullptr;
  cudaStream_t write_stream = nullptr;

  // crc_buf is a device uint32_t cell carved from the caller-provided temp; the context does not own it.
  StreamGzipCompressionContext(size_t max_input_size, cudaStream_t compute_stream, unsigned int *crc_buf);
  ~StreamGzipCompressionContext() noexcept;

  StreamGzipCompressionContext(const StreamGzipCompressionContext &) = delete;
  StreamGzipCompressionContext &operator=(const StreamGzipCompressionContext &) = delete;

  void init_crc(size_t *d_input_size, cudaStream_t stream);
  void update_crc(const uint8_t **input_buffers, const size_t *d_input_size, cudaEvent_t h2d_done, cudaStream_t stream);
  uint32_t finalize_crc(cudaStream_t stream);

  // called only by reader thread and read by host thread only after reader joins
  void update_isize(size_t partial_isize);
};

class StreamGzipCompressionWindow
{
private:
  void *host_memory_handle_ = nullptr; // pinned host staging, owned by the window

public:
  uint8_t *h_in = nullptr; // pinned host input staging       [input_buffer_size]
  uint8_t *h_out = nullptr; // pinned host output staging      [output_buffer_size]
  uint8_t *d_in = nullptr; // device input                    [input_buffer_size]
  uint8_t *d_out = nullptr; // device output (deflate payload) [output_buffer_size]
  uint8_t *d_temp = nullptr; // device scratch                  [temp_bytes]
  const uint8_t **d_in_ptr = nullptr; // device [1] -> d_in   (input_buffers, batch_size == 1)
  uint8_t **d_out_ptr = nullptr; // device [1] -> d_out  (compressed_buffers)
  size_t *d_in_size = nullptr; // device [1] uncompressed size of this window
  size_t *d_out_size = nullptr; // device [1] compressed size of this window
  nvcompStatus_t *d_status = nullptr; // device [1] compression status of this window
  size_t h_in_size = 0; // bytes read into h_in for this window
  size_t h_out_size = 0; // compressed bytes produced

  size_t input_buffer_size = 0; // max bytes per window
  size_t temp_bytes = 0; // size of d_temp

  cudaEvent_t h2d_done = nullptr; // input H2D copy complete (on the read stream)
  cudaEvent_t compress_done = nullptr; // CRC add + compress complete (on the compute stream)

  // This function returns PER WINDOW scratch requirement.
  static size_t
  device_scratch_bytes_requirement(size_t input_buffer_size, size_t output_buffer_size, size_t temp_buffer_bytes);

  StreamGzipCompressionWindow(
    const size_t input_buffer_size,
    const size_t output_buffer_size,
    const size_t temp_buffer_bytes,
    uint8_t *device_base
  );
  ~StreamGzipCompressionWindow() noexcept;

  StreamGzipCompressionWindow(StreamGzipCompressionWindow &) = delete;
  StreamGzipCompressionWindow(StreamGzipCompressionWindow &&) = delete;
  StreamGzipCompressionWindow &operator=(StreamGzipCompressionWindow &) = delete;
  StreamGzipCompressionWindow &operator=(StreamGzipCompressionWindow &&) = delete;

  // host-read one window and start its host->device copy on passed stream; record
  // h2d_done. Returns bytes read (0 == EOF).
  size_t read(std::istream &input, cudaStream_t stream);

  // compress this window on passed stream as a single non-final payload; record compress_done.
  void compress(cudaStream_t stream, gdeflate::gdeflate_compression_algo algorithm);

  // wait for this window's compress, copy D2H the payload back on passed stream, and append it to output.
  // Returns the compressed byte count.
  size_t write(std::ostream &output, cudaStream_t stream);
};

void write_or_throw(std::ostream &output, const void *data, size_t bytes);

// Thread-safe, closable FIFO of windows indices: the streaming pipeline hands windows between its
// reader / GPU / writer stages through these
class PipelineChannel
{
public:
  void push(int value);

  // Blocks until an item is available; returns false once the channel is closed AND drained.
  bool pop(int &out);

  // Signal end-of-stream: wake all waiters; subsequent pop()s drain what's left then return false.
  void close();

private:
  std::mutex mutex_{};
  std::condition_variable cv_{};
  std::deque<int> queue_{};
  bool closed_ = false;
};

// The read -> compress -> write windows pipeline.
class StreamGzipPipeline
{
public:
  StreamGzipPipeline(
    StreamGzipCompressionContext &context,
    std::istream &input,
    std::ostream &output,
    size_t output_buffer_size,
    size_t temp_bytes,
    gdeflate::gdeflate_compression_algo algorithm,
    void *device_temp
  );

  void run_reader(); // fill free windows from `input`, hand them to the GPU stage (own thread)
  void run_compress(); // CRC + compress each window in order, hand it to the writer (calling thread)
  void run_writer(); // retire each window to `output` in order, return the window (own thread)

  void rethrow_first_error(); // re-raise the first stage exception (if any) on the calling thread

  // Record the first error and close every channel so all stages unblock and drain. Stages call this on
  // their own failure; the launcher also calls it if a worker thread fails to start, to release the
  // already-running stages instead of deadlocking on join.
  void fail();

private:
  static constexpr int NUM_WINDOWS = STREAMING_NUM_WINDOWS;

  StreamGzipCompressionContext &context_;
  std::istream &input_;
  std::ostream &output_;
  gdeflate::gdeflate_compression_algo algorithm_; // gdeflate level for every window

  std::array<StreamGzipCompressionWindow, NUM_WINDOWS> windows_;
  PipelineChannel free_queue_{}; // empty windows ready to be filled by the reader
  PipelineChannel compress_queue_{}; // filled windows ready for processing
  PipelineChannel write_queue_{}; // compressed windows ready for the writer

  std::exception_ptr first_error_{};
  std::mutex error_mutex_{};
};

} // namespace nvcomp
