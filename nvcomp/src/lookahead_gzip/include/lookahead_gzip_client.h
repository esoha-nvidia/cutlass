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

#include <atomic>
#include <thread>
#include <vector>

#include "lookahead_gzip.h"
#include "nvcomp/shared_types.h"
#include "types.h"

#include <nvcomp/native/streaming_gzip.hpp>

/**
 *  NOTE: This is an internal C++ API for testing the lookahead_gzip algorithm. This will not be exposed to the end-user.
 *        Internally, one can chose between using this C++ API, or the C API via the lookahead_gzip.h header.
 */

namespace lookahead_gzip
{

template <bufferType_t T>
struct kernel_data_t;

template <bufferType_t T>
class LookaheadGzipBaseClient
{
public:
  LookaheadGzipBaseClient(lookaheadGzipConfig_t *config, size_t temp_bytes, uint8_t *device_temp_ptr) noexcept;
  LookaheadGzipBaseClient(const LookaheadGzipBaseClient &other) noexcept = delete;
  LookaheadGzipBaseClient &operator=(const LookaheadGzipBaseClient &other) noexcept = delete;

  virtual ~LookaheadGzipBaseClient() noexcept;

  // Retrieve the amount of scratch space required for decompression in bytes
  static void decompressGetTempSize(lookaheadGzipConfig_t *config, size_t *temp_bytes);

protected:
  // Perform the decompression
  void decompress(
    const uint32_t *const *device_compressed_ptrs,
    const size_t *device_compressed_bytes,
    char *const *device_uncompressed_ptrs,
    const size_t *device_uncompressed_buffer_bytes,
    size_t *device_uncompressed_chunk_bytes,
    uint8_t *device_temp_ptr,
    nvcompStatus_t *decomp_statuses = nullptr,
    cudaStream_t stream = nullptr
  );

  std::vector<kernel_data_t<T>> batch_data;
  uint32_t SM_count;
  uint32_t lhgzip_blocks_per_chunk;
  size_t device_temp_offset;
  uint32_t num_chunks;
};

class LookaheadGzipOneshotClient : public LookaheadGzipBaseClient<bufferType::LINEAR>
{
public:
  LookaheadGzipOneshotClient(lookaheadGzipConfig_t *config, size_t temp_bytes, uint8_t *device_temp_ptr);
  LookaheadGzipOneshotClient(const LookaheadGzipOneshotClient &other) noexcept = delete;
  LookaheadGzipOneshotClient &operator=(const LookaheadGzipOneshotClient &other) noexcept = delete;

  ~LookaheadGzipOneshotClient() noexcept = default;

  // Create config for the one-shot client (no freeing is necessary)
  static void createConfig(size_t num_chunks, cudaStream_t stream, lookaheadGzipConfig_t *config);

  // Retrieve the uncompressed buffer sizes
  static nvcompStatus_t getGzipSize(
    const uint8_t *const *device_compressed_ptrs,
    const size_t *device_compressed_bytes,
    size_t *device_uncompressed_chunk_bytes,
    const size_t num_chunks,
    nvcompStatus_t *device_statuses,
    cudaStream_t stream
  );

  // Perform the decompression
  void decompress(
    const uint32_t *const *device_compressed_ptrs,
    const size_t *device_compressed_bytes,
    char *const *device_uncompressed_ptrs,
    const size_t *device_uncompressed_buffer_bytes,
    size_t *device_uncompressed_chunk_bytes,
    uint8_t *device_temp_ptr,
    nvcompStatus_t *decomp_statuses = nullptr,
    cudaStream_t stream = nullptr
  );

  // Miscellaneous
  static size_t getMinLookaheadGzipCTAAmount() noexcept;
  static size_t getMinLookaheadGzipChunkSize() noexcept;
  void reset(cudaStream_t stream);
};

class LookaheadGzipStreamingClient : public LookaheadGzipBaseClient<bufferType::RING>
{
public:
  LookaheadGzipStreamingClient(lookaheadGzipConfig_t *config, size_t temp_bytes, uint8_t *device_temp_ptr);
  LookaheadGzipStreamingClient(const LookaheadGzipStreamingClient &other) noexcept = delete;
  LookaheadGzipStreamingClient &operator=(const LookaheadGzipStreamingClient &other) noexcept = delete;

  ~LookaheadGzipStreamingClient() noexcept;

  // Create and free configuration for the streaming client
  // @param stream is used to retrieve context SM count and device id
  // should be same as passed to decompress later
  static void createConfig(lookaheadGzipConfig_t *config, cudaStream_t stream);
  static void freeConfig(lookaheadGzipConfig_t *config);

  // Input stream handling
  size_t writeStart(std::istream &input_stream);
  void writeRemainder(std::istream &input_stream);

  // Output stream handling
  nvcompStatus_t read(std::ostream &output_stream, size_t &count);
  void advance(size_t count);

  // Perform the decompression
  void decompress(uint8_t *device_temp_ptr, nvcompStatus_t *decomp_statuses = nullptr, cudaStream_t stream = nullptr);

  // Miscellaneous
  void reset(cudaStream_t stream);
  bool is_done() const;

private:
  size_t uncompressed_read_position;
  std::thread send_input_thread;
  // Note:
  // when the device signals an error via set_error(),
  // then both is_error() and is_done() are true.
  // We only need to monitor done to know when we need to
  // exit our loops.
  std::atomic<bool> done_notification;
  bool done;
  int device_id;
  // Internal buffers
  char *common_buffer;
  void *input_ring_buffer;
  void *output_ring_buffer;
  void *ring_buffer_pointers;
};

} // namespace lookahead_gzip
