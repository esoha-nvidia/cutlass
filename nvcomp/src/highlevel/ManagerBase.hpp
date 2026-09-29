/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <algorithm>
#include <ciso646>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "allocators/host_pinned.hpp"
#include "common.h"
#include "CRC32.hpp"
#include "CudaUtils.h"
#include "highlevel/CompressionConfigs.hpp"
#include "highlevel/ManagerUtils.hpp"
#include "HWDecompress.hpp"
#include "nvcomp/ans.hpp"
#include "nvcomp/bitcomp.hpp"
#include "nvcomp/cascaded.hpp"
#include "nvcomp/deflate.hpp"
#include "nvcomp/formatSpec.hpp"
#include "nvcomp/gdeflate.hpp"
#include "nvcomp/gzip.hpp"
#include "nvcomp/lz4.hpp"
#include "nvcomp/nvcompManager.hpp"
#include "nvcomp/snappy.hpp"
#include "nvcomp/zstd.hpp"
#include "nvcomp_common_deps/hlif_shared.hpp"
#include "nvcomp_common_deps/hlif_shared_types.hpp"
#include "TemplateFunctions.cuh"

// clang-format off
// device_memory_pool.hpp requires device_guard.h and exception.hpp before it
// (it uses CudaDriver and DeviceGuard without including them - see comment in that header)
#include "device_guard.h"
#include "exception.hpp"
#include "allocators/device_memory_pool.hpp"
// clang-format on

namespace nvcomp
{

/**
 * @brief Query a format's buffer alignment requirements into a value.
 *
 * @param get_alignments_fn The format's GetRequiredAlignments function.
 * @param opts The compress/decompress options to query with.
 * @return The filled alignment requirements (input/output/temp).
 */
template <typename GetAlignmentsFn_t, typename Opts_t>
inline nvcompAlignmentRequirements_t query_alignment_requirements(GetAlignmentsFn_t get_alignments_fn, Opts_t opts)
{
  nvcompAlignmentRequirements_t alignment{};
  const auto status = get_alignments_fn(opts, &alignment);
  if (status != nvcompSuccess)
  {
    throw NVCompException(status, "Failed to query alignment requirements");
  }
  return alignment;
}

/**
 * @brief ManagerBase contains shared functionality amongst the different nvcompManager types
 *
 * - Intended that all Managers will inherit from this class directly or indirectly.
 *
 * - Contains a CPU/GPU-accessible memory pool for result statuses to avoid repeated
 *   allocations when tasked with multiple compressions / decompressions.
 *
 * - Templated on the particular format's FormatSpecHeader so that some operations can be shared here.
 *   This is likely to be inherited by template classes. In this case,
 *   some usage trickery is suggested to get around dependent name lookup issues.
 *   https://en.cppreference.com/w/cpp/language/dependent_name
 *
 */
template <
  typename FormatSpecHeader,
  typename DecompressFn_t,
  typename DecompressScratchFn_t,
  typename DecompressSizeFn_t,
  typename CompressFn_t,
  typename CompressScratchFn_t,
  typename MaxCompChunkSizeFn_t,
  typename CompressOpts_t,
  typename DecompressOpts_t,
  nvcompFormatType_t format_type>
struct ManagerBase : detail::nvcompManagerInternalBase
{

protected: // members
  CommonHeader *common_header_cpu;
  nvcompStatus_t *common_status_cpu;
  cudaStream_t user_stream;
  uint8_t *scratch_buffer;
  size_t scratch_buffer_size;
  AllocFn_t allocator;
  DeAllocFn_t deallocator;
  const size_t uncomp_chunk_size;
  size_t min_alignment;
  size_t max_comp_chunk_size;
  CompressOpts_t format_opts;
  DecompressOpts_t decompress_opts;
  ChecksumPolicy checksum_policy;
  FormatSpecHeader format_spec;
  BitstreamKind bitstream_kind;
  nvcompAlignmentRequirements_t compress_alignment;
  nvcompAlignmentRequirements_t decompress_alignment;

private: // members
  bool use_async_mem_ops;

private: // function handles
  DecompressFn_t decompress_fn;
  DecompressScratchFn_t decomp_scratch_size_fn;
  DecompressSizeFn_t decomp_size_fn;
  CompressFn_t compress_fn;
  CompressScratchFn_t comp_scratch_size_fn;
  MaxCompChunkSizeFn_t max_comp_size_fn;

public: // API
  /**
   * @brief Construct a ManagerBase
   *
   * @param user_stream The stream to use for all operations. Optional, defaults to the default stream
   */
  ManagerBase(
    size_t uncomp_chunk_size,
    CompressOpts_t format_opts,
    DecompressOpts_t decompress_opts,
    cudaStream_t user_stream,
    ChecksumPolicy checksum_policy,
    BitstreamKind bitstream_kind,
    DecompressFn_t decompress_fn,
    DecompressScratchFn_t decomp_scratch_size_fn,
    DecompressSizeFn_t decomp_size_fn,
    CompressFn_t compress_fn,
    CompressScratchFn_t comp_scratch_size_fn,
    MaxCompChunkSizeFn_t max_comp_size_fn,
    nvcompAlignmentRequirements_t compress_alignment,
    nvcompAlignmentRequirements_t decompress_alignment
  )
      : common_header_cpu(nullptr)
      , common_status_cpu(nullptr)
      , user_stream(user_stream)
      , scratch_buffer(nullptr)
      , scratch_buffer_size(0)
      , uncomp_chunk_size(uncomp_chunk_size)
      , format_opts(format_opts)
      , decompress_opts(decompress_opts)
      , checksum_policy(checksum_policy)
      , format_spec()
      , bitstream_kind(bitstream_kind)
      , compress_alignment(compress_alignment)
      , decompress_alignment(decompress_alignment)
      , use_async_mem_ops(false)
      , decompress_fn(decompress_fn)
      , decomp_scratch_size_fn(decomp_scratch_size_fn)
      , decomp_size_fn(decomp_size_fn)
      , compress_fn(compress_fn)
      , comp_scratch_size_fn(comp_scratch_size_fn)
      , max_comp_size_fn(max_comp_size_fn)
  {
    // In NVCOMP_NATIVE the input is tiled into chunks of `uncomp_chunk_size`
    // (compression: input + k * uncomp_chunk_size) and the output is tiled the
    // same way during decompression (decomp_buffer + k * uncomp_chunk_size).
    // For every chunk boundary to satisfy the compressor's alignment
    // requirements, `uncomp_chunk_size` must be a multiple of both the
    // compression input alignment and the decompression output alignment.
    if (bitstream_kind == BitstreamKind::NVCOMP_NATIVE)
    {
      const size_t required_alignment = std::max<size_t>(compress_alignment.input, decompress_alignment.output);
      if (uncomp_chunk_size % required_alignment != 0)
      {
        throw NVCompException(
          nvcompErrorInvalidValue,
          "Uncompressed chunk size (" + std::to_string(uncomp_chunk_size) +
            ") must be a multiple of the required buffer alignment (" + std::to_string(required_alignment) +
            " bytes) for the NVCOMP_NATIVE bitstream."
        );
      }
    }

    // Derive a single alignment that satisfies every buffer the format touches
    // in the NVCOMP_NATIVE compress/decompress paths.
    min_alignment = std::max<size_t>(
      {compress_alignment.input,
       compress_alignment.output,
       compress_alignment.temp,
       decompress_alignment.input,
       decompress_alignment.output,
       decompress_alignment.temp,
       sizeof(size_t)}
    );

    ManagerBase::check<FnType::MaxCompChunkSize>(max_comp_size_fn(uncomp_chunk_size, format_opts, &max_comp_chunk_size));
    max_comp_chunk_size = roundUpTo(max_comp_chunk_size, min_alignment);

    // Preallocate the pinned host scratch buffer
    allocate_host_scratch();

    use_async_mem_ops = CudaUtils::can_use_async_mem_ops(user_stream);

    allocator = [this](size_t alloc_bytes) {
      uint8_t *temp_scratch_buffer;
      if (use_async_mem_ops)
      {
        temp_scratch_buffer = static_cast<uint8_t *>(get_device_memory_pool().allocate(alloc_bytes, this->user_stream));
      }
      else
      {
        DeviceGuard device_guard(this->user_stream);
        CUDA_CHECK(cudaMalloc(&temp_scratch_buffer, alloc_bytes));
      }
      return temp_scratch_buffer;
    };

    deallocator = [this](void *ptr, size_t /* used by size-aware allocators */) {
      if (use_async_mem_ops)
      {
        CUDA_CHECK(cudaFreeAsync(ptr, this->user_stream));
      }
      else
      {
        CUDA_CHECK(cudaFree(ptr));
      }
    };

    if (bitstream_kind != BitstreamKind::NVCOMP_NATIVE)
    {
      if (checksum_policy != ChecksumPolicy::NoComputeNoVerify)
      {
        std::string error_message = "Only ChecksumPolicy::NoComputeNoVerify is allowed "
                                    "when using bitstream kind different than NVCOMP_NATIVE";
        throw NVCompException(nvcompErrorNotSupported, error_message);
      }
    }
  }

  // Disable copying
  ManagerBase(const ManagerBase &) = delete;
  ManagerBase &operator=(const ManagerBase &) = delete;
  ManagerBase() = delete;

  virtual ~ManagerBase()
  {
    try
    {
      deallocate_host_scratch();
      deallocate_gpu_scratch();
    }
    catch (const std::runtime_error &err)
    {
      std::cerr << "Fatal error in ManagerBase destructor:" << std::endl << err.what() << std::endl;
    }
  }

  size_t get_compressed_output_size(const uint8_t *comp_buffer) final;

  std::vector<size_t> get_compressed_output_size(const uint8_t *const *comp_buffers, size_t batch_size) final;

  size_t get_decompressed_output_size(const uint8_t *comp_buffer) final;

  std::vector<size_t> get_decompressed_output_size(const uint8_t *const *comp_buffers, size_t batch_size) final;

  CompressionConfig configure_compression(const size_t uncomp_buffer_size) final;

  std::vector<CompressionConfig> configure_compression(const std::vector<size_t> &uncomp_buffer_sizes) final;

  // Helper function to fill in the DecompressionConfig from the CommonHeader
  DecompressionConfig extract_decomp_config(const CommonHeader *common_header);

  DecompressionConfig configure_decompression(const uint8_t *comp_buffer, const size_t *comp_size = nullptr) final;

  std::vector<DecompressionConfig> configure_decompression(
    const uint8_t *const *comp_buffers,
    size_t batch_size,
    const size_t *comp_sizes = nullptr
  ) final;

  DecompressionConfig configure_decompression(const CompressionConfig &comp_config) final;

  std::vector<DecompressionConfig> configure_decompression(const std::vector<CompressionConfig> &comp_configs) final;

  void set_scratch_allocators(const AllocFn_t &alloc_fn, const DeAllocFn_t &dealloc_fn) final;

  void compress(
    const uint8_t *uncomp_buffer,
    uint8_t *comp_buffer,
    const CompressionConfig &comp_config,
    size_t *comp_size = nullptr
  ) final;

  void compress(
    const uint8_t *const *uncomp_buffers,
    uint8_t *const *comp_buffers,
    const std::vector<CompressionConfig> &comp_configs,
    size_t *comp_sizes = nullptr
  ) final;

  void decompress(
    uint8_t *decomp_buffer,
    const uint8_t *comp_buffer,
    const DecompressionConfig &decomp_config,
    size_t *comp_size = nullptr
  ) final;

  void decompress(
    uint8_t *const *decomp_buffers,
    const uint8_t *const *comp_buffers,
    const std::vector<DecompressionConfig> &decomp_configs,
    const size_t *comp_sizes = nullptr
  ) final;

  void decompress(
    uint8_t *const *decomp_buffers,
    const uint8_t *const *comp_buffers,
    const std::vector<DecompressionConfig> &decomp_configs,
    const size_t *comp_sizes,
    const size_t batch_count,
    const uint8_t *const *host_comp_buffers
  ) final;

  // This is used to replicate an input HLIF compressed buffer in
  // NVCOMP_NATIVE format.
  // This allows us to decompress / compress with the same data statistics
  // while benchmarking realistic sizes
  std::vector<uint8_t> do_replicate(const std::vector<uint8_t> &input_data, int extra_rep_count);

private: // helpers
  enum class FnType
  {
    Decompress = 0,
    DecompressScratch,
    DecompressSize,
    Compress,
    CompressScratch,
    MaxCompChunkSize
  };

  /**
   * @brief Function that verifes the return value of a callback function,
   * and raises an exception whenever the return value is not nvcompSuccess.
   * @param result Return value of the callback function to be evaluated.
   */
  template <FnType type>
  static inline void check(nvcompStatus_t result)
  {
    if (result == nvcompSuccess)
    {
      return;
    }

    const char *error_str = nullptr;
    if constexpr (type == FnType::Decompress)
    {
      error_str = "Could not perform decompression.";
    }
    else if constexpr (type == FnType::DecompressScratch)
    {
      error_str = "Could not determine the decompression scratch requirement.";
    }
    else if constexpr (type == FnType::DecompressSize)
    {
      error_str = "Could not determine the decompressed size.";
    }
    else if constexpr (type == FnType::Compress)
    {
      error_str = "Could not perform compression.";
    }
    else if constexpr (type == FnType::CompressScratch)
    {
      error_str = "Could not determine the compression scratch requirement";
    }
    else if constexpr (type == FnType::MaxCompChunkSize)
    {
      error_str = "Could not determine the maximum compressed chunk size.";
    }
    else
    {
      assert(false);
    }

    throw NVCompException(result, error_str);
  }

  /**
   * @brief Validate that a user-supplied buffer pointer satisfies the required alignment.
   *
   * @param ptr The buffer pointer to validate.
   * @param alignment The required alignment in bytes. Alignments of 0 or 1
   *        impose no constraint and are skipped.
   * @param what Human-readable description of the buffer for the error message.
   *
   * @throws NVCompException(nvcompErrorAlignment) if @p ptr is not a multiple of @p alignment.
   */
  static void check_buffer_alignment(const void *ptr, size_t alignment, const char *what)
  {
    if (alignment <= 1)
    {
      return;
    }
    if (reinterpret_cast<uintptr_t>(ptr) % alignment != 0)
    {
      throw NVCompException(
        nvcompErrorAlignment,
        std::string(what) + " buffer is not aligned to the required alignment of " + std::to_string(alignment) +
          " bytes."
      );
    }
  }

  /**
   * @brief Required helper that actually does the compression in NVCOMP_NATIVE for a single element
   * For param meaning see nvcompManager::compress()
   */
  void compress_chunked_single(
    const uint8_t *uncomp_buffer,
    uint8_t *comp_buffer,
    const CompressionConfig &comp_config,
    size_t *comp_size = nullptr
  );

  /**
   * @brief Required helper that actually does the compression in NVCOMP_NATIVE for multiple batches
   * For param meaning see nvcompManager::compress()
   */
  void compress_chunked_batched(
    const uint8_t *const *input_uncomp_buffers,
    uint8_t *const *output_comp_buffers,
    const std::vector<CompressionConfig> &comp_configs,
    size_t *comp_sizes = nullptr
  );

  /**
   * @brief Required helper that actually does the compression in RAW and WITH_UNCOMPRESSED_SIZE formats for batch of elements
   * For param meaning see nvcompManager::compress()
   */
  void compress_raw_batched(
    const uint8_t *const *uncomp_buffers,
    uint8_t *const *comp_buffers,
    const std::vector<CompressionConfig> &comp_configs,
    size_t *comp_sizes
  );

  /**
   * @brief Required helper that actually does the decompression
   *
   * @param decomp_buffer The location to output the decompressed data to (GPU accessible).
   * @param comp_buffer The compressed input data (GPU accessible).
   * @param decomp_config Resulted from configure_decompression given this decomp_buffer_size.
   */
  void decompress_chunked_single(
    uint8_t *decomp_buffer,
    const uint8_t *comp_buffer,
    const DecompressionConfig &decomp_config
  );

  /**
   * @brief Computes pinned memory size and decompress scratch size for batched decompression.
   * "host" prefix implies that the correponding variable is in host accessible memory.
   * HostSetup: reads headers from host_comp_buffers, fills host_global_chunk_offset and sizes.
   * DeviceSetup: uses decomp_configs to compute total_num_chunks and max_num_chunks.
   */
  void compute_decompress_hlif_mem_size(
    bool host_setup_mode,
    bool decomp_config_used,
    size_t batch_count,
    size_t &total_num_chunks,
    size_t &max_num_chunks,
    size_t &pinned_mem_size,
    size_t &total_decompress_scratch_req,
    const uint8_t *const *host_comp_buffers,
    const std::vector<DecompressionConfig> &decomp_configs
  );

  /**
   * @brief Host path: layout pinned chunk arrays, run setup_batched_decomp_llif_buffers_host, then decompress kernel.
   */
  void run_decompress_chunked_batched_host_path(
    const uint8_t *const *host_comp_buffers,
    uint8_t **pinned_input_comp_buffers,
    uint8_t **pinned_output_decomp_buffers,
    size_t batch_count,
    size_t total_num_chunks,
    int header_size,
    bool decomp_config_used,
    const std::vector<DecompressionConfig> &decomp_configs
  );

  /**
   * @brief Device path: pass pinned batch metadata directly to setup kernel, decompress, max-reduce statuses.
   */
  void run_decompress_chunked_batched_device_path(
    uint8_t **pinned_input_comp_buffers,
    uint8_t **pinned_output_decomp_buffers,
    size_t batch_count,
    size_t total_num_chunks,
    size_t max_num_chunks,
    size_t *pinned_global_chunk_offset,
    nvcompStatus_t *pinned_decomp_config_statuses,
    int header_size,
    const std::vector<DecompressionConfig> &decomp_configs
  );

  /**
   * @brief Performs batched decompression.
   * When host_comp_buffers is non-null, all setup is performed on host.
   * Otherwise, the device path runs where setup is performed on device.
   *
   * @param output_decomp_buffers The location to output the decompressed data for each batch (GPU accessible).
   * @param input_comp_buffers The compressed input data for each batch (GPU accessible).
   * @param decomp_configs Result from configure_decompression for each batch.
   * @param batch_count Number of batches (must equal decomp_configs.size()).
   * @param host_comp_buffers If non-null, host-accessible compressed buffers used to read headers (enables host setup path); pointer arrays are copied to pinned internally.
   */
  void decompress_chunked_batched(
    uint8_t *const *output_decomp_buffers,
    const uint8_t *const *input_comp_buffers,
    const std::vector<DecompressionConfig> &decomp_configs,
    const size_t batch_count,
    const uint8_t *const *host_comp_buffers = nullptr
  );

  void recompress(
    const uint8_t *decomp_buffer,
    const uint8_t *comp_buffer,
    const DecompressionConfig &decomp_config,
    size_t &recompress_size,
    float &recompress_throughput,
    float &recompress_ratio
  );

  void decompress_raw_batched(
    uint8_t *const *decomp_buffers,
    const uint8_t *const *comp_buffers,
    const std::vector<DecompressionConfig> &decomp_configs,
    const size_t *comp_sizes
  );

  size_t compute_lowlevel_compress_scratch_size(const size_t batch_size, const size_t max_uncomp_chunk_size);

  /**
   * @brief Computes the required scratch buffer size for compression
   */
  size_t compute_compress_scratch_buffer_size(const size_t batch_size, const bool with_compaction);

  /**
   * @brief Computes the required scratch buffer size for compression of multiple batches
   */
  size_t compute_batched_compress_scratch_buffer_size(
    const std::vector<CompressionConfig> &comp_configs,
    size_t total_num_chunks
  );

  size_t compute_lowlevel_decompress_scratch_size(const size_t batch_size, const size_t max_uncomp_chunk_size);

  /**
   * @brief Computes the required scratch buffer size for decompression
   */
  size_t compute_decompress_scratch_buffer_size(const size_t batch_size);

  /**
   * @brief Allocate device-accessible scratch space
   *
   * @note This scratch allocator does not come with any memory alignment guarantee,
   * therefore, users of this function should factor in the worst-case number of bytes
   * for the required alignment of the very first type the pointer gets casted to.
   */
  void allocate_gpu_scratch(size_t req_scratch_memory);

  void deallocate_gpu_scratch();

  /**
   * @brief Initialize the format_spec member from format_opts (shared by the
   * chunked-single and chunked-batched compress paths).
   */
  void init_format_spec();

  void allocate_host_scratch();

  void deallocate_host_scratch();

  void deallocate_gpu_mem();
};

} // namespace nvcomp

// Out-of-line definitions of the ManagerBase member functions.
#include "highlevel/ManagerBase/Benchmark.hpp"
#include "highlevel/ManagerBase/CompressImpl.hpp"
#include "highlevel/ManagerBase/ConfigAPI.hpp"
#include "highlevel/ManagerBase/DecompressImpl.hpp"
#include "highlevel/ManagerBase/DispatchAPI.hpp"
#include "highlevel/ManagerBase/ScratchMemory.hpp"
