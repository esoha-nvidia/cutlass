/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "nvcomp/nvcompManagerFactory.hpp"

#include "common.h"
#include "CudaUtils.h"
#include "exception.hpp"
#include "nvcomp.hpp"
#include "nvcomp/ans.hpp"
#include "nvcomp/bitcomp.hpp"
#include "nvcomp/cascaded.hpp"
#include "nvcomp/deflate.hpp"
#include "nvcomp/gdeflate.hpp"
#include "nvcomp/gzip.hpp"
#include "nvcomp/lz4.hpp"
#include "nvcomp/nvcompManager.hpp"
#include "nvcomp/snappy.hpp"
#include "nvcomp/zstd.hpp"
#include "nvcomp_common_deps/hlif_shared_types.hpp"

#include <assert.h>

namespace nvcomp
{

template <typename FormatSpecHeader, typename ManagerType, typename OptsConvFn_t>
std::shared_ptr<nvcompManagerBase> do_create_manager(
  const uint8_t *comp_buffer,
  ChecksumPolicy checksum_policy,
  OptsConvFn_t &&format_opts_fn,
  const CommonHeader &cpu_common_header,
  cudaStream_t stream
)
{
  validate_hlif_magic_number(cpu_common_header.magic_number);

  FormatSpecHeader format_spec;
  if constexpr (sizeof(FormatSpecHeader) > 0)
  {
    if (CudaUtils::is_device_pointer(reinterpret_cast<const void *>(comp_buffer)))
    {
      const FormatSpecHeader *gpu_format_header =
        reinterpret_cast<const FormatSpecHeader *>(comp_buffer + sizeof(CommonHeader));
      CUDA_CHECK(cudaMemcpyAsync(&format_spec, gpu_format_header, sizeof(FormatSpecHeader), cudaMemcpyDefault, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    else
    {
      CUDA_CHECK(cudaStreamSynchronize(stream));
      format_spec = *reinterpret_cast<const FormatSpecHeader *>(comp_buffer + sizeof(CommonHeader));
    }
  }

  auto [compress_opts, decompress_opts] = format_opts_fn(format_spec);

  return std::make_shared<ManagerType>(
    cpu_common_header.uncomp_chunk_size,
    compress_opts,
    decompress_opts,
    stream,
    checksum_policy
  );
}

void get_common_header(const uint8_t *comp_buffer, cudaStream_t stream, CommonHeader *cpu_common_header)
{
  if (CudaUtils::is_host_pointer(reinterpret_cast<const void *>(comp_buffer)))
  {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    *cpu_common_header = *reinterpret_cast<const CommonHeader *>(comp_buffer);
  }
  else
  {
    CUDA_CHECK(cudaMemcpyAsync(cpu_common_header, comp_buffer, sizeof(CommonHeader), cudaMemcpyDefault, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
}

nvcompFormatType_t get_compression_format(const uint8_t *comp_buffer, cudaStream_t stream)
{
  CommonHeader cpu_common_header;
  get_common_header(comp_buffer, stream, &cpu_common_header);
  return cpu_common_header.format;
}

std::shared_ptr<nvcompManagerBase> create_manager(
  const uint8_t *comp_buffer,
  cudaStream_t stream,
  ChecksumPolicy checksum_policy,
  nvcompDecompressBackend_t backend,
  bool use_de_sort
)
{
  CommonHeader cpu_common_header;
  get_common_header(comp_buffer, stream, &cpu_common_header);

  switch (cpu_common_header.format)
  {
    case nvcompFormatType_t::LZ4: {
      auto opts_fn = [backend, use_de_sort](auto format_spec) {
        const uint8_t msb = format_spec.bytes[3];
        nvcompBitshuffleMode_t mode = NVCOMP_BITSHUFFLE_NONE;
        if ((msb & 0x80) && !(msb & 0x40))
        {
          mode = NVCOMP_BITSHUFFLE_LSB_FIRST;
        }
        else if ((msb & 0x40) && !(msb & 0x80))
        {
          mode = NVCOMP_BITSHUFFLE_MSB_FIRST;
        }
        // Clear the bitshuffle mask byte before getting the data type.
        auto data_type_spec = format_spec;
        data_type_spec.bytes[3] = 0;
        const nvcompType_t data_type = data_type_spec.data_type;
        nvcompBatchedLZ4CompressOpts_t compress_opts = {data_type, mode, {0}};
        nvcompBatchedLZ4DecompressOpts_t decompress_opts = {backend, use_de_sort ? 1 : 0, data_type, mode, {0}};
        return std::make_pair(compress_opts, decompress_opts);
      };

      return do_create_manager<LZ4FormatSpecHeader, LZ4Manager>(
        comp_buffer,
        checksum_policy,
        opts_fn,
        cpu_common_header,
        stream
      );
    }
    case nvcompFormatType_t::Gzip: {
      auto opts_fn = [backend, use_de_sort](auto format_opts) {
        return std::make_pair(
          nvcompBatchedGzipCompressOpts_t{format_opts.algorithm, {0}},
          nvcompBatchedGzipDecompressOpts_t{backend, NVCOMP_GZIP_DECOMPRESS_ALGORITHM_NAIVE, use_de_sort ? 1 : 0, {0}}
        );
      };
      return do_create_manager<GzipFormatSpecHeader, GzipManager>(
        comp_buffer,
        checksum_policy,
        opts_fn,
        cpu_common_header,
        stream
      );
    }
    case nvcompFormatType_t::Snappy: {
      auto opts_fn = [backend, use_de_sort](auto) {
        return std::make_pair(
          nvcompBatchedSnappyCompressOpts_t{{0}},
          nvcompBatchedSnappyDecompressOpts_t{backend, use_de_sort ? 1 : 0, {0}}
        );
      };
      return do_create_manager<SnappyFormatSpecHeader, SnappyManager>(
        comp_buffer,
        checksum_policy,
        opts_fn,
        cpu_common_header,
        stream
      );
    }
    case nvcompFormatType_t::GDeflate: {
      auto opts_fn = [backend](auto format_spec) {
        return std::make_pair(
          nvcompBatchedGdeflateCompressOpts_t{format_spec.algorithm, {0}},
          nvcompBatchedGdeflateDecompressOpts_t{backend, {0}}
        );
      };
      return do_create_manager<GdeflateFormatSpecHeader, GdeflateManager>(
        comp_buffer,
        checksum_policy,
        opts_fn,
        cpu_common_header,
        stream
      );
    }
    case nvcompFormatType_t::Deflate: {
      auto opts_fn = [backend, use_de_sort](auto format_spec) {
        return std::make_pair(
          nvcompBatchedDeflateCompressOpts_t{format_spec.algorithm, {0}},
          nvcompBatchedDeflateDecompressOpts_t{backend, use_de_sort ? 1 : 0, {0}}
        );
      };
      return do_create_manager<DeflateFormatSpecHeader, DeflateManager>(
        comp_buffer,
        checksum_policy,
        opts_fn,
        cpu_common_header,
        stream
      );
    }
    case nvcompFormatType_t::Bitcomp: {
      auto opts_fn = [backend](auto format_spec) {
        return std::make_pair(
          nvcompBatchedBitcompCompressOpts_t{
            format_spec.algorithm,
            format_spec.data_type,
            format_spec.delta,
            format_spec.mode,
            {0}
          },
          nvcompBatchedBitcompDecompressOpts_t{backend, {0}}
        );
      };
      return do_create_manager<BitcompFormatSpecHeader, BitcompManager>(
        comp_buffer,
        checksum_policy,
        opts_fn,
        cpu_common_header,
        stream
      );
    }
    case nvcompFormatType_t::ANS: {
      // The format spec header in nvCOMP <= 5.2 was unused, and set to zero. Zero in > 5.2 (before major version bump),
      // indicates default subchunk count and data type. (works with any buffer)
      auto opts_fn = [backend](auto format_spec) {
        const uint8_t max_sub_chunk_count = format_spec.get_max_sub_chunk_count();
        const nvcompType_t data_type = format_spec.get_data_type();
        nvcompBatchedANSCompressOpts_t comp_opts = nvcompBatchedANSCompressDefaultOpts;
        comp_opts.max_sub_chunk_count = max_sub_chunk_count;
        comp_opts.data_type = data_type;
        nvcompBatchedANSDecompressOpts_t decomp_opts = nvcompBatchedANSDecompressDefaultOpts;
        decomp_opts.backend = backend;
        decomp_opts.max_sub_chunk_count = max_sub_chunk_count;
        decomp_opts.data_type = data_type;
        return std::make_pair(comp_opts, decomp_opts);
      };
      return do_create_manager<ANSFormatSpecHeader, ANSManager>(
        comp_buffer,
        checksum_policy,
        opts_fn,
        cpu_common_header,
        stream
      );
    }
    case nvcompFormatType_t::Cascaded: {
      auto opts_fn = [backend](auto format_spec) {
        return std::make_pair(
          nvcompBatchedCascadedCompressOpts_t{
            format_spec.internal_chunk_bytes,
            format_spec.data_type,
            format_spec.num_RLEs,
            format_spec.num_deltas,
            format_spec.use_bp,
            {0}
          },
          nvcompBatchedCascadedDecompressOpts_t{backend, {0}}
        );
      };
      return do_create_manager<CascadedFormatSpecHeader, CascadedManager>(
        comp_buffer,
        checksum_policy,
        opts_fn,
        cpu_common_header,
        stream
      );
    }
    case nvcompFormatType_t::Zstd: {
      auto opts_fn = [backend](auto) {
        return std::make_pair(nvcompBatchedZstdCompressOpts_t{{0}}, nvcompBatchedZstdDecompressOpts_t{backend, {0}});
      };
      return do_create_manager<ZstdFormatSpecHeader, ZstdManager>(
        comp_buffer,
        checksum_policy,
        opts_fn,
        cpu_common_header,
        stream
      );
    }
    case nvcompFormatType_t::NotSupportedError:
      [[fallthrough]];
    default:
      throw NVCompException(
        nvcompErrorInvalidValue,
        "The used compression format is not recognized by this version of "
        "nvCOMP."
      );
      break;
  }

  return nullptr;
}

} // namespace nvcomp
