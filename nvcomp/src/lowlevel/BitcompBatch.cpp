/*
 * Copyright (c) 2021-2026, NVIDIA CORPORATION. All rights reserved.
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

#include <cuda/std/cmath>

#include <cassert>
#include <limits>

#include "bitcomp/bitcomp_private.h"
#include "bitcomp/utilities.h"
#include "Check.h"
#include "common.h"
#include "Logging.h"
#include "lowlevel/nvcomp_private.h"
#include "nvcomp.h"
#include "nvcomp/bitcomp.h"
#include "nvcomp/native/bitcomp.h"

namespace
{

constexpr const char *bitcomp_compress_opts_format_str = "{{algo={:d}, data_type={:d}, mode={:d}, delta={:g}}}";
constexpr const char *bitcomp_decompress_opts_format_str = "{{backend={:d}}}";

#define BITCOMP_WITH_TRAILING_COMPRESS_OPTS_ARGS(callable, opts, ...)                                                  \
  callable(__VA_ARGS__, int(opts.algorithm), int(opts.data_type), int(opts.mode), opts.delta)

#define BITCOMP_WITH_TRAILING_DECOMPRESS_OPTS_ARGS(callable, opts, ...) callable(__VA_ARGS__, int(opts.backend))

#define BITCOMP_LOG_WITH_COMPRESS_OPTS(log_func, opts, ...)                                                            \
  BITCOMP_WITH_TRAILING_COMPRESS_OPTS_ARGS(log_func, opts, __func__, __VA_ARGS__, bitcomp_compress_opts_format_str)

#define BITCOMP_LOG_WITH_DECOMPRESS_OPTS(log_func, opts, ...)                                                          \
  BITCOMP_WITH_TRAILING_DECOMPRESS_OPTS_ARGS(log_func, opts, __func__, __VA_ARGS__, bitcomp_decompress_opts_format_str)

#define BITCOMP_CHECK_COMPRESS_OPTS(opts) NVCOMP_WRAP_CHECK_FUNC(check_compress_opts, opts)

#define BITCOMP_CHECK_DECOMPRESS_OPTS(opts) NVCOMP_WRAP_CHECK_FUNC(check_decompress_opts, opts)

double round_effective_delta(const double supplied_delta)
{
  return bitcomp::utilities::zeroMantissaBits(supplied_delta);
}

template <typename T>
T convert_effective_delta(const double supplied_delta)
{
  return static_cast<T>(round_effective_delta(supplied_delta));
}

template <>
half convert_effective_delta<half>(const double supplied_delta)
{
  return __double2half(round_effective_delta(supplied_delta));
}

template <typename T>
bool valid_lossy_delta(const double supplied_delta)
{
  if (!(supplied_delta > 0.0 && cuda::std::isnormal(supplied_delta)))
  {
    return false;
  }

  // Round the original double down before converting it. The resulting power
  // of two is exact in T whenever it is representable as a normal value.
  return cuda::std::isnormal(convert_effective_delta<T>(supplied_delta));
}

nvcompStatus_t check_compress_opts(const nvcompBatchedBitcompCompressOpts_t &opts)
{
  const int algo_type = opts.algorithm;
  const bool valid_algorithm = algo_type == int(BITCOMP_DEFAULT_ALGO) || algo_type == int(BITCOMP_SPARSE_ALGO);
  const bool integral_type = opts.data_type == NVCOMP_TYPE_BITS ||
                             (opts.data_type >= NVCOMP_TYPE_CHAR && opts.data_type <= NVCOMP_TYPE_ULONGLONG);
  const bool floating_point_type = opts.data_type == NVCOMP_TYPE_FLOAT16 || opts.data_type == NVCOMP_TYPE_FLOAT32 ||
                                   opts.data_type == NVCOMP_TYPE_FLOAT64;
  const bool lossy_mode = opts.mode == BITCOMP_LOSSY_FP_TO_SIGNED || opts.mode == BITCOMP_LOSSY_FP_TO_UNSIGNED;
  const bool valid_mode_and_type = (opts.mode == BITCOMP_LOSSLESS && (integral_type || floating_point_type)) ||
                                   (lossy_mode && floating_point_type);

  // Round the supplied delta (with type `double`) down before converting it to the input
  // precision. This preserves the caller's error bound across power-of-two
  // conversion boundaries to the smaller data types that have less precision. Reject values that are invalid before rounding or
  // are not normal in the selected input precision after conversion.
  bool valid_delta = true;
  if (lossy_mode)
  {
    switch (opts.data_type)
    {
      case NVCOMP_TYPE_FLOAT16:
        valid_delta = valid_lossy_delta<half>(opts.delta);
        break;
      case NVCOMP_TYPE_FLOAT32:
        valid_delta = valid_lossy_delta<float>(opts.delta);
        break;
      case NVCOMP_TYPE_FLOAT64:
        valid_delta = valid_lossy_delta<double>(opts.delta);
        break;
      default:
        valid_delta = false;
        break;
    }
  }

  const bool supported = valid_algorithm && valid_mode_and_type && valid_delta;
  if (supported)
  {
    return nvcompSuccess;
  }

  BITCOMP_WITH_TRAILING_COMPRESS_OPTS_ARGS(
    LOG_ERROR,
    opts,
    std::string("Unsupported options: ") + bitcomp_compress_opts_format_str
  );
  return nvcompErrorNotSupported;
}

nvcompStatus_t check_decompress_opts(const nvcompBatchedBitcompDecompressOpts_t &opts)
{
  const bool supported = opts.backend == NVCOMP_DECOMPRESS_BACKEND_DEFAULT ||
                         opts.backend == NVCOMP_DECOMPRESS_BACKEND_CUDA;
  if (supported)
  {
    return nvcompSuccess;
  }

  // Backend error
  switch (opts.backend)
  {
    case NVCOMP_DECOMPRESS_BACKEND_DEFAULT:
    case NVCOMP_DECOMPRESS_BACKEND_CUDA:
      break;
    case NVCOMP_DECOMPRESS_BACKEND_HARDWARE:
      LOG_ERROR("Hardware backend is not available for Bitcomp.");
      return nvcompErrorInvalidValue;
    default:
      LOG_ERROR(
        "Decompress backend invalid. Choose either NVCOMP_DECOMPRESS_BACKEND_DEFAULT or NVCOMP_DECOMPRESS_BACKEND_CUDA."
      );
      return nvcompErrorInvalidValue;
  }

  // Other errors
  BITCOMP_WITH_TRAILING_DECOMPRESS_OPTS_ARGS(
    LOG_ERROR,
    opts,
    std::string("Unsupported options: ") + bitcomp_decompress_opts_format_str
  );
  return nvcompErrorNotSupported;
}

bitcompIntFormat_t bitcomp_int_format_from_opts(const nvcompBatchedBitcompCompressOpts_t &opts)
{
  if (opts.mode == BITCOMP_LOSSY_FP_TO_SIGNED || opts.data_type == NVCOMP_TYPE_CHAR ||
      opts.data_type == NVCOMP_TYPE_SHORT || opts.data_type == NVCOMP_TYPE_INT ||
      opts.data_type == NVCOMP_TYPE_LONGLONG)
  {
    return BITCOMP_CUSTOM_INTEGER;
  }
  return BITCOMP_DEFAULT_FORMAT;
}

} // namespace

static_assert(sizeof(nvcompBatchedBitcompCompressOpts_t) == 64);
static_assert(sizeof(nvcompBatchedBitcompDecompressOpts_t) == 64);

nvcompStatus_t nvcompBatchedBitcompCompressGetMaxOutputChunkSize(
  size_t max_uncompressed_chunk_bytes,
  nvcompBatchedBitcompCompressOpts_t format_opts,
  size_t *max_compressed_chunk_bytes
)
{
  BITCOMP_LOG_WITH_COMPRESS_OPTS(
    nvcomp::logBatchedCompressGetMaxOutputChunkSize,
    format_opts,
    max_uncompressed_chunk_bytes,
    max_compressed_chunk_bytes
  );
  BITCOMP_CHECK_COMPRESS_OPTS(format_opts);

  NVCOMP_CHECK_NOT_NULL(max_compressed_chunk_bytes);
  NVCOMP_CHECK_ALIGNMENT(max_compressed_chunk_bytes);

  NVCOMP_CHECK_CHUNK_SIZE(max_uncompressed_chunk_bytes, nvcompBitcompCompressionMaxAllowedChunkSize);

  *max_compressed_chunk_bytes = bitcompMaxBuflen(max_uncompressed_chunk_bytes);
  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedBitcompCompressAsync(
  const void *const *device_uncompressed_chunk_ptrs,
  const size_t *device_uncompressed_chunk_bytes,
  size_t max_uncompressed_chunk_bytes, // unused, except for logging
  size_t num_chunks,
  void *device_temp_ptr, // unused, except for logging
  size_t temp_bytes, // unused, except for logging
  void *const *device_compressed_chunk_ptrs,
  size_t *device_compressed_chunk_bytes,
  const nvcompBatchedBitcompCompressOpts_t format_opts,
  nvcompStatus_t *device_statuses, // optional
  cudaStream_t stream
)
{
  BITCOMP_LOG_WITH_COMPRESS_OPTS(
    nvcomp::logBatchedCompressAsync,
    format_opts,
    device_uncompressed_chunk_ptrs,
    device_uncompressed_chunk_bytes,
    max_uncompressed_chunk_bytes,
    num_chunks,
    device_temp_ptr,
    temp_bytes,
    device_compressed_chunk_ptrs,
    device_compressed_chunk_bytes,
    device_statuses,
    stream
  );

  BITCOMP_CHECK_COMPRESS_OPTS(format_opts);

  NVCOMP_CHECK_NOT_NULL(device_uncompressed_chunk_ptrs);
  NVCOMP_CHECK_NOT_NULL(device_uncompressed_chunk_bytes);
  NVCOMP_CHECK_NOT_NULL(device_compressed_chunk_ptrs);
  NVCOMP_CHECK_NOT_NULL(device_compressed_chunk_bytes);

  // Check device pointer alignment
  NVCOMP_CHECK_ALIGNMENT(device_uncompressed_chunk_ptrs);
  NVCOMP_CHECK_ALIGNMENT(device_uncompressed_chunk_bytes);
  NVCOMP_CHECK_ALIGNMENT(device_compressed_chunk_ptrs);
  NVCOMP_CHECK_ALIGNMENT(device_compressed_chunk_bytes);
  NVCOMP_CHECK_ALIGNMENT(device_statuses);

  // Batch size is used as grid size, which CUDA limits to 2^31 - 1
  NVCOMP_CHECK_BATCH_SIZE(num_chunks, std::numeric_limits<int>::max());

  const auto algorithm = static_cast<bitcompAlgorithm_t>(format_opts.algorithm);

#define BITCOMP_LAUNCH_BATCH_ENCODER(T, data_type, mode, int_format, quantization_delta)                               \
  do                                                                                                                   \
  {                                                                                                                    \
    const nvcompStatus_t status = bitcomp::launchBatchEncoder<T, data_type, mode, int_format>(                         \
      device_uncompressed_chunk_ptrs,                                                                                  \
      device_compressed_chunk_ptrs,                                                                                    \
      algorithm,                                                                                                       \
      device_uncompressed_chunk_bytes,                                                                                 \
      device_compressed_chunk_bytes,                                                                                   \
      num_chunks,                                                                                                      \
      quantization_delta,                                                                                              \
      stream                                                                                                           \
    );                                                                                                                 \
    if (status != nvcompSuccess)                                                                                       \
    {                                                                                                                  \
      return status;                                                                                                   \
    }                                                                                                                  \
  } while (0)

// The lossy FP encoders differ only by the C++ type and its Bitcomp data-type
// tag: BITCOMP_LOSSY_FP_TO_SIGNED is always paired with BITCOMP_CUSTOM_INTEGER
// and BITCOMP_LOSSY_FP_TO_UNSIGNED with BITCOMP_DEFAULT_FORMAT. The mode and
// integer format are template arguments (compile-time), so the sign is still
// selected with a runtime branch; this macro factors out that fixed pairing so
// each floating-point type below is a single line.
#define BITCOMP_LAUNCH_LOSSY_BATCH_ENCODER(T, BITCOMP_DATA_TYPE)                                                       \
  do                                                                                                                   \
  {                                                                                                                    \
    if (format_opts.mode == BITCOMP_LOSSY_FP_TO_SIGNED)                                                                \
    {                                                                                                                  \
      BITCOMP_LAUNCH_BATCH_ENCODER(                                                                                    \
        T,                                                                                                             \
        BITCOMP_DATA_TYPE,                                                                                             \
        BITCOMP_LOSSY_FP_TO_SIGNED,                                                                                    \
        BITCOMP_CUSTOM_INTEGER,                                                                                        \
        convert_effective_delta<T>(format_opts.delta)                                                                  \
      );                                                                                                               \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
      BITCOMP_LAUNCH_BATCH_ENCODER(                                                                                    \
        T,                                                                                                             \
        BITCOMP_DATA_TYPE,                                                                                             \
        BITCOMP_LOSSY_FP_TO_UNSIGNED,                                                                                  \
        BITCOMP_DEFAULT_FORMAT,                                                                                        \
        convert_effective_delta<T>(format_opts.delta)                                                                  \
      );                                                                                                               \
    }                                                                                                                  \
  } while (0)

  if (format_opts.mode == BITCOMP_LOSSLESS)
  {
    switch (format_opts.data_type)
    {
      case NVCOMP_TYPE_BITS:
      case NVCOMP_TYPE_UCHAR:
        BITCOMP_LAUNCH_BATCH_ENCODER(
          unsigned char,
          BITCOMP_UNSIGNED_8BIT,
          BITCOMP_LOSSLESS,
          BITCOMP_DEFAULT_FORMAT,
          static_cast<unsigned char>(0)
        );
        break;
      case NVCOMP_TYPE_CHAR:
        BITCOMP_LAUNCH_BATCH_ENCODER(
          char,
          BITCOMP_SIGNED_8BIT,
          BITCOMP_LOSSLESS,
          BITCOMP_CUSTOM_INTEGER,
          static_cast<char>(0)
        );
        break;
      case NVCOMP_TYPE_FLOAT16:
      case NVCOMP_TYPE_USHORT:
        BITCOMP_LAUNCH_BATCH_ENCODER(
          unsigned short,
          BITCOMP_UNSIGNED_16BIT,
          BITCOMP_LOSSLESS,
          BITCOMP_DEFAULT_FORMAT,
          static_cast<unsigned short>(0)
        );
        break;
      case NVCOMP_TYPE_SHORT:
        BITCOMP_LAUNCH_BATCH_ENCODER(
          short,
          BITCOMP_SIGNED_16BIT,
          BITCOMP_LOSSLESS,
          BITCOMP_CUSTOM_INTEGER,
          static_cast<short>(0)
        );
        break;
      case NVCOMP_TYPE_FLOAT32:
      case NVCOMP_TYPE_UINT:
        BITCOMP_LAUNCH_BATCH_ENCODER(
          bitcomp::uint,
          BITCOMP_UNSIGNED_32BIT,
          BITCOMP_LOSSLESS,
          BITCOMP_DEFAULT_FORMAT,
          static_cast<bitcomp::uint>(0)
        );
        break;
      case NVCOMP_TYPE_INT:
        BITCOMP_LAUNCH_BATCH_ENCODER(int, BITCOMP_SIGNED_32BIT, BITCOMP_LOSSLESS, BITCOMP_CUSTOM_INTEGER, 0);
        break;
      case NVCOMP_TYPE_FLOAT64:
      case NVCOMP_TYPE_ULONGLONG:
        BITCOMP_LAUNCH_BATCH_ENCODER(
          unsigned long long,
          BITCOMP_UNSIGNED_64BIT,
          BITCOMP_LOSSLESS,
          BITCOMP_DEFAULT_FORMAT,
          0ULL
        );
        break;
      case NVCOMP_TYPE_LONGLONG:
        BITCOMP_LAUNCH_BATCH_ENCODER(long long, BITCOMP_SIGNED_64BIT, BITCOMP_LOSSLESS, BITCOMP_CUSTOM_INTEGER, 0LL);
        break;
      default:
        return nvcompErrorNotSupported;
    }
  }
  else
  {
    switch (format_opts.data_type)
    {
      case NVCOMP_TYPE_FLOAT16:
        BITCOMP_LAUNCH_LOSSY_BATCH_ENCODER(half, BITCOMP_FP16_DATA);
        break;
      case NVCOMP_TYPE_FLOAT32:
        BITCOMP_LAUNCH_LOSSY_BATCH_ENCODER(float, BITCOMP_FP32_DATA);
        break;
      case NVCOMP_TYPE_FLOAT64:
        BITCOMP_LAUNCH_LOSSY_BATCH_ENCODER(double, BITCOMP_FP64_DATA);
        break;
      default:
        return nvcompErrorNotSupported;
    }
  }

#undef BITCOMP_LAUNCH_LOSSY_BATCH_ENCODER
#undef BITCOMP_LAUNCH_BATCH_ENCODER

  // mark compression successful
  try
  {
    nvcomp::try_clear_device_statuses(num_chunks, device_statuses, stream);
  }
  catch ([[maybe_unused]] const std::exception &e)
  {
    return nvcompErrorCudaError;
  }
  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedBitcompDecompressGetRequiredAlignments(
  nvcompBatchedBitcompDecompressOpts_t decompress_opts,
  nvcompAlignmentRequirements_t *alignment_requirements
)
{
  BITCOMP_LOG_WITH_DECOMPRESS_OPTS(
    nvcomp::logBatchedDecompressGetRequiredAlignments,
    decompress_opts,
    alignment_requirements
  );
  BITCOMP_CHECK_DECOMPRESS_OPTS(decompress_opts);
  NVCOMP_CHECK_NOT_NULL(alignment_requirements);

  // The global header and block-offset table are read as uint64 values.
  // Output depends on the opts.data_type used during compression and its
  // alignment is the same as the alignment of the compression input. Since we
  // do not have access to compression options, the worst-case scenario is used,
  // which is 8 bytes.
  // The temp buffer is used for batchCompInfo_t.
  alignment_requirements->input = 8;
  alignment_requirements->output = 8;
  alignment_requirements->temp = alignof(batchCompInfo_t);

  static_assert(
    ((nvcompBitcompRequiredDecompressionAlignment) & (nvcompBitcompRequiredDecompressionAlignment - 1)) == 0,
    "Minimum decompression alignment is invalid."
  );
  assert(
    std::max(std::max(alignment_requirements->input, alignment_requirements->output), alignment_requirements->temp) <=
    nvcompBitcompRequiredDecompressionAlignment
  );

  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedBitcompDecompressAsync(
  const void *const *device_compressed_chunk_ptrs,
  const size_t *device_compressed_chunk_bytes, // unused, except for logging
  const size_t *device_uncompressed_buffer_bytes,
  size_t *device_uncompressed_chunk_bytes, // optional, to be used with device_status
  size_t num_chunks,
  void *const device_temp_ptr,
  size_t temp_bytes, // unused, except for logging
  void *const *device_uncompressed_chunk_ptrs,
  nvcompBatchedBitcompDecompressOpts_t decompress_opts,
  nvcompStatus_t *device_statuses, // optional
  cudaStream_t stream
)
{
  nvcomp::logBatchedDecompressAsync(
    __func__,
    device_compressed_chunk_ptrs,
    device_compressed_chunk_bytes,
    device_uncompressed_buffer_bytes,
    device_uncompressed_chunk_bytes,
    num_chunks,
    device_temp_ptr,
    temp_bytes,
    device_uncompressed_chunk_ptrs,
    decompress_opts.backend,
    device_statuses,
    stream
  );

  NVCOMP_CHECK_NOT_NULL(device_compressed_chunk_ptrs);
  NVCOMP_CHECK_NOT_NULL(device_uncompressed_buffer_bytes);
  NVCOMP_CHECK_NOT_NULL(device_uncompressed_chunk_ptrs);
  NVCOMP_CHECK_NOT_NULL(device_temp_ptr);
  if ((device_uncompressed_chunk_bytes == nullptr) != (device_statuses == nullptr))
  {
    LOG_ERROR("Both device_uncompressed_chunk_bytes and device_statuses should be valid or nullptr");
    return nvcompErrorInvalidValue;
  }
  BITCOMP_CHECK_DECOMPRESS_OPTS(decompress_opts);

  // Check device pointer alignment
  NVCOMP_CHECK_ALIGNMENT(device_compressed_chunk_ptrs);
  NVCOMP_CHECK_ALIGNMENT(device_uncompressed_buffer_bytes);
  NVCOMP_CHECK_ALIGNMENT(device_uncompressed_chunk_bytes);
  NVCOMP_CHECK_ALIGNMENT(device_uncompressed_chunk_ptrs);
  NVCOMP_CHECK_ALIGNMENT(device_statuses);

  if (device_temp_ptr != nullptr)
  {
    NVCOMP_CHECK_ALIGNMENT(device_temp_ptr, alignof(batchCompInfo_t));
  }

  // Batch size is used as grid size, which CUDA limits to 2^31 - 1
  NVCOMP_CHECK_BATCH_SIZE(num_chunks, std::numeric_limits<int>::max());

  if (num_chunks == 0)
  {
    return nvcompSuccess;
  }

  LOG_INFO("Launching SM decompression");

  batchCompInfo_t comp_info;
  try
  {
    const nvcompStatus_t status = bitcomp::utilities::bitcompGetBatchCompressedInfo(
      device_compressed_chunk_ptrs,
      num_chunks,
      comp_info,
      stream,
      device_temp_ptr
    );
    if (status != nvcompSuccess)
    {
      return status;
    }
  }
  catch ([[maybe_unused]] const std::exception &e)
  {
    return nvcompErrorCudaError;
  }

  return bitcomp::launchBatchDecoder(
    device_compressed_chunk_ptrs,
    device_uncompressed_chunk_ptrs,
    device_uncompressed_buffer_bytes,
    device_statuses,
    device_uncompressed_chunk_bytes,
    comp_info,
    num_chunks,
    stream
  );
}

nvcompStatus_t nvcompBatchedBitcompDecompressAsyncEx(
  const void *const *device_compressed_chunk_ptrs,
  const size_t *device_compressed_chunk_bytes, // not used
  const size_t *device_uncompressed_buffer_bytes,
  size_t *device_uncompressed_chunk_bytes,
  size_t num_chunks,
  void *const device_temp_ptr, // not used
  size_t temp_bytes, // not used
  void *const *device_uncompressed_chunk_ptrs,
  nvcompStatus_t *device_statuses,
  nvcompBatchedBitcompCompressOpts_t compress_opts,
  nvcompBatchedBitcompDecompressOpts_t decompress_opts,
  cudaStream_t stream,
  [[maybe_unused]] const void *const *host_comp_chunk_buffers
)
{
  nvcomp::logBatchedDecompressAsync(
    __func__,
    device_compressed_chunk_ptrs,
    device_compressed_chunk_bytes,
    device_uncompressed_buffer_bytes,
    device_uncompressed_chunk_bytes,
    num_chunks,
    device_temp_ptr,
    temp_bytes,
    device_uncompressed_chunk_ptrs,
    decompress_opts.backend,
    device_statuses,
    stream
  );

  NVCOMP_CHECK_NOT_NULL(device_compressed_chunk_ptrs);
  NVCOMP_CHECK_NOT_NULL(device_uncompressed_buffer_bytes);
  NVCOMP_CHECK_NOT_NULL(device_uncompressed_chunk_ptrs);
  NVCOMP_CHECK_NOT_NULL(device_temp_ptr);
  if ((device_uncompressed_chunk_bytes == nullptr) != (device_statuses == nullptr))
  {
    LOG_ERROR("Both device_uncompressed_chunk_bytes and device_statuses should be valid or nullptr");
    return nvcompErrorInvalidValue;
  }
  BITCOMP_CHECK_COMPRESS_OPTS(compress_opts);
  BITCOMP_CHECK_DECOMPRESS_OPTS(decompress_opts);

  // Check device pointer alignment
  NVCOMP_CHECK_ALIGNMENT(device_compressed_chunk_ptrs);
  NVCOMP_CHECK_ALIGNMENT(device_uncompressed_buffer_bytes);
  NVCOMP_CHECK_ALIGNMENT(device_uncompressed_chunk_bytes);
  NVCOMP_CHECK_ALIGNMENT(device_uncompressed_chunk_ptrs);
  NVCOMP_CHECK_ALIGNMENT(device_statuses);
  NVCOMP_CHECK_ALIGNMENT(device_statuses);

  LOG_INFO("Launching SM decompression");

  nvcompType_t data_type = compress_opts.data_type;
  if (compress_opts.mode == BITCOMP_LOSSLESS)
  {
    switch (data_type)
    {
      case NVCOMP_TYPE_FLOAT16:
        data_type = NVCOMP_TYPE_USHORT;
        break;
      case NVCOMP_TYPE_FLOAT32:
        data_type = NVCOMP_TYPE_UINT;
        break;
      case NVCOMP_TYPE_FLOAT64:
        data_type = NVCOMP_TYPE_ULONGLONG;
        break;
      default:
        break;
    }
  }

  const batchCompInfo_t comp_info{
    bitcomp::utilities::nvcomp_to_bitcomp_data_type(data_type),
    compress_opts.mode,
    static_cast<bitcompAlgorithm_t>(compress_opts.algorithm),
    bitcomp_int_format_from_opts(compress_opts),
    0
  };
  return bitcomp::launchBatchDecoder(
    device_compressed_chunk_ptrs,
    device_uncompressed_chunk_ptrs,
    device_uncompressed_buffer_bytes,
    device_statuses,
    device_uncompressed_chunk_bytes,
    comp_info,
    num_chunks,
    stream
  );
}

nvcompStatus_t nvcompBatchedBitcompGetDecompressSizeAsync(
  const void *const *device_compressed_chunk_ptrs,
  const size_t *device_compressed_chunk_bytes, // unused, except for logging
  size_t *device_uncompressed_chunk_bytes,
  size_t num_chunks,
  cudaStream_t stream
)
{
  nvcomp::logBatchedGetDecompressSizeAsync(
    __func__,
    device_compressed_chunk_ptrs,
    device_compressed_chunk_bytes,
    device_uncompressed_chunk_bytes,
    num_chunks,
    stream
  );

  NVCOMP_CHECK_NOT_NULL(device_compressed_chunk_ptrs);
  NVCOMP_CHECK_NOT_NULL(device_uncompressed_chunk_bytes);

  // Check device pointer alignment
  NVCOMP_CHECK_ALIGNMENT(device_compressed_chunk_ptrs, sizeof(void *));
  NVCOMP_CHECK_ALIGNMENT(device_uncompressed_chunk_bytes, sizeof(size_t));

  return bitcompBatchGetUncompressedSizesAsync(
    device_compressed_chunk_ptrs,
    device_uncompressed_chunk_bytes,
    num_chunks,
    stream
  );
}

nvcompStatus_t nvcompBatchedBitcompCompressGetRequiredAlignments(
  nvcompBatchedBitcompCompressOpts_t format_opts,
  nvcompAlignmentRequirements_t *alignment_requirements
)
{
  BITCOMP_LOG_WITH_COMPRESS_OPTS(nvcomp::logBatchedCompressGetRequiredAlignments, format_opts, alignment_requirements);
  BITCOMP_CHECK_COMPRESS_OPTS(format_opts);
  NVCOMP_CHECK_NOT_NULL(alignment_requirements);

  // Cast to appropriate type in loadInputToShared.
  alignment_requirements->input = nvcomp::sizeOfnvcompType(format_opts.data_type);
  // Cast to uint64_t* in bitcomp::header::getScalarAddress.
  alignment_requirements->output = 8;
  alignment_requirements->temp = 1; // Not used.

  // TODO(mpayrits): We need to decide if we are satisfied with the worst-case
  // decompression output (and potentially temp) alignment for algorithms where
  // these depend on the compression settings used.
  // We could redesign the interface to have functions like
  // nvcompBatched<alg>DecompressGetRequiredAlignments(
  //   T1* align_reqs, const T2* opts, const void* const* device_compressed_chunk_ptrs, T3 stream);
  // with the following logic (dccp is an alias for device_compressed_chunk_ptrs):
  // * opts == nullptr && dccp == nullptr: Return worst-case alignment.
  // * opts != nullptr && dccp != nullptr: Error.
  // * opts != nullptr && dccp == nullptr: Interpret *opts as the options used
  //   for compression, return appropriate alignment.
  // * opts == nullptr && dccp != nullptr: Parse dccp for information on
  //   alignment. Note that dccp[i] themselves must be aligned w.r.t. the worst
  //   case input alignment since that cannot be known in advance in this case.
  //   This case is different from nvcompBatched<alg>GetDecompressSizeAsync or
  //   nvcompBatched<alg>DecompressGetTempSize in that data is parsed on the
  //   device and transferred to the host. Documentation must stress that
  //   synchronization has to be done correctly.

  static_assert(
    ((nvcompBitcompRequiredCompressionAlignment) & (nvcompBitcompRequiredCompressionAlignment - 1)) == 0,
    "Minimum compression alignment is invalid."
  );
  assert(
    std::max(std::max(alignment_requirements->input, alignment_requirements->output), alignment_requirements->temp) <=
    nvcompBitcompRequiredCompressionAlignment
  );

  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedBitcompCompressGetTempSizeAsync(
  size_t num_chunks, // unused, except for logging
  size_t max_uncompressed_chunk_bytes, // unused, except for logging
  nvcompBatchedBitcompCompressOpts_t format_opts,
  size_t *temp_bytes,
  size_t max_total_uncompressed_bytes
) // unused, except for logging
{
  nvcomp::logBatchedCompressGetTempSizeAsync(
    __func__,
    num_chunks,
    max_uncompressed_chunk_bytes,
    temp_bytes,
    max_total_uncompressed_bytes,
    bitcomp_compress_opts_format_str,
    static_cast<int>(format_opts.algorithm),
    static_cast<int>(format_opts.data_type),
    static_cast<int>(format_opts.mode),
    format_opts.delta
  );

  BITCOMP_CHECK_COMPRESS_OPTS(format_opts);

  NVCOMP_CHECK_NOT_NULL(temp_bytes);
  NVCOMP_CHECK_ALIGNMENT(temp_bytes);

  *temp_bytes = 0;

  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedBitcompCompressGetTempSizeSync(
  [[maybe_unused]] const void *const *const device_uncompressed_chunk_ptrs,
  [[maybe_unused]] const size_t *const device_uncompressed_chunk_bytes,
  size_t num_chunks,
  size_t max_uncompressed_chunk_bytes,
  nvcompBatchedBitcompCompressOpts_t compress_opts,
  size_t *temp_bytes,
  size_t max_total_uncompressed_bytes,
  [[maybe_unused]] cudaStream_t stream
)
{
  return nvcompBatchedBitcompCompressGetTempSizeAsync(
    num_chunks,
    max_uncompressed_chunk_bytes,
    compress_opts,
    temp_bytes,
    max_total_uncompressed_bytes
  );
}

nvcompStatus_t nvcompBatchedBitcompDecompressGetTempSizeAsync(
  size_t num_chunks,
  size_t max_uncompressed_chunk_bytes,
  nvcompBatchedBitcompDecompressOpts_t decompress_opts,
  size_t *temp_bytes,
  [[maybe_unused]] size_t max_total_uncompressed_bytes
)
{
  BITCOMP_LOG_WITH_DECOMPRESS_OPTS(
    nvcomp::logBatchedDecompressGetTempSizeAsync,
    decompress_opts,
    num_chunks,
    max_uncompressed_chunk_bytes,
    temp_bytes,
    max_total_uncompressed_bytes
  );

  BITCOMP_CHECK_DECOMPRESS_OPTS(decompress_opts);
  NVCOMP_CHECK_NOT_NULL(temp_bytes);
  NVCOMP_CHECK_ALIGNMENT(temp_bytes);
  NVCOMP_CHECK_CHUNK_SIZE(max_uncompressed_chunk_bytes, nvcompBitcompDecompressionMaxAllowedChunkSize);

  *temp_bytes = sizeof(batchCompInfo_t);
  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedBitcompDecompressGetTempSizeSync(
  [[maybe_unused]] const void *const *const device_compressed_chunk_ptrs,
  [[maybe_unused]] const size_t *const device_compressed_chunk_bytes,
  size_t num_chunks,
  size_t max_uncompressed_chunk_bytes,
  size_t *temp_bytes,
  size_t max_total_uncompressed_bytes,
  nvcompBatchedBitcompDecompressOpts_t decompress_opts,
  nvcompStatus_t *device_statuses,
  cudaStream_t stream
)
{
  NVCOMP_CHECK_ALIGNMENT(device_statuses);

  nvcompStatus_t result = nvcomp::try_clear_device_statuses<true>(num_chunks, device_statuses, stream);
  if (result != nvcompSuccess)
  {
    return result;
  }

  return nvcompBatchedBitcompDecompressGetTempSizeAsync(
    num_chunks,
    max_uncompressed_chunk_bytes,
    decompress_opts,
    temp_bytes,
    max_total_uncompressed_bytes
  );
}
