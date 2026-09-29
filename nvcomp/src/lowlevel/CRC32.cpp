/*
 * Copyright (c) 2017-2021, NVIDIA CORPORATION. All rights reserved.
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

#include "nvcomp/crc32.h"

#include <sstream>

#include "Check.h"
#include "common.h"
#include "crc/cuCRC32.h"
#include "CudaUtils.h"
#include "Logging.h"
#include "nvcomp.h"
#include "nvcomp.hpp"
#include "StringUtils.h"
#include "type_macros.h"

using ::nvcomp::CharArray;
using ::nvcomp::CudaUtils;

namespace
{

constexpr auto crc32_spec_format_str =
  CharArray("{{poly=0x{:08x}, init=0x{:08x}, ref_in={:s}, ref_out={:s}, xorout=0x{:08x}}}");

constexpr auto crc32_kernel_conf_format_str =
  CharArray("{{kernel_kind={:s}, bytes_per_read={:d}, blocks_per_msg={:d}}}");

constexpr auto crc32_options_format_str = CharArray("{{spec=") + crc32_spec_format_str + CharArray(", kernel_conf=") +
                                          crc32_kernel_conf_format_str + CharArray("}}");

constexpr const char *boolToStr(bool b) { return b ? "true" : "false"; }

std::string kernelKindToStr(nvcompCRC32KernelKind_t kind)
{
  switch (kind)
  {
    case nvcompCRC32WarpKernel:
      return "warp";
    case nvcompCRC32BlockKernel:
      return "block";
    default:
      return "unknown (" + std::to_string(static_cast<int>(kind)) + ")";
  }
}

std::string segmentKindToStr(nvcompCRC32SegmentKind_t kind)
{
  switch (kind)
  {
    case nvcompCRC32OnlySegment:
      return "only";
    case nvcompCRC32FirstSegment:
      return "first";
    case nvcompCRC32MidSegment:
      return "mid";
    case nvcompCRC32LastSegment:
      return "last";
    default:
      return "unknown (" + std::to_string(static_cast<int>(kind)) + ")";
  }
}

// TODO: Add log macros that take sink functions as arguments to cuLibLogger.
// This allows doing two things at once without any additional macro usage -
// deconstructing structures for passing to formatting functions, as well as not
// evaluating said structures in the first place if the log level is too low to
// log. If we had that, we could use the deconstructing functions below.
// Until then, we resort to macros resolving to bare comma-separated arguments.
// These are a serious compromise and should be kept to a minimum in the rest of
// the codebase.

// template <typename F, typename... Args>
// auto with_trailing_spec_args(F&& f, const nvcompCRC32Spec_t& spec, Args&&... args)
// {
//   return std::forward<F>(f)(std::forward<Args>(args)..., spec.poly, spec.init,
//     boolToStr(spec.ref_in), boolToStr(spec.ref_out), spec.xorout);
// }
//
// template <typename F, typename... Args>
// auto with_trailing_kernel_conf_args(F&& f, const nvcompCRC32KernelConf_t& kernel_conf, Args&&... args)
// {
//   return std::forward<F>(f)(std::forward<Args>(args)..., kernelKindToStr(kernel_conf.kernel_kind),
//     kernel_conf.bytes_per_read, kernel_conf.blocks_per_msg);
// }
//
// template <typename F, typename... Args>
// auto with_trailing_options_args(F&& f, const nvcompBatchedCRC32Opts_t& opts, Args&&... args)
// {
//   return with_trailing_spec_args([&](auto&&... specArgs) {
//     return with_trailing_kernel_conf_args(std::forward<F>(f), opts.kernel_conf,
//       std::forward<Args>(args)..., NVCOMP_FWD(specArgs)...);
//   }, opts.spec);
// }

#define CRC32_SPEC_FORMAT_ARGS(spec) spec.poly, spec.init, boolToStr(spec.ref_in), boolToStr(spec.ref_out), spec.xorout

#define CRC32_KERNEL_CONF_FORMAT_ARGS(kernel_conf)                                                                     \
  kernelKindToStr(kernel_conf.kernel_kind), kernel_conf.bytes_per_read, kernel_conf.blocks_per_msg

#define CRC32_OPTIONS_FORMAT_ARGS(opts)                                                                                \
  CRC32_SPEC_FORMAT_ARGS(opts.spec), CRC32_KERNEL_CONF_FORMAT_ARGS(opts.kernel_conf)

// There is no check_spec function because nvcompCRC32Spec_t is always valid.

nvcompStatus_t check_kernel_conf(nvcompCRC32KernelConf_t kernel_conf)
{
  const bool validKernelKind = kernel_conf.kernel_kind == nvcompCRC32WarpKernel ||
                               kernel_conf.kernel_kind == nvcompCRC32BlockKernel;

  // Check that kernel_conf.bytes_per_read is a power of 2 and between 4 and 2048.
  const bool bytesPerReadInRange = kernel_conf.bytes_per_read >= 4 && kernel_conf.bytes_per_read <= 2048;
  const bool validBytesPerRead = ::nvcomp::is_power_of_two(std::size_t(kernel_conf.bytes_per_read)) &&
                                 bytesPerReadInRange;

  const bool validBlocksPerMsg = kernel_conf.kernel_kind != nvcompCRC32BlockKernel ||
                                 (kernel_conf.blocks_per_msg >= 1 && kernel_conf.blocks_per_msg < (1 << 16));

  if (!validKernelKind || !validBytesPerRead || !validBlocksPerMsg)
  {
    constexpr auto format_str = CharArray("Unsupported kernel configuration: ") + crc32_kernel_conf_format_str;
    LOG_ERROR(fmt::string_view(format_str.data), CRC32_KERNEL_CONF_FORMAT_ARGS(kernel_conf));
    return nvcompErrorNotSupported;
  }
  return nvcompSuccess;
}

#define CRC32_CHECK_KERNEL_CONF(kernel_conf) NVCOMP_WRAP_CHECK_FUNC(check_kernel_conf, kernel_conf)

nvcompStatus_t check_segment_kind(nvcompCRC32SegmentKind_t segment_kind)
{
  const bool validSegmentKind = segment_kind == nvcompCRC32OnlySegment || segment_kind == nvcompCRC32FirstSegment ||
                                segment_kind == nvcompCRC32MidSegment || segment_kind == nvcompCRC32LastSegment;

  if (!validSegmentKind)
  {
    LOG_ERROR("Unsupported segment kind: {:s}", segmentKindToStr(segment_kind));
    return nvcompErrorNotSupported;
  }
  return nvcompSuccess;
}

#define CRC32_CHECK_SEGMENT_KIND(segment_kind) NVCOMP_WRAP_CHECK_FUNC(check_segment_kind, segment_kind)

std::string getCuCRC32ErrorString(int err)
{
  switch (err)
  {
    case CUCRC32_SUCCESS:
      return "Success";
    case CUCRC32_ERROR_INVALID_PARAMS:
      return "Invalid parameters";
    case CUCRC32_ERROR_INVALID_KERNEL:
      return "Invalid kernel";
    case CUCRC32_ERROR_INVALID_READBYTES:
      return "Invalid bytes per read";
    case CUCRC32_ERROR_INVALID_BLKXMSG:
      return "Invalid blocks per message";
    default:
      assert(false);
      return "Unknown error";
  }
}

void checkCuCRC32(int status)
{
  if (status != CUCRC32_SUCCESS)
  {
    throw nvcomp::NVCompException(nvcompErrorInternal, "CRC32 calculation error: " + getCuCRC32ErrorString(status));
  }
}

// Convert nvcompCRC32Spec_t to crcSpec_t for use with the cuCRC32 library.
crcSpec_t getCuCRC32Spec(const nvcompCRC32Spec_t &nvcomp_spec)
{
  return {nvcomp_spec.poly, nvcomp_spec.init, int(nvcomp_spec.ref_in), int(nvcomp_spec.ref_out), nvcomp_spec.xorout};
}

// Convert a cuCRC32 configuration to a nvcompCRC32KernelConf_t.
nvcompCRC32KernelConf_t getNvcompKernelConf(uint32_t cucrc_conf)
{
  int kernel_kind = 0;
  int bytes_per_read = 0;
  int blocks_per_msg = 0;
  checkCuCRC32(cuCRC32ConfToParam(cucrc_conf, &kernel_kind, &bytes_per_read, &blocks_per_msg));

  return {nvcompCRC32KernelKind_t(kernel_kind), bytes_per_read, blocks_per_msg, {}};
}

// Convert nvcompCRC32KernelConf_t to a cuCRC32 configuration value.
uint32_t getCuCRC32Conf(const nvcompCRC32KernelConf_t &kernel_conf)
{
  uint32_t cucrc_conf = 0;
  checkCuCRC32(
    cuCRC32ParamToConf(int(kernel_conf.kernel_kind), kernel_conf.bytes_per_read, kernel_conf.blocks_per_msg, &cucrc_conf)
  );
  return cucrc_conf;
}

// Create a CRC context for use with the cuCRC32 library.
crcCtx_t getCuCRC32Ctx(const nvcompCRC32Spec_t &nvcomp_spec, uint32_t *output_buffer)
{
  // Convert the NVCOMP spec to a cuCRC32 spec
  crcSpec_t cucrc_spec = getCuCRC32Spec(nvcomp_spec);

  // Create and return CRC context
  return {output_buffer, cucrc_spec};
}

} // namespace

static_assert(sizeof(nvcompCRC32Spec_t) == 32);
static_assert(sizeof(nvcompCRC32KernelConf_t) == 32);
static_assert(sizeof(nvcompBatchedCRC32Opts_t) == 128);

nvcompStatus_t nvcompBatchedCRC32Async(
  const void *const *device_input_chunk_ptrs,
  const size_t *device_input_chunk_bytes,
  size_t num_chunks,
  uint32_t *device_crc32_ptr,
  nvcompBatchedCRC32Opts_t opts,
  nvcompCRC32SegmentKind_t segment_kind,
  nvcompStatus_t *device_statuses,
  cudaStream_t stream
)
{
  constexpr auto format_str = CharArray(
                                "device_input_chunk_ptrs={:#x}, device_input_chunk_bytes={:#x}, num_chunks={:d}, "
                                "device_crc32_ptr={:#x}, options=("
                              ) +
                              crc32_options_format_str +
                              CharArray("), segment_kind={:s}, device_statuses={:#x}, stream={:#x}");

  LOG_API(
    fmt::string_view(format_str.data),
    reinterpret_cast<uintptr_t>(device_input_chunk_ptrs),
    reinterpret_cast<uintptr_t>(device_input_chunk_bytes),
    num_chunks,
    reinterpret_cast<uintptr_t>(device_crc32_ptr),
    CRC32_OPTIONS_FORMAT_ARGS(opts),
    segmentKindToStr(segment_kind),
    reinterpret_cast<uintptr_t>(device_statuses),
    reinterpret_cast<uintptr_t>(stream)
  );

  // Error-check inputs.

  CRC32_CHECK_KERNEL_CONF(opts.kernel_conf);
  CRC32_CHECK_SEGMENT_KIND(segment_kind);

  if (device_input_chunk_ptrs != nullptr)
  {
    NVCOMP_CHECK_NOT_NULL(device_input_chunk_bytes);
  }
  else if (segment_kind != nvcompCRC32LastSegment)
  {
    LOG_ERROR("device_input_chunk_ptrs must not be nullptr if segment_kind is not nvcompCRC32LastSegment.");
    return ::nvcompStatus_t::nvcompErrorInvalidValue;
  }

  NVCOMP_CHECK_NOT_NULL(device_crc32_ptr);

  NVCOMP_CHECK_ALIGNMENT(device_input_chunk_ptrs);
  NVCOMP_CHECK_ALIGNMENT(device_input_chunk_bytes);
  NVCOMP_CHECK_ALIGNMENT(device_crc32_ptr);
  NVCOMP_CHECK_ALIGNMENT(device_statuses);

  try
  {
    const nvcompCRC32Spec_t &spec = opts.spec;
    const nvcompCRC32KernelConf_t &conf = opts.kernel_conf;

    // Create CRC context
    crcCtx_t curc_ctx = getCuCRC32Ctx(spec, device_crc32_ptr);

    // Mark CRC32 calculation successful
    nvcomp::try_clear_device_statuses(num_chunks, device_statuses, stream);

    if (segment_kind == nvcompCRC32FirstSegment || segment_kind == nvcompCRC32OnlySegment)
    {
      checkCuCRC32(cuCRC32Beg(&curc_ctx, uint32_t(num_chunks), stream));
    }

    // TODO: Anything better than a reinterpret_cast?

    if (device_input_chunk_ptrs != nullptr)
    {
      uint32_t cucrc_conf = getCuCRC32Conf(conf);
      checkCuCRC32(cuCRC32Add(
        &curc_ctx,
        cucrc_conf,
        uint32_t(num_chunks),
        reinterpret_cast<const unsigned long long *>(device_input_chunk_bytes),
        reinterpret_cast<const unsigned char *const *>(device_input_chunk_ptrs),
        stream
      ));
    }

    if (segment_kind == nvcompCRC32LastSegment || segment_kind == nvcompCRC32OnlySegment)
    {
      checkCuCRC32(cuCRC32End(&curc_ctx, uint32_t(num_chunks), stream));
    }
  }
  catch (const std::exception &e)
  {
    LOG_ERROR("{}", e.what());
    return nvcomp::Check::exception_to_error(e, "nvcompBatchedCRC32Async()");
  }

  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedCRC32GetHeuristicConf(
  const size_t *device_input_chunk_bytes,
  size_t num_chunks,
  nvcompCRC32KernelConf_t *kernel_conf,
  size_t max_input_chunk_bytes,
  cudaStream_t stream
)
{
  LOG_API(
    "device_input_chunk_bytes={:#x}, num_chunks={:d}, kernel_conf={:#x}"
    "max_input_chunk_bytes={:d}, stream={:#x}",
    reinterpret_cast<uintptr_t>(device_input_chunk_bytes),
    num_chunks,
    reinterpret_cast<uintptr_t>(kernel_conf),
    max_input_chunk_bytes,
    reinterpret_cast<uintptr_t>(stream)
  );

  // Error-check inputs.

  if (device_input_chunk_bytes == nvcompCRC32IgnoredInputChunkBytes)
  {
    if (max_input_chunk_bytes == nvcompCRC32DeducedMaxInputChunkBytes)
    {
      LOG_ERROR("max_input_chunk_bytes must be set if device_input_chunk_bytes is unused.");
      return ::nvcompStatus_t::nvcompErrorInvalidValue;
    }
  }
  else if (max_input_chunk_bytes != nvcompCRC32DeducedMaxInputChunkBytes)
  {
    LOG_ERROR("max_input_chunk_bytes must not be set if device_input_chunk_bytes is used.");
    return ::nvcompStatus_t::nvcompErrorInvalidValue;
  }

  NVCOMP_CHECK_NOT_NULL(kernel_conf);

  NVCOMP_CHECK_ALIGNMENT(device_input_chunk_bytes);
  NVCOMP_CHECK_ALIGNMENT(kernel_conf);

  try
  {
    // Create a temporary CRC context for the heuristic using the default CRC32 spec
    crcCtx_t curc_ctx = getCuCRC32Ctx(nvcompCRC32, nullptr);

    // Call the underlying heuristic function
    uint32_t cucrc_conf = 0;
    checkCuCRC32(cuCRC32ConfHeur(
      &curc_ctx,
      static_cast<uint32_t>(num_chunks),
      reinterpret_cast<const unsigned long long *>(device_input_chunk_bytes),
      &cucrc_conf,
      max_input_chunk_bytes,
      stream
    ));

    // Convert the configuration to our format
    *kernel_conf = getNvcompKernelConf(cucrc_conf);
  }
  catch (const std::exception &e)
  {
    LOG_ERROR("{}", e.what());
    return nvcomp::Check::exception_to_error(e, "nvcompBatchedCRC32GetHeuristicConf()");
  }

  return nvcompSuccess;
}

nvcompStatus_t nvcompBatchedCRC32SearchConf(
  const void *const *device_input_chunk_ptrs,
  const size_t *device_input_chunk_bytes,
  size_t num_chunks,
  uint32_t *device_crc32_ptr,
  nvcompCRC32Spec_t spec,
  nvcompCRC32KernelConf_t *kernel_conf,
  cudaStream_t stream
)
{
  constexpr auto format_str = CharArray(
                                "device_input_chunk_ptrs={:#x}, device_input_chunk_bytes={:#x}, num_chunks={:d}, "
                                "device_crc32_ptr={:#x}, spec="
                              ) +
                              crc32_spec_format_str + CharArray(", kernel_conf={:#x}, stream={:#x}");

  LOG_API(
    fmt::string_view(format_str.data),
    reinterpret_cast<uintptr_t>(device_input_chunk_ptrs),
    reinterpret_cast<uintptr_t>(device_input_chunk_bytes),
    num_chunks,
    reinterpret_cast<uintptr_t>(device_crc32_ptr),
    CRC32_SPEC_FORMAT_ARGS(spec),
    reinterpret_cast<uintptr_t>(kernel_conf),
    reinterpret_cast<uintptr_t>(stream)
  );

  // Error-check inputs.
  NVCOMP_CHECK_NOT_NULL(device_input_chunk_ptrs);
  NVCOMP_CHECK_NOT_NULL(device_input_chunk_bytes);
  NVCOMP_CHECK_NOT_NULL(device_crc32_ptr);
  NVCOMP_CHECK_NOT_NULL(kernel_conf);

  NVCOMP_CHECK_ALIGNMENT(device_input_chunk_ptrs);
  NVCOMP_CHECK_ALIGNMENT(device_input_chunk_bytes);
  NVCOMP_CHECK_ALIGNMENT(device_crc32_ptr);
  NVCOMP_CHECK_ALIGNMENT(kernel_conf);

  try
  {
    // Create CRC context
    crcCtx_t curc_ctx = getCuCRC32Ctx(spec, device_crc32_ptr);

    uint32_t cucrc_conf = 0;
    checkCuCRC32(cuCRC32ConfSearch(
      &curc_ctx,
      uint32_t(num_chunks),
      reinterpret_cast<const unsigned long long *>(device_input_chunk_bytes),
      reinterpret_cast<const unsigned char *const *>(device_input_chunk_ptrs),
      &cucrc_conf,
      stream
    ));

    *kernel_conf = getNvcompKernelConf(cucrc_conf);
  }
  catch (const std::exception &e)
  {
    LOG_ERROR("{}", e.what());
    return nvcomp::Check::exception_to_error(e, "nvcompBatchedCRC32SearchConf()");
  }

  return nvcompSuccess;
}
