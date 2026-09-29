/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
*/

#include <cub/device/device_radix_sort.cuh>

#include <unordered_map>

#include "exception.hpp"
#include "HWDecompress.hpp"

namespace nvcomp
{

struct DecompSupportVal
{
  // Deflate, Gzip, and Snappy support
  bool device_supports_decomp;
  bool device_supports_decomp_lz4;
  size_t max_de_supported_size;

  explicit DecompSupportVal(int device_id)
      : device_supports_decomp(false)
      , device_supports_decomp_lz4(false)
      , max_de_supported_size(0)
  {
    // First check driver
    if (!CTK_DE_BASE_SUPPORTED(CudaDriver::get_latest_supported_cuda_version()))
    {
      return;
    }

    int decompressSupportMask = 0;
    CUresult res = CudaDriver::cuDeviceGetAttribute(
      &decompressSupportMask,
      CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_ALGORITHM_MASK,
      device_id
    );
    device_supports_decomp = static_cast<bool>(decompressSupportMask);

    if (CTK_DE_LZ4_SUPPORTED(CudaDriver::get_latest_supported_cuda_version()))
    {
      device_supports_decomp_lz4 = device_supports_decomp;
    }
    if (res != CUDA_SUCCESS)
    {
      throw NVCompException(nvcompErrorCudaError, "nvCOMP error: Unable to get decompress attribute for device");
    }
    if (not device_supports_decomp)
    {
      return;
    }

    int max_supported_size = 0;
    res = CudaDriver::cuDeviceGetAttribute(
      &max_supported_size,
      CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_MAXIMUM_LENGTH,
      device_id
    );
    if (res != CUDA_SUCCESS)
    {
      throw NVCompException(nvcompErrorCudaError, "nvCOMP error: Unable to get decompress attribute for device");
    }
    max_de_supported_size = static_cast<size_t>(max_supported_size);
  }
};

const DecompSupportVal &get_decomp_support(int device_id)
{
  // Per-device static caching
  static std::unordered_map<int, DecompSupportVal> device_decomp_support = []() {
    std::unordered_map<int, DecompSupportVal> device_map;
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    device_map.reserve(device_count);
    for (int id = 0; id < device_count; ++id)
    {
      device_map.emplace(id, id);
    }
    return device_map;
  }();

  return device_decomp_support.at(device_id);
}

bool hw_supports_decomp_engine(nvcompFormatType_t comp_format, int device_id)
{
  auto &decomp_support = get_decomp_support(device_id);

  switch (comp_format)
  {
    case nvcompFormatType_t::Deflate:
    case nvcompFormatType_t::Gzip:
    case nvcompFormatType_t::Snappy:
      return decomp_support.device_supports_decomp;
    case nvcompFormatType_t::LZ4:
      return decomp_support.device_supports_decomp_lz4;
    default:
      return false;
  }
}

size_t get_decompress_engine_max_size(int device_id)
{
  const DecompSupportVal &decomp_support = get_decomp_support(device_id);
  return decomp_support.max_de_supported_size;
}

size_t get_sort_scratch_req(size_t num_chunks, [[maybe_unused]] size_t max_uncompressed_chunk_bytes)
{
  size_t temp_storage_bytes = 0;

  int max_bits = 32;
  int *dummy_keys{};
  int *dummy_keys2{};
  int *dummy_vals{};
  int *dummy_vals2{};
  cub::DeviceRadixSort::SortPairsDescending(
    nullptr,
    temp_storage_bytes,
    dummy_keys,
    dummy_keys2,
    dummy_vals,
    dummy_vals2,
    num_chunks,
    0,
    max_bits
  );
  temp_storage_bytes += 4 * num_chunks * sizeof(int);
  temp_storage_bytes += sizeof(int) - 1; // need to align to integer size
  return temp_storage_bytes;
}

} // namespace nvcomp
