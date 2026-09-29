#include "common.h"
#include "CRC32.hpp"
#include "device_guard.h"
#include "highlevel/CompressionConfigs.hpp"
#include "HWDecompress.hpp"
#include "LZ77_decomp.cuh"
#include "nvcomp/nvcompManager.hpp"
#include "nvcomp_common_deps/hlif_shared_types.hpp"

#include <cooperative_groups.h>

namespace nvcomp
{
template <typename FormatSpecHeader>
__global__ void setup_comp_llif_buffers(
  FormatSpecHeader format_spec,
  FormatSpecHeader *format_spec_header,
  uint8_t *scratch_comp_buffer,
  uint8_t *true_comp_buffer,
  const uint8_t *input_uncomp_buffer,
  const uint8_t **uncomp_buffers,
  uint8_t **comp_buffers,
  size_t *uncomp_sizes,
  const size_t total_uncomp_buffer_size,
  const size_t num_chunks,
  const size_t max_comp_chunk_size,
  const size_t max_uncomp_chunk_size,
  CommonHeader *common_header
)
{
  const size_t chunk_id = threadIdx.x + blockDim.x * blockIdx.x;
  if (chunk_id >= num_chunks)
  {
    return;
  }

  if (chunk_id == 0)
  {

    comp_buffers[chunk_id] = true_comp_buffer;
    common_header->comp_data_size = 0;
    common_header->include_per_chunk_comp_buffer_checksums = 0;
    common_header->include_per_chunk_decomp_buffer_checksums = 0;
    *format_spec_header = format_spec;
  }
  else
  {
    comp_buffers[chunk_id] = scratch_comp_buffer + (chunk_id - 1) * max_comp_chunk_size;
  }

  uncomp_buffers[chunk_id] = input_uncomp_buffer + chunk_id * max_uncomp_chunk_size;
  size_t uncomp_chunk_size = max_uncomp_chunk_size;
  if (chunk_id == num_chunks - 1)
  {
    const size_t rem_chunk_size = total_uncomp_buffer_size % max_uncomp_chunk_size;
    if (rem_chunk_size > 0)
    {
      uncomp_chunk_size = rem_chunk_size;
    }
  }

  uncomp_sizes[chunk_id] = uncomp_chunk_size;
}

template __global__ void setup_comp_llif_buffers<ANSFormatSpecHeader>(
  ANSFormatSpecHeader,
  ANSFormatSpecHeader *,
  uint8_t *,
  uint8_t *,
  const uint8_t *,
  const uint8_t **,
  uint8_t **,
  size_t *,
  const size_t,
  const size_t,
  const size_t,
  const size_t,
  CommonHeader *
);

template __global__ void setup_comp_llif_buffers<BitcompFormatSpecHeader>(
  BitcompFormatSpecHeader,
  BitcompFormatSpecHeader *,
  uint8_t *,
  uint8_t *,
  const uint8_t *,
  const uint8_t **,
  uint8_t **,
  size_t *,
  const size_t,
  const size_t,
  const size_t,
  const size_t,
  CommonHeader *
);

template __global__ void setup_comp_llif_buffers<CascadedFormatSpecHeader>(
  CascadedFormatSpecHeader,
  CascadedFormatSpecHeader *,
  uint8_t *,
  uint8_t *,
  const uint8_t *,
  const uint8_t **,
  uint8_t **,
  size_t *,
  const size_t,
  const size_t,
  const size_t,
  const size_t,
  CommonHeader *
);

template __global__ void setup_comp_llif_buffers<DeflateFormatSpecHeader>(
  DeflateFormatSpecHeader,
  DeflateFormatSpecHeader *,
  uint8_t *,
  uint8_t *,
  const uint8_t *,
  const uint8_t **,
  uint8_t **,
  size_t *,
  const size_t,
  const size_t,
  const size_t,
  const size_t,
  CommonHeader *
);

template __global__ void setup_comp_llif_buffers<GdeflateFormatSpecHeader>(
  GdeflateFormatSpecHeader,
  GdeflateFormatSpecHeader *,
  uint8_t *,
  uint8_t *,
  const uint8_t *,
  const uint8_t **,
  uint8_t **,
  size_t *,
  const size_t,
  const size_t,
  const size_t,
  const size_t,
  CommonHeader *
);

template __global__ void setup_comp_llif_buffers<GzipFormatSpecHeader>(
  GzipFormatSpecHeader,
  GzipFormatSpecHeader *,
  uint8_t *,
  uint8_t *,
  const uint8_t *,
  const uint8_t **,
  uint8_t **,
  size_t *,
  const size_t,
  const size_t,
  const size_t,
  const size_t,
  CommonHeader *
);

template __global__ void setup_comp_llif_buffers<LZ4FormatSpecHeader>(
  LZ4FormatSpecHeader,
  LZ4FormatSpecHeader *,
  uint8_t *,
  uint8_t *,
  const uint8_t *,
  const uint8_t **,
  uint8_t **,
  size_t *,
  const size_t,
  const size_t,
  const size_t,
  const size_t,
  CommonHeader *
);

template __global__ void setup_comp_llif_buffers<SnappyFormatSpecHeader>(
  SnappyFormatSpecHeader,
  SnappyFormatSpecHeader *,
  uint8_t *,
  uint8_t *,
  const uint8_t *,
  const uint8_t **,
  uint8_t **,
  size_t *,
  const size_t,
  const size_t,
  const size_t,
  const size_t,
  CommonHeader *
);

template __global__ void setup_comp_llif_buffers<ZstdFormatSpecHeader>(
  ZstdFormatSpecHeader,
  ZstdFormatSpecHeader *,
  uint8_t *,
  uint8_t *,
  const uint8_t *,
  const uint8_t **,
  uint8_t **,
  size_t *,
  const size_t,
  const size_t,
  const size_t,
  const size_t,
  CommonHeader *
);

template <typename FormatSpecHeader>
__global__ void setup_batched_comp_llif_buffers(
  uint8_t **input_uncomp_buffers, //Array of pointers to input uncompressed buffer of each batch
  uint8_t **output_comp_buffers, //Array of pointers to output compressed buffer of each batch
  CompressionConfig *compression_configs, //Array of pointers to compression config of each batch
  size_t total_num_chunks, //Total number of chunks across all batches
  size_t batch_count, //Number of batches
  FormatSpecHeader format_spec, //Format Spec to copy for each batch
  uint8_t *scratch_comp_buffer, //Start of scratch memory where we are storing output temporarily
  const uint8_t **uncomp_chunk_buffers, //Pointer to start of each uncompressed chunk (Scratch)
  uint8_t **comp_chunk_buffers, //Pointer to start of each compressed chunk (Scratch)
  size_t *uncomp_chunk_sizes, //Pointer to size of each uncompressed chunk(Scratch)
  const size_t max_comp_chunk_size, //Maximum compressed size of each chunk
  const size_t max_uncomp_chunk_size, //Maximum uncompressed size of each chunk
  const size_t min_alignment //Strongest required buffer alignment for the format
)
{
  const size_t global_chunk_id = threadIdx.x + blockDim.x * blockIdx.x;

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

  //Compute the pointers
  size_t batch_uncomp_buffer_size = compression_configs[batch_idx].uncompressed_buffer_size;

  uint8_t *this_batch_uncomp_buffer = input_uncomp_buffers[batch_idx];
  uint8_t *this_batch_comp_buffer = output_comp_buffers[batch_idx];

  CommonHeader *this_batch_common_header = reinterpret_cast<CommonHeader *>(this_batch_comp_buffer);
  this_batch_comp_buffer += sizeof(CommonHeader);

  FormatSpecHeader *this_batch_format_header = reinterpret_cast<FormatSpecHeader *>(this_batch_comp_buffer);
  this_batch_comp_buffer += sizeof(FormatSpecHeader);

  size_t *this_batch_comp_chunk_offsets = roundUpToAlignment<size_t>(this_batch_comp_buffer);
  size_t *this_batch_comp_sizes = this_batch_comp_chunk_offsets + this_batch_num_chunks;

  this_batch_comp_buffer = reinterpret_cast<uint8_t *>(
    roundUpTo(reinterpret_cast<uintptr_t>(this_batch_comp_sizes + this_batch_num_chunks), min_alignment)
  );

  //First chunk of each batch writes directly to final output buffer, and also writes header
  if ((this_batch_chunk_id == 0))
  {
    comp_chunk_buffers[global_chunk_id] = this_batch_comp_buffer;
    this_batch_common_header->comp_data_size = 0;
    this_batch_common_header->include_per_chunk_comp_buffer_checksums = 0;
    this_batch_common_header->include_per_chunk_decomp_buffer_checksums = 0;
    *this_batch_format_header = format_spec;
  }
  else
  { //Remaining chunks write to scratch space
    comp_chunk_buffers[global_chunk_id] = scratch_comp_buffer +
                                          (global_chunk_id - (batch_idx + 1)) * max_comp_chunk_size;
  }

  uncomp_chunk_buffers[global_chunk_id] = this_batch_uncomp_buffer + this_batch_chunk_id * max_uncomp_chunk_size;
  size_t uncomp_chunk_size = max_uncomp_chunk_size;
  if (this_batch_chunk_id == this_batch_num_chunks - 1)
  {
    const size_t rem_chunk_size = batch_uncomp_buffer_size % max_uncomp_chunk_size;
    if (rem_chunk_size > 0)
    {
      uncomp_chunk_size = rem_chunk_size;
    }
  }
  uncomp_chunk_sizes[global_chunk_id] = uncomp_chunk_size;
}

template __global__ void setup_batched_comp_llif_buffers<ANSFormatSpecHeader>(
  uint8_t **,
  uint8_t **,
  CompressionConfig *,
  size_t,
  size_t,
  ANSFormatSpecHeader,
  uint8_t *,
  const uint8_t **,
  uint8_t **,
  size_t *,
  const size_t,
  const size_t,
  const size_t
);

template __global__ void setup_batched_comp_llif_buffers<BitcompFormatSpecHeader>(
  uint8_t **,
  uint8_t **,
  CompressionConfig *,
  size_t,
  size_t,
  BitcompFormatSpecHeader,
  uint8_t *,
  const uint8_t **,
  uint8_t **,
  size_t *,
  const size_t,
  const size_t,
  const size_t
);

template __global__ void setup_batched_comp_llif_buffers<CascadedFormatSpecHeader>(
  uint8_t **,
  uint8_t **,
  CompressionConfig *,
  size_t,
  size_t,
  CascadedFormatSpecHeader,
  uint8_t *,
  const uint8_t **,
  uint8_t **,
  size_t *,
  const size_t,
  const size_t,
  const size_t
);

template __global__ void setup_batched_comp_llif_buffers<DeflateFormatSpecHeader>(
  uint8_t **,
  uint8_t **,
  CompressionConfig *,
  size_t,
  size_t,
  DeflateFormatSpecHeader,
  uint8_t *,
  const uint8_t **,
  uint8_t **,
  size_t *,
  const size_t,
  const size_t,
  const size_t
);

template __global__ void setup_batched_comp_llif_buffers<GdeflateFormatSpecHeader>(
  uint8_t **,
  uint8_t **,
  CompressionConfig *,
  size_t,
  size_t,
  GdeflateFormatSpecHeader,
  uint8_t *,
  const uint8_t **,
  uint8_t **,
  size_t *,
  const size_t,
  const size_t,
  const size_t
);

template __global__ void setup_batched_comp_llif_buffers<LZ4FormatSpecHeader>(
  uint8_t **,
  uint8_t **,
  CompressionConfig *,
  size_t,
  size_t,
  LZ4FormatSpecHeader,
  uint8_t *,
  const uint8_t **,
  uint8_t **,
  size_t *,
  const size_t,
  const size_t,
  const size_t
);

template __global__ void setup_batched_comp_llif_buffers<GzipFormatSpecHeader>(
  uint8_t **,
  uint8_t **,
  CompressionConfig *,
  size_t,
  size_t,
  GzipFormatSpecHeader,
  uint8_t *,
  const uint8_t **,
  uint8_t **,
  size_t *,
  const size_t,
  const size_t,
  const size_t
);

template __global__ void setup_batched_comp_llif_buffers<SnappyFormatSpecHeader>(
  uint8_t **,
  uint8_t **,
  CompressionConfig *,
  size_t,
  size_t,
  SnappyFormatSpecHeader,
  uint8_t *,
  const uint8_t **,
  uint8_t **,
  size_t *,
  const size_t,
  const size_t,
  const size_t
);

template __global__ void setup_batched_comp_llif_buffers<ZstdFormatSpecHeader>(
  uint8_t **,
  uint8_t **,
  CompressionConfig *,
  size_t,
  size_t,
  ZstdFormatSpecHeader,
  uint8_t *,
  const uint8_t **,
  uint8_t **,
  size_t *,
  const size_t,
  const size_t,
  const size_t
);

} // namespace nvcomp