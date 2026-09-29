#pragma once
#include <cuda_runtime.h>

#include "nvcomp.h"
#include "nvcomp/ans.h"
#include "nvcomp/bitcomp.h"
#include "nvcomp/cascaded.h"
#include "nvcomp/deflate.h"
#include "nvcomp/gdeflate.h"
#include "nvcomp/gzip.h"
#include "nvcomp/lz4.h"
#include "nvcomp/shared_types.h"
#include "nvcomp/snappy.h"
#include "nvcomp/zstd.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

nvcompStatus_t nvcompBatchedANSDecompressAsyncEx(
  const void *const *device_compressed_chunk_ptrs,
  const size_t *device_compressed_chunk_bytes,
  const size_t *device_uncompressed_buffer_bytes,
  size_t *device_uncompressed_chunk_bytes,
  size_t num_chunks,
  void *const device_temp_ptr,
  size_t temp_bytes,
  void *const *device_uncompressed_chunk_ptrs,
  nvcompStatus_t *device_statuses,
  nvcompBatchedANSCompressOpts_t compress_opts, // not used
  nvcompBatchedANSDecompressOpts_t decompress_opts,
  cudaStream_t stream,
  const void *const *host_comp_chunk_buffers
);

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
  const void *const *host_comp_chunk_buffers
);

nvcompStatus_t nvcompBatchedCascadedDecompressAsyncEx(
  const void *const *device_compressed_chunk_ptrs,
  const size_t *device_compressed_chunk_bytes,
  const size_t *device_uncompressed_buffer_bytes,
  size_t *device_uncompressed_chunk_bytes,
  size_t num_chunks,
  void *const device_temp_ptr, // not used
  size_t temp_bytes, // not used
  void *const *device_uncompressed_chunk_ptrs,
  nvcompStatus_t *device_statuses,
  nvcompBatchedCascadedCompressOpts_t compress_opts, // not used
  nvcompBatchedCascadedDecompressOpts_t decompress_opts,
  cudaStream_t stream,
  const void *const *host_comp_chunk_buffers
);

nvcompStatus_t nvcompBatchedDeflateDecompressAsyncEx(
  const void *const *device_compressed_chunk_ptrs,
  const size_t *device_compressed_chunk_bytes,
  const size_t *device_uncompressed_buffer_bytes,
  size_t *device_uncompressed_chunk_bytes,
  size_t num_chunks,
  void *const device_temp_ptr,
  size_t temp_bytes,
  void *const *device_uncompressed_chunk_ptrs,
  nvcompStatus_t *device_statuses,
  nvcompBatchedDeflateCompressOpts_t compress_opts, // not used
  nvcompBatchedDeflateDecompressOpts_t decompress_opts,
  cudaStream_t stream,
  const void *const *host_comp_chunk_buffers
);

nvcompStatus_t nvcompBatchedGdeflateDecompressAsyncEx(
  const void *const *device_compressed_chunk_ptrs,
  const size_t *device_compressed_chunk_bytes,
  const size_t *device_uncompressed_buffer_bytes,
  size_t *device_uncompressed_chunk_bytes,
  size_t num_chunks,
  void *const device_temp_ptr,
  size_t temp_bytes,
  void *const *device_uncompressed_chunk_ptrs,
  nvcompStatus_t *device_statuses,
  nvcompBatchedGdeflateCompressOpts_t compress_opts, // not used
  nvcompBatchedGdeflateDecompressOpts_t decompress_opts,
  cudaStream_t stream,
  const void *const *host_comp_chunk_buffers
);

nvcompStatus_t nvcompBatchedGzipDecompressAsyncEx(
  const void *const *device_compressed_chunk_ptrs,
  const size_t *device_compressed_chunk_bytes,
  const size_t *device_uncompressed_buffer_bytes,
  size_t *device_uncompressed_chunk_bytes,
  size_t num_chunks,
  void *const device_temp_ptr,
  size_t temp_bytes,
  void *const *device_uncompressed_chunk_ptrs,
  nvcompStatus_t *device_statuses,
  nvcompBatchedGzipCompressOpts_t compress_opts, // not used
  nvcompBatchedGzipDecompressOpts_t decompress_opts,
  cudaStream_t stream,
  const void *const *host_comp_chunk_buffers
);

nvcompStatus_t nvcompBatchedLZ4DecompressAsyncEx(
  const void *const *device_compressed_chunk_ptrs,
  const size_t *device_compressed_chunk_bytes,
  const size_t *device_uncompressed_buffer_bytes,
  size_t *device_uncompressed_chunk_bytes,
  size_t num_chunks,
  void *const device_temp_ptr,
  size_t temp_bytes,
  void *const *device_uncompressed_chunk_ptrs,
  nvcompStatus_t *device_statuses,
  nvcompBatchedLZ4CompressOpts_t compress_opts, // not used
  nvcompBatchedLZ4DecompressOpts_t decompress_opts,
  cudaStream_t stream,
  const void *const *host_comp_chunk_buffers
);

nvcompStatus_t nvcompBatchedSnappyDecompressAsyncEx(
  const void *const *device_compressed_chunk_ptrs,
  const size_t *device_compressed_chunk_bytes,
  const size_t *device_uncompressed_buffer_bytes,
  size_t *device_uncompressed_chunk_bytes,
  size_t num_chunks,
  void *const device_temp_ptr,
  size_t temp_bytes,
  void *const *device_uncompressed_chunk_ptrs,
  nvcompStatus_t *device_statuses,
  nvcompBatchedSnappyCompressOpts_t compress_opts, // not used
  nvcompBatchedSnappyDecompressOpts_t decompress_opts,
  cudaStream_t stream,
  const void *const *host_comp_chunk_buffers
);

nvcompStatus_t nvcompBatchedZstdDecompressAsyncEx(
  const void *const *device_compressed_chunk_ptrs,
  const size_t *device_compressed_chunk_bytes,
  const size_t *device_uncompressed_buffer_bytes,
  size_t *device_uncompressed_chunk_bytes,
  size_t num_chunks,
  void *const device_temp_ptr,
  size_t temp_bytes,
  void *const *device_uncompressed_chunk_ptrs,
  nvcompStatus_t *device_statuses,
  nvcompBatchedZstdCompressOpts_t compress_opts, // not used
  nvcompBatchedZstdDecompressOpts_t decompress_opts,
  cudaStream_t stream,
  const void *const *host_comp_chunk_buffers
);

#ifdef __cplusplus
}
#endif
