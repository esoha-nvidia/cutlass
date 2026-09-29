/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <type_traits>

#include "huffman.h"
#include "nvcomp/utils.hpp"

using namespace nvcomp;

namespace gdeflate
{

__global__ void huffman_tree(unsigned int *counts_, uint16_t *codelen, unsigned int *codes)
{
  __shared__ uint16_t left[256];
  __shared__ uint16_t right[256];
  __shared__ uint16_t depth[2 * 256];
  __shared__ uint16_t symbols[256];
  __shared__ unsigned int counts[2 * 256];

  // Load counts into shared memory
  for (unsigned int i = threadIdx.x; i < 256; i += WARP_SIZE_U)
  {
    counts[i] = counts_[i];
  }
  __syncwarp();

  // Initialize the tree builder
  WarpHuffmanTree<256, 15> wht(counts, symbols);

  // Build the Huffman tree
  wht.build(counts, left, right, depth);

  // Get code lengths
  wht.codelengths(codelen, symbols, depth);

  warp_encoder<256, 15> enc;
  enc.init(
    wht.num_symbols(),
    codelen,
    codes,
    reinterpret_cast<uint16_t *>(&counts[0]),
    reinterpret_cast<unsigned int *>(&counts[256])
  );
}

template <typename T>
__device__ T &identity_ref(T &arg)
{
  // Used to unpack an argument pack containing a single reference type.
  return arg;
}

template <bool Standard, typename... OptOutT>
__device__ void huffman_encode_tile(
  uint32_t *output,
  size_t *compressed_size,
  uint8_t *compressed_pad_bits,
  const unsigned char *input,
  const size_t n,
  [[maybe_unused]] OptOutT &...opt_out_pos
)
{
  static_assert(sizeof...(OptOutT) == size_t(Standard));
  using Twriter = std::conditional_t<Standard, warp_bitwriter_standard<uint32_t>, warp_bitwriter<uint32_t>>;

  if constexpr (Standard)
  {
    if (threadIdx.x == 0)
    {
      identity_ref(opt_out_pos...).store(0, cuda::std::memory_order_relaxed);
    }
  }

  __shared__ __align__(4) uint16_t left[gdeflate_literal_symbols];
  __shared__ __align__(4) uint16_t right[gdeflate_literal_symbols];
  __shared__ uint16_t depth[2 * gdeflate_literal_symbols];
  __shared__ uint16_t symbols[gdeflate_literal_symbols];
  __shared__ unsigned int counts[2 * gdeflate_literal_symbols];

  uint16_t *codelen = reinterpret_cast<uint16_t *>(counts);
  uint32_t *codes = reinterpret_cast<uint32_t *>(&counts[gdeflate_literal_symbols]);

  for (unsigned int i = threadIdx.x; i < gdeflate_literal_symbols; i += WARP_SIZE_U)
  {
    counts[i] = 0;
  }
  __syncwarp();

  // Compute counts of each character
  update_histogram(counts, input, n);

  // Add end of block character
  if (threadIdx.x == 0)
  {
    counts[256] = 1;
  }
  __syncwarp();

  // Initialize the tree builder
  WarpHuffmanTree<gdeflate_literal_symbols, gdeflate_max_codelen> wht(counts, symbols);

  // Build the Huffman tree
  wht.build(counts, left, right, depth);

  // Get code lengths
  // NOTE: this call does not need to use counts
  // so it is safe to alias codelen with counts
  wht.codelengths(codelen, symbols, depth);
  __syncwarp();

  // Build the canonical deflate Huffman encoder
  warp_encoder<gdeflate_literal_symbols, gdeflate_max_codelen> enc;
  enc.init(
    gdeflate_literal_symbols,
    codelen,
    codes,
    reinterpret_cast<uint16_t *>(left),
    reinterpret_cast<unsigned int *>(right)
  );

  // Section 3.2.7 of RFC 1951, the deflate specification, requires at least one
  // entry in the distance table. It also states that a single distance
  // codelength of zero means distances are not used so that is what we use.
  if (threadIdx.x == 0)
  {
    codelen[gdeflate_litlen_symbols] = 0;
  }
  __syncwarp();

  // Initialize the warp bitwriter
  Twriter bw(output);
  uint32_t bfinal = 1, btype = 2, header = 0;
  header = ((btype & 3) << 1) | (bfinal & 1);
  write_bits</*Header=*/true>(bw, header, 3, threadIdx.x == 0, opt_out_pos...); // Only thread 0 writes to its stream
  // TODO: add run length encoding of the codelengths
  pack_codelens(codelen, gdeflate_literal_symbols, 1, bw, opt_out_pos...);

  // Now encode the data
  for (unsigned int iter = 0; iter < (n + 1 + WARP_SIZE_U - 1) / WARP_SIZE_U; ++iter)
  {
    unsigned int i = threadIdx.x + iter * WARP_SIZE_U;
    bool active = (i <= n);
    uint16_t sym = i < n ? (uint16_t)input[i] : 256;
    uint16_t len = enc.get_codelen(sym);
    uint32_t code = enc.get_code(sym);
    write_bits(bw, code, len, active, opt_out_pos...);
  }

  // Write out final remainder state to the stream
  if constexpr (Standard)
  {
    const size_t total_bits = identity_ref(opt_out_pos...).load(cuda::std::memory_order_relaxed);
    const size_t bytes = roundUpDiv(total_bits, 8);
    *compressed_size = bytes;
    if (compressed_pad_bits != nullptr && threadIdx.x == 0)
    {
      *compressed_pad_bits = padBitsToByte(total_bits);
    }
  }
  else
  {
    uint32_t *compressed_stream_end = bw.finalize();
    *compressed_size = (size_t)(compressed_stream_end - output) * sizeof(uint32_t);
  }
}

// TODO: CUB BlockRadixSort limits this implementation to 1 warp per CTA
__global__ void
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900 || __CUDA_ARCH__ == 1000)
__launch_bounds__(WARP_SIZE, 24)
#else
__launch_bounds__(WARP_SIZE)
#endif // __CUDA_ARCH__
  huffman_encode(
    uint32_t *const *__restrict__ outputs,
    size_t *__restrict__ compressed_sizes,
    [[maybe_unused]] uint8_t *__restrict__ compressed_pad_bits,
    const unsigned char *const *__restrict__ inputs,
    const size_t *__restrict__ input_sizes,
    const unsigned int num_tiles_host,
    const int *device_num_tiles
  )
{
  assert(blockDim.x == WARP_SIZE_U);
  const unsigned int num_tiles = device_num_tiles ? static_cast<unsigned int>(*device_num_tiles) : num_tiles_host;
  unsigned int stride = gridDim.x * blockDim.y;
  for (unsigned int bid = blockIdx.x * blockDim.y + threadIdx.y; bid < num_tiles; bid += stride)
  {
    huffman_encode_tile</*Standard=*/false>(outputs[bid], &compressed_sizes[bid], nullptr, inputs[bid], input_sizes[bid]);
  }
}

// TODO: CUB BlockRadixSort limits this implementation to 1 warp per CTA
__global__ void
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900 || __CUDA_ARCH__ == 1000)
__launch_bounds__(WARP_SIZE, 24)
#else
__launch_bounds__(WARP_SIZE)
#endif // __CUDA_ARCH__
  huffman_encode_standard(
    uint32_t *const *__restrict__ outputs,
    size_t *__restrict__ compressed_sizes,
    uint8_t *__restrict__ compressed_pad_bits,
    const unsigned char *const *__restrict__ inputs,
    const size_t *__restrict__ input_sizes,
    const unsigned int num_tiles_host,
    const int *device_num_tiles
  )
{
  assert(blockDim.x == WARP_SIZE_U);

  __shared__ cuda::atomic<size_t, cuda::thread_scope_block> output_pos;

  const unsigned int num_tiles = device_num_tiles ? static_cast<unsigned int>(*device_num_tiles) : num_tiles_host;
  unsigned int stride = gridDim.x * blockDim.y;
  for (unsigned int bid = blockIdx.x * blockDim.y + threadIdx.y; bid < num_tiles; bid += stride)
  {
    __syncwarp(); // unnecessary, but benefits the perf on some test cases.
    huffman_encode_tile</*Standard=*/true>(
      outputs[bid],
      &compressed_sizes[bid],
      compressed_pad_bits ? &compressed_pad_bits[bid] : nullptr,
      inputs[bid],
      input_sizes[bid],
      output_pos
    );
  }
}

void huffman_encode_dispatch(
  uint32_t *const *outputs,
  size_t *compressed_sizes,
  uint8_t *compressed_pad_bits,
  const unsigned char *const *inputs,
  const size_t *input_sizes,
  const unsigned int batch_size,
  bool standard,
  cudaStream_t stream,
  const int *device_num_deflate_blocks
)
{
  decltype(&huffman_encode) kernel;
  if (standard)
  {
    kernel = huffman_encode_standard;
  }
  else
  {
    kernel = huffman_encode;
  }

  kernel<<<batch_size, WARP_SIZE, 0, stream>>>(
    outputs,
    compressed_sizes,
    compressed_pad_bits,
    inputs,
    input_sizes,
    batch_size,
    device_num_deflate_blocks
  );
}

} // namespace gdeflate
