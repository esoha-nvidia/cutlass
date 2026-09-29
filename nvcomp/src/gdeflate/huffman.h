/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include <cub/cub.cuh>

#include <cuda.h>

#include <cassert>
#include <cstdint>
#include <limits>

#include "bitwriter.h"
#include "common.h"
#include "gdeflate_constants.h"

#include <stdio.h>

// #include "gdeflate_decompress.h"
using namespace nvcomp;

namespace gdeflate
{

template <
  unsigned int N, // Max number of symbols
  unsigned int L, // Max code length
  typename Tn = uint16_t, // Type large enough to store N
  typename T = uint32_t>
class warp_encoder
{
  T *codes;
  const Tn *codelen;

public:
  static constexpr unsigned int width = sizeof(T) * 8;

  // Build the decoder using code lengths
  __device__ void init(unsigned int n, const Tn *codelen_, T *codes_, Tn *counts, T *basecode)
  {
    assert(n <= N);
    assert(blockDim.x == WARP_SIZE_U); // code below assumes there is only one warp on x dim

    codelen = codelen_;
    codes = codes_;

    for (unsigned int i = threadIdx.x; i < L + 1; i += WARP_SIZE_U)
    {
      counts[i] = 0;
    }
    __syncwarp();

    // Get histogram of code lengths
    Tn maxcodelen = 0;
    for (Tn i = threadIdx.x; i < WARP_SIZE_U * ((N - 1) / WARP_SIZE_U + 1); i += WARP_SIZE_U)
    {
      bool active = (i < n);
      Tn len = active ? codelen[i] : 0;
      unsigned int match = __match_any_sync(WARP_ALL, len);
      if ((len != 0) && (threadIdx.x == (__ffs(match) - 1)))
      {
        counts[len] += __popc(match);
      }
      __syncwarp();
      maxcodelen = len > maxcodelen ? len : maxcodelen;
    }
    // TODO: Change this to __reduce_max_sync for SM 80+
    // maxcodelen = __reduce_max_sync(WARP_ALL, (unsigned int)maxcodelen);
    maxcodelen = warpReduceMax((unsigned int)maxcodelen, WARP_ALL);

    // Compute basecodes for each code length
    for (unsigned int i = threadIdx.x; i < L + 1; i += WARP_SIZE_U)
    {
      basecode[i] = i <= maxcodelen ? 0 : 0xffffffff;
    }
    __syncwarp();

    if (threadIdx.x == 0)
    {
      T code = 0;
      for (unsigned int i = 1; i <= maxcodelen; ++i)
      {
        code = (code + counts[i - 1]) << 1;
        basecode[i] = code;
      }
    }
    __syncwarp();

    // Populate the symbol->code table
    for (unsigned int i = threadIdx.x; i < WARP_SIZE_U * ((N - 1) / WARP_SIZE_U + 1); i += WARP_SIZE_U)
    {
      Tn len = i < n ? codelen[i] : 0;
      unsigned int match = __match_any_sync(WARP_ALL, len);
      Tn offset = __popc(match & ltMask()) - 1;
      if (len != 0)
      {
        codes[i] = __brev((basecode[len] + offset) << (width - len)); // Left align and reverse
      }
      __syncwarp();
      if ((len != 0) && (threadIdx.x == (__ffs(match) - 1)))
      {
        basecode[len] += __popc(match);
      }
      __syncwarp();
    }
    __syncwarp();
  }

  // Get code length of a symbol
  __device__ Tn get_codelen(Tn sym) const
  {
    assert(sym < N);
    return codelen[sym];
  }

  // Get the code for a symbol
  // Code is already reversed and right aligned as required to write to the bitstream
  __device__ T get_code(Tn sym) const
  {
    assert(sym < N);
    return codes[sym];
  }
};

template <
  unsigned int N, // Max number of symbols
  unsigned int L, // Max code length
  typename Tn = uint16_t, // Type large enough to store N
  typename T = uint32_t>
class warp_decoder
{
  static_assert(L == 7 || L == 15, "Only max code lengths of 7 or 15 are supported");
  Tn *symbols;
  Tn *offset;
  T *basecode;

public:
  static constexpr unsigned int width = sizeof(T) * 8;

  // Build the decoder using code lengths
  __device__ void init(unsigned int n, const Tn *codelen, Tn *counts, Tn *symbols_, Tn *offset_, T *basecode_)
  {
    assert(n <= N);

    symbols = symbols_;
    offset = offset_;
    basecode = basecode_;

    for (unsigned int i = threadIdx.x; i < L + 1; i += WARP_SIZE_U)
    {
      counts[i] = 0;
    }
    __syncwarp();

    // Get histogram of code lengths
    Tn maxcodelen = 0;
    for (Tn i = threadIdx.x; i < WARP_SIZE_U * ((N - 1) / WARP_SIZE_U + 1); i += WARP_SIZE_U)
    {
      bool active = (i < n);
      Tn len = active ? codelen[i] : 0;
      unsigned int match = __match_any_sync(WARP_ALL, len);
      if ((len != 0) && (threadIdx.x == (__ffs(match) - 1)))
      {
        counts[len] += __popc(match);
      }
      maxcodelen = len > maxcodelen ? len : maxcodelen;
    }
    __syncwarp();
    // TODO: Change this to __reduce_max_sync for SM 80+
    // maxcodelen = __reduce_max_sync(WARP_ALL, (unsigned int)maxcodelen);
    maxcodelen = warpReduceMax((unsigned int)maxcodelen, WARP_ALL);

    // Compute offset using a prefix sum
    unsigned int count = threadIdx.x < L + 1 ? counts[threadIdx.x] : 0;
    unsigned int offset_val = prefixSum(count, WARP_ALL);

    if (threadIdx.x < L + 1)
    {
      offset[threadIdx.x] = offset_val;
    }
    __syncwarp();

    // Populate the symbol table and update offset to postfix sum
    for (Tn i = threadIdx.x; i < WARP_SIZE_U * ((N - 1) / WARP_SIZE_U + 1); i += WARP_SIZE_U)
    {
      Tn len = i < n ? codelen[i] : 0;
      unsigned int match = __match_any_sync(WARP_ALL, len);
      if (len != 0)
      {
        symbols[offset[len] + __popc(match & ltMask()) - 1] = i;
      }
      __syncwarp();
      if ((len != 0) && (threadIdx.x == (__ffs(match) - 1)))
      {
        offset[len] += __popc(match);
      }
    }

    for (unsigned int i = threadIdx.x; i < L + 1; i += WARP_SIZE_U)
    {
      basecode[i] = i <= maxcodelen ? 0 : 0xffffffff;
    }
    __syncwarp();

    if (threadIdx.x == 0)
    {
      T code = 0;
      for (unsigned int i = 1; i <= maxcodelen; ++i)
      {
        code = (code + counts[i - 1]) << 1;
        basecode[i] = code;
      }
    }
    __syncwarp();

    // Left align
    if (threadIdx.x <= maxcodelen)
    {
      basecode[threadIdx.x] <<= (width - threadIdx.x);
    }
    __syncwarp();
  }

  // Get code length
  __device__ Tn len4code(T code) const
  {
    Tn len = 1;
    if constexpr (L == 15)
    {
      // Decoder for literal/length and distance codes
      if (code >= basecode[8])
      {
        len = 8;
      }
      if (code >= basecode[len + 4])
      {
        len += 4;
      }
      if (code >= basecode[len + 2])
      {
        len += 2;
      }
      if (code >= basecode[len + 1])
      {
        len += 1;
      }
    }
    else if constexpr (L == 7)
    {
      // Decoder for code length codes
      if (code >= basecode[4])
      {
        len = 4;
      }
      if (code >= basecode[len + 2])
      {
        len += 2;
      }
      if (code >= basecode[len + 1])
      {
        len += 1;
      }
    }
    return len;
  }

  // Get symbol given the code and code length
  __device__ Tn sym4code(T code, Tn len) const
  {
    Tn o = offset[len - 1] + ((code - basecode[len]) >> (width - len));
    return symbols[o];
  }
};

// Class to perform branchless decodes for two Huffman trees with max code length of 15 each
// Performs one decode per thread of a warp (either lit/len or distance)
// Base codes and offsets for the Huffman decode process are distributed across threads
// in registers and shared using shuffles.
// Warp logically partitioned so the first 16 threads store the literal/length Huffman tables
// and the next 16 store the distance Huffman tables.
// The symbol table is stored in shared memory with the first 288 elements
// reserved for literal/length and the next 32 reserved for distance
// TODO: Can move the 320 symbols from the table to registers in a
// distributed fashion too adding at most 3 registers per thread
template <typename Tn = uint16_t, typename T = uint32_t>
class warp_dual_decoder
{
  const Tn *symbols;
  Tn offset;
  T basecode;

  static constexpr unsigned int width = sizeof(T) * 8;

public:
  // Constructor that reads in the table values from constant memory
  __device__ warp_dual_decoder(const Tn *symbols_, const Tn *offset_, const T *basecode_)
      : symbols(symbols_)
  {
    Tn symbol_offset = (threadIdx.x < 16) ? 0 : 288;

    // The first 16 correspond to literal/length
    // Next 16 are for distance
    basecode = basecode_[threadIdx.x];
    offset = symbol_offset + offset_[threadIdx.x];
  }

  // Get code length
  // base = 0 for literl/length and 16 for distance
  __device__ Tn len4code(T code, Tn base = 0) const
  {
    Tn len = 1;
    // Decoder for literal/length and distance codes
    if (code >= __shfl_sync(WARP_ALL, basecode, base + 8))
    {
      len = 8;
    }
    if (code >= __shfl_sync(WARP_ALL, basecode, base + len + 4))
    {
      len += 4;
    }
    if (code >= __shfl_sync(WARP_ALL, basecode, base + len + 2))
    {
      len += 2;
    }
    if (code >= __shfl_sync(WARP_ALL, basecode, base + len + 1))
    {
      len += 1;
    }
    return len;
  }

  // Get symbol given the code and code length
  // base = 0 for literl/length and 16 for distance
  __device__ Tn sym4code(T code, Tn len, Tn base = 0) const
  {
    T bcode = __shfl_sync(WARP_ALL, basecode, base + len);
    Tn soffset = __shfl_sync(WARP_ALL, offset, base + len - 1);
    Tn o = soffset + ((code - bcode) >> (width - len));
    return symbols[o];
  }
};

template <
  unsigned int N, // Max number of symbols
  unsigned int L, // Max code length
  typename Tn = uint16_t> // Type large enough to store 2*N
class WarpHuffmanTree
{
  Tn n_symbols;

public:
  __device__ WarpHuffmanTree(unsigned int *counts, Tn *symbols)
      : n_symbols(0)
  {
    assert(blockDim.x == 32); //this class assumes flat warp shape

    // Sort counts
    constexpr unsigned int items_per_thread = (N + WARP_SIZE_U - 1) / WARP_SIZE_U;
    typedef nvcomp::cub::BlockRadixSort<unsigned int, WARP_SIZE, items_per_thread, Tn> BlockRadixSort;
    __shared__ typename BlockRadixSort::TempStorage temp_storage;

    Tn s[items_per_thread];
    unsigned int c[items_per_thread];

// Load keys and values
#pragma unroll
    for (Tn i = 0; i < items_per_thread; ++i)
    {
      Tn start = items_per_thread * threadIdx.x;
      s[i] = start + i;
      c[i] = (start + i < N) ? counts[start + i] : 0;
      n_symbols += (c[i] == 0) ? 0 : 1;
      c[i] = c[i] == 0 ? std::numeric_limits<unsigned int>::max() : c[i];
    }

    // Sort
    BlockRadixSort(temp_storage).Sort(c, s);
    __syncwarp(); // TODO: BlockRadixSort here limits this implementation to 1 warp per CTA

// Store keys and values
#pragma unroll
    for (Tn i = 0; i < items_per_thread; ++i)
    {
      Tn start = items_per_thread * threadIdx.x;
      if ((start + i) < N)
      {
        symbols[start + i] = s[i];
        counts[start + i] = c[i];
      }
    }
    __syncwarp();

    // Get the total number of symbols with non-zero count
    n_symbols = warpReduceSum(n_symbols, WARP_ALL);
    assert(n_symbols > 0);
  }

  __device__ void build(unsigned int *counts, Tn *left, Tn *right, Tn *depth)
  {
    assert(blockDim.x == 32); //this class assumes flat warp shape
    // Only thread 0 builds the tree in a single threaded fashion
    if (threadIdx.x == 0)
    {
      Tn leaf_counter = 0;
      Tn branch_counter = 0;
      Tn n_branches = 0;

      // Build tree by combining nodes with the least counts until only 1 node remains
      while ((leaf_counter < n_symbols) || (n_branches - branch_counter > 1))
      {
        unsigned int ind[2];

        // Get the smallest two nodes
        for (int i = 0; i < 2; ++i)
        {
          if (branch_counter >= n_branches)
          {
            ind[i] = leaf_counter++;
          }
          else if (leaf_counter >= n_symbols)
          {
            ind[i] = N + branch_counter++;
          }
          else
          {
            ind[i] = (counts[leaf_counter] <= counts[N + branch_counter]) ? leaf_counter++ : (N + branch_counter++);
          }
        }

        // Merge the two nodes
        counts[N + n_branches] = counts[ind[0]] + counts[ind[1]];
        // Put higher count node on the left
        left[n_branches] = ind[1];
        right[n_branches] = ind[0];
        n_branches++;
      }

      // Last remaining node is the root of the tree
      int root = N + n_branches - 1;
      depth[root] = 0;

      // Now need to traverse the tree to get the codelengths of each symbol
      Tn q_size = 0, q_counter = 0;
      int *queue = reinterpret_cast<int *>(&counts[N]);

      // Add root to the queue and traverse the tree
      // TODO: Parallelize this tree traversal
      queue[q_size++] = root;
      Tn max_depth = 0;
      while (q_size > q_counter)
      {
        int node = queue[q_counter++];
        max_depth = depth[node] + 1;

        Tn l = left[node - N];
        Tn r = right[node - N];
        depth[l] = max_depth;
        depth[r] = max_depth;

        // Don't add leaf nodes to the queue
        if (l >= N)
        {
          queue[q_size++] = l;
        }
        if (r >= N)
        {
          queue[q_size++] = r;
        }
      }
    }
    __syncwarp();
  }

  __device__ void codelengths(Tn *codelen, const Tn *symbols, const Tn *depth)
  {
    assert(blockDim.x == 32); //this class assumes flat warp shape

    // Compute Kraft number scaled by 2^(L/2) for increased dynamic range
    float K = 0.;
    constexpr int norm_power = L / 2;

    for (unsigned int i = threadIdx.x; i < N; i += WARP_SIZE_U)
    {
      Tn symbol = symbols[i];
      Tn len = min(i < n_symbols ? depth[i] : 0, L); // Limit to max code length L
      codelen[symbol] = len;
      K += len > 0 ? scalbnf(1.f, norm_power - len) : 0; // K += 2^(norm_power - len)
    }
    K = warpReduceSum(K, WARP_ALL);

    // Return if Kraft number <= 1
    if (K <= scalbnf(1.f, norm_power))
    {
      return;
    }

    __syncwarp();
    // TODO: Add better depth limiting
    if (threadIdx.x == 0)
    {

      for (int i = 0; i < n_symbols; ++i)
      {
        Tn symbol = symbols[i];
        Tn len = codelen[symbol];
        if (len < L)
        {
          len += 1;
          K -= scalbnf(1.f, norm_power - len);
          codelen[symbol] = len;
        }
        if (K <= scalbnf(1.f, norm_power))
        {
          break;
        }
      }

      for (int i = n_symbols - 1; i >= 0; --i)
      {
        Tn symbol = symbols[i];
        Tn len = codelen[symbol];
        float K_ = K + scalbnf(1.f, norm_power - len);

        if (K_ > scalbnf(1.f, norm_power))
        {
          continue;
        }

        codelen[symbol] = len - 1;
        K = K_;

        if (K >= scalbnf(1.f, norm_power))
        {
          break;
        }
      }
    }
    __syncwarp();
  }

  __device__ Tn num_symbols() { return n_symbols; }
};

__global__ void huffman_tree(unsigned int *counts_, uint16_t *codelen, unsigned int *codes);

template <typename T>
__device__ void update_histogram(unsigned int *counts, const T *input, const size_t n)
{
  for (unsigned int i = threadIdx.x; i < n; i += WARP_SIZE_U)
  {
    T b = input[i];
    // TODO: Use warp match to replace shared memory atomics
    atomicAdd(&counts[b], 1);
  }
}

template <bool Header = false, typename Twriter, typename... OptOutT>
__device__ void
write_bits(Twriter &bw, uint32_t bits, unsigned int n, bool active, [[maybe_unused]] OptOutT &...opt_out_pos)
{
  static_assert(sizeof...(OptOutT) <= 1);
  if constexpr (sizeof...(OptOutT) == 1)
  {
    if constexpr (Header)
    {
      bw.standard_write_header(bits, n, opt_out_pos..., active);
    }
    else
    {
      bw.standard_write(bits, n, opt_out_pos..., active);
    }
  }
  else
  {
    bw.write(bits, n, active);
  }
}

template <typename Twriter, typename... OptOutT>
__device__ void pack_codelens(
  const uint16_t *codelen,
  unsigned int hlit,
  unsigned int hdist,
  Twriter &bw,
  [[maybe_unused]] OptOutT &...opt_out_pos
)
{
  assert(hlit >= gdeflate_literal_symbols);
  assert(hdist >= 1);

  __shared__ uint16_t left[16];
  __shared__ uint16_t right[16];
  __shared__ uint16_t depth[2 * 16];
  __shared__ uint16_t symbols[16];
  __shared__ unsigned int counts[2 * 16];
  uint16_t *lencode = reinterpret_cast<uint16_t *>(counts);
  uint32_t *codes = reinterpret_cast<uint32_t *>(&counts[16]);

  // Get counts for codelengths
  for (unsigned int i = threadIdx.x; i < 16; i += WARP_SIZE_U)
  {
    counts[i] = 0;
  }
  __syncwarp();

  // Compute counts of each codelength. Skip the gap of unused and invalid
  // symbols at the end of the litlen table.
  update_histogram(counts, codelen, hlit);
  update_histogram(counts, codelen + gdeflate_litlen_symbols, hdist);

  __syncwarp();

  // Build code length Huffman tree
  WarpHuffmanTree<16, 7> wht(counts, symbols);
  wht.build(counts, left, right, depth);
  wht.codelengths(lencode, symbols, depth);
  __syncwarp();

  // Encoder for code lengths
  // Here only 16 since we are not run-length encoding the codelength codelengths
  // TODO: Add RLE for codelengths
  warp_encoder<16, 7> enc;
  enc.init(16, lencode, codes, reinterpret_cast<uint16_t *>(left), reinterpret_cast<unsigned int *>(right));

  // Output hlit, hdist and hclen to block header
  uint32_t header = (19 - 4) & mask<uint32_t>(4); // 4 bit hclen
  header = (header << 5) | ((hdist - 1) & mask<uint32_t>(5)); // 5 bit hdist
  header = (header << 5) | ((hlit - gdeflate_literal_symbols) & mask<uint32_t>(5)); // 5 bit hlit
  write_bits</*Header=*/true>(bw, header, 14, threadIdx.x == 0, opt_out_pos...); // Only thread 0 writes to its stream

  // Output lencodes in mapped order
  bool active = (threadIdx.x < 19);
  unsigned int pos = active ? map[threadIdx.x] : 0;
  uint32_t len = pos < 16 ? lencode[pos] : 0;
  write_bits(bw, len, 3, active, opt_out_pos...);

  // Encode and write out code lengths
  const size_t total_symbol_count = hlit + hdist;
  for (unsigned int iter = 0; iter < (total_symbol_count + WARP_SIZE_U - 1) / WARP_SIZE_U; ++iter)
  {
    // We need the active variable rather than just doing a warp-stride loop over
    // i as bw requires the participation of the entire warp.
    unsigned int i = threadIdx.x + iter * WARP_SIZE_U;
    bool active = i < total_symbol_count;
    // Avoid the gap of unused/invalid symbols at the end of the litlen codelength
    // section. The last used litlen codelength and the first used distance codelength
    // must still be processed by neighbouring GDeflate streams (corresponding
    // to threads in our implementation) as per the GDeflate specification,
    // so simply setting active to false does not work.
    i += (i >= hlit) ? (gdeflate_litlen_symbols - hlit) : 0;
    // For inactive threads, cannot assign sym = 0, since 0 might not appear as
    // a codelength. Assign the codelength of the end-of-block marker, guaranteed
    // to be present, instead.
    uint16_t sym = codelen[active ? i : 256];
    uint16_t len = enc.get_codelen(sym);
    uint32_t code = enc.get_code(sym);
    write_bits(bw, code, len, active, opt_out_pos...);
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
  const int *device_num_deflate_blocks = nullptr
);

} // namespace gdeflate
