/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
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

#include <cuda/atomic>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <cooperative_groups.h>

#ifndef NVCOMPDX
#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif

#ifndef _MSC_VER
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#endif // _MSC_VER

// Need to suppress "effc++" warnings in fmt headers
#include <fmt/format.h>

#ifndef _MSC_VER
#pragma GCC diagnostic pop
#endif // _MSC_VER
#endif // NVCOMPDX

#include "Environment.hpp"
#include "LZ4Constants.cuh"
#include "LZ4Types.cuh"
#include "nvcomp/shared_types.h"
#ifndef NVCOMPDX
#include "exception.hpp"
#include "snappy/constants.cuh"
#endif // NVCOMPDX

#ifdef __CUDACC__
#include "snappy/util.cuh"
namespace cg = cooperative_groups;
#endif // __CUDACC__

namespace nvcomp
{
enum class LZ4CorrectnessErrorTypes
{
  EMPTY_BLOCK = 1,
  INVALID_LSIC = 2,
  INVALID_OFFSET = 3,
  LITERAL_SEQUENCE_TOO_LONG = 4,
  UNEXPECTED_END_OF_CHUNK = 5,
  INVALID_LAST_SEQUENCE = 6,
  INVALID_MATCH = 7,
  LAST_LITERAL_SEQUENCE_TOO_SHORT = 8,
  INVALID_LAST_MATCH = 9,
  CHUNK_TOO_LONG = 10,
};

template <bool CORRECTNESS_CHECK, typename CorrectnessErrorType, typename comp_decomp_id_type, class CheckerClass>
class CorrectnessChecker
{
public:
#ifdef __CUDACC__

  /*!
   * @brief Checks whether the checker has encountered an error.
   * @param checker The checker instance to check.
   * @return True if the checker has encountered an error, false otherwise.
   */
  static inline __device__ __host__ bool hasError(CheckerClass *checker)
  {
    if constexpr (CORRECTNESS_CHECK)
    {
      return checker->line_number_ != -1;
    }
    return false;
  }

  /*!
   * @brief Sets the error information in the checker.
   * @param error_type The type of error encountered.
   * @param comp_idx The index in the compressed data where the error
   * occurred.
   * @param decomp_idx The index in the decompressed data where the error
   * occurred.
   * @param line_number The line number where the error occurred.
   * @param function_name The name of the function where the error occurred.
   * @param checker The checker instance to update.
   */
  static inline __device__ void setError(
    const CorrectnessErrorType error_type,
    const comp_decomp_id_type comp_idx,
    const comp_decomp_id_type decomp_idx,
    const int line_number,
    const char *function_name,
    CheckerClass *checker
  )
  {
    if constexpr (CORRECTNESS_CHECK)
    {
      if (checker->getThreadRank() == 0)
      {
        checker->error_type_ = error_type;
        checker->comp_idx_ = comp_idx;
        checker->decomp_idx_ = decomp_idx;
        checker->line_number_ = line_number;
        checker->function_name_ptr_ = function_name;
        checker->function_name_size_ = 0;
        while (function_name[checker->function_name_size_] != '\0')
        {
          checker->function_name_size_++;
        }
      }
      checker->sync();
    }
  }

  /*!
   * @brief Sets an error if the condition is true.
   * @param condition The condition to check.
   * @param error_type The type of error to set if the condition is true.
   * @param comp_idx The index in the compressed data where the error occurred.
   * @param decomp_idx The index in the decompressed data where the error
   * occurred.
   * @param line_number The line number where the error occurred.
   * @param function_name The name of the function where the error occurred.
   * @param correctness_checker Pointer to the correctness checker object.
   * @return True if the condition is true and an error was set, false
   * otherwise.
   */
  static inline __device__ bool errorIfTrue(
    const bool condition,
    const CorrectnessErrorType error_type,
    const comp_decomp_id_type comp_idx,
    const comp_decomp_id_type decomp_idx,
    const int line_number,
    const char *function_name,
    CheckerClass *correctness_checker
  )
  {
    if constexpr (CORRECTNESS_CHECK)
    {
      if (condition)
      {
        setError(error_type, comp_idx, decomp_idx, line_number, function_name, correctness_checker);
      }
      return condition;
    }
    return false;
  }

#endif // __CUDACC__

#ifndef NVCOMPDX
  __host__ auto getErrorType() const { return error_type_; }

  __host__ auto getCompIdx() const { return comp_idx_; }

  __host__ auto getDecompIdx() const { return decomp_idx_; }

  __host__ auto getLineNumber() const { return line_number_; }

  __host__ std::string getFunctionName(cudaStream_t stream) const
  {
    std::string function_name;
    if (function_name_ptr_ != nullptr)
    {
      function_name.resize(function_name_size_);
      CUDA_CHECK(
        cudaMemcpyAsync(function_name.data(), function_name_ptr_, function_name_size_, cudaMemcpyDeviceToHost, stream)
      );
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    return function_name;
  }
#endif // NVCOMPDX

protected:
#ifndef NVCOMPDX
  __host__ static void printErrorsInternal(
    CheckerClass *d_correctness_checkers,
    size_t const batch_size,
    cudaStream_t stream,
    std::string const &default_filename,
    std::string const &env_var,
    std::string const &algo_name
  )
  {
    if constexpr (CORRECTNESS_CHECK)
    {
      std::string filename = nvcomp::getenv(env_var.c_str());
      if (filename.empty())
      {
        filename = default_filename;
      }

      std::vector<CheckerClass> correctness_checkers(batch_size);
      CUDA_CHECK(cudaMemcpyAsync(
        correctness_checkers.data(),
        d_correctness_checkers,
        batch_size * sizeof(CheckerClass),
        cudaMemcpyDeviceToHost,
        stream
      ));
      CUDA_CHECK(cudaStreamSynchronize(stream));

      std::FILE *output = nullptr;
      if (filename == "stdout")
      {
        output = stdout;
      }
      else if (filename == "stderr")
      {
        output = stderr;
      }
      else
      {
        output = std::fopen(filename.c_str(), "a");
        if (output == nullptr)
        {
          // Fallback to stderr if file cannot be opened
          std::cerr << "Warning: Could not open log file " << filename << " for writing. Falling back to stderr."
                    << std::endl;
          output = stderr;
        }
      }

      for (size_t i = 0; i < batch_size; ++i)
      {
        if (CheckerClass::hasError(correctness_checkers.data() + i))
        {
          // If the line number is negative, it means no error was found
          fmt::print(
            output,
            "{} decompression error in chunk {}. Error {}: {}, at "
            "compressed index {}, "
            "decompressed index {}, function {}, line number {}\n",
            algo_name,
            i,
            correctness_checkers[i].getErrorType(),
            correctness_checkers[i].getErrorMessage(),
            correctness_checkers[i].getCompIdx(),
            correctness_checkers[i].getDecompIdx(),
            correctness_checkers[i].getFunctionName(stream),
            correctness_checkers[i].getLineNumber()
          );
        }
      }

      if (output != stderr && output != stdout)
      {
        std::fclose(output);
      }
    }
  }
#endif // NVCOMPDX

  CorrectnessErrorType error_type_;
  comp_decomp_id_type comp_idx_; // the index in the compressed data where the error occurred
  comp_decomp_id_type decomp_idx_; // the index in the decompressed data where
  // the error occurred
  int line_number_; // the line number where the error occurred
  const char *function_name_ptr_; // the name of the function where the error occurred
  size_t function_name_size_; // the size of the function name
};

/*!
 * @brief A correctness checker for LZ4 decompression.
 * @details All functions are static and receive a pointer to the object
 *          to avoid the need for instantiation.
 * @tparam CORRECTNESS_CHECK If true, correctness checks are enabled.
 */
template <bool CORRECTNESS_CHECK>
class LZ4CorrectnessChecker
    : public CorrectnessChecker<
        CORRECTNESS_CHECK,
        LZ4CorrectnessErrorTypes,
        lz4::position_type,
        LZ4CorrectnessChecker<CORRECTNESS_CHECK>>
{
public:
#ifdef __CUDACC__

  /*!
   * @brief Initializes the correctness checker to its default values.
   * @param thread_group The thread group to associate with the checker.
   * @param checker The checker instance to initialize.
   */
  template <typename T>
  static inline __device__ void Initialize(T *thread_group, LZ4CorrectnessChecker<CORRECTNESS_CHECK> *checker)
  {
    if constexpr (CORRECTNESS_CHECK)
    {
      checker->line_number_ = -1; // Initialize to -1 to indicate no error
      checker->last_match_start_ = -1;
      checker->thread_group_ = thread_group;
    }
  }

  /*!
   * @brief Checks whether the chunk is empty or has only one non-zero byte.
   * @param chunk_size The size of the chunk.
   * @param first_byte The first byte of the chunk.
   * @param line_number The line number where the check is performed.
   * @param function_name The name of the function where the check is performed.
   * @param correctness_checker Pointer to the correctness checker object.
   * @return True if the chunk is empty or has only one non-zero byte, false
   * otherwise.
   */
  static inline __device__ bool checkEmptyChunk(
    const size_t chunk_size,
    const uint8_t first_byte,
    const int line_number,
    const char *function_name,
    LZ4CorrectnessChecker<CORRECTNESS_CHECK> *correctness_checker
  )
  {
    if constexpr (CORRECTNESS_CHECK)
    {
      if (chunk_size == 0)
      {
        // A valid nvCOMP LZ4 chunk cannot be 0 bytes.
        setError(LZ4CorrectnessErrorTypes::EMPTY_BLOCK, 0, 0, line_number, function_name, correctness_checker);
        return true;
      }
      else if ((chunk_size == 1) && (first_byte != 0))
      {
        // Empty input can be represented using a zero byte, interpreted as a
        // final token without literal and without a match.
        setError(
          LZ4CorrectnessErrorTypes::UNEXPECTED_END_OF_CHUNK,
          0,
          0,
          line_number,
          function_name,
          correctness_checker
        );
        return true;
      }
    }
    return false;
  }

  /*!
   * @brief Sets the start index of the latest match in the decompressed data.
   * @param decomp_idx The start index of the latest match in the decompressed
   * buffer.
   * @param correctness_checker Pointer to the correctness checker object.
   */
  static inline __device__ void
  setMatchStart(const lz4::position_type decomp_idx, LZ4CorrectnessChecker<CORRECTNESS_CHECK> *correctness_checker)
  {
    if constexpr (CORRECTNESS_CHECK)
    {
      if (correctness_checker->getThreadRank() == 0)
      {
        correctness_checker->last_match_start_ = decomp_idx;
      }
      correctness_checker->sync();
    }
  }

  /*!
   * @brief Checks whether the last match is valid, i.e. starts at least 12
   * bytes before the end of the decompressed data.
   * @param comp_idx The index in the compressed data where we currently are,
   * for error logging.
   * @param decomp_end The end index of the decompressed data.
   * @param line_number The line number where the check is performed.
   * @param function_name The name of the function where the check is performed.
   * @param correctness_checker Pointer to the correctness checker object.
   * @return True if the last match is invalid, false otherwise.
   */
  static inline __device__ bool checkLastMatchValidity(
    const lz4::position_type comp_idx,
    const lz4::position_type decomp_end,
    const int line_number,
    const char *function_name,
    LZ4CorrectnessChecker<CORRECTNESS_CHECK> *correctness_checker
  )
  {
    if constexpr (CORRECTNESS_CHECK)
    {
      if (correctness_checker->last_match_start_ >= 0 &&
          correctness_checker->last_match_start_ + LAST_VALID_MATCH_BYTES > decomp_end)
      {
        setError(
          LZ4CorrectnessErrorTypes::INVALID_LAST_MATCH,
          comp_idx,
          decomp_end,
          line_number,
          function_name,
          correctness_checker
        );
        return true;
      }
    }
    return false;
  }

  inline __device__ int getThreadRank() const { return thread_group_->thread_rank(); }

  inline __device__ void sync() const { thread_group_->sync(); }
#endif // __CUDACC__

#ifndef NVCOMPDX
  __host__ std::string getErrorMessage() const
  {
    switch (this->error_type_)
    {
      case LZ4CorrectnessErrorTypes::EMPTY_BLOCK:
        return "Empty block encountered";
      case LZ4CorrectnessErrorTypes::INVALID_LSIC:
        return "Invalid LSIC read";
      case LZ4CorrectnessErrorTypes::INVALID_OFFSET:
        return "Invalid offset";
      case LZ4CorrectnessErrorTypes::LITERAL_SEQUENCE_TOO_LONG:
        return "Literal sequence overflows input or output buffer";
      case LZ4CorrectnessErrorTypes::UNEXPECTED_END_OF_CHUNK:
        return "Unexpected end of chunk";
      case LZ4CorrectnessErrorTypes::INVALID_LAST_SEQUENCE:
        return "Last sequence contains a match";
      case LZ4CorrectnessErrorTypes::INVALID_MATCH:
        return "Invalid match";
      case LZ4CorrectnessErrorTypes::LAST_LITERAL_SEQUENCE_TOO_SHORT:
        return "Last literal sequence too short";
      case LZ4CorrectnessErrorTypes::INVALID_LAST_MATCH:
        return "Last match does not start at least 12 bytes before output end";
      case LZ4CorrectnessErrorTypes::CHUNK_TOO_LONG:
        return "Decompressed chunk is longer than allowed";
      default:
        return "Unknown error";
    }
  }

  /*!
   * @brief Prints the error information for each checker in the batch.
   * @param d_correctness_checkers Pointer to the device array of checkers.
   * @param batch_size The number of checkers in the batch.
   * @param stream The CUDA stream to use for asynchronous operations.
   */
  static __host__ void printErrors(
    LZ4CorrectnessChecker<CORRECTNESS_CHECK> *d_correctness_checkers,
    const size_t batch_size,
    cudaStream_t stream
  )
  {
    if constexpr (CORRECTNESS_CHECK)
    {
      CorrectnessChecker<
        CORRECTNESS_CHECK,
        LZ4CorrectnessErrorTypes,
        lz4::position_type,
        LZ4CorrectnessChecker<CORRECTNESS_CHECK>>::
        printErrorsInternal(
          d_correctness_checkers,
          batch_size,
          stream,
          "lz4_correctness_check.log",
          LZ4_CORRECTNESS_LOG_OUTPUT_ENV,
          "LZ4"
        );
    }
  }
#endif // NVCOMPDX

private:
  int64_t last_match_start_; // the start index of the last match in the
  // decompressed data
#ifdef __CUDACC__
  static_assert(
    sizeof(void *) == sizeof(cg::thread_block_tile<WARP_SIZE_U, cg::thread_block> *),
    "Thread group pointer size mismatch"
  );
  cg::thread_block_tile<WARP_SIZE_U, cg::thread_block> *thread_group_; // thread group for synchronization
#else
  // For host code, we don't need thread group synchronization
  void *thread_group_;
#endif // __CUDACC__
};

} // namespace nvcomp

namespace snappy
{

enum class SnappyCorrectnessErrorTypes
{
  EMPTY_BLOCK = 1,
  INVALID_UNCOMPRESSED_LENGTH = 2,
  COPY_AS_FIRST_SYMBOL = 3,
  WRONG_INPUT_LENGTH = 4,
  WRONG_OUTPUT_LENGTH = 5,
  INVALID_OFFSET = 6,
  OUT_OF_BOUNDS_WRITE = 7,
  OUT_OF_BOUNDS_LITERAL_READ = 8,
  OUTPUT_BUFFER_TOO_SMALL = 9,
};

// Assumes that functions are called by one whole warp
template <bool CORRECTNESS_CHECK>
class SnappyCorrectnessChecker
    : public nvcomp::CorrectnessChecker<
        CORRECTNESS_CHECK,
        SnappyCorrectnessErrorTypes,
        uint32_t,
        SnappyCorrectnessChecker<CORRECTNESS_CHECK>>
{
public:
  SnappyCorrectnessChecker() = default;

#ifdef __CUDACC__

  /*!
   * @brief Initializes the correctness checker to its default values.
   * @param checker The checker instance to initialize.
   * @param input_buffer Pointer to the input buffer.
   * @param input_buffer_size Size of the input buffer in bytes.
   * @param output_buffer_size Size of the output buffer in bytes.
   */
  static inline __device__ void Initialize(
    SnappyCorrectnessChecker<CORRECTNESS_CHECK> *checker,
    const uint8_t *input_buffer,
    const uint64_t input_buffer_size,
    const uint64_t output_buffer_size
  )
  {
    if constexpr (CORRECTNESS_CHECK)
    {
      if (checker->getThreadRank() == 0)
      {
        checker->line_number_ = -1; // Initialize to -1 to indicate no error
        for (int i = 0; i < 3; ++i)
        {
          checker->counters[i].store(0, cuda::std::memory_order_relaxed);
        }
        checker->input_buffer_ = input_buffer;
        assert(input_buffer_size <= UINT32_MAX);
        assert(output_buffer_size <= UINT32_MAX);
        checker->input_buffer_size_ = input_buffer_size;
        checker->output_buffer_size_ = output_buffer_size;
      }
      checker->sync();
    }
  }

  /*!
   * @brief Performs preliminary checks on the input and output buffers.
   * @param comp_idx The index in the compressed data where we currently are,
   * for error logging.
   * @param decomp_idx The index in the decompressed data where we currently
   * are, for error logging.
   * @param line_number The line number where the check is performed.
   * @param function_name The name of the function where the check is performed.
   * @param checker The checker instance to update.
   */
  static inline __device__ void preliminaryChecks(
    const uint32_t comp_idx,
    const uint32_t decomp_idx,
    const int line_number,
    const char *function_name,
    SnappyCorrectnessChecker<CORRECTNESS_CHECK> *checker
  )
  {
    if constexpr (CORRECTNESS_CHECK)
    {
      // Check that input is not empty
      bool exit = errorIfTrue(
        checker->input_buffer_size_ == 0,
        SnappyCorrectnessErrorTypes::EMPTY_BLOCK,
        0,
        0,
        line_number,
        function_name,
        checker
      );
      if (exit)
      {
        return;
      }

      // Check that the varints do not exceed the input buffer
      uint32_t num_varints = 1;
      while (checker->input_buffer_[num_varints - 1] >= 0x80)
      {
        num_varints++;
        if (num_varints >= checker->input_buffer_size_)
        {
          setError(
            SnappyCorrectnessErrorTypes::INVALID_UNCOMPRESSED_LENGTH,
            comp_idx,
            decomp_idx,
            line_number,
            function_name,
            checker
          );
          return;
        }
      }

      // check that the varints are correct
      int32_t error = 0;
      uint32_t header_size = 0;
      const uint32_t output_size =
        get_uncompressed_size(checker->input_buffer_, checker->input_buffer_size_, header_size, error);
      exit = errorIfTrue(
        error < 0 || header_size != num_varints,
        SnappyCorrectnessErrorTypes::INVALID_UNCOMPRESSED_LENGTH,
        0,
        0,
        line_number,
        function_name,
        checker
      );
      if (exit)
      {
        return;
      }

      checker->header_size_ = header_size;

      // Check that the input size is not zero after reading varints
      exit = errorIfTrue(
        (checker->input_buffer_size_ - num_varints) == 0 && output_size > 0,
        SnappyCorrectnessErrorTypes::INVALID_UNCOMPRESSED_LENGTH,
        0,
        0,
        line_number,
        function_name,
        checker
      );
      if (exit)
      {
        return;
      }

      // Check that the reported output size is not larger than the available
      // output buffer
      exit = errorIfTrue(
        checker->output_buffer_size_ < output_size,
        SnappyCorrectnessErrorTypes::OUTPUT_BUFFER_TOO_SMALL,
        0,
        0,
        line_number,
        function_name,
        checker
      );
      if (exit)
      {
        return;
      }

      checker->uncompressed_size_ = output_size;

      // Here an empty input would end
      if (checker->input_buffer_size_ == 1)
      {
        return;
      }

      // Check that the first symbol is not a copy
      exit = errorIfTrue(
        (checker->input_buffer_[num_varints] & 0x03) != 0,
        SnappyCorrectnessErrorTypes::COPY_AS_FIRST_SYMBOL,
        0,
        0,
        line_number,
        function_name,
        checker
      );
      if (exit)
      {
        return;
      }
    }
  }

  /*!
   * @brief Checks the validity of symbols (literal or match) during decompression.
   * @param source 4BpB gather index.
   * @param cursor Index where the symbol starts writing in the output buffer
   * @param end The end position in the output buffer after processing the
   * current symbol (exclusive).
   * @param comp_idx The index in the compressed data where we currently are,
   * for error logging.
   * @param uncomp_symbol_len The length of the uncompressed symbol.
   * @param active Whether the current thread is active (processing a symbol).
   * @param tag The tag byte indicating the type of symbol (literal or match).
   * @param line_number The line number where the check is performed.
   * @param function_name The name of the function where the check is performed.
   * @param checker The checker instance to update.
   * @return True if an error was detected, false otherwise.
   */
  static inline __device__ bool checkSymbols(
    const uint32_t source,
    const uint32_t cursor,
    const uint32_t end,
    const uint32_t comp_idx,
    uint32_t uncomp_symbol_len,
    const bool active,
    const uint32_t tag,
    const int line_number,
    const char *function_name,
    SnappyCorrectnessChecker<CORRECTNESS_CHECK> *checker
  )
  {
    if constexpr (CORRECTNESS_CHECK)
    {
      // Check that match does not go OOB and offset is not 0
      auto offset = SNAPPY_OFFSET_4BPB - (source - cursor);
      const bool invalid_offset = active && (tag > 0) && ((offset == 0) || (offset > cursor));
      bool should_exit = __ballot_sync(WARP_ALL, invalid_offset) > 0;
      if (should_exit)
      {
        setError(
          SnappyCorrectnessErrorTypes::INVALID_OFFSET,
          comp_idx,
          static_cast<int64_t>(cursor) - offset,
          line_number,
          function_name,
          checker
        );
        return true;
      }
      // Check that match does not overflow the output buffer or the uncompressed size
      const bool write_exceeds_buffer = end > checker->output_buffer_size_;
      const bool oob_write = active && (write_exceeds_buffer || end > checker->uncompressed_size_);
      should_exit = __ballot_sync(WARP_ALL, oob_write) > 0;
      if (should_exit)
      {
        offset = (tag == 0) ? 0 : offset;
        setError(
          write_exceeds_buffer ? SnappyCorrectnessErrorTypes::OUT_OF_BOUNDS_WRITE
                               : SnappyCorrectnessErrorTypes::WRONG_OUTPUT_LENGTH,
          comp_idx,
          static_cast<int64_t>(cursor) - offset,
          line_number,
          function_name,
          checker
        );
        return true;
      }
      // Check that literal sequence does not go out of bounds
      const bool oob_literal_read =
        active && (tag == 0) && ((source + uncomp_symbol_len + checker->header_size_) > checker->input_buffer_size_);
      should_exit = __ballot_sync(WARP_ALL, oob_literal_read) > 0;
      if (should_exit)
      {
        setError(
          SnappyCorrectnessErrorTypes::OUT_OF_BOUNDS_LITERAL_READ,
          source,
          cursor,
          line_number,
          function_name,
          checker
        );
        return true;
      }
    }
    return false;
  }

  /*!
   * @brief Checks that the actual input and output sizes match the expected
   * sizes after decompression.
   * @param actual_input_size The actual size of the input buffer used during
   * decompression.
   * @param actual_output_size The actual size of the output buffer produced
   * during decompression.
   * @param line_number The line number where the check is performed.
   * @param function_name The name of the function where the check is performed.
   * @param status Reference to the nvcompStatus_t variable to update in case of
   * error.
   * @param checker The checker instance to update.
   */
  static inline __device__ void checkInputOutputSizes(
    const uint32_t actual_input_size,
    const uint32_t actual_output_size,
    const int line_number,
    const char *function_name,
    nvcompStatus_t &status,
    SnappyCorrectnessChecker<CORRECTNESS_CHECK> *checker
  )
  {
    if constexpr (CORRECTNESS_CHECK)
    {
      if ((actual_input_size != checker->input_buffer_size_) || (actual_output_size != checker->uncompressed_size_))
      {
        setError(
          actual_input_size != checker->input_buffer_size_ ? SnappyCorrectnessErrorTypes::WRONG_INPUT_LENGTH
                                                           : SnappyCorrectnessErrorTypes::WRONG_OUTPUT_LENGTH,
          actual_input_size,
          actual_output_size,
          line_number,
          function_name,
          checker
        );
        if (checker->getThreadRank() == 0)
        {
          status = nvcompErrorCannotDecompress;
        }
        mapperExit(checker);
      }
      else
      {
        mapperFreeWindow(checker);
      }
    }
  }

  /*!
   * @brief Advances the finder counter by the specified number of symbols.
   * @param num_symbols The number of symbols processed by the finder.
   * @param checker The checker instance to update.
   */
  static inline __device__ void
  advanceFinder(const int num_symbols, SnappyCorrectnessChecker<CORRECTNESS_CHECK> *checker)
  {
    if constexpr (CORRECTNESS_CHECK)
    {
      if (checker->getThreadRank() == 0)
      {
        checker->counters[IX_FINDER].store(num_symbols, cuda::std::memory_order_relaxed);
      }
      checker->sync();
    }
  }

  /*!
   * @brief Advances the mapper counter by the specified number of symbols.
   * @param num_symbols The number of symbols processed by the mapper.
   * @param checker The checker instance to update.
   */
  static inline __device__ void
  advanceMapper(const int num_symbols, SnappyCorrectnessChecker<CORRECTNESS_CHECK> *checker)
  {
    if constexpr (CORRECTNESS_CHECK)
    {
      if (checker->getThreadRank() == 0)
      {
        checker->counters[IX_MAPPER].store(num_symbols, cuda::std::memory_order_relaxed);
      }
      checker->sync();
    }
  }

  /*!
   * @brief Stall the finder until the mapper is finished processing the found symbols.
   * @param checker The checker instance to update.
   */
  static inline __device__ bool finderWaitForMapper(SnappyCorrectnessChecker<CORRECTNESS_CHECK> *checker)
  {
    if constexpr (CORRECTNESS_CHECK)
    {
      const auto this_counter = checker->counters[IX_FINDER].load(cuda::std::memory_order_relaxed);
      auto mapper_counter = checker->counters[IX_MAPPER].load(cuda::std::memory_order_relaxed);
      while (this_counter > mapper_counter)
      {
        __nanosleep(SLEEP_DURATION);
        mapper_counter = checker->counters[IX_MAPPER].load(cuda::std::memory_order_relaxed);
        if (mapper_counter < 0)
        {
          return true;
        }
      }
    }
    return false;
  }

  /*!
   * @brief Stall the mapper until the finder has found new symbols.
   * @param checker The checker instance to update.
   */
  static inline __device__ void mapperWaitForFinder(SnappyCorrectnessChecker<CORRECTNESS_CHECK> *checker)
  {
    if constexpr (CORRECTNESS_CHECK)
    {
      while (checker->counters[IX_MAPPER].load(cuda::std::memory_order_relaxed) >=
             checker->counters[IX_FINDER].load(cuda::std::memory_order_relaxed))
      {
        __nanosleep(SLEEP_DURATION);
      }
    }
  }

  /*!
   * @brief Stall the mapper until the processor is finished processing it's window.
   * @param checker The checker instance to update.
   */
  static inline __device__ void mapperWaitForProcessor(SnappyCorrectnessChecker<CORRECTNESS_CHECK> *checker)
  {
    if constexpr (CORRECTNESS_CHECK)
    {
      while (checker->counters[IX_PROCESSOR].load(cuda::std::memory_order_acquire) >= 1)
      {
        __nanosleep(SLEEP_DURATION);
      }
    }
  }

  /*!
   * @brief Signals that the mapper has freed a window for processing.
   * @param checker The checker instance to update.
   */
  static inline __device__ void mapperFreeWindow(SnappyCorrectnessChecker<CORRECTNESS_CHECK> *checker)
  {
    if constexpr (CORRECTNESS_CHECK)
    {
      if (checker->getThreadRank() == 0)
      {
        checker->counters[IX_PROCESSOR]++;
      }
    }
  }

  /*!
   * @brief Signals that the mapper is exiting due to an error.
   * @param checker The checker instance to update.
   */
  static inline __device__ void mapperExit(SnappyCorrectnessChecker<CORRECTNESS_CHECK> *checker)
  {
    if constexpr (CORRECTNESS_CHECK)
    {
      if (checker->getThreadRank() == 0)
      {
        checker->counters[IX_MAPPER].store(INT32_MIN, cuda::std::memory_order_relaxed);
        checker->counters[IX_PROCESSOR].store(INT32_MIN, cuda::std::memory_order_relaxed);
      }
    }
  }

  /*!
   * @brief Stalls the processor until the mapper has mapped new symbols.
   * @param checker The checker instance to update.
   */
  static inline __device__ bool processorWaitForMapper(SnappyCorrectnessChecker<CORRECTNESS_CHECK> *checker)
  {
    if constexpr (CORRECTNESS_CHECK)
    {
      auto this_counter = checker->counters[IX_PROCESSOR].load(cuda::std::memory_order_relaxed);
      while (this_counter <= 0)
      {
        __nanosleep(SLEEP_DURATION);
        this_counter = checker->counters[IX_PROCESSOR].load(cuda::std::memory_order_relaxed);
        if (this_counter < 0)
        {
          return true;
        }
      }
    }
    return false;
  }

  /*!
   * @brief Signals that the processor has consumed a window.
   * @param checker The checker instance to update.
   */
  static inline __device__ void processorConsumeWindow(SnappyCorrectnessChecker<CORRECTNESS_CHECK> *checker)
  {
    if constexpr (CORRECTNESS_CHECK)
    {
      if (checker->getThreadRank() == 0)
      {
        checker->counters[IX_PROCESSOR]--;
      }
    }
  }

  inline __device__ int getThreadRank() const { return threadIdx.x % WARP_SIZE; }

  inline __device__ void sync() const { __syncwarp(); }

#endif // __CUDACC__

#ifndef NVCOMPDX
  __host__ std::string getErrorMessage()
  {
    switch (this->error_type_)
    {
      case SnappyCorrectnessErrorTypes::EMPTY_BLOCK:
        return "Empty block encountered";
      case SnappyCorrectnessErrorTypes::INVALID_UNCOMPRESSED_LENGTH:
        return "Reported uncompressed size is too large";
      case SnappyCorrectnessErrorTypes::COPY_AS_FIRST_SYMBOL:
        return "Copy operation as first symbol";
      case SnappyCorrectnessErrorTypes::WRONG_INPUT_LENGTH:
        return "Reported input size does not match the actual input size";
      case SnappyCorrectnessErrorTypes::WRONG_OUTPUT_LENGTH:
        return "Reported output size does not match the actual output size";
      case SnappyCorrectnessErrorTypes::INVALID_OFFSET:
        return "Invalid offset";
      case SnappyCorrectnessErrorTypes::OUT_OF_BOUNDS_WRITE:
        return "Out of bounds write operation";
      case SnappyCorrectnessErrorTypes::OUT_OF_BOUNDS_LITERAL_READ:
        return "Out of bounds literal read operation";
      case SnappyCorrectnessErrorTypes::OUTPUT_BUFFER_TOO_SMALL:
        return "Output buffer is too small for the reported decompressed size";
      default:
        return "Unknown error";
    }
  }

  static __host__ void printErrors(
    SnappyCorrectnessChecker<CORRECTNESS_CHECK> *d_correctness_checkers,
    const size_t batch_size,
    cudaStream_t stream
  )
  {
    if constexpr (CORRECTNESS_CHECK)
    {
      nvcomp::CorrectnessChecker<
        CORRECTNESS_CHECK,
        SnappyCorrectnessErrorTypes,
        uint32_t,
        SnappyCorrectnessChecker<CORRECTNESS_CHECK>>::
        printErrorsInternal(
          d_correctness_checkers,
          batch_size,
          stream,
          "snappy_correctness_check.log",
          SNAPPY_CORRECTNESS_LOG_OUTPUT_ENV,
          "Snappy"
        );
    }
  }
#endif // NVCOMPDX

private:
  const uint8_t *input_buffer_; // pointer to the input buffer
  uint32_t input_buffer_size_; // size of the input buffer
  uint32_t output_buffer_size_; // size of the output buffer
  uint32_t uncompressed_size_; // expected uncompressed size
  uint32_t header_size_; // size of the header (varints)
  /*
   * The first counter is for the finder to log how many symbols it has
   * produced. The second counter is for the mapper to log how many symbols it
   * has consumed from the finder. If negative, the mapper has exited due to an
   * error. The third counter is for the processor to log how many windows it
   * has processed. If the counter is 0 the processor has to wait for the mapper
   * to fill the next window. If the counter is 1 the mapper has to wait for the
   * processor to finish it's window successfully. A negative number indicates
   * that the mapper has exited due to an error. The mapping and processing is
   * done sequentially to avoid deadlocks in case one of them exits prematurely.
   */
  static constexpr int IX_FINDER = 0;
  static constexpr int IX_MAPPER = 1;
  static constexpr int IX_PROCESSOR = 2;
  static constexpr int SLEEP_DURATION = 100;
#ifdef NVCOMPDX
  static constexpr size_t SNAPPY_OFFSET_4BPB = 0; // Should not be used in nvCOMPDx
#else
  static constexpr size_t SNAPPY_OFFSET_4BPB = BIG_OFFSET;
#endif
#ifdef __CUDACC__
  cuda::atomic<int, cuda::thread_scope_block> counters[3];
#else
  int counters[3];
#endif
};

} // namespace snappy

namespace nvcomp_deflate
{
enum class DeflateCorrectnessErrorTypes
{
  EMPTY_BLOCK = 1,
  OUTPUT_BUFFER_TOO_SMALL = 2,
  UNEXPECTED_END_OF_CHUNK = 3,
  HCLEN_HDIST_TOO_BIG = 4,
  OVERSUBSCRIBED_DYNAMIC_HUFFMAN_TABLE = 5,
  INVALID_DYNAMIC_HUFFMAN_TABLE = 6,
  OUT_OF_BOUNDS_MATCH = 7,
  HUFFMAN_CODE_TOO_LONG = 8,
  INVALID_STORED_BLOCK_LENGTH = 9,
  INVALID_BLOCK_TYPE = 10,
  INVALID_MATCH_LENGTH = 11,
  INVALID_MATCH_OFFSET = 12,
  WRONG_GZIP_HEADER = 13,
};

struct inflate_state_s;

// Assumes that all functions except hasError are called by one thread
template <bool CORRECTNESS_CHECK>
class DeflateCorrectnessChecker
    : public nvcomp::CorrectnessChecker<
        CORRECTNESS_CHECK,
        DeflateCorrectnessErrorTypes,
        int64_t,
        DeflateCorrectnessChecker<CORRECTNESS_CHECK>>
{
public:
  DeflateCorrectnessChecker() = default;

#ifdef __CUDACC__

  /*!
   * @brief Initializes the correctness checker to its default values.
   * @param state_ptr Pointer to the inflate state structure.
   * @param input_buffer_start Pointer to the start of the input buffer.
   * @param checker The checker instance to initialize.
   */
  static inline __device__ void Initialize(
    inflate_state_s *state_ptr,
    const uint8_t *input_buffer_start,
    DeflateCorrectnessChecker<CORRECTNESS_CHECK> *checker
  )
  {
    if constexpr (CORRECTNESS_CHECK)
    {
      checker->line_number_ = -1; // -1 means no error
      checker->state_ptr_ = state_ptr;
      checker->input_buffer_start_ = input_buffer_start;
    }
  }

  /*!
   * @brief Sets the error information in the checker.
   * @param error_type The type of error encountered.
   * @param line_number The line number where the error occurred.
   * @param function_name The name of the function where the error occurred.
   * @param checker The checker instance to update.
   */
  static inline __device__ void setError(
    const DeflateCorrectnessErrorTypes error_type,
    const int line_number,
    const char *function_name,
    DeflateCorrectnessChecker<CORRECTNESS_CHECK> *checker
  )
  {
    if constexpr (CORRECTNESS_CHECK)
    {
      auto comp_id = reinterpret_cast<intptr_t>(checker->state_ptr_->cur) -
                     reinterpret_cast<intptr_t>(checker->input_buffer_start_);
      auto decomp_id = reinterpret_cast<intptr_t>(checker->state_ptr_->out) -
                       reinterpret_cast<intptr_t>(checker->state_ptr_->outbase);
      nvcomp::CorrectnessChecker<
        CORRECTNESS_CHECK,
        DeflateCorrectnessErrorTypes,
        int64_t,
        DeflateCorrectnessChecker<CORRECTNESS_CHECK>>::
        setError(error_type, comp_id, decomp_id, line_number, function_name, checker);
    }
  }

  /*!
   * @brief Sets an error if the condition is true.
   * @param condition The condition to check.
   * @param error_type The type of error to set if the condition is true.
   * @param line_number The line number where the check is performed.
   * @param function_name The name of the function where the check is performed.
   * @param checker The checker instance to update.
   * @return True if the condition is true and an error was set, false otherwise.
   */
  static inline __device__ bool errorIfTrue(
    const bool condition,
    const DeflateCorrectnessErrorTypes error_type,
    const int line_number,
    const char *function_name,
    DeflateCorrectnessChecker<CORRECTNESS_CHECK> *checker
  )
  {
    if constexpr (CORRECTNESS_CHECK)
    {
      auto comp_id = reinterpret_cast<intptr_t>(checker->state_ptr_->cur) -
                     reinterpret_cast<intptr_t>(checker->input_buffer_start_);
      auto decomp_id = reinterpret_cast<intptr_t>(checker->state_ptr_->out) -
                       reinterpret_cast<intptr_t>(checker->state_ptr_->outbase);
      return nvcomp::CorrectnessChecker<
        CORRECTNESS_CHECK,
        DeflateCorrectnessErrorTypes,
        int64_t,
        DeflateCorrectnessChecker<CORRECTNESS_CHECK>>::
        errorIfTrue(condition, error_type, comp_id, decomp_id, line_number, function_name, checker);
    }
    return false;
  }

  /*!
   * @brief Checks that input buffer size is not 0.
   * @param line_number The line number where the check is performed.
   * @param function_name The name of the function where the check is performed.
   * @param checker The checker instance to update.
   * @return True if an error was detected, false otherwise.
   */
  static inline __device__ bool preliminaryChecks(
    const int line_number,
    const char *function_name,
    DeflateCorrectnessChecker<CORRECTNESS_CHECK> *checker
  )
  {
    if constexpr (CORRECTNESS_CHECK)
    {
      // Check that the input size is not 0
      if (checker->state_ptr_->cur >= checker->state_ptr_->end)
      {
        setError(DeflateCorrectnessErrorTypes::EMPTY_BLOCK, line_number, function_name, checker);
        return true;
      }
    }
    return false;
  }

  /*!
   * @brief Validates a match operation during decompression.
   * @param ix_output The current output position in the decompressed buffer.
   * @param matchlen The length of the match sequence.
   * @param offset The backward offset to the match source.
   * @param line_number The line number where the check is performed.
   * @param function_name The name of the function where the check is performed.
   * @param checker The checker instance to update.
   * @return True if the match is valid, false otherwise.
   */
  static inline __device__ bool isValidMatch(
    const uint32_t ix_output,
    const uint16_t matchlen,
    const uint16_t offset,
    const int line_number,
    const char *function_name,
    DeflateCorrectnessChecker<CORRECTNESS_CHECK> *checker
  )
  {
    if constexpr (CORRECTNESS_CHECK)
    {
      // Check that the match does not go OOB
      if ((static_cast<int64_t>(ix_output) - offset < 0) ||
          (ix_output + matchlen > checker->state_ptr_->outend - checker->state_ptr_->outbase))
      {
        setError(DeflateCorrectnessErrorTypes::OUT_OF_BOUNDS_MATCH, line_number, function_name, checker);
        return false;
      }
      else if ((matchlen < 3) || (matchlen > 258))
      {
        // Check that the match length is at least 3
        setError(DeflateCorrectnessErrorTypes::INVALID_MATCH_LENGTH, line_number, function_name, checker);
        return false;
        // Note: this can never happen since the codes do not allow for it.
      }
      else if ((offset < 1) || (offset > 32'768))
      {
        // Check that the offset is not too big
        setError(DeflateCorrectnessErrorTypes::INVALID_MATCH_OFFSET, line_number, function_name, checker);
        return false;
      }
    }
    return true;
  }

  inline __device__ int getThreadRank() const { return threadIdx.x; }

  inline __device__ void sync() const
  {
    // Nothing since functions sometimes get called by 1 thread and others not
  }

#endif // __CUDACC__

#ifndef NVCOMPDX
  /*!
   * @brief Gets a human-readable error message for the current error type.
   * @return A string describing the error.
   */
  __host__ std::string getErrorMessage()
  {
    switch (this->error_type_)
    {
      case DeflateCorrectnessErrorTypes::EMPTY_BLOCK:
        return "Empty block encountered";
      case DeflateCorrectnessErrorTypes::OUTPUT_BUFFER_TOO_SMALL:
        return "Output buffer is too small";
      case DeflateCorrectnessErrorTypes::UNEXPECTED_END_OF_CHUNK:
        return "Unexpected end of chunk";
      case DeflateCorrectnessErrorTypes::HCLEN_HDIST_TOO_BIG:
        return "HCLEN or HDIST is too big";
      case DeflateCorrectnessErrorTypes::OVERSUBSCRIBED_DYNAMIC_HUFFMAN_TABLE:
        return "Oversubscribed dynamic Huffman table";
      case DeflateCorrectnessErrorTypes::INVALID_DYNAMIC_HUFFMAN_TABLE:
        return "Invalid dynamic Huffman table";
      case DeflateCorrectnessErrorTypes::OUT_OF_BOUNDS_MATCH:
        return "Out of bounds match";
      case DeflateCorrectnessErrorTypes::HUFFMAN_CODE_TOO_LONG:
        return "Huffman code too long";
      case DeflateCorrectnessErrorTypes::INVALID_STORED_BLOCK_LENGTH:
        return "Invalid stored block length";
      case DeflateCorrectnessErrorTypes::INVALID_BLOCK_TYPE:
        return "Invalid block type";
      case DeflateCorrectnessErrorTypes::INVALID_MATCH_LENGTH:
        return "Invalid match length";
      case DeflateCorrectnessErrorTypes::INVALID_MATCH_OFFSET:
        return "Invalid match offset";
      case DeflateCorrectnessErrorTypes::WRONG_GZIP_HEADER:
        return "Wrong GZIP header";
      default:
        return "Unknown error";
    }
  }

  /*!
   * @brief Prints the error information for each checker in the batch.
   * @param d_correctness_checkers Pointer to the device array of checkers.
   * @param batch_size The number of checkers in the batch.
   * @param stream The CUDA stream to use for asynchronous operations.
   */
  static __host__ void printErrors(
    DeflateCorrectnessChecker<CORRECTNESS_CHECK> *d_correctness_checkers,
    const size_t batch_size,
    cudaStream_t stream
  )
  {
    if constexpr (CORRECTNESS_CHECK)
    {
      nvcomp::CorrectnessChecker<
        CORRECTNESS_CHECK,
        DeflateCorrectnessErrorTypes,
        int64_t,
        DeflateCorrectnessChecker<CORRECTNESS_CHECK>>::
        printErrorsInternal(
          d_correctness_checkers,
          batch_size,
          stream,
          "deflate_correctness_check.log",
          DEFLATE_CORRECTNESS_LOG_OUTPUT_ENV,
          "Deflate"
        );
    }
  }
#endif // NVCOMPDX

private:
  inflate_state_s *state_ptr_; // Pointer to the inflate state
  const uint8_t *input_buffer_start_; // Pointer to the start of the input buffer
};

} // namespace nvcomp_deflate
