/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <cuda_runtime.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "device_guard.h"
#include "lookahead_gzip.h"
#include "nvcomp.h"
#include "progress_stream.hpp"

#include <cuLibLogger/cuLibLogger.h>
#include <exception.hpp>
#include <nvcomp/native/streaming_gzip.hpp>

constexpr const char *const REQUIRED_PARAMETER = "_REQUIRED_";

struct parameter_type
{
  std::string short_flag;
  std::string long_flag;
  std::string description;
  std::string default_value;
};

struct args_type
{
  bool decompress = false;
  bool compress = false;
  std::string input_file;
  std::string output_file;
  bool enable_progress = false;
  int device_id = 0;
  int algorithm = nvcompBatchedGzipCompressDefaultOpts.algorithm;

  args_type()
      : input_file()
      , output_file()
  {}

  ~args_type() = default;
};

void usage(const std::string &name, const std::vector<parameter_type> &parameters)
{
  std::cout << "Usage: " << name << " [OPTIONS]" << std::endl;
  for (const parameter_type &parameter : parameters)
  {
    std::cout << "  -" << parameter.short_flag << ", --" << parameter.long_flag;
    std::cout << " : " << parameter.description << std::endl;
    if (parameter.default_value == REQUIRED_PARAMETER)
    {
      std::cout << "    required" << std::endl;
    }
    else if (!parameter.default_value.empty())
    {
      std::cout << "    default=" << parameter.default_value << std::endl;
    }
  }
}

args_type parse_args(int argc, char **argv)
{
  args_type args;

  const std::vector<parameter_type> params{
    {"h", "help", "Show options.", ""},
    {"d", "decompress", "Switch nvlzcat into decompression mode (default).", ""},
    {"c", "compress", "Switch nvlzcat into compression mode. It will output gzip compliant buffer on stdout", ""},
    {"f",
     "input_file",
     "The input file to be compressed/decompressed. If no file is specified, "
     "then std::cin is considered as input.",
     ""},
    {"o",
     "output_file",
     "The output file for the compressed/decompressed output. If no file is "
     "specified, then std::cout is considered as output.",
     ""},
    {"p", "progress", "Signal progress. Only supported if input file is specified.", ""},
    {"a",
     "algorithm",
     "Gzip compression level. Only supported in compression mode.\n"
     "        0: highest-throughput, lowest compression ratio - entropy-only compression\n"
     "        1: high-throughput, low compression ratio (default)\n"
     "        2: medium-throughput, medium compression ratio, beats Zlib level 1 on ratio\n"
     "        3: placeholder for further compression levels, currently maps to level 2\n"
     "        4: lower-throughput, higher compression ratio, beats Zlib level 6 on ratio\n"
     "        5: lowest-throughput, highest compression ratio",
     ""},
    {"g", "gpu", "Target GPU to run the decompression, defaults to 0", ""},
  };

  char **argv_end = argv + argc;
  const std::string name(argv[0]);
  argv += 1;

  while (argv != argv_end)
  {
    std::string arg(*(argv++));
    bool found = false;
    for (const parameter_type &param : params)
    {
      if (arg == "-" + param.short_flag || arg == "--" + param.long_flag)
      {
        found = true;

        // found the parameter
        if (param.long_flag == "help")
        {
          usage(name, params);
          std::exit(0);
        }

        if (param.long_flag == "progress")
        {
          args.enable_progress = true;
          break;
        }

        if (param.long_flag == "decompress")
        {
          args.decompress = true;
          break;
        }

        if (param.long_flag == "compress")
        {
          args.compress = true;
          break;
        }

        // Remaining flags require additional parameter
        if (argv >= argv_end)
        {
          usage(name, params);
          throw std::runtime_error("Missing argument for '" + arg + "'");
        }

        if (param.long_flag == "gpu")
        {
          try
          {
            args.device_id = std::stoi(*(argv++));
          }
          catch (const std::exception &e)
          {
            throw std::runtime_error(std::string("Invalid GPU device ID: ") + e.what());
          }
          break;
        }

        if (param.long_flag == "algorithm")
        {
          try
          {
            args.algorithm = std::stoi(*(argv++));
          }
          catch (const std::exception &e)
          {
            throw std::runtime_error(std::string("Invalid gzip compression algorithm: ") + e.what());
          }
          break;
        }

        if (param.long_flag == "input_file")
        {
          args.input_file = *(argv++);
          if (!std::filesystem::exists(args.input_file))
          {
            throw std::runtime_error("Input file does not exist '" + args.input_file + "'");
          }
          break;
        }
        else if (param.long_flag == "output_file")
        {
          args.output_file = *(argv++);
          break;
        }
        else
        {
          usage(name, params);
          throw std::runtime_error("Unhandled parameter '" + arg + "'");
        }
      }
    }
    if (!found)
    {
      usage(name, params);
      throw std::runtime_error("Unknown argument '" + arg + "'");
    }
  }

  if (args.enable_progress && args.input_file.empty())
  {
    throw std::runtime_error("Progress flag requires input file to be specified");
  }

  return args;
}

// Selects the temp-size and run entry points for a given streaming mode.
struct CompressMode
{
  nvcompBatchedGzipCompressOpts_t opts = nvcompBatchedGzipCompressDefaultOpts;

  nvcompStatus_t getTempSize(size_t *bytes) const { return nvcompGzipStreamingCompressGetTempSize(opts, bytes); }
  nvcompStatus_t
  run(std::istream &input, std::ostream &output, size_t temp_bytes, void *d_temp, cudaStream_t stream) const
  {
    return nvcompGzipStreamingCompress(input, output, temp_bytes, d_temp, opts, stream);
  }
};

struct DecompressMode
{
  nvcompStatus_t getTempSize(size_t *bytes) const { return nvcompGzipStreamingDecompressGetTempSize(bytes); }
  nvcompStatus_t
  run(std::istream &input, std::ostream &output, size_t temp_bytes, void *d_temp, cudaStream_t stream) const
  {
    return nvcompGzipStreamingDecompress(input, output, temp_bytes, d_temp, stream);
  }
};

// RAII streaming context shared by both modes: sizes and allocates the device workspace + a stream, and
// releases them on destruction. `Mode` is CompressMode or DecompressMode.
template <typename Mode>
class StreamingContext
{
  void destroy() noexcept
  {
    if (d_temp)
    {
      CUDA_CHECK_LOG(cudaFree(d_temp));
    }
    if (stream)
    {
      CUDA_CHECK_LOG(cudaStreamDestroy(stream));
    }
  }

  Mode mode;

public:
  cudaStream_t stream = nullptr;
  void *d_temp = nullptr;
  size_t temp_bytes = 0;

  explicit StreamingContext(Mode mode = Mode{})
      : mode(mode)
  {
    try
    {
      CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

      if (mode.getTempSize(&temp_bytes) != nvcompSuccess)
      {
        throw std::runtime_error("Failed to determine required GPU memory");
      }

      CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));
    }
    catch (...)
    {
      destroy();
      throw;
    }
  }

  StreamingContext(const StreamingContext &) = delete;
  StreamingContext &operator=(const StreamingContext &) = delete;

  ~StreamingContext() noexcept { destroy(); }

  // Run the mode's streaming op over this context's workspace.
  nvcompStatus_t run(std::istream &input, std::ostream &output)
  {
    return mode.run(input, output, temp_bytes, d_temp, stream);
  }
};

int main(int argc, char *argv[])
{
  try
  {
    args_type args = parse_args(argc, argv);

    nvcomp::DeviceGuard guard(args.device_id);

    std::unique_ptr<std::istream> file_stream;
    if (!args.input_file.empty())
    {
      if (args.enable_progress)
      {
        file_stream = std::make_unique<ProgressIfstream>(args.input_file);
      }
      else
      {
        file_stream = std::make_unique<std::ifstream>(args.input_file);
      }
    }
    else
    {
      if (isatty(STDIN_FILENO))
      {
        // stdin is a terminal, mimic zcat behavior
        throw std::runtime_error("Compressed data not read from a terminal. Try piping.");
      }
    }

    std::istream &input_stream = (file_stream) ? *file_stream : std::cin;

    std::ofstream output_file_stream;
    if (!args.output_file.empty())
    {
      output_file_stream.open(args.output_file);
    }
    std::ostream &output_stream = output_file_stream.is_open() ? output_file_stream : std::cout;

    if (args.compress && args.decompress)
    {
      throw std::runtime_error(
        std::string("nvlzcat invoked with both `-c` and `-d` flags. Please select either compression or decompression.")
      );
    }
    else if (args.compress)
    {
      // Run compression
      nvcompBatchedGzipCompressOpts_t opts = nvcompBatchedGzipCompressDefaultOpts;
      opts.algorithm = args.algorithm;
      StreamingContext<CompressMode> ctx{CompressMode{opts}};
      nvcompStatus_t status = ctx.run(input_stream, output_stream);

      if (status != nvcompSuccess)
      {
        throw std::runtime_error(std::string("Compression failed - ") + nvcompGetStatusString(status));
      }
    }
    else
    {
      // Run decompression (explicit `-d`, or the default when neither flag is given)
      StreamingContext<DecompressMode> ctx{};
      nvcompStatus_t status = ctx.run(input_stream, output_stream);

      if (status != nvcompSuccess)
      {
        throw std::runtime_error(std::string("Decompression failed - ") + nvcompGetStatusString(status));
      }
    }

    return EXIT_SUCCESS;
  }
  catch (const std::exception &e)
  {
    std::cerr << "nvlzcat: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }
}
