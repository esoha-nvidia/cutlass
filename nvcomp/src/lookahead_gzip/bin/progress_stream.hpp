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

#pragma once

#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <streambuf>
#include <string>

// RAII guard that saves and restores the full formatting state of a stream.
class FormatGuard
{
public:
  explicit FormatGuard(std::ostream &os)
      : os_(os)
      , state_(nullptr)
  {
    state_.copyfmt(os_);
  }

  ~FormatGuard() { os_.copyfmt(state_); }

  FormatGuard(const FormatGuard &) = delete;
  FormatGuard &operator=(const FormatGuard &) = delete;

private:
  std::ostream &os_;
  std::ios state_;
};

// Helper function to format bytes in human-readable format
inline std::string formatBytes(size_t bytes)
{
  const char *units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
  int unit = 0;
  double size = static_cast<double>(bytes);

  while (size >= 1024.0 && unit < 4)
  {
    size /= 1024.0;
    unit++;
  }

  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2) << std::setw(7) << size << " " << units[unit];
  return oss.str();
}

// Progress bar display function
inline void showProgressBar(size_t bytes_read, size_t total_bytes)
{
  static bool cursor_hidden = false;

  // Hide cursor on first call
  if (!cursor_hidden)
  {
    std::cerr << "\033[?25l"; // Hide cursor
    cursor_hidden = true;
  }

  const int bar_width = 50;
  double progress = total_bytes > 0 ? static_cast<double>(bytes_read) / total_bytes : 0.0;
  int filled = static_cast<int>(bar_width * progress);

  std::cerr << "\r[";
  for (int i = 0; i < bar_width; ++i)
  {
    if (i < filled)
    {
      std::cerr << "=";
    }
    else if (i == filled && filled < bar_width)
    {
      std::cerr << ">";
    }
    else
    {
      std::cerr << " ";
    }
  }

  {
    FormatGuard guard(std::cerr);
    std::cerr << "] " << std::fixed << std::setprecision(1) << std::setw(5) << (progress * 100.0) << "% "
              << formatBytes(bytes_read) << "/" << formatBytes(total_bytes);
  }
  std::cerr << std::flush;

  // Show cursor when complete
  if (bytes_read >= total_bytes)
  {
    std::cerr << "\033[?25h";
    cursor_hidden = false;
  }
}

// Custom streambuf that reports read progress
class ReportingStreamBuf : public std::streambuf
{
private:
  std::streambuf *backend_;
  std::function<void(size_t, size_t)> report_callback_;
  size_t bytes_read_ = 0;
  size_t total_bytes_ = 0;
  bool reported_ = false;

protected:
  int_type underflow() override
  {
    int_type c = backend_->sgetc();
    if (c != EOF)
    {
      bytes_read_++;
      reported_ = true;
      if (report_callback_)
      {
        report_callback_(bytes_read_, total_bytes_);
      }
    }
    return c;
  }

  int_type uflow() override
  {
    int_type c = backend_->sbumpc();
    if (c != EOF)
    {
      bytes_read_++;
      reported_ = true;
      if (report_callback_)
      {
        report_callback_(bytes_read_, total_bytes_);
      }
    }
    return c;
  }

  // bulk read
  std::streamsize xsgetn(char *s, std::streamsize n) override
  {
    std::streamsize read = backend_->sgetn(s, n);
    if (read > 0)
    {
      bytes_read_ += read;
      reported_ = true;
      if (report_callback_)
      {
        report_callback_(bytes_read_, total_bytes_);
      }
    }
    return read;
  }

public:
  ReportingStreamBuf(std::streambuf *buf, std::function<void(size_t, size_t)> callback, size_t total)
      : backend_(buf)
      , report_callback_(callback)
      , total_bytes_(total) {};

  ~ReportingStreamBuf()
  {
    if (reported_)
    {
      std::cerr << "\033[?25h"; // Show cursor
      std::cerr << std::endl; // Final newline to clean up progress line
    }
  }

  ReportingStreamBuf(const ReportingStreamBuf &) = delete;
  ReportingStreamBuf &operator=(const ReportingStreamBuf &) = delete;

  ReportingStreamBuf(ReportingStreamBuf &&) = default;
  ReportingStreamBuf &operator=(ReportingStreamBuf &&) = default;

  size_t getBytesRead() const { return bytes_read_; };
};

// Wrapper class for convenient usage with progress reporting
class ProgressIfstream : public std::istream
{
  std::ifstream file_;
  size_t file_size_;
  ReportingStreamBuf buf_;

public:
  ProgressIfstream(const std::string &filename)
      : std::istream(&buf_)
      , file_(filename, std::ios::binary | std::ios::ate)
      , file_size_(file_.tellg())
      , buf_(
          file_.rdbuf(),
          [last_update = size_t(0)](size_t bytes, size_t total) mutable {
            // Update if we've read 1% more or 1MB more, or if we're done
            if (bytes - last_update >= total / 100 || bytes - last_update >= 1024 * 1024 || bytes >= total)
            {
              showProgressBar(bytes, total);
              last_update = bytes;
            }
          },
          file_size_
        )
  {
    file_.seekg(0);
  };

  ProgressIfstream(const ProgressIfstream &) = delete;
  ProgressIfstream(ProgressIfstream &&) = delete;
  ProgressIfstream &operator=(const ProgressIfstream &) = delete;
  ProgressIfstream &operator=(ProgressIfstream &&) = delete;

  bool is_open() const { return file_.is_open(); };
  size_t bytes_read() const { return buf_.getBytesRead(); };
};
