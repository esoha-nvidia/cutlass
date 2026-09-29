/*
 * Copyright 1993-2020 NVIDIA Corporation. All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */
#pragma once

#include <string.h>
#include <assert.h>

#include <mutex>
#include <string>
#include <functional>

#ifndef LOGGER_FMT_PREINCLUDED
#define FMT_USE_WINDOWS_H 0
#include "fmt/format.h"
#include "fmt/chrono.h"
#include "fmt/printf.h"
#endif

#ifndef LOG_ENABLED
#define LOG_ENABLED 1
#endif

#ifndef LOGGER_NAME
#define LOGGER_NAME "CUDA LIB"
#endif

#ifndef LOGGER_LOG_LEVEL_ENV
#define LOGGER_LOG_LEVEL_ENV "CUDA_LIB_LOG_LEVEL"
#endif

#ifndef LOGGER_LOG_MASK_ENV
#define LOGGER_LOG_MASK_ENV "CUDA_LIB_LOG_MASK"
#endif

#ifndef LOGGER_LOG_FILE_ENV
#define LOGGER_LOG_FILE_ENV "CUDA_LIB_LOG_FILE"
#endif

#ifndef LOGGER_BEGIN_NAMESPACE
#define LOGGER_BEGIN_NAMESPACE \
  namespace cuLibLogger {
#endif

#ifndef LOGGER_END_NAMESPACE
#define LOGGER_END_NAMESPACE \
  }
#endif

#ifndef LOGGER_MAX_BUFFER_SIZE
#define LOGGER_MAX_BUFFER_SIZE 2048
#endif

LOGGER_BEGIN_NAMESPACE

class Logger {
public:
    enum Status {
        Success = 0,
        InvalidValue = 1
    };

    enum Level {
        Off = 0,
        Error = 1,
        Trace = 2,
        Hint = 3,
        Info = 4,
        Api = 5,
        Debug = 6
    };

    enum Mask {
        OffMask = 0,
        ErrorMask = 1 << (Level::Error - 1),
        TraceMask = 1 << (Level::Trace - 1),
        HintMask = 1 << (Level::Hint - 1),
        InfoMask = 1 << (Level::Info - 1),
        ApiMask = 1 << (Level::Api - 1),
        DebugMask = 1 << (Level::Debug - 1)
    };

    using CallbackType = std::function<void(int, const char*, const char*)>;
    using CallbackDataType = std::function<void(int, const char*, const char*, void*)>;

    Logger(const Logger& other) = delete;
    Logger(Logger&& other) = delete;
    const Logger& operator = (const Logger& other) = delete;
    const Logger& operator = (Logger&& other) = delete;

    static inline Logger& Instance();
    template<typename... Args>
    inline void Log(Level logLevel, Mask logMask, const fmt::string_view& format, const Args&... args);
    template<typename... Args>
    inline void Log(const char* api, int line, Level logLevel, Mask logMask, const fmt::string_view& format, const Args&... args);
    inline void SetMostRecentApi(const char* api);
    inline bool IsEnabled();
    inline bool ShouldLogMessage(Level level, Mask mask);
    inline Status SetCallback(CallbackType callback);
    inline Status SetCallbackData(CallbackDataType callbackData, void* userData);
    inline Status SetFile(FILE* logFile);
    inline Status OpenFile(const char* logFile);
    inline Status SetLevel(Level level);
    inline Level GetLevel();
    inline Status SetMask(Mask mask);
    inline Mask GetMask();
    inline Status ForceDisable();

private:
    class LogSink {
    public:
        static inline LogSink& Instance();
        inline void Log(const fmt::string_view& message);
        inline Status OpenFile(const char* fileName);
        inline Status SetFile(FILE* file);

    private:
        LogSink();
        ~LogSink();

        inline void CloseFile();

        std::mutex mutex;
        std::FILE* file = nullptr;
        bool shouldCloseFile = false;
    };

    using OutputBuffer = fmt::basic_memory_buffer<char, LOGGER_MAX_BUFFER_SIZE>;

    Logger();

    inline void GetFormattedTime(OutputBuffer& out);
    inline const char* GetLevelText(Level logLevel);
    inline void Format(OutputBuffer& out, const char* api, int line, Level logLevel, const std::string& message);

    CallbackType callback = nullptr;
    CallbackDataType callbackData = nullptr;
    Level level = Level::Off;
    Mask mask = Mask::OffMask;
    bool forceDisableLogger = false;
    const std::string name = LOGGER_NAME;
    void* userData = nullptr;
}; // class Logger

LOGGER_END_NAMESPACE

#include "cuLibLogger-inl.h"

// Logger API
#if LOG_ENABLED

LOGGER_BEGIN_NAMESPACE

inline cuLibLogger::Logger::Status loggerSetCallback(cuLibLogger::Logger::CallbackType callback) {
    auto& logger = cuLibLogger::Logger::Instance();
    return logger.SetCallback(callback);
}

inline cuLibLogger::Logger::Status loggerSetCallbackData(cuLibLogger::Logger::CallbackDataType callback, void* userData) {
    auto& logger = cuLibLogger::Logger::Instance();
    return logger.SetCallbackData(callback, userData);
}

inline cuLibLogger::Logger::Status loggerSetFile(FILE* file) {
    auto& logger = cuLibLogger::Logger::Instance();
    return logger.SetFile(file);
}

inline cuLibLogger::Logger::Status loggerOpenFile(const char* logFile) {
    auto& logger = cuLibLogger::Logger::Instance();
    return logger.OpenFile(logFile);
}

inline cuLibLogger::Logger::Status loggerSetLevel(cuLibLogger::Logger::Level level) {
    auto& logger = cuLibLogger::Logger::Instance();
    return logger.SetLevel(level);
}

inline cuLibLogger::Logger::Status loggerSetMask(cuLibLogger::Logger::Mask mask) {
    auto& logger = cuLibLogger::Logger::Instance();
    return logger.SetMask(mask);
}

inline cuLibLogger::Logger::Status loggerForceDisable() {
    auto& logger = cuLibLogger::Logger::Instance();
    return logger.ForceDisable();
}

LOGGER_END_NAMESPACE

#define LOG_ERROR(...)                                                                                                          \
  do {                                                                                                                          \
    auto& logger = cuLibLogger::Logger::Instance();                                                                             \
    if (logger.ShouldLogMessage(cuLibLogger::Logger::Level::Error, cuLibLogger::Logger::Mask::ErrorMask)) {                     \
      logger.Log(cuLibLogger::Logger::Level::Error, cuLibLogger::Logger::Mask::ErrorMask, __VA_ARGS__);                         \
    }                                                                                                                           \
  } while (0)

#define LOG_TRACE(...)                                                                                                          \
  do {                                                                                                                          \
    auto& logger = cuLibLogger::Logger::Instance();                                                                             \
    if (logger.ShouldLogMessage(cuLibLogger::Logger::Level::Trace, cuLibLogger::Logger::Mask::TraceMask)) {                     \
      logger.Log(cuLibLogger::Logger::Level::Trace, cuLibLogger::Logger::Mask::TraceMask, __VA_ARGS__);                         \
    }                                                                                                                           \
  } while (0)

#define LOG_HINT(...)                                                                                                           \
  do {                                                                                                                          \
    auto& logger = cuLibLogger::Logger::Instance();                                                                             \
    if (logger.ShouldLogMessage(cuLibLogger::Logger::Level::Hint, cuLibLogger::Logger::Mask::HintMask)) {                       \
      logger.Log(cuLibLogger::Logger::Level::Hint, cuLibLogger::Logger::Mask::HintMask, __VA_ARGS__);                           \
    }                                                                                                                           \
  } while (0)

#define LOG_INFO(...)                                                                                                           \
  do {                                                                                                                          \
    auto& logger = cuLibLogger::Logger::Instance();                                                                             \
    if (logger.ShouldLogMessage(cuLibLogger::Logger::Level::Info, cuLibLogger::Logger::Mask::InfoMask)) {                       \
      logger.Log(cuLibLogger::Logger::Level::Info, cuLibLogger::Logger::Mask::InfoMask, __VA_ARGS__);                           \
    }                                                                                                                           \
  } while (0)

#define LOG_API(...)                                                                                                            \
  do {                                                                                                                          \
    auto& logger = cuLibLogger::Logger::Instance();                                                                             \
    logger.SetMostRecentApi(__FUNCTION__);                                                                                      \
    if (logger.ShouldLogMessage(cuLibLogger::Logger::Level::Api, cuLibLogger::Logger::Mask::ApiMask)) {                         \
      logger.Log(cuLibLogger::Logger::Level::Api, cuLibLogger::Logger::Mask::ApiMask, __VA_ARGS__);                             \
    }                                                                                                                           \
  } while (0)

#ifndef NDEBUG
#define LOG_DEBUG(...)                                                                                                          \
  do {                                                                                                                          \
    auto& logger = cuLibLogger::Logger::Instance();                                                                             \
    if (logger.ShouldLogMessage(cuLibLogger::Logger::Level::Debug, cuLibLogger::Logger::Mask::DebugMask)) {                     \
      logger.Log(__FUNCTION__, __LINE__, cuLibLogger::Logger::Level::Debug, cuLibLogger::Logger::Mask::DebugMask, __VA_ARGS__); \
    }                                                                                                                           \
  } while (0)
#else
#define LOG_DEBUG(...)
#endif  // DEBUG

#else

LOGGER_BEGIN_NAMESPACE

inline cuLibLogger::Logger::Status loggerSetCallback(cuLibLogger::Logger::CallbackType /*callback*/) { return cuLibLogger::Logger::Status::Success; }
inline cuLibLogger::Logger::Status loggerSetCallbackData(cuLibLogger::Logger::CallbackDataType /*callback*/, void* /*userData*/) { return cuLibLogger::Logger::Status::Success; }
inline cuLibLogger::Logger::Status loggerSetFile(FILE* /*file*/) { return cuLibLogger::Logger::Status::Success; }
inline cuLibLogger::Logger::Status loggerOpenFile(const char* /*logFile*/) { return cuLibLogger::Logger::Status::Success; }
inline cuLibLogger::Logger::Status loggerSetLevel(cuLibLogger::Logger::Level /*level*/) { return cuLibLogger::Logger::Status::Success; }
inline cuLibLogger::Logger::Status loggerSetMask(cuLibLogger::Logger::Mask /*mask*/) { return cuLibLogger::Logger::Status::Success; }
inline cuLibLogger::Logger::Status loggerForceDisable() { return cuLibLogger::Logger::Status::Success;}

LOGGER_END_NAMESPACE

#define LOG_ERROR(...)
#define LOG_TRACE(...)
#define LOG_HINT(...)
#define LOG_INFO(...)
#define LOG_API(...)
#define LOG_DEBUG(...)

#endif
