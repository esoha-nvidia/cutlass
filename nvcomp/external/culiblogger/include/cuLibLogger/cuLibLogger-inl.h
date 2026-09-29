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

#include "cuLibLogger.h"

#include "utils.h"

LOGGER_BEGIN_NAMESPACE

// This is required because c++14 and below do not allow inline variables,
// therefore, we cannot declare it as static member of the logger class.
template <typename T>
struct TLS {
    static thread_local T mostRecentApi;
};

template <typename T>
thread_local T TLS<T>::mostRecentApi = "";

inline std::string CreateLogFileName() {
    const char* logFile = utils::GetEnv(LOGGER_LOG_FILE_ENV);
    if (logFile && strcmp(logFile, "")) {
        const int pid = utils::GetPid();
        return fmt::sprintf(logFile, pid);
    }

    return {};
}

inline Logger& Logger::Instance() {
    static Logger logger;
    return logger;
}

template<typename... Args>
inline void Logger::Log(Level logLevel, Mask logMask, const fmt::string_view& format, const Args&... args) {
    if (!ShouldLogMessage(logLevel, logMask)) {
        return;
    }

    Log(TLS<const char*>::mostRecentApi, -1, logLevel, logMask, format, args...);
}

template<typename... Args>
inline void Logger::Log(const char* api, int line, Level logLevel, Mask logMask, const fmt::string_view& format, const Args&... args) {
    if (!ShouldLogMessage(logLevel, logMask)) {
        return;
    }

    const std::string& message = fmt::format(format, args...);

    if (callback != nullptr) {
        callback(static_cast<int>(logLevel), api, message.c_str());
    }

    if (callbackData != nullptr) {
        callbackData(static_cast<int>(logLevel), api, message.c_str(), userData);
    }

    OutputBuffer out;
    Format(out, api, line, logLevel, message);
    LogSink::Instance().Log(fmt::string_view(out.data(), out.size()));
}

inline void Logger::SetMostRecentApi(const char* api) {
    if (!IsEnabled()) {
        return;
    }

    TLS<const char*>::mostRecentApi = api;
}

inline bool Logger::IsEnabled() {
#if LOG_ENABLED
    return !forceDisableLogger && (level != Level::Off || mask != Mask::OffMask);
#else
    return false;
#endif
}

inline bool Logger::ShouldLogMessage(Level level, Mask mask) {
#if LOG_ENABLED
    return !forceDisableLogger && ((level <= this->level) || ((mask & this->mask) != Mask::OffMask));
#else
    return false;
#endif
}

inline Logger::Logger() {
#if LOG_ENABLED
    const char* logLevel = utils::GetEnv(LOGGER_LOG_LEVEL_ENV);
    const char* logMask = utils::GetEnv(LOGGER_LOG_MASK_ENV);
    if (logLevel || logMask) {
        if (logLevel && strcmp(logLevel, "")) {
            SetLevel(static_cast<Level>(std::atoi(logLevel)));
        }
        else if (logMask && strcmp(logMask, "")) {
            SetMask(static_cast<Mask>(std::atoi(logMask)));
        }
        
        if (level != Level::Off || mask != Mask::OffMask) {
            // Init the log sink
            LogSink::Instance();
        }
    }
#endif
}

inline Logger::Status Logger::SetCallback(CallbackType callback) {
    this->callback = callback;
    return Status::Success;
}

inline Logger::Status Logger::SetCallbackData(CallbackDataType callbackData, void* userData) {
    this->callbackData = callbackData;
    this->userData = userData;
    return Status::Success;
}

inline Logger::Status Logger::SetFile(FILE* file) {
    return LogSink::Instance().SetFile(file);
}

inline Logger::Status Logger::OpenFile(const char* logFile) {
    return LogSink::Instance().OpenFile(logFile);
}

inline Logger::Status Logger::ForceDisable() {
    forceDisableLogger = true;
    return Status::Success;
}

inline Logger::Status Logger::SetLevel(Level level) {
    switch (level) {
        case Level::Error:
        case Level::Trace:
        case Level::Hint:
        case Level::Info:
        case Level::Api:
        case Level::Debug:
        case Level::Off:
            this->level = level;
            this->mask = Mask::OffMask;
            return Status::Success;
        default:
            this->level = Level::Off;
            return Status::InvalidValue;
    }
}

inline Logger::Level Logger::GetLevel() {
    return level;
}

inline Logger::Status Logger::SetMask(Mask mask) {
    this->mask = mask;
    this->level = Level::Off;
    return Status::Success;
}

inline Logger::Mask Logger::GetMask() {
    return mask;
}

inline void Logger::GetFormattedTime(OutputBuffer& out) {
    std::time_t now = std::time(nullptr);
#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable:4996)
#endif
    const std::tm& tm = *std::localtime(&now);
#ifdef _WIN32
#pragma warning(pop)
#endif
    fmt::format_to(std::back_inserter(out), "[{:%Y-%m-%d %H:%M:%S}]", tm);
}

inline const char* Logger::GetLevelText(Level logLevel) {
    switch (logLevel) {
        case Level::Off:
            return "Off";
        case Level::Error:
            return "Error";
        case Level::Trace:
            return "Trace";
        case Level::Hint:
            return "Hint";
        case Level::Info:
            return "Info";
        case Level::Api:
            return "Api";
        case Level::Debug:
            return "Debug";
        default:
            return "Invalid log level";
    }
}

inline void Logger::Format(OutputBuffer& out, const char* api, int line, Level logLevel, const std::string& message) {
    // format "[time][logger name][thread][level][api name] message"
    assert(api[0] != '\0');
    GetFormattedTime(out);
    fmt::format_to(std::back_inserter(out), "[{}][{}][{}][{}]", name, utils::GetTid(), GetLevelText(logLevel), api);

#ifndef NDEBUG
    if (logLevel == Level::Debug)
    {
        fmt::format_to(std::back_inserter(out), "[{}]", line);
    }
#else
    (void)line;
#endif

    fmt::format_to(std::back_inserter(out), " {}\n", message);
}

// LogSink
inline Logger::LogSink& Logger::LogSink::Instance() {
    static LogSink logSink;
    return logSink;
}

inline void Logger::LogSink::Log(const fmt::string_view& message) {
    if (!file) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex);
    fwrite(message.data(), 1, message.size(), file);
    fflush(file);
}

inline Logger::LogSink::LogSink() {
    const std::string& logFile = CreateLogFileName();
    if (logFile.empty())
    {
        SetFile(stdout);
    }
    else
    {
        OpenFile(logFile.c_str());
    }
}

inline Logger::LogSink::~LogSink() {
    CloseFile();
}

inline Logger::Status Logger::LogSink::OpenFile(const char* fileName) {
    CloseFile();
    if (fileName) {
#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable:4996)
#endif
        file = fopen(fileName, "w");
#ifdef _WIN32
#pragma warning(pop)
#endif
        if (file) {
            shouldCloseFile = true;
        }
        else {
            return Logger::Status::InvalidValue;
        }
    }

    return Logger::Status::Success;
}

inline Logger::Status Logger::LogSink::SetFile(FILE* file) {
    CloseFile();
    this->file = file;
    shouldCloseFile = false;
    return Logger::Status::Success;
}

inline void Logger::LogSink::CloseFile() {
    if (file) {
        fflush(file);
        if (shouldCloseFile) {
            fclose(file);
        }
        file = nullptr;
    }
}

LOGGER_END_NAMESPACE
