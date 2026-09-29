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

#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvToolsExtCuda.h>

#include "cuLibLogger.h"

#ifndef NVTX_DOMAIN_NAME
#define NVTX_DOMAIN_NAME "CUDA LIB"
#endif

#ifndef NVTX_LEVEL_ENV
#define NVTX_LEVEL_ENV "NVTX_LEVEL"
#endif

#ifndef NVTX_INTERNAL_SUPPORT_ENV
#define NVTX_INTERNAL_SUPPORT_ENV "NVTX_INTERNAL_SUPPORT"
#endif

LOGGER_BEGIN_NAMESPACE

class Nvtx {
public:
    enum Level {
        Off = 0,
        Trace = 1,
        Api = 2,
    };

    static inline Nvtx& Instance() {
        static Nvtx nvtx;
        return nvtx;
    }

    inline bool IsInternalNvtxEnabled() {
        return isInternalNvtxEnabled;
    }

    inline bool ShouldLogEvent(Level level) {
        return level <= this->level;
    }

    inline void Mark(nvtxStringHandle_t text) {
        nvtxEventAttributes_t eventAttrib = PrepareEventAttribute(text);
        nvtxDomainMarkEx(defaultDomain, &eventAttrib);
    }

    inline void RangePush(nvtxStringHandle_t text) {
        nvtxEventAttributes_t eventAttrib = PrepareEventAttribute(text);
        nvtxDomainRangePushEx(defaultDomain, &eventAttrib);
    }

    inline void RangePush(const char* text) {
        nvtxEventAttributes_t eventAttrib = PrepareEventAttribute(text);
        nvtxDomainRangePushEx(defaultDomain, &eventAttrib);
    }

    inline void RangePop() {
        nvtxDomainRangePop(defaultDomain);
    }

    inline nvtxRangeId_t RangeStart(nvtxStringHandle_t text) {
        nvtxEventAttributes_t eventAttrib = PrepareEventAttribute(text);
        return nvtxDomainRangeStartEx(defaultDomain, &eventAttrib);
    }

    inline void RangeEnd(const nvtxRangeId_t& rangeId) {
        nvtxDomainRangeEnd(defaultDomain, rangeId);
    }

    inline void NameThread(uint32_t threadId, const char* name) {
        nvtxNameOsThreadA(threadId, name);
    }

    inline void NameCuDevice(CUdevice device, const char* name) {
        nvtxNameCuDeviceA(device, name);
    }

    inline void NameCuStream(CUstream stream, const char* name) {
        nvtxNameCuStreamA(stream, name);
    }

    inline void NameCuContext(CUcontext context, const char* name) {
        nvtxNameCuContextA(context, name);
    }

    inline void NameCuEvent(CUevent event, const char* name) {
        nvtxNameCuEventA(event, name);
    }

    inline nvtxStringHandle_t RegisterString(const char* text) {
        return nvtxDomainRegisterStringA(defaultDomain, text);
    }

private:
    Nvtx() {
        const char* nvtxLevel = utils::GetEnv(NVTX_LEVEL_ENV);
        if (nvtxLevel) {
            level = static_cast<Level>(atoi(nvtxLevel));
        }

#ifdef NVTX_INTERNAL_SUPPORT
        const char* isInternalNvtxEnabledEnv = utils::GetEnv(NVTX_INTERNAL_SUPPORT_ENV);
        if (isInternalNvtxEnabledEnv) {
            isInternalNvtxEnabled = true;
        }
#endif

        if (level != Level::Off || isInternalNvtxEnabled) {
            defaultDomain = nvtxDomainCreate(NVTX_DOMAIN_NAME);
        }
    }

    inline nvtxEventAttributes_t PrepareEventAttribute(nvtxStringHandle_t text) {
        nvtxEventAttributes_t eventAttrib;
        memset(&eventAttrib, 0, sizeof(eventAttrib));
        eventAttrib.version = NVTX_VERSION;
        eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        eventAttrib.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
        eventAttrib.message.registered = text;
        return eventAttrib;
    }

    inline nvtxEventAttributes_t PrepareEventAttribute(const char* text) {
        nvtxEventAttributes_t eventAttrib;
        memset(&eventAttrib, 0, sizeof(eventAttrib));
        eventAttrib.version = NVTX_VERSION;
        eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
        eventAttrib.message.ascii = text;
        return eventAttrib;
    }

    bool isInternalNvtxEnabled = false;
    Level level = Level::Off;
    nvtxDomainHandle_t defaultDomain;
};

struct NvtxScoped {
    inline NvtxScoped(cuLibLogger::Nvtx& nvtx, nvtxStringHandle_t text, bool shouldLogEvent) : shouldLogEvent(shouldLogEvent), nvtx(nvtx) {
        if (shouldLogEvent) {
            nvtx.RangePush(text);
        }
    }

    inline NvtxScoped(cuLibLogger::Nvtx& nvtx, const char* text, bool shouldLogEvent) : shouldLogEvent(shouldLogEvent), nvtx(nvtx) {
        if (shouldLogEvent) {
            nvtx.RangePush(text);
        }
    }

    inline ~NvtxScoped() {
        if (shouldLogEvent) {
            nvtx.RangePop();
        }
    }

private:
    bool shouldLogEvent;
    cuLibLogger::Nvtx& nvtx;
};

LOGGER_END_NAMESPACE

// public NVTX support

// with string registration
#define NVTX_SCOPED_PUBLIC(LEVEL, TEXT) \
  static cuLibLogger::Nvtx& nvtx = cuLibLogger::Nvtx::Instance(); \
  static nvtxStringHandle_t stringId = nvtx.ShouldLogEvent(LEVEL) ? nvtx.RegisterString(TEXT) : nvtxStringHandle_t(0); \
  cuLibLogger::NvtxScoped nvtxScoped(nvtx, stringId, nvtx.ShouldLogEvent(LEVEL))

#define NVTX_SCOPED_PUBLIC_TRACE_STR(TEXT) \
  NVTX_SCOPED_PUBLIC(cuLibLogger::Nvtx::Level::Trace, TEXT)

#define NVTX_SCOPED_PUBLIC_API_STR(TEXT) \
  NVTX_SCOPED_PUBLIC(cuLibLogger::Nvtx::Level::Api, TEXT)

#define NVTX_SCOPED_PUBLIC_TRACE() \
  NVTX_SCOPED_PUBLIC_TRACE_STR(__FUNCTION__)

#define NVTX_SCOPED_PUBLIC_API() \
  NVTX_SCOPED_PUBLIC_API_STR(__FUNCTION__)

// without string registration
#define NVTX_SCOPED_PUBLIC_NO_REG(LEVEL, TEXT) \
  static cuLibLogger::Nvtx& nvtx = cuLibLogger::Nvtx::Instance(); \
  cuLibLogger::NvtxScoped nvtxScoped(nvtx, TEXT, nvtx.ShouldLogEvent(LEVEL))

#define NVTX_SCOPED_PUBLIC_TRACE_STR_NO_REG(TEXT) \
  NVTX_SCOPED_PUBLIC_NO_REG(cuLibLogger::Nvtx::Level::Trace, TEXT)

#define NVTX_SCOPED_PUBLIC_API_STR_NO_REG(TEXT) \
  NVTX_SCOPED_PUBLIC_NO_REG(cuLibLogger::Nvtx::Level::Api, TEXT)

// internal NVTX support
#ifdef NVTX_INTERNAL_SUPPORT

#define NVTX_MARK(TEXT) \
  do { \
    static cuLibLogger::Nvtx& nvtx = cuLibLogger::Nvtx::Instance(); \
    if (nvtx.IsInternalNvtxEnabled()) { \
      static nvtxStringHandle_t stringId = nvtx.RegisterString(TEXT); \
      nvtx.Mark(stringId); \
    } \
  } while(0)

#define NVTX_SCOPED(TEXT) \
  static cuLibLogger::Nvtx& nvtx = cuLibLogger::Nvtx::Instance(); \
  static nvtxStringHandle_t stringId = nvtx.IsInternalNvtxEnabled() ? nvtx.RegisterString(TEXT) : nvtxStringHandle_t(0); \
  cuLibLogger::NvtxScoped nvtxScoped(nvtx, stringId, nvtx.IsInternalNvtxEnabled())

#define NVTX_RANGE_PUSH(TEXT) \
  do { \
    static cuLibLogger::Nvtx& nvtx = cuLibLogger::Nvtx::Instance(); \
    if (nvtx.IsInternalNvtxEnabled()) { \
      static nvtxStringHandle_t stringId = nvtx.RegisterString(TEXT); \
      nvtx.RangePush(stringId); \
    } \
  } while(0)

#define NVTX_RANGE_POP() \
  do { \
    static cuLibLogger::Nvtx& nvtx = cuLibLogger::Nvtx::Instance(); \
    if (nvtx.IsInternalNvtxEnabled()) { \
      nvtx.RangePop(); \
    } \
  } while(0)

#define NVTX_RANGE_START(STRING_ID) \
  cuLibLogger::Nvtx::Instance().IsInternalNvtxEnabled() ? cuLibLogger::Nvtx::Instance().RangeStart(STRING_ID) : nvtxRangeId_t(0)

#define NVTX_RANGE_END(RANGE_ID)  \
  do { \
    static cuLibLogger::Nvtx& nvtx = cuLibLogger::Nvtx::Instance(); \
    if (nvtx.IsInternalNvtxEnabled()) { \
      nvtx.RangeEnd(RANGE_ID); \
    } \
  } while(0)

#define NVTX_NAME_THREAD(TID, NAME) \
  do { \
    static cuLibLogger::Nvtx& nvtx = cuLibLogger::Nvtx::Instance(); \
    if (nvtx.IsInternalNvtxEnabled()) { \
      nvtx.NameThread(TID, NAME); \
    } \
  } while(0)

#define NVTX_NAME_DEVICE(DEVICE, NAME) \
  do { \
    static cuLibLogger::Nvtx& nvtx = cuLibLogger::Nvtx::Instance(); \
    if (nvtx.IsInternalNvtxEnabled()) { \
      nvtx.NameCuDevice(DEVICE, NAME); \
    } \
  } while(0)

#define NVTX_NAME_STREAM(STREAM, NAME) \
  do {  \
    static cuLibLogger::Nvtx& nvtx = cuLibLogger::Nvtx::Instance(); \
    if (nvtx.IsInternalNvtxEnabled()) { \
      nvtx.NameCuStream(STREAM, NAME); \
    } \
  } while(0)

#define NVTX_NAME_CONTEXT(CONTEXT, NAME) \
  do { \
    static cuLibLogger::Nvtx& nvtx = cuLibLogger::Nvtx::Instance(); \
    if (nvtx.IsInternalNvtxEnabled()) { \
      nvtx.NameCuContext(CONTEXT, NAME); \
    } \
  } while(0)

#define NVTX_NAME_EVENT(EVENT, NAME) \
  do { \
    static cuLibLogger::Nvtx& nvtx = cuLibLogger::Nvtx::Instance(); \
    if (nvtx.IsInternalNvtxEnabled()) { \
      nvtx.NameCuEvent(EVENT, NAME); \
    } \
  } while(0)

#define NVTX_REGISTER_STRING(TEXT) \
    cuLibLogger::Nvtx::Instance().IsInternalNvtxEnabled() ? cuLibLogger::Nvtx::Instance().RegisterString(TEXT) : nvtxStringHandle_t(0)

#else

template <typename T>
inline void doNothing(T& /*t*/) {}

#define NVTX_MARK(TEXT)
#define NVTX_SCOPED(TEXT)
#define NVTX_RANGE_PUSH(TEXT)
#define NVTX_RANGE_POP()
#define NVTX_RANGE_START(STRING_ID) nvtxRangeId_t(0)
#define NVTX_RANGE_END(RANGE_ID) doNothing(RANGE_ID)
#define NVTX_NAME_THREAD(TID, NAME)
#define NVTX_NAME_DEVICE(DEVICE, NAME)
#define NVTX_NAME_STREAM(STREAM, NAME)
#define NVTX_NAME_CONTEXT(CONTEXT, NAME)
#define NVTX_NAME_EVENT(EVENT, NAME)
#define NVTX_REGISTER_STRING(TEXT)

#endif
