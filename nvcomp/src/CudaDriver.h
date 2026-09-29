/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// TODO: This file among others should be part of nvcomp_utils, a common dependency

#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cudaTypedefs.h>

#include <cassert>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "common.h"
#include "exception.hpp"
#include "HWDecompressMacros.h"

namespace nvcomp
{

// The macros below are to be used within the body of the CudaDriver class.
// Only DECLARE_DRIVER_API_FNS should be directly used. This takes a name of
// a CUDA Driver API call and the preconditions that that call requires.
// It then defines two public member functions: one with the exact name of the
// Driver API call and one with the same name suffixed with "_unchecked".
// The former automatically checks and, if necessary, enforces the preconditions
// for the call, whereas the latter does not. The _unchecked variants are
// intended for code segments with several CUDA Driver API calls where the user
// may want to enforce the most restrictive preconditions only once with
// CudaDriver::make_preconditions_guard and avoid the overhead of preconditions
// being enforced with every call (it should, however, be remarked, that the
// overhead may be negligible in most cases). Note that the _unchecked version
// of a call is identical to the checked version for functions with no
// preconditions, but both versions are still defined.
//
// Several CUDA Driver API call names are macros that expand to a different
// string, usually the same name with a suffix like "_v2" or "_v3". The names of
// the member functions of CudaDriver are these expanded strings. All macros
// below, except for DECLARE_DRIVER_API_FNS, assume that FuncName is this expanded
// string, which is automatically true when they are called indirectly.
//
// For a more efficient implementation, a private traits type is also defined
// for each CUDA Driver API call. Any members declared between a
// DECLARE_DRIVER_API_FNS invocation and an access specifier will be public.

#define DRIVER_API_TRAITS_NAME(FuncName) FuncName##Traits

#define DECLARE_DRIVER_API_TRAITS(FuncName, ApiVersion, FuncPtrType, SymbolName)                                       \
  struct DRIVER_API_TRAITS_NAME(FuncName)                                                                              \
  {                                                                                                                    \
    static constexpr const char *QueryString = SymbolName;                                                             \
    static constexpr unsigned int DriverApiVersion = ApiVersion;                                                       \
    using FunctionType = FuncPtrType;                                                                                  \
  }

#define DECLARE_GENERIC_DRIVER_API_FN(FuncName, Traits, Preconds)                                                      \
  template <typename... Args>                                                                                          \
  static CUresult FuncName(Args &&...args)                                                                             \
  {                                                                                                                    \
    return invokeFunction<Traits, Preconds>(std::forward<Args>(args)...);                                              \
  }

#define DRIVER_API_UNCHKD_FUNC_NAME(FuncName) FuncName##_unchecked

#define DECLARE_DRIVER_API_FNS(FuncName, ApiVersion, Preconds)                                                         \
  DECLARE_DRIVER_API_FNS_EX(FuncName, ApiVersion, #FuncName, Preconds)

#define DECLARE_DRIVER_API_FNS_EX(FuncName, ApiVersion, SymbolName, Preconds)                                          \
private:                                                                                                               \
  DECLARE_DRIVER_API_TRAITS(FuncName, ApiVersion, decltype(&::FuncName), SymbolName);                                  \
                                                                                                                       \
public:                                                                                                                \
  DECLARE_GENERIC_DRIVER_API_FN(FuncName, DRIVER_API_TRAITS_NAME(FuncName), Preconds)                                  \
  DECLARE_GENERIC_DRIVER_API_FN(                                                                                       \
    DRIVER_API_UNCHKD_FUNC_NAME(FuncName),                                                                             \
    DRIVER_API_TRAITS_NAME(FuncName),                                                                                  \
    CallPreconditions::None                                                                                            \
  )

/**
 * @brief Abstraction for the CUDA Driver API via the CUDA Runtime API, so that
 * we don't need to link against libcuda.so / nvcudaXX.dll. Driver entry points are
 * resolved through cudaGetDriverEntryPointByVersion.
 * Therefore, whenever nvCOMP
 * is used on a machine without GPU support (either due to missing hardware or missing
 * driver), nvCOMP can gracefully terminate.
 *
 * @note Function grouping follows the official CUDA Driver API groups.
 */
class CudaDriver
{
public:
  CudaDriver() noexcept = delete;
  ~CudaDriver() noexcept = delete;

  /**
   * @brief Enumerates the kinds of preconditions that must hold for successfully
   * calling a CUDA Driver API function.
   */
  enum class CallPreconditions
  {
    /// There are no preconditions on using the function.
    None,
    /// The driver API must have been initialized by cuInit(0).
    NeedsInit,
    /// The driver API must have been initialized by cuInit(0) and a CUDA context must be active.
    NeedsContext,
  };

  /**
   * @brief Generate a scoped guard that enforces the requested preconditions on
   * construction and may restore state on destruction.
   *
   * For @ref CallPreconditions::NeedsInit, it is ensured that the CUDA driver
   * runtime is initialized on construction and nothing is done on destruction.
   *
   * For @ref CallPreconditions::NeedsContext, it is ensured that the CUDA driver
   * runtime is initialized on construction as well as that a CUDA context is
   * active. If no context is active, the thread is bound to the primary context
   * of the device with index zero. On destruction, the primary context of the
   * device with index zero is unbound if no context was active at construction.
   * If a context was active at construction, destruction does nothing.
   */
  template <CallPreconditions Preconds>
  static auto make_preconditions_guard();

  /**
   * @brief Trivial structure representing the absence of actions needed upon
   * destruction in @ref make_preconditions_guard.
   */
  struct TrivialGuard
  {};

  // Initialization
  DECLARE_DRIVER_API_FNS(cuInit, 2000, CallPreconditions::None)

  // Context Management
  DECLARE_DRIVER_API_FNS(cuCtxSetCurrent, 4000, CallPreconditions::NeedsInit)
  DECLARE_DRIVER_API_FNS(cuCtxGetCurrent, 4000, CallPreconditions::NeedsInit)
  DECLARE_DRIVER_API_FNS(cuCtxPushCurrent, 4000, CallPreconditions::NeedsInit)
  DECLARE_DRIVER_API_FNS(cuCtxPopCurrent, 4000, CallPreconditions::NeedsInit)
  DECLARE_DRIVER_API_FNS(cuCtxGetDevice, 2000, CallPreconditions::NeedsInit)
  DECLARE_DRIVER_API_FNS(cuMemBatchDecompressAsync, 12060, CallPreconditions::NeedsContext)

  // Note: cuStreamGetCtx may become a macro alias for cuStreamGetCtx_v2 in
  // a future X.Y version of the CUDA toolkit. To avoid double definition
  // errors, the following line should then be guarded by:
  // #if CUDART_VERSION < X0Y0
  DECLARE_DRIVER_API_FNS(cuStreamGetCtx, 9020, CallPreconditions::NeedsContext)
  DECLARE_DRIVER_API_FNS_EX(cuStreamGetCtx_v2, 12050, "cuStreamGetCtx", CallPreconditions::NeedsContext)

  // Returns the latest CUDA version supported by the current driver; result is cached after the first call.
  // Note: despite its name, cudaDriverGetVersion returns the latest CUDA version supported by the driver,
  //       not the driver version itself.
  static int get_latest_supported_cuda_version();

  static CUresult cuStreamGetCtx_portable(CUstream hStream, CUcontext *pctx)
  {
    // CTK 12.x -> decide based on the driver version
    // CTK 13.x -> call directly cuStreamGetCtx_v2
#if CUDART_VERSION >= 13000
    CUgreenCtx green_ctx{};
    return CudaDriver::cuStreamGetCtx_v2(hStream, pctx, &green_ctx);
#else
    if (get_latest_supported_cuda_version() >= 12050)
    {
      CUgreenCtx green_ctx{};
      return CudaDriver::cuStreamGetCtx_v2(hStream, pctx, &green_ctx);
    }
    else
    {
      return CudaDriver::cuStreamGetCtx(hStream, pctx);
    }
#endif
  }

  // Primary Context Management
  DECLARE_DRIVER_API_FNS(cuDevicePrimaryCtxRetain, 7000, CallPreconditions::NeedsInit)
  DECLARE_DRIVER_API_FNS(cuDevicePrimaryCtxRelease, 11000, CallPreconditions::NeedsInit)

  // Device Management
  DECLARE_DRIVER_API_FNS(cuDeviceGet, 2000, CallPreconditions::NeedsInit)
  DECLARE_DRIVER_API_FNS(cuDeviceGetCount, 2000, CallPreconditions::NeedsInit)
  DECLARE_DRIVER_API_FNS(cuDeviceGetAttribute, 2000, CallPreconditions::NeedsInit)
  DECLARE_DRIVER_API_FNS(cuPointerGetAttribute, 4000, CallPreconditions::NeedsInit)

  // Error Handling
  DECLARE_DRIVER_API_FNS(cuGetErrorString, 6000, CallPreconditions::None)
private:
  template <typename Traits>
  static typename Traits::FunctionType loadDriverEntryPoint()
  {
    using FuncType = typename Traits::FunctionType;
    FuncType fn_ptr = nullptr;
    cudaDriverEntryPointQueryResult driverStatus{};
    CUDA_CHECK(cudaGetDriverEntryPointByVersion(
      Traits::QueryString,
      reinterpret_cast<void **>(&fn_ptr),
      Traits::DriverApiVersion,
      cudaEnableDefault,
      &driverStatus
    ));
    if (driverStatus != cudaDriverEntryPointSuccess)
    {
      std::string msg;
      switch (driverStatus)
      {
        case cudaDriverEntryPointSymbolNotFound:
          msg = "Symbol not found";
          break;
        case cudaDriverEntryPointVersionNotSufficent:
          msg = "Runtime version not sufficient";
          break;
        default:
          msg = "Unknown driver entry point query failure";
          break;
      }
      throw NVCompException(
        nvcompErrorCudaError,
        std::string("Unable to retrieve driver entry point for ") + Traits::QueryString + ": " + msg
      );
    }
    if (fn_ptr == nullptr)
    {
      throw NVCompException(
        nvcompErrorInternal,
        std::string("Unable to acquire driver entry point for ") + Traits::QueryString
      );
    }
    return fn_ptr;
  }

  template <typename Traits, CallPreconditions Preconds, typename... Args>
  static CUresult invokeFunction(Args &&...args)
  {
    const auto fn_ptr = getFunction<Traits>();
    [[maybe_unused]] const auto preconds_guard = make_preconditions_guard<Preconds>();
    return fn_ptr(std::forward<Args>(args)...);
  }

  template <typename Traits>
  static auto getFunction()
  {
    using FuncType = typename Traits::FunctionType;
    static const FuncType fn_ptr = loadDriverEntryPoint<Traits>();
    return fn_ptr;
  }

  static void ensureInit()
  {
    [[maybe_unused]] static const CUresult res = []() {
      CUresult res = CudaDriver::cuInit(0 /* flags */);
      CU_CHECK(res);
      return res;
    }();
  }

  static std::vector<CUdevice> getDevices()
  {
    int device_count = 0;
    CU_CHECK(CudaDriver::cuDeviceGetCount(&device_count));

    std::vector<CUdevice> devices(device_count);
    for (int i = 0; i < device_count; ++i)
    {
      CU_CHECK(CudaDriver::cuDeviceGet(&devices[i], i));
    }
    return devices;
  }
};

// TODO: Consider having explicit specializations for these and defining them
// in a source file rather than using if constexpr. Then we could also get rid
// of get_driver_api_error_string in error_handling.{h,cpp}.
template <CudaDriver::CallPreconditions Preconds>
auto CudaDriver::make_preconditions_guard()
{
  if constexpr (Preconds == CallPreconditions::NeedsContext)
  {
    ensureInit();
    static const std::vector<CUdevice> devices = getDevices();

    // C++17 lambdas are not default constructible, use explicit function object
    // instead.
    struct Dtor
    {
      void operator()(CUdevice *device)
      {
        // Note: ctx is not accessed, just used as scratch space for an unused
        // output parameter.
        // Note 2: The loading of symbols might also throw exceptions
        try
        {
          CUcontext ctx;
          CU_CHECK(CudaDriver::cuCtxPopCurrent(&ctx));
          CU_CHECK(CudaDriver::cuDevicePrimaryCtxRelease(*device));
        }
        catch (const std::runtime_error &err)
        {
          std::cerr << "Fatal error in CudaDriver::preconditions_guard destructor:" << std::endl
                    << err.what() << std::endl;
        }
      }
    };

    using RetT = std::unique_ptr<CUdevice, Dtor>;

    CUcontext context = nullptr;
    CU_CHECK(CudaDriver::cuCtxGetCurrent(&context));

    if (context == nullptr)
    {
      // No context is active, temporarily retain the primary context of
      // the current GPU in the system according to the Runtime API.
      int device_id;
      CUDA_CHECK(cudaGetDevice(&device_id));
      CU_CHECK(CudaDriver::cuDevicePrimaryCtxRetain(&context, devices[device_id]));
      if (context == nullptr)
      {
        throw NVCompException(nvcompErrorCudaError, "Primary context of the current GPU is null.");
      }
      CU_CHECK(CudaDriver::cuCtxPushCurrent(context));
      // Note, that we are not deleting in the custom deleter devices[device_id] .
      return RetT(const_cast<CUdevice *>(&devices[device_id]));
    }
    else
    {
      // Return std::unique_ptr that doesn't own anything.
      return RetT();
    }
  }
  else if constexpr (Preconds == CallPreconditions::NeedsInit)
  {
    ensureInit();
    return TrivialGuard{};
  }
  else
  {
    static_assert(Preconds == CallPreconditions::None);
    return TrivialGuard{};
  }
}

inline int CudaDriver::get_latest_supported_cuda_version()
{
  static const int version = []() {
    int v = 0;
    CUDA_CHECK(cudaDriverGetVersion(&v));
    return v;
  }();
  return version;
}

#undef DECLARE_DRIVER_API_FNS_EX
#undef DECLARE_DRIVER_API_FNS
#undef DRIVER_API_UNCHKD_FUNC_NAME
#undef DECLARE_GENERIC_DRIVER_API_FN
#undef DECLARE_DRIVER_API_TRAITS
#undef DRIVER_API_TRAITS_NAME

inline std::string get_driver_api_error_string(CUresult e)
{
  const char *msg{};
  // Given this function is already going to be invoked as part of a failure path, it is safe to just discard its return value
  (void)CudaDriver::cuGetErrorString(e, &msg);
  return msg;
}

} // namespace nvcomp
