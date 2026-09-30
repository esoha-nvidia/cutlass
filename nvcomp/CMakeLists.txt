# In-tree nvCOMP host library for CUTLASS examples. Compiles the copied
# include/ + src/ as a static library (no prebuilt libnvcomp.so).
# Do not enable CUDA_SEPARABLE_COMPILATION on this archive: gdeflate headers
# define non-inline __device__ tables, and nvlink then reports multiple
# definition. The in-kernel ANS entry lives in nvcomp_ans_device (RDC).

include(GenerateExportHeader)
include(CheckCXXCompilerFlag)
find_package(CUDAToolkit REQUIRED)

set(nvcomp_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(nvcomp_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

set(PROJECT_VERSION_MAJOR 6)
set(PROJECT_VERSION_MINOR 0)
set(PROJECT_VERSION_PATCH 0)
set(PROJECT_VERSION_TWEAK 0)

if(CUTLASS_NVCC_ARCHS_ENABLED)
  set(NVCOMP_GPU_ARCHS "${CUTLASS_NVCC_ARCHS_ENABLED}")
elseif(CUTLASS_NVCC_ARCHS)
  set(NVCOMP_GPU_ARCHS "${CUTLASS_NVCC_ARCHS}")
else()
  set(NVCOMP_GPU_ARCHS "100")
endif()

file(
  GLOB NVCOMP_CUDA_SOURCES
  ${nvcomp_SOURCE_DIR}/src/*.cu
  ${nvcomp_SOURCE_DIR}/src/ans/*.cu
  ${nvcomp_SOURCE_DIR}/src/lowlevel/*.cu
  ${nvcomp_SOURCE_DIR}/src/highlevel/*.cu
  ${nvcomp_SOURCE_DIR}/src/crc/cuCRC32.cu
  ${nvcomp_SOURCE_DIR}/src/zstd/*.cu
  ${nvcomp_SOURCE_DIR}/src/gdeflate/*.cu
  ${nvcomp_SOURCE_DIR}/src/gzip/*.cu
  ${nvcomp_SOURCE_DIR}/src/optimal_parse/*.cu
  ${nvcomp_SOURCE_DIR}/src/cascaded/*.cu
  ${nvcomp_SOURCE_DIR}/src/lookahead_gzip/*.cu
  ${nvcomp_SOURCE_DIR}/src/bitcomp/*.cu)
list(FILTER NVCOMP_CUDA_SOURCES EXCLUDE REGEX "/ans_device\\.cu$")

file(
  GLOB NVCOMP_CPP_SOURCES
  ${nvcomp_SOURCE_DIR}/src/*.cpp
  ${nvcomp_SOURCE_DIR}/src/lowlevel/*.cpp
  ${nvcomp_SOURCE_DIR}/src/highlevel/*.cpp
  ${nvcomp_SOURCE_DIR}/src/gdeflate/*.cpp
  ${nvcomp_SOURCE_DIR}/src/gzip/*.cpp
  ${nvcomp_SOURCE_DIR}/src/cascaded/*.cpp
  ${nvcomp_SOURCE_DIR}/src/allocators/*.cpp)

add_library(nvcomp STATIC ${NVCOMP_CUDA_SOURCES} ${NVCOMP_CPP_SOURCES})

configure_file(
  "${nvcomp_SOURCE_DIR}/src/nvcomp_version.h.in"
  "${nvcomp_BINARY_DIR}/include/nvcomp/version.h"
  @ONLY)

generate_export_header(
  nvcomp
  BASE_NAME NVCOMP
  EXPORT_FILE_NAME "${nvcomp_BINARY_DIR}/include/nvcomp_export.h")

set(_nvcomp_cccl_include "")
if(CUTLASS_CUDA_CCCL_INCLUDE_DIR)
  set(_nvcomp_cccl_include "${CUTLASS_CUDA_CCCL_INCLUDE_DIR}")
endif()

target_compile_definitions(
  nvcomp
  PUBLIC NVCOMP_STATIC_DEFINE
  PRIVATE
    nvcomp_EXPORTS
    ENABLE_GDEFLATE
    IMPLICIT_BUFFER_ALIGNMENT
    LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE
    THRUST_CUB_WRAPPED_NAMESPACE=nvcomp
    FMT_HEADER_ONLY)

target_compile_options(
  nvcomp
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-pthread>)

check_cxx_compiler_flag(-mavx2 NVCOMP_COMPILER_OPT_AVX2)
if(NVCOMP_COMPILER_OPT_AVX2)
  target_compile_options(
    nvcomp
    PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-mavx2>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-mavx2>)
endif()

set_target_properties(
  nvcomp
  PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_ARCHITECTURES "${NVCOMP_GPU_ARCHS}")

if(NOT EXISTS "${nvcomp_SOURCE_DIR}/external/fmt/include/fmt/format.h")
  message(FATAL_ERROR "nvCOMP needs header-only fmt at ${nvcomp_SOURCE_DIR}/external/fmt/include")
endif()
if(NOT EXISTS "${nvcomp_SOURCE_DIR}/external/culiblogger/include/cuLibLogger/cuLibLogger.h")
  message(FATAL_ERROR
    "nvCOMP needs culiblogger headers at ${nvcomp_SOURCE_DIR}/external/culiblogger/include")
endif()

target_include_directories(
  nvcomp
  PUBLIC
    "$<BUILD_INTERFACE:${nvcomp_SOURCE_DIR}/include>"
    "$<BUILD_INTERFACE:${nvcomp_BINARY_DIR}/include>"
  PRIVATE
    "${nvcomp_SOURCE_DIR}/src"
    "${nvcomp_SOURCE_DIR}/src/bitcomp"
    "${nvcomp_SOURCE_DIR}/external/fmt/include"
    "${nvcomp_SOURCE_DIR}/external/culiblogger/include")

if(_nvcomp_cccl_include)
  target_include_directories(nvcomp PRIVATE "${_nvcomp_cccl_include}")
endif()

target_link_libraries(nvcomp PUBLIC CUDA::cudart)

# Relocatable device ANS entry used by 00_basic_gemm. Compiled separately so
# nvlink does not see the rest of libnvcomp.a.
add_library(nvcomp_ans_device OBJECT "${nvcomp_SOURCE_DIR}/src/ans/ans_device.cu")
target_compile_definitions(
  nvcomp_ans_device
  PRIVATE
    nvcomp_EXPORTS
    NVCOMP_STATIC_DEFINE
    ENABLE_GDEFLATE
    IMPLICIT_BUFFER_ALIGNMENT
    LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE
    THRUST_CUB_WRAPPED_NAMESPACE=nvcomp
    FMT_HEADER_ONLY)
target_compile_options(
  nvcomp_ans_device
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-pthread>)
target_include_directories(
  nvcomp_ans_device
  PRIVATE
    "${nvcomp_SOURCE_DIR}/include"
    "${nvcomp_BINARY_DIR}/include"
    "${nvcomp_SOURCE_DIR}/src"
    "${nvcomp_SOURCE_DIR}/src/bitcomp"
    "${nvcomp_SOURCE_DIR}/external/fmt/include"
    "${nvcomp_SOURCE_DIR}/external/culiblogger/include")
if(_nvcomp_cccl_include)
  target_include_directories(nvcomp_ans_device PRIVATE "${_nvcomp_cccl_include}")
endif()
set_target_properties(
  nvcomp_ans_device
  PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "${NVCOMP_GPU_ARCHS}")
add_dependencies(nvcomp_ans_device nvcomp)

message(STATUS "nvCOMP: building in-tree static library for archs ${NVCOMP_GPU_ARCHS}")
