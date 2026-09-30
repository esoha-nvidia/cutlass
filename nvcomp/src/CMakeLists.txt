include(GenerateExportHeader)

file(
  GLOB
  CUDA_SOURCES
  ./*.cu
  ./ans/*.cu
  ./lowlevel/*.cu
  ./highlevel/*.cu
  ./crc/cuCRC32.cu
  ./zstd/*.cu
  ./gdeflate/*.cu
  ./gzip/*.cu
  ./optimal_parse/*.cu
  ./cascaded/*.cu
  ./lookahead_gzip/*.cu)

file(
  GLOB
  CPP_SOURCES
  ./*.cpp
  ./lowlevel/*.cpp
  ./highlevel/*.cpp
  ./gdeflate/*.cpp
  ./gzip/*.cpp
  ./cascaded/*.cpp)

# Shared high-level interface (PimplManager,
# CompressionConfig/DecompressionConfig) used by both nvcomp and nvcomp_cpu;
# build as object library and link into both to avoid nvcomp_cpu linking nvcomp
list(FILTER CPP_SOURCES EXCLUDE REGEX
     "/(PimplManager|CompressionConfigs)\\.cpp$")
set(NVCOMP_HLIF_SHARED_CPP
    ${CMAKE_CURRENT_SOURCE_DIR}/highlevel/PimplManager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/highlevel/CompressionConfigs.cpp)

add_library(nvcomp_hlif_shared_objects OBJECT ${NVCOMP_HLIF_SHARED_CPP})
target_include_directories(
  nvcomp_hlif_shared_objects
  PRIVATE "${nvcomp_SOURCE_DIR}/include" "${nvcomp_SOURCE_DIR}/src"
          "${nvcomp_BINARY_DIR}/include")
target_compile_definitions(nvcomp_hlif_shared_objects PRIVATE nvcomp_EXPORTS)
set_target_properties(nvcomp_hlif_shared_objects
                      PROPERTIES POSITION_INDEPENDENT_CODE ON)

set(NVCOMP_OBJECTS_NAME ${NVCOMP_LIBRARY_NAME}_obj)

add_library(${NVCOMP_OBJECTS_NAME} OBJECT ${CUDA_SOURCES} ${CPP_SOURCES})
target_compile_definitions(
  ${NVCOMP_OBJECTS_NAME}
  PRIVATE LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE
          THRUST_CUB_WRAPPED_NAMESPACE=nvcomp nvcomp_EXPORTS)

add_library(
  ${NVCOMP_STATIC_LIBRARY_NAME} STATIC
  $<TARGET_OBJECTS:nvcomp_allocator_objects>
  $<TARGET_OBJECTS:nvcomp_bitcomp_objects>
  $<TARGET_OBJECTS:nvcomp_hlif_shared_objects> $<TARGET_OBJECTS:nvcomp_obj>)
add_library(
  ${NVCOMP_LIBRARY_NAME} SHARED
  $<TARGET_OBJECTS:nvcomp_allocator_objects>
  $<TARGET_OBJECTS:nvcomp_bitcomp_objects>
  $<TARGET_OBJECTS:nvcomp_hlif_shared_objects> $<TARGET_OBJECTS:nvcomp_obj>)

# RtlDllShutdownInProgress (used in DeviceMemoryPool destructor on Windows) is
# exported from ntdll.dll.
if(WIN32)
  target_link_libraries(${NVCOMP_LIBRARY_NAME} PRIVATE ntdll)
  target_link_libraries(${NVCOMP_STATIC_LIBRARY_NAME} INTERFACE ntdll)
endif()

# Note: The python wheel we create cannot have symlinks (ZIP archives do not
# preserve symlinks), and therefore we are falling back to the major version of
# nvCOMP to avoid .so duplication within our python wheels.
if(BUILD_WHEEL)
  set_target_properties(
    ${NVCOMP_LIBRARY_NAME} PROPERTIES VERSION ${PROJECT_VERSION_MAJOR}
                                      SOVERSION ${PROJECT_VERSION_MAJOR})
else()
  set_target_properties(
    ${NVCOMP_LIBRARY_NAME} PROPERTIES VERSION ${PROJECT_VERSION}
                                      SOVERSION ${PROJECT_VERSION_MAJOR})
endif()

target_link_libraries(${NVCOMP_LIBRARY_NAME} PRIVATE CUDA::cudart_static)
target_link_libraries(${NVCOMP_STATIC_LIBRARY_NAME} PRIVATE CUDA::cudart_static)

if(MSVC)
  set_target_properties(
    ${NVCOMP_LIBRARY_NAME}
    PROPERTIES RUNTIME_OUTPUT_NAME
               "${NVCOMP_LIBRARY_NAME}${ARCH_SUFFIX}_${NVCOMP_VERSION_MAJOR}")
endif()

# Configure library version
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/nvcomp_version.h.in"
               "${nvcomp_BINARY_DIR}/include/nvcomp/version.h" @ONLY)

generate_export_header(${NVCOMP_LIBRARY_NAME} BASE_NAME NVCOMP EXPORT_FILE_NAME
                       "${nvcomp_BINARY_DIR}/include/nvcomp_export.h")

cpmaddpackage(
  NAME
  CCCL
  GITHUB_REPOSITORY
  nvidia/cccl
  VERSION
  3.4.2
  PATCH_COMMAND
  git
  apply
  "${nvcomp_SOURCE_DIR}/cmake/cccl-fix-undef.patch"
  "${nvcomp_SOURCE_DIR}/cmake/cccl-fix-destructor-throw.patch"
  ||
  echo
  "Patch failed or reapplied")

# Debugging WAR for CCCL ptxas warning: Stack size for entry function
# `cub::detail::DeviceScanKernel...` cannot be statically determined
add_compile_definitions($<$<CONFIG:Debug>:CCCL_DISABLE_WARPSPEED_SCAN>)

target_link_libraries(
  ${NVCOMP_OBJECTS_NAME} PRIVATE $<BUILD_LOCAL_INTERFACE:CCCL::CCCL>
                                 CUDA::cudart_static)

if(TARGET nvtx3-cpp)
  target_link_libraries(${NVCOMP_OBJECTS_NAME} PRIVATE nvtx3-cpp)
endif()

add_subdirectory(gdeflate/cpu)
add_subdirectory(allocators)
add_subdirectory(bitcomp)
if(NOT MSVC)
  add_subdirectory(lookahead_gzip/bin)
endif()

target_link_libraries(
  nvcomp_allocator_objects PRIVATE $<BUILD_LOCAL_INTERFACE:CCCL::CCCL>
                                   CUDA::cudart_static)

set_target_properties(
  ${NVCOMP_OBJECTS_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON
                                    CUDA_ARCHITECTURES "${GPU_ARCHS}")

target_compile_options(
  ${NVCOMP_OBJECTS_NAME}
  PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda
          --expt-relaxed-constexpr>)

if(MSVC)
  target_link_options(${NVCOMP_OBJECTS_NAME} PUBLIC
                      $<$<CXX_COMPILER_ID:MSVC>:/NODEFAULTLIB:libcmt.lib>)
else()
  target_compile_options(
    ${NVCOMP_OBJECTS_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler
                                   -pthread>)
endif()

target_include_directories(
  ${NVCOMP_OBJECTS_NAME}
  PRIVATE "${nvcomp_SOURCE_DIR}/src"
          "$<BUILD_INTERFACE:${nvcomp_SOURCE_DIR}/external/fmt/include>")

# Add includes here, so that we don't need to add them each time when we build
# tests
foreach(LIB_NAME ${NVCOMP_OBJECTS_NAME} ${NVCOMP_STATIC_LIBRARY_NAME}
                 ${NVCOMP_LIBRARY_NAME})
  target_include_directories(
    ${LIB_NAME}
    PUBLIC
      "$<BUILD_INTERFACE:${nvcomp_SOURCE_DIR}/include>"
      "$<BUILD_INTERFACE:${nvcomp_SOURCE_DIR}/src/>"
      "$<BUILD_INTERFACE:${nvcomp_SOURCE_DIR}/external/culiblogger/include>"
      "$<BUILD_INTERFACE:${nvcomp_BINARY_DIR}/include>" # Necessary because of
                                                        # our generated
                                                        # version.h and
                                                        # *_export.h files
      "$<BUILD_INTERFACE:${nvcomp_SOURCE_DIR}/src/bitcomp>")
endforeach()

# Add to install interface, so generated cmake install file has include path
foreach(RELEASED_LIB_NAME ${NVCOMP_STATIC_LIBRARY_NAME} ${NVCOMP_LIBRARY_NAME})
  target_include_directories(${RELEASED_LIB_NAME}
                             PUBLIC "$<INSTALL_INTERFACE:include>")
endforeach()

list(APPEND NVCOMP_STATIC_EXPORT_TARGETS ${NVCOMP_STATIC_LIBRARY_NAME})
list(APPEND NVCOMP_DYNAMIC_EXPORT_TARGETS ${NVCOMP_LIBRARY_NAME})

# Create namespace aliases for use as a subproject (e.g., via CPM). The nvcomp::
# namespace is only created on install, not when built as a subproject.
add_library(${PROJECT_NAME}::${NVCOMP_STATIC_LIBRARY_NAME} ALIAS
            ${NVCOMP_STATIC_LIBRARY_NAME})
add_library(${PROJECT_NAME}::${NVCOMP_LIBRARY_NAME} ALIAS
            ${NVCOMP_LIBRARY_NAME})

# nvcomp_cpu

# Note: we need to combine nvcomp's CPU objects with libdeflate_static so that
# the executables that link against nvcomp_cpu_static can resolve all of their
# symbols. Shared HLIF (PimplManager, configs) come from
# nvcomp_hlif_shared_objects.
add_library(
  ${NVCOMP_CPU_STATIC_LIBRARY_NAME} STATIC
  $<TARGET_OBJECTS:libdeflate_static>
  $<TARGET_OBJECTS:nvcomp_gdeflate_cpu_objects>
  $<TARGET_OBJECTS:nvcomp_hlif_shared_objects>)
add_dependencies(${NVCOMP_CPU_STATIC_LIBRARY_NAME} libdeflate_static)
target_include_directories(
  ${NVCOMP_CPU_STATIC_LIBRARY_NAME}
  PUBLIC "$<BUILD_INTERFACE:${nvcomp_SOURCE_DIR}/external/libdeflate>"
         "$<INSTALL_INTERFACE:include>")

add_library(
  ${NVCOMP_CPU_LIBRARY_NAME} SHARED
  $<TARGET_OBJECTS:nvcomp_gdeflate_cpu_objects>
  $<TARGET_OBJECTS:nvcomp_hlif_shared_objects>)
add_dependencies(${NVCOMP_CPU_LIBRARY_NAME} libdeflate_static)
target_include_directories(${NVCOMP_CPU_LIBRARY_NAME}
                           PUBLIC "$<INSTALL_INTERFACE:include>")

if(MSVC)
  set_target_properties(
    ${NVCOMP_CPU_LIBRARY_NAME}
    PROPERTIES
      RUNTIME_OUTPUT_NAME
      "${NVCOMP_CPU_LIBRARY_NAME}${ARCH_SUFFIX}_${NVCOMP_VERSION_MAJOR}")
endif()

target_link_libraries(${NVCOMP_CPU_LIBRARY_NAME} PRIVATE libdeflate_static)

# Note: The python wheel we create cannot have symlinks (ZIP archives do not
# preserve symlinks), and therefore we are falling back to the major version of
# nvCOMP to avoid .so duplication within our python wheels.
if(BUILD_WHEEL)
  set_target_properties(
    ${NVCOMP_CPU_LIBRARY_NAME} PROPERTIES VERSION ${PROJECT_VERSION_MAJOR}
                                          SOVERSION ${PROJECT_VERSION_MAJOR})
else()
  set_target_properties(
    ${NVCOMP_CPU_LIBRARY_NAME} PROPERTIES VERSION ${PROJECT_VERSION}
                                          SOVERSION ${PROJECT_VERSION_MAJOR})
endif()

list(APPEND NVCOMP_STATIC_EXPORT_TARGETS ${NVCOMP_CPU_STATIC_LIBRARY_NAME})
list(APPEND NVCOMP_DYNAMIC_EXPORT_TARGETS ${NVCOMP_CPU_LIBRARY_NAME})

# Create namespace aliases for CPU libraries
add_library(${PROJECT_NAME}::${NVCOMP_CPU_STATIC_LIBRARY_NAME} ALIAS
            ${NVCOMP_CPU_STATIC_LIBRARY_NAME})
add_library(${PROJECT_NAME}::${NVCOMP_CPU_LIBRARY_NAME} ALIAS
            ${NVCOMP_CPU_LIBRARY_NAME})

set(NVCOMP_STATIC_EXPORT_TARGETS
    "${NVCOMP_STATIC_EXPORT_TARGETS}"
    PARENT_SCOPE)
set(NVCOMP_DYNAMIC_EXPORT_TARGETS
    "${NVCOMP_DYNAMIC_EXPORT_TARGETS}"
    PARENT_SCOPE)

verify_object_symbols(${NVCOMP_LIBRARY_NAME} ${PROJECT_NAME})
verify_object_symbols(${NVCOMP_CPU_LIBRARY_NAME} ${PROJECT_NAME})
