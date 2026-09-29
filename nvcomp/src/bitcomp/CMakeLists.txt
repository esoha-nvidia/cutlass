file(GLOB CUDA_SOURCES CONFIGURE_DEPENDS *.cu)

add_library(nvcomp_bitcomp_objects OBJECT ${CUDA_SOURCES})

set_target_properties(
  nvcomp_bitcomp_objects PROPERTIES POSITION_INDEPENDENT_CODE ON
                                    CUDA_ARCHITECTURES "${GPU_ARCHS}")
target_compile_options(
  nvcomp_bitcomp_objects
  PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda
          --expt-relaxed-constexpr>)
target_link_options(nvcomp_bitcomp_objects PUBLIC
                    $<$<CXX_COMPILER_ID:MSVC>:/NODEFAULTLIB:libcmt.lib>)

# specify that we want to export objects that we are building
target_compile_definitions(nvcomp_bitcomp_objects PRIVATE nvcomp_EXPORTS)

target_include_directories(
  nvcomp_bitcomp_objects
  PUBLIC $<INSTALL_INTERFACE:include>
         "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>"
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
          "$<BUILD_INTERFACE:${nvcomp_BINARY_DIR}/include>"
          "${nvcomp_SOURCE_DIR}/include" "${nvcomp_SOURCE_DIR}/src"
  INTERFACE $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include>)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-mavx2 COMPILER_OPT_AVX2_SUPPORTED)
if(COMPILER_OPT_AVX2_SUPPORTED)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -mavx2")
endif()

target_link_libraries(
  nvcomp_bitcomp_objects PRIVATE $<BUILD_LOCAL_INTERFACE:CCCL::CCCL>
                                 CUDA::cudart_static)
