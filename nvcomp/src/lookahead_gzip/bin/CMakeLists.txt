# nvlzcat executable

add_executable(nvlzcat nvlzcat.cpp)
set_property(TARGET nvlzcat PROPERTY CUDA_ARCHITECTURES ${GPU_ARCHS})
set_property(TARGET nvlzcat PROPERTY CXX_STANDARD 17)
target_include_directories(
  nvlzcat PRIVATE "${nvcomp_SOURCE_DIR}/src/lookahead_gzip/include")
target_include_directories(
  nvlzcat SYSTEM PRIVATE "${nvcomp_SOURCE_DIR}/external/fmt/include"
                         "${nvcomp_SOURCE_DIR}/external/culiblogger/include")
target_compile_definitions(nvlzcat PRIVATE FMT_HEADER_ONLY)
target_link_libraries(
  nvlzcat
  PRIVATE ${NVCOMP_STATIC_LIBRARY_NAME} $<BUILD_LOCAL_INTERFACE:CCCL::CCCL>
          CUDA::cudart_static $<$<CXX_COMPILER_ID:GNU>:stdc++fs>)
install(
  TARGETS nvlzcat
  DESTINATION ${CMAKE_INSTALL_BINDIR}
  COMPONENT lib)
