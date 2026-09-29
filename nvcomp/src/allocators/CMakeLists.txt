file(GLOB CPP_SOURCES *.cpp)

add_library(nvcomp_allocator_objects OBJECT ${CPP_SOURCES})
set_property(TARGET nvcomp_allocator_objects PROPERTY POSITION_INDEPENDENT_CODE
                                                      ON)

target_compile_definitions(
  nvcomp_allocator_objects
  PRIVATE nvcomp_EXPORTS LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE)
target_include_directories(
  nvcomp_allocator_objects
  PRIVATE "$<BUILD_INTERFACE:${nvcomp_SOURCE_DIR}/include>"
          "$<BUILD_INTERFACE:${nvcomp_SOURCE_DIR}/src>"
          "$<BUILD_INTERFACE:${nvcomp_SOURCE_DIR}/external/culiblogger/include>"
          "$<BUILD_INTERFACE:${nvcomp_SOURCE_DIR}/external/fmt/include>"
          "$<BUILD_INTERFACE:${nvcomp_BINARY_DIR}/include>")
