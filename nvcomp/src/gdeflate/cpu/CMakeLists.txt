file(GLOB CPP_SOURCES *.cpp)

add_library(nvcomp_gdeflate_cpu_objects OBJECT ${CPP_SOURCES})
add_dependencies(nvcomp_gdeflate_cpu_objects libdeflate_static)

target_compile_definitions(nvcomp_gdeflate_cpu_objects PRIVATE nvcomp_EXPORTS)

set_property(TARGET nvcomp_gdeflate_cpu_objects
             PROPERTY POSITION_INDEPENDENT_CODE ON)

# Note: The exported headers of lib(g)deflate do not expose the constant
# (GDEFLATE_PAGE_SIZE) we are interested in, hence the direct path inclusion.
target_include_directories(
  nvcomp_gdeflate_cpu_objects
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
         $<INSTALL_INTERFACE:include>
         "$<BUILD_INTERFACE:${nvcomp_BINARY_DIR}/include>" # We need this for
                                                           # nvcomp_export.h
  PRIVATE "${nvcomp_SOURCE_DIR}/src/gdeflate" "${nvcomp_SOURCE_DIR}/src"
          "${nvcomp_SOURCE_DIR}/include"
          "${nvcomp_SOURCE_DIR}/external/libdeflate/lib")

target_link_libraries(nvcomp_gdeflate_cpu_objects PRIVATE libdeflate_static)
