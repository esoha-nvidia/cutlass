
# Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


cutlass_example_add_executable(
  00_basic_gemm
  basic_gemm.cu
)

target_link_libraries(00_basic_gemm PRIVATE cublas)

set(NVCOMP_ROOT "/home/esoha/nvcomp-dev/nvcomp" CACHE PATH "nvCOMP source or install prefix")

find_package(nvcomp QUIET HINTS
  "${NVCOMP_ROOT}/build/python/libnvcomp/nvidia/libnvcomp"
  /usr/local
)

if(TARGET nvcomp::nvcomp)
  target_link_libraries(00_basic_gemm PRIVATE nvcomp::nvcomp)
  get_target_property(_nvcomp_loc nvcomp::nvcomp IMPORTED_LOCATION)
  if(NOT _nvcomp_loc)
    get_target_property(_nvcomp_loc nvcomp::nvcomp IMPORTED_LOCATION_RELEASE)
  endif()
  if(_nvcomp_loc)
    get_filename_component(_nvcomp_lib_dir "${_nvcomp_loc}" DIRECTORY)
    set_property(TARGET 00_basic_gemm APPEND PROPERTY BUILD_RPATH "${_nvcomp_lib_dir}")
  endif()
else()
  find_path(NVCOMP_INCLUDE_DIR nvcomp.h
    HINTS
      "${NVCOMP_ROOT}/include"
      "${NVCOMP_ROOT}/build/python/libnvcomp/nvidia/libnvcomp/include"
      /usr/local/include
  )
  find_path(NVCOMP_GENERATED_INCLUDE_DIR nvcomp_export.h
    HINTS
      "${NVCOMP_ROOT}/build/include"
      "${NVCOMP_ROOT}/build/python/libnvcomp/nvidia/libnvcomp/include"
      /usr/local/include
      "${NVCOMP_INCLUDE_DIR}"
  )
  find_library(NVCOMP_LIBRARY nvcomp
    HINTS
      "${NVCOMP_ROOT}/build/lib"
      "${NVCOMP_ROOT}/build/python/libnvcomp/nvidia/libnvcomp/lib"
      /usr/local/lib
      /usr/local/lib64
  )
  if(NOT NVCOMP_INCLUDE_DIR OR NOT NVCOMP_LIBRARY)
    message(FATAL_ERROR
      "nvCOMP not found. Set -DNVCOMP_ROOT=... (source tree with build/lib, or install prefix).")
  endif()
  target_include_directories(00_basic_gemm PRIVATE
    ${NVCOMP_INCLUDE_DIR}
    ${NVCOMP_GENERATED_INCLUDE_DIR}
  )
  target_link_libraries(00_basic_gemm PRIVATE ${NVCOMP_LIBRARY})
  get_filename_component(_nvcomp_lib_dir "${NVCOMP_LIBRARY}" DIRECTORY)
  set_property(TARGET 00_basic_gemm APPEND PROPERTY BUILD_RPATH "${_nvcomp_lib_dir}")
endif()
  
