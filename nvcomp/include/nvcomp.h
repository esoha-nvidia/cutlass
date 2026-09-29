/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
*/

#pragma once

#ifdef __has_include
#if __has_include(<cuda_runtime.h>)
#include <cuda_runtime.h>
#else
#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif
typedef struct CUstream_st *cudaStream_t;
#endif
#endif

// clang-format off
#include "nvcomp/shared_types.h"
#include "nvcomp/version.h"
#include "nvcomp_export.h"

#include "nvcomp/ans.h"
#include "nvcomp/bitcomp.h"
#include "nvcomp/cascaded.h"
#include "nvcomp/crc32.h"
#include "nvcomp/deflate.h"
#include "nvcomp/gdeflate.h"
#include "nvcomp/gzip.h"
#include "nvcomp/lz4.h"
#include "nvcomp/snappy.h"
#include "nvcomp/zstd.h"
// clang-format on

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Retrieve the nvCOMP library properties.
 *
 * @param[out] properties Retrieved nvCOMP properties in an nvcompProperties_t struct.
 *
 * @return nvcompErrorInvalidValue if properties is nullptr, nvcompSuccess otherwise.
 */
NVCOMP_EXPORT
nvcompStatus_t nvcompGetProperties(nvcompProperties_t *properties);

/**
 * @brief Returns the description string for an error code.
 *
 * @param nvcompStatus_t Status to convert to string
 *
 * @return Pointer to a NULL-terminated string.
 */
NVCOMP_EXPORT
const char *nvcompGetStatusString(nvcompStatus_t status);

#ifdef __cplusplus
}
#endif
