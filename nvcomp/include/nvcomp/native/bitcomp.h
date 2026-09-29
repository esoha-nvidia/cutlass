/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES.
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

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <nvcomp/native/bitcomp_mode.h>
#include <nvcomp/shared_types.h>
#include <nvcomp_export.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#else
// cuda_fp16.h does not define __half (and thus half), for C, because __half is a C++ class
typedef uint16_t half;
#endif

struct bitcompContext;
typedef struct bitcompContext *bitcompHandle_t;

typedef enum bitcompResult_t
{
  BITCOMP_SUCCESS = 0,
  BITCOMP_INVALID_PARAMETER = -1,
  BITCOMP_INVALID_COMPRESSED_DATA = -2,
  BITCOMP_INVALID_ALIGNMENT = -3,
  BITCOMP_INVALID_INPUT_LENGTH = -4,
  BITCOMP_CUDA_KERNEL_LAUNCH_ERROR = -5,
  BITCOMP_CUDA_API_ERROR = -6,
  BITCOMP_UNKNOWN_ERROR = -7,
} bitcompResult_t;

typedef enum bitcompDataType_t
{
  // Integral types for lossless compression
  BITCOMP_UNSIGNED_8BIT = 0,
  BITCOMP_SIGNED_8BIT,
  BITCOMP_UNSIGNED_16BIT,
  BITCOMP_SIGNED_16BIT,
  BITCOMP_UNSIGNED_32BIT,
  BITCOMP_SIGNED_32BIT,
  BITCOMP_UNSIGNED_64BIT,
  BITCOMP_SIGNED_64BIT,
  // Floating point types used for lossy compression
  BITCOMP_FP16_DATA,
  BITCOMP_FP32_DATA,
  BITCOMP_FP64_DATA
} bitcompDataType_t;

typedef enum bitcompAlgorithm_t
{
  BITCOMP_DEFAULT_ALGO = 0, // Default algorithm
  BITCOMP_SPARSE_ALGO = 1 // Recommended for very sparse data (lots of zeros)
} bitcompAlgorithm_t;

//***********************************************************************************************
// Plan creation and destruction

/**
 * @brief Create a bitcomp plan for compression and decompression, lossy or lossless.
 *
 * The lossless compression can be used on any data type, viewed as integral type.
 * Choosing the right integral type will have an effect on the compression ratio.
 *
 * Lossy compression:
 * The lossy compression is only available for floating point data types, and is based
 * on a quantization of the floating point values to integers.
 * The floating point values are divided by the delta provided during the compression, and converted
 * to integers. These integers are then compressed with a lossless encoder.
 * The maximum error between the uncompressed data and the original data is <= delta/2.
 * Values that would overflow during quantization (e.g. large input values and a very small delta),
 * as well as NaN, +Inf, -Inf will be handled correctly by the compression.
 * The integers can be either signed or unsigned.
 *
 * The same plan can be used on several devices or on the host, but associating the plan
 * with a stream, or turning on remote compression acceleration will make a plan device-specific.
 * Using a plan concurrently on more than one device is not supported.
 *
 * @param[out] handle Handle created.
 * @param[in] n size of the uncompressed data in bytes.
 * @param[in] dataType Datatype of the uncompressed data.
 * @param[in] mode Compression mode, lossless or lossy to signed / lossy to unsigned.
 * @param[in] algo Which compression algorithm to use.
 * @return Returns BITCOMP_SUCCESS if successful, or an error.
 */
NVCOMP_EXPORT
bitcompResult_t bitcompCreatePlan(
  bitcompHandle_t *handle,
  size_t n,
  bitcompDataType_t dataType,
  bitcompMode_t mode,
  bitcompAlgorithm_t algo
);

/**
 * @brief Create a handle from existing compressed data.
 *
 * @param[out] handle Handle created.
 * @param[in] data Pointer to the compressed data, from which all the handle parameters will be extracted.
 * @return Returns BITCOMP_SUCCESS if successful, or an error.
 */
NVCOMP_EXPORT
bitcompResult_t bitcompCreatePlanFromCompressedData(bitcompHandle_t *handle, const void *data);

/**
 * @brief Destroy an existing bitcomp handle.
 *
 * @param[in] handle Handle to destroy.
 * @return Returns BITCOMP_SUCCESS if successful, or an error.
 */
NVCOMP_EXPORT
bitcompResult_t bitcompDestroyPlan(bitcompHandle_t handle);

//***********************************************************************************************
// Modification of plan attributes

/**
 * @brief Associate a bitcomp handle to a stream. All the subsequent operations will be done in the stream.
 *
 * @param[in, out] handle Bitcomp handle
 * @param[in] stream Stream to use.
 * @return Returns BITCOMP_SUCCESS if successful, or an error.
 */
NVCOMP_EXPORT
bitcompResult_t bitcompSetStream(bitcompHandle_t handle, cudaStream_t stream);

/**
 * @brief Turn on compression acceleration when the compressed output is not in the global memory
 * of the device running the compression (e.g. host pinned memory, or another device's memory)
 * This is optional and only affects the performance.
 * NOTE: This makes the handle become device-specific. A plan that has this acceleration turned on
 * should always be used on the same device.
 *
 * @param[in, out] handle Bitcomp handle.
 * @return Returns BITCOMP_SUCCESS if successful, or an error.
 */
NVCOMP_EXPORT
bitcompResult_t bitcompAccelerateRemoteCompression(bitcompHandle_t handle);

//***********************************************************************************************
// Compression and decompression on the device

/**
 * @brief Compression for FP16 (half) data, running asynchronously on the device.
 *
 * @param[in] handle Bitcomp handle.
 * @param[in] input Pointer to the uncompressed data. Must be accessible from the device.
 * @param[out] output Pointer to the compressed data. Must be accessible from the device and 64-bit aligned.
 * @param[in] delta Delta used for the integer quantization of the data.
 * The maximum error between the uncompressed data and the original data should be <= delta/2.
 * To ensure this, the delta value is rounded down to the nearest power of two (i.e. no mantissa).
 * @return Returns BITCOMP_SUCCESS if successful, or an error.
 */
NVCOMP_EXPORT
bitcompResult_t bitcompCompressLossy_fp16(const bitcompHandle_t handle, const half *input, void *output, half delta);

/**
 * @brief Compression for 32-bit floating point data, running asynchronously on the device.
 *
 * @param[in] handle Bitcomp handle.
 * @param[in] input Pointer to the uncompressed data. Must be accessible from the device.
 * @param[out] output Pointer to the compressed data. Must be accessible from the device and 64-bit aligned.
 * @param[in] delta Delta used for the integer quantization of the data.
 * The maximum error between the uncompressed data and the original data should be <= delta/2.
 * To ensure this, the delta value is rounded down to the nearest power of two (i.e. no mantissa).
 * @return Returns BITCOMP_SUCCESS if successful, or an error.
 */
NVCOMP_EXPORT
bitcompResult_t bitcompCompressLossy_fp32(const bitcompHandle_t handle, const float *input, void *output, float delta);

/**
 * @brief Compression for 64-bit floating point data, running asynchronously on the device.
 *
 * @param[in] handle Bitcomp handle.
 * @param[in] input Pointer to the uncompressed data. Must be accessible from the device.
 * @param[out] output Pointer to the compressed data. Must be accessible from the device and 64-bit aligned.
 * @param[in] delta Delta used for the integer quantization of the data.
 * The maximum error between the uncompressed data and the original data should be <= delta/2.
 * To ensure this, the delta value is rounded down to the nearest power of two (i.e. no mantissa).
 * @return Returns BITCOMP_SUCCESS if successful, or an error.
 */
NVCOMP_EXPORT
bitcompResult_t bitcompCompressLossy_fp64(const bitcompHandle_t handle, const double *input, void *output, double delta);

NVCOMP_EXPORT
bitcompResult_t bitcompCompressLossless(const bitcompHandle_t handle, const void *input, void *output);

/**
 * @brief Decompression, running asynchronously on the device.
 *
 * @param[in] handle Bitcomp handle.
 * @param[in] input Pointer to the compressed data. Must be accessible from the device and 64-bit aligned.
 * @param[out] output Pointer to where the uncompressed data will be written.
 * @return Returns BITCOMP_SUCCESS if successful, or an error.
 */
NVCOMP_EXPORT
bitcompResult_t bitcompUncompress(const bitcompHandle_t handle, const void *input, void *output);

/**
 * @brief Partial decompression, running asynchronously on the device.
 *
 * @param[in] handle Bitcomp handle.
 * @param[in] input Pointer to the compressed data. Must be accessible from the device and 64-bit aligned.
 * @param[out] output Pointer to where the partial uncompressed data will be written.
 * @param[in] start Offset in bytes relative to the original uncompressed size where to start decompressing.
 * @param[in] length Length in bytes of the partial decompression.
 * @return Returns BITCOMP_SUCCESS if successful, or an error.
 */
NVCOMP_EXPORT
bitcompResult_t
bitcompPartialUncompress(const bitcompHandle_t handle, const void *input, void *output, size_t start, size_t length);

//***********************************************************************************************
// Compression and decompression on the host

/**
 * @brief Lossy compression for FP16 (half) data, running on the host processor. This call is blocking.
 * If a non-NULL stream was set in the handle, this call will synchronize the stream.
 * before compressing the data.
 * All arrays must be device accessible.
 *
 * @param[in] handle Bitcomp handle.
 * @param[in] input Pointer to the uncompressed data. Must be accessible from the host.
 * @param[out] output Pointer to the compressed data. Must be accessible from the host and 64-bit aligned.
 * @param[in] delta Delta used for the integer quantization of the data.
 * The maximum error between the uncompressed data and the original data should be <= delta/2.
 * To ensure this, the delta value is rounded down to the nearest power of two (i.e. no mantissa).
 * @return Returns BITCOMP_SUCCESS if successful, or an error
 */
NVCOMP_EXPORT
bitcompResult_t bitcompHostCompressLossy_fp16(const bitcompHandle_t handle, const half *input, void *output, half delta);

/**
 * @brief Lossy compression for 32-bit floats, running on the host processor. This call is blocking.
 * If a non-NULL stream was set in the handle, this call will synchronize the stream
 * before compressing the data.
 * All arrays must be device accessible.
 *
 * @param[in] handle Bitcomp handle.
 * @param[in] input Pointer to the uncompressed data. Must be accessible from the host.
 * @param[out] output Pointer to the compressed data. Must be accessible from the host and 64-bit aligned.
 * @param[in] delta Delta used for the integer quantization of the data.
 * The maximum error between the uncompressed data and the original data should be <= delta/2.
 * To ensure this, the delta value is rounded down to the nearest power of two (i.e. no mantissa).
 * @return Returns BITCOMP_SUCCESS if successful, or an error.
 */
NVCOMP_EXPORT
bitcompResult_t
bitcompHostCompressLossy_fp32(const bitcompHandle_t handle, const float *input, void *output, float delta);

/**
 * @brief Lossy compression for 64-bit floats, running on the host processor. This call is blocking.
 * If a non-NULL stream was set in the handle, this call will synchronize the stream
 * before compressing the data.
 * All arrays must be device accessible.
 *
 * @param[in] handle Bitcomp handle.
 * @param[in] input Pointer to the uncompressed data. Must be accessible from the host.
 * @param[out] output Pointer to the compressed data. Must be accessible from the host and 64-bit aligned.
 * @param[in] delta Delta used for the integer quantization of the data.
 * The maximum error between the uncompressed data and the original data should be <= delta/2.
 * To ensure this, the delta value is rounded down to the nearest power of two (i.e. no mantissa).
 * @return Returns BITCOMP_SUCCESS if successful, or an error.
 */
NVCOMP_EXPORT
bitcompResult_t
bitcompHostCompressLossy_fp64(const bitcompHandle_t handle, const double *input, void *output, double delta);

/**
 * @brief Lossless compression (integral datatypes), running on the host processor. This call is blocking.
 * If a non-NULL stream was set in the handle, this call will synchronize the stream
 * before compressing the data.
 * All arrays must be device accessible.
 *
 * @param[in] handle Bitcomp handle.
 * @param[in] input Pointer to the uncompressed data. Must be accessible from the host.
 * @param[out] output Pointer to the compressed data. Must be accessible from the host and 64-bit aligned.
 * @return Returns BITCOMP_SUCCESS if successful, or an error.
 */
NVCOMP_EXPORT
bitcompResult_t bitcompHostCompressLossless(const bitcompHandle_t handle, const void *input, void *output);

/**
 * @brief Decompression, running on the host processor. This call is blocking.
 * If a non-NULL stream was set in the handle, this call will synchronize the stream
 * before decompressing the data.
 * All arrays must be device accessible.
 *
 * @param[in] handle Bitcomp handle.
 * @param[in] input Pointer to the compressed data. Must be accessible from the host and 64-bit aligned.
 * @param[out] output Pointer to the uncompressed data. Must be accessible from the host.
 * @return Returns BITCOMP_SUCCESS if successful, or an error.
 */
NVCOMP_EXPORT
bitcompResult_t bitcompHostUncompress(const bitcompHandle_t handle, const void *input, void *output);

/**
 * @brief Partial decompression, running on the host processor. This call is blocking.
 * If a non-NULL stream was set in the handle, this call will synchronize the stream
 * before decompressing the data.
 * All arrays must be device accessible.
 *
 * @param[in] handle Bitcomp handle.
 * @param[in] input Pointer to the compressed data. Must be accessible from the host and 64-bit aligned.
 * @param[out] output Pointer to where the partial uncompressed data will be written.
 * @param[in] start Offset in bytes relative to the original uncompressed size where to start decompressing.
 * @param[in] length Length in bytes of the partial decompression.
 * @return Returns BITCOMP_SUCCESS if successful, or an error
 */
NVCOMP_EXPORT
bitcompResult_t
bitcompHostPartialUncompress(const bitcompHandle_t handle, const void *input, void *output, size_t start, size_t length);

// *******************************************************************************************************************
// Utilities

/**
 * @brief Query the maximum size (worst case scenario) that the compression could
 * generate given an input size.
 *
 * @param[in] nbytes Size of the uncompressed data, in bytes.
 * @return Returns the maximum size of the compressed data, in bytes.
 */
NVCOMP_EXPORT
size_t bitcompMaxBuflen(size_t nbytes);

/**
 * @brief Query the compressed size from a compressed buffer.
 * The pointers don't have to be device-accessible. This is a blocking call.
 * The compression must have completed before calling this function.
 *
 * @param[in] compressedData Pointer to the compressed data.
 * @param[out] size Size of the compressed data, in bytes.
 * @return Returns BITCOMP_SUCCESS if successful, or an error.
 */
NVCOMP_EXPORT
bitcompResult_t bitcompGetCompressedSize(const void *compressedData, size_t *size);

/**
 * @brief Query the compressed size from a compressed buffer, asynchronously.
 * Both pointers must be device-accessible.
 *
 * @param[in] compressedData Pointer to the compressed data.
 * @param[out] size Size of the compressed data, in bytes.
 * @param[in] stream Stream for asynchronous operation.
 * @return Returns BITCOMP_SUCCESS if successful, or an error.
 */
NVCOMP_EXPORT
bitcompResult_t bitcompGetCompressedSizeAsync(const void *compressedData, size_t *size, cudaStream_t stream);

/**
 * @brief Query the uncompressed size from a compressed buffer
 *
 * @param[in] compressedData Pointer to the compressed data buffer,
 * The pointer doesn't have to be device-accessible.
 * @param[out] size Size of the uncompressed data, in bytes.
 * @return Returns BITCOMP_SUCCESS if successful, or an error.
 */
NVCOMP_EXPORT
bitcompResult_t bitcompGetUncompressedSize(const void *compressedData, size_t *size);

/**
 * @brief Query the uncompressed size from a handle.
 *
 * @param[in] handle handle.
 * @param[out] bytes Size in bytes of the uncompressed data.
 * @return Returns BITCOMP_SUCCESS if successful, or an error.
 */
NVCOMP_EXPORT
bitcompResult_t bitcompGetUncompressedSizeFromHandle(const bitcompHandle_t handle, size_t *bytes);

/**
 * @brief Query the uncompressed datatype from a handle.
 *
 * @param[in] handle handle.
 * @param[out] dataType Data type of the uncompressed data.
 * @return Returns BITCOMP_SUCCESS if successful, or an error.
 */
NVCOMP_EXPORT
bitcompResult_t bitcompGetDataTypeFromHandle(const bitcompHandle_t handle, bitcompDataType_t *dataType);

/**
 * @brief: Query compressed data information.
 * @param[in] compressedData Compressed data pointer. Doesn't have to be device-accessible.
 * @param[in, out] compressedDataSize Takes size of the compressed buffer. Stores actual size of the compressed data
 * If the size of the compressed buffer is smaller than the actual size of the compressed data,
 * BITCOMP_INVALID_PARAMETER will be returned.
 * @param[out] uncompressedSize The size of the uncompressed data in bytes.
 * @param[out] dataType The type of the compressed data.
 * @param[out] mode Compression mode (lossy or lossless).
 * @param[out] algo Bitcomp algorithm used (default, or sparse).
 * @return Returns BITCOMP_SUCCESS if successful, or an error.
 */
NVCOMP_EXPORT
bitcompResult_t bitcompGetCompressedInfo(
  const void *compressedData,
  size_t *compressedDataSize,
  size_t *uncompressedSize,
  bitcompDataType_t *dataType,
  bitcompMode_t *mode,
  bitcompAlgorithm_t *algo
);

#ifdef __cplusplus
}
#endif
