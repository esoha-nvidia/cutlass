/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * Written by Mauro Bisson <maurob@nvidia.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef __CRC32_H__
#define __CRC32_H__

#include <cuda_runtime.h>

typedef struct
{
  unsigned int poly;
  unsigned int init;
  int ref_in;
  int ref_out;
  unsigned int xorout;
} crcSpec_t;

typedef struct
{
  unsigned int *crcBuf;
  crcSpec_t spec;
} crcCtx_t;

#define CUCRC32_SUCCESS (0)
#define CUCRC32_ERROR_INVALID_PARAMS (1)
#define CUCRC32_ERROR_INVALID_KERNEL (2)
#define CUCRC32_ERROR_INVALID_READBYTES (3)
#define CUCRC32_ERROR_INVALID_BLKXMSG (4)

int cuCRC32ConfSearch(
  const crcCtx_t *ctx,
  unsigned int nMsg,
  const unsigned long long *msgLen_d,
  const unsigned char *const *msg_d,
  unsigned int *conf,
  cudaStream_t stream = 0
);

int cuCRC32Beg(const crcCtx_t *ctx, unsigned int nMsg, cudaStream_t stream = 0);

int cuCRC32Add(
  const crcCtx_t *ctx,
  unsigned int conf,
  unsigned int nMsg,
  const unsigned long long *msgLen_d,
  const unsigned char *const *msg_d,
  cudaStream_t stream = 0
);

int cuCRC32End(const crcCtx_t *ctx, unsigned int nMsg, cudaStream_t stream = 0);

int cuCRC32ParamToConf(int kernel, int readBytes, int blocksPerMsg, unsigned int *conf);

int cuCRC32ConfToParam(unsigned int conf, int *kernel, int *readBytes, int *blocksPerMsg);

int cuCRC32ConfHeur(
  const crcCtx_t *ctx,
  unsigned int nMsg,
  const unsigned long long *msgLen_d,
  unsigned int *conf,
  unsigned long long maxMsgLen = 0,
  cudaStream_t stream = 0
);

int cuCRC32DispatchHeur(
  int arch,
  int n_sm,
  int max_grid_dim_y,
  unsigned int nMsg,
  unsigned long long maxMsgLen,
  unsigned int *conf
);

#endif
