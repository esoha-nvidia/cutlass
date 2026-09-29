/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
*/

#include "bitcomp_private.h"
#include "cpu_transpose.h"
#include "header.h"
#include "rle_common.h"

#include <stdio.h>

#pragma once

namespace bitcomp
{

namespace rle
{

// ****************************************************************************
// RLE encoder

inline int encoder(const unsigned char *bufin, unsigned char *bufout)
{
  int lenData = 2; // First 2 bytes = the number of codes
  int lenCode = 0;
  int lenExt = 0;
  // Worst case scenario for RLE is 4096 codes
  // AAA B CCC D EEE F GGG H III J ...
  // i.e, a repeated group is followed by a non-repeat group
  int tmpCode[4096];
  // Worst case scenario is 256 extensions
  // Ax32 Bx32 Cx32 Dx32
  // or
  // Ax32 BCDE...(32 long) Bx32 CDE...(32 long) ...
  int tmpExt[256]; // Long = 33+ for non-dup, or 35+ for dups.

  int icur = 0;
  int nonrepeat = 0;
  while (icur < NOMINAL_BLOCK_SIZE)
  {
    unsigned char cur = bufin[icur];
    //Look for at least 3 repetitions of the current byte
    int repeat = 1;
    while ((icur + repeat < NOMINAL_BLOCK_SIZE) && (cur == bufin[icur + repeat]))
    {
      repeat++;
    }
    if (repeat >= 3)
    {
      // Write the control code if a non-repeat string just ended
      if (nonrepeat)
      {
        // sumn += nonrepeat;
        // nn++;
        // Non-repeat lengths are encoded as (length-1)
        nonrepeat--;
        if (nonrepeat < 32) // short length, in 1 byte
        {
          tmpCode[lenCode++] = CTRL_NONDUP | nonrepeat;
        }
        else
        {
          tmpCode[lenCode++] = CTRL_NONDUP | CTRL_LONG | (nonrepeat >> 8);
          tmpExt[lenExt++] = nonrepeat & 0xff;
        }
        nonrepeat = 0;
      }
      // Now process the repeat chain, encoded with (length-3)
      // sumr += repeat;
      // nr++;
      icur += repeat;
      repeat -= 3;
      unsigned char control;
      if (cur == 0)
      {
        control = CTRL_DUP00;
      }
      else if (cur == 0xff)
      {
        control = CTRL_DUPFF;
      }
      else
      {
        control = CTRL_DUP;
        bufout[lenData++] = cur;
      }
      if (repeat < 32) // short length, in 1 byte
      {
        tmpCode[lenCode++] = control | repeat;
      }
      else
      {
        tmpCode[lenCode++] = control | CTRL_LONG | (repeat >> 8);
        tmpExt[lenExt++] = repeat & 0xff;
      }
    }
    else
    {
      // Not part of a repetition
      nonrepeat++;
      bufout[lenData++] = cur;
      icur++;
    }
  }
  // Write the control code of the last non-repeat string
  if (nonrepeat)
  {
    // Non-repeat lengths are encoded as (length-1)
    nonrepeat--;
    if (nonrepeat < 32) // short length, in 1 byte
    {
      tmpCode[lenCode++] = CTRL_NONDUP | nonrepeat;
    }
    else
    {
      tmpCode[lenCode++] = CTRL_NONDUP | CTRL_LONG | (nonrepeat >> 8);
      tmpExt[lenExt++] = nonrepeat & 0xff;
    }
  }

  // Total compressed length in words
  int total = lenData + lenExt + lenCode;
  int lcomp = (total + 3) / 4;

  // Store the number of codes at the very beginning of the data
  bufout[0] = lenCode >> 8;
  bufout[1] = lenCode & 0xff;

  // Store the codes and code extensions backward from the end
  for (int i = 0; i < lenCode; i++)
  {
    bufout[lcomp * 4 - 1 - i] = tmpCode[i];
  }
  for (int i = 0; i < lenExt; i++)
  {
    bufout[lcomp * 4 - lenCode - 1 - i] = tmpExt[i];
  }

  return lcomp;
}

inline void decoder(const unsigned char *bufin, int lcomp, unsigned char *bufout)
{
  int lenCode = (bufin[0] << 8) + bufin[1];
  int codeStart = lcomp * 4 - 1;
  int iData = 2;
  int iExt = 0;
  int icur = 0;

  for (int iCode = 0; iCode < lenCode; iCode++)
  {
    unsigned char control = bufin[codeStart - iCode];
    int length = control & 0x1f;
    if (control & CTRL_LONG)
    {
      length = (length << 8) + bufin[codeStart - lenCode - iExt];
      iExt++;
    }
    switch (control & CTRL_DUP)
    {
      case CTRL_NONDUP:
        for (int i = 0; i < length + 1; i++)
        {
          bufout[icur++] = bufin[iData++];
        }
        break;
      case CTRL_DUP00:
        for (int i = 0; i < length + 3; i++)
        {
          bufout[icur++] = 0;
        }
        break;
      case CTRL_DUPFF:
        for (int i = 0; i < length + 3; i++)
        {
          bufout[icur++] = 0xff;
        }
        break;
      case CTRL_DUP:
        char val = bufin[iData++];
        for (int i = 0; i < length + 3; i++)
        {
          bufout[icur++] = val;
        }
        break;
    }
  }
}

} // namespace rle

} // namespace bitcomp
