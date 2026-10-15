# GEMM + ANS fusion attempts

Summary of every attempt to fuse nvCOMP ANS with CUTLASS GEMM on `profile_gemm` (Sep 22 – Oct 8, 2026). None of these commits put ANS in the four SM100 epilogue warps after store. Fusion in this history is either **same-CTA-after-GEMM** or **two kernels**.

Ampere lives in `examples/00_basic_gemm/basic_gemm.cu`. Blackwell lives in `examples/70_blackwell_gemm/70_blackwell_fp16_gemm_nvcomp.cu`.

## What never landed

- ANS inside `LinearCombination`
- ANS in the **4 epilogue warps after each store**
- A 1-warp ANS compressor
- nvCOMPDx on the SM100 persistent kernel

## 1. Hook in the CUTLASS epilogue functor (not ANS)

**`677e161a` (Sep 22)** — `PrintTileEpilogue` subclasses `LinearCombination` and `printf`s the first fragment. That showed the functor never holds a 128×128 tile, so ANS cannot live there.

## 2. Host nvCOMP after GEMM (called “epilogue” in comments)

**`6a80ac0c`** — GEMM, then pack tiles and `nvcompBatchedANSCompressAsync`. Separate kernel/API, not fused.

## 3. nvCOMPDx as a second kernel (unfused)

**`415ec745`**, **`f4db1782`**, **`2955e9a1`** — `compress_*_nvcompdx_kernel`: 256 threads, `BlockWarp<8>`, `SM<1000>`. Pack a 128×128 tile, then `compressor.execute()`. **`--nvcomp-only`** compresses a matrix with no GEMM.

Still two launches.

## 4. First real fusion: same CTA after Ampere GEMM (nvCOMPDx)

**`68c4f130`**, **`--fuse-nvcomp` in `01afed27`**

`GemmFusedAns` runs the SIMT `128×128×8` GEMM (`OpClassSimt`, 256 threads), `__syncthreads()`, then packs **that CTA’s** tile from `D` and runs nvCOMPDx on reused SMEM.

Comments already say CUTLASS `device::Gemm` has no post-epilogue hook, so the example launches `Kernel<GemmFusedAns>` itself. This is **after the whole GEMM**, not inside LinearCombination.

Follow-ups: round-trip decompress (`b780c2fd`, `fcadde43`), nvCOMPDx vs LLIF bitstream compare (`e2be7ff3`), launch bounds (`b47fd059`).

## 5. Same fusion, LLIF instead of nvCOMPDx

**`1599a55a`** — drop nvCOMPDx; fused path calls `nvcompDeviceANSCompressChunk` (needed RDC).

**`8644473c` / `fcf94447` / `c817d236`** — `nvcompBatchedANSCompressGetDeviceLaunchParams`, smem alignment, zero compressed padding so device/host bitstreams match.

**`00c6e2ac`** — inline `compress_chunk` in the GEMM TU, **no RDC**. This is the Ampere fused shape that stuck.

## 6. Unfused pack+compress in one kernel (not fused with GEMM)

**`12a42de4`**, **`e727cc21`**, **`--nvcomp-unfused` in `efe1d199`** — `nvcompBatchedANSCompressFromColMajorTilesAsync`: stock `compress_kernel` packs column-major `C` then ANS. One kernel for pack+compress, **after** GEMM.

**`485c43be`** — `gemm_unfused_kernel` with the same 256-thread tile so NCU can compare fused vs unfused GEMM.

**`369a54c7`** — `--nvcomp-only` runs GEMM first, then that pack+compress path.

## 7. Keep 256 threads while changing the Ampere MMA

Fusion still required **8 warps**. Later Ampere commits only change the MMA, not the fusion shape:

| Commit | Change |
|---|---|
| **`43f8b20c`** | TF32 TensorOp `128×128×32` (still 256 threads) |
| **`c133eda1`** | fp16 tiles / `NVCOMP_TYPE_FLOAT16` |
| **`892f99d9`** | bf16 |
| **`ffbfb7bd`** | back to fp16 |

## 8. Blackwell: unfused first

**`5864c387`** — `70_blackwell_fp16_gemm_nvcomp`: SM100 persistent GEMM, then stock `compress_kernel`. No fusion.

## 9. Blackwell attempt A: same CTA after the persistent GEMM

**`4b413fba`** — `gemm_fused_ans_kernel`: full `GemmKernel`, then `fused_ans_compress_tiles` on the same 256-thread persistent CTA, grid-stride on **`blockIdx.x` only**.

This is the one that hit, in order:

1. `cudaFuncSetAttribute`: invalid argument (ANS static `__shared__` + 227 KiB GEMM)
2. `misaligned address` (TMA base not 128-byte aligned)
3. 3105/4096 tiles failed (every CTA thought it was CTA 0)

## 10. Blackwell attempt A, patched in place

**`27469f0b`** — 16 KiB SMEM carveout, `extern __shared__ __align__(128)`, 3-D linear CTA id, `cudaMemset` on sizes.

GEMM Passed, ratio jumped to ~39×, then **tile 152** `nvcompErrorCannotDecompress`. CLC warps never reconverge; overlaying `compress_chunk` on GEMM SMEM after named barriers produced junk headers.

## 11. Blackwell attempt B (current): two kernels, still called `--fuse-nvcomp`

**`71c0dc6e`**

1. `gemm_fused_ans_kernel` — GEMM only
2. `fused_ans_compress_kernel` — one 256-thread CTA per 128×128 tile, clean SMEM, `compress_chunk`

Same stream, not same CTA, not in-epilogue. That is what the tree does now.

## End state

Ampere fusion that works is attempt **4/5**: same 256-thread CTA after SIMT/TensorOp GEMM (`--fuse-nvcomp` on `00_basic_gemm`).

Blackwell never got that; the surviving `--fuse-nvcomp` on `70_blackwell_fp16_gemm_nvcomp` is attempt **11** (GEMM kernel then ANS kernel).
