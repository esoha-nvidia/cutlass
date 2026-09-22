import sys
from contextlib import contextmanager

import numpy as np
import torch

import cutlass.operators as ops

torch.manual_seed(2025)


@contextmanager
def nvtx_range(name: str):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()


@contextmanager
def profile_range(name: str):
    """NVTX label plus cudaProfilerStart/Stop (nsys --capture-range=cudaProfilerApi)."""
    torch.cuda.nvtx.range_push(name)
    torch.cuda.profiler.start()
    try:
        yield
    finally:
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
        torch.cuda.nvtx.range_pop()


# --- 1. cuBLAS TF32 GEMM (no CUTLASS JIT) ---
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

a = torch.randn(128, 128, device="cuda", dtype=torch.float32)
b = torch.randn(128, 128, device="cuda", dtype=torch.float32)

# Warmup: cuBLAS handle / kernels, outside the captured NVTX range.
c = torch.mm(a, b)
torch.cuda.synchronize()

with nvtx_range("cublas_gemm"):
    c = torch.mm(a, b)

ref = np.matmul(a.cpu().numpy(), b.cpu().numpy())
torch.testing.assert_close(c.cpu(), torch.from_numpy(ref), atol=5e-2, rtol=1e-2)
print("cuBLAS TF32 GEMM matched NumPy (within TF32 tolerance)")

# --- 2. CUTLASS Operator GEMM with fused epilogue (Blackwell / sm_100f) ---
if not (status := ops.utils.device.device_or_env_supports("100f")):
    print(f"Skipping fused epilogue: needs sm_100f.\n{status.error}")
    sys.exit(0)

L, M, N, K = 1, 256, 256, 256
A = torch.randint(-2, 3, (L, M, K), device="cuda", dtype=torch.float16)
B = torch.randint(-2, 3, (L, K, N), device="cuda", dtype=torch.float16)
C = torch.randint(-2, 3, (L, M, N), device="cuda", dtype=torch.float16)


def my_epilogue(accum, C, alpha, beta, extra_scalar):
    Aux = (alpha * accum) + (beta * C)
    D = extra_scalar * Aux
    return D, Aux


alpha, beta, extra_scalar = 1.0, 2.0, 0.5
D = torch.empty((L, M, N), device="cuda", dtype=torch.float16)
Aux = torch.empty((L, M, N), device="cuda", dtype=torch.float16)

epi_args = ops.EpilogueArguments(
    my_epilogue,
    C=C,
    alpha=alpha,
    beta=beta,
    extra_scalar=extra_scalar,
    D=D,
    Aux=Aux,
)
args = ops.GemmArguments(
    A=A, B=B, out=D, accumulator_type=torch.float32, epilogue=epi_args
)
target_sm = ops.utils.device.device_or_env_target_sm()
operator = ops.get_operators(args, target_sm=target_sm, limit=1)[0]

# Warmup: JIT-compile outside the captured NVTX range.
operator.run(args)
torch.cuda.synchronize()

with profile_range("cutlass_gemm"):
    operator.run(args)

D_ref, Aux_ref = my_epilogue(A @ B, C, alpha, beta, extra_scalar)
torch.testing.assert_close(D, D_ref)
torch.testing.assert_close(Aux, Aux_ref)
print("CUTLASS fused epilogue matched PyTorch GEMM + epilogue")
