```bash
ssh oci-hsg-cs-001-login-02
```

```bash
srun --partition=batch -A coreai_devtech_all --job-name devtech-benchmarking:shell --qos=interactive --gpus-per-node=4 --time=4:00:00 --container-image=nvcr.io/nvidia/pytorch:26.08-py3   --container-mounts=/lustre/fsw/portfolios/coreai/users/esoha --pty /bin/bash -

# 26.08-py3 is CUDA 13.4. Default nvidia-cutlass-dsl is CUDA 12 and will
# fail on import (OpOperands / mixed _cutlass_ir.so). Install the cu13 extra.
pip uninstall -y nvidia-cutlass-dsl nvidia-cutlass-dsl-libs-base \
  nvidia-cutlass-dsl-libs-core nvidia-cutlass-dsl-libs-cu12 \
  nvidia-cutlass-dsl-libs-cu13 nvidia-cutlass-operators

pip install "nvidia-cutlass-dsl[cu13]==4.8.0"
pip install nvidia-cutlass-operators[torch]

python -c "import cutlass; import torch; print(cutlass.__version__, torch.cuda.is_available(), torch.version.cuda)"

cd /lustre/fsw/portfolios/coreai/users/esoha/cutlass
python ./cutlass_test.py

# Capture only the CUTLASS GEMM (JIT/setup happen before cudaProfilerStart).
# torch.cuda.nvtx uses unregistered strings; nsys --capture-range=nvtx ignores
# those unless NSYS_NVTX_PROFILER_REGISTER_ONLY=0. cudaProfilerApi is reliable.
../nsight-systems-2026.4.1/bin/nsys profile \
  -t cuda,nvtx,cublas \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --stats=true -o cutlass_gemm \
  python ./cutlass_test.py

# NVTX capture (plain torch.cuda.nvtx strings) — needs this env var:
# NSYS_NVTX_PROFILER_REGISTER_ONLY=0 ../nsight-systems-2026.4.1/bin/nsys profile \
#   -t cuda,nvtx,cublas \
#   --capture-range=nvtx --nvtx-capture=cutlass_gemm --capture-range-end=stop \
#   --stats=true -o cutlass_gemm \
#   python ./cutlass_test.py
```

## nvCOMP (in-tree LLIF)

Host batched ANS and in-kernel fused ANS both come from the copy of nvCOMP under `nvcomp/`. `00_basic_gemm` compiles that as a **static** library (no relocatable device code) and inlines `compress_chunk` in the GEMM CTA. It does not use MathDx / nvCOMPDx.

Do **not** pass `-DBUILD_NVCOMPDX=ON` — that FetchContent-clones internal GitLab over SSH, which this cluster cannot reach.

For c++:
```bash
cd /lustre/fsw/portfolios/coreai/users/esoha/cutlass
mkdir -p build && cd build

export CUDACXX=$(which nvcc)   # usually /usr/local/cuda/bin/nvcc

cmake .. \
  -DCUTLASS_NVCC_ARCHS=100a \
  -DCUTLASS_ENABLE_TESTS=OFF \
  -DCMAKE_CUDA_FLAGS="-lineinfo"

make 00_basic_gemm -j$(nproc)

./examples/00_basic_gemm/00_basic_gemm
./examples/00_basic_gemm/00_basic_gemm --fuse-nvcomp
./examples/00_basic_gemm/00_basic_gemm --nvcomp-only
./examples/00_basic_gemm/00_basic_gemm --nvcomp-unfused

../../nsight-systems-2026.4.1/bin/nsys profile -t cuda,nvtx,cublas \
  -o cutlass_gemm --force-overwrite true ./examples/00_basic_gemm/00_basic_gemm

../../nsight-systems-2026.4.1/bin/nsys profile -t cuda,nvtx,cublas \
  -o cutlass_gemm_nvcomp --force-overwrite true ./examples/00_basic_gemm/00_basic_gemm --fuse-nvcomp

../../nsight-systems-2026.4.1/bin/nsys profile -t cuda,nvtx,cublas \
  -o cutlass_nvcomp_only --force-overwrite true ./examples/00_basic_gemm/00_basic_gemm --nvcomp-only

../../nsight-systems-2026.4.1/bin/nsys profile -t cuda,nvtx,cublas \
  -o cutlass_nvcomp_unfused --force-overwrite true ./examples/00_basic_gemm/00_basic_gemm --nvcomp-unfused

# --set full + --import-source needs -lineinfo (CMAKE_CUDA_FLAGS above).
# NCU function basename: fused is gemm_fused_ans_kernel, unfused GEMM is gemm_unfused_kernel.
NCU=../../nsight/ncu/nsight_compute/ncu

$NCU --set full --import-source yes \
  -k regex:'gemm_fused_ans_kernel' \
  -o cutlass_gemm_nvcomp_ncu --force-overwrite \
  ./examples/00_basic_gemm/00_basic_gemm --fuse-nvcomp

$NCU --set full --import-source yes \
  -k regex:'(de)?compress_kernel' \
  -o cutlass_nvcomp_only_ncu --force-overwrite \
  ./examples/00_basic_gemm/00_basic_gemm --nvcomp-only

$NCU --set full --import-source yes \
  -k regex:'gemm_unfused_kernel|(de)?compress_kernel' \
  -o cutlass_nvcomp_unfused_ncu --force-overwrite \
  ./examples/00_basic_gemm/00_basic_gemm --nvcomp-unfused

$NCU --set full --import-source yes \
  -k regex:'gemm_unfused_kernel' \
  -o cutlass_gemm_ncu --force-overwrite \
  ./examples/00_basic_gemm/00_basic_gemm

# Blackwell tcgen05 GEMM then ANS (1SM 128x128 tiles, FP16 C).
# Unfused GEMM basename is typically device_kernel; fused GEMM is
# gemm_fused_ans_kernel, fused ANS is fused_ans_compress_kernel.
make 70_blackwell_fp16_gemm_nvcomp -j$(nproc)

./examples/70_blackwell_gemm/70_blackwell_fp16_gemm_nvcomp --m=8192 --n=8192 --k=2048 --iterations=1
./examples/70_blackwell_gemm/70_blackwell_fp16_gemm_nvcomp --fuse-nvcomp --iterations=1 --m=8192 --n=8192 --k=2048

../../nsight-systems-2026.4.1/bin/nsys profile -t cuda,nvtx \
  -o blackwell_gemm_nvcomp_unfused --force-overwrite true \
  ./examples/70_blackwell_gemm/70_blackwell_fp16_gemm_nvcomp --m=8192 --n=8192 --k=2048 --iterations=1

../../nsight-systems-2026.4.1/bin/nsys profile -t cuda,nvtx \
  -o blackwell_gemm_nvcomp --force-overwrite true \
  ./examples/70_blackwell_gemm/70_blackwell_fp16_gemm_nvcomp --fuse-nvcomp --iterations=1 --m=8192 --n=8192 --k=2048

$NCU --set full --import-source yes \
  -k regex:'device_kernel|compress_kernel' \
  -o blackwell_gemm_nvcomp_unfused_ncu --force-overwrite \
  ./examples/70_blackwell_gemm/70_blackwell_fp16_gemm_nvcomp --m=8192 --n=8192 --k=2048 --iterations=1

$NCU --set full --import-source yes \
  -k regex:'gemm_fused_ans_kernel|fused_ans_compress_kernel' \
  -o blackwell_gemm_nvcomp_ncu --force-overwrite \
  ./examples/70_blackwell_gemm/70_blackwell_fp16_gemm_nvcomp --fuse-nvcomp --iterations=1 --m=8192 --n=8192 --k=2048
```

