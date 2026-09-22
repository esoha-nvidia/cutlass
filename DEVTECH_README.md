```
srun -A coreai_devtech_all -N1 -p gb200 -J coreai_devtech_all-esoha.nvcomp \
  --container-image="nvcr.io/nvidia/pytorch:26.08-py3" \
  --container-mounts="/home/esoha" --pty bash

# 26.08-py3 is CUDA 13.4. Default nvidia-cutlass-dsl is CUDA 12 and will
# fail on import (OpOperands / mixed _cutlass_ir.so). Install the cu13 extra.
pip uninstall -y nvidia-cutlass-dsl nvidia-cutlass-dsl-libs-base \
  nvidia-cutlass-dsl-libs-core nvidia-cutlass-dsl-libs-cu12 \
  nvidia-cutlass-dsl-libs-cu13 nvidia-cutlass-operators

pip install "nvidia-cutlass-dsl[cu13]==4.8.0"
pip install nvidia-cutlass-operators[torch]

python -c "import cutlass; import torch; print(cutlass.__version__, torch.cuda.is_available(), torch.version.cuda)"

cd /home/esoha/cutlass
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

From c++:
```
cd /home/esoha/cutlass
mkdir -p build && cd build

export CUDACXX=$(which nvcc)   # usually /usr/local/cuda/bin/nvcc

cmake .. \
  -DCUTLASS_NVCC_ARCHS=100a \
  -DCUTLASS_ENABLE_TESTS=OFF

make 00_basic_gemm -j$(nproc)

./examples/00_basic_gemm/00_basic_gemm

../../nsight-systems-2026.4.1/bin/nsys profile ./examples/00_basic_gemm/00_basic_gemm
```

