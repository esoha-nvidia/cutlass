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

# Capture only the NVTX range around the GEMM (skips JIT / setup).
# Use -c cutlass_gemm or -c cublas_gemm.
../nsight-systems-2026.4.1/bin/nsys profile \
  -t cuda,nvtx,cublas \
  --capture-range=nvtx --nvtx-capture=cutlass_gemm --capture-range-end=stop \
  --stats=true -o cutlass_gemm \
  python ./cutlass_test.py
```
