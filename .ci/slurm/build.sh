#!/usr/bin/env bash
set -euo pipefail
ddla_root=${1:?run directory}
ddla_control=$ddla_root/control
ddla_out=$ddla_root/results/build
mkdir -p "$ddla_out"
module -t list > "$ddla_out/modules.txt" 2>&1
printf '%s\n' "$MPI_HOME" "$CUDA_HOME" "$NCCL_ROOT" "$OPENBLAS_ROOT" > "$ddla_out/dependency-roots.txt"
cmake -S "$ddla_root/source" -B "$ddla_root/build" \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON -DDDLA_USE_CUDA=ON -DDDLA_USE_HIP=OFF -DDDLA_USE_CCL=ON \
    -DDDLA_ENABLE_FORTRAN=OFF -DDDLA_ENABLE_PYTHON=OFF \
    -DCMAKE_CXX_COMPILER=/opt/devtools/gcc/13.3.0/bin/g++ \
    -DCMAKE_CUDA_HOST_COMPILER=/opt/devtools/gcc/13.3.0/bin/g++ \
    -DCMAKE_CUDA_COMPILER=/opt/devtools/nvidia/cuda-12.9.1/bin/nvcc \
    -DCUDAToolkit_ROOT=/opt/devtools/nvidia/cuda-12.9.1 -DCMAKE_CUDA_ARCHITECTURES=70 \
    -DDDLA_CPU_BLAS_LIBRARY="$OPENBLAS_ROOT/lib/libopenblas.so" -DDDLA_TEST_NP=4 \
    -DMPIEXEC_EXECUTABLE="$ddla_control/mpiexec.sh" -DMPIEXEC_NUMPROC_FLAG=-n -DMPIEXEC_PREFLAGS= \
    > "$ddla_out/configure.log" 2>&1
cp "$ddla_root/build/CMakeCache.txt" "$ddla_out/CMakeCache.txt"
cmake --build "$ddla_root/build" --parallel "${2:?build parallelism}" > "$ddla_out/build.log" 2>&1
nvcc -arch=sm_70 "$ddla_control/gpu_probe.cu" -o "$ddla_root/build/gpu_probe" > "$ddla_out/probe-build.log" 2>&1
ldd "$ddla_root/build/src/libddla.so" > "$ddla_out/ldd.txt"
python3 - "$ddla_out/ldd.txt" <<'PY'
from pathlib import Path
import sys
text = Path(sys.argv[1]).read_text()
assert 'not found' not in text, text
for library in ['libcudart.so.12', 'libcublas.so.12', 'libcusolver.so.11']:
    line = next(line for line in text.splitlines() if line.lstrip().startswith(library + ' '))
    assert '/cuda-12.9.1/' in line, line
line = next(line for line in text.splitlines() if line.lstrip().startswith('libnccl.so.2 '))
assert '/nccl_2.29.3_cuda12.9_sai_v2.29.3-1-sai.2/' in line, line
PY
ctest --test-dir "$ddla_root/build" --show-only=json-v1 > "$ddla_root/results/inventory.json"
python3 "$ddla_control/runtime.py" plan "$ddla_root"
