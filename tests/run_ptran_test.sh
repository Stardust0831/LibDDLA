#!/bin/bash
#SBATCH -p v100g32fat
#SBATCH -J test_ptran
#SBATCH -A renxg
#SBATCH --nodes=1
#SBATCH --gres=gpu:6
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=3
#SBATCH --output=/data/home/renxg/app/github/LibDDLA/tests/ptran_test_%j.log
#SBATCH --error=/data/home/renxg/app/github/LibDDLA/tests/ptran_test_%j.err

module load gcc/11.3.0
module load openmpi/4.1.8-cuda
module load cmake/3.25.3
source /data/home/renxg/app/nvhpc/setup_nvhpc

LibDDLA_PATH=/data/home/renxg/app/github/LibDDLA
BUILD_DIR=${LibDDLA_PATH}/build
INSTALL_DIR=${LibDDLA_PATH%LibDDLA}_install

export CPATH=$INSTALL_DIR/include:$CPATH
export LIBRARY_PATH=$INSTALL_DIR/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$INSTALL_DIR/lib:$LD_LIBRARY_PATH
export OMPI_MCA_btl_openib_allow_ib=1

echo "=== Job started on $(hostname) ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1 | head -2
echo "=== Rebuilding library ==="

cd ${LibDDLA_PATH}
cmake -B ${BUILD_DIR} -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
      -DCMAKE_CXX_COMPILER=g++ \
      -DDDLA_USE_CUDA=ON \
      -DDDLA_USE_CCL=ON \
      -DCMAKE_CUDA_ARCHITECTURES=70

cmake --build ${BUILD_DIR} -j 8
cmake --install ${BUILD_DIR} --prefix ${INSTALL_DIR}

if [ $? -ne 0 ]; then
    echo "❌ Library build failed"
    exit 1
fi

echo "=== Compiling test_ptran ==="
cd ${LibDDLA_PATH}/tests
mpicxx -g -O2 -lcudart -lddla -fopenmp -lcublas -lcusolver -lcurand \
    test_ptran.cpp -o ./test_ptran -std=c++17 -DDDLA_USE_CUDA -DDDLA_USE_CCL

if [ $? -ne 0 ]; then
    echo "❌ test_ptran compilation failed"
    exit 1
fi

echo "============ Running test_ptran (6 procs, 2x3 grid) ============"
mpirun -np 6 --mca btl_tcp_if_include ib0,ib1 ./test_ptran
echo "=== test_ptran exit: $? ==="