#!/bin/bash
#SBATCH -p v100g32fat
#SBATCH -J validate
#SBATCH -A renxg
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=3
#SBATCH --output=/data/home/renxg/app/github/LibDDLA/tests/validate_%j.log
#SBATCH --error=/data/home/renxg/app/github/LibDDLA/tests/validate_%j.err

module load gcc/11.3.0
module load openmpi/4.1.8-cuda
source /data/home/renxg/app/nvhpc/setup_nvhpc

cd /data/home/renxg/app/github/LibDDLA/tests
export LD_LIBRARY_PATH=/data/home/renxg/app/github/LibDDLA_install/lib:$LD_LIBRARY_PATH
export OMPI_MCA_btl_openib_allow_ib=1

echo "=== Job started on $(hostname) ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1 | head -4

echo "============ 1. test_pzgemm (4 procs) ============"
mpirun -np 4 --mca btl_tcp_if_include ib0,ib1 ./test_pzgemm
echo "=== test_pzgemm exit: $? ==="

echo "============ 2. benchmark_pgemm (4 procs) ============"
mpirun -np 4 --mca btl_tcp_if_include ib0,ib1 ./benchmark_pgemm 5000 10000 15000
echo "=== benchmark_pgemm exit: $? ==="
