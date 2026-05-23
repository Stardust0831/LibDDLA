#!/bin/bash
#SBATCH -p v100g32
##SBATCH --nodelist gpu005
#SBATCH -J test
##SBATCH -A xgren
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=3
#SBATCH --output=../../log_test
#SBATCH --error=../../err_test


module load gcc/11.3.0

module load openmpi/4.1.8-cuda
module load cmake/3.25.3

source /data/home/renxg/app/nvhpc/setup_nvhpc

cd ..
LibDDLA_PATH="${PWD}_install"
cd tests
export CPATH=$LibDDLA_PATH/include:$CPATH
export LIBRARY_PATH=$LibDDLA_PATH/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$LibDDLA_PATH/lib:$LD_LIBRARY_PATH


echo "========================="
echo 'LD_LIBRARY_PATH:' $LD_LIBRARY_PATH
echo "========================="
echo 'PATH:' $PATH
echo "========================="
echo 'CPATH:' $CPATH
echo "========================="
echo 'C_INCLUDE_PATH:' $C_INCLUDE_PATH
echo "========================="
echo 'LIBRARY_PATH:' $LIBRARY_PATH
echo "========================="
echo 'CPLUS_INCLUDE_PATH:' $CPLUS_INCLUDE_PATH
echo "========================="

echo "任务运行节点列表: ${SLURM_NODELIST}"


echo Begin Time: `date`
### * * * Running the tasks * * * ###
which mpicxx

# FILENAME=test_sv_gemm
# FILENAME=test_aware
# FILENAME=test_pgeadd
# FILENAME=test_potrf_solvermp
FILENAME=test_potrf_potrs

# nvidia-smi

rm ../../${FILENAME}
# mpicxx -g -O2 -lcudart -lddla -fopenmp -lnccl -lcublas -lcusolver -lcurand  ${FILENAME}.cpp -o ${FILENAME} -std=c++11 -DENABLE_CUDA -DENABLE_CCL

mpicxx -g -O2 -lcudart -lddla -fopenmp -lcublas -lcusolver -lcurand -lcal -lcusolverMp ${FILENAME}.cpp -o ../../${FILENAME} -std=c++17 -DENABLE_CUDA -DENABLE_CCL
np=$((SLURM_NTASKS_PER_NODE * SLURM_NNODES))
echo "np: $np"
# ldd ./${FILENAME}
# mpirun -n $np --mca btl ^openib ./${FILENAME} 
# export OMPI_MCA_btl_vader_single_copy_mechanism=none
# export OMPI_MCA_bcol_uma_enable=0
export OMPI_MCA_btl_openib_allow_ib=1
mpirun -n $np --mca btl_tcp_if_include ib0,ib1 ../../${FILENAME}
# export OMPI_MCA_btl_vader_single_copy_mechanism=none
# export OMPI_MCA_bcol_uma_enable=0
# mpirun -n $np ./${FILENAME}
echo End Time: `date`
