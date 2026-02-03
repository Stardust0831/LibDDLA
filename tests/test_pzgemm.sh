#!/bin/bash
#SBATCH -p gpu4090_128
##SBATCH --nodelist 
##SBATCH --exclude 
#SBATCH -J pzgemm
#SBATCH -A xgren
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=./log_pzgemm
#SBATCH --error=./err_pzgemm

ulimit -s unlimited
ulimit -c unlimited


unset CPATH

module purge

source ~/app/gcc/250808/setup_gcc
source ~/abacus/251205/toolchain_amd/build/setup_openmpi_extern4
source ~/app/hpc/250711/Linux_x86_64/setup_nvhpc
LibDDLA_PATH="${PWD}_install"
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

FILENAME=test_pzgemm

rm ${FILENAME}
g++ -lmpi -g -O2 -lcudart -lddla -fopenmp -lnccl -lcublas -lcusolver -lcurand  ${FILENAME}.cpp -o ${FILENAME} -std=c++11 -DENABLE_CUDA
# hipcc -gdwarf-4 -lmpi -g -O2 -lamdhip64 -lgalaxyhip -lddla -fopenmp -lrccl -lhipblas -lhipsolver -lhiprand  ${FILENAME}.cpp -o ${FILENAME} -std=c++11 -DENABLE_HIP
np=$((SLURM_NTASKS_PER_NODE * SLURM_NNODES))
echo "np: $np"
# ldd ./${FILENAME}
mpirun -n $np ./${FILENAME} --mca btl ^openib
# mpirun -n $np ./${FILENAME}
echo End Time: `date`
