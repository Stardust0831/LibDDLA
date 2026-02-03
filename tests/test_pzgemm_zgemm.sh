#!/bin/bash
#SBATCH -p kshdnormal02
##SBATCH --nodelist f17r1n19
#SBATCH --exclude b17r3n04
#SBATCH -J pzgemm
##SBATCH -A xgren
#SBATCH --nodes=1
#SBATCH --gres=dcu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=./log_pzgemm_zgemm
#SBATCH --error=./err_pzgemm_zgemm

ulimit -s unlimited
ulimit -c unlimited


unset CPATH

module purge

# module load compiler/dtk/22.10.1
module load compiler/dtk/25.04.2
# module load compiler/rocm/dtk-23.10
export LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/lib64/:$LIBRARY_PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/lib64/:$LD_LIBRARY_PATH
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.11.0/gcc-7.3.1

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
export LANGUAGE=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

echo Begin Time: `date`
### * * * Running the tasks * * * ###

FILENAME=test_pzgemm_zgemm

rm ${FILENAME}
# g++ -lmpi -g -O2 -lcudart -lddla -fopenmp -lnccl -lcublas -lcusolver -lcurand  ${FILENAME}.cpp -o ${FILENAME} -std=c++11 -DENABLE_CUDA
g++ -gdwarf-4 -lmpi -g -O2 -lamdhip64 -lgalaxyhip -lddla -fopenmp -lrccl -lhipblas -lhipsolver -lhiprand  ${FILENAME}.cpp -o ${FILENAME} -std=c++11 -DENABLE_HIP -D__HIP_PLATFORM_AMD__
np=$((SLURM_NTASKS_PER_NODE * SLURM_NNODES))
echo "np: $np"
# ldd ./${FILENAME}
# mpirun -n $np ./${FILENAME} --mca btl ^openib
mpirun -n $np ./${FILENAME}
echo End Time: `date`
