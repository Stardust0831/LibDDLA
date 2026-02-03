#!/bin/bash
#SBATCH -p kshdnormal
##SBATCH --nodelist gpu007
#SBATCH -J install
##SBATCH -A xgren
#SBATCH --nodes=1
#SBATCH --gres=dcu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=./log_install
#SBATCH --error=./err_install

ulimit -s unlimited
ulimit -c unlimited

unset CPATH
module purge


# module load compiler/dtk/22.10.1
# module load compiler/dtk/23.10
module load compiler/dtk/25.04.2
export CPATH=$ROCM_PATH/include/rocrand:$CPATH
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.11.0/gcc-7.3.1
module load compiler/cmake/3.23.3

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

export LANGUAGE=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

export OMPI_CXX=$CXX
export OMPI_CC=$CC
export OMPI_FC=$FC
# echo 'ROCM_PATH：' $ROCM_PATH

echo Begin Time: `date`
### * * * Running the tasks * * * ###
BUILD_DIR=./build_test
INSTALL_DIR="${PWD}_install"
echo 'Build Dir:' $BUILD_DIR
echo 'Install Dir:' $INSTALL_DIR
echo "任务运行节点列表: ${SLURM_NODELIST}"
rm -rf ${BUILD_DIR}
rm -rf ${INSTALL_DIR}
mkdir ${INSTALL_DIR}
cmake -B $BUILD_DIR -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
        -DROCM_PATH=$ROCM_PATH \
        -DMPI_CXX_COMPILER=mpicxx \
        -DENABLE_HIP=ON \
        -DCMAKE_PREFIX_PATH=$ROCM_PATH \
        -DCMAKE_CXX_COMPILER=hipcc \
        # -DCMAKE_HIP_COMPILER_ROCM_ROOT=$ROCM_PATH \
        # -DCMAKE_HIP_COMPILER=hipcc \
        # -DCMAKE_Fortran_COMPILER=gfortran \
        
        # -DBUILD_TESTS=ON \

cmake --build $BUILD_DIR -j `nproc` 

cmake --install $BUILD_DIR --prefix $INSTALL_DIR

# cd ${BUILD_DIR}
# make test
echo End Time: `date`
