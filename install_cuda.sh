#!/bin/bash
#SBATCH -p gpu4090_128
##SBATCH --nodelist gpu005
#SBATCH -J trtri
#SBATCH -A xgren
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=./log_install
#SBATCH --error=./err_install

ulimit -s unlimited
ulimit -c unlimited

unset CPATH
module purge


source ~/app/gcc/250808/setup_gcc
source ~/abacus/251205/toolchain_amd/build/setup_openmpi_extern4
source ~/app/hpc/250711/Linux_x86_64/setup_nvhpc
source ~/abacus/250815/abacus-develop-LTSv3.10.0/toolchain/build/setup_cmake
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


export OMPI_CXX=$CXX
export OMPI_CC=$CC
export OMPI_FC=$FC


echo Begin Time: `date`
### * * * Running the tasks * * * ###
BUILD_DIR=./build_test
INSTALL_DIR=/data/home/hbchen/work/libddla_install
rm -rf ${BUILD_DIR}
rm -rf ${INSTALL_DIR}
mkdir ${INSTALL_DIR}
cmake -B $BUILD_DIR -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
        -DCMAKE_CXX_COMPILER=g++ \
        -DMPI_CXX_COMPILER=mpicxx \
        -DCMAKE_Fortran_COMPILER=gfortran \
        -DENABLE_CUDA=ON \
        # -DBUILD_TESTS=ON \

cmake --build $BUILD_DIR -j `nproc` 

cmake --install $BUILD_DIR --prefix $INSTALL_DIR

# cd ${BUILD_DIR}
# make test
echo End Time: `date`
