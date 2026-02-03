#include <ddla.h>
#include <cassert>
#include <ddla_connector.h>
#include <ddla_utils.h>
#include <ddla_stream.h>
namespace DDLA{

void pzlapiv(
    const char& direc, const char& rowcol, const char& pivroc,
    const int& m, const int& n,
    std::complex<double>* d_A,const DDLA::DdlaDesc& array_descA,
    const int* ipiv, const DDLA::DdlaDesc& array_descIP,
    int* iwork
)
{
    DdlaHandle_t ddla_handle = array_descA.ddla_handle();
    
    assert(direc=='F');
    assert(rowcol=='R');
    assert(pivroc=='C');
    assert(m<=array_descA.m());
    assert(n==array_descA.n());

    int nprows = array_descA.nprows();
    int myprow = array_descA.myprow();

    int mb = array_descA.mb();
    int lldA = array_descA.lld();

    std::complex<double>*temp_A_target;
    DEVICE_CHECK(deviceMallocAsync(&temp_A_target, sizeof(std::complex<double>)*array_descA.n_loc(), ddla_handle->stream));

    deviceStream_t stream = ddla_handle->stream;
    deblasHandle_t blasH = ddla_handle->blasH;
    

    // 初始化 NCCL
    MPI_Comm col_comm = ddla_handle->col_comm;
    ncclComm_t col_nccl_comm=ddla_handle->nccl_col_comm;

    int i_loc;
    int owner_row;
    int target_row;
    int target_i_global,target_i_loc;
    for(int i=0;i<m;i++){
        i_loc = array_descA.indx_g2l_r(i);
        owner_row = DDLA::indxg2p(i, mb, array_descA.irsrc(), nprows);
        if(i_loc>=0){
            target_i_global = ipiv[i_loc] - 1;
        }
        MPI_Bcast(&target_i_global, 1, MPI_INT, owner_row, col_comm);
        if(target_i_global==i)
            continue;
        target_row = DDLA::indxg2p(target_i_global,array_descIP.mb(),array_descA.irsrc(),nprows);
        target_i_loc = array_descIP.indx_g2l_r(target_i_global);
        // if(i_loc>=0)
        //     printf("myid:%d, i:%d, i_loc:%d, target_i_global:%d, target_i_loc:%d, owner_row:%d, target_row:%d\n",mpi_comm_global_h.myid,i,i_loc,target_i_global,target_i_loc,owner_row,target_row);
        // else if(target_i_loc>=0)
        //     printf("myid:%d, i:%d, i_loc:%d, target_i_global:%d, target_i_loc:%d, owner_row:%d, target_row:%d\n",mpi_comm_global_h.myid,i,i_loc,target_i_global,target_i_loc,owner_row,target_row);
        if(target_row==owner_row){
            if(myprow==owner_row)
                BLAS_CHECK(deblasZswap(blasH,array_descA.n_loc(),d_A+i_loc,lldA,d_A+target_i_loc,lldA));
                
        }else{
            if(myprow==target_row){
                DEVICE_CHECK(deviceMemcpy2DAsync(
                    temp_A_target,1*sizeof(std::complex<double>),
                    d_A+target_i_loc, lldA*sizeof(std::complex<double>),
                    1*sizeof(std::complex<double>), array_descA.n_loc(),
                    deviceMemcpyDeviceToDevice, stream
                ));
                CCL_CHECK(ncclSend(temp_A_target, array_descA.n_loc()*2, ncclFloat64, owner_row, col_nccl_comm, stream));
            }else if(myprow==owner_row){
                CCL_CHECK(ncclRecv(temp_A_target, array_descA.n_loc()*2, ncclFloat64, target_row, col_nccl_comm, stream));
                BLAS_CHECK(deblasZswap(blasH,array_descA.n_loc(),d_A+i_loc,lldA,temp_A_target,1));
            }
            if(myprow == owner_row){
                CCL_CHECK(ncclSend(temp_A_target, array_descA.n_loc()*2, ncclFloat64, target_row, col_nccl_comm, stream));
            }else if(myprow==target_row){
                CCL_CHECK(ncclRecv(temp_A_target, array_descA.n_loc()*2, ncclFloat64, owner_row, col_nccl_comm, stream));
                BLAS_CHECK(deblasZswap(blasH,array_descA.n_loc(),d_A+target_i_loc,lldA,temp_A_target,1));
            }
        }
    }

    DEVICE_CHECK(deviceFreeAsync(temp_A_target, stream));
    DEVICE_CHECK(deviceStreamSynchronize(stream));

}

}