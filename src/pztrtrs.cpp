#include <ddla.h>
#include <cassert>
#include <ddla_connector.h>
#include <ddla_utils.h>
#include <ddla_stream.h>
namespace DDLA{

void pztrtrs(
    const char& uplo, const char& diag, const int& m, const int& n,
    std::complex<double>* d_A, const DDLA::DdlaDesc& array_descA,
    std::complex<double>* d_B, const DDLA::DdlaDesc& array_descB
)
{
    DdlaHandle_t ddla_handle = array_descA.ddla_handle();
    
    MPI_Comm comm_group = ddla_handle->comm;
    int rank=ddla_handle->myid;
    int size=ddla_handle->nprocs;
    
    assert(array_descA.m() == array_descA.n());
    assert(array_descA.mb()==array_descA.nb());
    assert(array_descA.mb()==array_descB.mb());
    assert(array_descA.m() == array_descB.m());
    assert(uplo=='L'||uplo=='U');
    assert(diag=='U'||diag=='N');
    int nb = array_descA.mb();
    int lldA = array_descA.lld();
    int lldB = array_descB.lld();

    int nprows = array_descA.nprows();
    int npcols = array_descA.npcols();
    // printf("nprows:%d, npcols:%d\n",nprows,npcols);

    // 初始化 NCCL  
    ncclComm_t comm=ddla_handle->nccl_comm;
    ncclComm_t row_comm=ddla_handle->nccl_row_comm;
    ncclComm_t col_comm=ddla_handle->nccl_col_comm;

    deviceStream_t stream=ddla_handle->stream;
    deblasHandle_t blasH=ddla_handle->blasH;
    desolverHandle_t solverH=ddla_handle->solverH;

    #ifdef ENABLE_CUDA
    cublasFillMode_t uplo_device = (uplo == 'U') ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
    cublasDiagType_t diag_device = (diag == 'U') ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT;
    cublasOperation_t trans_device = CUBLAS_OP_N;
    cublasSideMode_t side_device = CUBLAS_SIDE_LEFT;
    #endif
    #ifdef ENABLE_HIP
    hipblasFillMode_t uplo_device = (uplo == 'U') ? HIPBLAS_FILL_MODE_UPPER : HIPBLAS_FILL_MODE_LOWER;
    hipblasDiagType_t diag_device = (diag == 'U') ? HIPBLAS_DIAG_UNIT : HIPBLAS_DIAG_NON_UNIT;
    hipblasOperation_t trans_device = HIPBLAS_OP_N;
    hipblasSideMode_t side_device = HIPBLAS_SIDE_LEFT;
    #endif
    // double start_time = MPI_Wtime();
    std::complex<double>* d_block_diag,*d_block_A,*d_block_B;
    DEVICE_CHECK(deviceMallocAsync(&d_block_diag, nb * nb * sizeof(std::complex<double>), stream));
    DEVICE_CHECK(deviceMallocAsync(&d_block_B, nb * array_descB.n_loc() * sizeof(std::complex<double>), stream));
    DEVICE_CHECK(deviceMallocAsync(&d_block_A, array_descA.m_loc() * nb * sizeof(std::complex<double>), stream));
    int mm_row_start,mm_row_step;
    // int mm_col_start=0;
    int n_s_start,n_s_end,n_s_step;
    if(uplo == 'U'){
        int m_loc = array_descA.m_loc();
        n_s_start = m%nb==0?m-nb:m - m%nb;
        n_s_end = -nb;
        n_s_step = -nb;
        mm_row_start = m_loc;
    }else{
        n_s_start = 0;
        n_s_end = m%nb==0?m:m - m%nb + nb;
        n_s_step = nb;
        mm_row_start = 0;
    }
    int l_row_s, l_col_s;
    int owner_row, owner_col;
    
    int lld_block_A;
    int64_t A_offset, B_offset;
    for(int n_s=n_s_start;n_s!=n_s_end;n_s+=n_s_step){
        int nb_real = std::min(nb, m - n_s);
        // printf("n_s=%d, nb_real=%d\n",n_s, nb_real);
        mm_row_step = (uplo=='L')?nb_real:-nb_real;

        l_row_s = array_descA.indx_g2l_r(n_s);
        l_col_s = array_descA.indx_g2l_c(n_s);

        owner_row = DDLA::indxg2p(n_s, nb, array_descA.irsrc(), nprows);
        owner_col = DDLA::indxg2p(n_s, nb, array_descA.icsrc(), npcols);
        // printf("owner_row:%d,owner_col:%d\n",owner_row,owner_col);

        const std::complex<double> one = {1.0, 0.0};
        const std::complex<double> minus_one = {-1.0, 0.0};
        if(l_row_s>=0&&l_col_s>=0){
            DEVICE_CHECK(deviceMemcpy2DAsync(
                d_block_diag, nb_real * sizeof(std::complex<double>),
                d_A + l_row_s + l_col_s * lldA, lldA * sizeof(std::complex<double>),
                nb_real * sizeof(std::complex<double>), nb_real,
                deviceMemcpyDeviceToDevice,stream
            ));
        }
        DEVICE_CHECK(deviceStreamSynchronize(stream));
        // 广播当前块行
        if(l_row_s>=0){
            CCL_CHECK(ncclBroadcast(d_block_diag,d_block_diag,nb_real * nb_real * 2,ncclFloat64,owner_col,row_comm,stream));
            BLAS_CHECK(deblasZtrsm(
                blasH, side_device, uplo_device, trans_device, diag_device,
                nb_real, array_descB.n_loc(), &one,
                d_block_diag, nb_real,
                d_B+l_row_s, lldB
            ));
            DEVICE_CHECK(deviceMemcpy2DAsync(
                d_block_B,nb_real*sizeof(std::complex<double>),
                (d_B+l_row_s),lldB*sizeof(std::complex<double>),
                nb_real*sizeof(std::complex<double>),array_descB.n_loc(),
                deviceMemcpyDeviceToDevice,stream
            ));
            mm_row_start+=mm_row_step;
        }
        DEVICE_CHECK(deviceStreamSynchronize(stream));
        CCL_CHECK(ncclBroadcast(d_block_B,d_block_B,nb_real * array_descB.n_loc() * 2,ncclFloat64,owner_row,col_comm,stream));
        lld_block_A = uplo=='L'?array_descA.m_loc()-mm_row_start:mm_row_start;
        A_offset = uplo=='L'?mm_row_start+l_col_s*lldA:l_col_s*lldA;
        
        if(l_col_s>=0&&lld_block_A>0){
            // printf("myid:%d, n_s:%d, lld_block_A:%d, A_offset:%ld\n",mpi_comm_global_h.myid,n_s,lld_block_A,A_offset);
            DEVICE_CHECK(deviceMemcpy2DAsync(
                d_block_A,lld_block_A*sizeof(std::complex<double>),
                d_A+A_offset,lldA*sizeof(std::complex<double>),
                lld_block_A*sizeof(std::complex<double>),nb_real,
                deviceMemcpyDeviceToDevice,stream
            ));
        }
        DEVICE_CHECK(deviceStreamSynchronize(stream));
        B_offset = uplo=='L'?mm_row_start:0;
        if(lld_block_A>0){
            CCL_CHECK(ncclBroadcast(d_block_A,d_block_A,lld_block_A * nb_real * 2,ncclFloat64,owner_col,row_comm,stream));
            BLAS_CHECK(deblasZgemm(
                blasH, trans_device, trans_device,lld_block_A,array_descB.n_loc(),nb_real,
                &minus_one,
                d_block_A,lld_block_A,
                d_block_B,nb_real,
                &one,
                d_B+B_offset, lldB
            ));
        }
        DEVICE_CHECK(deviceStreamSynchronize(stream));
    }
    DEVICE_CHECK(deviceFreeAsync(d_block_A,stream));
    DEVICE_CHECK(deviceFreeAsync(d_block_B,stream));
    DEVICE_CHECK(deviceFreeAsync(d_block_diag,stream));
    DEVICE_CHECK(deviceStreamSynchronize(stream));
}

}