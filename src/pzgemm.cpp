#include <ddla.h>
#include <cassert>
#include <ddla_connector.h>
#include <ddla_utils.h>
#include <ddla_stream.h>
namespace DDLA{

void pzgemm(
    const char& transa, const char& transb,
    const int& m, const int& n, const int& k,
    const std::complex<double>& alpha,
    const std::complex<double>* d_A, const DDLA::DdlaDesc& array_descA,
    const std::complex<double>* d_B, const DDLA::DdlaDesc& array_descB,
    const std::complex<double>& beta,
    std::complex<double>* d_C, const DDLA::DdlaDesc& array_descC
)
{
    DdlaHandle_t ddla_handle = array_descA.ddla_handle();

    assert(transa == 'N' && transb == 'N');
    assert(array_descA.nb() == array_descB.mb());
    assert(array_descA.mb() == array_descC.nb());
    assert(array_descB.nb() == array_descC.mb());

    MPI_Comm row_comm = ddla_handle->row_comm;
    MPI_Comm col_comm = ddla_handle->col_comm;
    ncclComm_t row_nccl_comm = ddla_handle->nccl_row_comm;
    ncclComm_t col_nccl_comm = ddla_handle->nccl_col_comm;

    int nprows = array_descC.nprows();
    int npcols = array_descC.npcols();
    int myprow = array_descC.myprow();
    int mypcol = array_descC.mypcol();

    // int mbA = array_descA.mb();
    // int nbB = array_descB.nb();
    int m_loc_A = array_descA.m_loc();
    int n_loc_B = array_descB.n_loc();
    int nb = array_descA.nb();
    int lldA = array_descA.lld();
    int lldB = array_descB.lld();

    const std::complex<double> one = {1.0,0.0};

    const int num_blocks = 1; // the number of blocks to broadcast each time

    deviceStream_t stream = ddla_handle->stream;
    deviceStream_t stream_data = ddla_handle->stream_data;
    deblasHandle_t blasH = ddla_handle->blasH;

    BLAS_CHECK(deblasZscal(blasH, m_loc_A*n_loc_B, &beta, d_C, 1));

    const int buffer_max = 2;
    std::complex<double> *d_A_temp[buffer_max],*d_B_temp[buffer_max];
    for(int i=0;i<buffer_max;i++){
        DEVICE_CHECK(deviceMallocAsync(&d_A_temp[i], sizeof(std::complex<double>)*m_loc_A*nb*num_blocks, stream_data));
        DEVICE_CHECK(deviceMallocAsync(&d_B_temp[i], sizeof(std::complex<double>)*nb*n_loc_B*num_blocks, stream_data));
    }

    int temp_buffer = 0;
    int k_s = 0 ,owner_col_A,owner_row_B,kb;
    auto data_trans = [&kb, &num_blocks,&nb,&k,&array_descA,&array_descB,
        &d_A,&d_B,&d_A_temp,&d_B_temp,&m_loc_A,&n_loc_B,&lldA,&lldB,
        &mypcol,&myprow,&row_nccl_comm,&col_nccl_comm,&stream_data,&temp_buffer](int k_s) 
    {
        kb = std::min(num_blocks*nb, k - k_s);

        int owner_col_A = DDLA::indxg2p(k_s, nb, array_descA.icsrc(), array_descA.npcols());
        int owner_row_B = DDLA::indxg2p(k_s, nb, array_descB.irsrc(), array_descB.nprows());
        if(kb<=0) return;
        // broadcast A block column
        if(mypcol == owner_col_A){
            DEVICE_CHECK(deviceMemcpy2DAsync(
                d_A_temp[temp_buffer], m_loc_A * sizeof(std::complex<double>),
                d_A + array_descA.indx_g2l_c(k_s) * lldA, lldA * sizeof(std::complex<double>),
                m_loc_A * sizeof(std::complex<double>), kb,
                deviceMemcpyDeviceToDevice, stream_data
            ));
        }
        CCL_CHECK(ncclBroadcast(
            d_A_temp[temp_buffer], d_A_temp[temp_buffer],
            m_loc_A * kb * 2, ncclFloat64,
            owner_col_A, row_nccl_comm, stream_data
        ));
        // broadcast B block row
        if(myprow == owner_row_B){
            DEVICE_CHECK(deviceMemcpy2DAsync(
                d_B_temp[temp_buffer], kb * sizeof(std::complex<double>),
                d_B + array_descB.indx_g2l_r(k_s), lldB * sizeof(std::complex<double>),
                kb * sizeof(std::complex<double>), n_loc_B,
                deviceMemcpyDeviceToDevice, stream_data
            ));
        }
        CCL_CHECK(ncclBroadcast(
            d_B_temp[temp_buffer], d_B_temp[temp_buffer],
            kb * n_loc_B * 2, ncclFloat64,
            owner_row_B, col_nccl_comm, stream_data
        ));
    };
    data_trans(k_s);
    for(k_s=0; k_s<k; k_s+=num_blocks*nb){
        DEVICE_CHECK(deviceStreamSynchronize(stream_data));
        DEVICE_CHECK(deviceStreamSynchronize(stream));
        BLAS_CHECK(deblasZgemm(
            blasH, DEBLAS_OP_N, DEBLAS_OP_N,
            m_loc_A, n_loc_B, kb,
            &alpha,
            d_A_temp[temp_buffer], m_loc_A,
            d_B_temp[temp_buffer], kb,
            &one,
            d_C, m_loc_A
        ));
        temp_buffer = (temp_buffer + 1) % buffer_max;
        data_trans(k_s + num_blocks*nb);
    }
    DEVICE_CHECK(deviceStreamSynchronize(stream));
    DEVICE_CHECK(deviceStreamSynchronize(stream_data));
    for(int i=0;i<buffer_max;i++){
        DEVICE_CHECK(deviceFreeAsync(d_A_temp[i], stream_data));
        DEVICE_CHECK(deviceFreeAsync(d_B_temp[i], stream_data));
    }
    return;
}

}