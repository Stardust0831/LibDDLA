#include <ddla.h>
#include <cassert>
#include <ddla_connector.h>
#include <ddla_utils.h>
#include <ddla_stream.h>
#include <vector>
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

    if(transa != 'N' || transb != 'N')
    {
        if(array_descA.nprows() != array_descA.npcols()){
            throw std::runtime_error("the trans multiplication is now implemented for mxn(m!=n) mpi grid");
        }
    }
    {
        int mbA,kbA,kbB,nbB,mbC,nbC;
        mbC = array_descC.mb();
        nbC = array_descC.nb();
        if(transa == 'N'){
            mbA = array_descA.mb();
            kbA = array_descA.nb();
        }else{
            mbA = array_descA.nb();
            kbA = array_descA.mb();
        }

        if(transb == 'N'){
            kbB = array_descB.mb();
            nbB = array_descB.nb();
        }else{
            kbB = array_descB.nb();
            nbB = array_descB.mb();
        }
        assert(mbA == mbC);
        assert(kbA == kbB);
        assert(nbB == nbC);
    }

    #ifdef ENABLE_CCL
    ncclComm_t row_nccl_comm = ddla_handle->nccl_row_comm;
    ncclComm_t col_nccl_comm = ddla_handle->nccl_col_comm;
    #else
    MPI_Comm row_nccl_comm = ddla_handle->row_comm;
    MPI_Comm col_nccl_comm = ddla_handle->col_comm;
    #endif

    int nprows = array_descC.nprows();
    int npcols = array_descC.npcols();
    int myprow = array_descC.myprow();
    int mypcol = array_descC.mypcol();

    int m_loc_A = array_descA.m_loc();
    int n_loc_A = array_descA.n_loc();
    int m_loc_B = array_descB.m_loc();
    int n_loc_B = array_descB.n_loc();
    int m_loc_C = array_descC.m_loc();
    int n_loc_C = array_descC.n_loc();

    int nb;
    if(transa=='N')
        nb = array_descA.nb();
    else
        nb = array_descA.mb();
    int lldA = array_descA.lld();
    int lldB = array_descB.lld();

    const std::complex<double> one = {1.0,0.0};

    deviceStream_t stream = ddla_handle->stream;
    deviceStream_t stream_data = ddla_handle->stream_data;
    deblasHandle_t blasH = ddla_handle->blasH;

    deblasOperation_t opA = (transa == 'N') ? DEBLAS_OP_N :
                            (transa == 'T') ? DEBLAS_OP_T : DEBLAS_OP_C;
    deblasOperation_t opB = (transb == 'N') ? DEBLAS_OP_N :
                            (transb == 'T') ? DEBLAS_OP_T : DEBLAS_OP_C;

    BLAS_CHECK(deblasZscal(blasH, m_loc_C*n_loc_C, &beta, d_C, 1));

    const int buffer_max = 2;
    std::complex<double> *d_A_temp[buffer_max],*d_B_temp[buffer_max];
    for(int i=0;i<buffer_max;i++){
        DEVICE_CHECK(deviceMallocAsync(&d_A_temp[i], sizeof(std::complex<double>) * (transa=='N'?m_loc_C:(std::max(n_loc_A, m_loc_C))) * nb, stream_data));
        DEVICE_CHECK(deviceMallocAsync(&d_B_temp[i], sizeof(std::complex<double>) * nb * (transb=='N'?n_loc_C:(std::max(m_loc_B, n_loc_C))), stream_data));
        DEVICE_CHECK(deviceMemsetAsync(d_A_temp[i], 0, sizeof(std::complex<double>) * (transa=='N'?m_loc_C:(std::max(n_loc_A, m_loc_C))) * nb, stream_data));
        DEVICE_CHECK(deviceMemsetAsync(d_B_temp[i], 0, sizeof(std::complex<double>) * nb * (transb=='N'?n_loc_C:(std::max(m_loc_B, n_loc_C))), stream_data));
    }

    int temp_buffer = 0;
    int k_s = 0, kb;
    auto get_data = [&kb,&nb,&k,&array_descA,&array_descB,
        &d_A,&d_B,&d_A_temp,&d_B_temp,&m_loc_A,&n_loc_A,&m_loc_B,&n_loc_B,&lldA,&lldB,
        &mypcol,&myprow,&row_nccl_comm,&col_nccl_comm,&stream_data,&temp_buffer,&transa,&transb,&m_loc_C,&n_loc_C, &ddla_handle](int k_s) 
    {
        kb = std::min(nb, k - k_s);
        if(kb<=0) return;
        int src_A;
        
        if(transa != 'N'){
            int owner_row_A = DDLA::indxg2p(k_s, nb, array_descA.irsrc(), array_descA.nprows());
            if(myprow == owner_row_A){
                DEVICE_CHECK(deviceMemcpy2DAsync(
                    d_A_temp[temp_buffer], kb * sizeof(std::complex<double>),
                    d_A + array_descA.indx_g2l_r(k_s), lldA * sizeof(std::complex<double>),
                    kb * sizeof(std::complex<double>), n_loc_A,
                    deviceMemcpyDeviceToDevice, stream_data
                ));
                if(myprow != mypcol){
                    // printf("before ccl send\n");
                    CCL_CHECK(cclSend(d_A_temp[temp_buffer], kb * n_loc_A, mypcol, col_nccl_comm, stream_data));
                    // printf("after ccl send\n");
                }
            }else{
                if(myprow == mypcol){
                    // printf("before ccl recv\n");
                    CCL_CHECK(cclRecv(d_A_temp[temp_buffer], kb * n_loc_A, owner_row_A, col_nccl_comm, stream_data));
                    // printf("after ccl recv\n");
                }
            }
            src_A = myprow;
        }else{
            src_A = DDLA::indxg2p(k_s, nb, array_descA.icsrc(), array_descA.npcols());
            if(mypcol == src_A){
                DEVICE_CHECK(deviceMemcpy2DAsync(
                    d_A_temp[temp_buffer], m_loc_A * sizeof(std::complex<double>),
                    d_A + array_descA.indx_g2l_c(k_s) * lldA, lldA * sizeof(std::complex<double>),
                    m_loc_A * sizeof(std::complex<double>), kb,
                    deviceMemcpyDeviceToDevice, stream_data
                ));
            }
        }
        
        // printf("before bcast\n");
        // broadcast A block
        CCL_CHECK(cclBcast(d_A_temp[temp_buffer], m_loc_C * kb, src_A, row_nccl_comm, stream_data));
        // end communicate A
        // printf("after bcast\n");
        int src_B;
        // start communicate B
        if(transb != 'N'){
            int owner_col_B = DDLA::indxg2p(k_s, nb, array_descB.icsrc(), array_descB.npcols());
            if(mypcol == owner_col_B){
                DEVICE_CHECK(deviceMemcpy2DAsync(
                    d_B_temp[temp_buffer], m_loc_B * sizeof(std::complex<double>),
                    d_B + array_descB.indx_g2l_c(k_s) * lldB, lldB * sizeof(std::complex<double>),
                    m_loc_B * sizeof(std::complex<double>), kb,
                    deviceMemcpyDeviceToDevice, stream_data
                ));
                if(myprow != mypcol){
                    CCL_CHECK(cclSend(d_B_temp[temp_buffer], kb * m_loc_B, myprow, row_nccl_comm, stream_data));
                }
            }else{
                if(myprow == mypcol){
                    CCL_CHECK(cclRecv(d_B_temp[temp_buffer], kb * m_loc_B, owner_col_B, row_nccl_comm, stream_data));
                }
            }
            src_B = mypcol;
        }
        else{
            src_B = DDLA::indxg2p(k_s, nb, array_descB.irsrc(), array_descB.nprows());
            if(myprow == src_B){
                DEVICE_CHECK(deviceMemcpy2DAsync(
                    d_B_temp[temp_buffer], kb * sizeof(std::complex<double>),
                    d_B + array_descB.indx_g2l_r(k_s), lldB * sizeof(std::complex<double>),
                    kb * sizeof(std::complex<double>), n_loc_C,
                    deviceMemcpyDeviceToDevice, stream_data
                ));
            }
        }
        // broadcast B block
        CCL_CHECK(cclBcast(
            d_B_temp[temp_buffer], kb * n_loc_C, src_B, col_nccl_comm, stream_data
        ));
    };
    get_data(k_s);
    for(k_s=0; k_s<k; k_s+=nb){
        DEVICE_CHECK(deviceStreamSynchronize(stream_data));
        DEVICE_CHECK(deviceStreamSynchronize(stream));
        BLAS_CHECK(deblasZgemm(
            blasH, opA, opB,
            m_loc_C, n_loc_C, kb,
            &alpha,
            d_A_temp[temp_buffer], transa=='N'?m_loc_C:kb,
            d_B_temp[temp_buffer], transb=='N'?kb:n_loc_C,
            &one,
            d_C, m_loc_C
        ));
        temp_buffer = (temp_buffer + 1) % buffer_max;
        get_data(k_s + nb);
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