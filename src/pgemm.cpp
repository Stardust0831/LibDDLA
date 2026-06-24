#include <ddla/ddla.h>
#include <cassert>
#include <ddla/ddla_connector.h>
#include <ddla/ddla_stream.h>
#include <vector>
#include <ddla/transport_block.h>
#include <ddla/ptran.h>
#include <ddla/ddla_comm.h>
#include <ddla/scal.h>
#include <ddla/gemm.h>

namespace ddla{

template <typename T>
void pgemm(
    const char& transa, const char& transb,
    const int& m, const int& n, const int& k,
    const T& alpha,
    const T* d_A, const DdlaDesc& array_descA,
    const T* d_B, const DdlaDesc& array_descB,
    const T& beta,
    T* d_C, const DdlaDesc& array_descC
)
{
    DdlaHandle_t ddla_handle = array_descA.ddla_handle();

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

    #ifdef DDLA_USE_CCL
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

    const int m_loc_C = array_descC.m_loc();
    const int n_loc_C = array_descC.n_loc();

    deviceStream_t stream = ddla_handle->stream;
    deviceStream_t stream_data = ddla_handle->stream_data;
    deblasHandle_t blasH = ddla_handle->blasH;

    // Pre-transpose (and conjugate for 'C') operands so that the SUMMA loop
    // always sees non-transposed data.  This makes rectangular process grids
    // work because transport_block only has to copy panels.
    const T* d_A_use = d_A;
    const T* d_B_use = d_B;
    const DdlaDesc* descA_use = &array_descA;
    const DdlaDesc* descB_use = &array_descB;

    T* d_AT = nullptr;
    T* d_BT = nullptr;

    if(transa != 'N'){
        bool conjA = (transa == 'C');
        DdlaDesc* descAT = new DdlaDesc(ddla_handle);
        descAT->init(array_descA.n(), array_descA.m(),
                     array_descA.nb(), array_descA.mb(),
                     array_descA.icsrc(), array_descA.irsrc());
        DEVICE_CHECK(deviceMalloc(&d_AT,
            sizeof(T) * descAT->m_loc() * descAT->n_loc()));
        ptran(d_A, array_descA, d_AT, *descAT, conjA);
        DEVICE_CHECK(deviceDeviceSynchronize());
        descA_use = descAT;
        d_A_use = d_AT;
    }

    if(transb != 'N'){
        bool conjB = (transb == 'C');
        DdlaDesc* descBT = new DdlaDesc(ddla_handle);
        descBT->init(array_descB.n(), array_descB.m(),
                     array_descB.nb(), array_descB.mb(),
                     array_descB.icsrc(), array_descB.irsrc());
        DEVICE_CHECK(deviceMalloc(&d_BT,
            sizeof(T) * descBT->m_loc() * descBT->n_loc()));
        ptran(d_B, array_descB, d_BT, *descBT, conjB);
        DEVICE_CHECK(deviceDeviceSynchronize());
        descB_use = descBT;
        d_B_use = d_BT;
    }

    // Physical transpose/conjugate means the effective gemm operation is 'N'.
    deblasOperation_t opA = DEBLAS_OP_N;
    deblasOperation_t opB = DEBLAS_OP_N;

    BLAS_CHECK(deblasScal(blasH, m_loc_C*n_loc_C, beta, d_C, 1));

    const int buffer_max = 2;
    T *d_A_temp[buffer_max],*d_B_temp[buffer_max];
    int nb;
    if(transa == 'N')
        nb = array_descA.nb();
    else
        nb = array_descA.mb();

    int count_a = m_loc_C * nb;
    int count_b = nb * n_loc_C;
    #ifdef DDLA_USE_GPU_CPU_TUNNEL
    std::vector<T> h_temp(std::max(count_a, count_b));
    #endif
    for(int i=0;i<buffer_max;i++){
        DEVICE_CHECK(deviceMallocAsync(&d_A_temp[i], sizeof(T) * count_a, stream_data));
        DEVICE_CHECK(deviceMallocAsync(&d_B_temp[i], sizeof(T) * count_b, stream_data));
    }

    int temp_buffer = 0;
    int k_s = 0 , kb;
    auto get_data = [&](int k_s)
    {
        kb = std::min(nb, k - k_s);
        if(kb<=0) return;

        // A panel: rows [0, m), cols [k_s, k_s+kb)
        transport_block(
            'C', 'N', m, kb,
            d_A_use, 0, k_s, *descA_use,
            d_A_temp[temp_buffer]
        );

        // B panel: rows [k_s, k_s+kb), cols [0, n)
        transport_block(
            'R', 'N', kb, n,
            d_B_use, k_s, 0, *descB_use,
            d_B_temp[temp_buffer]
        );
    };
    get_data(k_s);
    for(k_s=0; k_s<k; k_s+=nb){
        DEVICE_CHECK(deviceStreamSynchronize(stream_data));
        DEVICE_CHECK(deviceStreamSynchronize(stream));
        BLAS_CHECK(deblasGemm(
            blasH, opA, opB,
            m_loc_C, n_loc_C, kb,
            alpha,
            d_A_temp[temp_buffer], m_loc_C,
            d_B_temp[temp_buffer], kb,
            1.0,
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

    if(d_AT) DEVICE_CHECK(deviceFree(d_AT));
    if(d_BT) DEVICE_CHECK(deviceFree(d_BT));

    if(transa != 'N') delete descA_use;
    if(transb != 'N') delete descB_use;

    return;
}

template void pgemm<float>(
    const char& transa, const char& transb,
    const int& m, const int& n, const int& k,
    const float& alpha,
    const float* d_A, const DdlaDesc& array_descA,
    const float* d_B, const DdlaDesc& array_descB,
    const float& beta,
    float* d_C, const DdlaDesc& array_descC
);

template void pgemm<double>(
    const char& transa, const char& transb,
    const int& m, const int& n, const int& k,
    const double& alpha,
    const double* d_A, const DdlaDesc& array_descA,
    const double* d_B, const DdlaDesc& array_descB,
    const double& beta,
    double* d_C, const DdlaDesc& array_descC
);

template void pgemm<std::complex<float>>(
    const char& transa, const char& transb,
    const int& m, const int& n, const int& k,
    const std::complex<float>& alpha,
    const std::complex<float>* d_A, const DdlaDesc& array_descA,
    const std::complex<float>* d_B, const DdlaDesc& array_descB,
    const std::complex<float>& beta,
    std::complex<float>* d_C, const DdlaDesc& array_descC
);

template void pgemm<std::complex<double>>(
    const char& transa, const char& transb,
    const int& m, const int& n, const int& k,
    const std::complex<double>& alpha,
    const std::complex<double>* d_A, const DdlaDesc& array_descA,
    const std::complex<double>* d_B, const DdlaDesc& array_descB,
    const std::complex<double>& beta,
    std::complex<double>* d_C, const DdlaDesc& array_descC
);

} // namespace ddla
