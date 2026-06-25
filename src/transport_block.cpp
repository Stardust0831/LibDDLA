#include <ddla/ddla.h>
#include <cassert>
#include <ddla/ddla_connector.h>
#include <ddla/ddla_stream.h>
#include <ddla/trsm.h>
#include <ddla/ddla_comm.h>
#include <ddla/geam.h>
#include <ddla/scal.h>
#ifdef DDLA_USE_GPU_CPU_TUNNEL
#include <vector>
#endif

namespace ddla{

template <typename T>
void transport_block(
    const char& sData, const char& trans,
    const int& m, const int& n,
    const T* d_A, const int& ia, const int& ja,
    const DdlaDesc& array_descA,
    const DdlaDesc& target_desc,
    T* d_block_A
)
{
    if(m==0 || n==0)
        return;
    DdlaHandle_t ddla_handle = array_descA.ddla_handle();

    assert(sData == 'C' || sData == 'R');
    assert(trans == 'N' || trans == 'T' || trans == 'C');
    #ifdef DDLA_USE_CCL
    ncclComm_t row_nccl_comm = ddla_handle->nccl_row_comm;
    ncclComm_t col_nccl_comm = ddla_handle->nccl_col_comm;
    #else
    MPI_Comm row_nccl_comm = ddla_handle->row_comm;
    MPI_Comm col_nccl_comm = ddla_handle->col_comm;
    #endif

    #ifdef DDLA_USE_GPU_CPU_TUNNEL
    std::vector<T> h_temp(array_descA.nb() * (std::max(array_descA.m_loc(), array_descA.n_loc())));
    #endif

    int lld = array_descA.lld();

    if(trans == 'N'){
        int i_loc = num_loc(ia, array_descA.mb(), array_descA.myprow(), array_descA.irsrc(), array_descA.nprows());
        int j_loc = num_loc(ja, array_descA.nb(), array_descA.mypcol(), array_descA.icsrc(), array_descA.npcols());

        int m_loc = num_loc(ia + m, array_descA.mb(), array_descA.myprow(), array_descA.irsrc(), array_descA.nprows());
        int n_loc = num_loc(ja + n, array_descA.nb(), array_descA.mypcol(), array_descA.icsrc(), array_descA.npcols());

        int owner_row = indxg2p(ia, array_descA.mb(), array_descA.irsrc(), array_descA.nprows());
        int owner_col = indxg2p(ja, array_descA.nb(), array_descA.icsrc(), array_descA.npcols());

        if(sData == 'R' && n_loc > j_loc){
            if(array_descA.myprow() == owner_row){
                DEVICE_CHECK(deviceMemcpy2DAsync(
                    d_block_A, m * sizeof(T),
                    d_A + i_loc + j_loc * lld, lld * sizeof(T),
                    m * sizeof(T), n_loc - j_loc,
                    deviceMemcpyDeviceToDevice, ddla_handle->stream
                ));
            }
            #ifdef DDLA_USE_GPU_CPU_TUNNEL
            MPI_CHECK(cclBcast(h_temp.data(), d_block_A, m * (n_loc - j_loc), owner_row, ddla_handle->col_comm, ddla_handle->stream));
            #else
            CCL_CHECK(cclBcast(d_block_A, m * (n_loc - j_loc), owner_row, col_nccl_comm, ddla_handle->stream));
            #endif
        }else if(sData == 'C' && m_loc > i_loc){
            if(array_descA.mypcol() == owner_col){
                DEVICE_CHECK(deviceMemcpy2DAsync(
                    d_block_A, (m_loc - i_loc) * sizeof(T),
                    d_A + i_loc + j_loc * lld, lld * sizeof(T),
                    (m_loc - i_loc) * sizeof(T), n,
                    deviceMemcpyDeviceToDevice, ddla_handle->stream
                ));
            }
            #ifdef DDLA_USE_GPU_CPU_TUNNEL
            MPI_CHECK(cclBcast(h_temp.data(), d_block_A, (m_loc - i_loc) * n, owner_col, ddla_handle->row_comm, ddla_handle->stream));
            #else
            CCL_CHECK(cclBcast(d_block_A, (m_loc - i_loc) * n, owner_col, row_nccl_comm, ddla_handle->stream));
            #endif
        }
    }else{
        // transposed cases: use the target descriptor to decide which rows/cols
        // of the transposed block are needed on this process.
        // This path relies on the ScaLAPACK compatibility between array_descA
        // and target_desc (e.g. for A^T, target_desc.mb() == array_descA.nb()
        // and target_desc.irsrc() == array_descA.icsrc()).
        T one = T(1);
        T zero = T(0);
        // Choose geam operation: OP_T for 'T', OP_C for 'C' (conjugate transpose).
        deblasOperation_t geam_op = (trans == 'C') ? DEBLAS_OP_C : DEBLAS_OP_T;
        if(sData == 'R'){
            // Requested block: original A columns [ia, ia+m), rows [ja, ja+n).
            // Output: m_target_rows x m buffer, column-major with lda = m.
            int owner_col = indxg2p(ia, array_descA.nb(), array_descA.icsrc(), array_descA.npcols());

            int local_col_start = num_loc(ia, array_descA.nb(), array_descA.mypcol(), array_descA.icsrc(), array_descA.npcols());
            int local_col_end   = num_loc(ia + m, array_descA.nb(), array_descA.mypcol(), array_descA.icsrc(), array_descA.npcols());
            int local_cols = local_col_end - local_col_start;

            int m_target_rows = target_desc.m_loc();

            if(array_descA.mypcol() == owner_col && local_cols > 0 && m_target_rows > 0){
                // source submatrix is m_target_rows x local_cols in A, transpose to local_cols x m_target_rows
                BLAS_CHECK(deblasGeam(
                    ddla_handle->blasH, geam_op, DEBLAS_OP_N,
                    local_cols, m_target_rows,
                    one,
                    d_A + local_col_start * lld, lld,
                    zero,
                    d_block_A, local_cols,
                    d_block_A, local_cols
                ));
            }
            if(m_target_rows > 0){
                #ifdef DDLA_USE_GPU_CPU_TUNNEL
                MPI_CHECK(cclBcast(h_temp.data(), d_block_A, m * m_target_rows, owner_col, ddla_handle->row_comm, ddla_handle->stream));
                #else
                CCL_CHECK(cclBcast(d_block_A, m * m_target_rows, owner_col, row_nccl_comm, ddla_handle->stream));
                #endif
            }
        }else if(sData == 'C'){
            // Requested block: original A rows [ja, ja+n), columns [ia, ia+m).
            // Output: n_target_cols x n buffer, column-major with lda = n.
            int owner_row = indxg2p(ja, array_descA.mb(), array_descA.irsrc(), array_descA.nprows());

            int local_row_start = num_loc(ja, array_descA.mb(), array_descA.myprow(), array_descA.irsrc(), array_descA.nprows());
            int local_row_end   = num_loc(ja + n, array_descA.mb(), array_descA.myprow(), array_descA.irsrc(), array_descA.nprows());
            int local_rows = local_row_end - local_row_start;

            int n_target_cols = target_desc.n_loc();

            if(array_descA.myprow() == owner_row && local_rows > 0 && n_target_cols > 0){
                // source submatrix is local_rows x n_target_cols in A, transpose to n_target_cols x local_rows
                BLAS_CHECK(deblasGeam(
                    ddla_handle->blasH, geam_op, DEBLAS_OP_N,
                    n_target_cols, local_rows,
                    one,
                    d_A + local_row_start, lld,
                    zero,
                    d_block_A, n_target_cols,
                    d_block_A, n_target_cols
                ));
            }
            if(n_target_cols > 0){
                #ifdef DDLA_USE_GPU_CPU_TUNNEL
                MPI_CHECK(cclBcast(h_temp.data(), d_block_A, n * n_target_cols, owner_row, ddla_handle->col_comm, ddla_handle->stream));
                #else
                CCL_CHECK(cclBcast(d_block_A, n * n_target_cols, owner_row, col_nccl_comm, ddla_handle->stream));
                #endif
            }
        }
    }
    return;
}

template <typename T>
void transport_block(
    const char& sData, const char& trans,
    const int& m, const int& n,
    const T* d_A, const int& ia, const int& ja, const DdlaDesc& array_descA,
    T* d_block_A
)
{
    transport_block(sData, trans, m, n, d_A, ia, ja, array_descA, array_descA, d_block_A);
}

template void transport_block<float>
(
    const char& sData, const char& trans,
    const int& m, const int& n,
    const float* d_A, const int& ia, const int& ja, const DdlaDesc& array_descA,
    float* d_block_A
);

template void transport_block<double>
(
    const char& sData, const char& trans,
    const int& m, const int& n,
    const double* d_A, const int& ia, const int& ja, const DdlaDesc& array_descA,
    double* d_block_A
);

template void transport_block<std::complex<float>>
(
    const char& sData, const char& trans,
    const int& m, const int& n,
    const std::complex<float>* d_A, const int& ia, const int& ja, const DdlaDesc& array_descA,
    std::complex<float>* d_block_A
);

template void transport_block<std::complex<double>>
(
    const char& sData, const char& trans,
    const int& m, const int& n,
    const std::complex<double>* d_A, const int& ia, const int& ja, const DdlaDesc& array_descA,
    std::complex<double>* d_block_A
);


template void transport_block<float>
(
    const char& sData, const char& trans,
    const int& m, const int& n,
    const float* d_A, const int& ia, const int& ja,
    const DdlaDesc& array_descA,
    const DdlaDesc& target_desc,
    float* d_block_A
);

template void transport_block<double>
(
    const char& sData, const char& trans,
    const int& m, const int& n,
    const double* d_A, const int& ia, const int& ja,
    const DdlaDesc& array_descA,
    const DdlaDesc& target_desc,
    double* d_block_A
);

template void transport_block<std::complex<float>>
(
    const char& sData, const char& trans,
    const int& m, const int& n,
    const std::complex<float>* d_A, const int& ia, const int& ja,
    const DdlaDesc& array_descA,
    const DdlaDesc& target_desc,
    std::complex<float>* d_block_A
);

template void transport_block<std::complex<double>>
(
    const char& sData, const char& trans,
    const int& m, const int& n,
    const std::complex<double>* d_A, const int& ia, const int& ja,
    const DdlaDesc& array_descA,
    const DdlaDesc& target_desc,
    std::complex<double>* d_block_A
);


} // namespace DDLA
