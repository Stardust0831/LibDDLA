#include <ddla/ddla.h>
#include "ddla_desc.h"
#include <cassert>
#include <vector>
#include "ddla_connector.h"
#include "ddla_stream_impl.h"
#include "require_backend.h"
#include "comm_traits.h"

namespace ddla{

template <DdlaBackend Backend, typename T>
void pgetrf(
    const DdlaHandle_t& handle, int m, int n,
    T* A, const int* array_descA,
    int* ipiv, // host
    int& info  // host
)
{
    check_desc(array_descA, handle);
    int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
    ddlaGetGridDims(handle, nprows, npcols);
    ddlaGetGridCoords(handle, myprow, mypcol);


    DdlaHandle_t ddla_handle = handle;
    detail::require_backend<Backend>(ddla_handle, "pgetrf");
    assert(m <= array_descA[DDLA_M_] && n <= array_descA[DDLA_N_]);


    int nb = array_descA[DDLA_MB_];
    int nb_real;
    assert(array_descA[DDLA_MB_]==array_descA[DDLA_NB_]);
    int lld = array_descA[DDLA_LLD_];

    int m_loc = num_loc(m, array_descA[DDLA_MB_], myprow, array_descA[DDLA_RSRC_], nprows);
    int n_loc = num_loc(n, array_descA[DDLA_NB_], mypcol, array_descA[DDLA_CSRC_], npcols);


    int mm_row_start = 0;
    int mm_col_start = 0;
    int i_loc,j_loc;
    int owner_row,owner_col;

    T *temp_block;
    DDLA_CHECK(ddlaMalloc(ddla_handle, &temp_block, sizeof(T)*nb*nb));

    T *temp_L;
    DDLA_CHECK(ddlaMalloc(ddla_handle, &temp_L, sizeof(T)*m_loc*nb));

    T *temp_U;
    DDLA_CHECK(ddlaMalloc(ddla_handle, &temp_U, sizeof(T)*nb*n_loc));

    info = 0;

    for(int n_s=0;n_s<std::min(m,n);n_s+=nb){
        nb_real = std::min(nb, std::min(m,n)-n_s);

        i_loc = indx_g2l_r(array_descA, handle, n_s);
        j_loc = indx_g2l_c(array_descA, handle, n_s);

        owner_row = indxg2p(n_s, nb, array_descA[DDLA_RSRC_], nprows);
        owner_col = indxg2p(n_s, nb, array_descA[DDLA_CSRC_], npcols);


        // start pgetf2

        pgetf2_panel<Backend>(handle, 
            m, n, nb_real,
            A, n_s, array_descA,
            ipiv, info
        );
        // finish pgetf2
        if(info != 0){
            break;
        }
        // update trailing matrix
        if(i_loc>=0)
            mm_row_start +=nb; // update row start   
        if(j_loc>=0){
            mm_col_start+=nb;
            if(i_loc>=0){
                DDLA_CHECK(ddlaMemcpy2D(ddla_handle, temp_block, nb_real * sizeof(T), A + i_loc + j_loc * lld,
                                        lld * sizeof(T), nb_real * sizeof(T), nb_real));
                
            }
            if(mm_row_start<m_loc){
                DDLA_CHECK(ddlaMemcpy2D(ddla_handle, temp_L, (m_loc-mm_row_start) * sizeof(T),
                                        A + mm_row_start + j_loc * lld, lld * sizeof(T),
                                        (m_loc - mm_row_start) * sizeof(T), nb_real));
            }
        }
        if(mm_row_start<m_loc){
            commBcast<Backend>(ddla_handle, CommScope::Row, temp_L, (std::size_t)(m_loc - mm_row_start) * nb_real, owner_col);
        }
        // broadcast block column
        if(i_loc>=0){
            commBcast<Backend>(ddla_handle, CommScope::Row, temp_block, (std::size_t)nb_real * nb_real, owner_col);
            if(mm_col_start<n_loc){
                trsm<Backend, T>(
                    ddla_handle, 'L', 'L', 'N', 'U',
                    nb_real, n_loc - mm_col_start, T(1),
                    temp_block, nb_real,
                    A + i_loc + mm_col_start * lld, lld);
                DDLA_CHECK(ddlaMemcpy2D(ddla_handle, temp_U, nb_real * sizeof(T), A + i_loc + mm_col_start * lld,
                                        lld * sizeof(T), nb_real * sizeof(T), n_loc - mm_col_start));
            }   
        }
        if(mm_col_start<n_loc){
            commBcast<Backend>(ddla_handle, CommScope::Col, temp_U, (std::size_t)nb_real * (n_loc - mm_col_start), owner_row);
        }
        if(mm_row_start<m_loc&&mm_col_start<n_loc){
            gemm<Backend, T>(ddla_handle, 'N', 'N',
                m_loc - mm_row_start, n_loc - mm_col_start, nb_real,
                -1.0,
                temp_L, m_loc - mm_row_start,
                temp_U, nb_real,
                1.0,
                A + mm_row_start + mm_col_start * lld, lld);
        }
    }
    DDLA_CHECK(ddlaFree(ddla_handle, temp_block));
    DDLA_CHECK(ddlaFree(ddla_handle, temp_L));
    DDLA_CHECK(ddlaFree(ddla_handle, temp_U));
    DDLA_CHECK(ddlaSynchronize(ddla_handle));

}

#define INSTANTIATE_PGETRF(BACKEND, TYPE)                                   \
    template void pgetrf<BACKEND, TYPE>(const DdlaHandle_t&, int m, int n, \
    TYPE* A, const int* array_descA, \
    int* ipiv, \
    int& info)

#if DDLA_HAS_CPU
INSTANTIATE_PGETRF(DdlaBackend::CPU, float);
INSTANTIATE_PGETRF(DdlaBackend::CPU, double);
INSTANTIATE_PGETRF(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_PGETRF(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_PGETRF(DdlaBackend::GPU, float);
INSTANTIATE_PGETRF(DdlaBackend::GPU, double);
INSTANTIATE_PGETRF(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_PGETRF(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_PGETRF


}
