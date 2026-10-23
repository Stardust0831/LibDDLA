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
void pgetf2_panel(
    const DdlaHandle_t& handle, int m, int n, int nb_real,
    T* A, int n_start, const int* array_descA,
    int* ipiv, // host
    int& info  // host
)
{
    check_desc(array_descA, handle);
    int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
    ddlaGetGridDims(handle, nprows, npcols);
    ddlaGetGridCoords(handle, myprow, mypcol);


    assert(m <= array_descA[DDLA_M_] && n <= array_descA[DDLA_N_]);
    DdlaHandle_t ddla_handle = handle;
    detail::require_backend<Backend>(ddla_handle, "pgetf2_panel");



    int nb = array_descA[DDLA_MB_];
    assert(array_descA[DDLA_MB_]==array_descA[DDLA_NB_]);
    int lld = array_descA[DDLA_LLD_];

    int m_loc = num_loc(m, array_descA[DDLA_MB_], myprow, array_descA[DDLA_RSRC_], nprows);
    int n_loc = num_loc(n, array_descA[DDLA_NB_], mypcol, array_descA[DDLA_CSRC_], npcols);

    const int panel = std::min(32, nb/2>0?nb/2:1);
    int panel_real;


    int mm_row_start = num_loc(n_start, nb, myprow, array_descA[DDLA_RSRC_], nprows);
    int mm_col_start = num_loc(n_start, nb, mypcol, array_descA[DDLA_CSRC_], npcols);
    int i_loc,j_loc;
    int owner_row;

    T *temp_U;
    DDLA_CHECK(ddlaMalloc(ddla_handle, &temp_U, sizeof(T)*nb_real*panel));

    info = 0;

    int j_s = indx_g2l_c(array_descA, handle, n_start);
    

    for(int n_s=n_start;n_s<n_start+nb_real;n_s+=panel){
        panel_real = std::min(panel, nb_real+n_start-n_s);

        i_loc = indx_g2l_r(array_descA, handle, n_s);
        j_loc = indx_g2l_c(array_descA, handle, n_s);

        owner_row = indxg2p(n_s, nb, array_descA[DDLA_RSRC_], nprows);
        // start pgetf2
        pgetf2<Backend>(handle, 
            m, n, panel_real,
            A, n_s, array_descA,
            ipiv, info
        );
        if(info != 0){
            DDLA_CHECK(ddlaFree(ddla_handle, temp_U));
            return;
        }
        // finish pgetf2
        // update trailing matrix
        if(i_loc>=0)
            mm_row_start +=panel; // update row start   
        mm_col_start+=panel;

        // broadcast block column
        if(mm_col_start<j_s + nb_real && j_loc>=0){
            if(i_loc>=0){
                trsm<Backend, T>(
                    ddla_handle, 'L', 'L', 'N', 'U',
                    panel_real, j_s + nb_real - mm_col_start, T(1),
                    A + i_loc + j_loc * lld, lld,
                    A + i_loc + mm_col_start * lld, lld);
                DDLA_CHECK(ddlaMemcpy2D(ddla_handle, temp_U, panel_real * sizeof(T),
                                        A + i_loc + mm_col_start * lld, lld * sizeof(T), panel_real * sizeof(T),
                                        j_s + nb_real - mm_col_start));
            } 
            commBcast<Backend>(ddla_handle, CommScope::Col, temp_U, (std::size_t)panel_real * (j_s + nb_real - mm_col_start), owner_row);
        }
        if(mm_row_start<m_loc && mm_col_start<j_s + nb_real && j_loc>=0){
            gemm<Backend, T>(ddla_handle, 'N', 'N',
                m_loc - mm_row_start, j_s + nb_real - mm_col_start, panel_real,
                -1.0,
                A + mm_row_start + j_loc * lld, lld,
                temp_U, panel_real,
                1.0,
                A + mm_row_start + mm_col_start * lld, lld
            );
        }
    }
    info = 0;
    DDLA_CHECK(ddlaFree(ddla_handle, temp_U));

}

#define INSTANTIATE_PGETF2_PANEL(BACKEND, TYPE)                                   \
    template void pgetf2_panel<BACKEND, TYPE>(const DdlaHandle_t&, int m, int n, int nb_real, \
    TYPE* A, int n_start, const int* array_descA, \
    int* ipiv, \
    int& info)

#if DDLA_HAS_CPU
INSTANTIATE_PGETF2_PANEL(DdlaBackend::CPU, float);
INSTANTIATE_PGETF2_PANEL(DdlaBackend::CPU, double);
INSTANTIATE_PGETF2_PANEL(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_PGETF2_PANEL(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_PGETF2_PANEL(DdlaBackend::GPU, float);
INSTANTIATE_PGETF2_PANEL(DdlaBackend::GPU, double);
INSTANTIATE_PGETF2_PANEL(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_PGETF2_PANEL(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_PGETF2_PANEL


}
