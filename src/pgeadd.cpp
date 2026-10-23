#include <ddla/ddla.h>
#include "ddla_desc.h"
#include <cassert>
#include "ddla_connector.h"
#include "ddla_stream_impl.h"
#include "require_backend.h"
#include <vector>
#include "transport_block.h"
#include "comm_traits.h"

namespace ddla{

template <DdlaBackend Backend, typename T>
void pgeadd(
    const DdlaHandle_t& handle, char transa, char transb,
    int m, int n,
    const T& alpha,
    const T* A, const int* array_descA,
    const T& beta,
    const T* B, const int* array_descB,
    T* C, const int* array_descC
)
{
    check_desc(array_descA, handle);
    check_desc(array_descB, handle);
    check_desc(array_descC, handle);
    int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
    ddlaGetGridDims(handle, nprows, npcols);
    ddlaGetGridCoords(handle, myprow, mypcol);


    DdlaHandle_t ddla_handle = handle;
    detail::require_backend<Backend>(ddla_handle, "pgeadd");

    if(transa != 'N' || transb != 'N')
    {
        if(nprows != npcols){
            throw std::runtime_error("the trans multiplication is now implemented for mxn(m!=n) mpi grid");
        }
    }
    {
        int mbA, nbA, mbB, nbB, mbC, nbC;
        mbC = array_descC[DDLA_MB_];
        nbC = array_descC[DDLA_NB_];
        if(transa == 'N'){
            mbA = array_descA[DDLA_MB_];
            nbA = array_descA[DDLA_NB_];
        }else{
            mbA = array_descA[DDLA_NB_];
            nbA = array_descA[DDLA_MB_];
        }

        if(transb == 'N'){
            mbB = array_descB[DDLA_MB_];
            nbB = array_descB[DDLA_NB_];
        }else{
            mbB = array_descB[DDLA_NB_];
            nbB = array_descB[DDLA_MB_];
        }
        assert(mbA == mbB && mbA == mbC);
        assert(nbA == nbB && nbA == nbC);
    }


    int m_loc_C = num_loc(m, array_descC[DDLA_MB_], myprow, array_descC[DDLA_RSRC_], nprows);
    int n_loc_C = num_loc(n, array_descC[DDLA_NB_], mypcol, array_descC[DDLA_CSRC_], npcols);


    if(transa == 'N' && transb == 'N'){
        geam<Backend, T>(
            ddla_handle, transa, transb,
            m_loc_C, n_loc_C,
            alpha, 
            A, array_descA[DDLA_LLD_],
            beta,
            B, array_descB[DDLA_LLD_],
            C, array_descC[DDLA_LLD_]
        );
        
    }else if (transa != 'N' && transb != 'N'){
        if(myprow == mypcol){
            geam<Backend, T>(
                ddla_handle, transa, transb,
                m_loc_C, n_loc_C,
                alpha, 
                A, array_descA[DDLA_LLD_],
                beta,
                B, array_descB[DDLA_LLD_],
                C, array_descC[DDLA_LLD_]
            );
        }else{
            T* temp;
            DDLA_CHECK(ddlaMalloc(ddla_handle, (void**)&temp, m_loc_C * n_loc_C * sizeof(T)));
            geam<Backend, T>(
                ddla_handle, transa, transb,
                n_loc_C, m_loc_C,
                alpha,
                A, array_descA[DDLA_LLD_],
                beta,
                B, array_descB[DDLA_LLD_],
                temp, n_loc_C
            );
            int trans_rank = ddla_handle->rc_to_rank(mypcol, myprow);
            commGroupStart<Backend>(ddla_handle);
            if(myprow > mypcol){
                commSend<Backend>(ddla_handle, CommScope::Grid, temp, (std::size_t)m_loc_C * n_loc_C, trans_rank);
                commRecv<Backend>(ddla_handle, CommScope::Grid, C, (std::size_t)m_loc_C * n_loc_C, trans_rank);
            }else{
                commRecv<Backend>(ddla_handle, CommScope::Grid, C, (std::size_t)m_loc_C * n_loc_C, trans_rank);
                commSend<Backend>(ddla_handle, CommScope::Grid, temp, (std::size_t)m_loc_C * n_loc_C, trans_rank);
            }
            commGroupEnd<Backend>(ddla_handle);
            DDLA_CHECK(ddlaFree(ddla_handle, temp));
        }
    }else{
        if(myprow == mypcol){
            geam<Backend, T>(
                ddla_handle, transa, transb,
                m_loc_C, n_loc_C,
                alpha, 
                A, array_descA[DDLA_LLD_],
                beta,
                B, array_descB[DDLA_LLD_],
                C, array_descC[DDLA_LLD_]
            );
        }else{
            T* temp;
            DDLA_CHECK(ddlaMalloc(ddla_handle, (void**)&temp, m_loc_C * n_loc_C * sizeof(T)));
            const T* comm = transa != 'N' ? A : B;
            const T* nt = transa == 'N' ? A : B;
            const T& trans_scale = transa != 'N' ? alpha : beta;
            const T& nontrans_scale = transa == 'N' ? alpha : beta;
            int trans_rank = ddla_handle->rc_to_rank(mypcol, myprow);
            commGroupStart<Backend>(ddla_handle);
            if(myprow > mypcol){
                commSend<Backend>(ddla_handle, CommScope::Grid, comm, (std::size_t)m_loc_C * n_loc_C, trans_rank);
                commRecv<Backend>(ddla_handle, CommScope::Grid, temp, (std::size_t)m_loc_C * n_loc_C, trans_rank);
            }else{
                commRecv<Backend>(ddla_handle, CommScope::Grid, temp, (std::size_t)m_loc_C * n_loc_C, trans_rank);
                commSend<Backend>(ddla_handle, CommScope::Grid, comm, (std::size_t)m_loc_C * n_loc_C, trans_rank);
            }
            commGroupEnd<Backend>(ddla_handle);
            geam<Backend, T>(
                ddla_handle, (transa != 'N' ? transa : transb), 'N',
                m_loc_C, n_loc_C,
                trans_scale,
                temp, n_loc_C,
                nontrans_scale,
                nt, m_loc_C,
                C, m_loc_C
            );
            DDLA_CHECK(ddlaFree(ddla_handle, temp));
        }
    }

    return;
    
}

#define INSTANTIATE_PGEADD(BACKEND, TYPE)                                   \
    template void pgeadd<BACKEND, TYPE>(const DdlaHandle_t&, char transa, char transb, \
    int m, int n, \
    const TYPE& alpha, \
    const TYPE* A, const int* array_descA, \
    const TYPE& beta, \
    const TYPE* B, const int* array_descB, \
    TYPE* C, const int* array_descC)

#if DDLA_HAS_CPU
INSTANTIATE_PGEADD(DdlaBackend::CPU, float);
INSTANTIATE_PGEADD(DdlaBackend::CPU, double);
INSTANTIATE_PGEADD(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_PGEADD(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_PGEADD(DdlaBackend::GPU, float);
INSTANTIATE_PGEADD(DdlaBackend::GPU, double);
INSTANTIATE_PGEADD(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_PGEADD(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_PGEADD



}
