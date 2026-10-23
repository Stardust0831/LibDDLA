#include <ddla/ddla.h>
#include "ddla_desc.h"
#include <cassert>
#include <complex>
#include "ddla_connector.h"
#include "ddla_stream_impl.h"
#include "require_backend.h"
#include "comm_traits.h"

#include "pgetf2_kernels.h"

namespace ddla{

namespace {

struct MaxLoc {
    double value;
    int index;
};

// The pivot metric of ?amax: |x| for real types, |re| + |im| for complex ones (what the vendor ?amax compares).
inline double pivot_metric(float v) { return std::abs(v); }
inline double pivot_metric(double v) { return std::abs(v); }
inline double pivot_metric(const std::complex<float>& v) { return std::abs(v.real()) + std::abs(v.imag()); }
inline double pivot_metric(const std::complex<double>& v) { return std::abs(v.real()) + std::abs(v.imag()); }

// The two per-column steps of the panel factorization.  A GPU handle runs the fused kernels of pgetf2_kernels.cpp (one
// launch and one host round trip for the pivot search, one launch for the scale and the rank-1 update): replacing them by
// iamax + scal + geru costs two extra synchronisations and launches per column, +11..24 % on a pgesv (measured 2026-10-20).
// A CPU handle has no launch latency to save and calls the BLAS routines directly.

// Local pivot of one column: metric (|x| or |re|+|im|), zero-based index into the column, and the element.
template <DdlaBackend Backend, typename T>
void find_local_pivot(const DdlaHandle_t& handle, const T* column, int length, void* d_workspace,
                      double& metric, int& index, T& value)
{
    if constexpr (Backend == DdlaBackend::GPU) {
        detail::pgetf2_find_local_pivot(handle, column, length, d_workspace, metric, index, value);
    } else {
        (void)d_workspace;
        int one_based = 0;
        iamax<Backend, T>(handle, length, column, 1, one_based);
        index = one_based - 1;
        DDLA_CHECK(ddlaMemcpy(handle, &value, column + index, sizeof(T)));
        DDLA_CHECK(ddlaSynchronize(handle));
        metric = pivot_metric(value);
    }
}

// column := column * inverse_pivot, then trailing -= column * pivot_row^T for the length_col columns to the right.
template <DdlaBackend Backend, typename T>
void scale_update(const DdlaHandle_t& handle, int length_row, int length_col, const T& inverse_pivot,
                  T* column, const T* pivot_row, T* trailing, int lld)
{
    if constexpr (Backend == DdlaBackend::GPU) {
        detail::pgetf2_scale_update(handle, length_row, length_col, inverse_pivot, column, pivot_row, trailing, lld);
    } else {
        if (length_row <= 0) return;
        scal<Backend, T>(handle, length_row, inverse_pivot, column, 1);
        if (length_col > 0) {
            geru<Backend, T>(handle, length_row, length_col, T(-1), column, 1, pivot_row, 1, trailing, lld);
        }
    }
}

template <DdlaBackend Backend, typename T>
std::size_t pivot_workspace_bytes()
{
    if constexpr (Backend == DdlaBackend::GPU) return detail::pgetf2_pivot_workspace_size<T>();
    else return 0;
}

template <typename T>
struct PivotBroadcast {
    int max_row = 0;
    int max_prow = 0;
    T max_value = T{};
};

} // namespace

// Factors the leading-block panel starting at global column n_s of width
// nb_real (<= nb), for the logical m-by-n sub-matrix anchored at global (0,0)
// of a possibly-larger descriptor.  All local extents are derived from the
// logical m/n (num_loc), so the panel kernel never touches rows/columns beyond
// the leading block.
template <DdlaBackend Backend, typename T>
void pgetf2(
    const DdlaHandle_t& handle, const int& m, const int& n, const int& nb_real,
    T* A, const int& n_s, const int* array_descA,
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
    detail::require_backend<Backend>(ddla_handle, "pgetf2");

    MPI_Comm row_comm = ddla_handle->row_comm;
    MPI_Comm col_comm = ddla_handle->col_comm;


    int m_loc = num_loc(m, array_descA[DDLA_MB_], myprow, array_descA[DDLA_RSRC_], nprows);
    int n_loc = num_loc(n, array_descA[DDLA_NB_], mypcol, array_descA[DDLA_CSRC_], npcols);
    int lld = array_descA[DDLA_LLD_];
    int nb = array_descA[DDLA_NB_];

    
    int max_row;
    int max_prow;
    T max_value{};

    T *temp;
    const size_t row_buffer_elems = static_cast<size_t>(n_loc > 0 ? n_loc : 1);
    DDLA_CHECK(ddlaMalloc(ddla_handle, &temp, sizeof(T)*row_buffer_elems));

    T *temp_peer = nullptr;
    DDLA_CHECK(ddlaMalloc(ddla_handle, &temp_peer, sizeof(T)*row_buffer_elems));

    void* d_pivot_workspace = nullptr;   // the GPU pivot kernel's result slot (zero bytes, hence null, on a CPU handle)
    DDLA_CHECK(ddlaMalloc(ddla_handle, &d_pivot_workspace, pivot_workspace_bytes<Backend, T>()));

    int i_loc = indx_g2l_r(array_descA, handle, n_s);
    int j_loc = indx_g2l_c(array_descA, handle, n_s);

    int owner_row = indxg2p(n_s, nb, array_descA[DDLA_RSRC_], nprows);
    int owner_col = indxg2p(n_s, nb, array_descA[DDLA_CSRC_], npcols);

    int mm_row_start = num_loc(n_s, nb, myprow, array_descA[DDLA_RSRC_], nprows);
    int mm_col_start = num_loc(n_s, nb, mypcol, array_descA[DDLA_CSRC_], npcols);

    info = 0;
    for(int i_tf2 = 0; i_tf2 < nb_real; i_tf2++){
        // find max_rows and value
        int i_panel, j_panel;
        if(i_loc >= 0)
            i_panel = i_loc + i_tf2;
        else
            i_panel = mm_row_start;
        if(j_loc >= 0)
            j_panel = j_loc + i_tf2;
        else
            j_panel = mm_col_start;
        if(j_loc >= 0){
            MaxLoc local_max{-1.0, -1};
            MaxLoc global_max{-1.0, -1};
            T local_max_value{};
            if(i_panel<m_loc){
                int local_max_index = 0;
                find_local_pivot<Backend, T>(
                    ddla_handle, A + j_panel * lld + i_panel, m_loc - i_panel,
                    d_pivot_workspace,
                    local_max.value, local_max_index, local_max_value);
                const int local_row = i_panel + local_max_index;
                local_max.index = indx_l2g_r(array_descA, handle, local_row);
            }
            MPI_CHECK(MPI_Allreduce(&local_max, &global_max, 1,
                                    MPI_DOUBLE_INT, MPI_MAXLOC, col_comm));
            max_row = global_max.index;
            max_prow = indxg2p(max_row, nb, array_descA[DDLA_RSRC_], nprows);
            if(myprow == max_prow){
                max_value = local_max_value;
            }
            MPI_CHECK(MPI_Bcast(&max_value, static_cast<int>(sizeof(T)), MPI_BYTE,
                                max_prow, col_comm));
        }

        PivotBroadcast<T> pivot;
        if(mypcol == owner_col){
            pivot.max_row = max_row;
            pivot.max_prow = max_prow;
            pivot.max_value = max_value;
        }
        MPI_CHECK(MPI_Bcast(&pivot, static_cast<int>(sizeof(pivot)), MPI_BYTE, owner_col, row_comm));
        max_row = pivot.max_row;
        max_prow = pivot.max_prow;
        max_value = pivot.max_value;

        int max_loc_row = indx_g2l_r(array_descA, handle, max_row);
        if(myprow == owner_row){
            ipiv[i_panel] = max_row + 1; // 1-based index like fortran
        }
        // exchange rows
        if(owner_row == max_prow){
            if(myprow == owner_row && max_loc_row != i_panel)
                swap<Backend, T>(
                    ddla_handle, n_loc,
                    A + i_panel, lld,
                    A + max_loc_row, lld
                );
        }else{
            if(myprow == owner_row){
                DDLA_CHECK(ddlaMemcpy2D(ddla_handle, temp, sizeof(T), A + i_panel, lld * sizeof(T), sizeof(T),
                                        n_loc));
                commSendrecv<Backend>(ddla_handle, CommScope::Col, temp, temp_peer, (std::size_t)n_loc, max_prow);
                DDLA_CHECK(ddlaMemcpy2D(ddla_handle, A + i_panel, lld * sizeof(T), temp_peer, sizeof(T),
                                        sizeof(T), n_loc));
            }else if(myprow == max_prow){
                DDLA_CHECK(ddlaMemcpy2D(ddla_handle, temp, sizeof(T), A + max_loc_row, lld * sizeof(T), sizeof(T),
                                        n_loc));
                commSendrecv<Backend>(ddla_handle, CommScope::Col, temp, temp_peer, (std::size_t)n_loc, owner_row);
                DDLA_CHECK(ddlaMemcpy2D(ddla_handle, A + max_loc_row, lld * sizeof(T), temp_peer, sizeof(T),
                                        sizeof(T), n_loc));
            }
        }
        if(std::abs(max_value) == 0.0){
            info = n_s+i_tf2+1;
            break;
        }
        if(j_loc>=0){
            const T inverse_pivot = T(1.0) / max_value;
            int64_t a_off;
            int length_row;
            
            if(i_loc>=0){
                a_off = (i_panel + 1) + j_panel * lld;
                length_row = m_loc - (i_panel + 1);
            }else{
                a_off = mm_row_start + j_panel * lld;
                length_row = m_loc - mm_row_start;
            }
            int length_col = nb_real - i_tf2 - 1;
            if(myprow == owner_row && length_col>0){
                DDLA_CHECK(ddlaMemcpy2D(ddla_handle, temp, 1 * sizeof(T), A + i_panel + (j_panel + 1) * lld,
                                        lld * sizeof(T), 1*sizeof(T), length_col));
            }
            if(length_col>0)
                commBcast<Backend>(ddla_handle, CommScope::Col, temp, (std::size_t)length_col, owner_row);
            // multipliers l = a / pivot, then the rank-1 update of the panel columns to the right: A -= l * u^T
            T* trailing = length_col > 0 ? A + a_off + lld : A + a_off;
            scale_update<Backend, T>(
                ddla_handle, length_row, length_col, inverse_pivot,
                A + a_off, temp, trailing, lld);
        }
    }
    DDLA_CHECK(ddlaFree(ddla_handle, temp));
    DDLA_CHECK(ddlaFree(ddla_handle, temp_peer));
    DDLA_CHECK(ddlaFree(ddla_handle, d_pivot_workspace));
    if(info != 0){
        DDLA_CHECK(ddlaSynchronize(ddla_handle));
    }
}

#define INSTANTIATE_PGETF2(BACKEND, TYPE)                                   \
    template void pgetf2<BACKEND, TYPE>(const DdlaHandle_t&, const int& m, const int& n, const int& nb_real, \
    TYPE* A, const int& n_s, const int* array_descA, \
    int* ipiv, \
    int& info)

#if DDLA_HAS_CPU
INSTANTIATE_PGETF2(DdlaBackend::CPU, float);
INSTANTIATE_PGETF2(DdlaBackend::CPU, double);
INSTANTIATE_PGETF2(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_PGETF2(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_PGETF2(DdlaBackend::GPU, float);
INSTANTIATE_PGETF2(DdlaBackend::GPU, double);
INSTANTIATE_PGETF2(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_PGETF2(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_PGETF2


} // namespace ddla
