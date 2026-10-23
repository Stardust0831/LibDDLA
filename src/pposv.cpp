#include <ddla/ddla.h>
#include "ddla_desc.h"
#include <cassert>
#include "require_backend.h"

namespace ddla{

template <DdlaBackend Backend, typename T>
void pposv(
    const DdlaHandle_t& handle, const char& side, const char& uplo, const char& trans,
    const int & n, const int& nrhs,
    T* A, const int& ia, const int& ja, const int* array_descA,
    T* B, const int& ib, const int& jb, const int* array_descB,
    int& info, // host pointer
    bool is_head, int location
)
{
    check_desc(array_descB, handle);
    check_desc(array_descA, handle);
    DdlaHandle_t ddla_handle = handle;
    detail::require_backend<Backend>(ddla_handle, "pposv");
    // Solve operates on the leading n x n / n x nrhs sub-matrices anchored at
    // global (0,0); ia/ja/ib/jb are reserved and must be 1 (1-based).
    assert(ia == 1 && ja == 1 && ib == 1 && jb == 1);
    bool is_nega = ppotrf<Backend>(handle, uplo, n, A, ia, ja, array_descA, info, is_head, location);
    // is_nega must be *passed to* ppotrs (which applies the corresponding sign
    // flip), not used to skip the solve entirely -- the previous `&& !is_nega`
    // guard silently returned an unsolved B whenever the head correction fired.

    // ppotrs handles the head-correction B permutation itself (keyed on
    // `location`), so there is nothing else to do here.
    if(info == 0)
        ppotrs<Backend>(handle, side, uplo, trans, n, nrhs, A, array_descA, B, array_descB, is_nega, location);
    return;
}

#define INSTANTIATE_PPOSV(BACKEND, TYPE)                                   \
    template void pposv<BACKEND, TYPE>(const DdlaHandle_t&, const char& side, const char& uplo, const char& trans, \
    const int & n, const int& nrhs, \
    TYPE* A, const int& ia, const int& ja, const int* array_descA, \
    TYPE* B, const int& ib, const int& jb, const int* array_descB, \
    int& info, \
    bool is_head, int location)

#if DDLA_HAS_CPU
INSTANTIATE_PPOSV(DdlaBackend::CPU, float);
INSTANTIATE_PPOSV(DdlaBackend::CPU, double);
INSTANTIATE_PPOSV(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_PPOSV(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_PPOSV(DdlaBackend::GPU, float);
INSTANTIATE_PPOSV(DdlaBackend::GPU, double);
INSTANTIATE_PPOSV(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_PPOSV(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_PPOSV


} // namespace ddla
