#include <ddla/ddla.h>
#include <cassert>
#include "require_gpu.h"

namespace ddla{

template <typename T>
void pposv(
    const char& side, const char& uplo, const char& trans,
    const int & n, const int& nrhs,
    T* d_A, const int& ia, const int& ja, const DdlaDesc& array_descA,
    T* d_B, const int& ib, const int& jb, const DdlaDesc& array_descB,
    int& info, // host pointer
    bool is_head, int location
)
{
    DdlaHandle_t ddla_handle = array_descA.ddla_handle();
    detail::require_gpu_backend(ddla_handle, "pposv");
    // Solve operates on the leading n x n / n x nrhs sub-matrices anchored at
    // global (0,0); ia/ja/ib/jb are reserved and must be 1 (1-based).
    assert(ia == 1 && ja == 1 && ib == 1 && jb == 1);
    bool is_nega = ppotrf(uplo, n, d_A, ia, ja, array_descA, info, is_head, location);
    // is_nega must be *passed to* ppotrs (which applies the corresponding sign
    // flip), not used to skip the solve entirely -- the previous `&& !is_nega`
    // guard silently returned an unsolved B whenever the head correction fired.

    // ppotrs handles the head-correction B permutation itself (keyed on
    // `location`), so there is nothing else to do here.
    if(info == 0)
        ppotrs(side, uplo, trans, n, nrhs, d_A, array_descA, d_B, array_descB, is_nega, location);
    return;
}

template void pposv<std::complex<float>>(
    const char& side, const char& uplo, const char& trans,
    const int & n, const int& nrhs,
    std::complex<float>* d_A, const int& ia, const int& ja, const DdlaDesc& array_descA,
    std::complex<float>* d_B, const int& ib, const int& jb, const DdlaDesc& array_descB,
    int& info, // host pointer
    bool is_head, int location
);

template void pposv<float>(
    const char& side, const char& uplo, const char& trans,
    const int & n, const int& nrhs,
    float* d_A, const int& ia, const int& ja, const DdlaDesc& array_descA,
    float* d_B, const int& ib, const int& jb, const DdlaDesc& array_descB,
    int& info, // host pointer
    bool is_head, int location
);

template void pposv<double>(
    const char& side, const char& uplo, const char& trans,
    const int & n, const int& nrhs,
    double* d_A, const int& ia, const int& ja, const DdlaDesc& array_descA,
    double* d_B, const int& ib, const int& jb, const DdlaDesc& array_descB,
    int& info, // host pointer
    bool is_head, int location
);

template void pposv<std::complex<double>>(
    const char& side, const char& uplo, const char& trans,
    const int & n, const int& nrhs,
    std::complex<double>* d_A, const int& ia, const int& ja, const DdlaDesc& array_descA,
    std::complex<double>* d_B, const int& ib, const int& jb, const DdlaDesc& array_descB,
    int& info, // host pointer
    bool is_head, int location
);

} // namespace ddla
