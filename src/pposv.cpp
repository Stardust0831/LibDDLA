#include <ddla/ddla.h>
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
    bool is_nega = ppotrf(uplo, n, d_A, ia, ja, array_descA, info, is_head, location);
    // is_nega must be *passed to* ppotrs (which applies the corresponding sign
    // flip), not used to skip the solve entirely -- the previous `&& !is_nega`
    // guard silently returned an unsolved B whenever the head correction fired.

    // When location != -1, ppotrf already relocated the head element to the
    // last global index via an in-place symmetric permutation of A -- but it
    // only ever touches A, never B. Without also permuting B here, ppotrs
    // (called below with location forced to -1, since the head element is
    // now genuinely last) would solve against the permuted A but an
    // un-permuted B, silently giving a wrong X. Apply the same row swap to B
    // before the solve and again after (self-inverse), so X comes back in
    // the caller's original ordering. This mirrors the manual pswap
    // bookkeeping any direct ppotrf+ppotrs caller must otherwise do (see
    // tests/test_api_grid_ppotrf_head.cpp's permute_rhs_rows).
    const bool needs_permute = is_head && location != -1 && location != n;
    if(needs_permute)
        pswap(nrhs, d_B, location, jb, array_descB, array_descB.m(),
                    d_B, n,        jb, array_descB, array_descB.m());
    if(info == 0)
        ppotrs(side, uplo, trans, n, nrhs, d_A, array_descA, d_B, array_descB, is_nega, -1);
    if(needs_permute)
        pswap(nrhs, d_B, location, jb, array_descB, array_descB.m(),
                    d_B, n,        jb, array_descB, array_descB.m());
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

} // namespace DDLA