#include <ddla/ddla.h>
#include "ddla_desc.h"
#include <cassert>
#include "ddla_stream_impl.h"
#include "require_backend.h"

namespace ddla {

/**
 * @brief Distributed linear-system solver without pivoting (driver).
 *
 * Convenience wrapper: pgetrf_nopiv (LU) + pgetrs_nopiv (solve).
 * Solves A * X = B without pivoting.
 *
 * @tparam T   Scalar type.
 * @param n       Order of square matrix A.
 * @param nrhs    Number of right-hand sides.
 * @param d_A     Device pointer to A (input: coefficient; output: LU factors).
 * @param array_descA  ScaLAPACK int[9] descriptor for A.
 * @param d_B     Device pointer to RHS / solution B (input/output).
 * @param array_descB  ScaLAPACK int[9] descriptor for B.
 * @throws std::runtime_error if LU factorization fails (info != 0).
 */
template <DdlaBackend Backend, typename T>
void pgesv_nopiv(
    const DdlaHandle_t& handle, const char& side, const char& trans, const int& n, const int& nrhs,
    T* d_A, const int* array_descA,
    T* d_B, const int* array_descB
)
{
    check_desc(array_descB, handle);
    check_desc(array_descA, handle);
    DdlaHandle_t ddla_handle = handle;
    detail::require_backend<Backend>(ddla_handle, "pgesv_nopiv");
    assert(n <= array_descA[DDLA_M_] && n <= array_descA[DDLA_N_]);
    int info = 1;
    pgetrf_nopiv<Backend>(handle, n, n, d_A, array_descA, info);
    if (info != 0) {
        throw std::runtime_error("pgesv_nopiv: pgetrf_nopiv returned info = " + std::to_string(info));
    }
    pgetrs_nopiv<Backend>(handle, side, trans, n, nrhs, d_A, array_descA, d_B, array_descB);
}

#define INSTANTIATE_PGESV_NOPIV(BACKEND, TYPE)                                   \
    template void pgesv_nopiv<BACKEND, TYPE>(const DdlaHandle_t&, const char& side, const char& trans, const int& n, const int& nrhs, \
    TYPE* d_A, const int* array_descA, \
    TYPE* d_B, const int* array_descB)

#if DDLA_HAS_CPU
INSTANTIATE_PGESV_NOPIV(DdlaBackend::CPU, float);
INSTANTIATE_PGESV_NOPIV(DdlaBackend::CPU, double);
INSTANTIATE_PGESV_NOPIV(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_PGESV_NOPIV(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_PGESV_NOPIV(DdlaBackend::GPU, float);
INSTANTIATE_PGESV_NOPIV(DdlaBackend::GPU, double);
INSTANTIATE_PGESV_NOPIV(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_PGESV_NOPIV(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_PGESV_NOPIV


} // namespace ddla
