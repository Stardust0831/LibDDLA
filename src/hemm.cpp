// ---------------------------------------------------------------------------
// Unified backend-templated Hermitian matrix-matrix product
//   C := alpha*A*B + beta*C    (side = 'L')   or
//   C := alpha*B*A + beta*C    (side = 'R').
//
// Structure follows src/gemm.cpp and src/herk.cpp exactly: one definition
// covering CPU (Fortran BLAS) and GPU (cuBLAS/hipBLAS), selected with
// `if constexpr` on the Backend template parameter, with the arms themselves
// guarded by the BUILD CAPABILITY macros (DDLA_HAS_CPU / DDLA_HAS_GPU) and the
// vendor chosen per scalar arm by the PER-TU macros (DDLA_USE_CUDA /
// DDLA_USE_HIP).
//
// Why the capability guard matters here: the previous design put a `#else`
// CPU arm inside src/hemm.h selected by `!defined(DDLA_USE_CUDA) &&
// !defined(DDLA_USE_HIP)`.  In a dual build (DDLA_USE_CPU + DDLA_USE_CUDA)
// every TU defines DDLA_USE_CUDA, so that arm was never compiled -- while
// DDLA_HAS_CPU was 1 and the CPU instantiation of this function still existed.
// A CPU-backend call would therefore have reached cublasChemm with HOST
// pointers.  Guarding on DDLA_HAS_CPU removes that class of bug.
//
// Scalar convention: xHEMM takes alpha and beta of the operands' own type T
// (both real for the real scalar types, complex for the complex ones), so the
// signature takes `const T&` for both.  For the real scalar types the routine
// is xSYMM; there is no transpose flag in this family.
// ---------------------------------------------------------------------------
#include <ddla/ddla.h>
#include <ddla/ddla_handle_t.h>

#include <complex>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "blas_enums.h"
#include "ddla_connector.h"
#include "ddla_stream_impl.h"

#if DDLA_HAS_CPU
extern "C" {
// Real: xHEMM has no real-scalar BLAS counterpart, so the real types use xSYMM.
void ssymm_(const char*, const char*, const int*, const int*, const float*,
            const float*, const int*, const float*, const int*,
            const float*, float*, const int*);
void dsymm_(const char*, const char*, const int*, const int*, const double*,
            const double*, const int*, const double*, const int*,
            const double*, double*, const int*);
// Complex: xHEMM with complex alpha/beta.
void chemm_(const char*, const char*, const int*, const int*, const std::complex<float>*,
            const std::complex<float>*, const int*, const std::complex<float>*, const int*,
            const std::complex<float>*, std::complex<float>*, const int*);
void zhemm_(const char*, const char*, const int*, const int*, const std::complex<double>*,
            const std::complex<double>*, const int*, const std::complex<double>*, const int*,
            const std::complex<double>*, std::complex<double>*, const int*);
}
#endif

// ---------------------------------------------------------------------------
// GPU arm: the vendor calls.  Kept as free deblasHemm overloads so the
// dispatch above stays a single BLAS_CHECK call per scalar type, exactly as
// ddla::gemm does it inline.  Private to this translation unit -- the public
// interface is the templated ddla::hemm declared in <ddla/ddla.h>.
// ---------------------------------------------------------------------------
#if DDLA_HAS_GPU

namespace ddla {

inline deblasStatus_t deblasHemm(
    deblasHandle_t handle, deblasSideMode_t side, deblasFillMode_t uplo,
    int m, int n,
    const float& alpha,
    const float* A, int lda,
    const float* B, int ldb,
    const float& beta,
    float* C, int ldc
)
{
    // Real HEMM == SYMM.
    #if defined(DDLA_USE_HIP)
    return hipblasSsymm(handle, side, uplo, m, n, &alpha, A, lda, B, ldb, &beta, C, ldc);
    #elif defined(DDLA_USE_CUDA)
    return cublasSsymm(handle, side, uplo, m, n, &alpha, A, lda, B, ldb, &beta, C, ldc);
    #else
    throw std::runtime_error("deblasHemm: no GPU vendor selected at compile time");
    #endif
}

inline deblasStatus_t deblasHemm(
    deblasHandle_t handle, deblasSideMode_t side, deblasFillMode_t uplo,
    int m, int n,
    const double& alpha,
    const double* A, int lda,
    const double* B, int ldb,
    const double& beta,
    double* C, int ldc
)
{
    #if defined(DDLA_USE_HIP)
    return hipblasDsymm(handle, side, uplo, m, n, &alpha, A, lda, B, ldb, &beta, C, ldc);
    #elif defined(DDLA_USE_CUDA)
    return cublasDsymm(handle, side, uplo, m, n, &alpha, A, lda, B, ldb, &beta, C, ldc);
    #else
    throw std::runtime_error("deblasHemm: no GPU vendor selected at compile time");
    #endif
}

inline deblasStatus_t deblasHemm(
    deblasHandle_t handle, deblasSideMode_t side, deblasFillMode_t uplo,
    int m, int n,
    const std::complex<float>& alpha,
    const std::complex<float>* A, int lda,
    const std::complex<float>* B, int ldb,
    const std::complex<float>& beta,
    std::complex<float>* C, int ldc
)
{
    #if defined(DDLA_USE_HIP)
    return hipblasChemm(handle, side, uplo, m, n,
        (hipblasComplex*)&alpha, (hipblasComplex*)A, lda,
        (hipblasComplex*)B, ldb, (hipblasComplex*)&beta, (hipblasComplex*)C, ldc);
    #elif defined(DDLA_USE_CUDA)
    return cublasChemm(handle, side, uplo, m, n,
        (cuFloatComplex*)&alpha, (cuFloatComplex*)A, lda,
        (cuFloatComplex*)B, ldb, (cuFloatComplex*)&beta, (cuFloatComplex*)C, ldc);
    #else
    throw std::runtime_error("deblasHemm: no GPU vendor selected at compile time");
    #endif
}

inline deblasStatus_t deblasHemm(
    deblasHandle_t handle, deblasSideMode_t side, deblasFillMode_t uplo,
    int m, int n,
    const std::complex<double>& alpha,
    const std::complex<double>* A, int lda,
    const std::complex<double>* B, int ldb,
    const std::complex<double>& beta,
    std::complex<double>* C, int ldc
)
{
    #if defined(DDLA_USE_HIP)
    return hipblasZhemm(handle, side, uplo, m, n,
        (hipblasDoubleComplex*)&alpha, (hipblasDoubleComplex*)A, lda,
        (hipblasDoubleComplex*)B, ldb, (hipblasDoubleComplex*)&beta, (hipblasDoubleComplex*)C, ldc);
    #elif defined(DDLA_USE_CUDA)
    return cublasZhemm(handle, side, uplo, m, n,
        (cuDoubleComplex*)&alpha, (cuDoubleComplex*)A, lda,
        (cuDoubleComplex*)B, ldb, (cuDoubleComplex*)&beta, (cuDoubleComplex*)C, ldc);
    #else
    throw std::runtime_error("deblasHemm: no GPU vendor selected at compile time");
    #endif
}

} // namespace ddla

#endif // DDLA_HAS_GPU

namespace ddla {

template <DdlaBackend Backend, typename T>
void hemm(
    const DdlaHandle_t& handle,
    char side, char uplo,
    int m, int n,
    const T& alpha,
    const T* A, int lda,
    const T* B, int ldb,
    const T& beta,
    T* C, int ldc)
{
    static_assert(Backend != DdlaBackend::CPU || DDLA_HAS_CPU,
                  "CPU hemm is not available in this LibDDLA build");
    static_assert(Backend != DdlaBackend::GPU || DDLA_HAS_GPU,
                  "GPU hemm is not available in this LibDDLA build");

    if (handle == nullptr) {
        throw std::runtime_error("hemm: null handle");
    }
    const DdlaBackend actual = ddlaGetBackend(handle);
    if (actual != Backend) {
        throw std::runtime_error(
            std::string("hemm: template backend ") +
            (Backend == DdlaBackend::CPU ? "CPU" : "GPU") +
            " does not match handle backend " +
            (actual == DdlaBackend::CPU ? "CPU" : "GPU"));
    }

    if constexpr (Backend == DdlaBackend::CPU) {
#if DDLA_HAS_CPU
        // Real data: HEMM == SYMM; complex data calls xHEMM.  There is no
        // transpose flag in this family, so nothing needs normalising.
        if constexpr (std::is_same_v<T, float>) {
            const char s = side, u = uplo;
            ssymm_(&s, &u, &m, &n, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
        } else if constexpr (std::is_same_v<T, double>) {
            const char s = side, u = uplo;
            dsymm_(&s, &u, &m, &n, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
        } else if constexpr (std::is_same_v<T, std::complex<float>>) {
            const char s = side, u = uplo;
            chemm_(&s, &u, &m, &n, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
            const char s = side, u = uplo;
            zhemm_(&s, &u, &m, &n, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
        } else {
            static_assert(sizeof(T) == 0, "hemm: unsupported scalar type T");
        }
#else
        (void)handle; (void)side; (void)uplo; (void)m; (void)n;
        (void)alpha; (void)A; (void)lda; (void)B; (void)ldb;
        (void)beta; (void)C; (void)ldc;
#endif
    } else {
#if DDLA_HAS_GPU
        // The real types map onto xSYMM inside deblasHemm.
        if constexpr (std::is_same_v<T, float> ||
                      std::is_same_v<T, double> ||
                      std::is_same_v<T, std::complex<float>> ||
                      std::is_same_v<T, std::complex<double>>) {
            BLAS_CHECK(deblasHemm(handle->blasH,
                                  detail::to_side_mode(side),
                                  detail::to_fill_mode(uplo),
                                  m, n, alpha, A, lda, B, ldb, beta, C, ldc));
        } else {
            static_assert(sizeof(T) == 0, "hemm: unsupported scalar type T");
        }
#else
        (void)handle; (void)side; (void)uplo; (void)m; (void)n;
        (void)alpha; (void)A; (void)lda; (void)B; (void)ldb;
        (void)beta; (void)C; (void)ldc;
#endif
    }
}

#define INSTANTIATE_HEMM(BACKEND, TYPE)                                        \
    template void hemm<BACKEND, TYPE>(                                         \
        const DdlaHandle_t&, char, char, int, int,                             \
        const TYPE&, const TYPE*, int, const TYPE*, int,                       \
        const TYPE&, TYPE*, int)

#if DDLA_HAS_CPU
INSTANTIATE_HEMM(DdlaBackend::CPU, float);
INSTANTIATE_HEMM(DdlaBackend::CPU, double);
INSTANTIATE_HEMM(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_HEMM(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_HEMM(DdlaBackend::GPU, float);
INSTANTIATE_HEMM(DdlaBackend::GPU, double);
INSTANTIATE_HEMM(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_HEMM(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_HEMM

} // namespace ddla
