// ---------------------------------------------------------------------------
// Unified backend-templated triangular solve (side):
//   side='L': B := alpha*op(A)^{-1}*B   (A is m-by-m)
//   side='R': B := alpha*B*op(A)^{-1}   (A is n-by-n)
//
// Structure follows src/herk.cpp exactly: one definition covering CPU (Fortran
// BLAS) and GPU (cuBLAS/hipBLAS), selected with `if constexpr` on the Backend
// template parameter, with the arms themselves guarded by the BUILD CAPABILITY
// macros (DDLA_HAS_CPU / DDLA_HAS_GPU) and the vendor chosen per scalar arm by
// the PER-TU macros (DDLA_USE_CUDA / DDLA_USE_HIP).
//
// Why the capability guard matters here: the previous design put a `#else`
// CPU arm inside src/trsm.h selected by `!defined(DDLA_USE_CUDA) &&
// !defined(DDLA_USE_HIP)`.  In a dual build (DDLA_USE_CPU + DDLA_USE_CUDA)
// every TU defines DDLA_USE_CUDA, so that arm was never compiled -- while
// DDLA_HAS_CPU was 1 and the CPU instantiation of this function still existed.
// A CPU-backend call would therefore have reached cublasZtrsm with HOST
// pointers.  Guarding on DDLA_HAS_CPU removes that class of bug.
//
// The public entry point (include/ddla/ddla.h) takes LAPACK-style characters,
// which the GPU arm converts with the blas_enums.h helpers; the CPU arm consumes
// the characters directly.
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
void strsm_(const char*, const char*, const char*, const char*, const int*, const int*,
            const float*, const float*, const int*, float*, const int*);
void dtrsm_(const char*, const char*, const char*, const char*, const int*, const int*,
            const double*, const double*, const int*, double*, const int*);
void ctrsm_(const char*, const char*, const char*, const char*, const int*, const int*,
            const std::complex<float>*, const std::complex<float>*, const int*,
            std::complex<float>*, const int*);
void ztrsm_(const char*, const char*, const char*, const char*, const int*, const int*,
            const std::complex<double>*, const std::complex<double>*, const int*,
            std::complex<double>*, const int*);
}
#endif

// ---------------------------------------------------------------------------
// GPU arm: the vendor calls.  Kept as free deblasTrsm overloads so the dispatch
// below stays a single BLAS_CHECK call per scalar type, exactly as
// ddla::gemm does it inline.  Private to this translation unit -- the public
// interface is the templated ddla::trsm declared in <ddla/ddla.h>.
//
// A and B are deliberately NON-const: the vendor prototypes take non-const
// pointers even though xTRSM only reads A and reads/writes B.
// ---------------------------------------------------------------------------
#if DDLA_HAS_GPU

namespace ddla {

inline deblasStatus_t deblasTrsm(
    deblasHandle_t handle, deblasSideMode_t side, deblasFillMode_t uplo, deblasOperation_t trans, deblasDiagType_t diag,
    int m, int n,
    const std::complex<double>& alpha,
    std::complex<double> *A, int lda,
    std::complex<double> *B, int ldb
)
{
    #if defined(DDLA_USE_HIP)
    return hipblasZtrsm(handle, side, uplo, trans, diag, m, n, (hipblasDoubleComplex*)&alpha, (hipblasDoubleComplex*)A, lda, (hipblasDoubleComplex*)B, ldb);
    #elif defined(DDLA_USE_CUDA)
    return cublasZtrsm(handle, side, uplo, trans, diag, m, n, (cuDoubleComplex*)&alpha, (cuDoubleComplex*)A, lda, (cuDoubleComplex*)B, ldb);
    #else
    // Neither vendor macro: this can only happen if the translation unit was
    // built without a GPU language while DDLA_HAS_GPU is set, which the build
    // system does not do (see the LANGUAGE CUDA/HIP source lists).  Throw
    // rather than fall through to a host pointer.
    throw std::runtime_error("deblasTrsm: no GPU vendor selected at compile time");
    #endif
}

inline deblasStatus_t deblasTrsm(
    deblasHandle_t handle, deblasSideMode_t side, deblasFillMode_t uplo, deblasOperation_t trans, deblasDiagType_t diag,
    int m, int n,
    const std::complex<float>& alpha,
    std::complex<float> *A, int lda,
    std::complex<float> *B, int ldb
)
{
    #if defined(DDLA_USE_HIP)
    return hipblasCtrsm(handle, side, uplo, trans, diag, m, n, (hipblasComplex*)&alpha, (hipblasComplex*)A, lda, (hipblasComplex*)B, ldb);
    #elif defined(DDLA_USE_CUDA)
    return cublasCtrsm(handle, side, uplo, trans, diag, m, n, (cuFloatComplex*)&alpha, (cuFloatComplex*)A, lda, (cuFloatComplex*)B, ldb);
    #else
    throw std::runtime_error("deblasTrsm: no GPU vendor selected at compile time");
    #endif
}

inline deblasStatus_t deblasTrsm(
    deblasHandle_t handle, deblasSideMode_t side, deblasFillMode_t uplo, deblasOperation_t trans, deblasDiagType_t diag,
    int m, int n,
    const float& alpha,
    float *A, int lda,
    float *B, int ldb
)
{
    #if defined(DDLA_USE_HIP)
    return hipblasStrsm(handle, side, uplo, trans, diag, m, n, &alpha, A, lda, B, ldb);
    #elif defined(DDLA_USE_CUDA)
    return cublasStrsm(handle, side, uplo, trans, diag, m, n, &alpha, A, lda, B, ldb);
    #else
    throw std::runtime_error("deblasTrsm: no GPU vendor selected at compile time");
    #endif
}

inline deblasStatus_t deblasTrsm(
    deblasHandle_t handle, deblasSideMode_t side, deblasFillMode_t uplo, deblasOperation_t trans, deblasDiagType_t diag,
    int m, int n,
    const double& alpha,
    double *A, int lda,
    double *B, int ldb
)
{
    #if defined(DDLA_USE_HIP)
    return hipblasDtrsm(handle, side, uplo, trans, diag, m, n, &alpha, A, lda, B, ldb);
    #elif defined(DDLA_USE_CUDA)
    return cublasDtrsm(handle, side, uplo, trans, diag, m, n, &alpha, A, lda, B, ldb);
    #else
    throw std::runtime_error("deblasTrsm: no GPU vendor selected at compile time");
    #endif
}

} // namespace ddla

#endif // DDLA_HAS_GPU

namespace ddla {

template <DdlaBackend Backend, typename T>
void trsm(
    const DdlaHandle_t& handle,
    char side, char uplo, char trans, char diag,
    int m, int n,
    const T& alpha,
    T* A, int lda,
    T* B, int ldb)
{
    static_assert(Backend != DdlaBackend::CPU || DDLA_HAS_CPU,
                  "CPU trsm is not available in this LibDDLA build");
    static_assert(Backend != DdlaBackend::GPU || DDLA_HAS_GPU,
                  "GPU trsm is not available in this LibDDLA build");

    if (handle == nullptr) {
        throw std::runtime_error("trsm: null handle");
    }
    const DdlaBackend actual = ddlaGetBackend(handle);
    if (actual != Backend) {
        throw std::runtime_error(
            std::string("trsm: template backend ") +
            (Backend == DdlaBackend::CPU ? "CPU" : "GPU") +
            " does not match handle backend " +
            (actual == DdlaBackend::CPU ? "CPU" : "GPU"));
    }

    if constexpr (Backend == DdlaBackend::CPU) {
#if DDLA_HAS_CPU
        // The Fortran entry points take the LAPACK characters directly, so no
        // conversion is needed on this path.
        if constexpr (std::is_same_v<T, float>) {
            strsm_(&side, &uplo, &trans, &diag, &m, &n, &alpha, A, &lda, B, &ldb);
        } else if constexpr (std::is_same_v<T, double>) {
            dtrsm_(&side, &uplo, &trans, &diag, &m, &n, &alpha, A, &lda, B, &ldb);
        } else if constexpr (std::is_same_v<T, std::complex<float>>) {
            ctrsm_(&side, &uplo, &trans, &diag, &m, &n, &alpha, A, &lda, B, &ldb);
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
            ztrsm_(&side, &uplo, &trans, &diag, &m, &n, &alpha, A, &lda, B, &ldb);
        } else {
            static_assert(sizeof(T) == 0, "trsm: unsupported scalar type T");
        }
#else
        (void)handle; (void)side; (void)uplo; (void)trans; (void)diag;
        (void)m; (void)n; (void)alpha; (void)A; (void)lda; (void)B; (void)ldb;
#endif
    } else {
#if DDLA_HAS_GPU
        if constexpr (std::is_same_v<T, float> ||
                      std::is_same_v<T, double> ||
                      std::is_same_v<T, std::complex<float>> ||
                      std::is_same_v<T, std::complex<double>>) {
            BLAS_CHECK(deblasTrsm(handle->blasH,
                                  detail::to_side_mode(side),
                                  detail::to_fill_mode(uplo),
                                  detail::to_blas_op(trans),
                                  detail::to_diag_type(diag),
                                  m, n, alpha, A, lda, B, ldb));
        } else {
            static_assert(sizeof(T) == 0, "trsm: unsupported scalar type T");
        }
#else
        (void)handle; (void)side; (void)uplo; (void)trans; (void)diag;
        (void)m; (void)n; (void)alpha; (void)A; (void)lda; (void)B; (void)ldb;
#endif
    }
}

#define INSTANTIATE_TRSM(BACKEND, TYPE)                                        \
    template void trsm<BACKEND, TYPE>(                                         \
        const DdlaHandle_t&, char, char, char, char, int, int,                 \
        const TYPE&, TYPE*, int, TYPE*, int)

#if DDLA_HAS_CPU
INSTANTIATE_TRSM(DdlaBackend::CPU, float);
INSTANTIATE_TRSM(DdlaBackend::CPU, double);
INSTANTIATE_TRSM(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_TRSM(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_TRSM(DdlaBackend::GPU, float);
INSTANTIATE_TRSM(DdlaBackend::GPU, double);
INSTANTIATE_TRSM(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_TRSM(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_TRSM

} // namespace ddla
