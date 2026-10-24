// ---------------------------------------------------------------------------
// Unified backend-templated triangular matrix multiply
// (C := alpha*op(A)*B or C := alpha*B*op(A), per @p side).
//
// Structure follows src/herk.cpp exactly: one definition covering CPU (Fortran
// BLAS) and GPU (cuBLAS/hipBLAS), selected with `if constexpr` on the Backend
// template parameter, with the arms themselves guarded by the BUILD CAPABILITY
// macros (DDLA_HAS_CPU / DDLA_HAS_GPU) and the vendor chosen per scalar arm by
// the PER-TU macros (DDLA_USE_CUDA / DDLA_USE_HIP).
//
// Why the capability guard matters here: the previous design put a `#else`
// CPU arm inside src/trmm.h selected by `!defined(DDLA_USE_CUDA) &&
// !defined(DDLA_USE_HIP)`.  In a dual build (DDLA_USE_CPU + DDLA_USE_CUDA)
// every TU defines DDLA_USE_CUDA, so that arm was never compiled -- while
// DDLA_HAS_CPU was 1 and the CPU instantiation of this function still existed.
// A CPU-backend call would therefore have reached cublasStrmm with HOST
// pointers.  Guarding on DDLA_HAS_CPU removes that class of bug.
//
// The vendor (cuBLAS/hipBLAS) TRMM has a separate C output while the Fortran
// BLAS TRMM overwrites B in place; the CPU arm copies B into C and multiplies
// in place there so both arms deliver C := alpha*op(A)*B with B untouched.
// ---------------------------------------------------------------------------
#include <ddla/ddla.h>
#include <ddla/ddla_handle_t.h>

#include <complex>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "blas_enums.h"
#include "ddla_connector.h"
#include "ddla_stream_impl.h"

#if DDLA_HAS_CPU
extern "C" {
void strmm_(const char*, const char*, const char*, const char*, const int*, const int*,
            const float*, const float*, const int*, float*, const int*);
void dtrmm_(const char*, const char*, const char*, const char*, const int*, const int*,
            const double*, const double*, const int*, double*, const int*);
void ctrmm_(const char*, const char*, const char*, const char*, const int*, const int*,
            const std::complex<float>*, const std::complex<float>*, const int*,
            std::complex<float>*, const int*);
void ztrmm_(const char*, const char*, const char*, const char*, const int*, const int*,
            const std::complex<double>*, const std::complex<double>*, const int*,
            std::complex<double>*, const int*);
}
#endif

// ---------------------------------------------------------------------------
// GPU arm: the vendor calls.  Kept as free gpublasTrmm overloads so the
// dispatch below stays a single BLAS_CHECK call per scalar type, exactly as
// ddla::gemm does it inline.  Private to this translation unit -- the public
// interface is the templated ddla::trmm declared in <ddla/ddla.h>.
// ---------------------------------------------------------------------------
#if DDLA_HAS_GPU

namespace ddla {

inline gpublasStatus_t gpublasTrmm(
    gpublasHandle_t handle, gpublasSideMode_t side, gpublasFillMode_t uplo,
    gpublasOperation_t trans, gpublasDiagType_t diag,
    int m, int n,
    const float& alpha,
    const float* A, int lda,
    float* B, int ldb,
    float* C, int ldc
)
{
#if DDLA_GPU_VENDOR
    return gpublasStrmm(handle, side, uplo, trans, diag, m, n, &alpha, A, lda, B, ldb, C, ldc);
#else
    // Neither vendor macro: this can only happen if the translation unit was
    // built without a GPU language while DDLA_HAS_GPU is set, which the build
    // system does not do (see the LANGUAGE CUDA/HIP source lists).  Throw
    // rather than fall through to a host pointer.
    throw std::runtime_error("gpublasTrmm: no GPU vendor selected at compile time");
#endif
}

inline gpublasStatus_t gpublasTrmm(
    gpublasHandle_t handle, gpublasSideMode_t side, gpublasFillMode_t uplo,
    gpublasOperation_t trans, gpublasDiagType_t diag,
    int m, int n,
    const double& alpha,
    const double* A, int lda,
    double* B, int ldb,
    double* C, int ldc
)
{
#if DDLA_GPU_VENDOR
    return gpublasDtrmm(handle, side, uplo, trans, diag, m, n, &alpha, A, lda, B, ldb, C, ldc);
#else
    throw std::runtime_error("gpublasTrmm: no GPU vendor selected at compile time");
#endif
}

inline gpublasStatus_t gpublasTrmm(
    gpublasHandle_t handle, gpublasSideMode_t side, gpublasFillMode_t uplo,
    gpublasOperation_t trans, gpublasDiagType_t diag,
    int m, int n,
    const std::complex<float>& alpha,
    const std::complex<float>* A, int lda,
    std::complex<float>* B, int ldb,
    std::complex<float>* C, int ldc
)
{
#if DDLA_GPU_VENDOR
    return gpublasCtrmm(handle, side, uplo, trans, diag, m, n,
                        reinterpret_cast<const gpublasComplex*>(&alpha),
                        reinterpret_cast<const gpublasComplex*>(A), lda,
                        reinterpret_cast<gpublasComplex*>(B), ldb,
                        reinterpret_cast<gpublasComplex*>(C), ldc);
#else
    throw std::runtime_error("gpublasTrmm: no GPU vendor selected at compile time");
#endif
}

inline gpublasStatus_t gpublasTrmm(
    gpublasHandle_t handle, gpublasSideMode_t side, gpublasFillMode_t uplo,
    gpublasOperation_t trans, gpublasDiagType_t diag,
    int m, int n,
    const std::complex<double>& alpha,
    const std::complex<double>* A, int lda,
    std::complex<double>* B, int ldb,
    std::complex<double>* C, int ldc
)
{
#if DDLA_GPU_VENDOR
    return gpublasZtrmm(handle, side, uplo, trans, diag, m, n,
                        reinterpret_cast<const gpublasDoubleComplex*>(&alpha),
                        reinterpret_cast<const gpublasDoubleComplex*>(A), lda,
                        reinterpret_cast<gpublasDoubleComplex*>(B), ldb,
                        reinterpret_cast<gpublasDoubleComplex*>(C), ldc);
#else
    throw std::runtime_error("gpublasTrmm: no GPU vendor selected at compile time");
#endif
}

} // namespace ddla

#endif // DDLA_HAS_GPU

namespace ddla {

template <DdlaBackend Backend, typename T>
void trmm(
    const DdlaHandle_t& handle,
    char side, char uplo, char trans, char diag,
    int m, int n,
    const T& alpha,
    const T* A, int lda,
    const T* B, int ldb,
    T* C, int ldc)
{
    static_assert(Backend != DdlaBackend::CPU || DDLA_HAS_CPU,
                  "CPU trmm is not available in this LibDDLA build");
    static_assert(Backend != DdlaBackend::GPU || DDLA_HAS_GPU,
                  "GPU trmm is not available in this LibDDLA build");

    if (handle == nullptr) {
        throw std::runtime_error("trmm: null handle");
    }
    const DdlaBackend actual = ddlaGetBackend(handle);
    if (actual != Backend) {
        throw std::runtime_error(
            std::string("trmm: template backend ") +
            (Backend == DdlaBackend::CPU ? "CPU" : "GPU") +
            " does not match handle backend " +
            (actual == DdlaBackend::CPU ? "CPU" : "GPU"));
    }

    if constexpr (Backend == DdlaBackend::CPU) {
#if DDLA_HAS_CPU
        // Fortran xTRMM multiplies in place, so stage B into C first and leave
        // B untouched -- that is the contract the vendor arm also delivers.
        // The characters go to the Fortran entry points unconverted.
        if constexpr (std::is_same_v<T, float>) {
            for (int j = 0; j < n; ++j)
                std::memcpy(C + static_cast<std::ptrdiff_t>(j) * ldc,
                            B + static_cast<std::ptrdiff_t>(j) * ldb,
                            sizeof(float) * m);
            strmm_(&side, &uplo, &trans, &diag, &m, &n, &alpha, A, &lda, C, &ldc);
        } else if constexpr (std::is_same_v<T, double>) {
            for (int j = 0; j < n; ++j)
                std::memcpy(C + static_cast<std::ptrdiff_t>(j) * ldc,
                            B + static_cast<std::ptrdiff_t>(j) * ldb,
                            sizeof(double) * m);
            dtrmm_(&side, &uplo, &trans, &diag, &m, &n, &alpha, A, &lda, C, &ldc);
        } else if constexpr (std::is_same_v<T, std::complex<float>>) {
            for (int j = 0; j < n; ++j)
                std::memcpy(C + static_cast<std::ptrdiff_t>(j) * ldc,
                            B + static_cast<std::ptrdiff_t>(j) * ldb,
                            sizeof(std::complex<float>) * m);
            ctrmm_(&side, &uplo, &trans, &diag, &m, &n, &alpha, A, &lda, C, &ldc);
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
            for (int j = 0; j < n; ++j)
                std::memcpy(C + static_cast<std::ptrdiff_t>(j) * ldc,
                            B + static_cast<std::ptrdiff_t>(j) * ldb,
                            sizeof(std::complex<double>) * m);
            ztrmm_(&side, &uplo, &trans, &diag, &m, &n, &alpha, A, &lda, C, &ldc);
        } else {
            static_assert(sizeof(T) == 0, "trmm: unsupported scalar type T");
        }
#else
        (void)handle; (void)side; (void)uplo; (void)trans; (void)diag;
        (void)m; (void)n; (void)alpha; (void)A; (void)lda;
        (void)B; (void)ldb; (void)C; (void)ldc;
#endif
    } else {
#if DDLA_HAS_GPU
        if constexpr (std::is_same_v<T, float> ||
                      std::is_same_v<T, double> ||
                      std::is_same_v<T, std::complex<float>> ||
                      std::is_same_v<T, std::complex<double>>) {
            // The vendor xTRMM prototypes take a non-const B pointer even
            // though they only read it, and the real gpublasTrmm overloads keep
            // that shape; the cast back is confined to this one call.
            T* B_mutable = const_cast<T*>(B);
            BLAS_CHECK(gpublasTrmm(handle->blasH,
                                   detail::to_side_mode(side),
                                   detail::to_fill_mode(uplo),
                                   detail::to_blas_op(trans),
                                   detail::to_diag_type(diag),
                                   m, n, alpha, A, lda,
                                   B_mutable, ldb, C, ldc));
        } else {
            static_assert(sizeof(T) == 0, "trmm: unsupported scalar type T");
        }
#else
        (void)handle; (void)side; (void)uplo; (void)trans; (void)diag;
        (void)m; (void)n; (void)alpha; (void)A; (void)lda;
        (void)B; (void)ldb; (void)C; (void)ldc;
#endif
    }
}

#define INSTANTIATE_TRMM(BACKEND, TYPE)                                        \
    template void trmm<BACKEND, TYPE>(                                         \
        const DdlaHandle_t&, char, char, char, char, int, int,                 \
        const TYPE&, const TYPE*, int, const TYPE*, int, TYPE*, int)

#if DDLA_HAS_CPU
INSTANTIATE_TRMM(DdlaBackend::CPU, float);
INSTANTIATE_TRMM(DdlaBackend::CPU, double);
INSTANTIATE_TRMM(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_TRMM(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_TRMM(DdlaBackend::GPU, float);
INSTANTIATE_TRMM(DdlaBackend::GPU, double);
INSTANTIATE_TRMM(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_TRMM(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_TRMM

} // namespace ddla
