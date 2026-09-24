// ---------------------------------------------------------------------------
// Unified backend-templated symmetric rank-k update (C := alpha*A*A^T +
// beta*C  ('N')  or  C := alpha*A^T*A + beta*C).
//
// Structure follows src/herk.cpp exactly: one definition covering CPU (Fortran
// BLAS) and GPU (cuBLAS/hipBLAS), selected with `if constexpr` on the Backend
// template parameter, with the arms themselves guarded by the BUILD CAPABILITY
// macros (DDLA_HAS_CPU / DDLA_HAS_GPU) and the vendor chosen per scalar arm by
// the PER-TU macros (DDLA_USE_CUDA / DDLA_USE_HIP).
//
// Why the capability guard matters here: the previous design put a `#else`
// CPU arm inside src/syrk.h selected by `!defined(DDLA_USE_CUDA) &&
// !defined(DDLA_USE_HIP)`.  In a dual build (DDLA_USE_CPU + DDLA_USE_CUDA)
// every TU defines DDLA_USE_CUDA, so that arm was never compiled -- while
// DDLA_HAS_CPU was 1 and the CPU instantiation of this function still existed.
// A CPU-backend call would therefore have reached cublasSsyrk with HOST
// pointers.  Guarding on DDLA_HAS_CPU removes that class of bug.
//
// REAL SCALAR TYPES ONLY, as in the header this replaces: the callers route
// real data to xSYRK and the complex types to xHERK (which takes a REAL alpha
// and beta), so a complex instantiation here would have no user.  Instantiating
// a complex T is a compile-time error pointing at ddla::herk.
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
// Fortran declarations, guarded by the BUILD CAPABILITY rather than the
// per-TU vendor macro: a dual build defines DDLA_USE_CUDA everywhere while
// still instantiating the CPU templates, so `#if !defined(DDLA_USE_CUDA)`
// would omit these exactly where they are needed.
void ssyrk_(const char*, const char*, const int*, const int*,
            const float*, const float*, const int*, const float*, float*, const int*);
void dsyrk_(const char*, const char*, const int*, const int*,
            const double*, const double*, const int*, const double*, double*, const int*);
}
#endif

// ---------------------------------------------------------------------------
// GPU arm: the vendor calls.  Kept as free deblasSyrk overloads so the
// dispatch below stays a single BLAS_CHECK call per scalar type, exactly as
// ddla::gemm does it inline.  Private to this translation unit -- the public
// interface is the templated ddla::syrk declared in <ddla/ddla.h>.
//
// Only the two REAL overloads exist: xSYRK has no complex counterpart, and the
// complex rank-k updates go through ddla::herk / deblasHerk.
// ---------------------------------------------------------------------------
#if DDLA_HAS_GPU

namespace ddla {

inline deblasStatus_t deblasSyrk(
    deblasHandle_t handle,
    deblasFillMode_t uplo, deblasOperation_t trans,
    int n, int k,
    const float& alpha,
    const float* A, int lda,
    const float& beta,
    float* C, int ldc
)
{
    #if defined(DDLA_USE_HIP)
    return hipblasSsyrk(
        handle, uplo, trans, n, k,
        &alpha, A, lda,
        &beta, C, ldc
    );
    #elif defined(DDLA_USE_CUDA)
    return cublasSsyrk(
        handle, uplo, trans, n, k,
        &alpha, A, lda,
        &beta, C, ldc
    );
    #else
    // Neither vendor macro: this can only happen if the translation unit was
    // built without a GPU language while DDLA_HAS_GPU is set, which the build
    // system does not do (see the LANGUAGE CUDA/HIP source lists).  Throw
    // rather than fall through to a host pointer.
    throw std::runtime_error("deblasSyrk: no GPU vendor selected at compile time");
    #endif
}

inline deblasStatus_t deblasSyrk(
    deblasHandle_t handle,
    deblasFillMode_t uplo, deblasOperation_t trans,
    int n, int k,
    const double& alpha,
    const double* A, int lda,
    const double& beta,
    double* C, int ldc
)
{
    #if defined(DDLA_USE_HIP)
    return hipblasDsyrk(
        handle, uplo, trans, n, k,
        &alpha, A, lda,
        &beta, C, ldc
    );
    #elif defined(DDLA_USE_CUDA)
    return cublasDsyrk(
        handle, uplo, trans, n, k,
        &alpha, A, lda,
        &beta, C, ldc
    );
    #else
    throw std::runtime_error("deblasSyrk: no GPU vendor selected at compile time");
    #endif
}

} // namespace ddla

#endif // DDLA_HAS_GPU

namespace ddla {

template <DdlaBackend Backend, typename T>
void syrk(
    const DdlaHandle_t& handle,
    char uplo, char trans,
    int n, int k,
    const T& alpha,
    const T* A, int lda,
    const T& beta,
    T* C, int ldc)
{
    static_assert(Backend != DdlaBackend::CPU || DDLA_HAS_CPU,
                  "CPU syrk is not available in this LibDDLA build");
    static_assert(Backend != DdlaBackend::GPU || DDLA_HAS_GPU,
                  "GPU syrk is not available in this LibDDLA build");

    if (handle == nullptr) {
        throw std::runtime_error("syrk: null handle");
    }
    const DdlaBackend actual = ddlaGetBackend(handle);
    if (actual != Backend) {
        throw std::runtime_error(
            std::string("syrk: template backend ") +
            (Backend == DdlaBackend::CPU ? "CPU" : "GPU") +
            " does not match handle backend " +
            (actual == DdlaBackend::CPU ? "CPU" : "GPU"));
    }

    if constexpr (Backend == DdlaBackend::CPU) {
#if DDLA_HAS_CPU
        // A conjugate-transpose request is normalised to a plain transpose:
        // for real operands the two are identical and some vendor copies of
        // the underlying routine reject 'C'.
        if constexpr (std::is_same_v<T, float>) {
            const char u = uplo, t = detail::trans_real(trans);
            ssyrk_(&u, &t, &n, &k, &alpha, A, &lda, &beta, C, &ldc);
        } else if constexpr (std::is_same_v<T, double>) {
            const char u = uplo, t = detail::trans_real(trans);
            dsyrk_(&u, &t, &n, &k, &alpha, A, &lda, &beta, C, &ldc);
        } else {
            static_assert(sizeof(T) == 0, "syrk: complex scalar types should use herk");
        }
#else
        (void)handle; (void)uplo; (void)trans; (void)n; (void)k;
        (void)alpha; (void)A; (void)lda; (void)beta; (void)C; (void)ldc;
#endif
    } else {
#if DDLA_HAS_GPU
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
            BLAS_CHECK(deblasSyrk(handle->blasH,
                                  detail::to_fill_mode(uplo),
                                  detail::to_blas_op(detail::trans_real(trans)),
                                  n, k, alpha, A, lda, beta, C, ldc));
        } else {
            static_assert(sizeof(T) == 0, "syrk: complex scalar types should use herk");
        }
#else
        (void)handle; (void)uplo; (void)trans; (void)n; (void)k;
        (void)alpha; (void)A; (void)lda; (void)beta; (void)C; (void)ldc;
#endif
    }
}

#define INSTANTIATE_SYRK(BACKEND, TYPE)                                        \
    template void syrk<BACKEND, TYPE>(                                         \
        const DdlaHandle_t&, char, char, int, int,                             \
        const TYPE&, const TYPE*, int, const TYPE&, TYPE*, int)

// Real types only -- see the complex static_assert above.
#if DDLA_HAS_CPU
INSTANTIATE_SYRK(DdlaBackend::CPU, float);
INSTANTIATE_SYRK(DdlaBackend::CPU, double);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_SYRK(DdlaBackend::GPU, float);
INSTANTIATE_SYRK(DdlaBackend::GPU, double);
#endif

#undef INSTANTIATE_SYRK

} // namespace ddla
