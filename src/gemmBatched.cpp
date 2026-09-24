// ---------------------------------------------------------------------------
// Unified backend-templated batched matrix multiply over arrays of pointers:
// for i in [0, batchCount):  C[i] := alpha*op(A[i])*op(B[i]) + beta*C[i].
//
// Structure follows src/herk.cpp exactly: one definition covering CPU (Fortran
// BLAS) and GPU (cuBLAS/hipBLAS), selected with `if constexpr` on the Backend
// template parameter, with the arms themselves guarded by the BUILD CAPABILITY
// macros (DDLA_HAS_CPU / DDLA_HAS_GPU) and the vendor chosen per scalar arm by
// the PER-TU macros (DDLA_USE_CUDA / DDLA_USE_HIP).
//
// Why the capability guard matters here: the previous design put a `#else`
// CPU arm inside src/gemmBatched.h selected by `!defined(DDLA_USE_CUDA) &&
// !defined(DDLA_USE_HIP)`.  In a dual build (DDLA_USE_CPU + DDLA_USE_CUDA)
// every TU defines DDLA_USE_CUDA, so that arm was never compiled -- while
// DDLA_HAS_CPU was 1 and the CPU instantiation of this function still existed.
// A CPU-backend call would therefore have reached cublasZgemmBatched with HOST
// pointers.  Guarding on DDLA_HAS_CPU removes that class of bug.
//
// The array-of-pointer arguments keep the `const T* const Aarray[]` shape: the
// pointers themselves are read-only to this wrapper (the vendor prototypes and
// the Fortran declaration differ on constness, which is why the GPU arm casts
// -- exactly as the pre-unification wrapper did).
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
// Fortran BLAS entry points for the CPU arm below.  Guarded by the BUILD
// CAPABILITY (DDLA_HAS_CPU), not the per-TU vendor macro: in a dual build
// every TU defines DDLA_USE_CUDA/DDLA_USE_HIP while still instantiating the
// CPU templates, so `#if !defined(DDLA_USE_CUDA)` would leave these
// declarations out exactly where they are needed.
void sgemm_(const char*, const char*, const int*, const int*, const int*,
            const float*, const float*, const int*, const float*, const int*,
            const float*, float*, const int*);
void dgemm_(const char*, const char*, const int*, const int*, const int*,
            const double*, const double*, const int*, const double*, const int*,
            const double*, double*, const int*);
void cgemm_(const char*, const char*, const int*, const int*, const int*,
            const std::complex<float>*, const std::complex<float>*, const int*,
            const std::complex<float>*, const int*,
            const std::complex<float>*, std::complex<float>*, const int*);
void zgemm_(const char*, const char*, const int*, const int*, const int*,
            const std::complex<double>*, const std::complex<double>*, const int*,
            const std::complex<double>*, const int*,
            const std::complex<double>*, std::complex<double>*, const int*);
}
#endif // DDLA_HAS_CPU

// ---------------------------------------------------------------------------
// GPU arm: the vendor calls.  Kept as free deblasGemmBatched overloads so the
// dispatch below stays a single BLAS_CHECK call per scalar type, exactly as
// ddla::gemm does it inline.  Private to this translation unit -- the public
// interface is the templated ddla::gemmBatched declared in <ddla/ddla.h>.
// ---------------------------------------------------------------------------
#if DDLA_HAS_GPU

namespace ddla {

inline deblasStatus_t deblasGemmBatched(
    deblasHandle_t handle, deblasOperation_t transa, deblasOperation_t transb,
    int m, int n, int k,
    const std::complex<double>& alpha,
    const std::complex<double> *const Aarray[], int lda,
    const std::complex<double> *const Barray[], int ldb,
    const std::complex<double>& beta,
    std::complex<double> *const Carray[], int ldc,
    int batchCount
)
{
    #if defined(DDLA_USE_HIP)
    return hipblasZgemmBatched(
        handle, transa, transb, m, n, k,
        (hipblasDoubleComplex*)(&alpha),
        (hipblasDoubleComplex**)(Aarray), lda,
        (hipblasDoubleComplex**)(Barray), ldb,
        (hipblasDoubleComplex*)(&beta),
        (hipblasDoubleComplex**)(Carray), ldc,
        batchCount
    );
    #elif defined(DDLA_USE_CUDA)
    return cublasZgemmBatched(
        handle, transa, transb, m, n, k,
        (cuDoubleComplex*)(&alpha),
        (cuDoubleComplex**)(Aarray), lda,
        (cuDoubleComplex**)(Barray), ldb,
        (cuDoubleComplex*)(&beta),
        (cuDoubleComplex**)(Carray), ldc,
        batchCount
    );
    #else
    // Neither vendor macro: this can only happen if the translation unit was
    // built without a GPU language while DDLA_HAS_GPU is set, which the build
    // system does not do (see the LANGUAGE CUDA/HIP source lists).  Throw
    // rather than fall through to a host pointer.
    throw std::runtime_error("deblasGemmBatched: no GPU vendor selected at compile time");
    #endif
}

inline deblasStatus_t deblasGemmBatched(
    deblasHandle_t handle, deblasOperation_t transa, deblasOperation_t transb,
    int m, int n, int k,
    const std::complex<float>& alpha,
    const std::complex<float> *const Aarray[], int lda,
    const std::complex<float> *const Barray[], int ldb,
    const std::complex<float>& beta,
    std::complex<float> *const Carray[], int ldc,
    int batchCount
)
{
    #if defined(DDLA_USE_HIP)
    return hipblasCgemmBatched(
        handle, transa, transb, m, n, k,
        (hipblasComplex*)(&alpha),
        (hipblasComplex**)(Aarray), lda,
        (hipblasComplex**)(Barray), ldb,
        (hipblasComplex*)(&beta),
        (hipblasComplex**)(Carray), ldc,
        batchCount
    );
    #elif defined(DDLA_USE_CUDA)
    return cublasCgemmBatched(
        handle, transa, transb, m, n, k,
        (cuFloatComplex*)(&alpha),
        (cuFloatComplex**)(Aarray), lda,
        (cuFloatComplex**)(Barray), ldb,
        (cuFloatComplex*)(&beta),
        (cuFloatComplex**)(Carray), ldc,
        batchCount
    );
    #else
    throw std::runtime_error("deblasGemmBatched: no GPU vendor selected at compile time");
    #endif
}

inline deblasStatus_t deblasGemmBatched(
    deblasHandle_t handle, deblasOperation_t transa, deblasOperation_t transb,
    int m, int n, int k,
    const float& alpha,
    const float *const Aarray[], int lda,
    const float *const Barray[], int ldb,
    const float& beta,
    float *const Carray[], int ldc,
    int batchCount
)
{
    #if defined(DDLA_USE_HIP)
    return hipblasSgemmBatched(
        handle, transa, transb, m, n, k,
        &alpha,
        Aarray, lda,
        Barray, ldb,
        &beta,
        Carray, ldc,
        batchCount
    );
    #elif defined(DDLA_USE_CUDA)
    return cublasSgemmBatched(
        handle, transa, transb, m, n, k,
        &alpha,
        Aarray, lda,
        Barray, ldb,
        &beta,
        Carray, ldc,
        batchCount
    );
    #else
    throw std::runtime_error("deblasGemmBatched: no GPU vendor selected at compile time");
    #endif
}

inline deblasStatus_t deblasGemmBatched(
    deblasHandle_t handle, deblasOperation_t transa, deblasOperation_t transb,
    int m, int n, int k,
    const double& alpha,
    const double *const Aarray[], int lda,
    const double *const Barray[], int ldb,
    const double& beta,
    double *const Carray[], int ldc,
    int batchCount
)
{
    #if defined(DDLA_USE_HIP)
    return hipblasDgemmBatched(
        handle, transa, transb, m, n, k,
        &alpha,
        Aarray, lda,
        Barray, ldb,
        &beta,
        Carray, ldc,
        batchCount
    );
    #elif defined(DDLA_USE_CUDA)
    return cublasDgemmBatched(
        handle, transa, transb, m, n, k,
        &alpha,
        Aarray, lda,
        Barray, ldb,
        &beta,
        Carray, ldc,
        batchCount
    );
    #else
    throw std::runtime_error("deblasGemmBatched: no GPU vendor selected at compile time");
    #endif
}

} // namespace ddla

#endif // DDLA_HAS_GPU

namespace ddla {

template <DdlaBackend Backend, typename T>
void gemmBatched(
    const DdlaHandle_t& handle,
    char transa, char transb,
    int m, int n, int k,
    const T& alpha,
    const T* const Aarray[], int lda,
    const T* const Barray[], int ldb,
    const T& beta,
    T* const Carray[], int ldc,
    int batchCount)
{
    static_assert(Backend != DdlaBackend::CPU || DDLA_HAS_CPU,
                  "CPU gemmBatched is not available in this LibDDLA build");
    static_assert(Backend != DdlaBackend::GPU || DDLA_HAS_GPU,
                  "GPU gemmBatched is not available in this LibDDLA build");

    if (handle == nullptr) {
        throw std::runtime_error("gemmBatched: null handle");
    }
    const DdlaBackend actual = ddlaGetBackend(handle);
    if (actual != Backend) {
        throw std::runtime_error(
            std::string("gemmBatched: template backend ") +
            (Backend == DdlaBackend::CPU ? "CPU" : "GPU") +
            " does not match handle backend " +
            (actual == DdlaBackend::CPU ? "CPU" : "GPU"));
    }

    if constexpr (Backend == DdlaBackend::CPU) {
#if DDLA_HAS_CPU
        // Fortran BLAS has no batched entry point, so the CPU arm is an
        // explicit loop over the per-matrix gemm calls.  The characters go to
        // the Fortran entry points unconverted.
        if constexpr (std::is_same_v<T, float>) {
            for (int i = 0; i < batchCount; ++i)
                sgemm_(&transa, &transb, &m, &n, &k, &alpha, Aarray[i], &lda,
                       Barray[i], &ldb, &beta, Carray[i], &ldc);
        } else if constexpr (std::is_same_v<T, double>) {
            for (int i = 0; i < batchCount; ++i)
                dgemm_(&transa, &transb, &m, &n, &k, &alpha, Aarray[i], &lda,
                       Barray[i], &ldb, &beta, Carray[i], &ldc);
        } else if constexpr (std::is_same_v<T, std::complex<float>>) {
            for (int i = 0; i < batchCount; ++i)
                cgemm_(&transa, &transb, &m, &n, &k, &alpha, Aarray[i], &lda,
                       Barray[i], &ldb, &beta, Carray[i], &ldc);
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
            for (int i = 0; i < batchCount; ++i)
                zgemm_(&transa, &transb, &m, &n, &k, &alpha, Aarray[i], &lda,
                       Barray[i], &ldb, &beta, Carray[i], &ldc);
        } else {
            static_assert(sizeof(T) == 0, "gemmBatched: unsupported scalar type T");
        }
#else
        (void)handle; (void)transa; (void)transb; (void)m; (void)n; (void)k;
        (void)alpha; (void)Aarray; (void)lda; (void)Barray; (void)ldb;
        (void)beta; (void)Carray; (void)ldc; (void)batchCount;
#endif
    } else {
#if DDLA_HAS_GPU
        if constexpr (std::is_same_v<T, float> ||
                      std::is_same_v<T, double> ||
                      std::is_same_v<T, std::complex<float>> ||
                      std::is_same_v<T, std::complex<double>>) {
            BLAS_CHECK(deblasGemmBatched(handle->blasH,
                                         detail::to_blas_op(transa),
                                         detail::to_blas_op(transb),
                                         m, n, k, alpha,
                                         Aarray, lda, Barray, ldb, beta,
                                         Carray, ldc, batchCount));
        } else {
            static_assert(sizeof(T) == 0, "gemmBatched: unsupported scalar type T");
        }
#else
        (void)handle; (void)transa; (void)transb; (void)m; (void)n; (void)k;
        (void)alpha; (void)Aarray; (void)lda; (void)Barray; (void)ldb;
        (void)beta; (void)Carray; (void)ldc; (void)batchCount;
#endif
    }
}

#define INSTANTIATE_GEMM_BATCHED(BACKEND, TYPE)                                \
    template void gemmBatched<BACKEND, TYPE>(                                  \
        const DdlaHandle_t&, char, char, int, int, int,                        \
        const TYPE&, const TYPE* const[], int, const TYPE* const[], int,       \
        const TYPE&, TYPE* const[], int, int)

#if DDLA_HAS_CPU
INSTANTIATE_GEMM_BATCHED(DdlaBackend::CPU, float);
INSTANTIATE_GEMM_BATCHED(DdlaBackend::CPU, double);
INSTANTIATE_GEMM_BATCHED(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_GEMM_BATCHED(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_GEMM_BATCHED(DdlaBackend::GPU, float);
INSTANTIATE_GEMM_BATCHED(DdlaBackend::GPU, double);
INSTANTIATE_GEMM_BATCHED(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_GEMM_BATCHED(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_GEMM_BATCHED

} // namespace ddla
