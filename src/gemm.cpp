#include <ddla/gemm.h>

#include <complex>
#include <stdexcept>
#include <string>

#include "ddla_stream_impl.h"

#if DDLA_HAS_CPU
extern "C" {
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
#endif

namespace ddla {

#if DDLA_HAS_CPU
inline void cpu_gemm(
    char transa, char transb, int m, int n, int k,
    const float& alpha, const float* A, int lda,
    const float* B, int ldb, const float& beta, float* C, int ldc)
{
    sgemm_(&transa, &transb, &m, &n, &k,
           &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}

inline void cpu_gemm(
    char transa, char transb, int m, int n, int k,
    const double& alpha, const double* A, int lda,
    const double* B, int ldb, const double& beta, double* C, int ldc)
{
    dgemm_(&transa, &transb, &m, &n, &k,
           &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}

inline void cpu_gemm(
    char transa, char transb, int m, int n, int k,
    const std::complex<float>& alpha, const std::complex<float>* A, int lda,
    const std::complex<float>* B, int ldb,
    const std::complex<float>& beta, std::complex<float>* C, int ldc)
{
    cgemm_(&transa, &transb, &m, &n, &k,
           &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}

inline void cpu_gemm(
    char transa, char transb, int m, int n, int k,
    const std::complex<double>& alpha, const std::complex<double>* A, int lda,
    const std::complex<double>* B, int ldb,
    const std::complex<double>& beta, std::complex<double>* C, int ldc)
{
    zgemm_(&transa, &transb, &m, &n, &k,
           &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}
#endif

inline const char* backend_name(DdlaBackend backend)
{
    return backend == DdlaBackend::CPU ? "CPU" :
           backend == DdlaBackend::GPU ? "GPU" : "AUTO";
}

template <DdlaBackend Backend, typename T>
void gemm(
    const DdlaHandle_t& handle,
    char transa, char transb,
    int m, int n, int k,
    const T& alpha,
    const T* A, int lda,
    const T* B, int ldb,
    const T& beta,
    T* C, int ldc)
{
    static_assert(Backend == DdlaBackend::CPU || Backend == DdlaBackend::GPU,
                  "gemm backend must be DdlaBackend::CPU or DdlaBackend::GPU");
    static_assert(Backend != DdlaBackend::CPU || DDLA_HAS_CPU,
                  "CPU gemm is not available in this LibDDLA build");
    static_assert(Backend != DdlaBackend::GPU || DDLA_HAS_GPU,
                  "GPU gemm is not available in this LibDDLA build");

    if (handle == nullptr) {
        throw std::runtime_error("gemm: null handle");
    }
    const DdlaBackend actual = ddla_get_backend(handle);
    if (actual != Backend) {
        throw std::runtime_error(
            std::string("gemm: template backend ") + backend_name(Backend) +
            " does not match handle backend " + backend_name(actual));
    }

    if constexpr (Backend == DdlaBackend::CPU) {
#if DDLA_HAS_CPU
        cpu_gemm(transa, transb, m, n, k,
                 alpha, A, lda, B, ldb, beta, C, ldc);
#endif
    } else {
#if DDLA_HAS_GPU
        const deblasOperation_t opA = transa == 'N' ? DEBLAS_OP_N :
                                      transa == 'T' ? DEBLAS_OP_T : DEBLAS_OP_C;
        const deblasOperation_t opB = transb == 'N' ? DEBLAS_OP_N :
                                      transb == 'T' ? DEBLAS_OP_T : DEBLAS_OP_C;
        BLAS_CHECK(deblasGemm(
            handle->blasH, opA, opB,
            m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
#endif
    }
}

#define INSTANTIATE_GEMM(BACKEND, TYPE)                                      \
    template void gemm<BACKEND, TYPE>(                               \
        const DdlaHandle_t&, char, char, int, int, int,                       \
        const TYPE&, const TYPE*, int, const TYPE*, int,                      \
        const TYPE&, TYPE*, int)

#if DDLA_HAS_CPU
INSTANTIATE_GEMM(DdlaBackend::CPU, float);
INSTANTIATE_GEMM(DdlaBackend::CPU, double);
INSTANTIATE_GEMM(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_GEMM(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_GEMM(DdlaBackend::GPU, float);
INSTANTIATE_GEMM(DdlaBackend::GPU, double);
INSTANTIATE_GEMM(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_GEMM(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_GEMM

} // namespace ddla
