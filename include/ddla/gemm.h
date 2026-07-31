#ifndef GEMM_H
#define GEMM_H

#include "ddla_connector.h"
#include "ddla_handle_t.h"

namespace ddla{

/**
 * @brief Backend-neutral local GEMM.
 *
 * CPU specializations consume host pointers and call the linked BLAS. GPU
 * specializations consume device pointers and call cuBLAS or hipBLAS.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void gemm(
    const DdlaHandle_t& handle,
    char transa, char transb,
    int m, int n, int k,
    const T& alpha,
    const T* A, int lda,
    const T* B, int ldb,
    const T& beta,
    T* C, int ldc);

#ifdef DDLA_USE_CPU
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

inline deblasStatus_t deblasGemm(
    deblasHandle_t handle, deblasOperation_t transa, deblasOperation_t transb, 
    int m, int n, int k, 
    const float& alpha, 
    const float *A, int lda, 
    const float *B, int ldb,
    const float& beta,
    float *C, int ldc
)
{
#if defined(DDLA_USE_CUDA)
    return cublasSgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
#elif defined(DDLA_USE_HIP)
    return hipblasSgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
#elif defined(DDLA_USE_CPU)
    (void)handle;
    sgemm_(&transa, &transb, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
    return DEBLAS_STATUS_SUCCESS;
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}


inline deblasStatus_t deblasGemm(
    deblasHandle_t handle, deblasOperation_t transa, deblasOperation_t transb, 
    int m, int n, int k, 
    const double& alpha, 
    const double *A, int lda, 
    const double *B, int ldb,
    const double& beta,
    double *C, int ldc
)
{
#if defined(DDLA_USE_CUDA)
    return cublasDgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
#elif defined(DDLA_USE_HIP)
    return hipblasDgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
#elif defined(DDLA_USE_CPU)
    (void)handle;
    dgemm_(&transa, &transb, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
    return DEBLAS_STATUS_SUCCESS;
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}


inline deblasStatus_t deblasGemm(
    deblasHandle_t handle, deblasOperation_t transa, deblasOperation_t transb, 
    int m, int n, int k, 
    const std::complex<float>& alpha, 
    const std::complex<float> *A, int lda, 
    const std::complex<float> *B, int ldb,
    const std::complex<float>& beta,
    std::complex<float> *C, int ldc
)
{
#if defined(DDLA_USE_CUDA)
    return cublasCgemm(handle, transa, transb, m, n, k, (cuFloatComplex*)&alpha, (cuFloatComplex*)A, lda, (cuFloatComplex*)B, ldb, (cuFloatComplex*)&beta, (cuFloatComplex*)C, ldc);
#elif defined(DDLA_USE_HIP)
    return hipblasCgemm(handle, transa, transb, m, n, k, (hipblasComplex*)&alpha, (hipblasComplex*)A, lda, (hipblasComplex*)B, ldb, (hipblasComplex*)&beta, (hipblasComplex*)C, ldc);
#elif defined(DDLA_USE_CPU)
    (void)handle;
    cgemm_(&transa, &transb, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
    return DEBLAS_STATUS_SUCCESS;
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif

}

inline deblasStatus_t deblasGemm(
    deblasHandle_t handle, deblasOperation_t transa, deblasOperation_t transb, 
    int m, int n, int k, 
    const std::complex<double>& alpha, 
    const std::complex<double> *A, int lda, 
    const std::complex<double> *B, int ldb,
    const std::complex<double>& beta,
    std::complex<double> *C, int ldc
)
{
#if defined(DDLA_USE_CUDA)
    return cublasZgemm(handle, transa, transb, m, n, k, (cuDoubleComplex*)&alpha, (cuDoubleComplex*)A, lda, (cuDoubleComplex*)B, ldb, (cuDoubleComplex*)&beta, (cuDoubleComplex*)C, ldc);
#elif defined(DDLA_USE_HIP)
    return hipblasZgemm(handle, transa, transb, m, n, k, (hipblasDoubleComplex*)&alpha, (hipblasDoubleComplex*)A, lda, (hipblasDoubleComplex*)B, ldb, (hipblasDoubleComplex*)&beta, (hipblasDoubleComplex*)C, ldc);
#elif defined(DDLA_USE_CPU)
    (void)handle;
    zgemm_(&transa, &transb, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
    return DEBLAS_STATUS_SUCCESS;
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}


} // namespace ddla

#endif // GEMM_H
