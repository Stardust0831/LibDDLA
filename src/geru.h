#ifndef GERU_H
#define GERU_H

// Public <ddla/ddla.h> must be included before any redeclaration: it
// owns the public default arguments (Backend = default_backend_v), and
// hipcc/clang rejects redeclarations that introduce defaults later.
#include <ddla/ddla.h>
#include "ddla_connector.h"
#include <ddla/ddla_handle_t.h>

namespace ddla{

inline gpublasStatus_t gpublasGeru(gpublasHandle_t handle, int m, int n, const float& alpha, const float *x, int incx, const float *y, int incy, float *A, int lda) {
#if DDLA_GPU_VENDOR
    return gpublasSger(handle, m, n, &alpha, x, incx, y, incy, A, lda);
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}

inline gpublasStatus_t gpublasGeru(gpublasHandle_t handle, int m, int n, const double& alpha, const double *x, int incx, const double *y, int incy, double *A, int lda) {
#if DDLA_GPU_VENDOR
    return gpublasDger(handle, m, n, &alpha, x, incx, y, incy, A, lda);
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}

inline gpublasStatus_t gpublasGeru(gpublasHandle_t handle, int m, int n, const std::complex<float>& alpha, const std::complex<float> *x, int incx, const std::complex<float> *y, int incy, std::complex<float> *A, int lda) {
#if DDLA_GPU_VENDOR
    return gpublasCgeru(handle, m, n, (gpublasComplex*)&alpha, (gpublasComplex*)x, incx, (gpublasComplex*)y, incy, (gpublasComplex*)A, lda);
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}

inline gpublasStatus_t gpublasGeru(gpublasHandle_t handle, int m, int n, const std::complex<double>& alpha, const std::complex<double> *x, int incx, const std::complex<double> *y, int incy, std::complex<double> *A, int lda) {
#if DDLA_GPU_VENDOR
    return gpublasZgeru(handle, m, n, (gpublasDoubleComplex*)&alpha, (gpublasDoubleComplex*)x, incx, (gpublasDoubleComplex*)y, incy, (gpublasDoubleComplex*)A, lda);
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}

/**
 * @brief Backend-neutral BLAS Level-2 geru: A := alpha * x * y^T + A
 * (unconjugated outer product).
 *
 * CPU specializations consume host pointers and run a plain vendor-neutral
 * host loop; GPU specializations consume device pointers and call
 * cuBLAS/hipBLAS via gpublasGeru.
 */
template <DdlaBackend Backend, typename T>
void geru(const DdlaHandle_t& handle, int m, int n, const T& alpha,
          const T* x, int incx, const T* y, int incy, T* A, int lda);

} // namespace ddla

#endif // GERU_H