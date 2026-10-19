#ifndef AXPY_H
#define AXPY_H

// Public <ddla/ddla.h> must be included before any redeclaration: it
// owns the public default arguments (Backend = default_backend_v), and
// hipcc/clang rejects redeclarations that introduce defaults later.
#include <ddla/ddla.h>
#include "ddla_connector.h"
#include <ddla/ddla_handle_t.h>

namespace ddla{

inline gpublasStatus_t gpublasAxpy(gpublasHandle_t handle, const int64_t& n, const float& alpha, const float *x, int incx, float *y, int incy) {
#if DDLA_GPU_VENDOR
    return gpublasSaxpy(handle, n, &alpha, x, incx, y, incy);
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}

inline gpublasStatus_t gpublasAxpy(gpublasHandle_t handle, const int64_t& n, const double& alpha, const double *x, int incx, double *y, int incy) {
#if DDLA_GPU_VENDOR
    return gpublasDaxpy(handle, n, &alpha, x, incx, y, incy);
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}

inline gpublasStatus_t gpublasAxpy(gpublasHandle_t handle, const int64_t& n, const std::complex<float>& alpha, const std::complex<float> *x, int incx, std::complex<float> *y, int incy) {
#if DDLA_GPU_VENDOR
    return gpublasCaxpy(handle, n, (gpublasComplex*)&alpha, (gpublasComplex*)x, incx, (gpublasComplex*)y, incy);
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}

inline gpublasStatus_t gpublasAxpy(gpublasHandle_t handle, const int64_t& n, const std::complex<double>& alpha, const std::complex<double> *x, int incx, std::complex<double> *y, int incy) {
#if DDLA_GPU_VENDOR
    return gpublasZaxpy(handle, n, (gpublasDoubleComplex*)&alpha, (gpublasDoubleComplex*)x, incx, (gpublasDoubleComplex*)y, incy);
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}


/**
 * @brief Backend-neutral BLAS Level-1 axpy: y := alpha * x + y.
 *
 * CPU specializations consume host pointers and run a plain vendor-neutral
 * host loop; GPU specializations consume device pointers and call
 * cuBLAS/hipBLAS via gpublasAxpy. Mirrors ddla::scal's shape.
 */
template <DdlaBackend Backend, typename T>
void axpy(const DdlaHandle_t& handle, int n, const T& alpha,
          const T* x, int incx, T* y, int incy);

} // namespace ddla

#endif // AXPY_H
