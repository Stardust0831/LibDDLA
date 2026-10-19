#ifndef SCAL_H
#define SCAL_H

// Public <ddla/ddla.h> must be included before any redeclaration: it
// owns the public default arguments (Backend = default_backend_v), and
// hipcc/clang rejects redeclarations that introduce defaults later.
#include <ddla/ddla.h>
#include "ddla_connector.h"
#include <ddla/ddla_handle_t.h>

namespace ddla{

inline gpublasStatus_t gpublasScal(gpublasHandle_t handle, int64_t n, const float& alpha, float *x, int64_t incx)
{
    #if DDLA_GPU_VENDOR
    return gpublasSscal(handle, n, &alpha, x, incx);
    #else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
    #endif
}

inline gpublasStatus_t gpublasScal(gpublasHandle_t handle, int64_t n, const double& alpha, double *x, int64_t incx)
{
#if DDLA_GPU_VENDOR
    return gpublasDscal(handle, n, &alpha, x, incx);
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}

inline gpublasStatus_t gpublasScal(gpublasHandle_t handle, int64_t n, const float& alpha, std::complex<float> *x, int64_t incx)
{
#if DDLA_GPU_VENDOR
    return gpublasCsscal(handle, n, &alpha, (gpublasComplex*)x, incx);
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}

inline gpublasStatus_t gpublasScal(gpublasHandle_t handle, int64_t n, const double& alpha, std::complex<double> *x, int64_t incx)
{
#if DDLA_GPU_VENDOR
    return gpublasZdscal(handle, n, &alpha, (gpublasDoubleComplex*)x, incx);
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}

inline gpublasStatus_t gpublasScal(gpublasHandle_t handle, int64_t n, const std::complex<float>& alpha, std::complex<float> *x, int64_t incx) {
#if DDLA_GPU_VENDOR
    return gpublasCscal(handle, n, (gpublasComplex*)&alpha, (gpublasComplex*)x, incx);
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}

inline gpublasStatus_t gpublasScal(gpublasHandle_t handle, int64_t n, const std::complex<double>& alpha, std::complex<double> *x, int64_t incx) {
#if DDLA_GPU_VENDOR
    return gpublasZscal(handle, n, (gpublasDoubleComplex*)&alpha, (gpublasDoubleComplex*)x, incx);
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}

/**
 * @brief Backend-neutral BLAS Level-1 scale: x := alpha * x.
 *
 * Mirrors ddla::gemm's shape: one function, backend chosen by the template
 * argument, `if constexpr` dispatch inside. CPU specializations consume host
 * pointers and call the linked BLAS (?scal); GPU specializations consume
 * device pointers and call cuBLAS/hipBLAS via gpublasScal.
 *
 * Note: this covers the same-type-alpha case (alpha and x share type T).
 * The mixed real-alpha/complex-x variants (csscal/zdscal) are still reached
 * only through the raw gpublasScal overloads above.
 */
template <DdlaBackend Backend, typename T>
void scal(const DdlaHandle_t& handle, int n, const T& alpha, T* x, int incx);

} // namespace ddla

#endif // SCAL_H
