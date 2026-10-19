#ifndef IAMAX_H
#define IAMAX_H

// Public <ddla/ddla.h> must be included before any redeclaration: it
// owns the public default arguments (Backend = default_backend_v), and
// hipcc/clang rejects redeclarations that introduce defaults later.
#include <ddla/ddla.h>
#include "ddla_connector.h"
#include <ddla/ddla_handle_t.h>

namespace ddla{

inline gpublasStatus_t gpublasIamax(gpublasHandle_t handle, int n, const float *x, int incx, int *result) {
#if DDLA_GPU_VENDOR
    return gpublasIsamax(handle, n, x, incx, result);
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}


inline gpublasStatus_t gpublasIamax(gpublasHandle_t handle, int n, const double *x, int incx, int *result) {
#if DDLA_GPU_VENDOR
    return gpublasIdamax(handle, n, x, incx, result);
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}

inline gpublasStatus_t gpublasIamax(gpublasHandle_t handle, int n, const std::complex<float> *x, int incx, int *result) {
#if DDLA_GPU_VENDOR
    return gpublasIcamax(handle, n, (gpublasComplex*)x, incx, result);
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}

inline gpublasStatus_t gpublasIamax(gpublasHandle_t handle, int n, const std::complex<double> *x, int incx, int *result) {
#if DDLA_GPU_VENDOR
    return gpublasIzamax(handle, n, (gpublasDoubleComplex*)x, incx, result);
#else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
#endif
}

/**
 * @brief Backend-neutral BLAS Level-1 iamax: 1-based index of the element
 * with largest magnitude (|Re| + |Im| for complex, matching cuBLAS/hipBLAS
 * iamax semantics).
 *
 * CPU specializations consume host pointers and run a plain vendor-neutral
 * host loop; GPU specializations consume device pointers and call
 * cuBLAS/hipBLAS via gpublasIamax.
 */
template <DdlaBackend Backend, typename T>
void iamax(const DdlaHandle_t& handle, int n, const T* x, int incx, int& result);

} // namespace ddla

#endif // IAMAX_H