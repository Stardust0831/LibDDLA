#ifndef LASWP_H
#define LASWP_H

#include "ddla_connector.h"
#include <stdexcept>
#include <vector>
#include "swap.h"

namespace ddla{

#if GPU_HAS_SOLVER_LASWP

inline gpusolverStatus_t gpusolverLaswp(
    gpusolverHandle_t handle,
    int n,
    double *A,
    int lda,
    int k1,
    int k2,
    const int *ipiv,
    int incx
)
{
    return gpusolverDlaswp(handle, n, A, lda, k1, k2, ipiv, incx);
}


inline gpusolverStatus_t gpusolverLaswp(
    gpusolverHandle_t handle,
    int n,
    float *A,
    int lda,
    int k1,
    int k2,
    const int *ipiv,
    int incx
)
{
    return gpusolverSlaswp(handle, n, A, lda, k1, k2, ipiv, incx);
}

inline gpusolverStatus_t gpusolverLaswp(
    gpusolverHandle_t handle,
    int n,
    std::complex<double> *A,
    int lda,
    int k1,
    int k2,
    const int *ipiv,
    int incx
)
{
    return gpusolverZlaswp(handle, n, (gpusolverDoubleComplex*)A, lda, k1, k2, ipiv, incx);
}

inline gpusolverStatus_t gpusolverLaswp(
    gpusolverHandle_t handle,
    int n,
    std::complex<float> *A,
    int lda,
    int k1,
    int k2,
    const int *ipiv,
    int incx
)
{
    return gpusolverClaswp(handle, n, (gpusolverComplex*)A, lda, k1, k2, ipiv, incx);
}
#endif // GPU_HAS_SOLVER_LASWP

/**
 * @brief Apply a series of row interchanges to matrix A based on ipiv from getrf.
 *
 * @note cuSOLVER and hipSOLVER do not provide a native laswp routine.
 *       This implementation uses gpublasSwap (cublas/hipblas swap) to exchange
 *       rows one-by-one.
 * @note ipiv is assumed to be a **device** pointer (consistent with the output
 *       of gpusolverGetrf).  It is copied to host internally before applying swaps.
 * @note k1, k2 and ipiv entries follow the 1-based convention of LAPACK/cuSOLVER.
 *
 * @tparam T      Scalar type (float, double, complex<float>, complex<double>).
 * @param handle  BLAS handle (gpublasHandle_t) needed for swap operations.
 * @param n       Number of columns of A (length of each row).
 * @param A       Device pointer to column-major matrix A.
 * @param lda     Leading dimension of A.
 * @param k1      First row index (1-based) to pivot.
 * @param k2      Last row index (1-based) to pivot.
 * @param ipiv    Device array of pivot indices (1-based), length >= (k2-k1)*|incx|+1.
 * @param incx    Increment between successive entries of ipiv.
 * @return gpublasStatus_t  Success or the first swap error encountered.
 */
template<typename T>
inline gpublasStatus_t gpublasLaswp(
    gpublasHandle_t handle,
    int n,
    T *A,
    int lda,
    int k1,
    int k2,
    const int *ipiv, //device
    int incx
)
{
    if (n <= 0 || k1 > k2)
        return GPUBLAS_STATUS_SUCCESS;

    int n_pivots = k2 - k1 + 1;
    std::vector<int> host_ipiv(n_pivots);
#if DDLA_GPU_VENDOR
    GPU_CHECK(gpuMemcpy(host_ipiv.data(), ipiv + (k1 - 1) * incx, n_pivots * sizeof(int), gpuMemcpyDeviceToHost));
#else
    throw std::runtime_error("gpublasLaswp: needs a GPU build");
#endif

    gpublasStatus_t status = GPUBLAS_STATUS_SUCCESS;
    for (int i = 0; i < n_pivots; ++i) {
        int current_row = k1 + i;   // 1-based
        int piv_row     = host_ipiv[i]; // 1-based
        if (piv_row != current_row) {
            status = gpublasSwap(handle, n, A + current_row - 1, lda, A + piv_row - 1, lda);
            if (status != GPUBLAS_STATUS_SUCCESS)
                break;
        }
    }
    return status;
}



}

#endif
