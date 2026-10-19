#ifndef SWAP_H
#define SWAP_H

#include "ddla_connector.h"

namespace ddla{
    inline gpublasStatus_t gpublasSwap(gpublasHandle_t handle, int n, std::complex<double> *x, int incx, std::complex<double> *y, int incy)
    {
        #if DDLA_GPU_VENDOR
        return gpublasZswap(handle, n, (gpublasDoubleComplex*)x, incx, (gpublasDoubleComplex*)y, incy);
        #else
        throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
        #endif
    }

    inline gpublasStatus_t gpublasSwap(gpublasHandle_t handle, int n, std::complex<float> *x, int incx, std::complex<float> *y, int incy)
    {
        #if DDLA_GPU_VENDOR
        return gpublasCswap(handle, n, (gpublasComplex*)x, incx, (gpublasComplex*)y, incy);
        #else
        throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
        #endif
    }

    inline gpublasStatus_t gpublasSwap(gpublasHandle_t handle, int n, float *x, int incx, float *y, int incy)
    {
        #if DDLA_GPU_VENDOR
        return gpublasSswap(handle, n, x, incx, y, incy);
        #else
        throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
        #endif
    }

    inline gpublasStatus_t gpublasSwap(gpublasHandle_t handle, int n, double *x, int incx, double *y, int incy)
    {
        #if DDLA_GPU_VENDOR
        return gpublasDswap(handle, n, x, incx, y, incy);
        #else
        throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
        #endif
    }
}// DDLA


#endif // SWAP_H