#ifndef GETRF_H
#define GETRF_H

#include "ddla_connector.h"
#include <stdexcept>

namespace ddla{

inline gpusolverStatus_t gpusolverGetrf(
    gpusolverHandle_t handle,
    int m,
    int n,
    double *A,
    int lda,
    int *devIpiv,
    int *devInfo
)
{
    double *Workspace = nullptr;
    int Lwork;

    gpuStream_t stream;
    SOLVER_CHECK(gpusolverGetStream(handle, &stream));

    #if DDLA_GPU_VENDOR
    SOLVER_CHECK(gpusolverDgetrf_bufferSize(handle, m, n, A, lda, &Lwork));
    #else
    throw std::runtime_error("not ENABLE CUDA and ENABLE HIP\n");
    #endif
    
    #if DDLA_GPU_VENDOR
    if(Lwork > 0)
        GPU_CHECK(gpuMallocAsync(reinterpret_cast<void**>(&Workspace), Lwork * sizeof(double), stream));
    #endif

    gpusolverStatus_t status{};
    #if DDLA_GPU_VENDOR
    status = gpusolverDgetrf(handle, m, n, A, lda, Workspace, Lwork, devIpiv, devInfo);
    #else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
    #endif

    #if DDLA_GPU_VENDOR
    if(Lwork > 0)
        GPU_CHECK(gpuFreeAsync(Workspace, stream));
    #endif
    return status;
}

inline gpusolverStatus_t gpusolverGetrf(
    gpusolverHandle_t handle,
    int m,
    int n,
    float *A,
    int lda,
    int *devIpiv,
    int *devInfo
)
{
    float *Workspace = nullptr;
    int Lwork;

    gpuStream_t stream;
    SOLVER_CHECK(gpusolverGetStream(handle, &stream));

    #if DDLA_GPU_VENDOR
    SOLVER_CHECK(gpusolverSgetrf_bufferSize(handle, m, n, A, lda, &Lwork));
    #else
    throw std::runtime_error("not ENABLE CUDA and ENABLE HIP\n");
    #endif

    #if DDLA_GPU_VENDOR
    if(Lwork > 0)
        GPU_CHECK(gpuMallocAsync(reinterpret_cast<void**>(&Workspace), Lwork * sizeof(float), stream));
    #endif

    gpusolverStatus_t status{};
    #if DDLA_GPU_VENDOR
    status = gpusolverSgetrf(handle, m, n, A, lda, Workspace, Lwork, devIpiv, devInfo);
    #else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
    #endif

    #if DDLA_GPU_VENDOR
    if(Lwork > 0)
        GPU_CHECK(gpuFreeAsync(Workspace, stream));
    #endif
    return status;
}

inline gpusolverStatus_t gpusolverGetrf(
    gpusolverHandle_t handle,
    int m,
    int n,
    std::complex<double> *A,
    int lda,
    int *devIpiv,
    int *devInfo
)
{
    std::complex<double> *Workspace = nullptr;
    int Lwork;

    gpuStream_t stream;
    SOLVER_CHECK(gpusolverGetStream(handle, &stream));

    #if DDLA_GPU_VENDOR
    SOLVER_CHECK(gpusolverZgetrf_bufferSize(handle, m, n, (gpusolverDoubleComplex*)A, lda, &Lwork));
    #else
    throw std::runtime_error("not ENABLE CUDA and ENABLE HIP\n");
    #endif
    #if DDLA_GPU_VENDOR
    if(Lwork > 0)
        GPU_CHECK(gpuMallocAsync(reinterpret_cast<void**>(&Workspace), Lwork * sizeof(std::complex<double>), stream));
    #endif

    gpusolverStatus_t status{};
    #if DDLA_GPU_VENDOR
    status = gpusolverZgetrf(handle, m, n, (gpusolverDoubleComplex*)A, lda, (gpusolverDoubleComplex*)Workspace, Lwork, devIpiv, devInfo);
    #else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
    #endif
    #if DDLA_GPU_VENDOR
    if(Lwork > 0)
        GPU_CHECK(gpuFreeAsync(Workspace, stream));
    #endif
    return status;
}

inline gpusolverStatus_t gpusolverGetrf(
    gpusolverHandle_t handle,
    int m,
    int n,
    std::complex<float> *A,
    int lda,
    int *devIpiv,
    int *devInfo
)
{
    std::complex<float> *Workspace = nullptr;
    int Lwork;

    gpuStream_t stream;
    SOLVER_CHECK(gpusolverGetStream(handle, &stream));

    #if DDLA_GPU_VENDOR
    SOLVER_CHECK(gpusolverCgetrf_bufferSize(handle, m, n, (gpusolverComplex*)A, lda, &Lwork));
    #else
    throw std::runtime_error("not ENABLE CUDA and ENABLE HIP\n");
    #endif

    #if DDLA_GPU_VENDOR
    if(Lwork > 0)
        GPU_CHECK(gpuMallocAsync(reinterpret_cast<void**>(&Workspace), Lwork * sizeof(std::complex<float>), stream));
    #endif

    gpusolverStatus_t status{};
    #if DDLA_GPU_VENDOR
    status = gpusolverCgetrf(handle, m, n, (gpusolverComplex*)A, lda, (gpusolverComplex*)Workspace, Lwork, devIpiv, devInfo);
    #else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
    #endif

    #if DDLA_GPU_VENDOR
    if(Lwork > 0)
        GPU_CHECK(gpuFreeAsync(Workspace, stream));
    #endif
    return status;
}

}

#endif
