#ifndef POTRF_H
#define POTRF_H

#include "ddla_connector.h"
#include <stdexcept>

namespace ddla{

inline gpusolverStatus_t gpusolverPotrf(
    gpusolverHandle_t handle,
    gpublasFillMode_t uplo,
    int n,
    std::complex<double> *A,
    int lda,
    int *devInfo
)
{
    std::complex<double> *Workspace = nullptr;
    int Lwork;

    gpuStream_t stream;
    SOLVER_CHECK(gpusolverGetStream(handle, &stream));

    #if DDLA_GPU_VENDOR
    SOLVER_CHECK(gpusolverZpotrf_bufferSize(handle, gpusolverFillMode(uplo), n, (gpusolverDoubleComplex*)A, lda, &Lwork));
    #else
    throw std::runtime_error("not ENABLE CUDA and ENABLE HIP\n");
    #endif

    #if DDLA_GPU_VENDOR
    if(Lwork > 0)
        GPU_CHECK(gpuMallocAsync(reinterpret_cast<void**>(&Workspace), Lwork*sizeof(std::complex<double>), stream));
    #endif

    gpusolverStatus_t status{};
    #if DDLA_GPU_VENDOR
    status = gpusolverZpotrf(handle, gpusolverFillMode(uplo), n, (gpusolverDoubleComplex*)A, lda, (gpusolverDoubleComplex*)Workspace, Lwork, devInfo);
    #else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
    #endif

    #if DDLA_GPU_VENDOR
    if(Lwork > 0)
        GPU_CHECK(gpuFreeAsync(Workspace, stream));
    #endif
    return status;
}

inline gpusolverStatus_t gpusolverPotrf(
    gpusolverHandle_t handle,
    gpublasFillMode_t uplo,
    int n,
    std::complex<float> *A,
    int lda,
    int *devInfo
)
{
    std::complex<float> *Workspace = nullptr;
    int Lwork;

    gpuStream_t stream;
    SOLVER_CHECK(gpusolverGetStream(handle, &stream));

    #if DDLA_GPU_VENDOR
    SOLVER_CHECK(gpusolverCpotrf_bufferSize(handle, gpusolverFillMode(uplo), n, (gpusolverComplex*)A, lda, &Lwork));
    #else
    throw std::runtime_error("not ENABLE CUDA and ENABLE HIP\n");
    #endif

    #if DDLA_GPU_VENDOR
    if(Lwork > 0)
        GPU_CHECK(gpuMallocAsync(reinterpret_cast<void**>(&Workspace), Lwork*sizeof(std::complex<float>), stream));
    #endif

    gpusolverStatus_t status{};
    #if DDLA_GPU_VENDOR
    status = gpusolverCpotrf(handle, gpusolverFillMode(uplo), n, (gpusolverComplex*)A, lda, (gpusolverComplex*)Workspace, Lwork, devInfo);
    #else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
    #endif

    #if DDLA_GPU_VENDOR
    if(Lwork > 0)
        GPU_CHECK(gpuFreeAsync(Workspace, stream));
    #endif
    return status;
}

inline gpusolverStatus_t gpusolverPotrf(
    gpusolverHandle_t handle,
    gpublasFillMode_t uplo,
    int n,
    float *A,
    int lda,
    int *devInfo
)
{
    float *Workspace = nullptr;
    int Lwork;

    gpuStream_t stream;
    SOLVER_CHECK(gpusolverGetStream(handle, &stream));

    #if DDLA_GPU_VENDOR
    SOLVER_CHECK(gpusolverSpotrf_bufferSize(handle, gpusolverFillMode(uplo), n, A, lda, &Lwork));
    #else
    throw std::runtime_error("not ENABLE CUDA and ENABLE HIP\n");
    #endif

    #if DDLA_GPU_VENDOR
    if(Lwork > 0)
        GPU_CHECK(gpuMallocAsync(reinterpret_cast<void**>(&Workspace), Lwork*sizeof(float), stream));
    #endif

    gpusolverStatus_t status{};
    #if DDLA_GPU_VENDOR
    status = gpusolverSpotrf(handle, gpusolverFillMode(uplo), n, A, lda, Workspace, Lwork, devInfo);
    #else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
    #endif

    #if DDLA_GPU_VENDOR
    if(Lwork > 0)
        GPU_CHECK(gpuFreeAsync(Workspace, stream));
    #endif
    return status;
}

inline gpusolverStatus_t gpusolverPotrf(
    gpusolverHandle_t handle,
    gpublasFillMode_t uplo,
    int n,
    double *A,
    int lda,
    int *devInfo
)
{
    double *Workspace = nullptr;
    int Lwork;

    gpuStream_t stream;
    SOLVER_CHECK(gpusolverGetStream(handle, &stream));

    #if DDLA_GPU_VENDOR
    SOLVER_CHECK(gpusolverDpotrf_bufferSize(handle, gpusolverFillMode(uplo), n, A, lda, &Lwork));
    #else
    throw std::runtime_error("not ENABLE CUDA and ENABLE HIP\n");
    #endif

    #if DDLA_GPU_VENDOR
    if(Lwork > 0)
        GPU_CHECK(gpuMallocAsync(reinterpret_cast<void**>(&Workspace), Lwork*sizeof(double), stream));
    #endif

    gpusolverStatus_t status{};
    #if DDLA_GPU_VENDOR
    status = gpusolverDpotrf(handle, gpusolverFillMode(uplo), n, A, lda, Workspace, Lwork, devInfo);
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
