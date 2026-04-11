#ifndef DDLA_UTILS_H
#define DDLA_UTILS_H

#pragma once
#include <mpi.h>
#include <chrono>
#include <fstream>
#include <complex>

#ifndef MPI_CHECK
#define MPI_CHECK(call)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        int status = call;                                                                                             \
        if (status != MPI_SUCCESS)                                                                                     \
        {                                                                                                              \
            fprintf(stderr, "MPI error at %s:%d : %d\n", __FILE__, __LINE__, status);\
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    }  while(0)
#endif
#ifdef ENABLE_CUDA
#define DEVICE_CHECK(call)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t status = call;                                                                                     \
        if (status != cudaSuccess)                                                                                     \
        {                                                                                                              \
            fprintf(stderr, "CUDA error at %s:%d : %s\n", __FILE__, __LINE__, cudaGetErrorString(status));             \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

#define BLAS_CHECK(err)                                                                                              \
    do {                                                                                                               \
        cublasStatus_t err_ = (err);                                                                                   \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                                           \
            std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                                       \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)
    
#define SOLVER_CHECK(err)                                                                        \
    do {                                                                                           \
        cusolverStatus_t err_ = (err);                                                             \
        if (err_ != CUSOLVER_STATUS_SUCCESS) {                                                     \
            printf("cusolver error %d at %s:%d\n", err_, __FILE__, __LINE__);                      \
            throw std::runtime_error("cusolver error");                                            \
        }                                                                                          \
    } while (0)
#endif

#ifdef ENABLE_CCL
#ifndef CCL_CHECK
#define CCL_CHECK(call)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        ncclResult_t status = call;                                                                                    \
        if (status != ncclSuccess)                                                                                     \
        {                                                                                                              \
            fprintf(stderr, "NCCL error at %s:%d : %d\n", __FILE__, __LINE__, status);                                 \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)
#endif
#else
#define CCL_CHECK(call) MPI_CHECK(call)
#endif

#ifdef ENABLE_HIP
#ifndef DEVICE_CHECK
#define DEVICE_CHECK(error)                    \
    do {                                             \
        hipError_t status = (error);                 \
        if(status != hipSuccess)                       \
        {                                             \
            fprintf(stderr,                           \
                    "Hip error: '%s'(%d) at %s:%d\n", \
                    hipGetErrorString(status),         \
                    status,                            \
                    __FILE__,                         \
                    __LINE__);                        \
            exit(EXIT_FAILURE);                       \
        }                                      \
    } while (0)
#endif

#ifndef BLAS_CHECK
#define BLAS_CHECK(error)                              \
    do {                                                   \
        hipblasStatus_t status = (error);                    \
        if(status != HIPBLAS_STATUS_SUCCESS)                         \
        {                                                           \
            fprintf(stderr, "hipBLAS error: ");                     \
            if(status == HIPBLAS_STATUS_NOT_INITIALIZED)             \
                fprintf(stderr, "HIPBLAS_STATUS_NOT_INITIALIZED");  \
            if(status == HIPBLAS_STATUS_ALLOC_FAILED)                \
                fprintf(stderr, "HIPBLAS_STATUS_ALLOC_FAILED");     \
            if(status == HIPBLAS_STATUS_INVALID_VALUE)               \
                fprintf(stderr, "HIPBLAS_STATUS_INVALID_VALUE");    \
            if(status == HIPBLAS_STATUS_MAPPING_ERROR)               \
                fprintf(stderr, "HIPBLAS_STATUS_MAPPING_ERROR");    \
            if(status == HIPBLAS_STATUS_EXECUTION_FAILED)            \
                fprintf(stderr, "HIPBLAS_STATUS_EXECUTION_FAILED"); \
            if(status == HIPBLAS_STATUS_INTERNAL_ERROR)              \
                fprintf(stderr, "HIPBLAS_STATUS_INTERNAL_ERROR");   \
            if(status == HIPBLAS_STATUS_NOT_SUPPORTED)               \
                fprintf(stderr, "HIPBLAS_STATUS_NOT_SUPPORTED");    \
            if(status == HIPBLAS_STATUS_INVALID_ENUM)                \
                fprintf(stderr, "HIPBLAS_STATUS_INVALID_ENUM");     \
            if(status == HIPBLAS_STATUS_UNKNOWN)                     \
                fprintf(stderr, "HIPBLAS_STATUS_UNKNOWN");          \
            fprintf(stderr, "error in deblas\n");                                  \
            exit(EXIT_FAILURE);                                     \
        }   \
    } while (0)
#endif

#define SOLVER_CHECK(call)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        hipsolverStatus_t status = call;                                                                               \
        if (status != HIPSOLVER_STATUS_SUCCESS)                                                                       \
        {                                                                                                              \
            fprintf(stderr, "HIPSOLVER error at %s:%d : %d\n", __FILE__, __LINE__, status); \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)


#endif

#ifndef DERAND_CHECK
#define DERAND_CHECK(call)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        derandStatus_t status = call;                                                                               \
        if (status != DERAND_STATUS_SUCCESS)                                                                       \
        {                                                                                                              \
            fprintf(stderr, "DERAND error at %s:%d : %d\n", __FILE__, __LINE__, status); \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)
#endif



#endif // DDLA_UTILS_H