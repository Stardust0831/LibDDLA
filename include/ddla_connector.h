#ifndef DDLA_CONNECTOR_H
#define DDLA_CONNECTOR_H

#include <mpi.h>
#include <iostream>
#ifdef ENABLE_CUDA
#ifdef ENABLE_CCL
#include <nccl.h>
#endif
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <curand.h>
#endif
#ifdef ENABLE_HIP
#ifdef ENABLE_CCL
#include <rccl/rccl.h>
#endif
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hipsolver/hipsolver.h>
#include <hiprand/hiprand.h>
#endif

#include <cmath>
#include <complex>
#include "ddla_utils.h"

namespace DDLA{

#ifdef ENABLE_CCL
using cclOp=ncclRedOp_t;
const auto cclSum=ncclRedOp_t::ncclSum;
#else
using cclOp=MPI_Op;
const auto cclSum=MPI_SUM;
#endif
#ifdef ENABLE_CUDA
using deviceStream_t = cudaStream_t;
using deviceError_t = cudaError_t;
using deblasStatus_t = cublasStatus_t;
using deblasHandle_t = cublasHandle_t;
using desolverHandle_t = cusolverDnHandle_t;
using desolverStatus_t = cusolverStatus_t;
#define desolverGetStream cusolverDnGetStream
#define deviceMemcpyAsync cudaMemcpyAsync
#define deviceMemcpy cudaMemcpy
#define deviceMemcpy2DAsync cudaMemcpy2DAsync
#define deviceMallocAsync cudaMallocAsync
#define deviceMalloc cudaMalloc
#define deviceMemsetAsync cudaMemsetAsync
#define deviceFreeAsync cudaFreeAsync
#define deviceFree cudaFree
using deviceDataType_t = cudaDataType_t;
constexpr auto DEVICE_R_64F = deviceDataType_t::CUDA_R_64F;
constexpr auto DEVICE_C_64F = deviceDataType_t::CUDA_C_64F;
constexpr auto DEVICE_R_32F = deviceDataType_t::CUDA_R_32F;
constexpr auto DEVICE_C_32F = deviceDataType_t::CUDA_C_32F;
using derandGenerator_t = curandGenerator_t;
using derandStatus_t = curandStatus_t;
constexpr auto DERAND_STATUS_SUCCESS = derandStatus_t::CURAND_STATUS_SUCCESS;
#define derandCreateGenerator curandCreateGenerator
#define derandSetPseudoRandomGeneratorSeed curandSetPseudoRandomGeneratorSeed
#define derandGenerateUniform curandGenerateUniform
#define derandGenerateUniformDouble curandGenerateUniformDouble
#define derandDestroyGenerator curandDestroyGenerator
using derandRngType = curandRngType;
constexpr auto DERAND_RNG_PSEUDO_DEFAULT = derandRngType::CURAND_RNG_PSEUDO_DEFAULT;
#define deviceMemGetInfo cudaMemGetInfo
using deblasSideMode_t = cublasSideMode_t;
constexpr auto DEBLAS_SIDE_LEFT = deblasSideMode_t::CUBLAS_SIDE_LEFT;
constexpr auto DEBLAS_SIDE_RIGHT = deblasSideMode_t::CUBLAS_SIDE_RIGHT;
using deblasFillMode_t = cublasFillMode_t;
constexpr auto DEBLAS_FILL_MODE_LOWER = deblasFillMode_t::CUBLAS_FILL_MODE_LOWER;
constexpr auto DEBLAS_FILL_MODE_UPPER = deblasFillMode_t::CUBLAS_FILL_MODE_UPPER;
using deblasDiagType_t = cublasDiagType_t;
constexpr auto DEBLAS_DIAG_UNIT = deblasDiagType_t::CUBLAS_DIAG_UNIT;
constexpr auto DEBLAS_DIAG_NON_UNIT = deblasDiagType_t::CUBLAS_DIAG_NON_UNIT;
using deblasOperation_t = cublasOperation_t;
constexpr auto DEBLAS_OP_N = deblasOperation_t::CUBLAS_OP_N;
constexpr auto DEBLAS_OP_T = deblasOperation_t::CUBLAS_OP_T;
constexpr auto DEBLAS_OP_C = deblasOperation_t::CUBLAS_OP_C;

#endif
#ifdef ENABLE_HIP
using deviceStream_t = hipStream_t;
using deviceError_t = hipError_t;
using deblasStatus_t = hipblasStatus_t;
using deblasHandle_t = hipblasHandle_t;
using desolverHandle_t = hipsolverHandle_t;
using desolverStatus_t = hipsolverStatus_t;
#define desolverGetStream hipsolverGetStream
#define deviceMemcpyAsync hipMemcpyAsync
#define deviceMemcpy hipMemcpy
#define deviceMemcpy2DAsync hipMemcpy2DAsync
#define deviceMallocAsync hipMallocAsync
#define deviceMalloc hipMalloc
#define deviceMemsetAsync hipMemsetAsync
#define deviceFreeAsync hipFreeAsync
#define deviceFree hipFree
using deviceDataType_t = hipDataType;
constexpr auto DEVICE_R_64F = deviceDataType_t::HIP_R_64F;
constexpr auto DEVICE_C_64F = deviceDataType_t::HIP_C_64F;
constexpr auto DEVICE_R_32F = deviceDataType_t::HIP_R_32F;
constexpr auto DEVICE_C_32F = deviceDataType_t::HIP_C_32F;
using derandGenerator_t = hiprandGenerator_t;
using derandStatus_t = hiprandStatus_t;
constexpr auto DERAND_STATUS_SUCCESS = derandStatus_t::HIPRAND_STATUS_SUCCESS;
#define derandCreateGenerator hiprandCreateGenerator
#define derandSetPseudoRandomGeneratorSeed hiprandSetPseudoRandomGeneratorSeed
#define derandGenerateUniform hiprandGenerateUniform
#define derandGenerateUniformDouble hiprandGenerateUniformDouble
#define derandDestroyGenerator hiprandDestroyGenerator
using derandRngType = hiprandRngType;
constexpr auto DERAND_RNG_PSEUDO_DEFAULT = derandRngType::HIPRAND_RNG_PSEUDO_DEFAULT;
#define deviceMemGetInfo hipMemGetInfo
using deblasSideMode_t = hipblasSideMode_t;
constexpr auto DEBLAS_SIDE_LEFT = deblasSideMode_t::HIPBLAS_SIDE_LEFT;
constexpr auto DEBLAS_SIDE_RIGHT = deblasSideMode_t::HIPBLAS_SIDE_RIGHT;
using deblasFillMode_t = hipblasFillMode_t;
constexpr auto DEBLAS_FILL_MODE_LOWER = deblasFillMode_t::HIPBLAS_FILL_MODE_LOWER;
constexpr auto DEBLAS_FILL_MODE_UPPER = deblasFillMode_t::HIPBLAS_FILL_MODE_UPPER;
using deblasDiagType_t = hipblasDiagType_t;
constexpr auto DEBLAS_DIAG_UNIT = deblasDiagType_t::HIPBLAS_DIAG_UNIT;
constexpr auto DEBLAS_DIAG_NON_UNIT = deblasDiagType_t::HIPBLAS_DIAG_NON_UNIT;
using deblasOperation_t = hipblasOperation_t;
constexpr auto DEBLAS_OP_N = deblasOperation_t::HIPBLAS_OP_N;
constexpr auto DEBLAS_OP_T = deblasOperation_t::HIPBLAS_OP_T;
constexpr auto DEBLAS_OP_C = deblasOperation_t::HIPBLAS_OP_C;
#endif

#ifdef ENABLE_CUDA
using deviceMemcpyKind=cudaMemcpyKind;
constexpr auto deviceMemcpyHostToDevice = deviceMemcpyKind::cudaMemcpyHostToDevice;
constexpr auto deviceMemcpyDeviceToHost = deviceMemcpyKind::cudaMemcpyDeviceToHost;
constexpr auto deviceMemcpyDeviceToDevice = deviceMemcpyKind::cudaMemcpyDeviceToDevice;
#endif
#ifdef ENABLE_HIP
using deviceMemcpyKind=hipMemcpyKind;
constexpr auto deviceMemcpyHostToDevice = deviceMemcpyKind::hipMemcpyHostToDevice;
constexpr auto deviceMemcpyDeviceToHost = deviceMemcpyKind::hipMemcpyDeviceToHost;
constexpr auto deviceMemcpyDeviceToDevice = deviceMemcpyKind::hipMemcpyDeviceToDevice;
#endif


#ifdef ENABLE_CUDA
inline cudaError_t deviceStreamSynchronize(cudaStream_t stream) {
    return cudaStreamSynchronize(stream);
}
#endif
#ifdef ENABLE_HIP
inline hipError_t deviceStreamSynchronize(hipStream_t stream) {
    return hipStreamSynchronize(stream);
}
#endif

#ifdef ENABLE_CUDA
inline cudaError_t deviceDeviceSynchronize(){
    return cudaDeviceSynchronize();
}
#endif

#ifdef ENABLE_HIP
inline hipError_t deviceDeviceSynchronize(){
    return hipDeviceSynchronize();
}
#endif

inline deblasStatus_t deblasIzamax(deblasHandle_t handle, int n, const std::complex<double> *x, int incx, int *result) {
    #ifdef ENABLE_CUDA
    return cublasIzamax(handle, n, (cuDoubleComplex*)x, incx, result);
    #endif
    #ifdef ENABLE_HIP
    return hipblasIzamax(handle, n, (hipblasDoubleComplex*)x, incx, result);
    #endif
}

inline deblasStatus_t deblasZscal(deblasHandle_t handle, int64_t n, const std::complex<double> *alpha, std::complex<double> *x, int64_t incx) {
    #ifdef ENABLE_CUDA
    return cublasZscal(handle, n, (cuDoubleComplex*)alpha, (cuDoubleComplex*)x, incx);
    #endif
    #ifdef ENABLE_HIP
    return hipblasZscal(handle, n, (hipblasDoubleComplex*)alpha, (hipblasDoubleComplex*)x, incx);
    #endif
}
inline deblasStatus_t deblasZdscal(deblasHandle_t handle, int64_t n, const double *alpha, std::complex<double> *x, int64_t incx) {
    #ifdef ENABLE_CUDA
    return cublasZdscal(handle, n, alpha, (cuDoubleComplex*)x, incx);
    #endif
    #ifdef ENABLE_HIP
    return hipblasZdscal(handle, n, alpha, (hipblasDoubleComplex*)x, incx);
    #endif
}

inline deblasStatus_t deblasScal(deblasHandle_t handle, int64_t n, const float& alpha, float *x, int64_t incx)
{
    #ifdef ENABLE_CUDA
    return cublasSscal(handle, n, &alpha, x, incx);
    #endif
    #ifdef ENABLE_HIP
    return hipblasSscal(handle, n, &alpha, x, incx);
    #endif
}

inline deblasStatus_t deblasScal(deblasHandle_t handle, int64_t n, const double& alpha, double *x, int64_t incx)
{
    #ifdef ENABLE_CUDA
    return cublasDscal(handle, n, &alpha, x, incx);
    #endif
    #ifdef ENABLE_HIP
    return hipblasDscal(handle, n, &alpha, x, incx);
    #endif
}

inline deblasStatus_t deblasScal(deblasHandle_t handle, int64_t n, const float& alpha, std::complex<float> *x, int64_t incx)
{
    #ifdef ENABLE_CUDA
    return cublasCsscal(handle, n, &alpha, (cuFloatComplex*)x, incx);
    #endif
    #ifdef ENABLE_HIP
    return hipblasCsscal(handle, n, &alpha, (hipblasComplex*)x, incx);
    #endif
}

inline deblasStatus_t deblasScal(deblasHandle_t handle, int64_t n, const double& alpha, std::complex<double> *x, int64_t incx)
{
    #ifdef ENABLE_CUDA
    return cublasZdscal(handle, n, &alpha, (cuDoubleComplex*)x, incx);
    #endif
    #ifdef ENABLE_HIP
    return hipblasZdscal(handle, n, &alpha, (hipblasDoubleComplex*)x, incx);
    #endif
}

inline deblasStatus_t deblasScal(deblasHandle_t handle, int64_t n, const std::complex<float>& alpha, std::complex<float> *x, int64_t incx) {
    #ifdef ENABLE_CUDA
    return cublasCscal(handle, n, (cuFloatComplex*)&alpha, (cuFloatComplex*)x, incx);
    #endif
    #ifdef ENABLE_HIP
    return hipblasCscal(handle, n, (hipblasComplex*)&alpha, (hipblasComplex*)x, incx);
    #endif
}

inline deblasStatus_t deblasScal(deblasHandle_t handle, int64_t n, const std::complex<double>& alpha, std::complex<double> *x, int64_t incx) {
    #ifdef ENABLE_CUDA
    return cublasZscal(handle, n, (cuDoubleComplex*)&alpha, (cuDoubleComplex*)x, incx);
    #endif
    #ifdef ENABLE_HIP
    return hipblasZscal(handle, n, (hipblasDoubleComplex*)&alpha, (hipblasDoubleComplex*)x, incx);
    #endif
}





inline deblasStatus_t deblasZaxpy(deblasHandle_t handle, const int64_t& n, const std::complex<double> *alpha, const std::complex<double> *x, int incx, std::complex<double> *y, int incy) {
    #ifdef ENABLE_CUDA
    return cublasZaxpy(handle, n, (cuDoubleComplex*)alpha, (cuDoubleComplex*)x, incx, (cuDoubleComplex*)y, incy);
    #endif
    #ifdef ENABLE_HIP
    return hipblasZaxpy(handle, n, (hipblasDoubleComplex*)alpha, (hipblasDoubleComplex*)x, incx, (hipblasDoubleComplex*)y, incy);
    #endif
}

inline deblasStatus_t deblasZgeru(deblasHandle_t handle, int m, int n, const std::complex<double> *alpha, const std::complex<double> *x, int incx, const std::complex<double> *y, int incy, std::complex<double> *A, int lda) {
    #ifdef ENABLE_CUDA
    return cublasZgeru(handle, m, n, (cuDoubleComplex*)alpha, (cuDoubleComplex*)x, incx, (cuDoubleComplex*)y, incy, (cuDoubleComplex*)A, lda);
    #endif
    #ifdef ENABLE_HIP
    return hipblasZgeru(handle, m, n, (hipblasDoubleComplex*)alpha, (hipblasDoubleComplex*)x, incx, (hipblasDoubleComplex*)y, incy, (hipblasDoubleComplex*)A, lda);
    #endif
}
#ifdef ENABLE_CUDA
inline deblasStatus_t deblasZswap(cublasHandle_t handle, int n, std::complex<double> *x, int incx, std::complex<double> *y, int incy) {
  return cublasZswap(handle, n, (cuDoubleComplex*)x, incx, (cuDoubleComplex*)y, incy);
}
#endif
#ifdef ENABLE_HIP
inline deblasStatus_t deblasZswap(deblasHandle_t handle, int n, std::complex<double> *x, int incx, std::complex<double> *y, int incy) {
  return hipblasZswap(handle, n, (hipblasDoubleComplex*)x, incx, (hipblasDoubleComplex*)y, incy);
}
#endif

#ifdef ENABLE_CUDA
inline deblasStatus_t deblasZtrsm(
    deblasHandle_t handle, deblasSideMode_t side, deblasFillMode_t uplo, deblasOperation_t trans, deblasDiagType_t diag, 
    int m, int n, 
    const std::complex<double> *alpha, 
    std::complex<double> *A, int lda, 
    std::complex<double> *B, int ldb
    ) 
{
  return cublasZtrsm(handle, side, uplo, trans, diag, m, n, (cuDoubleComplex*)alpha, (cuDoubleComplex*)A, lda, (cuDoubleComplex*)B, ldb);
}
#endif
#ifdef ENABLE_HIP
inline deblasStatus_t deblasZtrsm(
    deblasHandle_t handle, deblasSideMode_t side, deblasFillMode_t uplo, hipblasOperation_t trans, deblasDiagType_t diag, 
    int m, int n, 
    const std::complex<double> *alpha,
    std::complex<double> *A, int lda,
    std::complex<double> *B, int ldb
)
{
  return hipblasZtrsm(handle, side, uplo, trans, diag, m, n, (hipblasDoubleComplex*)alpha, (hipblasDoubleComplex*)A, lda, (hipblasDoubleComplex*)B, ldb);
}
#endif


#ifdef ENABLE_CUDA
inline deblasStatus_t deblasZgemm(
    deblasHandle_t handle, deblasOperation_t transa, deblasOperation_t transb, 
    int m, int n, int k, 
    const std::complex<double> *alpha, 
    const std::complex<double> *A, int lda, 
    const std::complex<double> *B, int ldb,
    const std::complex<double> *beta,
    std::complex<double> *C, int ldc
)
{
    return cublasZgemm(handle, transa, transb, m, n, k, (cuDoubleComplex*)alpha, (cuDoubleComplex*)A, lda, (cuDoubleComplex*)B, ldb, (cuDoubleComplex*)beta, (cuDoubleComplex*)C, ldc);
}
#endif

#ifdef ENABLE_CUDA
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
    return cublasSgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
}
#endif

#ifdef ENABLE_CUDA
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
    return cublasDgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
}
#endif

#ifdef ENABLE_CUDA
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
    return cublasCgemm(handle, transa, transb, m, n, k, (cuFloatComplex*)&alpha, (cuFloatComplex*)A, lda, (cuFloatComplex*)B, ldb, (cuFloatComplex*)&beta, (cuFloatComplex*)C, ldc);
}
#endif

#ifdef ENABLE_CUDA
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
    return cublasZgemm(handle, transa, transb, m, n, k, (cuDoubleComplex*)&alpha, (cuDoubleComplex*)A, lda, (cuDoubleComplex*)B, ldb, (cuDoubleComplex*)&beta, (cuDoubleComplex*)C, ldc);
}
#endif

#ifdef ENABLE_HIP
inline deblasStatus_t deblasGemm(
    deblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb,
    int m, int n, int k,
    const float& alpha,
    const float *A, int lda,
    const float *B, int ldb,
    const float& beta,
    float *C, int ldc
)
{
    return hipblasSgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
}
#endif

#ifdef ENABLE_HIP
inline deblasStatus_t deblasGemm(
    deblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb,
    int m, int n, int k,
    const double& alpha,
    const double *A, int lda,
    const double *B, int ldb,
    const double& beta,
    double *C, int ldc
)
{
    return hipblasDgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
}
#endif

#ifdef ENABLE_HIP
inline deblasStatus_t deblasGemm(
    deblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb,
    int m, int n, int k,
    const std::complex<float>& alpha,
    const std::complex<float> *A, int lda,
    const std::complex<float> *B, int ldb,
    const std::complex<float>& beta,
    std::complex<float> *C, int ldc
)
{
    return hipblasCgemm(handle, transa, transb, m, n, k, (hipblasComplex*)&alpha, (hipblasComplex*)A, lda, (hipblasComplex*)B, ldb, (hipblasComplex*)&beta, (hipblasComplex*)C, ldc);
}
#endif

#ifdef ENABLE_HIP
inline deblasStatus_t deblasZgemm(
    deblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb,
    int m, int n, int k,
    const std::complex<double>* alpha,
    const std::complex<double> *A, int lda,
    const std::complex<double> *B, int ldb,
    const std::complex<double>* beta,
    std::complex<double> *C, int ldc
)
{
    return hipblasZgemm(handle, transa, transb, m, n, k, (hipblasDoubleComplex*)alpha, (hipblasDoubleComplex*)A, lda, (hipblasDoubleComplex*)B, ldb, (hipblasDoubleComplex*)beta, (hipblasDoubleComplex*)C, ldc);
}
#endif

#ifdef ENABLE_HIP
inline deblasStatus_t deblasGemm(
    deblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb,
    int m, int n, int k,
    const std::complex<double>& alpha,
    const std::complex<double> *A, int lda,
    const std::complex<double> *B, int ldb,
    const std::complex<double>& beta,
    std::complex<double> *C, int ldc
)
{
    return hipblasZgemm(handle, transa, transb, m, n, k, (hipblasDoubleComplex*)&alpha, (hipblasDoubleComplex*)A, lda, (hipblasDoubleComplex*)B, ldb, (hipblasDoubleComplex*)&beta, (hipblasDoubleComplex*)C, ldc);
}
#endif

inline int MPI_Allreduce_ddla(const float* sendbuff, float* recvbuff, int count, MPI_Op op, MPI_Comm comm)
{
    return MPI_Allreduce(sendbuff, recvbuff, count, MPI_FLOAT, op, comm);
}

inline int MPI_Allreduce_ddla(const std::complex<float>* sendbuff, std::complex<float>* recvbuff, int count, MPI_Op op, MPI_Comm comm)
{
    return MPI_Allreduce(sendbuff, recvbuff, count * 2, MPI_FLOAT, op, comm);
}

inline int MPI_Allreduce_ddla(const double* sendbuff, double* recvbuff, int count, MPI_Op op, MPI_Comm comm)
{
    return MPI_Allreduce(sendbuff, recvbuff, count, MPI_DOUBLE, op, comm);
}

inline int MPI_Allreduce_ddla(const std::complex<double>* sendbuff, std::complex<double>* recvbuff, int count, MPI_Op op, MPI_Comm comm)
{
    return MPI_Allreduce(sendbuff, recvbuff, count, MPI_DOUBLE_COMPLEX, op, comm);
}


#ifdef ENABLE_CCL
template<typename T>
inline ncclResult_t cclSend(const T* sendbuff, size_t count, int peer, ncclComm_t comm, deviceStream_t stream)
{
    return ncclSend(sendbuff, count * sizeof(T), ncclInt8, peer, comm, stream);
}

template<typename T>
inline ncclResult_t cclRecv(T* recvbuff, size_t count, int peer, ncclComm_t comm, deviceStream_t stream)
{
    return ncclRecv(recvbuff, count * sizeof(T), ncclInt8, peer, comm, stream);
}

template<typename T>
inline ncclResult_t cclBroadcast(const T* sendbuff, T* recvbuff, size_t count, int root, ncclComm_t comm, deviceStream_t stream)
{
    return ncclBroadcast(sendbuff, recvbuff, count * sizeof(T), ncclInt8, root, comm, stream);
}

template<typename T>
inline ncclResult_t cclBcast(T* buff, size_t count, int root, ncclComm_t comm, deviceStream_t stream)
{
    return ncclBcast(buff, count * sizeof(T), ncclInt8, root, comm, stream);
}

inline ncclResult_t cclAllReduce(const float* sendbuff, float* recvbuff, int count, cclOp op, ncclComm_t comm, deviceStream_t stream)
{
    return ncclAllReduce(sendbuff, recvbuff, count, ncclFloat32, op, comm, stream);
}

inline ncclResult_t cclAllReduce(const std::complex<float>* sendbuff, std::complex<float>* recvbuff, int count, cclOp op, ncclComm_t comm, deviceStream_t stream)
{
    return ncclAllReduce(sendbuff, recvbuff, count * 2, ncclFloat32, op, comm, stream);
}

inline ncclResult_t cclAllReduce(const double* sendbuff, double* recvbuff, int count, cclOp op, ncclComm_t comm, deviceStream_t stream)
{
    return ncclAllReduce(sendbuff, recvbuff, count, ncclFloat64, op, comm, stream);
}

inline ncclResult_t cclAllReduce(const std::complex<double>* sendbuff, std::complex<double>* recvbuff, int count, cclOp op, ncclComm_t comm, deviceStream_t stream)
{
    return ncclAllReduce(sendbuff, recvbuff, count * 2, ncclFloat64, op, comm, stream);
}

#else

template<typename T>
inline int cclSend(const T* sendbuff, size_t count, int peer, MPI_Comm comm, deviceStream_t stream)
{
    DEVICE_CHECK(deviceStreamSynchronize(stream));
    return MPI_Send(sendbuff, count * sizeof(T), MPI_BYTE, peer, 0, comm);
}

template<typename T>
inline int cclRecv(T* recvbuff, size_t count, int peer, MPI_Comm comm, deviceStream_t stream)
{
    DEVICE_CHECK(deviceStreamSynchronize(stream));
    return MPI_Recv(recvbuff, count * sizeof(T), MPI_BYTE, peer, 0, comm, MPI_STATUS_IGNORE);
}

template<typename T>
inline int cclBcast(T* buff, size_t count, int root, MPI_Comm comm, deviceStream_t stream)
{
    DEVICE_CHECK(deviceStreamSynchronize(stream));
    return MPI_Bcast(buff, count * sizeof(T), MPI_BYTE, root, comm);
}

template<typename T>
inline int cclAllReduce(const T* sendbuff, T* recvbuff, int count, cclOp op, MPI_Comm comm, deviceStream_t stream)
{
    DEVICE_CHECK(deviceStreamSynchronize(stream));
    return MPI_Allreduce_ddla(sendbuff, recvbuff, count, op, comm);
}

template<typename T>
inline int cclBroadcast(const T* sendbuff, T* recvbuff, int count, int root, MPI_Comm comm, deviceStream_t stream)
{
    DEVICE_CHECK(deviceStreamSynchronize(stream));
    return MPI_Broadcast(sendbuff, recvbuff, count * sizeof(T), MPI_BYTE, root, comm);
}
#endif

#ifdef ENABLE_GPU_CPU_TUNNEL
template<typename T>
inline int cclBcast(T* h_sendbuff, T* d_sendbuff, size_t count, int root, MPI_Comm comm, deviceStream_t stream)
{
    DEVICE_CHECK(deviceMemcpyAsync(h_sendbuff, d_sendbuff, count * sizeof(T), deviceMemcpyDeviceToHost, stream));
    DEVICE_CHECK(deviceStreamSynchronize(stream));
    int value =  MPI_Bcast(h_sendbuff, count * sizeof(T), MPI_BYTE, root, comm);
    DEVICE_CHECK(deviceMemcpyAsync(d_sendbuff, h_sendbuff, count * sizeof(T), deviceMemcpyHostToDevice, stream));
    return value;
}

template<typename T>
inline int cclSend(T* h_sendbuff, const T* d_sendbuff, size_t count, int peer, MPI_Comm comm, deviceStream_t stream)
{
    DEVICE_CHECK(deviceMemcpyAsync(h_sendbuff, d_sendbuff, count * sizeof(T), deviceMemcpyDeviceToHost, stream));
    DEVICE_CHECK(deviceStreamSynchronize(stream));
    return MPI_Send(h_sendbuff, count * sizeof(T), MPI_BYTE, peer, 0, comm);
}

template<typename T>
inline int cclRecv(T* h_recvbuff, T* d_recvbuff, size_t count, int peer, MPI_Comm comm, deviceStream_t stream)
{
    DEVICE_CHECK(deviceStreamSynchronize(stream));
    int value = MPI_Recv(h_recvbuff, count * sizeof(T), MPI_BYTE, peer, 0, comm, MPI_STATUS_IGNORE);
    DEVICE_CHECK(deviceMemcpyAsync(d_recvbuff, h_recvbuff, count * sizeof(T), deviceMemcpyHostToDevice, stream));
    return value;
}

template<typename T>
inline int cclAllReduce(T* h_sendbuff, const T* d_sendbuff, T* h_recvbuff, T* d_recvbuff, int count, cclOp op, MPI_Comm comm, deviceStream_t stream)
{
    DEVICE_CHECK(deviceMemcpyAsync(h_sendbuff, d_sendbuff, count * sizeof(T), deviceMemcpyDeviceToHost, stream));
    DEVICE_CHECK(deviceMemcpyAsync(h_recvbuff, d_recvbuff, count * sizeof(T), deviceMemcpyDeviceToHost, stream));
    DEVICE_CHECK(deviceStreamSynchronize(stream));
    int value = MPI_Allreduce_ddla(h_sendbuff, h_recvbuff, count, op, comm);
    DEVICE_CHECK(deviceMemcpyAsync(d_recvbuff, h_recvbuff, count * sizeof(T), deviceMemcpyHostToDevice, stream));
    return value;
}
#endif


void random_generator(void* c_data, const int64_t& lengthOfData, const deviceDataType_t& compute_type);
// col major
void write_matrix(std::complex<double>* A, const int& m,const int& n, const char* filename);

} // namespace DDLA

#endif // DDLA_CONNECTOR_H