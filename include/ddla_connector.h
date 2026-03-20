#ifndef DDLA_CONNECTOR_H
#define DDLA_CONNECTOR_H

#include <mpi.h>
#include <iostream>
#ifdef ENABLE_CUDA
#include <nccl.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <curand.h>
#endif
#ifdef ENABLE_HIP
#include <rccl/rccl.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hipsolver/hipsolver.h>
#include <hiprand/hiprand.h>
#endif

#include <cmath>
#include <complex>

namespace DDLA{


#ifdef ENABLE_CUDA
using deviceStream_t = cudaStream_t;
using deviceError_t = cudaError_t;
using deblasStatus_t = cublasStatus_t;
using deblasHandle_t = cublasHandle_t;
using desolverHandle_t = cusolverDnHandle_t;
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
using deBlasOperation_t = cublasOperation_t;
constexpr auto DEBLAS_OP_N = cublasOperation_t::CUBLAS_OP_N;
constexpr auto DEBLAS_OP_T = cublasOperation_t::CUBLAS_OP_T;
constexpr auto DEBLAS_OP_C = cublasOperation_t::CUBLAS_OP_C;

#endif
#ifdef ENABLE_HIP
using deviceStream_t = hipStream_t;
using deviceError_t = hipError_t;
using deblasStatus_t = hipblasStatus_t;
using deblasHandle_t = hipblasHandle_t;
using desolverHandle_t = hipsolverHandle_t;
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
using deBlasOperation_t = hipblasOperation_t;
constexpr auto DEBLAS_OP_N = deBlasOperation_t::HIPBLAS_OP_N;
constexpr auto DEBLAS_OP_T = deBlasOperation_t::HIPBLAS_OP_T;
constexpr auto DEBLAS_OP_C = deBlasOperation_t::HIPBLAS_OP_C;
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

inline deblasStatus_t deblasScal(deblasHandle_t handle, int64_t n, const double *alpha, double *x, int64_t incx)
{
    #ifdef ENABLE_CUDA
    return cublasDscal(handle, n, alpha, x, incx);
    #endif
    #ifdef ENABLE_HIP
    return hipblasDscal(handle, n, alpha, x, incx);
    #endif
}

inline deblasStatus_t deblasScal(deblasHandle_t handle, int64_t n, const double *alpha, std::complex<double> *x, int64_t incx)
{
    #ifdef ENABLE_CUDA
    return cublasZdscal(handle, n, alpha, (cuDoubleComplex*)x, incx);
    #endif
    #ifdef ENABLE_HIP
    return hipblasZdscal(handle, n, alpha, (hipblasDoubleComplex*)x, incx);
    #endif
}

inline deblasStatus_t deblasScal(deblasHandle_t handle, int64_t n, const std::complex<double> *alpha, std::complex<double> *x, int64_t incx) {
    #ifdef ENABLE_CUDA
    return cublasZscal(handle, n, (cuDoubleComplex*)alpha, (cuDoubleComplex*)x, incx);
    #endif
    #ifdef ENABLE_HIP
    return hipblasZscal(handle, n, (hipblasDoubleComplex*)alpha, (hipblasDoubleComplex*)x, incx);
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
    deblasHandle_t handle, deblasSideMode_t side, deblasFillMode_t uplo, cublasOperation_t trans, deblasDiagType_t diag, 
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
    hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, hipblasOperation_t trans, hipblasDiagType_t diag, 
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
    deblasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, 
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
    deblasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, 
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
    deblasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, 
    int m, int n, int k, 
    const double *alpha, 
    const double *A, int lda, 
    const double *B, int ldb,
    const double *beta,
    double *C, int ldc
)
{
    return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
#endif

#ifdef ENABLE_HIP
inline deblasStatus_t deblasZgemm(
    deblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb,
    int m, int n, int k,
    const std::complex<double> *alpha,
    const std::complex<double> *A, int lda,
    const std::complex<double> *B, int ldb,
    const std::complex<double> *beta,
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
    const std::complex<double> *alpha,
    const std::complex<double> *A, int lda,
    const std::complex<double> *B, int ldb,
    const std::complex<double> *beta,
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
    const double *alpha,
    const double *A, int lda,
    const double *B, int ldb,
    const double *beta,
    double *C, int ldc
)
{
    return hipblasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
#endif

inline ncclResult_t cclSend(const double* sendbuff, size_t count, int peer, ncclComm_t comm, deviceStream_t stream)
{
    return ncclSend(sendbuff, count, ncclFloat64, peer, comm, stream);
}

inline ncclResult_t cclSend(const std::complex<double>* sendbuff, size_t count, int peer, ncclComm_t comm, deviceStream_t stream)
{
    return ncclSend(sendbuff, count*2, ncclFloat64, peer, comm, stream);
}

inline ncclResult_t  cclRecv(double* recvbuff, size_t count, int peer, ncclComm_t comm, deviceStream_t stream)
{
    return ncclRecv(recvbuff, count, ncclFloat64, peer, comm, stream);
}

inline ncclResult_t  cclRecv(std::complex<double>* recvbuff, size_t count, int peer, ncclComm_t comm, deviceStream_t stream)
{
    return ncclRecv(recvbuff, count*2, ncclFloat64, peer, comm, stream);
}

inline ncclResult_t  cclBroadcast(const double* sendbuff, double* recvbuff, size_t count, int root, ncclComm_t comm, deviceStream_t stream)
{
    return ncclBroadcast(sendbuff, recvbuff, count, ncclFloat64, root, comm, stream);
}

inline ncclResult_t  cclBroadcast(const std::complex<double>* sendbuff, std::complex<double>* recvbuff, size_t count, int root, ncclComm_t comm, deviceStream_t stream)
{
    return ncclBroadcast(sendbuff, recvbuff, count*2, ncclFloat64, root, comm, stream);
}

} // namespace DDLA

#endif // DDLA_CONNECTOR_H