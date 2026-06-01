/**
 * @brief Single-GPU Cholesky factorization from the bottom-right corner:
 *        A = U * U^H,  U is upper-triangular.
 *
 *        Standard potrf (LAPACK/cusolver) only computes A = L * L^H.
 *        This implementation provides a custom GPU kernel that directly
 *        factors a Hermitian positive-definite matrix into U * U^H,
 *        working column-by-column from right to left (bottom-right style).
 *
 *        The scalar recurrence for U * U^H (upper triangle stored):
 *          - for column j = n-1 down to 0:
 *              A(j,j) = sqrt( A(j,j) - sum_{k>j} |A(j,k)|^2 )
 *              for i = j-1 down to 0:
 *                  A(i,j) = ( A(i,j) - sum_{k>j} A(i,k)*conj(A(j,k)) ) / A(j,j)
 *
 *        This kernel runs on a single diagonal block of size jb.
 *        The outer blocked algorithm is identical in structure to MAGMA's:
 *        herk → factorize panel → gemm → trsm.
 *
 * @note   Block-level potrf uses the custom right-looking kernel;
 *         the trailing update (gemm + trsm) uses standard cublas calls.
 *         trsm:  X * U^H = B   →  side=Right, uplo=Upper, trans=ConjTrans.
 */

#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <complex>

// --------------- platform abstraction (standalone demo) --------------------
#ifndef ENABLE_CUDA
#define ENABLE_CUDA
#endif

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#define DEVICE_CHECK(c) do { cudaError_t e = (c); if (e != cudaSuccess) { \
    fprintf(stderr,"CUDA err %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1); }} while(0)
#define BLAS_CHECK(c)   do { cublasStatus_t e = (c); if (e != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr,"cuBLAS err %s:%d: %d\n",__FILE__,__LINE__,(int)e); exit(1); }} while(0)

using stream_t  = cudaStream_t;
using blasH_t   = cublasHandle_t;
#endif
// ---------------------------------------------------------------------------

/**
 * @brief GPU kernel: Cholesky factorization A = U * U^H.
 *
 *        Processes one column per iteration (right-to-left).
 *        The upper-triangular part of the jb×jb matrix `A` (leading dim `lda`)
 *        is factorized in-place.  The strictly lower-triangular part is not
 *        touched.
 *
 *        Internally uses a parallel reduction for the inner-product sum
 *        (sum_{k>j} A(i,k) * conj(A(j,k))) via a local memory scratchpad.
 *
 * @param jb   Block size.
 * @param A    Device pointer to the jb×jb matrix (upper triangle referenced).
 * @param lda  Leading dimension of A.
 * @param info_ptr  Device pointer to int: set to 0 on success, >0 if non-P.D.
 *
 *        Launch: 1 block, jb threads.  Block size must be >= jb.
 *        If jb > 1024, split into multiple blocks (not shown for brevity).
 */
template<typename T>
__global__ void potrf_UxUH_kernel(int jb, T* A, int lda, int* info_ptr)
{
    extern __shared__ T sdata[];   // shared memory for partial sums and col pivots

    if (threadIdx.x == 0) *info_ptr = 0;
    __syncthreads();

    for (int j = jb - 1; j >= 0; --j) {

        // ---- 1) compute |A(j,k)|^2 sum for k > j ----
        T sum = T(0.0);
        for (int k = j + 1; k < jb; ++k) {
            T akj = A[j + k * lda];      // upper triangle: column-wise, row < col
            sum  += akj * conj(akj);
        }

        // diagonal element
        T diag = A[j + j * lda] - sum;
        T d_real = diag.real();

        if (d_real <= 0.0) {
            if (threadIdx.x == 0) *info_ptr = j + 1;
            return;
        }
        T piv = T(sqrt(static_cast<double>(d_real)), 0.0);
        if (threadIdx.x == 0) A[j + j * lda] = piv;
        __syncthreads();

        // ---- 2) compute non-diagonal rows i < j, column j ----
        // Each thread handles one row i above the diagonal.
        // We need bdot = sum_{k>j} A(i,k) * conj(A(j,k)),
        // then A(i,j) = (A(i,j) - bdot) / piv.
        for (int i = threadIdx.x; i < j; i += blockDim.x) {
            T bdot = T(0.0);
            for (int k = j + 1; k < jb; ++k) {
                bdot += A[i + k * lda] * conj(A[j + k * lda]);
            }
            A[i + j * lda] = (A[i + j * lda] - bdot) / piv;
        }
        __syncthreads();
    }
}

/**
 * @brief  Blocked Cholesky A = U * U^H  from the bottom-right corner.
 *
 *         On input,  the upper triangle of dA holds the Hermitian matrix.
 *         On output, the upper triangle holds U (the factor);
 *                    the strictly-lower triangle is left untouched.
 *
 * @tparam T  std::complex<double> or std::complex<float>.
 * @param n     Order of matrix dA.
 * @param dA    Device pointer, dimension (lda, n).
 * @param lda   Leading dimension (lda >= n).
 * @param nb    Panel block size.
 * @param stream  CUDA/HIP stream.
 * @param blas    cuBLAS / hipBLAS handle.
 * @return 0 on success, >0 if the matrix is not positive-definite (1-based).
 */
template<typename T>
int potrf_bottom_right(
    int n,
    T* dA, int lda,
    int nb,
    stream_t stream,
    blasH_t  blas)
{
    const T c_one      =  T(1.0);
    const T c_neg_one  = -T(1.0);

    constexpr auto OP_N = cublasOperation_t::CUBLAS_OP_N;
    constexpr auto OP_C = cublasOperation_t::CUBLAS_OP_C;

    if (n <= 0) return 0;
    assert(nb <= 1024 && "potrf_UxUH_kernel uses 1 block approach; nb <= 1024");

    int* d_info = nullptr;
    DEVICE_CHECK(cudaMalloc(&d_info, sizeof(int)));

    for (int j = n - nb; j >= 0; j -= nb) {
        int jb = std::min(nb, n - j);

        // -------- 1. herk: update diagonal block with right-side columns --------
        if (j + jb < n) {
            BLAS_CHECK(cublasZherk(
                blas,
                CUBLAS_FILL_MODE_UPPER,
                OP_N,
                jb,
                n - j - jb,
                &c_neg_one,
                (const cuDoubleComplex*)(dA + j + size_t(j + jb) * lda), lda,
                &c_one,
                (cuDoubleComplex*)(dA + j + size_t(j) * lda), lda));
        }

        // -------- 2. factorize diagonal block:  U * U^H  (custom kernel) --------
        {
            int shm_bytes = jb * sizeof(T);
            potrf_UxUH_kernel<T><<<1, jb, shm_bytes, stream>>>(
                jb, dA + j + size_t(j) * lda, lda, d_info);

            int info = 0;
            DEVICE_CHECK(cudaMemcpyAsync(&info, d_info, sizeof(int),
                                         cudaMemcpyDeviceToHost, stream));
            DEVICE_CHECK(cudaStreamSynchronize(stream));
            if (info != 0) {
                DEVICE_CHECK(cudaFree(d_info));
                return j + info;
            }
        }

        if (j == 0) break;

        // -------- 3. gemm: update left block column --------
        // A(0:j, j:j+jb) -= A(0:j, j+jb:n) * A(j:j+jb, j+jb:n)^H
        if (j + jb < n) {
            BLAS_CHECK(cublasZgemm(
                blas,
                OP_N, OP_C,
                j, jb, n - j - jb,
                &c_neg_one,
                (const cuDoubleComplex*)(dA + 0 + size_t(j + jb) * lda), lda,
                (const cuDoubleComplex*)(dA + j + size_t(j + jb) * lda), lda,
                &c_one,
                (cuDoubleComplex*)(dA + 0 + size_t(j) * lda), lda));
        }

        // -------- 4. trsm: solve  X * U^H = B  →  X = B * inv(U)^H --------
        // U is stored in the upper triangle of dA(j:j+jb, j:j+jb).
        // side=RIGHT, uplo=UPPER, trans=CONJ_TRANS, diag=NON_UNIT.
        BLAS_CHECK(cublasZtrsm(
            blas,
            CUBLAS_SIDE_RIGHT,
            CUBLAS_FILL_MODE_UPPER,
            OP_C,
            CUBLAS_DIAG_NON_UNIT,
            j, jb,
            &c_one,
            (const cuDoubleComplex*)(dA + j + size_t(j) * lda), lda,
            (cuDoubleComplex*)(dA + 0 + size_t(j) * lda), lda));
    }

    DEVICE_CHECK(cudaFree(d_info));
    return 0;
}

// ---- explicit instantiation for std::complex<double> ----
template int potrf_bottom_right<std::complex<double>>(
    int, std::complex<double>*, int, int, stream_t, blasH_t);
