// ---------------------------------------------------------------------------
// Backend-templated LU factorization with partial pivoting (LAPACK ?getrf):  A = P * L * U.
//
// Structure follows src/scal.cpp / src/geam.cpp: one definition covering CPU (Fortran LAPACK ?getrf on host pointers)
// and GPU (cuSOLVER / hipSOLVER getrf on device pointers, enqueued on the handle's stream), selected with
// `if constexpr` on the Backend template parameter, the arms guarded by the BUILD CAPABILITY macros
// (DDLA_HAS_CPU / DDLA_HAS_GPU).  The vendor calls are private to this translation unit; the public interface is the
// templated ddla::getrf declared in <ddla/ddla.h>.
//
// GPU arm: the vendor routine needs a scratch buffer.  It is sized with the ?getrf_bufferSize query, allocated
// stream-ordered on the solver's stream (the stream the handle bound it to) and freed there, so the call stays
// asynchronous: nothing is synchronized, `info` is valid once the caller synchronizes the stream.
// ---------------------------------------------------------------------------
#include <ddla/ddla.h>
#include <ddla/ddla_handle_t.h>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "ddla_connector.h"
#include "ddla_stream_impl.h"

#if DDLA_HAS_CPU
// Host LAPACK ?getrf (Fortran ABI; the CPU BLAS library must provide LAPACK -- OpenBLAS and MKL do).
extern "C" {
void sgetrf_(const int*, const int*, float*, const int*, int*, int*);
void dgetrf_(const int*, const int*, double*, const int*, int*, int*);
void cgetrf_(const int*, const int*, std::complex<float>*, const int*, int*, int*);
void zgetrf_(const int*, const int*, std::complex<double>*, const int*, int*, int*);
}
#endif

namespace ddla {

namespace {

inline const char* getrf_backend_name(DdlaBackend backend)
{
    return backend == DdlaBackend::CPU ? "CPU" : "GPU";
}

#if DDLA_HAS_CPU
inline void cpu_getrf(int m, int n, float* A, int lda, int* ipiv, int* info)
{
    sgetrf_(&m, &n, A, &lda, ipiv, info);
}

inline void cpu_getrf(int m, int n, double* A, int lda, int* ipiv, int* info)
{
    dgetrf_(&m, &n, A, &lda, ipiv, info);
}

inline void cpu_getrf(int m, int n, std::complex<float>* A, int lda, int* ipiv, int* info)
{
    cgetrf_(&m, &n, A, &lda, ipiv, info);
}

inline void cpu_getrf(int m, int n, std::complex<double>* A, int lda, int* ipiv, int* info)
{
    zgetrf_(&m, &n, A, &lda, ipiv, info);
}
#endif // DDLA_HAS_CPU

#if DDLA_HAS_GPU
#if DDLA_GPU_VENDOR

// Per scalar type: the buffer-size query and the factorization itself.
inline gpusolverStatus_t getrf_buffer_size(gpusolverHandle_t h, int m, int n, float* A, int lda, int* lwork)
{
    return gpusolverSgetrf_bufferSize(h, m, n, A, lda, lwork);
}
inline gpusolverStatus_t getrf_buffer_size(gpusolverHandle_t h, int m, int n, double* A, int lda, int* lwork)
{
    return gpusolverDgetrf_bufferSize(h, m, n, A, lda, lwork);
}
inline gpusolverStatus_t getrf_buffer_size(gpusolverHandle_t h, int m, int n, std::complex<float>* A, int lda, int* lwork)
{
    return gpusolverCgetrf_bufferSize(h, m, n, (gpusolverComplex*)A, lda, lwork);
}
inline gpusolverStatus_t getrf_buffer_size(gpusolverHandle_t h, int m, int n, std::complex<double>* A, int lda, int* lwork)
{
    return gpusolverZgetrf_bufferSize(h, m, n, (gpusolverDoubleComplex*)A, lda, lwork);
}

inline gpusolverStatus_t getrf_call(gpusolverHandle_t h, int m, int n, float* A, int lda, float* work, int lwork,
                                    int* ipiv, int* info)
{
    return gpusolverSgetrf(h, m, n, A, lda, work, lwork, ipiv, info);
}
inline gpusolverStatus_t getrf_call(gpusolverHandle_t h, int m, int n, double* A, int lda, double* work, int lwork,
                                    int* ipiv, int* info)
{
    return gpusolverDgetrf(h, m, n, A, lda, work, lwork, ipiv, info);
}
inline gpusolverStatus_t getrf_call(gpusolverHandle_t h, int m, int n, std::complex<float>* A, int lda,
                                    std::complex<float>* work, int lwork, int* ipiv, int* info)
{
    return gpusolverCgetrf(h, m, n, (gpusolverComplex*)A, lda, (gpusolverComplex*)work, lwork, ipiv, info);
}
inline gpusolverStatus_t getrf_call(gpusolverHandle_t h, int m, int n, std::complex<double>* A, int lda,
                                    std::complex<double>* work, int lwork, int* ipiv, int* info)
{
    return gpusolverZgetrf(h, m, n, (gpusolverDoubleComplex*)A, lda, (gpusolverDoubleComplex*)work, lwork, ipiv, info);
}

template <typename T>
gpusolverStatus_t gpu_getrf(gpusolverHandle_t handle, int m, int n, T* A, int lda, int* ipiv, int* info)
{
    gpuStream_t stream;
    SOLVER_CHECK(gpusolverGetStream(handle, &stream));

    int lwork = 0;
    SOLVER_CHECK(getrf_buffer_size(handle, m, n, A, lda, &lwork));

    T* work = nullptr;
    if (lwork > 0)
        GPU_CHECK(gpuMallocAsync(reinterpret_cast<void**>(&work), static_cast<std::size_t>(lwork) * sizeof(T), stream));

    const gpusolverStatus_t status = getrf_call(handle, m, n, A, lda, work, lwork, ipiv, info);

    if (lwork > 0)
        GPU_CHECK(gpuFreeAsync(work, stream));
    return status;
}

#else // DDLA_GPU_VENDOR

template <typename T>
gpusolverStatus_t gpu_getrf(gpusolverHandle_t, int, int, T*, int, int*, int*)
{
    throw std::runtime_error("getrf: no GPU vendor selected at compile time");
}

#endif // DDLA_GPU_VENDOR
#endif // DDLA_HAS_GPU

} // anonymous namespace

template <DdlaBackend Backend, typename T>
void getrf(const DdlaHandle_t& handle, int m, int n, T* A, int lda, int* ipiv, int* info)
{
    static_assert(Backend != DdlaBackend::CPU || DDLA_HAS_CPU,
                  "CPU getrf is not available in this LibDDLA build");
    static_assert(Backend != DdlaBackend::GPU || DDLA_HAS_GPU,
                  "GPU getrf is not available in this LibDDLA build");

    if (handle == nullptr) {
        throw std::runtime_error("getrf: null handle");
    }
    const DdlaBackend actual = ddlaGetBackend(handle);
    if (actual != Backend) {
        throw std::runtime_error(
            std::string("getrf: template backend ") + getrf_backend_name(Backend) +
            " does not match handle backend " + getrf_backend_name(actual));
    }
    if (m < 0 || n < 0) throw std::runtime_error("getrf: negative matrix size");
    if (info == nullptr) throw std::runtime_error("getrf: null info");

    if (m == 0 || n == 0) {
        // Nothing to factor: success, reported where the caller will look for it.
        if constexpr (Backend == DdlaBackend::CPU) {
            *info = 0;
        } else {
            DDLA_CHECK(ddlaMemset(handle, info, 0, sizeof(int)));
        }
        return;
    }
    if (A == nullptr) throw std::runtime_error("getrf: null A");
    if (ipiv == nullptr) throw std::runtime_error("getrf: null ipiv");
    if (lda < std::max(1, m)) throw std::runtime_error("getrf: lda too small");

    if constexpr (Backend == DdlaBackend::CPU) {
#if DDLA_HAS_CPU
        cpu_getrf(m, n, A, lda, ipiv, info);
#endif
    } else {
#if DDLA_HAS_GPU
        SOLVER_CHECK(gpu_getrf(handle->solverH, m, n, A, lda, ipiv, info));
#endif
    }
}

#define INSTANTIATE_GETRF(BACKEND, TYPE)                                     \
    template void getrf<BACKEND, TYPE>(                                      \
        const DdlaHandle_t&, int, int, TYPE*, int, int*, int*)

#if DDLA_HAS_CPU
INSTANTIATE_GETRF(DdlaBackend::CPU, float);
INSTANTIATE_GETRF(DdlaBackend::CPU, double);
INSTANTIATE_GETRF(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_GETRF(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_GETRF(DdlaBackend::GPU, float);
INSTANTIATE_GETRF(DdlaBackend::GPU, double);
INSTANTIATE_GETRF(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_GETRF(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_GETRF

} // namespace ddla
