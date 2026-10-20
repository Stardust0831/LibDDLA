// ---------------------------------------------------------------------------
// Backend-templated Cholesky factorization of a Hermitian positive-definite matrix (LAPACK ?potrf):
//     A = U^H * U  (uplo = 'U')        or        A = L * L^H  (uplo = 'L').
//
// Structure follows src/getrf.cpp: one definition covering CPU (Fortran LAPACK ?potrf on host pointers) and GPU
// (cuSOLVER / hipSOLVER potrf on device pointers, enqueued on the handle's stream), selected with `if constexpr`
// on the Backend template parameter, the arms guarded by DDLA_HAS_CPU / DDLA_HAS_GPU.  The vendor calls are private
// to this translation unit; the public interface is the templated ddla::potrf declared in <ddla/ddla.h>.
//
// GPU arm: scratch sized with ?potrf_bufferSize, allocated and freed stream-ordered on the solver's stream, so the
// call stays asynchronous; `info` is valid once the caller synchronizes the stream.
// ---------------------------------------------------------------------------
#include <ddla/ddla.h>
#include <ddla/ddla_handle_t.h>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "blas_enums.h"
#include "ddla_connector.h"
#include "ddla_stream_impl.h"

#if DDLA_HAS_CPU
// Host LAPACK ?potrf (Fortran ABI; the CPU BLAS library must provide LAPACK -- OpenBLAS and MKL do).
extern "C" {
void spotrf_(const char*, const int*, float*, const int*, int*);
void dpotrf_(const char*, const int*, double*, const int*, int*);
void cpotrf_(const char*, const int*, std::complex<float>*, const int*, int*);
void zpotrf_(const char*, const int*, std::complex<double>*, const int*, int*);
}
#endif

namespace ddla {

namespace {

inline const char* potrf_backend_name(DdlaBackend backend)
{
    return backend == DdlaBackend::CPU ? "CPU" : "GPU";
}

#if DDLA_HAS_CPU
inline void cpu_potrf(char uplo, int n, float* A, int lda, int* info)
{
    spotrf_(&uplo, &n, A, &lda, info);
}

inline void cpu_potrf(char uplo, int n, double* A, int lda, int* info)
{
    dpotrf_(&uplo, &n, A, &lda, info);
}

inline void cpu_potrf(char uplo, int n, std::complex<float>* A, int lda, int* info)
{
    cpotrf_(&uplo, &n, A, &lda, info);
}

inline void cpu_potrf(char uplo, int n, std::complex<double>* A, int lda, int* info)
{
    zpotrf_(&uplo, &n, A, &lda, info);
}
#endif // DDLA_HAS_CPU

#if DDLA_HAS_GPU
#if DDLA_GPU_VENDOR

// Per scalar type: the buffer-size query and the factorization itself.
inline gpusolverStatus_t potrf_buffer_size(gpusolverHandle_t h, gpublasFillMode_t uplo, int n, float* A, int lda, int* lwork)
{
    return gpusolverSpotrf_bufferSize(h, gpusolverFillMode(uplo), n, A, lda, lwork);
}
inline gpusolverStatus_t potrf_buffer_size(gpusolverHandle_t h, gpublasFillMode_t uplo, int n, double* A, int lda, int* lwork)
{
    return gpusolverDpotrf_bufferSize(h, gpusolverFillMode(uplo), n, A, lda, lwork);
}
inline gpusolverStatus_t potrf_buffer_size(gpusolverHandle_t h, gpublasFillMode_t uplo, int n, std::complex<float>* A,
                                           int lda, int* lwork)
{
    return gpusolverCpotrf_bufferSize(h, gpusolverFillMode(uplo), n, (gpusolverComplex*)A, lda, lwork);
}
inline gpusolverStatus_t potrf_buffer_size(gpusolverHandle_t h, gpublasFillMode_t uplo, int n, std::complex<double>* A,
                                           int lda, int* lwork)
{
    return gpusolverZpotrf_bufferSize(h, gpusolverFillMode(uplo), n, (gpusolverDoubleComplex*)A, lda, lwork);
}

inline gpusolverStatus_t potrf_call(gpusolverHandle_t h, gpublasFillMode_t uplo, int n, float* A, int lda, float* work,
                                    int lwork, int* info)
{
    return gpusolverSpotrf(h, gpusolverFillMode(uplo), n, A, lda, work, lwork, info);
}
inline gpusolverStatus_t potrf_call(gpusolverHandle_t h, gpublasFillMode_t uplo, int n, double* A, int lda, double* work,
                                    int lwork, int* info)
{
    return gpusolverDpotrf(h, gpusolverFillMode(uplo), n, A, lda, work, lwork, info);
}
inline gpusolverStatus_t potrf_call(gpusolverHandle_t h, gpublasFillMode_t uplo, int n, std::complex<float>* A, int lda,
                                    std::complex<float>* work, int lwork, int* info)
{
    return gpusolverCpotrf(h, gpusolverFillMode(uplo), n, (gpusolverComplex*)A, lda, (gpusolverComplex*)work, lwork, info);
}
inline gpusolverStatus_t potrf_call(gpusolverHandle_t h, gpublasFillMode_t uplo, int n, std::complex<double>* A, int lda,
                                    std::complex<double>* work, int lwork, int* info)
{
    return gpusolverZpotrf(h, gpusolverFillMode(uplo), n, (gpusolverDoubleComplex*)A, lda,
                           (gpusolverDoubleComplex*)work, lwork, info);
}

template <typename T>
gpusolverStatus_t gpu_potrf(gpusolverHandle_t handle, gpublasFillMode_t uplo, int n, T* A, int lda, int* info)
{
    gpuStream_t stream;
    SOLVER_CHECK(gpusolverGetStream(handle, &stream));

    int lwork = 0;
    SOLVER_CHECK(potrf_buffer_size(handle, uplo, n, A, lda, &lwork));

    T* work = nullptr;
    if (lwork > 0)
        GPU_CHECK(gpuMallocAsync(reinterpret_cast<void**>(&work), static_cast<std::size_t>(lwork) * sizeof(T), stream));

    const gpusolverStatus_t status = potrf_call(handle, uplo, n, A, lda, work, lwork, info);

    if (lwork > 0)
        GPU_CHECK(gpuFreeAsync(work, stream));
    return status;
}

#else // DDLA_GPU_VENDOR

template <typename T>
gpusolverStatus_t gpu_potrf(gpusolverHandle_t, gpublasFillMode_t, int, T*, int, int*)
{
    throw std::runtime_error("potrf: no GPU vendor selected at compile time");
}

#endif // DDLA_GPU_VENDOR
#endif // DDLA_HAS_GPU

} // anonymous namespace

template <DdlaBackend Backend, typename T>
void potrf(const DdlaHandle_t& handle, char uplo, int n, T* A, int lda, int* info)
{
    static_assert(Backend != DdlaBackend::CPU || DDLA_HAS_CPU,
                  "CPU potrf is not available in this LibDDLA build");
    static_assert(Backend != DdlaBackend::GPU || DDLA_HAS_GPU,
                  "GPU potrf is not available in this LibDDLA build");

    if (handle == nullptr) {
        throw std::runtime_error("potrf: null handle");
    }
    const DdlaBackend actual = ddlaGetBackend(handle);
    if (actual != Backend) {
        throw std::runtime_error(
            std::string("potrf: template backend ") + potrf_backend_name(Backend) +
            " does not match handle backend " + potrf_backend_name(actual));
    }
    // 'U' / 'L' in either case; anything else throws instead of silently picking a triangle.
    const gpublasFillMode_t fill = detail::to_fill_mode(uplo);
    if (n < 0) throw std::runtime_error("potrf: negative matrix size");
    if (info == nullptr) throw std::runtime_error("potrf: null info");

    if (n == 0) {
        // Nothing to factor: success, reported where the caller will look for it.
        if constexpr (Backend == DdlaBackend::CPU) {
            *info = 0;
        } else {
            DDLA_CHECK(ddlaMemset(handle, info, 0, sizeof(int)));
        }
        return;
    }
    if (A == nullptr) throw std::runtime_error("potrf: null A");
    if (lda < std::max(1, n)) throw std::runtime_error("potrf: lda too small");

    if constexpr (Backend == DdlaBackend::CPU) {
#if DDLA_HAS_CPU
        cpu_potrf(fill == GPUBLAS_FILL_MODE_UPPER ? 'U' : 'L', n, A, lda, info);
#endif
    } else {
#if DDLA_HAS_GPU
        SOLVER_CHECK(gpu_potrf(handle->solverH, fill, n, A, lda, info));
#endif
    }
}

#define INSTANTIATE_POTRF(BACKEND, TYPE)                                     \
    template void potrf<BACKEND, TYPE>(                                      \
        const DdlaHandle_t&, char, int, TYPE*, int, int*)

#if DDLA_HAS_CPU
INSTANTIATE_POTRF(DdlaBackend::CPU, float);
INSTANTIATE_POTRF(DdlaBackend::CPU, double);
INSTANTIATE_POTRF(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_POTRF(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_POTRF(DdlaBackend::GPU, float);
INSTANTIATE_POTRF(DdlaBackend::GPU, double);
INSTANTIATE_POTRF(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_POTRF(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_POTRF

} // namespace ddla
