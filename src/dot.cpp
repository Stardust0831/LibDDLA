// ---------------------------------------------------------------------------
// Unified backend-templated conjugate dot product (result := x^H * y).
//
// Structure follows src/herk.cpp exactly: one definition covering CPU (Fortran
// BLAS) and GPU (cuBLAS/hipBLAS), selected with `if constexpr` on the Backend
// template parameter, with the arms themselves guarded by the BUILD CAPABILITY
// macros (DDLA_HAS_CPU / DDLA_HAS_GPU) and the vendor chosen per scalar arm by
// the PER-TU macros (DDLA_USE_CUDA / DDLA_USE_HIP).
//
// Why the capability guard matters here: the previous design put a `#else`
// CPU arm inside src/dot.h selected by `!defined(DDLA_USE_CUDA) &&
// !defined(DDLA_USE_HIP)`.  In a dual build (DDLA_USE_CPU + DDLA_USE_CUDA)
// every TU defines DDLA_USE_CUDA, so that arm was never compiled -- while
// DDLA_HAS_CPU was 1 and the CPU instantiation of this function still existed.
// A CPU-backend call would therefore have reached cublasDdotc with HOST
// pointers.  Guarding on DDLA_HAS_CPU removes that class of bug.
//
// POINTER SEMANTICS -- the result is written THROUGH A POINTER, not returned
// by value, and the pointer's address space depends on the backend:
//   * GPU handles: `result` is a DEVICE pointer.  cuBLAS/hipBLAS default to
//     HOST pointer mode, in which the vendor routine dereferences the argument
//     as a host address; a non-host-mapped device buffer then faults.  Every
//     GPU overload therefore opens a detail::blas_device_pointer_mode_scope,
//     which switches the handle to DEVICE mode for the duration of the call
//     and restores the previous mode on scope exit.
//   * CPU handles: `result` is a HOST pointer.
//
// The file is named dot.cpp to mirror the private header it replaces
// (src/dot.h); the wrapper family itself is Dotc (conjugate in the first
// argument) and the templated entry point is ddla::dotc, so this TU defines
// deblasDotc.  For the real scalar types xDOTC degenerates to the plain dot
// product.
//
// CPU ARM OF THE COMPLEX OVERLOADS: they accumulate in plain C++ instead of
// calling xDOTC_/zDOTC_.  The Fortran ABI for a function returning a COMPLEX
// value is compiler-dependent (hidden first argument vs. register return), so
// the project avoids declaring those symbols; xNRM2 returns a REAL scalar and
// is safe, complex-valued function returns are not.  The real cases call the
// BLAS as usual.
// ---------------------------------------------------------------------------
#include <ddla/ddla.h>
#include <ddla/ddla_handle_t.h>

#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "blas_enums.h"
#include "blas_pointer_mode.h"
#include "ddla_connector.h"
#include "ddla_stream_impl.h"

#if DDLA_HAS_CPU
extern "C" {
// Real-valued (or real-typed) function returns: safe ABI.  The complex
// xDOTC/zDOTC are deliberately NOT declared -- see the header comment above;
// the complex CPU arm accumulates in C++ instead.
float  sdot_(const int*, const float*, const int*, const float*, const int*);
double ddot_(const int*, const double*, const int*, const double*, const int*);
}
#endif

// ---------------------------------------------------------------------------
// GPU arm: the vendor calls.  Kept as free deblasDotc overloads so the
// dispatch above stays a single BLAS_CHECK call per scalar type, exactly as
// ddla::gemm does it inline.  Private to this translation unit -- the public
// interface is the templated ddla::dotc declared in <ddla/ddla.h>.
//
// The complex overloads cast the const inputs away: the vendor prototypes take
// non-const operands even though xDOTC only reads them.  This matches the
// pre-unification header (src/dot.h) verbatim.
//
// The pointer-mode scope is the FIRST statement of each vendor arm so that the
// handle is in DEVICE mode before the vendor routine dereferences `result`,
// and the destructor restores the caller's mode on the way out (including on
// the throwing path, since the scope is an automatic object).
// ---------------------------------------------------------------------------
#if DDLA_HAS_GPU

namespace ddla {

inline deblasStatus_t deblasDotc(
    deblasHandle_t handle, int n,
    const float* x, int incx,
    const float* y, int incy,
    float* result
)
{
    #if defined(DDLA_USE_HIP)
    detail::blas_device_pointer_mode_scope pm(handle);
    return hipblasSdot(handle, n, x, incx, y, incy, result);
    #elif defined(DDLA_USE_CUDA)
    detail::blas_device_pointer_mode_scope pm(handle);
    return cublasSdot(handle, n, x, incx, y, incy, result);
    #else
    // Neither vendor macro: this can only happen if the translation unit was
    // built without a GPU language while DDLA_HAS_GPU is set, which the build
    // system does not do (see the LANGUAGE CUDA/HIP source lists).  Throw
    // rather than fall through to a host pointer.
    throw std::runtime_error("deblasDotc: no GPU vendor selected at compile time");
    #endif
}

inline deblasStatus_t deblasDotc(
    deblasHandle_t handle, int n,
    const double* x, int incx,
    const double* y, int incy,
    double* result
)
{
    #if defined(DDLA_USE_HIP)
    detail::blas_device_pointer_mode_scope pm(handle);
    return hipblasDdot(handle, n, x, incx, y, incy, result);
    #elif defined(DDLA_USE_CUDA)
    detail::blas_device_pointer_mode_scope pm(handle);
    return cublasDdot(handle, n, x, incx, y, incy, result);
    #else
    throw std::runtime_error("deblasDotc: no GPU vendor selected at compile time");
    #endif
}

inline deblasStatus_t deblasDotc(
    deblasHandle_t handle, int n,
    const std::complex<float>* x, int incx,
    const std::complex<float>* y, int incy,
    std::complex<float>* result
)
{
    #if defined(DDLA_USE_HIP)
    detail::blas_device_pointer_mode_scope pm(handle);
    return hipblasCdotc(handle, n, (hipblasComplex*)x, incx, (hipblasComplex*)y, incy,
                        (hipblasComplex*)result);
    #elif defined(DDLA_USE_CUDA)
    detail::blas_device_pointer_mode_scope pm(handle);
    return cublasCdotc(handle, n, (cuFloatComplex*)x, incx, (cuFloatComplex*)y, incy,
                       (cuFloatComplex*)result);
    #else
    throw std::runtime_error("deblasDotc: no GPU vendor selected at compile time");
    #endif
}

inline deblasStatus_t deblasDotc(
    deblasHandle_t handle, int n,
    const std::complex<double>* x, int incx,
    const std::complex<double>* y, int incy,
    std::complex<double>* result
)
{
    #if defined(DDLA_USE_HIP)
    detail::blas_device_pointer_mode_scope pm(handle);
    return hipblasZdotc(handle, n, (hipblasDoubleComplex*)x, incx, (hipblasDoubleComplex*)y, incy,
                        (hipblasDoubleComplex*)result);
    #elif defined(DDLA_USE_CUDA)
    detail::blas_device_pointer_mode_scope pm(handle);
    return cublasZdotc(handle, n, (cuDoubleComplex*)x, incx, (cuDoubleComplex*)y, incy,
                       (cuDoubleComplex*)result);
    #else
    throw std::runtime_error("deblasDotc: no GPU vendor selected at compile time");
    #endif
}

} // namespace ddla

#endif // DDLA_HAS_GPU

namespace ddla {

template <DdlaBackend Backend, typename T>
void dotc(
    const DdlaHandle_t& handle,
    int n,
    const T* x, int incx,
    const T* y, int incy,
    T* result)
{
    static_assert(Backend != DdlaBackend::CPU || DDLA_HAS_CPU,
                  "CPU dotc is not available in this LibDDLA build");
    static_assert(Backend != DdlaBackend::GPU || DDLA_HAS_GPU,
                  "GPU dotc is not available in this LibDDLA build");

    if (handle == nullptr) {
        throw std::runtime_error("dotc: null handle");
    }
    const DdlaBackend actual = ddlaGetBackend(handle);
    if (actual != Backend) {
        throw std::runtime_error(
            std::string("dotc: template backend ") +
            (Backend == DdlaBackend::CPU ? "CPU" : "GPU") +
            " does not match handle backend " +
            (actual == DdlaBackend::CPU ? "CPU" : "GPU"));
    }

    if constexpr (Backend == DdlaBackend::CPU) {
#if DDLA_HAS_CPU
        // HOST result pointer, set by the arm that computed the value: the
        // real types call the BLAS, the complex types accumulate in C++
        // (xDOTC's complex-return ABI is not portable -- see the file header).
        if constexpr (std::is_same_v<T, float>) {
            *result = sdot_(&n, x, &incx, y, &incy);
        } else if constexpr (std::is_same_v<T, double>) {
            *result = ddot_(&n, x, &incx, y, &incy);
        } else if constexpr (std::is_same_v<T, std::complex<float>>) {
            std::complex<float> acc(0.0f, 0.0f);
            for (int i = 0; i < n; ++i)
                acc += std::conj(x[static_cast<std::ptrdiff_t>(i) * incx])
                     * y[static_cast<std::ptrdiff_t>(i) * incy];
            *result = acc;
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
            std::complex<double> acc(0.0, 0.0);
            for (int i = 0; i < n; ++i)
                acc += std::conj(x[static_cast<std::ptrdiff_t>(i) * incx])
                     * y[static_cast<std::ptrdiff_t>(i) * incy];
            *result = acc;
        } else {
            static_assert(sizeof(T) == 0, "dotc: unsupported scalar type T");
        }
#else
        (void)handle; (void)n; (void)x; (void)incx; (void)y; (void)incy; (void)result;
#endif
    } else {
#if DDLA_HAS_GPU
        // DEVICE result pointer: the vendor arm forces DEVICE pointer mode.
        if constexpr (std::is_same_v<T, float> ||
                      std::is_same_v<T, double> ||
                      std::is_same_v<T, std::complex<float>> ||
                      std::is_same_v<T, std::complex<double>>) {
            BLAS_CHECK(deblasDotc(handle->blasH, n, x, incx, y, incy, result));
        } else {
            static_assert(sizeof(T) == 0, "dotc: unsupported scalar type T");
        }
#else
        (void)handle; (void)n; (void)x; (void)incx; (void)y; (void)incy; (void)result;
#endif
    }
}

#define INSTANTIATE_DOTC(BACKEND, TYPE)                                        \
    template void dotc<BACKEND, TYPE>(                                         \
        const DdlaHandle_t&, int, const TYPE*, int, const TYPE*, int, TYPE*)

#if DDLA_HAS_CPU
INSTANTIATE_DOTC(DdlaBackend::CPU, float);
INSTANTIATE_DOTC(DdlaBackend::CPU, double);
INSTANTIATE_DOTC(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_DOTC(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_DOTC(DdlaBackend::GPU, float);
INSTANTIATE_DOTC(DdlaBackend::GPU, double);
INSTANTIATE_DOTC(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_DOTC(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_DOTC

} // namespace ddla
