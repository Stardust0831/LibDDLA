#ifndef DDLA_HELPERS_H
#define DDLA_HELPERS_H

#include <complex>

#if defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)
#include <thrust/complex.h>

namespace ddla::detail {

// Kernel arithmetic uses thrust::complex; the public API keeps std::complex.
// This maps types only: it does not establish pointer alignment or permit
// dereferencing a reinterpreted host scalar. Construct by-value scalars below.
template <typename T>
struct DeviceScalar {
    using type = T;
    using real = T;
};

template <typename R>
struct DeviceScalar<std::complex<R>> {
    using type = thrust::complex<R>;
    using real = R;
};

template <typename T>
using DeviceScalarT = typename DeviceScalar<T>::type;

template <typename T>
inline T to_device_scalar(const T& value) { return value; }

template <typename R>
inline thrust::complex<R> to_device_scalar(const std::complex<R>& value)
{
    return {value.real(), value.imag()};
}

#if defined(__CUDACC__) || defined(__HIP__)
template <typename T>
__device__ __forceinline__ T device_conj(const T& value, bool = true) { return value; }

template <typename R>
__device__ __forceinline__ thrust::complex<R> device_conj(const thrust::complex<R>& value, bool conjugate = true)
{
    return conjugate ? thrust::conj(value) : value;
}
#endif

} // namespace ddla::detail
#endif // DDLA_USE_CUDA || DDLA_USE_HIP

#endif // DDLA_HELPERS_H
