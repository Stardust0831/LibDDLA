#pragma once
#include <ddla/ddla.h>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace trtrm_test {
using namespace ddla;
template <typename T> struct Scalar { using real = T; using wide = long double; static constexpr char name = std::is_same_v<T,float> ? 's' : 'd'; };
template <typename R> struct Scalar<std::complex<R>> { using real = R; using wide = std::complex<long double>; static constexpr char name = std::is_same_v<R,float> ? 'c' : 'z'; };
template <typename T> T value(double r, double i = 0) {
    if constexpr (std::is_arithmetic_v<T>) return T(r);
    else return T(r, i);
}
template <typename T> T data(int i, int j, int seed) {
    return value<T>(std::sin(0.13 * i + 0.17 * j + seed), std::cos(0.23 * i - 0.19 * j + seed));
}
template <typename T> T poison() { return value<T>(std::numeric_limits<typename Scalar<T>::real>::quiet_NaN()); }
template <typename T> struct Buffer {
    DdlaHandle_t h; T* p = nullptr; void* allocation = nullptr; std::size_t size;
    Buffer(DdlaHandle_t handle, std::size_t count, bool component_alignment = false) : h(handle), size(count) {
        // Exercise legal std::complex alignment without assuming the stricter
        // alignment of the corresponding device arithmetic type.
        const std::size_t shift = component_alignment && !std::is_arithmetic_v<T> ? alignof(T) : 0;
        DDLA_CHECK(ddlaMalloc(h, &allocation, size * sizeof(T) + shift));
        p = reinterpret_cast<T*>(static_cast<char*>(allocation) + shift);
    }
    ~Buffer() { if (allocation) ddlaFree(h, allocation); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void put(const std::vector<T>& a) {
        if (a.size() != size) throw std::runtime_error("buffer size mismatch");
        DDLA_CHECK(ddlaMemcpy(h, p, a.data(), size * sizeof(T)));
        DDLA_CHECK(ddlaSynchronize(h));
    }
    std::vector<T> get() const {
        std::vector<T> a(size);
        DDLA_CHECK(ddlaMemcpy(h, a.data(), p, size * sizeof(T)));
        DDLA_CHECK(ddlaSynchronize(h));
        return a;
    }
};
}
