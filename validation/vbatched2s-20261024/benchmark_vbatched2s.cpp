#include <ddla/ddla.h>
#include <ddla/ddla_handle_t.h>
#include <mpi.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

struct Storage
{
    ddla::DdlaHandle_t handle;
    std::vector<void*> pointers;
    template <typename T> T* copy(const std::vector<T>& host)
    {
        T* device = nullptr;
        ddla::DDLA_CHECK(ddla::ddlaMalloc(handle, &device, host.size() * sizeof(T)));
        pointers.push_back(device);
        ddla::DDLA_CHECK(ddla::ddlaMemcpy(handle, device, host.data(), host.size() * sizeof(T)));
        ddla::DDLA_CHECK(ddla::ddlaSynchronize(handle));
        return device;
    }
    ~Storage()
    {
        for (void* pointer : pointers)
            ddla::ddlaFree(handle, pointer);
        ddla::ddlaSynchronize(handle);
    }
};

template <typename T>
void measure(ddla::DdlaHandle_t handle, const char* type, int batches, int iterations)
{
    Storage storage{handle, {}};
    std::vector<int> m(batches), n(batches), k(batches), n1(batches);
    std::vector<std::size_t> a(batches + 1), b(batches + 1), ab(batches + 1), tmp(batches + 1), c(batches + 1);
    for (int i = 0; i < batches; ++i)
    {
        m[i] = 8 + (i % 3) * 4;
        n[i] = 16 + (i % 5) * 8;
        k[i] = 12 + (i % 4) * 4;
        n1[i] = 12 + (i % 3) * 4;
        a[i+1] = a[i] + m[i] * k[i];
        b[i+1] = b[i] + k[i] * n[i];
        ab[i+1] = ab[i] + n[i] * n1[i];
        tmp[i+1] = tmp[i] + m[i] * n[i];
        c[i+1] = c[i] + m[i] * n1[i];
    }
    auto matrix = [&](const std::vector<std::size_t>& offsets, T initial) {
        T* data = storage.copy(std::vector<T>(offsets.back(), initial));
        std::vector<T*> pointers(batches);
        for (int i = 0; i < batches; ++i) pointers[i] = data + offsets[i];
        return std::make_pair(data, storage.copy(pointers));
    };
    auto A = matrix(a, T(0.01));
    auto B = matrix(b, T(0.02));
    auto AB = matrix(ab, T(0.03));
    auto temporary = matrix(tmp, T{});
    auto C = matrix(c, T{});
    int* dm = storage.copy(m);
    int* dn = storage.copy(n);
    int* dk = storage.copy(k);
    int* dn1 = storage.copy(n1);
    auto invoke = [&] {
        ddla::gemmVbatched2s(
            handle, 'N', 'N', dm, dn, dk, T(1),
            const_cast<const T* const*>(A.second), dm,
            const_cast<const T* const*>(B.second), dk, T{}, temporary.second, dm,
            'N', 'N', dm, dn1, dn, T(1),
            const_cast<const T* const*>(AB.second), dm, dn, T{}, C.second, dm,
            true, batches, &batches, 1);
    };
    for (int i = 0; i < 20; ++i) invoke();
    ddla::DDLA_CHECK(ddla::ddlaSynchronize(handle));
    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) invoke();
    ddla::DDLA_CHECK(ddla::ddlaSynchronize(handle));
    const double us = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count()
        * 1e6 / iterations;
    std::vector<T> result(c.back());
    ddla::DDLA_CHECK(ddla::ddlaMemcpy(handle, result.data(), C.first, result.size() * sizeof(T)));
    ddla::DDLA_CHECK(ddla::ddlaSynchronize(handle));
    double error = 0;
    for (int i = 0; i < batches; ++i)
        for (std::size_t j = c[i]; j < c[i+1]; ++j)
            error = std::max(error, static_cast<double>(std::abs(result[j] - T(0.000006 * k[i] * n[i]))));
    if (error > 1e-12) throw std::runtime_error("microbenchmark numerical mismatch");
    std::cout << std::setprecision(12) << "{\"type\":\"" << type << "\",\"batches\":" << batches
              << ",\"iterations\":" << iterations << ",\"microseconds\":" << us
              << ",\"max_error\":" << error << "}" << std::endl;
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    ddla::DdlaHandle_t handle = nullptr;
    ddla::DDLA_CHECK(ddla::ddlaInit(handle, ddla::DdlaBackend::GPU));
    ddla::DDLA_CHECK(ddla::ddlaSet(handle, MPI_COMM_WORLD));
    const int iterations = argc > 1 ? std::atoi(argv[1]) : 1000;
    for (int batches : {1, 64, 512})
    {
        measure<double>(handle, "double", batches, iterations);
        measure<std::complex<double>>(handle, "complex_double", batches, iterations);
    }
    ddla::DDLA_CHECK(ddla::ddlaDestroy(handle));
    MPI_Finalize();
}
