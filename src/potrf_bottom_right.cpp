#include "potrf_bottom_right_internal.h"
#include "potrf_bottom_right_transform.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <complex>
#include <type_traits>

#include <ddla/ddla.h>

#include "ddla_connector.h"
#include "ddla_stream_impl.h"
#include "require_gpu.h"
#include "potrf.h"

namespace ddla{

namespace detail {

namespace {

constexpr int kBlockSize = 128;

template <typename T>
void update_leading(
    const DdlaHandle_t& handle,
    const char uplo,
    int leading_n, int block_n,
    const T* d_panel, int lda,
    T* d_leading)
{
    const bool upper = uplo == 'U';
    // d_leading -= d_panel * d_panel^H (or its transpose, for the lower case).
    // The unified syrk/herk take a LAPACK-style character flag, so the
    // transpose choice is the same 'N'/'T'/'C' string the CPU arm hands to
    // Fortran -- one convention across both backends.
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
        const T minus_one = T(-1);
        const T one = T(1);
        syrk<DdlaBackend::GPU, T>(handle, uplo, upper ? 'N' : 'T',
                                  leading_n, block_n,
                                  minus_one, d_panel, lda,
                                  one, d_leading, lda);
    } else {
        using Real = std::conditional_t<
            std::is_same_v<T, std::complex<float>>, float, double>;
        const Real minus_one = Real(-1);
        const Real one = Real(1);
        herk<DdlaBackend::GPU, T>(handle, uplo, upper ? 'N' : 'C',
                                  leading_n, block_n,
                                  minus_one, d_panel, lda,
                                  one, d_leading, lda);
    }
}

} // namespace

template <typename T>
void potrf_bottom_right_block(
    const char& uplo, const int& n, T* d_A, const int& lda,
    const int& global_offset,
    T* d_work, int* d_info, int& info,
    const DdlaHandle_t& handle)
{
    assert(uplo == 'U' || uplo == 'L');
    assert(n >= 0);
    assert(global_offset >= 0);
    assert(handle != nullptr);

    info = 0;
    if(n == 0){
        return;
    }

    assert(d_A != nullptr);
    assert(d_work != nullptr);
    assert(d_info != nullptr);
    assert(lda >= n);

    const runtimeStream_t stream = handle->stream;
    RUNTIME_CHECK(runtimeMemsetAsync(d_info, 0, sizeof(int), stream));
    const deblasFillMode_t solver_uplo = uplo == 'U'
        ? DEBLAS_FILL_MODE_LOWER
        : DEBLAS_FILL_MODE_UPPER;
    if(uplo == 'U'){
        reverse_upper_to_lower(n, d_A, lda, d_work, n, stream);
    }else{
        reverse_lower_to_upper(n, d_A, lda, d_work, n, stream);
    }
    SOLVER_CHECK(desolverPotrf(
        handle->solverH, solver_uplo,
        n, d_work, n, d_info));

    int solver_info = 0;
    RUNTIME_CHECK(runtimeMemcpyAsync(
        &solver_info, d_info, sizeof(int),
        runtimeMemcpyDeviceToHost, stream));
    RUNTIME_CHECK(runtimeStreamSynchronize(stream));
    if(solver_info != 0){
        info = global_offset + n - solver_info + 1;
        return;
    }

    if(uplo == 'U'){
        reverse_lower_to_upper(n, d_work, n, d_A, lda, stream);
    }else{
        reverse_upper_to_lower(n, d_work, n, d_A, lda, stream);
    }
}

template void potrf_bottom_right_block<float>(
    const char&, const int&, float*, const int&, const int&, float*, int*, int&,
    const DdlaHandle_t&);
template void potrf_bottom_right_block<double>(
    const char&, const int&, double*, const int&, const int&, double*, int*, int&,
    const DdlaHandle_t&);
template void potrf_bottom_right_block<std::complex<float>>(
    const char&, const int&, std::complex<float>*, const int&, const int&,
    std::complex<float>*, int*, int&, const DdlaHandle_t&);
template void potrf_bottom_right_block<std::complex<double>>(
    const char&, const int&, std::complex<double>*, const int&, const int&,
    std::complex<double>*, int*, int&, const DdlaHandle_t&);

} // namespace detail

template <typename T>
void potrf_bottom_right(
    const char& uplo, const int& n, T* d_A, const int& lda,
    int& info, const DdlaHandle_t& handle)
{
    detail::require_gpu_backend(handle, "potrf_bottom_right");
    assert(uplo == 'U' || uplo == 'L');
    assert(n >= 0);
    assert(handle != nullptr);

    info = 0;
    if(n == 0){
        return;
    }

    assert(d_A != nullptr);
    assert(lda >= n);

    const runtimeStream_t stream = handle->stream;
    T* d_work = nullptr;
    int* d_info = nullptr;
    RUNTIME_CHECK(runtimeMallocAsync(
        reinterpret_cast<void**>(&d_work),
        static_cast<std::size_t>(detail::kBlockSize)
            * detail::kBlockSize * sizeof(T),
        stream));
    RUNTIME_CHECK(runtimeMallocAsync(
        reinterpret_cast<void**>(&d_info), sizeof(int), stream));

    const int last_block_start =
        ((n - 1) / detail::kBlockSize) * detail::kBlockSize;
    for(int block_start = last_block_start;
        block_start >= 0;
        block_start -= detail::kBlockSize){
        const int block_n = std::min(detail::kBlockSize, n - block_start);
        T* const d_diag = d_A + block_start
                        + static_cast<std::size_t>(block_start) * lda;
        detail::potrf_bottom_right_block(
            uplo, block_n, d_diag, lda, block_start,
            d_work, d_info, info, handle);
        if(info != 0){
            break;
        }
        if(block_start == 0){
            continue;
        }

        T* const d_panel = uplo == 'U'
            ? d_A + static_cast<std::size_t>(block_start) * lda
            : d_A + block_start;
        const T one = T(1);
        // For real T a conjugate transpose is a plain transpose; for the
        // complex types 'C' is required.  One flag, both backends.
        const char trans = (std::is_same_v<T, float> || std::is_same_v<T, double>)
            ? 'T' : 'C';
        if(uplo == 'U'){
            trsm<DdlaBackend::GPU, T>(handle, 'R', 'U', 'C', 'N',
                                      block_start, block_n, one,
                                      d_diag, lda, d_panel, lda);
        }else{
            trsm<DdlaBackend::GPU, T>(handle, 'L', 'L', trans, 'N',
                                      block_n, block_start, one,
                                      d_diag, lda, d_panel, lda);
        }
        detail::update_leading(
            handle,
            uplo,
            block_start, block_n,
            d_panel, lda,
            d_A);
    }

    RUNTIME_CHECK(runtimeFreeAsync(d_work, stream));
    RUNTIME_CHECK(runtimeFreeAsync(d_info, stream));
    RUNTIME_CHECK(runtimeStreamSynchronize(stream));
}

template void potrf_bottom_right<float>(
    const char&, const int&, float*, const int&, int&, const DdlaHandle_t&);
template void potrf_bottom_right<double>(
    const char&, const int&, double*, const int&, int&, const DdlaHandle_t&);
template void potrf_bottom_right<std::complex<float>>(
    const char&, const int&, std::complex<float>*, const int&, int&,
    const DdlaHandle_t&);
template void potrf_bottom_right<std::complex<double>>(
    const char&, const int&, std::complex<double>*, const int&, int&,
    const DdlaHandle_t&);

} // namespace ddla
