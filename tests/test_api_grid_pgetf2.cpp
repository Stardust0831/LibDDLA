#include "api_grid_test_common.h"

using namespace api_grid_test;

void check_pgetf2(const ddla::DdlaHandle_t& handle, const Shape& base)
{
    const int nb = base.nb;
    const int n = square_size(handle, base);
    ddla::DdlaDesc descA(handle);
    descA.init(n, n, nb, nb, 0, 0);

    {
        auto h_A = make_local<Complex>(descA, [=](int i, int j){ return dominant_value(i, j, n); });
        DeviceBuffer<Complex> d_A(handle, h_A.size());
        upload(handle, d_A.ptr, h_A);
        DEVICE_CHECK(deviceStreamSynchronize(handle->stream));

        std::vector<int> ipiv(descA.m_loc(), -1);
        int info = -1;
        ddla::pgetf2(n, std::min(nb, n), d_A.ptr, 0, descA, ipiv.data(), info);
        require_close(handle, "pgetf2(normal info)", std::abs(info), 0.0);
    }

    {
        const int remote_row = handle->nprows_ > 1 ? nb : 1;
        auto h_A = make_local<Complex>(descA, [=](int i, int j){
            if(j == 0) return Complex(i == remote_row ? 9.0 : 1.0 / (i + 1), 0.0);
            return i == j ? Complex(2.0, 0.0) : Complex(0.0, 0.0);
        });
        DeviceBuffer<Complex> d_A(handle, h_A.size());
        upload(handle, d_A.ptr, h_A);
        DEVICE_CHECK(deviceStreamSynchronize(handle->stream));

        std::vector<int> ipiv(descA.m_loc(), -1);
        int info = -1;
        ddla::pgetf2(n, 1, d_A.ptr, 0, descA, ipiv.data(), info);
        require_close(handle, "pgetf2(cross-row info)", std::abs(info), 0.0);
        const double pivot_err = handle->myprow_ == 0
                               ? std::abs(ipiv[0] - (remote_row + 1)) : 0.0;
        require_close(handle, "pgetf2(cross-row pivot)", pivot_err, 0.0);
    }

    {
        const int competing_row = handle->nprows_ > 1 ? nb : 3;
        auto h_A = make_local<Complex>(descA, [=](int i, int j){
            if(j == 0){
                if(i == 0) return Complex(4.0, 4.0);
                if(i == competing_row) return Complex(7.0, 0.0);
                return Complex(0.25, 0.0);
            }
            return i == j ? Complex(2.0, 0.0) : Complex(0.0, 0.0);
        });
        DeviceBuffer<Complex> d_A(handle, h_A.size());
        upload(handle, d_A.ptr, h_A);
        DEVICE_CHECK(deviceStreamSynchronize(handle->stream));

        std::vector<int> ipiv(descA.m_loc(), -1);
        int info = -1;
        ddla::pgetf2(n, 1, d_A.ptr, 0, descA, ipiv.data(), info);
        require_close(handle, "pgetf2(complex metric info)", std::abs(info), 0.0);
        const double pivot_err = handle->myprow_ == 0 ? std::abs(ipiv[0] - 1) : 0.0;
        require_close(handle, "pgetf2(complex metric pivot)", pivot_err, 0.0);
    }

    {
        auto h_A = make_local<Complex>(descA, [](int i, int j){
            if(i != j) return Complex(0.0, 0.0);
            return Complex(i == 0 ? 1e-12 : 1.0, 0.0);
        });
        DeviceBuffer<Complex> d_A(handle, h_A.size());
        upload(handle, d_A.ptr, h_A);
        DEVICE_CHECK(deviceStreamSynchronize(handle->stream));

        std::vector<int> ipiv(descA.m_loc(), -1);
        int info = -1;
        ddla::pgetf2(n, 1, d_A.ptr, 0, descA, ipiv.data(), info);
        require_close(handle, "pgetf2(tiny nonzero pivot)", std::abs(info), 0.0);
    }

    {
        auto h_A = make_local<Complex>(descA, [](int i, int j){
            if(j == 0) return Complex(0.0, 0.0);
            return i == j ? Complex(1.0, 0.0) : Complex(0.0, 0.0);
        });
        DeviceBuffer<Complex> d_A(handle, h_A.size());
        upload(handle, d_A.ptr, h_A);
        DEVICE_CHECK(deviceStreamSynchronize(handle->stream));

        std::vector<int> ipiv(descA.m_loc(), -1);
        int info = -1;
        ddla::pgetf2(n, 1, d_A.ptr, 0, descA, ipiv.data(), info);
        require_close(handle, "pgetf2(exact singular info)", std::abs(info - 1), 0.0);
    }
}

int main(int argc, char** argv)
{
    return run_grid_test(argc, argv, "test_api_grid_pgetf2", check_pgetf2);
}
