#include "api_grid_test_common.h"

using namespace api_grid_test;

void check_pgetf2(const ddla::DdlaHandle_t& handle, const Shape& base)
{
    const int nb = base.nb;
    const int n = square_size(handle, base);
    ddla::DdlaDesc descA(handle);
    descA.init(n, n, nb, nb, 0, 0);

    auto h_A = make_local<Complex>(descA, [=](int i, int j){ return dominant_value(i, j, n); });
    DeviceBuffer<Complex> d_A(handle, h_A.size());
    upload(handle, d_A.ptr, h_A);
    DEVICE_CHECK(deviceStreamSynchronize(handle->stream));

    std::vector<int> ipiv(descA.m_loc());
    int info = 0;
    ddla::pgetf2(n, std::min(nb, n), d_A.ptr, 0, descA, ipiv.data(), info);
    if(info != 0) MPI_Abort(handle->comm, 1);
    require_close(handle, "pgetf2(info)", 0.0, 0.0);
}

int main(int argc, char** argv)
{
    return run_grid_test(argc, argv, "test_api_grid_pgetf2", check_pgetf2);
}
