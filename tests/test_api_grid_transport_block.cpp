#include "api_grid_test_common.h"

using namespace api_grid_test;

void check_transport_block(const ddla::DdlaHandle_t& handle, const Shape& base)
{
    const int nb = base.nb;
    const int m = round_up_for_grid(base.m, nb, handle->nprows_);
    const int n = round_up_for_grid(base.n, nb, handle->npcols_);
    ddla::DdlaDesc desc(handle);
    desc.init(m, n, nb, nb, 0, 0);

    auto h_A = make_local<Complex>(desc, [](int i, int j){ return general_value(i, j, 7); });
    DeviceBuffer<Complex> d_A(handle, h_A.size());
    upload(handle, d_A.ptr, h_A);

    const int rows = std::min(nb, m);
    DeviceBuffer<Complex> d_row(handle, static_cast<size_t>(rows) * desc.n_loc());
    ddla::transport_block('R', 'N', rows, n, d_A.ptr, 0, 0, desc, d_row.ptr);
    auto h_row = download(handle, d_row.ptr, static_cast<size_t>(rows) * desc.n_loc());
    double err_row = 0.0;
    for(int jloc = 0; jloc < desc.n_loc(); ++jloc){
        const int j = desc.indx_l2g_c(jloc);
        for(int r = 0; r < rows; ++r){
            err_row = std::max(err_row, std::abs(h_row[r + jloc * rows] - general_value(r, j, 7)));
        }
    }
    require_close(handle, "transport_block(R,N)", err_row, 1e-12);

    const int cols = std::min(nb, n);
    DeviceBuffer<Complex> d_col(handle, static_cast<size_t>(desc.m_loc()) * cols);
    ddla::transport_block('C', 'N', m, cols, d_A.ptr, 0, 0, desc, d_col.ptr);
    auto h_col = download(handle, d_col.ptr, static_cast<size_t>(desc.m_loc()) * cols);
    double err_col = 0.0;
    for(int c = 0; c < cols; ++c){
        for(int iloc = 0; iloc < desc.m_loc(); ++iloc){
            const int i = desc.indx_l2g_r(iloc);
            err_col = std::max(err_col, std::abs(h_col[iloc + c * desc.m_loc()] - general_value(i, c, 7)));
        }
    }
    require_close(handle, "transport_block(C,N)", err_col, 1e-12);
}

int main(int argc, char** argv)
{
    return run_grid_test(argc, argv, "test_api_grid_transport_block", check_transport_block);
}
