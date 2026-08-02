#include "api_grid_test_common.h"

using namespace api_grid_test;

void check_pposv(const ddla::DdlaHandle_t& handle, const Shape& base)
{
    if(skip_non_square_grid(handle, "pposv")) return;

    const int nb = base.nb;
    const int n = square_size(handle, base);
    const int nrhs = nrhs_size(base);
    ddla::DdlaDesc descA(handle), descB(handle);
    descA.init(n, n, nb, nb, 0, 0);
    descB.init(n, nrhs, nb, nb, 0, 0);

    auto h_A = make_local<Complex>(descA, [=](int i, int j){ return hpd_value(i, j, n); });
    auto h_B = build_rhs(descB, n, hpd_value, n);

    DeviceBuffer<Complex> d_A(handle, h_A.size());
    DeviceBuffer<Complex> d_B(handle, h_B.size());
    upload(handle, d_A.ptr, h_A);
    upload(handle, d_B.ptr, h_B);
    check_ddla_sync(handle);

    int info = -1;
    ddla::pposv('L', 'L', 'N', n, nrhs, d_A.ptr, 1, 1, descA,
                d_B.ptr, 1, 1, descB, info);
    if(info != 0) MPI_Abort(ddla_get_communicator(handle), 1);
    check_solution(handle, descB, d_B.ptr, h_B.size(), "pposv", 5e-9);
}

namespace {

// Same construction as test_api_grid_ppotrf_head.cpp: diagonally-dominant
// Hermitian matrix, except the diagonal at `head_idx` (0-based) is negated,
// so the Schur complement at that pivot goes negative and the head
// correction (is_nega == true) fires.
inline Complex head_value(int i, int j, int n, int head_idx)
{
    if(i == j){
        const double diag = 5.0 + 0.2 * n + 0.05 * i;
        return Complex(i == head_idx ? -diag : diag, 0.0);
    }
    const int lo = std::min(i, j);
    const int hi = std::max(i, j);
    const Complex val(0.01 * ((lo + 2 * hi) % 5 - 2),
                      0.006 * ((3 * lo + hi) % 7 - 3));
    return i < j ? val : std::conj(val);
}

inline std::vector<Complex> build_head_rhs(const ddla::DdlaDesc& descB, int n, int head_idx)
{
    return make_local<Complex>(descB, [&](int i, int j){
        Complex sum(0.0, 0.0);
        for(int l = 0; l < n; ++l){
            sum += head_value(i, l, n, head_idx) * x_value(l, j);
        }
        return sum;
    });
}

} // namespace

// pposv(is_head=true, location=<interior>) regression: this is exactly the
// case that was silently broken until pposv learned to permute B around the
// call to ppotrs (see src/pposv.cpp). Unlike test_api_grid_ppotrf_head.cpp,
// this test does *not* manually permute B -- pposv is meant to hide that
// bookkeeping from its caller, so its absence here is the point.
void check_pposv_head(const ddla::DdlaHandle_t& handle, const Shape& base)
{
    if(skip_non_square_grid(handle, "pposv_head")) return;

    const int nb = base.nb;
    const int n = square_size(handle, base);
    const int nrhs = nrhs_size(base);
    const int head_idx0 = std::max(1, n / 3) - 1; // 0-based, interior

    ddla::DdlaDesc descA(handle), descB(handle);
    descA.init(n, n, nb, nb, 0, 0);
    descB.init(n, nrhs, nb, nb, 0, 0);

    auto h_A = make_local<Complex>(descA, [=](int i, int j){
        return head_value(i, j, n, head_idx0);
    });
    auto h_B = build_head_rhs(descB, n, head_idx0);

    DeviceBuffer<Complex> d_A(handle, h_A.size());
    DeviceBuffer<Complex> d_B(handle, h_B.size());
    upload(handle, d_A.ptr, h_A);
    upload(handle, d_B.ptr, h_B);
    check_ddla_sync(handle);

    const int head_idx_1based = head_idx0 + 1;
    int info = -1;
    ddla::pposv('L', 'L', 'N', n, nrhs, d_A.ptr, 1, 1, descA,
                d_B.ptr, 1, 1, descB, info, true, head_idx_1based);
    if(info != 0) MPI_Abort(ddla_get_communicator(handle), 1);
    check_solution(handle, descB, d_B.ptr, h_B.size(), "pposv_head(location=interior)", 5e-9);
}

int main(int argc, char** argv)
{
    return run_grid_test(argc, argv, "test_api_grid_pposv", [](const ddla::DdlaHandle_t& handle, const Shape& base){
        check_pposv(handle, base);
        check_pposv_head(handle, base);
    });
}
