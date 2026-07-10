#include <algorithm>
#include <complex>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#include <mpi.h>

#include <ddla/ddla.h>
#include <ddla/ddla_connector.h>
#include <ddla/ddla_stream.h>

using namespace ddla;

namespace {

using Complex = std::complex<double>;

constexpr int kBlockSize = 128;
constexpr int kPanelWidth = 32;
constexpr int kRepeats = 7;

void initialize_panel_matrix(int n, const ddla::DdlaDesc& desc, Complex* d_A,
                             const ddla::DdlaHandle_t& handle)
{
    const size_t count = static_cast<size_t>(desc.lld()) * desc.n_loc();
    DEVICE_CHECK(deviceMemsetAsync(d_A, 0, count * sizeof(Complex), handle->stream));

    const Complex diagonal(2.0, 0.0);
    for(int i = 0; i < std::min(n, kPanelWidth); ++i){
        const int iloc = desc.indx_g2l_r(i);
        const int jloc = desc.indx_g2l_c(i);
        if(iloc >= 0 && jloc >= 0){
            DEVICE_CHECK(deviceMemcpyAsync(
                d_A + iloc + static_cast<size_t>(jloc) * desc.lld(),
                &diagonal, sizeof(Complex), deviceMemcpyHostToDevice, handle->stream));
        }
    }
    DEVICE_CHECK(deviceStreamSynchronize(handle->stream));
}

double benchmark_size(int n, int repeats, const ddla::DdlaHandle_t& handle)
{
    ddla::DdlaDesc desc(handle);
    desc.init(n, n, kBlockSize, kBlockSize, 0, 0);

    const size_t count = static_cast<size_t>(desc.lld()) * desc.n_loc();
    Complex* d_A = nullptr;
    DEVICE_CHECK(deviceMallocAsync(reinterpret_cast<void**>(&d_A),
                                   std::max<size_t>(count, 1) * sizeof(Complex),
                                   handle->stream));
    initialize_panel_matrix(n, desc, d_A, handle);

    std::vector<int> ipiv(desc.m_loc(), -1);
    std::vector<double> times;
    if(handle->myid == 0){
        times.reserve(repeats);
    }

    for(int repeat = 0; repeat < repeats; ++repeat){
        std::fill(ipiv.begin(), ipiv.end(), -1);
        int info = -1;

        MPI_CHECK(MPI_Barrier(handle->comm));
        const double start = MPI_Wtime();
        ddla::pgetf2(n, kPanelWidth, d_A, 0, desc, ipiv.data(), info);
        DEVICE_CHECK(deviceStreamSynchronize(handle->stream));
        const double elapsed = MPI_Wtime() - start;

        int global_info = 0;
        MPI_CHECK(MPI_Allreduce(&info, &global_info, 1, MPI_INT, MPI_MAX, handle->comm));
        if(global_info != 0){
            if(handle->myid == 0){
                std::cerr << "pgetf2 failed for n=" << n
                          << ", info=" << global_info << std::endl;
            }
            MPI_Abort(handle->comm, 1);
        }

        double max_elapsed = 0.0;
        MPI_CHECK(MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX,
                             0, handle->comm));
        if(handle->myid == 0){
            times.push_back(max_elapsed);
        }
    }

    DEVICE_CHECK(deviceFreeAsync(d_A, handle->stream));
    DEVICE_CHECK(deviceStreamSynchronize(handle->stream));

    if(handle->myid != 0){
        return 0.0;
    }
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

} // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int nprocs = 0;
    int rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(nprocs != 4){
        if(rank == 0){
            std::cerr << "benchmark_pgetf2 requires exactly 4 MPI ranks for a 2x2 grid"
                      << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::vector<int> sizes = {500, 5000, 10000, 15000};
    if(argc > 1){
        sizes.clear();
        for(int i = 1; i < argc; ++i){
            const int n = std::atoi(argv[i]);
            if(n <= 0){
                if(rank == 0){
                    std::cerr << "matrix sizes must be positive integers" << std::endl;
                }
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            sizes.push_back(n);
        }
    }

    ddla::DdlaHandle_t handle = nullptr;
    ddla::ddla_init(handle);
    ddla::ddla_set(handle, MPI_COMM_WORLD, 2, 2);

    if(handle->myid == 0){
        std::cout << "=== pgetf2 benchmark: complex<double>, 4 MPI ranks, 2x2 grid ==="
                  << std::endl;
    }

    for(size_t i = 0; i < sizes.size(); ++i){
        const bool warmup = i == 0 && sizes[i] == 500;
        const int repeats = warmup ? 1 : kRepeats;
        const double median = benchmark_size(sizes[i], repeats, handle);
        if(handle->myid == 0){
            std::cout << (warmup ? "WARMUP" : "RESULT")
                      << " n=" << sizes[i]
                      << " type=complex<double>"
                      << " grid=2x2"
                      << " ranks=4"
                      << " nb=" << kBlockSize
                      << " panel=" << kPanelWidth
                      << " repeats=" << repeats
                      << " median_time_s=" << std::fixed << std::setprecision(6)
                      << median << std::endl;
        }
    }

    ddla::ddla_destroy(handle);
    MPI_Finalize();
    return 0;
}
