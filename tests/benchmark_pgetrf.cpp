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
#include <ddla/scal.h>

using namespace ddla;

using Complex = std::complex<double>;

void fill_matrix(int n, const DdlaDesc& desc, Complex* d_A, const DdlaHandle_t& handle)
{
    const size_t nelem = static_cast<size_t>(desc.m_loc()) * desc.n_loc();
    random_generator(d_A, nelem, DEVICE_C_64F);
    BLAS_CHECK(deblasScal(handle->blasH, nelem, Complex(1.0e-4, 0.0), d_A, 1));

    const Complex diag(2.0, 0.0);
    for(int i = 0; i < n; ++i){
        const int iloc = desc.indx_g2l_r(i);
        const int jloc = desc.indx_g2l_c(i);
        if(iloc >= 0 && jloc >= 0){
            DEVICE_CHECK(deviceMemcpyAsync(d_A + iloc + jloc * desc.lld(), &diag,
                                          sizeof(Complex), deviceMemcpyHostToDevice,
                                          handle->stream));
        }
    }
    DEVICE_CHECK(deviceStreamSynchronize(handle->stream));
}

double benchmark_pgetrf(int n, const DdlaHandle_t& handle)
{
    const int nb = std::min(128, n);
    DdlaDesc desc(handle);
    desc.init(n, n, nb, nb, 0, 0);

    const size_t nelem = static_cast<size_t>(desc.m_loc()) * desc.n_loc();
    Complex* d_A = nullptr;
    DEVICE_CHECK(deviceMallocAsync(reinterpret_cast<void**>(&d_A),
                                  nelem * sizeof(Complex), handle->stream));
    fill_matrix(n, desc, d_A, handle);

    std::vector<int> ipiv(desc.m_loc());
    int info = -1;

    MPI_Barrier(handle->comm);
    const double start = MPI_Wtime();
    pgetrf(n, n, d_A, desc, ipiv.data(), info);
    DEVICE_CHECK(deviceStreamSynchronize(handle->stream));
    MPI_Barrier(handle->comm);
    const double elapsed = MPI_Wtime() - start;

    int global_info = 0;
    MPI_Allreduce(&info, &global_info, 1, MPI_INT, MPI_MAX, handle->comm);
    if(global_info != 0){
        if(handle->myid == 0){
            std::cerr << "pgetrf failed for n=" << n << ", info=" << global_info << std::endl;
        }
        MPI_Abort(handle->comm, 1);
    }

    double max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, handle->comm);

    DEVICE_CHECK(deviceFreeAsync(d_A, handle->stream));
    DEVICE_CHECK(deviceStreamSynchronize(handle->stream));

    if(handle->myid == 0){
        std::cout << "RESULT n=" << n
                  << " type=complex<double>"
                  << " grid=2x2"
                  << " ranks=4"
                  << " nb=" << nb
                  << " time_s=" << std::fixed << std::setprecision(6)
                  << max_elapsed
                  << std::endl;
    }
    return max_elapsed;
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int nprocs = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(nprocs != 4){
        if(rank == 0){
            std::cerr << "benchmark_pgetrf requires exactly 4 MPI ranks for a 2x2 grid"
                      << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    DdlaHandle_t handle = nullptr;
    ddla_init(handle);
    ddla_set(handle, MPI_COMM_WORLD, 2, 2);

    std::vector<int> sizes = {500, 5000, 10000, 15000};
    if(argc > 1){
        sizes.clear();
        for(int i = 1; i < argc; ++i){
            sizes.push_back(std::atoi(argv[i]));
        }
    }

    if(handle->myid == 0){
        std::cout << "=== pgetrf benchmark: complex<double>, 4 MPI ranks, 2x2 grid ==="
                  << std::endl;
    }

    for(int n : sizes){
        benchmark_pgetrf(n, handle);
    }

    ddla_destroy(handle);
    MPI_Finalize();
    return 0;
}
