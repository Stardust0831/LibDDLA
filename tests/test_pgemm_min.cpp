#include <cassert>
#include <cmath>
#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <complex>
#include <string>
#include <ddla/ddla.h>
#include <ddla/ddla_connector.h>
#include <ddla/ddla_stream.h>

using namespace ddla;

std::complex<double> op_value(char trans, const std::vector<std::complex<double>>& mat,
                              int m, int n, int i, int j)
{
    if(trans == 'N'){
        return mat[i + j * m];
    }else if(trans == 'T'){
        return mat[j + i * m];
    }else{
        return std::conj(mat[j + i * m]);
    }
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    DdlaHandle_t handle;
    ddla_init(handle);
    ddla_set(handle, MPI_COMM_WORLD, 2, 3);

    int m=100, n=100, k=100;
    int nb=10;

    DdlaDesc descA(handle), descB(handle), descC(handle);
    descA.init(m, k, nb, nb, 0, 0);
    descB.init(k, n, nb, nb, 0, 0);
    descC.init(m, n, nb, nb, 0, 0);

    int myid;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    std::vector<std::complex<double>> h_A(descA.m_loc() * descA.n_loc());
    std::vector<std::complex<double>> h_B(descB.m_loc() * descB.n_loc());
    std::vector<std::complex<double>> h_C(descC.m_loc() * descC.n_loc());

    for(size_t i=0;i<h_A.size();i++) h_A[i] = std::complex<double>((i%17)-8, (i%13)-6);
    for(size_t i=0;i<h_B.size();i++) h_B[i] = std::complex<double>((i%11)-5, (i%19)-9);
    for(size_t i=0;i<h_C.size();i++) h_C[i] = std::complex<double>(0,0);

    std::complex<double>* d_A;
    std::complex<double>* d_B;
    std::complex<double>* d_C;
    DEVICE_CHECK(deviceMalloc(&d_A, sizeof(std::complex<double>) * h_A.size()));
    DEVICE_CHECK(deviceMalloc(&d_B, sizeof(std::complex<double>) * h_B.size()));
    DEVICE_CHECK(deviceMalloc(&d_C, sizeof(std::complex<double>) * h_C.size()));

    DEVICE_CHECK(deviceMemcpy(d_A, h_A.data(), sizeof(std::complex<double>) * h_A.size(), deviceMemcpyHostToDevice));
    DEVICE_CHECK(deviceMemcpy(d_B, h_B.data(), sizeof(std::complex<double>) * h_B.size(), deviceMemcpyHostToDevice));
    DEVICE_CHECK(deviceMemcpy(d_C, h_C.data(), sizeof(std::complex<double>) * h_C.size(), deviceMemcpyHostToDevice));

    std::complex<double> alpha(1,0), beta(0,0);

    auto gather = [&](const DdlaDesc& desc, const std::vector<std::complex<double>>& local){
        int mg = desc.m();
        int ng = desc.n();
        std::vector<std::complex<double>> global(mg * ng);
        std::vector<int> recvcounts(nprocs), displs(nprocs);
        int sz = local.size();
        MPI_Allgather(&sz, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        displs[0] = 0;
        for(int i=1;i<nprocs;i++) displs[i] = displs[i-1] + recvcounts[i-1];
        std::vector<std::complex<double>> all(displs[nprocs-1] + recvcounts[nprocs-1]);
        MPI_Allgatherv(local.data(), sz, MPI_C_DOUBLE_COMPLEX,
                       all.data(), recvcounts.data(), displs.data(), MPI_C_DOUBLE_COMPLEX,
                       MPI_COMM_WORLD);
        int npcols = desc.npcols();
        for(int src=0; src<nprocs; src++){
            int prow = src / npcols;
            int pcol = src % npcols;
            int off = displs[src];
            int ml = num_loc(mg, desc.mb(), prow, desc.irsrc(), desc.nprows());
            int nl = num_loc(ng, desc.nb(), pcol, desc.icsrc(), desc.npcols());
            if(ml * nl != recvcounts[src]) continue;
            for(int j=0;j<nl;j++){
                int gj = indxl2g(j, desc.nb(), pcol, desc.icsrc(), desc.npcols());
                for(int i=0;i<ml;i++){
                    int gi = indxl2g(i, desc.mb(), prow, desc.irsrc(), desc.nprows());
                    global[gi + gj * mg] = all[off + i + j * ml];
                }
            }
        }
        return global;
    };

    std::vector<std::pair<char,char>> cases = {{'N','T'}, {'T','N'}, {'T','T'}, {'C','N'}, {'N','C'}, {'C','C'}};
    for(auto& tc : cases){
        char transa = tc.first;
        char transb = tc.second;

        std::vector<std::complex<double>> h_zero_C(h_C.size());
        DEVICE_CHECK(deviceMemcpy(d_C, h_zero_C.data(), sizeof(std::complex<double>) * h_C.size(), deviceMemcpyHostToDevice));
        pgemm(transa, transb, m, n, k, alpha, d_A, descA, d_B, descB, beta, d_C, descC);
        DEVICE_CHECK(deviceStreamSynchronize(handle->stream));

        std::vector<std::complex<double>> h_C_out(descC.m_loc() * descC.n_loc());
        DEVICE_CHECK(deviceMemcpy(h_C_out.data(), d_C, sizeof(std::complex<double>) * h_C_out.size(), deviceMemcpyDeviceToHost));

        auto g_A = gather(descA, h_A);
        auto g_B = gather(descB, h_B);
        auto g_C = gather(descC, h_C_out);

        int ma = (transa == 'N') ? m : k;
        int na = (transa == 'N') ? k : m;
        int mb_ = (transb == 'N') ? k : n;
        int nb_ = (transb == 'N') ? n : k;

        double max_err = 0;
        for(int j=0;j<n;j++){
            for(int i=0;i<m;i++){
                std::complex<double> ref(0,0);
                for(int l=0;l<k;l++){
                    ref += alpha * op_value(transa, g_A, ma, na, i, l) * op_value(transb, g_B, mb_, nb_, l, j);
                }
                max_err = std::max(max_err, std::abs(g_C[i + j*m] - ref));
            }
        }
        double global_max;
        MPI_Reduce(&max_err, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if(myid == 0){
            std::cout << "pgemm(" << transa << "," << transb << ") max_err " << global_max << std::endl;
        }
    }

    DEVICE_CHECK(deviceFree(d_A));
    DEVICE_CHECK(deviceFree(d_B));
    DEVICE_CHECK(deviceFree(d_C));
    ddla_destroy(handle);
    MPI_Finalize();
    return 0;
}
