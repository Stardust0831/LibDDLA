#include <ddla.h>
#include <cassert>
#include <vector>
#include <ddla_stream.h>
namespace DDLA{

template <typename T>
void ppotrs(
    const char& side, const char& uplo, const char& trans,
    const int& n, const int& nrhs,
    T* d_A, const DDLA::DdlaDesc& array_descA,
    T* d_B, const DDLA::DdlaDesc& array_descB
)
{
    DdlaHandle_t ddla_handle = array_descA.ddla_handle();
    
    assert(trans == 'N');
    assert(side == 'L');
    double start_time = MPI_Wtime();
    ptrtrs(
        side, uplo, uplo == 'L' ? 'N' : 'C', 'N', n, nrhs,
        d_A, array_descA,
        d_B, array_descB
    );
    // { 
    //     std::vector<std::complex<double>> a(array_descB.m_loc()*array_descB.n_loc());
    //     DEVICE_CHECK(deviceMemcpyAsync(a.data(), d_B, sizeof(std::complex<double>)*array_descB.m_loc()*array_descB.n_loc(), deviceMemcpyDeviceToHost, ddla_handle->stream));
    //     DEVICE_CHECK(deviceStreamSynchronize(ddla_handle->stream));
    //     std::string filename = "after_ptrtrs_1_myid_";
    //     filename += std::to_string(ddla_handle->myid);
    //     filename += ".txt";
    //     DDLA::write_matrix(a.data(), array_descB.m_loc(), array_descB.n_loc(), filename.c_str());
    // }
    ptrtrs(
        side, uplo, uplo == 'L' ? 'C' : 'N', 'N', n, nrhs,
        d_A, array_descA,
        d_B, array_descB
    );
    printf("myid:%d 2xtrtrs time:%lf\n",ddla_handle->myid,MPI_Wtime()-start_time);

}

template void ppotrs<std::complex<double>>(
    const char& side, const char& uplo, const char& trans,
    const int& n, const int& nrhs,
    std::complex<double>* d_A, const DDLA::DdlaDesc& array_descA,
    std::complex<double>* d_B, const DDLA::DdlaDesc& array_descB
);

template void ppotrs<std::complex<float>>(
    const char& side, const char& uplo, const char& trans,
    const int& n, const int& nrhs,
    std::complex<float>* d_A, const DDLA::DdlaDesc& array_descA,
    std::complex<float>* d_B, const DDLA::DdlaDesc& array_descB
);


}