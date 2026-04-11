#include <ddla.h>
namespace DDLA{

template <typename T>
void pposv(
    const char& side, const char& uplo, const char& trans,
    const int & n, const int& nrhs,
    T* d_A, const int& ia, const int& ja, const DDLA::DdlaDesc& array_descA,
    T* d_B, const int& ib, const int& jb, const DDLA::DdlaDesc& array_descB,
    int& info // host pointer
)
{
    ppotrf(uplo, n, d_A, ia, ja, array_descA, info);
    if(info == 0)
        ppotrs(side, uplo, trans, n, nrhs, d_A, array_descA, d_B, array_descB);
    return;
}

template void pposv<std::complex<float>>(
    const char& side, const char& uplo, const char& trans,
    const int & n, const int& nrhs,
    std::complex<float>* d_A, const int& ia, const int& ja, const DDLA::DdlaDesc& array_descA,
    std::complex<float>* d_B, const int& ib, const int& jb, const DDLA::DdlaDesc& array_descB,
    int& info // host pointer
);

template void pposv<std::complex<double>>(
    const char& side, const char& uplo, const char& trans,
    const int & n, const int& nrhs,
    std::complex<double>* d_A, const int& ia, const int& ja, const DDLA::DdlaDesc& array_descA,
    std::complex<double>* d_B, const int& ib, const int& jb, const DDLA::DdlaDesc& array_descB,
    int& info // host pointer
);

} // namespace DDLA