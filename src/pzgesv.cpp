#include <ddla.h>
#include <cassert>
#include <vector>
#include <ddla_connector.h>
#include <ddla_utils.h>
#include <ddla_stream.h>
namespace DDLA{

void pzgesv(
    const int& n, const int& nrhs,
    std::complex<double>* d_A, const DDLA::DdlaDesc& array_descA,
    std::complex<double>* d_B, const DDLA::DdlaDesc& array_descB
)
{
    std::vector<int> ipiv(array_descA.m_loc());
    int info = 1;
    pzgetrf(
        n, n,
        d_A, array_descA,
        ipiv.data(),
        info
    );
    if(info !=0){
        printf("Error in pzgetrf, info = %d\n", info);
        throw std::runtime_error("info !=0\n");
    }
    pzgetrs(
        'N', n, nrhs,
        d_A, array_descA,
        ipiv.data(),
        d_B, array_descB
    );
}






}