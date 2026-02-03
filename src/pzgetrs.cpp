#include <ddla.h>
#include <cassert>
#include <vector>
#include <ddla_stream.h>
namespace DDLA{

void pzgetrs(
    const char& trans, const int& n, const int& nrhs,
    std::complex<double>* d_A, const DDLA::DdlaDesc& array_descA,
    const int* ipiv, // host
    std::complex<double>* d_B, const DDLA::DdlaDesc& array_descB
)
{
    DdlaHandle_t ddla_handle = array_descA.ddla_handle();
    
    assert(trans == 'N');
    char direc = 'F';
    char rowcol = 'R';
    char pivroc='C';
    printf("myid:%d, start pzlapiv\n",ddla_handle->myid);
    double start_time_swap = MPI_Wtime();
    pzlapiv(
        direc, rowcol, pivroc,
        n, nrhs,
        d_B, array_descB,
        ipiv, array_descA,
        nullptr
    );
    printf("myid:%d, pzlapiv time:%lf\n",ddla_handle->myid,MPI_Wtime()-start_time_swap);
    double start_time = MPI_Wtime();
    pztrtrs(
        'L', 'U', n, nrhs,
        d_A, array_descA,
        d_B, array_descB
    );
    pztrtrs(
        'U', 'N', n, nrhs,
        d_A, array_descA,
        d_B, array_descB
    );
    printf("myid:%d 2xtrtrs time:%lf\n",ddla_handle->myid,MPI_Wtime()-start_time);

}

}