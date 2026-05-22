#ifndef DDLA_H
#define DDLA_H

#include "ddla_desc.h"
#include <complex>

namespace ddla{

template<typename T>
void ptrtrs(
    const char& side, const char& uplo, const char& trans, const char& diag,
    const int& m, const int& n,
    T* d_A, const DdlaDesc& array_descA,
    T* d_B, const DdlaDesc& array_descB
);

// now implements only support direc == 'F' and rowcol == 'R' and pivroc == 'C'
template <typename T>
void plapiv(
    const char& direc, const char& rowcol, const char& pivroc,
    const int& m, const int& n,
    T* d_A,const DdlaDesc& array_descA,
    const int* ipiv, const DdlaDesc& array_descIP,
    int* iwork
);

template <typename T>
void pswap(
    const int& N, 
    T* A, int ia, int ja, const DdlaDesc& array_descA, const int& inca,
    T* B, int ib, int jb, const DdlaDesc& array_descB, const int& incb
);

template <typename T>
void pgetf2(
    const int& m, const int& nb_real,
    T* d_A, const int& n_s, const DdlaDesc& array_descA,
    int* ipiv, // host
    int& info  // host
);

template <typename T>
void pgetf2_panel(
    const int& m, const int& nb_real,
    T* d_A, const int& n_start, const DdlaDesc& array_descA,
    int* ipiv, // host
    int& info  // host
);

template <typename T>
void pgetrf(
    const int& m, const int& n,
    T* d_A, const DdlaDesc& array_descA,
    int* ipiv, // host
    int& info  // host
);

// now implements only support no-transpose case
template <typename T>
void pgetrs(
    const char& trans, const int& n, const int& nrhs,
    T* d_A, const DdlaDesc& array_descA,
    const int* ipiv, // host
    T* d_B, const DdlaDesc& array_descB
);

template <typename T>
void pgesv(
    const int& n, const int& nrhs,
    T* d_A, const DdlaDesc& array_descA,
    T* d_B, const DdlaDesc& array_descB
);

template <typename T>
void pgemm(
    const char& transa, const char& transb,
    const int& m, const int& n, const int& k,
    const T& alpha,
    const T* d_A, const DdlaDesc& array_descA,
    const T* d_B, const DdlaDesc& array_descB,
    const T& beta,
    T* d_C, const DdlaDesc& array_descC
);

template <typename T>
void pgeadd(
    const char& transa, const char& transb,
    const int& m, const int& n,
    const T& alpha,
    const T* d_A, const DdlaDesc& array_descA,
    const T& beta,
    const T* d_B, const DdlaDesc& array_descB,
    T* d_C, const DdlaDesc& array_descC
);

template<typename T>
bool ppotrf(
    const char& uplo, const int& n,
    T* A, const int& ia, const int& ja, const DdlaDesc& array_descA,
    int& info, // host pointer
    bool is_head = false, int location = -1
);

template <typename T>
void ppotrs(
    const char& side, const char& uplo, const char& trans,
    const int& n, const int& nrhs,
    T* d_A, const DdlaDesc& array_descA,
    T* d_B, const DdlaDesc& array_descB,
    bool is_nega = false, int location = -1
);

template <typename T>
void pposv(
    const char& side, const char& uplo, const char& trans,
    const int & n, const int& nrhs,
    T* d_A, const int& ia, const int& ja, const DdlaDesc& array_descA,
    T* d_B, const int& ib, const int& jb, const DdlaDesc& array_descB,
    int& info, // host pointer
    bool is_head = false, int location = -1
);


} // namespace DDLA

#endif // DDLA_H