#ifndef DDLA_H
#define DDLA_H

#include "ddla_desc.h"
#include <complex>

namespace DDLA{

void pztrtrs(
    const char& uplo, const char& diag, 
    const int& m, const int& n,
    std::complex<double>* d_A, const DDLA::DdlaDesc& array_descA,
    std::complex<double>* d_B, const DDLA::DdlaDesc& array_descB
);

template<typename T>
void ptrtrs(
    const char& side, const char& uplo, const char& trans, const char& diag,
    const int& m, const int& n,
    T* d_A, const DDLA::DdlaDesc& array_descA,
    T* d_B, const DDLA::DdlaDesc& array_descB
);

// now implements only support direc == 'F' and rowcol == 'R' and pivroc == 'C'
void pzlapiv(
    const char& direc, const char& rowcol, const char& pivroc,
    const int& m, const int& n,
    std::complex<double>* d_A,const DDLA::DdlaDesc& array_descA,
    const int* ipiv, const DDLA::DdlaDesc& array_descIP,
    int* iwork
);

void pzgetf2(
    const int& m, const int& nb_real,
    std::complex<double>* d_A, const int& n_s, const DDLA::DdlaDesc& array_descA,
    int* ipiv, // host
    int& info  // host
);

void pzgetf2_panel(
    const int& m, const int& nb_real,
    std::complex<double>* d_A, const int& n_start, const DDLA::DdlaDesc& array_descA,
    int* ipiv, // host
    int& info  // host
);

void pzgetrf(
    const int& m, const int& n,
    std::complex<double>* d_A, const DDLA::DdlaDesc& array_descA,
    int* ipiv, // host
    int& info  // host
);

// now implements only support no-transpose case
void pzgetrs(
    const char& trans, const int& n, const int& nrhs,
    std::complex<double>* d_A, const DDLA::DdlaDesc& array_descA,
    const int* ipiv, // host
    std::complex<double>* d_B, const DDLA::DdlaDesc& array_descB
);

void pzgesv(
    const int& n, const int& nrhs,
    std::complex<double>* d_A, const DDLA::DdlaDesc& array_descA,
    std::complex<double>* d_B, const DDLA::DdlaDesc& array_descB
);

// now implements only support no-transpose case
void pzgemm(
    const char& transa, const char& transb,
    const int& m, const int& n, const int& k,
    const std::complex<double>& alpha,
    const std::complex<double>* d_A, const DDLA::DdlaDesc& array_descA,
    const std::complex<double>* d_B, const DDLA::DdlaDesc& array_descB,
    const std::complex<double>& beta,
    std::complex<double>* d_C, const DDLA::DdlaDesc& array_descC
);

template <typename T>
void pgemm(
    const char& transa, const char& transb,
    const int& m, const int& n, const int& k,
    const T& alpha,
    const T* d_A, const DDLA::DdlaDesc& array_descA,
    const T* d_B, const DDLA::DdlaDesc& array_descB,
    const T& beta,
    T* d_C, const DDLA::DdlaDesc& array_descC
);

template <typename T>
void pgeadd(
    const char& transa, const char& transb,
    const int& m, const int& n,
    const T& alpha,
    const T* d_A, const DDLA::DdlaDesc& array_descA,
    const T& beta,
    const T* d_B, const DDLA::DdlaDesc& array_descB,
    T* d_C, const DDLA::DdlaDesc& array_descC
);

template<typename T>
void ppotrf(
    const char& uplo, const int& n,
    T* A, const int& ia, const int& ja, const DdlaDesc& array_descA,
    int& info // host pointer
);

template <typename T>
void ppotrs(
    const char& side, const char& uplo, const char& trans,
    const int& n, const int& nrhs,
    T* d_A, const DDLA::DdlaDesc& array_descA,
    T* d_B, const DDLA::DdlaDesc& array_descB
);

template <typename T>
void pposv(
    const char& side, const char& uplo, const char& trans,
    const int & n, const int& nrhs,
    T* d_A, const int& ia, const int& ja, const DDLA::DdlaDesc& array_descA,
    T* d_B, const int& ib, const int& jb, const DDLA::DdlaDesc& array_descB,
    int& info // device pointer
);


} // namespace DDLA

#endif // DDLA_H