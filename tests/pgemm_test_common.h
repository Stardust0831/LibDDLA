#ifndef DDLA_PGEMM_TEST_COMMON_H
#define DDLA_PGEMM_TEST_COMMON_H
#include "matrix_test_common.h"
#include "pgemm_impl.h"

using pgemm_impl::Algorithm;

template <typename T> std::complex<double> op_value(char t, int i, int j, int salt) {
    auto x = wide(t == 'N' ? value<T>(i,j,salt) : value<T>(j,i,salt));
    return t == 'C' ? std::conj(x) : x;
}

#endif
