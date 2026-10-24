#ifndef DDLA_MATRIX_TEST_COMMON_H
#define DDLA_MATRIX_TEST_COMMON_H
#include <ddla/ddla.h>
#include <ddla/ddla_config.h>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

using namespace ddla;

template <typename T> T scalar(double re, double im = 0) {
    if constexpr (std::is_same_v<T,float> || std::is_same_v<T,double>) return T(re);
    else return T(re, im);
}
template <typename T> T value(int i, int j, int salt) {
    return scalar<T>(((i*17+j*11+salt*7)%31-15)/16.0,
                     ((i*7+j*19+salt*13)%29-14)/16.0);
}
template <typename T> std::complex<double> wide(T x) {
    if constexpr (std::is_same_v<T,float> || std::is_same_v<T,double>) return {double(x),0};
    else return {double(x.real()),double(x.imag())};
}

template <typename T> struct Matrix {
    DdlaHandle_t h;
    int d[DDLA_DLEN_], nr, nc, pr, pc, r, c;
    T* data = nullptr;
    std::vector<T> host;
    Matrix(DdlaHandle_t handle, int m, int n, int mb, int nb, int rs, int cs, int salt, bool poison)
        : h(handle) {
        ddlaGetGridDims(h,pr,pc); ddlaGetGridCoords(h,r,c);
        const int rows = m+5, cols = n+4;
        DDLA_CHECK(ddlaDescInit(d,h,rows,cols,mb,nb,rs,cs,num_loc(rows,mb,r,rs,pr)+3));
        nr=num_loc(rows,mb,r,rs,pr); nc=num_loc(cols,nb,c,cs,pc);
        host.assign(std::max(std::size_t(1),std::size_t(d[DDLA_LLD_])*nc),scalar<T>(-987.25,654.5));
        for(int j=0;j<nc;++j) for(int i=0;i<nr;++i) {
            int gi=indxl2g(i,mb,r,rs,pr), gj=indxl2g(j,nb,c,cs,pc);
            if(gi<m && gj<n) host[i+std::size_t(j)*d[DDLA_LLD_]] = poison ? scalar<T>(std::numeric_limits<double>::quiet_NaN()) : value<T>(gi,gj,salt);
        }
        DDLA_CHECK(ddlaMalloc(h,reinterpret_cast<void**>(&data),host.size()*sizeof(T)));
        reset();
    }
    ~Matrix() { ddlaFree(h,data); }
    void reset() { DDLA_CHECK(ddlaMemcpy(h,data,host.data(),host.size()*sizeof(T))); }
    std::vector<T> download() {
        std::vector<T> out(host.size());
        DDLA_CHECK(ddlaMemcpy(h,out.data(),data,out.size()*sizeof(T)));
        DDLA_CHECK(ddlaSynchronize(h)); return out;
    }
};

#endif
