#include <ddla/ddla_connector.h>
#include <chrono>
#include <fstream>
#include <type_traits>

namespace ddla{

template <typename T>
void random_generate(T* data, const int64_t& lengthOfData)
{
    if (lengthOfData <= 0) return;
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    derandGenerator_t gen;
    DERAND_CHECK(derandCreateGenerator(&gen, DERAND_RNG_PSEUDO_DEFAULT));
    DERAND_CHECK(derandSetPseudoRandomGeneratorSeed(gen, static_cast<unsigned long long>(seed)));
    if constexpr (std::is_same_v<T, float>) {
        DERAND_CHECK(derandGenerateUniform(gen, data, lengthOfData));
    } else if constexpr (std::is_same_v<T, double>) {
        DERAND_CHECK(derandGenerateUniformDouble(gen, data, lengthOfData));
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        DERAND_CHECK(derandGenerateUniform(gen, reinterpret_cast<float*>(data), lengthOfData * 2));
    } else if constexpr (std::is_same_v<T, std::complex<double>>) {
        DERAND_CHECK(derandGenerateUniformDouble(gen, reinterpret_cast<double*>(data), lengthOfData * 2));
    } else {
        static_assert(sizeof(T) == 0, "random_generate supports only float, double, std::complex<float>, std::complex<double>");
    }
    DERAND_CHECK(derandDestroyGenerator(gen));
}

void write_matrix(std::complex<double>* A, const int& m,const int& n, const char* filename)
{

    std::ofstream outfile;
    outfile.open(filename, std::ios::out | std::ios::trunc);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            outfile<<"(";
            if(std::abs(A[i+j*m].real())<1e-10)
                outfile<<"0";
            else
                outfile<<A[i+j*m].real();
            outfile<<",";
            if(std::abs(A[i+j*m].imag())<1e-10)
                outfile<<"0";
            else
                outfile<<A[i+j*m].imag();
            outfile<<") ";
        }
        outfile<<"\n";
    }
    outfile.close();
    return;
}


template void random_generate<float>(float*, const int64_t&);
template void random_generate<double>(double*, const int64_t&);
template void random_generate<std::complex<float>>(std::complex<float>*, const int64_t&);
template void random_generate<std::complex<double>>(std::complex<double>*, const int64_t&);

} // namespace ddla