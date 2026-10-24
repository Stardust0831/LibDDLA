#ifndef DDLA_PGEMM_IMPL_H
#define DDLA_PGEMM_IMPL_H

#include <ddla/ddla.h>

namespace ddla { namespace pgemm_impl {

// Private entry points shared with the regression tests and benchmark. Public
// pgemm always selects Auto; there is no process-wide algorithm setting.
enum class Algorithm { Auto, AB, AC, BC };

Algorithm select(char ta, char tb, int m, int n, int k,
                 const int* da, const int* db, const int* dc, int pr, int pc);

template <DdlaBackend Backend, typename T>
void run(const DdlaHandle_t& h, char ta, char tb,
         int m, int n, int k, const T& alpha,
         const T* a, const int* da, const T* b, const int* db,
         const T& beta, T* c, const int* dc, Algorithm algorithm);

template <DdlaBackend Backend, typename T>
void stationary(const DdlaHandle_t& h, bool keep_a, char ta, char tb,
                int m, int n, int k, const T& alpha,
                const T* a, const int* da, const T* b, const int* db,
                const T& beta, T* c, const int* dc);

}} // namespace ddla::pgemm_impl
#endif
