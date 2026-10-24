#include "pgemm_impl.h"
#include <algorithm>
#include <cstdint>

namespace ddla { namespace pgemm_impl {

Algorithm select(char ta, char tb, int m, int n, int k,
                 const int* a, const int* b, const int* c, int pr, int pc)
{
    if (pr == 1 && pc == 1) return Algorithm::AB;
    // ScaLAPACK 2.2.2 PBLAS/SRC/pdgemm_.c, communication-volume model.
    // DNROC estimates the largest local extent in whole blocks. Use wide
    // arithmetic before rounding: dimensions themselves are public ints.
    auto loc = [](int extent, int block, int procs) {
        const std::int64_t blocks = (std::int64_t(extent) + block - 1) / block;
        return double((blocks + procs - 1) / procs) * block;
    };
    const double reduce_cost = 1.3;
    double ab = k, ac = m, bc = n, x, y, z, w;
    if (tb == 'N') {
        if (ta == 'N') {
            x = loc(m,c[DDLA_MB_],pr); y = loc(n,c[DDLA_NB_],pc);
            ab *= (pc == 1 ? 0 : x) + (pr == 1 ? 0 : y);
            x = loc(k,b[DDLA_MB_],pr); y = loc(n,b[DDLA_NB_],pc);
            z = loc(k,a[DDLA_NB_],pc);
            ac *= (pc == 1 ? 0 : x) + std::max(x,z) + reduce_cost*(pr == 1 ? 0 : y);
            x = loc(m,a[DDLA_MB_],pr); y = loc(k,a[DDLA_NB_],pc);
            w = loc(k,b[DDLA_MB_],pr);
            bc *= reduce_cost*(pc == 1 ? 0 : x) + (pr == 1 ? 0 : y) + std::max(y,w);
        } else {
            x = loc(m,c[DDLA_MB_],pr); y = loc(n,c[DDLA_NB_],pc);
            z = loc(m,a[DDLA_NB_],pc);
            ab *= (pc == 1 ? 0 : x) + std::max(x,z) + (pr == 1 ? 0 : y);
            x = loc(k,b[DDLA_MB_],pr); y = loc(n,b[DDLA_NB_],pc);
            ac *= (pc == 1 ? 0 : x) + reduce_cost*(pr == 1 ? 0 : y);
            x = loc(k,a[DDLA_MB_],pr); y = loc(m,b[DDLA_NB_],pc);
            w = loc(m,c[DDLA_MB_],pr);
            bc *= (pc == 1 ? 0 : x) + reduce_cost*(pr == 1 ? 0 : y) + std::max(y,w);
        }
    } else {
        if (ta == 'N') {
            x = loc(m,c[DDLA_MB_],pr); y = loc(n,c[DDLA_NB_],pc);
            w = loc(n,b[DDLA_MB_],pr);
            ab *= (pc == 1 ? 0 : x) + (pr == 1 ? 0 : y) + std::max(y,w);
            x = loc(n,b[DDLA_MB_],pr); y = loc(k,b[DDLA_NB_],pc);
            z = loc(n,c[DDLA_NB_],pc);
            ac *= reduce_cost*(pc == 1 ? 0 : x) + std::max(x,z) + (pr == 1 ? 0 : y);
            x = loc(m,a[DDLA_MB_],pr); y = loc(k,a[DDLA_NB_],pc);
            bc *= reduce_cost*(pc == 1 ? 0 : x) + (pr == 1 ? 0 : y);
        } else {
            x = loc(m,c[DDLA_MB_],pr); y = loc(n,c[DDLA_NB_],pc);
            z = loc(m,a[DDLA_NB_],pc); w = loc(n,b[DDLA_MB_],pr);
            ab *= (pc == 1 ? 0 : x) + std::max(x,z) + (pr == 1 ? 0 : y) + std::max(y,w);
            x = loc(n,b[DDLA_MB_],pr); y = loc(k,b[DDLA_NB_],pc);
            z = loc(n,c[DDLA_NB_],pc); w = loc(k,a[DDLA_MB_],pr);
            ac *= reduce_cost*(pc == 1 ? 0 : x) + std::max(x,z) + (pr == 1 ? 0 : y) + std::max(y,w);
            x = loc(k,a[DDLA_MB_],pr); y = loc(m,a[DDLA_NB_],pc);
            z = loc(k,b[DDLA_NB_],pc); w = loc(m,c[DDLA_MB_],pr);
            bc *= (pc == 1 ? 0 : x) + std::max(x,z) + reduce_cost*(pr == 1 ? 0 : y) + std::max(y,w);
        }
    }
    if (ab <= 1.3*bc && ab <= 1.3*ac) return Algorithm::AB;
    return bc <= ac && 1.3*bc <= ab ? Algorithm::BC : Algorithm::AC;
}

}} // namespace ddla::pgemm_impl
