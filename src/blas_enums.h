#ifndef DDLA_BLAS_ENUMS_H
#define DDLA_BLAS_ENUMS_H

// ---------------------------------------------------------------------------
// LAPACK-style character flags <-> vendor BLAS enum conversion.
//
// The unified backend-templated BLAS entries (src/gemm.cpp and the siblings
// modelled on it) take the LAPACK convention -- plain `char` for
// uplo/side/trans/diag -- because that is what the public header uses for
// every routine that exposes these flags (ddla.h's gemm, ptrtrs, ...).  The
// vendor arms need `gpublasFillMode_t` and friends, so the mapping has to
// happen somewhere; doing it here keeps it in ONE place instead of repeating
// the same three ternaries in every scalar overload of every family, which is
// how the pre-unification wrappers carried 48 copies of it.
//
// The CPU (Fortran) arms need no conversion at all: they consume the character
// directly, which is why only the GPU arms call these.
//
// Accepted characters are the LAPACK ones:
//   uplo  'U' upper / 'L' lower
//   side  'L' left  / 'R' right
//   diag  'U' unit  / 'N' non-unit
//   trans 'N' none  / 'T' transpose / 'C' conjugate-transpose
//
// An unrecognised character throws rather than silently defaulting: a typo'd
// 'u' reaching cuBLAS as UPPER would produce a wrong answer that no
// convergence or residual check is guaranteed to catch on every input.
//
// The functions are `constexpr`-eligible and header-only so they can be used
// from any TU, and they compile in the CPU-only lane too (there the vendored
// `gpublas*` aliases are plain ints/chars, so the comparisons still hold).
// ---------------------------------------------------------------------------

#include <stdexcept>

#include "ddla_connector.h"

namespace ddla {
namespace detail {

inline gpublasFillMode_t to_fill_mode(char uplo)
{
    if (uplo == 'U' || uplo == 'u') return GPUBLAS_FILL_MODE_UPPER;
    if (uplo == 'L' || uplo == 'l') return GPUBLAS_FILL_MODE_LOWER;
    throw std::runtime_error("ddla: invalid uplo flag (expected 'U' or 'L')");
}

inline gpublasSideMode_t to_side_mode(char side)
{
    if (side == 'L' || side == 'l') return GPUBLAS_SIDE_LEFT;
    if (side == 'R' || side == 'r') return GPUBLAS_SIDE_RIGHT;
    throw std::runtime_error("ddla: invalid side flag (expected 'L' or 'R')");
}

inline gpublasDiagType_t to_diag_type(char diag)
{
    if (diag == 'U' || diag == 'u') return GPUBLAS_DIAG_UNIT;
    if (diag == 'N' || diag == 'n') return GPUBLAS_DIAG_NON_UNIT;
    throw std::runtime_error("ddla: invalid diag flag (expected 'U' or 'N')");
}

inline gpublasOperation_t to_blas_op(char trans)
{
    if (trans == 'N' || trans == 'n') return GPUBLAS_OP_N;
    if (trans == 'T' || trans == 't') return GPUBLAS_OP_T;
    if (trans == 'C' || trans == 'c') return GPUBLAS_OP_C;
    throw std::runtime_error("ddla: invalid trans flag (expected 'N', 'T' or 'C')");
}

// The symmetric/Hermitian rank-k families (xSYRK, xHERK, xSYR2K, xHER2K)
// accept only 'N' or 'T'/'C' semantics -- there is no "no-transpose vs
// transpose" distinction beyond that, and LAPACK itself normalises any
// conjugate-transpose request to a plain transpose for the real types.  This
// mirrors what the pre-unification wrappers did with a local `trans_real`.
inline char trans_real(char trans)
{
    return (trans == 'N' || trans == 'n') ? 'N' : 'T';
}

} // namespace detail
} // namespace ddla

#endif // DDLA_BLAS_ENUMS_H
