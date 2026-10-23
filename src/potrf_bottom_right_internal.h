#ifndef DDLA_POTRF_BOTTOM_RIGHT_INTERNAL_H
#define DDLA_POTRF_BOTTOM_RIGHT_INTERNAL_H

#include <ddla/ddla_handle_t.h>

namespace ddla::detail {

// Factor one diagonal block in backend memory from the bottom-right. With
// uplo='U', compute A = U * U^H; with uplo='L', compute A = L^H * L. The
// selected triangle is reverse-packed into work, factored with the opposite
// standard POTRF triangle, and mapped back. work holds at least n*n elements.
// info_buffer and work are caller-owned and may be reused across diagonal blocks.
template <DdlaBackend Backend, typename T>
void potrf_bottom_right_block(
    const char& uplo, const int& n, T* A, const int& lda,
    const int& global_offset,
    T* work, int* info_buffer, int& info,
    const DdlaHandle_t& handle
);

} // namespace ddla::detail

#endif // DDLA_POTRF_BOTTOM_RIGHT_INTERNAL_H
