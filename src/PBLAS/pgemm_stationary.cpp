#include "pgemm_impl.h"
#include "comm_traits.h"
#include "transport_block.h"
#include <algorithm>

namespace ddla { namespace pgemm_impl {
namespace {

// Keep panel copies on the calling CPU thread. Starting a second OpenMP
// pool here can leave spinning workers competing with the threaded BLAS.
// The BLAS thread count is untouched, and the handle budget is restored.
template <DdlaBackend Backend> class HostPanelThreads {
    DdlaHandle_t h_;
    int saved_;
public:
    explicit HostPanelThreads(DdlaHandle_t h) : h_(h), saved_(h->host_threads) {
        if constexpr (Backend == DdlaBackend::CPU) h_->host_threads = 1;
    }
    ~HostPanelThreads() {
        if constexpr (Backend == DdlaBackend::CPU) h_->host_threads = saved_;
    }
    HostPanelThreads(const HostPanelThreads&) = delete;
    HostPanelThreads& operator=(const HostPanelThreads&) = delete;
};


} // namespace

template <DdlaBackend Backend, typename T>
void stationary(const DdlaHandle_t& h, bool keep_a, char ta, char tb,
                int m, int n, int k, const T& alpha,
                const T* a, const int* da, const T* b, const int* db,
                const T& beta, T* c, const int* dc)
{
    HostPanelThreads<Backend> panel_threads(h);
    const int pr = h->nprows_, pc = h->npcols_, r = h->myprow_, col = h->mypcol_;
    const int* fixed = keep_a ? da : db;
    const int* thin = keep_a ? db : da;
    const T* x = keep_a ? b : a;
    const char tx = keep_a ? tb : ta;
    const bool k_rows = keep_a ? ta != 'N' : tb == 'N';
    const bool thin_k_rows = keep_a ? tb == 'N' : ta != 'N';
    const int kb = fixed[k_rows ? DDLA_MB_ : DDLA_NB_];
    const int ks = fixed[k_rows ? DDLA_RSRC_ : DDLA_CSRC_];
    const int fb = fixed[k_rows ? DDLA_NB_ : DDLA_MB_];
    const int fs = fixed[k_rows ? DDLA_CSRC_ : DDLA_RSRC_];
    const int kloc = num_loc(k, kb, k_rows ? r : col, ks, k_rows ? pr : pc);
    const int free_extent = keep_a ? m : n;
    const int floc = num_loc(free_extent, fb, k_rows ? col : r, fs, k_rows ? pc : pr);
    const int panels = keep_a ? n : m;
    const int panel_block = dc[keep_a ? DDLA_NB_ : DDLA_MB_];
    const bool transpose_output = keep_a == k_rows;
    const CommScope reduction = k_rows ? CommScope::Col : CommScope::Row;
    if ((thin_k_rows==k_rows && thin[k_rows?DDLA_RSRC_:DDLA_CSRC_]!=ks) ||
        (!transpose_output && fs!=dc[keep_a?DDLA_RSRC_:DDLA_CSRC_]))
        throw std::invalid_argument("pgemm: stationary branch requires aligned non-transposed panel owners");
    vector<T> panel(h), partial(h), aligned(h), temporary(h);
    int aligned_desc[DDLA_DLEN_], temporary_desc[DDLA_DLEN_], result_desc[DDLA_DLEN_];
    auto describe = [&](vector<T>& storage, int* desc,
                        int rows, int cols, int mb, int nb, int rs, int cs) {
        DDLA_CHECK(ddlaDescInit(desc, h, rows, cols, mb, nb, rs, cs));
        storage.clear();
        storage.resize(std::max(std::size_t(1), std::size_t(desc[DDLA_LLD_]) *
            num_loc(cols, nb, col, cs, pc)));
    };

    for (int p0 = 0; p0 < panels; ) {
        const int width = std::min(panel_block, panels - p0);
        const int c_owner = indxg2p(p0, panel_block,
                                   dc[keep_a ? DDLA_CSRC_ : DDLA_RSRC_], keep_a ? pc : pr);
        const int root = transpose_output ? 0 : c_owner;
        panel.clear(); partial.clear();
        panel.resize(std::max(std::size_t(1), std::size_t(kloc) * width));
        partial.resize(std::max(std::size_t(1), std::size_t(floc) * width));

        // A descriptor view of this one physical row/column block. Preserve
        // the input LLD, and shift only the pointer on ranks owning the block.
        int view[DDLA_DLEN_];
        std::copy(thin, thin + DDLA_DLEN_, view);
        view[DDLA_M_] = thin_k_rows ? k : width;
        view[DDLA_N_] = thin_k_rows ? width : k;
        const int owner = indxg2p(p0, panel_block,
            thin[thin_k_rows ? DDLA_CSRC_ : DDLA_RSRC_], thin_k_rows ? pc : pr);
        view[thin_k_rows ? DDLA_CSRC_ : DDLA_RSRC_] = owner;
        const int offset = num_loc(p0, panel_block, thin_k_rows ? col : r,
            thin[thin_k_rows ? DDLA_CSRC_ : DDLA_RSRC_], thin_k_rows ? pc : pr);
        const T* input = x;
        if ((thin_k_rows ? col : r) == owner)
            input += thin_k_rows ? std::size_t(offset) * thin[DDLA_LLD_] : offset;
        const int* input_desc = view;
        char panel_op = tx;
        if (thin_k_rows != k_rows) {
            // The thin panel's k runs along the other process dimension. One
            // transpose lays it out like the stationary k blocks (any source
            // coordinates), so the broadcast below is always untransposed.
            describe(aligned, aligned_desc, view[DDLA_N_], view[DDLA_M_], view[DDLA_NB_], view[DDLA_MB_],
                     k_rows ? ks : 0, k_rows ? 0 : ks);
            pgeadd<Backend>(h, tx=='C'?'C':'T', 'N', aligned_desc[DDLA_M_], aligned_desc[DDLA_N_],
                T(1), input, 1, 1, view, T(0), static_cast<const T*>(nullptr), 1, 1, aligned_desc, aligned.data(), 1, 1, aligned_desc);
            input = aligned.data();
            input_desc = aligned_desc;
            panel_op = tx == 'N' ? 'T' : 'N';
        }
        transport_block<Backend>(h, k_rows ? 'C' : 'R', 'N',
            input_desc[DDLA_M_], input_desc[DDLA_N_], input, 0, 0, input_desc, panel.data());
        const int panel_ld = k_rows ? std::max(1, kloc) : width;

        if (floc > 0 && kloc > 0) {
            if (keep_a)
                gemm<Backend>(h, ta, panel_op, floc, width, kloc, alpha, a, da[DDLA_LLD_],
                              panel.data(), panel_ld, T(0), partial.data(), floc);
            else
                gemm<Backend>(h, panel_op, tb, width, floc, kloc, alpha, panel.data(), panel_ld,
                              b, db[DDLA_LLD_], T(0), partial.data(), width);
        } else {
            DDLA_CHECK(ddlaMemset(h, partial.data(), 0, std::size_t(floc)*width*sizeof(T)));
        }
        // In place: the root's partial becomes the reduced panel.
        commReduceSum<Backend>(h, reduction, partial.data(), partial.data(), std::size_t(floc)*width, root);

        int output[DDLA_DLEN_];
        std::copy(dc, dc + DDLA_DLEN_, output);
        const int cm = keep_a ? m : width, cn = keep_a ? width : n;
        output[DDLA_M_] = cm;
        output[DDLA_N_] = cn;
        output[keep_a ? DDLA_CSRC_ : DDLA_RSRC_] = c_owner;
        // Without a transpose the roots are C's owners and their partials
        // already have C's local layout; only a descriptor is needed.
        DDLA_CHECK(ddlaDescInit(result_desc, h, cm, cn, dc[DDLA_MB_], dc[DDLA_NB_],
                                output[DDLA_RSRC_], output[DDLA_CSRC_]));
        const T* result = partial.data();
        const int* rdesc = result_desc;
        char result_op = 'N';

        if (transpose_output) {
            // A transposed stationary input leaves the result distributed in
            // the opposite direction. Transpose the local reduced slab; the
            // final pgeadd redistributes it back into C's orientation.
            describe(temporary, temporary_desc, cn, cm, dc[DDLA_NB_], dc[DDLA_MB_],
                               k_rows ? root : fs, k_rows ? fs : root);
            if ((k_rows ? r : col) == root && floc > 0)
                geam<Backend>(h, 'T', 'N', keep_a ? width : floc, keep_a ? floc : width,
                    T(1), partial.data(), keep_a ? floc : width, T(0), static_cast<const T*>(nullptr), 1,
                    temporary.data(), temporary_desc[DDLA_LLD_]);
            result = temporary.data();
            rdesc = temporary_desc;
            result_op = 'T';
        }

        // Every rank joins because a transposed result moves between ranks.
        // Only C's owners hold part of this panel, so beta is applied once,
        // and the descriptor view keeps the rest of C, including padding, untouched.
        const int co = num_loc(p0, panel_block, keep_a ? col : r,
                              dc[keep_a ? DDLA_CSRC_ : DDLA_RSRC_], keep_a ? pc : pr);
        T* dest = c + (keep_a ? std::size_t(co)*dc[DDLA_LLD_] : co);
        pgeadd<Backend>(h, result_op, 'N', cm, cn, T(1), result, 1, 1, rdesc,
                        beta, dest, 1, 1, output, dest, 1, 1, output);
        p0 += width;
    }
}

#define DDLA_INSTANTIATE_STATIONARY(B, T) \
    template void stationary<B,T>(const DdlaHandle_t&, bool, char, char, int, int, int, \
        const T&, const T*, const int*, const T*, const int*, const T&, T*, const int*)
#define DDLA_INSTANTIATE_STATIONARY_BACKEND(B) \
    DDLA_INSTANTIATE_STATIONARY(B, float); \
    DDLA_INSTANTIATE_STATIONARY(B, double); \
    DDLA_INSTANTIATE_STATIONARY(B, std::complex<float>); \
    DDLA_INSTANTIATE_STATIONARY(B, std::complex<double>)
#if DDLA_HAS_CPU
DDLA_INSTANTIATE_STATIONARY_BACKEND(DdlaBackend::CPU);
#endif
#if DDLA_HAS_GPU
DDLA_INSTANTIATE_STATIONARY_BACKEND(DdlaBackend::GPU);
#endif
#undef DDLA_INSTANTIATE_STATIONARY_BACKEND
#undef DDLA_INSTANTIATE_STATIONARY

}} // namespace ddla::pgemm_impl
