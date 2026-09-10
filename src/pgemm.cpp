#include <ddla/ddla.h>
#include "ddla_desc.h"
#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <string>
#include "ddla_connector.h"
#include <ddla/ddla_config.h>
#include <ddla/ddla_desc.h>
#include "ddla_stream_impl.h"
#include <vector>
#include "transport_block.h"
#include "gemm.h"
#include "scal.h"

// CPU-only includes (no GPU vendor headers pulled in by these; safe in any
// build -- DDLA_HAS_CPU below still gates whether the CPU pgemm/transport
// paths are ever compiled).
#include <cstdlib>
#include <complex>
#include <new>

namespace ddla {

inline const char* pgemm_backend_name(DdlaBackend backend)
{
    return backend == DdlaBackend::CPU ? "CPU" : "GPU";
}


template <DdlaBackend Backend, typename T>
void pgemm(
    const DdlaHandle_t& handle, const char& transa, const char& transb,
    const int& m, const int& n, const int& k,
    const T& alpha,
    const T* A, const int* descA,
    const T* B, const int* descB,
    const T& beta,
    T* C, const int* descC)
{
    static_assert(Backend != DdlaBackend::CPU || DDLA_HAS_CPU,
                  "CPU pgemm is not available in this LibDDLA build");
    static_assert(Backend != DdlaBackend::GPU || DDLA_HAS_GPU,
                  "GPU pgemm is not available in this LibDDLA build");

    DdlaHandle_t h = handle;

    if (h == nullptr) {
        throw std::runtime_error("pgemm: null handle");
    }
    // Each descriptor is validated against this handle's process grid
    // (dense block-cyclic type, slot ranges, LLD vs LOCr); a descriptor
    // built against a different process grid is rejected there.
    check_desc(descA, handle);
    check_desc(descB, handle);
    check_desc(descC, handle);
    int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
    ddlaGetGridDims(handle, nprows, npcols);
    ddlaGetGridCoords(handle, myprow, mypcol);

    const DdlaBackend actual_backend = ddlaGetBackend(h);
    if (actual_backend != Backend) {
        throw std::runtime_error(
            std::string("pgemm: template backend ") + pgemm_backend_name(Backend) +
            " does not match descriptor handle backend " +
            pgemm_backend_name(actual_backend));
    }

    // ---- SUMMA implementation (double-buffered panel pipeline) -------------
    assert(transa == 'N' || transa == 'T' || transa == 'C');
    assert(transb == 'N' || transb == 'T' || transb == 'C');
    assert(m >= 0 && n >= 0 && k >= 0);

    const int opA_m = (transa == 'N') ? descA[DDLA_M_] : descA[DDLA_N_];
    const int opA_n = (transa == 'N') ? descA[DDLA_N_] : descA[DDLA_M_];
    const int opB_m = (transb == 'N') ? descB[DDLA_M_] : descB[DDLA_N_];
    const int opB_n = (transb == 'N') ? descB[DDLA_N_] : descB[DDLA_M_];
    assert(m <= opA_m);
    assert(k <= opA_n);
    assert(k <= opB_m);
    assert(n <= opB_n);
    assert(m <= descC[DDLA_M_] && n <= descC[DDLA_N_]);
    {
        int mbA, kbA, kbB, nbB, mbC, nbC;
        mbC = descC[DDLA_MB_];
        nbC = descC[DDLA_NB_];
        if (transa == 'N') {
            mbA = descA[DDLA_MB_];
            kbA = descA[DDLA_NB_];
        } else {
            mbA = descA[DDLA_NB_];
            kbA = descA[DDLA_MB_];
        }
        if (transb == 'N') {
            kbB = descB[DDLA_MB_];
            nbB = descB[DDLA_NB_];
        } else {
            kbB = descB[DDLA_NB_];
            nbB = descB[DDLA_MB_];
        }
        assert(mbA == mbC);
        assert(kbA == kbB);
        assert(nbB == nbC);
    }

    int nb;
    if (transa == 'N')
        nb = descA[DDLA_NB_];
    else
        nb = descA[DDLA_MB_];

    const int m_loc_C = num_loc(m, descC[DDLA_MB_], myprow, descC[DDLA_RSRC_], nprows);
    const int n_loc_C = num_loc(n, descC[DDLA_NB_], mypcol, descC[DDLA_CSRC_], npcols);
    const int lldC = descC[DDLA_LLD_];

    // ---- Degenerate-size fast paths (F2) -----------------------------------
    // ScaLAPACK's PxGEMM contract: an empty output (m==0 or n==0) is a no-op,
    // and k==0 reduces to C := beta*C with A/B untouched. Both cases used to
    // fall through the k-loop below, which never executes when k==0, so C
    // was silently left unscaled by beta -- this restores that contract for
    // both backends without ever allocating the SUMMA panel buffers.
    if (m == 0 || n == 0) {
        return;
    }
    if (k == 0) {
        // F2 fast path: C := beta*C via one unified ddla::scal<Backend,T>
        // call per column.
        if (m_loc_C > 0 && n_loc_C > 0) {
            for (int j = 0; j < n_loc_C; ++j) {
                ddla::scal<Backend, T>(
                    h, m_loc_C, beta,
                    C + static_cast<std::size_t>(j) * lldC, 1);
            }
        }
        return;
    }

    // Buffer sizing
    const int m_loc_A = num_loc((transa == 'N') ? m : k, descA[DDLA_MB_], myprow, descA[DDLA_RSRC_], nprows);
    const int n_loc_A = num_loc((transa == 'N') ? k : m, descA[DDLA_NB_], mypcol, descA[DDLA_CSRC_], npcols);
    const int m_loc_B = num_loc((transb == 'N') ? k : n, descB[DDLA_MB_], myprow, descB[DDLA_RSRC_], nprows);
    const int n_loc_B = num_loc((transb == 'N') ? n : k, descB[DDLA_NB_], mypcol, descB[DDLA_CSRC_], npcols);

    // ---- k-panel blocking: one local gemm per run of consecutive k-blocks ---
    // A local gemm is cheaper per flop when it covers a run of k-blocks instead
    // of a single one: with nb = 128 and a 1024-wide k panel that is 8 panel
    // transports (8 broadcasts) feeding ONE local gemm of k = 1024. The target
    // panel width defaults to DDLA_PGEMM_KPANEL (1024); a value below nb
    // degenerates to the classic one-block-per-gemm pipeline. The transposed
    // transport path has no multi-block gather yet, so any transposed operand
    // stays one block per gemm.
    constexpr int kDefaultPanelWidth = 1024;
    const int kfull = k / nb;
    const int ktail = k - kfull * nb;
    const int nb_eff = (nb > 0) ? nb : 1;
    int panel_width = kDefaultPanelWidth;
    {
        const char* s = std::getenv("DDLA_PGEMM_KPANEL");
        if (s != nullptr && *s != '\0') {
            const int v = std::atoi(s);
            if (v > 0) panel_width = v;
        }
    }
    int blocks_per_gemm = std::max(1, panel_width / nb_eff);
    // The transposed transport path (trans != 'N') has no multi-block gather
    // yet -- it would move only the first block of a run and leave the rest of
    // the panel uninitialized -- so those operands stay one block per gemm.
    if (transa != 'N' || transb != 'N') blocks_per_gemm = 1;

    struct KStage { int start; int cnt; int kk; };  // [start, start+kk) slice of k
    std::vector<KStage> stages;
    stages.reserve(static_cast<std::size_t>(kfull / blocks_per_gemm + 2));
    for (int b = 0; b < kfull; b += blocks_per_gemm) {
        const int cnt = std::min(blocks_per_gemm, kfull - b);
        stages.push_back(KStage{b * nb, cnt, cnt * nb});
    }
    if (ktail > 0) stages.push_back(KStage{kfull * nb, 1, ktail});
    const int panel_max = blocks_per_gemm * nb;  // widest stage (>= nb)

    // Scratch buffers (GPU stream-scratch or CPU heap) for the SUMMA panel
    // pipeline, freed explicitly below.  DDLA_CHECK and RUNTIME_CHECK throw
    // std::runtime_error on failure (uncaught it terminates the rank), and the
    // transport_block/gemm/scal calls below can throw on a null handle or a
    // backend mismatch -- both validated above, but a failure between these
    // allocations and the frees at the end leaks the scratch buffers on that
    // rank.  The process terminates via the uncaught exception, so the OS
    // reclaims them; this is not a runaway leak in practice.
    const int buffer_max = 2;
    T* d_A_temp[buffer_max] = {nullptr, nullptr};
    T* d_B_temp[buffer_max] = {nullptr, nullptr};
    int count_a = ((transa == 'N') ? std::max(m_loc_A, m_loc_C) : std::max(n_loc_A, m_loc_C)) * panel_max;
    int count_b = panel_max * ((transb == 'N') ? std::max(n_loc_B, n_loc_C) : std::max(m_loc_B, n_loc_C));
    count_a = std::max(1, count_a);
    count_b = std::max(1, count_b);
    // One-block staging tile. The B panel is stored as (kk x n_loc) with
    // ld = kk so one gemm sees the whole run, but transport_block writes a
    // (w x n_loc) tile with ld = w; every block is staged here and copied
    // into the panel with the panel pitch.
    int count_blk = nb * ((transb == 'N') ? std::max(n_loc_B, n_loc_C) : std::max(m_loc_B, n_loc_C));
    count_blk = std::max(1, count_blk);  // keep d_B_blk a real allocation even at zero local extent
    // ddlaMalloc/ddlaFree dispatch on the handle's backend at run time -- host
    // heap on CPU, and on GPU the stream-ordered cudaMallocAsync/hipMallocAsync
    // pool on the handle's stream (no device-wide synchronize per call).  Going
    // through the handle rather than the type-templated runtimeMallocAsync also
    // keeps the stream type out of these call sites, so no compile-time
    // backend split is needed.
    T* d_B_blk = nullptr;
    DDLA_CHECK(ddlaMalloc(reinterpret_cast<void**>(&d_B_blk), count_blk * sizeof(T), h));
    for (int i = 0; i < buffer_max; i++) {
        DDLA_CHECK(ddlaMalloc(reinterpret_cast<void**>(&d_A_temp[i]), count_a * sizeof(T), h));
        DDLA_CHECK(ddlaMalloc(reinterpret_cast<void**>(&d_B_temp[i]), count_b * sizeof(T), h));
    }

    // Wait out the in-flight panel transport before the local gemm reads it
    // (GPU only: CPU BLAS is synchronous and every MPI call in
    // transport_block<DdlaBackend::CPU> is blocking).
    auto sync_panel = [&]() {
        if constexpr (Backend != DdlaBackend::CPU) {
            RUNTIME_CHECK<Backend>(runtimeStreamSynchronize<Backend>(h->stream));
            RUNTIME_CHECK<Backend>(runtimeStreamSynchronize<Backend>(h->stream_data));
        }
    };

    int temp_buffer = 0;

    auto get_data = [&](int s) {
        if (s < 0 || s >= static_cast<int>(stages.size())) return;
        const int ks = stages[s].start;
        const int cnt = stages[s].cnt;
        const int kk = stages[s].kk;

        // Gather the stage's blocks one transport_block call each. The A tile
        // (sData 'C', transa 'N') is (rows x w) with ld = rows, so it lands
        // straight in the panel at column offset b * w; the B tile (sData 'R',
        // transb 'N') is (w x cols) with ld = w, which only matches the panel
        // pitch kk when the stage is one block wide -- otherwise it is staged
        // in d_B_blk and copied in with the panel pitch. w = nb for every full
        // block; the tail stage has w = ktail.
        for (int b = 0; b < cnt; ++b) {
            const int g = ks + b * nb;
            const int w = std::min(nb, k - g);

            if (transa == 'N') {
                ddla::transport_block<Backend, T>(handle,
                    'C', transa,
                    m, w,
                    A, 0, g, descA,
                    d_A_temp[temp_buffer] + (std::size_t)b * w * m_loc_A);
            } else {
                ddla::transport_block<Backend, T>(handle,
                    'R', transa,
                    w, m,
                    A, g, 0, descA,
                    d_A_temp[temp_buffer]);
            }

            if (transb == 'N') {
                if (cnt != 1) {  // multi-block: tile ld w != panel ld kk, stage + repack
                    ddla::transport_block<Backend, T>(handle,
                        'R', transb,
                        w, n,
                        B, g, 0, descB,
                        d_B_blk);
                    ddla::copy2D<Backend, T>(h,
                        d_B_temp[temp_buffer] + (std::size_t)b * w, kk,
                        d_B_blk, w,
                        w, n_loc_B);
                } else {
                    ddla::transport_block<Backend, T>(handle,
                        'R', transb,
                        w, n,
                        B, g, 0, descB,
                        d_B_temp[temp_buffer]);
                }
            } else {
                ddla::transport_block<Backend, T>(handle,
                    'C', transb,
                    n, w,
                    B, 0, g, descB,
                    d_B_temp[temp_buffer]);
            }
        }
    };

    get_data(0);
    bool first_gemm = true;
    for (std::size_t s = 0; s < stages.size(); ++s) {
        const int kb = stages[s].kk;
        sync_panel();
        if (m_loc_C > 0 && n_loc_C > 0) {
            const T gemm_beta = first_gemm ? beta : static_cast<T>(1.0);
            ddla::gemm<Backend>(
                h, transa, 'N',
                m_loc_C, n_loc_C, kb,
                alpha,
                d_A_temp[temp_buffer], transa == 'N' ? static_cast<int>(m_loc_C) : kb,
                d_B_temp[temp_buffer], kb,
                gemm_beta,
                C, lldC);
            first_gemm = false;
        }
        temp_buffer = (temp_buffer + 1) % buffer_max;
        get_data(static_cast<int>(s) + 1);
    }
    sync_panel();
    for (int i = 0; i < buffer_max; i++) {
        DDLA_CHECK(ddlaFree(d_A_temp[i], h));
        DDLA_CHECK(ddlaFree(d_B_temp[i], h));
    }
    DDLA_CHECK(ddlaFree(d_B_blk, h));
}

// ---------------------------------------------------------------------------
// Explicit instantiations
// ---------------------------------------------------------------------------
#define INSTANTIATE_PGEMM(BACKEND, TYPE)                                     \
    template void pgemm<BACKEND, TYPE>(                              const DdlaHandle_t&, \
        const char&, const char&, const int&, const int&, const int&,          \
        const TYPE&, const TYPE*, const int*,                            \
        const TYPE*, const int*,                                         \
        const TYPE&, TYPE*, const int*)

#if DDLA_HAS_CPU
INSTANTIATE_PGEMM(DdlaBackend::CPU, float);
INSTANTIATE_PGEMM(DdlaBackend::CPU, double);
INSTANTIATE_PGEMM(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_PGEMM(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_PGEMM(DdlaBackend::GPU, float);
INSTANTIATE_PGEMM(DdlaBackend::GPU, double);
INSTANTIATE_PGEMM(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_PGEMM(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_PGEMM

} // namespace ddla
