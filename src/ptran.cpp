#include <ddla/ptran.h>
#include <ddla/ddla.h>
#include <ddla/ddla_connector.h>
#include <ddla/ddla_stream.h>
#include <ddla/ddla_comm.h>
#include <ddla/geam.h>
#include <vector>
#include <algorithm>
#include <cassert>

namespace ddla{

template <typename T>
static inline MPI_Datatype mpi_type();

template <>
MPI_Datatype mpi_type<float>(){ return MPI_FLOAT; }
template <>
MPI_Datatype mpi_type<double>(){ return MPI_DOUBLE; }
template <>
MPI_Datatype mpi_type<std::complex<float>>(){ return MPI_CXX_FLOAT_COMPLEX; }
template <>
MPI_Datatype mpi_type<std::complex<double>>(){ return MPI_CXX_DOUBLE_COMPLEX; }

/**
 * @brief Distributed out-of-place matrix transpose (optionally conjugate).
 *
 * Strategy (NCCL path):
 *   1. Build a communication plan by scanning all global blocks.
 *   2. Pack: geam-transpose owned blocks into a contiguous d_sendbuf (D2D, async).
 *      Local blocks (src==dst) are transposed directly into d_AT.
 *   3. Communicate: all send/recv issued in a single ncclGroup — device-to-device,
 *      no host staging, no per-block synchronisation.
 *   4. Scatter: deviceMemcpy2D from d_recvbuf into d_AT (D2D, async).
 *
 * This replaces the old per-block MPI_Send/Recv + D2H/H2D loop.
 */
template <typename T>
void ptran(const T* d_A, const DdlaDesc& descA,
           T* d_AT, const DdlaDesc& descAT,
           bool conj)
{
    assert(descAT.m() == descA.n());
    assert(descAT.n() == descA.m());
    assert(descAT.mb() == descA.nb());
    assert(descAT.nb() == descA.mb());
    assert(descAT.irsrc() == descA.icsrc());
    assert(descAT.icsrc() == descA.irsrc());
    assert(descAT.nprows() == descA.nprows());
    assert(descAT.npcols() == descA.npcols());

    DdlaHandle_t handle = descA.ddla_handle();
    int myrank = handle->myid;
    int Pr = descA.nprows();
    int Pc = descA.npcols();
    int nprocs = Pr * Pc;

    int mA = descA.m();
    int nA = descA.n();
    int mbA = descA.mb();
    int nbA = descA.nb();
    int irsrcA = descA.irsrc();
    int icsrcA = descA.icsrc();
    int mbAT = descAT.mb();
    int nbAT = descAT.nb();
    int irsrcAT = descAT.irsrc();
    int icsrcAT = descAT.icsrc();

    int nbr = (mA + mbA - 1) / mbA;
    int nbc = (nA + nbA - 1) / nbA;

    // ---- Phase 1: build communication plan ----
    struct BlockInfo {
        int g_row, g_col;       // global block origin
        int bm, bn;             // block dimensions (in A)
        int src_rank, dst_rank; // process ranks
        int lrow_A, lcol_A;     // local offset in d_A   (valid if src==myrank)
        int lrow_AT, lcol_AT;   // local offset in d_AT  (valid if dst==myrank)
        int offset;             // offset into send/recv buffer
    };

    std::vector<BlockInfo> local_blocks;  // src == dst == myrank
    std::vector<BlockInfo> send_blocks;   // src == myrank, dst != myrank
    std::vector<BlockInfo> recv_blocks;   // dst == myrank, src != myrank

    // Upper bounds: a process owns at most ceil(nbr/Pr)*ceil(nbc/Pc) blocks.
    int max_own = ((nbr + Pr - 1) / Pr) * ((nbc + Pc - 1) / Pc);
    local_blocks.reserve(max_own);
    send_blocks.reserve(max_own);
    recv_blocks.reserve(max_own);

    for(int br = 0; br < nbr; ++br){
        for(int bc = 0; bc < nbc; ++bc){
            int g_row = br * mbA;
            int g_col = bc * nbA;
            int bm = std::min(mbA, mA - g_row);
            int bn = std::min(nbA, nA - g_col);
            if(bm <= 0 || bn <= 0) continue;

            int src_row = indxg2p(g_row, mbA, irsrcA, Pr);
            int src_col = indxg2p(g_col, nbA, icsrcA, Pc);
            int dst_row = indxg2p(g_col, mbAT, irsrcAT, Pr);
            int dst_col = indxg2p(g_row, nbAT, icsrcAT, Pc);

            int src_rank = handle->rc_to_rank(src_row, src_col);
            int dst_rank = handle->rc_to_rank(dst_row, dst_col);

            BlockInfo info;
            info.g_row = g_row; info.g_col = g_col;
            info.bm = bm; info.bn = bn;
            info.src_rank = src_rank; info.dst_rank = dst_rank;
            info.offset = 0;

            if(src_rank == myrank){
                info.lrow_A = indxg2l(g_row, mbA, Pr);
                info.lcol_A = indxg2l(g_col, nbA, Pc);
            }
            if(dst_rank == myrank){
                info.lrow_AT = indxg2l(g_col, mbAT, Pr);
                info.lcol_AT = indxg2l(g_row, nbAT, Pc);
            }

            if(src_rank == myrank && dst_rank == myrank){
                local_blocks.push_back(info);
            } else if(src_rank == myrank){
                send_blocks.push_back(info);
            } else if(dst_rank == myrank){
                recv_blocks.push_back(info);
            }
        }
    }

    // Sort by peer rank, then by (g_row, g_col) so that both send and recv
    // sides visit blocks in the same order (required for NCCL group matching).
    auto cmp_send = [](const BlockInfo& a, const BlockInfo& b){
        if(a.dst_rank != b.dst_rank) return a.dst_rank < b.dst_rank;
        if(a.g_row != b.g_row) return a.g_row < b.g_row;
        return a.g_col < b.g_col;
    };
    auto cmp_recv = [](const BlockInfo& a, const BlockInfo& b){
        if(a.src_rank != b.src_rank) return a.src_rank < b.src_rank;
        if(a.g_row != b.g_row) return a.g_row < b.g_row;
        return a.g_col < b.g_col;
    };
    std::sort(send_blocks.begin(), send_blocks.end(), cmp_send);
    std::sort(recv_blocks.begin(), recv_blocks.end(), cmp_recv);

    // Compute contiguous offsets
    int send_total = 0;
    for(auto& b : send_blocks){
        b.offset = send_total;
        send_total += b.bm * b.bn;
    }
    int recv_total = 0;
    for(auto& b : recv_blocks){
        b.offset = recv_total;
        recv_total += b.bm * b.bn;
    }

    // ---- Phase 2: allocate device buffers ----
    T* d_sendbuf = nullptr;
    T* d_recvbuf = nullptr;
    if(send_total > 0) DEVICE_CHECK(deviceMalloc(&d_sendbuf, sizeof(T) * send_total));
    if(recv_total > 0) DEVICE_CHECK(deviceMalloc(&d_recvbuf, sizeof(T) * recv_total));

    deblasOperation_t op = conj ? DEBLAS_OP_C : DEBLAS_OP_T;
    T one = T(1);
    T zero = T(0);

    // ---- Phase 3: pack send data (geam transpose → contiguous d_sendbuf) ----
    for(auto& b : send_blocks){
        // src sub-block of A: bm x bn, lda = descA.lld()
        // transposed output:   bn x bm, ldc = bn (compact in sendbuf)
        BLAS_CHECK(deblasGeam(handle->blasH, op, DEBLAS_OP_N,
                              b.bn, b.bm,
                              one,
                              d_A + b.lrow_A + b.lcol_A * descA.lld(), descA.lld(),
                              zero,
                              d_sendbuf + b.offset, b.bn,
                              d_sendbuf + b.offset, b.bn));
    }

    // ---- Phase 4: local transpose (geam directly d_A → d_AT) ----
    for(auto& b : local_blocks){
        BLAS_CHECK(deblasGeam(handle->blasH, op, DEBLAS_OP_N,
                              b.bn, b.bm,
                              one,
                              d_A + b.lrow_A + b.lcol_A * descA.lld(), descA.lld(),
                              zero,
                              d_AT + b.lrow_AT + b.lcol_AT * descAT.lld(), descAT.lld(),
                              d_AT + b.lrow_AT + b.lcol_AT * descAT.lld(), descAT.lld()));
    }

    // ---- Phase 5: communication ----
    #ifdef DDLA_USE_CCL
    // NCCL: batch all send/recv in one group — device-to-device, no host staging.
    if(send_total > 0 || recv_total > 0){
        CCL_CHECK(ncclGroupStart());
        for(auto& b : send_blocks){
            CCL_CHECK(cclSend(d_sendbuf + b.offset, b.bm * b.bn,
                              b.dst_rank, handle->nccl_comm, handle->stream));
        }
        for(auto& b : recv_blocks){
            CCL_CHECK(cclRecv(d_recvbuf + b.offset, b.bm * b.bn,
                              b.src_rank, handle->nccl_comm, handle->stream));
        }
        CCL_CHECK(ncclGroupEnd());
    }
    #else
    // MPI fallback: Alltoallv with host staging (single collective, not per-block).
    if(send_total > 0 || recv_total > 0){
        std::vector<int> sendcounts(nprocs, 0), recvcounts(nprocs, 0);
        std::vector<int> sdispls(nprocs, 0), rdispls(nprocs, 0);
        for(auto& b : send_blocks) sendcounts[b.dst_rank] += b.bm * b.bn;
        for(auto& b : recv_blocks) recvcounts[b.src_rank] += b.bm * b.bn;
        for(int p = 1; p < nprocs; p++){
            sdispls[p] = sdispls[p-1] + sendcounts[p-1];
            rdispls[p] = rdispls[p-1] + recvcounts[p-1];
        }
        std::vector<T> h_sendbuf(send_total), h_recvbuf(recv_total);
        if(send_total > 0){
            DEVICE_CHECK(deviceMemcpyAsync(h_sendbuf.data(), d_sendbuf,
                sizeof(T) * send_total, deviceMemcpyDeviceToHost, handle->stream));
            DEVICE_CHECK(deviceStreamSynchronize(handle->stream));
        }
        MPI_Alltoallv(h_sendbuf.data(), sendcounts.data(), sdispls.data(), mpi_type<T>(),
                      h_recvbuf.data(), recvcounts.data(), rdispls.data(), mpi_type<T>(),
                      handle->comm);
        if(recv_total > 0){
            DEVICE_CHECK(deviceMemcpyAsync(d_recvbuf, h_recvbuf.data(),
                sizeof(T) * recv_total, deviceMemcpyHostToDevice, handle->stream));
            DEVICE_CHECK(deviceStreamSynchronize(handle->stream));
        }
    }
    #endif

    // ---- Phase 6: scatter recv data into d_AT (D2D memcpy2D) ----
    for(auto& b : recv_blocks){
        // recvbuf block: bn x bm, column-major, ldc = bn
        // destination in d_AT: bn x bm, column-major, ldc = descAT.lld()
        DEVICE_CHECK(deviceMemcpy2DAsync(
            d_AT + b.lrow_AT + b.lcol_AT * descAT.lld(), descAT.lld() * sizeof(T),
            d_recvbuf + b.offset, b.bn * sizeof(T),
            b.bn * sizeof(T), b.bm,
            deviceMemcpyDeviceToDevice, handle->stream));
    }

    DEVICE_CHECK(deviceStreamSynchronize(handle->stream));

    if(d_sendbuf) DEVICE_CHECK(deviceFree(d_sendbuf));
    if(d_recvbuf) DEVICE_CHECK(deviceFree(d_recvbuf));
}

template void ptran<float>(const float* d_A, const DdlaDesc& descA, float* d_AT, const DdlaDesc& descAT, bool conj);
template void ptran<double>(const double* d_A, const DdlaDesc& descA, double* d_AT, const DdlaDesc& descAT, bool conj);
template void ptran<std::complex<float>>(const std::complex<float>* d_A, const DdlaDesc& descA, std::complex<float>* d_AT, const DdlaDesc& descAT, bool conj);
template void ptran<std::complex<double>>(const std::complex<double>* d_A, const DdlaDesc& descA, std::complex<double>* d_AT, const DdlaDesc& descAT, bool conj);

} // namespace ddla
