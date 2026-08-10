#include "ddla_stream_impl.h"
#include <stdexcept>
#include <cstring>

namespace ddla {

// ---------------------------------------------------------------------------
// Backend availability (compile-time capabilities from generated config)
// ---------------------------------------------------------------------------
bool ddlaBackendAvailable(DdlaBackend backend)
{
    switch (backend) {
    case DdlaBackend::CPU:
#if DDLA_HAS_CPU
        return true;
#else
        return false;
#endif
    case DdlaBackend::GPU:
#if DDLA_HAS_GPU
        return true;
#else
        return false;
#endif
    }
    return false;
}

// ---------------------------------------------------------------------------
// ddlaInit — allocate opaque handle, validate backend
// ---------------------------------------------------------------------------
ddlaStatus_t ddlaInit(DdlaHandle_t& handle)
{
    return ddlaInit(handle, default_backend_v);
}

ddlaStatus_t ddlaInit(DdlaHandle_t& handle, DdlaBackend requested_backend)
{
    // Validate availability
    if (!ddlaBackendAvailable(requested_backend)) {
        throw std::runtime_error(
            "Requested backend is not available in this build of LibDDLA");
    }

    // Allocate fresh handle — do NOT read the incoming value (callers may
    // pass uninitialized pointer variables; reading them is UB).
    handle = new DdlaStream();
    handle->backend = requested_backend;
    return ddlaStatus_t::DDLA_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// ddlaGetBackend
// ---------------------------------------------------------------------------
DdlaBackend ddlaGetBackend(const DdlaHandle_t& handle)
{
    if (handle == nullptr) {
        throw std::runtime_error("ddlaGetBackend: handle is null");
    }
    return handle->backend;
}

// ---------------------------------------------------------------------------
// Shared by both ddlaSet overloads: validate the handle backend is
// available in this build, and collectively verify every MPI rank picked
// the same backend. Packs {value, -value} into a single MPI_MAX Allreduce
// to recover both the max and the min in one collective call instead of two.
// ---------------------------------------------------------------------------
static void resolve_and_validate_backend(DdlaHandle_t handle, const MPI_Comm& comm)
{
    const DdlaBackend resolved = handle->backend;

    if (!ddlaBackendAvailable(resolved)) {
        throw std::runtime_error(
            "Resolved backend is not available in this build of LibDDLA");
    }

    int send[2] = { static_cast<int>(resolved), -static_cast<int>(resolved) };
    int recv[2] = { 0, 0 };
    MPI_CHECK(MPI_Allreduce(send, recv, 2, MPI_INT, MPI_MAX, comm));
    int max_backend = recv[0];
    int min_backend = -recv[1];
    if (max_backend != min_backend) {
        throw std::runtime_error(
            "Not all MPI ranks selected the same backend");
    }
}

// ---------------------------------------------------------------------------
// ddlaSet — initialize process grid, verify backend consistency collectively
// ---------------------------------------------------------------------------
ddlaStatus_t ddlaSet(DdlaHandle_t handle, const MPI_Comm& comm, const char& major)
{
    if (handle == nullptr) {
        throw std::runtime_error("ddlaSet: handle is null");
    }

    resolve_and_validate_backend(handle, comm);

    // CPU handles must not execute GPU device setup
    handle->init(comm, major);
    handle->initialized = true;
    return ddlaStatus_t::DDLA_STATUS_SUCCESS;
}

ddlaStatus_t ddlaSet(DdlaHandle_t handle, const MPI_Comm& comm,
              const int& nprows, const int& npcols, const char& major)
{
    if (handle == nullptr) {
        throw std::runtime_error("ddlaSet: handle is null");
    }

    resolve_and_validate_backend(handle, comm);

    handle->init(nprows, npcols, comm, major);
    handle->initialized = true;
    return ddlaStatus_t::DDLA_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// ddlaDestroy — idempotent cleanup
// ---------------------------------------------------------------------------
ddlaStatus_t ddlaDestroy(DdlaHandle_t& handle)
{
    if (handle == nullptr) return ddlaStatus_t::DDLA_STATUS_SUCCESS;
    if (handle->destroyed) {
        handle = nullptr;
        return ddlaStatus_t::DDLA_STATUS_SUCCESS;
    }
    handle->clean();
    handle->destroyed = true;
    delete handle;
    handle = nullptr;
    return ddlaStatus_t::DDLA_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// ddlaGetStream — CPU returns nullptr; GPU returns the stream handle
// ---------------------------------------------------------------------------
void* ddlaGetStream(const DdlaHandle_t& handle)
{
    if (handle == nullptr) return nullptr;
    if (handle->backend == DdlaBackend::CPU) return nullptr;
#if defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)
    return reinterpret_cast<void*>(handle->stream);
#else
    // GPU backend requested but this is a CPU-only build — unreachable
    // if ddlaBackendAvailable is checked correctly before init.
    return nullptr;
#endif
}

// ---------------------------------------------------------------------------
// Memory helpers — dispatch by handle backend, with uniform validation
// ---------------------------------------------------------------------------
ddlaStatus_t ddlaMalloc(void** ptr, std::size_t bytes, const DdlaHandle_t& handle)
{
    if (handle == nullptr) return ddlaStatus_t::DDLA_STATUS_INVALID_HANDLE;
    if (ptr == nullptr) return ddlaStatus_t::DDLA_STATUS_INVALID_VALUE;

    // Zero-byte allocation: set *ptr = nullptr, return success
    if (bytes == 0) {
        *ptr = nullptr;
        return ddlaStatus_t::DDLA_STATUS_SUCCESS;
    }

    if (handle->backend == DdlaBackend::CPU) {
        *ptr = std::malloc(bytes);
        return (*ptr != nullptr) ? ddlaStatus_t::DDLA_STATUS_SUCCESS
                                 : ddlaStatus_t::DDLA_STATUS_ALLOC_FAILED;
    }
    return runtimeMallocAsync(ptr, bytes, handle->stream)
        == runtimeSuccess ? ddlaStatus_t::DDLA_STATUS_SUCCESS
                          : ddlaStatus_t::DDLA_STATUS_INTERNAL_ERROR;
}

ddlaStatus_t ddlaFree(void* ptr, const DdlaHandle_t& handle)
{
    if (handle == nullptr) return ddlaStatus_t::DDLA_STATUS_INVALID_HANDLE;
    // Freeing nullptr is a valid no-op on both CPU and GPU
    if (ptr == nullptr) return ddlaStatus_t::DDLA_STATUS_SUCCESS;
    if (handle->backend == DdlaBackend::CPU) {
        std::free(ptr);
        return ddlaStatus_t::DDLA_STATUS_SUCCESS;
    }
    return runtimeFreeAsync(ptr, handle->stream)
        == runtimeSuccess ? ddlaStatus_t::DDLA_STATUS_SUCCESS
                          : ddlaStatus_t::DDLA_STATUS_INTERNAL_ERROR;
}

ddlaStatus_t ddlaMemcpy(void* dst, const void* src, std::size_t bytes,
                DdlaMemoryCopyKind kind, const DdlaHandle_t& handle)
{
    if (handle == nullptr) return ddlaStatus_t::DDLA_STATUS_INVALID_HANDLE;

    // Zero-byte copy: no-op, do not dereference src/dst
    if (bytes == 0) return ddlaStatus_t::DDLA_STATUS_SUCCESS;

    // Nonzero copy: reject null source or destination
    if (dst == nullptr || src == nullptr) return ddlaStatus_t::DDLA_STATUS_INVALID_VALUE;

    // Reject invalid copy kind values
    switch (kind) {
    case DdlaMemoryCopyKind::HostToDevice:
    case DdlaMemoryCopyKind::DeviceToHost:
    case DdlaMemoryCopyKind::DeviceToDevice:
        break;
    default:
        return ddlaStatus_t::DDLA_STATUS_INVALID_VALUE;
    }

    if (handle->backend == DdlaBackend::CPU) {
        std::memcpy(dst, src, bytes);
        return ddlaStatus_t::DDLA_STATUS_SUCCESS;
    }
    runtimeMemcpyKind dkind;
    switch (kind) {
    case DdlaMemoryCopyKind::HostToDevice:   dkind = runtimeMemcpyHostToDevice;   break;
    case DdlaMemoryCopyKind::DeviceToHost:   dkind = runtimeMemcpyDeviceToHost;   break;
    case DdlaMemoryCopyKind::DeviceToDevice: dkind = runtimeMemcpyDeviceToDevice; break;
    default: return ddlaStatus_t::DDLA_STATUS_INVALID_VALUE; // unreachable (validated above)
    }
    return runtimeMemcpyAsync(dst, src, bytes, dkind, handle->stream)
        == runtimeSuccess ? ddlaStatus_t::DDLA_STATUS_SUCCESS
                          : ddlaStatus_t::DDLA_STATUS_INTERNAL_ERROR;
}

ddlaStatus_t ddlaSynchronize(const DdlaHandle_t& handle)
{
    if (handle == nullptr) return ddlaStatus_t::DDLA_STATUS_INVALID_HANDLE;
    if (handle->backend == DdlaBackend::CPU) return ddlaStatus_t::DDLA_STATUS_SUCCESS;
    return runtimeStreamSynchronize(handle->stream)
        == runtimeSuccess ? ddlaStatus_t::DDLA_STATUS_SUCCESS
                          : ddlaStatus_t::DDLA_STATUS_INTERNAL_ERROR;
}

// ---------------------------------------------------------------------------
// ddlaSetStream — GPU handles only; CPU handles have no device stream
// ---------------------------------------------------------------------------
ddlaStatus_t ddlaSetStream(const DdlaHandle_t& handle, void* stream)
{
    if (handle == nullptr) return ddlaStatus_t::DDLA_STATUS_INVALID_HANDLE;
    if (handle->backend == DdlaBackend::CPU) {
        throw std::runtime_error(
            "ddlaSetStream: setting a stream requires a GPU handle");
    }
#if defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)
    handle->stream = static_cast<runtimeStream_t>(stream);
#endif
    // Re-bind the BLAS/solver handles so subsequent calls honor the new
    // stream (mirrors what DdlaStream::init does at creation time).
#if defined(DDLA_USE_CUDA)
    BLAS_CHECK(cublasSetStream(handle->blasH, handle->stream));
    SOLVER_CHECK(cusolverDnSetStream(handle->solverH, handle->stream));
#elif defined(DDLA_USE_HIP)
    BLAS_CHECK(hipblasSetStream(handle->blasH, handle->stream));
    SOLVER_CHECK(hipsolverSetStream(handle->solverH, handle->stream));
#endif
    return ddlaStatus_t::DDLA_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// Public accessors
// ---------------------------------------------------------------------------
MPI_Comm ddlaGetCommunicator(const DdlaHandle_t& handle)
{
    if (handle == nullptr) return MPI_COMM_NULL;
    return handle->comm;
}

int ddlaGetRank(const DdlaHandle_t& handle)
{
    if (handle == nullptr) return -1;
    if (handle->comm == MPI_COMM_NULL) return -1;
    int rank;
    MPI_Comm_rank(handle->comm, &rank);
    return rank;
}

int ddlaGetSize(const DdlaHandle_t& handle)
{
    if (handle == nullptr) return 0;
    if (handle->comm == MPI_COMM_NULL) return 0;
    int size;
    MPI_Comm_size(handle->comm, &size);
    return size;
}

void ddlaGetGridCoords(const DdlaHandle_t& handle,
                          int& myprow, int& mypcol)
{
    if (handle == nullptr) {
        myprow = -1; mypcol = -1;
        return;
    }
    myprow = handle->myprow_;
    mypcol = handle->mypcol_;
}

void ddlaGetGridDims(const DdlaHandle_t& handle,
                        int& nprows, int& npcols)
{
    if (handle == nullptr) {
        nprows = 0; npcols = 0;
        return;
    }
    nprows = handle->nprows_;
    npcols = handle->npcols_;
}

void ddlaRankToRc(const DdlaHandle_t& handle,
                     int rank, int& row, int& col)
{
    if (handle == nullptr) {
        row = -1; col = -1;
        return;
    }
    handle->rank_to_rc(rank, row, col);
}

int ddlaRcToRank(const DdlaHandle_t& handle, int row, int col)
{
    if (handle == nullptr) return -1;
    return handle->rc_to_rank(row, col);
}

MPI_Comm ddlaGetRowCommunicator(const DdlaHandle_t& handle)
{
    if (handle == nullptr) return MPI_COMM_NULL;
    return handle->row_comm;
}

MPI_Comm ddlaGetColCommunicator(const DdlaHandle_t& handle)
{
    if (handle == nullptr) return MPI_COMM_NULL;
    return handle->col_comm;
}

} // namespace ddla
