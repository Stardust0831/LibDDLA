#ifndef DDLA_HANDLE_T_H
#define DDLA_HANDLE_T_H

#include <cstddef>
#include <mpi.h>
#include <ddla/ddla_config.h>

namespace ddla {

// ---------------------------------------------------------------------------
// Backend selection
// ---------------------------------------------------------------------------
enum class DdlaBackend {
    CPU,  ///< CPU-only (host BLAS, MPI communication)
    GPU   ///< GPU (CUDA or HIP, depending on build configuration)
};

/// Compile-time default used by backend-templated compute interfaces.
/// Dual CPU+GPU builds prefer GPU, matching ddlaInit(handle).
inline constexpr DdlaBackend default_backend_v =
#if DDLA_HAS_GPU
    DdlaBackend::GPU;
#elif DDLA_HAS_CPU
    DdlaBackend::CPU;
#else
    // Unreachable: CMake requires at least one backend (CPU, CUDA, or HIP).
    // Kept as a hard error so the header stays self-consistent.
#error "LibDDLA requires at least one backend (CPU or GPU)"
#endif

/// Memory copy direction for ddlaMemcpy.
enum class DdlaMemoryCopyKind {
    HostToDevice,
    DeviceToHost,
    DeviceToDevice
};

/// Status codes returned by the public API functions.
enum class ddlaStatus_t {
    DDLA_STATUS_SUCCESS = 0,        ///< Operation completed successfully.
    DDLA_STATUS_INVALID_HANDLE,     ///< Null or invalid handle.
    DDLA_STATUS_INVALID_VALUE,      ///< Null pointer, invalid enum value, etc.
    DDLA_STATUS_ALLOC_FAILED,       ///< Memory allocation failed.
    DDLA_STATUS_BACKEND_UNAVAILABLE,///< Requested backend not compiled in.
    DDLA_STATUS_INTERNAL_ERROR      ///< Vendor library / device error.
};

// ---------------------------------------------------------------------------
// Opaque handle
// ---------------------------------------------------------------------------
class DdlaStream;                      // concrete type in private header
using DdlaHandle_t = DdlaStream*;      // ABI: opaque pointer

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

/// Create an uninitialized handle using the compile-time default backend
/// (`default_backend_v`: GPU when compiled with GPU support, else CPU).
ddlaStatus_t ddlaInit(DdlaHandle_t& handle);

/// Create a handle requesting a specific backend.
/// @throws std::runtime_error if the requested backend is unavailable.
ddlaStatus_t ddlaInit(DdlaHandle_t& handle, DdlaBackend requested_backend);

/// Initialize process grid and allocate backend resources.
/// Backend is resolved and validated collectively.
ddlaStatus_t ddlaSet(DdlaHandle_t handle, const MPI_Comm& comm = MPI_COMM_WORLD,
                      const char& major = 'R');

/// Initialize with explicit grid dimensions.
ddlaStatus_t ddlaSet(DdlaHandle_t handle, const MPI_Comm& comm,
                      const int& nprows, const int& npcols,
                      const char& major = 'R');

/// Release all resources; idempotent (safe to call multiple times).
ddlaStatus_t ddlaDestroy(DdlaHandle_t& handle);

// ---------------------------------------------------------------------------
// Backend queries
// ---------------------------------------------------------------------------

/// True if the library was compiled with support for the given backend.
bool ddlaBackendAvailable(DdlaBackend backend);

/// Return the resolved backend for this handle.
/// @throws std::runtime_error if @p handle is null.
DdlaBackend ddlaGetBackend(const DdlaHandle_t& handle);

// ---------------------------------------------------------------------------
// Memory and synchronization (vendor-neutral)
// ---------------------------------------------------------------------------

/// Allocate backend-appropriate memory.
///
/// CPU handles allocate host memory (malloc).  GPU handles allocate device
/// memory on the handle's default stream.
///
/// - Returns DDLA_STATUS_INVALID_VALUE if @p ptr is null,
///   DDLA_STATUS_INVALID_HANDLE if @p handle is null.
/// - Zero-byte allocation: sets @p *ptr = nullptr and returns
///   DDLA_STATUS_SUCCESS.  No backend call is made.
/// - Nonzero CPU allocation: DDLA_STATUS_SUCCESS on success,
///   DDLA_STATUS_ALLOC_FAILED if malloc fails.
/// - Nonzero GPU allocation: DDLA_STATUS_SUCCESS on success,
///   DDLA_STATUS_INTERNAL_ERROR on device error.
///
/// @param[out] ptr   Set to the allocated pointer on success.
/// @param bytes      Allocation size in bytes.
/// @param handle     Backend handle (must not be null).
ddlaStatus_t ddlaMalloc(void** ptr, std::size_t bytes, const DdlaHandle_t& handle);

/// Free backend-appropriate memory.
///
/// - Returns DDLA_STATUS_INVALID_HANDLE if @p handle is null.
/// - Freeing a null pointer is a valid no-op (DDLA_STATUS_SUCCESS).
/// - CPU handles free host memory; GPU handles free device memory on the
///   handle's default stream.
ddlaStatus_t ddlaFree(void* ptr, const DdlaHandle_t& handle);

/// Copy data between host and device (or within host for CPU handles).
///
/// - Returns 1 if @p handle is null.
/// - Zero-byte copy: returns 0 without dereferencing @p src or @p dst.
/// - Nonzero copy: returns 1 if @p src or @p dst is null, or if @p kind
///   is not one of the three defined DdlaMemoryCopyKind enumerators.
/// - CPU handles perform an ordinary host memcpy (the @p kind is ignored
///   after validation).
/// - GPU handles use the appropriate runtimeMemcpyKind on the handle's
///   default stream.
///
/// CPU handles require host pointers.  GPU handles require pointers
/// allocated in the selected accelerator memory space.  No implicit
/// migration between address spaces is performed.
ddlaStatus_t ddlaMemcpy(void* dst, const void* src, std::size_t bytes,
                         DdlaMemoryCopyKind kind, const DdlaHandle_t& handle);

/// Synchronize the handle's default compute stream.
ddlaStatus_t ddlaSynchronize(const DdlaHandle_t& handle);

/// Set the compute stream of a GPU handle; subsequent library calls on this
/// handle are enqueued on @p stream.
///
/// The handle takes ownership of @p stream: it is destroyed by ddlaDestroy,
/// so the caller must not destroy it separately.
///
/// @throws std::runtime_error if the handle uses the CPU backend (CPU
/// handles have no device stream).
ddlaStatus_t ddlaSetStream(const DdlaHandle_t& handle, void* stream);

// ---------------------------------------------------------------------------
// Public accessors (replace direct field access)
// ---------------------------------------------------------------------------

/// Return the duplicated MPI communicator owned by the handle.
MPI_Comm ddlaGetCommunicator(const DdlaHandle_t& handle);

/// Return the row communicator (processes in the same grid row).
MPI_Comm ddlaGetRowCommunicator(const DdlaHandle_t& handle);

/// Return the column communicator (processes in the same grid column).
MPI_Comm ddlaGetColCommunicator(const DdlaHandle_t& handle);

/// Rank of this process within the handle's communicator.
int ddlaGetRank(const DdlaHandle_t& handle);

/// Size of the handle's communicator.
int ddlaGetSize(const DdlaHandle_t& handle);

/// Process-grid row and column coordinates of this rank.
void ddlaGetGridCoords(const DdlaHandle_t& handle,
                          int& myprow, int& mypcol);

/// Number of process rows and columns in the grid.
void ddlaGetGridDims(const DdlaHandle_t& handle,
                        int& nprows, int& npcols);

/// Convert a rank to (row, col) process-grid coordinates.
void ddlaRankToRc(const DdlaHandle_t& handle,
                     int rank, int& row, int& col);

/// Convert (row, col) process-grid coordinates to a rank.
int ddlaRcToRank(const DdlaHandle_t& handle, int row, int col);

/// Return the compute stream (GPU: device stream; CPU: nullptr/0).
void* ddlaGetStream(const DdlaHandle_t& handle);

} // namespace ddla

#endif // DDLA_HANDLE_T_H
