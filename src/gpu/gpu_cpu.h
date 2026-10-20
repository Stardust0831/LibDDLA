#ifndef DDLA_GPU_CPU_H
#define DDLA_GPU_CPU_H

// CPU-only stand-ins for the gpu* names (used when neither DDLA_USE_CUDA nor DDLA_USE_HIP is defined).
// They let backend-templated code name the handle, status and enum types in a CPU build; the vendor
// functions themselves (gpublas<P>gemm, gpuMalloc, ...) do not exist here and every call to them sits
// behind `#if DDLA_GPU_VENDOR` or an `if constexpr` on the backend.

namespace ddla {

// A CPU handle has no stream or event; both stay null.
using gpuStream_t = void*;
using gpuEvent_t = void*;

using gpublasStatus_t = int;
constexpr auto GPUBLAS_STATUS_SUCCESS = 0;
using gpublasHandle_t = void*;
using gpusolverHandle_t = void*;
using gpusolverStatus_t = int;
constexpr auto GPUSOLVER_STATUS_SUCCESS = 0;
using gpurandGenerator_t = void*;
using gpurandStatus_t = int;
constexpr auto GPURAND_STATUS_SUCCESS = 0;
using gpurandRngType = int;
constexpr auto GPURAND_RNG_PSEUDO_DEFAULT = 0;
using gpublasSideMode_t = int;
constexpr auto GPUBLAS_SIDE_LEFT = 0;
constexpr auto GPUBLAS_SIDE_RIGHT = 1;
using gpublasFillMode_t = int;
constexpr auto GPUBLAS_FILL_MODE_LOWER = 0;
constexpr auto GPUBLAS_FILL_MODE_UPPER = 1;
using gpublasDiagType_t = int;
constexpr auto GPUBLAS_DIAG_UNIT = 0;
constexpr auto GPUBLAS_DIAG_NON_UNIT = 1;
using gpublasOperation_t = char;
constexpr auto GPUBLAS_OP_N = 'N';
constexpr auto GPUBLAS_OP_T = 'T';
constexpr auto GPUBLAS_OP_C = 'C';

} // namespace ddla

#define gpusolverGetStream(solverH, stream) ((void)0)
#define gpurandCreateGenerator(gen, rng) ((void)0)
#define gpurandSetPseudoRandomGeneratorSeed(gen, seed) ((void)0)
#define gpurandGenerateUniform(gen, data, n) ((void)0)
#define gpurandGenerateUniformDouble(gen, data, n) ((void)0)
#define gpurandDestroyGenerator(gen) ((void)0)
#define gpurandSetStream(gen, stream) ((void)0)

#define GPU_HAS_SOLVER_LASWP 0

#endif // DDLA_GPU_CPU_H
