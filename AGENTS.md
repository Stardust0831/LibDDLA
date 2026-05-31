# AGENTS.md

## Build

Requires **both CUDA/HIP and MPI**. Choose exactly one backend:

```bash
mkdir build && cd build

# CUDA (with NCCL):
cmake .. \
  -DENABLE_CUDA=ON \
  -DENABLE_CCL=ON \
  -DBUILD_TESTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES="80"

# HIP/ROCm (with RCCL):
cmake .. \
  -DENABLE_HIP=ON \
  -DENABLE_CCL=ON \
  -DBUILD_TESTS=ON \
  -DCMAKE_HIP_ARCHITECTURES="gfx90a"

make -j
```

Key CMake options:
- `ENABLE_CUDA=ON` or `ENABLE_HIP=ON` — **required**, exactly one
- `ENABLE_CCL=ON` — use NCCL/RCCL for inter-GPU comm; OFF falls back to MPI
- `ENABLE_GPU_CPU_TUNNEL=ON` — route data through host for MPI-based comm
- `ENABLE_DEBUG=ON` — debug builds
- `BUILD_TESTS=ON` — build test executables (tests are not built by default)

If `ENABLE_CCL` and `ENABLE_GPU_CPU_TUNNEL` are both ON, CCL is linked but unused — pick one communication path.

## Running tests

Test binaries live in `tests/`. They link against the shared library `libddla.so` and must be run under `mpirun`:

```bash
# Build and install lib first
cmake .. -DENABLE_CUDA=ON -DENABLE_CCL=ON -DBUILD_TESTS=ON
make -j
make install DESTDIR=./install

# Run
cd tests
mpirun -np 4 ./test_pzgemm
```

Slurm scripts are in `tests/test_cuda.sh` and `tests/test_hip.sh` — these show the full env setup (module loads, NVHPC setup, InfiniBand config).

## Architecture

**Namespace**: `ddla` (lowercase). Old code used `DDLA` (uppercase); check includes.

**Headers**: `include/ddla/` — one header per concern. The main API is `ddla.h`.

**Source layout**:

| File | Purpose |
|------|---------|
| `pgetrf.cpp` | Distributed LU factorization (panel-by-panel) |
| `pgetrs.cpp` | LU solve (pivot + forward + backward triangular) |
| `pgesv.cpp` | LU driver = pgetrf + pgetrs |
| `ptrtrs.cpp` | Distributed triangular solve |
| `pgemm.cpp` | Matrix multiply: `C = alpha*op(A)*op(B) + beta*C` |
| `pgeadd.cpp` | Matrix addition: `C = alpha*op(A) + beta*op(B)` |
| `plapiv.cpp` | Apply pivot permutation |
| `pswap.cpp` | Swap rows/columns between distributed matrices |
| `ppotrf.cpp` | Cholesky factorization (external GPU solver + batched gemm) |
| `ppotrs.cpp` | Cholesky solve (two triangular solves) |
| `pposv.cpp` | Cholesky driver = ppotrf + ppotrs |
| `pgetf2.cpp` | Inner unblocked panel LU for pgetrf |
| `pgetf2_panel.cpp` | Alternative panel LU (rank-revealing variant) |
| `transport_block.cpp` | Extract a sub-block from a distributed matrix with optional transpose |
| `ddla_stream.cpp` | DdlaStream: device set, NCCL comms, BLAS/solver handles |
| `ddla_handle_t.cpp` | Handle init / set / destroy |
| `ddla_desc.cpp` | DdlaDesc: local sizes, index mapping (g2l, l2g) |

**All functions are templates** instantiated for `float`, `double`, `std::complex<float>`, `std::complex<double>`. The instantiation block is at the bottom of each `.cpp` file (`template void pgemm<float>(...);` etc.).

**ppotrf / ppotrs / pposv are only instantiated for complex types** (not float/double), unlike the others which have all four.

## Communication model

Three paths, selected at compile time:

1. **`ENABLE_CCL=ON`** — NCCL/RCCL `ncclSend`/`ncclBcast`/`ncclAllReduce` directly on device buffers
2. **`ENABLE_GPU_CPU_TUNNEL=ON`** — `deviceMemcpy D2H → MPI_Bcast → H2D` via host staging buffers
3. **Neither** — `deviceStreamSynchronize` then MPI on device pointers (unsafe in general but works on unified-memory or with CUDA-aware MPI)

`ddla_comm.h` overloads `cclSend`, `cclRecv`, `cclBcast`, `cclAllReduce`, `cclBroadcast` for each path. Implementation files `#include <ddla/ddla_comm.h>` and use the `CCL_CHECK()` macro.

## Data distribution

Matrices are **2D block-cyclic** over a `nprows × npcols` process grid, same as ScaLAPACK. The `DdlaDesc` class (`ddla_desc.h`) holds global dims (`m`, `n`), block sizes (`mb`, `nb`), process-grid info (`myprow`, `mypcol`, `nprows`, `npcols`), source rows/cols (`irsrc`, `icsrc`), and local dimensions (`m_loc`, `n_loc`, `lld`).

Free functions for index mapping (also in `ddla_desc.h`):
- `indxg2p(global_idx, nb, srcproc, nprocs)` → process rank
- `indxg2l(global_idx, nb, nprocs)` → local index
- `indxl2g(local_idx, nb, myproc, srcproc, nprocs)` → global index
- `num_loc(n, nb, myproc, srcproc, nprocs)` → local count

## Coding conventions

- `BLAS_CHECK()`, `DEVICE_CHECK()`, `SOLVER_CHECK()`, `CCL_CHECK()`, `MPI_CHECK()` macros wrap every call — always use them
- BLAS wrappers are type-overloaded in individual headers (`gemm.h`, `trsm.h`, etc.); use `deblasGemm`, `deblasTrsm`, `deblasScal`, etc.
- Include path: `#include <ddla/ddla.h>` (the top-level `include/` dir is added via `include_directories`)
- Functions use `const &` for scalars and array descriptors
- No multi-file functions — each function lives in exactly one `.cpp`
- Avoid adding comments to implementation code unless explaining a non-obvious algorithm step
- Doxygen-style `@brief`/`@param`/`@tparam` comments go in the header declaration, not the `.cpp`

## Version

LibDDLA uses semantic versioning. Version numbers are read from `src/version.h` (three `#define` macros: `LIBDDLA_MAJOR_VERSION`, `LIBDDLA_MINOR_VERSION`, `LIBDDLA_MICRO_VERSION`). Bump there, not in CMakeLists.
