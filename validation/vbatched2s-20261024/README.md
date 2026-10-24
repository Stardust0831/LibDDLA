Validation of DDLA two-stage GEMM dimension-check consolidation

Baseline: `10873422d01f8a8798ac5462673446494849b96a`.
Candidate: `dcd0881d806ef0e4df9f74367a0304f6d383c24d`.
The implementation change is confined to `src/BLAS/gemm_vbatched.cpp`; the other changed file extends its regression tests.

Both dimension sets are checked in one kernel launch and one host readback, with separate maxima and error flags. Both stages must validate before either multiplication starts. The requested transient summary allocation grows from 16 to 32 bytes. Matrix workspaces, GEMM kernels, segmentation, and accumulation order are unchanged.

Validation used V100 16 GB GPUs, CUDA 12.9.1, GNU 13.3, Open MPI 5.0.10, and Release builds targeting `sm_70`. Each HSE rank used one GPU and four physical CPU cores, with serial BLAS and explicit CPU affinity. Each case ran baseline/candidate, candidate/baseline, baseline/candidate in one allocation. Elapsed times include process startup. GPU memory was sampled every 0.5 s; the table reports the largest sampled per-process peak across the three repetitions for each variant. One MPI process used each GPU.

All HSE calculations used one ABACUS binary from `1aaeb8ff453dce854ef67e7f2e49d30700250194` and LibRI `a0661cbb4b4f89ceeb23e040d49accb00943093f`. That LibRI revision uses DDLA's old API, so [a separate test adapter](libri-public-api.patch) was applied identically for both variants. ABACUS was configured with explicit `DDLA_INCLUDE_DIR` and `DDLA_ROOT` values to find the new installation. Only the DDLA shared library changed between performance measurements; `ldd` output and process mappings were recorded. The adapter is outside the performance PR.

The structures are displaced Si64 and Si128 supercells, displaced CsPbI3 (20 atoms), and displaced Si8 with a 2×2×2 k-point mesh and complex EXX tensors. Inputs use HSE, `scf_thr=1e-9`, `scf_ene_thr=1e-8`, `exx_separate_loop=1`, `exx_symmetry_realspace=0`, and `nspin=1`. Si8 includes both forces and stress. The other force cases calculate forces. See the archived inputs and pseudopotential/orbital SHA256 records for complete settings.

The unprofiled timing and correctness suite contains 39 completed HSE runs: 36 GPU contraction runs and three CPU contraction references. All 27 paired numerical comparisons passed. CPU references retain the GPU eigensolver and switch `device_exx` to `cpu`. This ABACUS branch also selects its older RI initialization path for CPU contractions. The first CsPbI3 CPU reference exceeded the 30 GiB host-memory limit; its retry reserved 60 GiB through a four-GPU allocation, while still using two MPI ranks, two GPUs, and four threads per rank. CPU reference elapsed times are excluded from the DDLA performance ratios.

| Comparison | Max energy difference (eV/atom) | Max force difference (eV/Å) | Max stress difference (kbar) | Same SCF counts |
|---|---:|---:|---:|---|
| Candidate vs baseline GPU | 1.819e-12 | 1.000e-10 | 1.000e-10 | True |
| Candidate GPU vs CPU contractions | 7.276e-13 | 2.800e-09 | 2.980e-08 | True |

Differences are computed from the logged energies and force/stress components. GPU comparisons use tolerances of `1e-8 eV/atom`, `1e-6 eV/Å`, and `1e-4 kbar`; CPU references use `1e-6`, `1e-4`, and `1e-3` in the corresponding units.

| HSE case | GPUs | Baseline median (s) | Candidate median (s) | Wall speedup |
|---|---:|---:|---:|---:|
| si64 single_point | 1 | 131.318 | 130.914 | 1.003× |
| si64 forces | 1 | 175.581 | 178.999 | 0.981× |
| cspbi3_20 single_point | 2 | 140.321 | 140.323 | 1.000× |
| cspbi3_20 forces | 2 | 181.945 | 181.600 | 1.002× |
| si8 forces | 2 | 69.320 | 71.680 | 0.967× |
| si128 forces | 2 | 314.125 | 313.117 | 1.003× |

Wall times are medians of three runs per variant. Speedup is baseline divided by candidate; values below one indicate a measured slowdown. The changes are small and coexist with run-to-run variation. These results do not establish a general HSE speedup. All samples are retained in [hse-measurements.json](hse-measurements.json).

| Case | EXX energy median, base → candidate (s) | EXX force median, base → candidate (s) | GPU peak, base / candidate (MiB) |
|---|---:|---:|---:|
| si64 single_point | 25.48 → 24.85 | — | 6558 / 6558 |
| si64 forces | 27.45 → 27.68 | 33.47 → 33.80 | 6578 / 6578 |
| cspbi3_20 single_point | 13.31 → 13.01 | — | 2766 / 2766 |
| cspbi3_20 forces | 13.60 → 14.47 | 18.09 → 18.09 | 2786 / 2786 |
| si8 forces | 7.99 → 8.97 | 9.43 → 8.72 | 1970 / 1970 |
| si128 forces | 52.56 → 52.81 | 61.84 → 62.57 | 7346 / 7348 |

The isolated two-stage variable-batch microbenchmark improved by 1.27–1.61× (21–38% less elapsed time per call) over 1, 64, and 512 variable-size matrices, for double and complex double. It uses three paired repetitions, 20 warmups and 1,000 measured calls per sample. It checks every output against its analytical value. These are complete DDLA calls, including validation and computation.

Nsight Systems confirms that 720 two-stage calls reduce validation launches from 1,440 to 720 and remove 720 allocation/free, copy, and explicit synchronization calls. The 1,440 multiplication kernel launches are unchanged.

The separate Si8 HSE trace shows only 114 two-stage calls per rank over the entire calculation. Per rank, validation launches fall from 342 to 228; allocation/free and explicit synchronization calls each fall by 114. The observed microbenchmark saving corresponds to milliseconds per rank at this call count, explaining the small wall-time effect. Profiled elapsed times are excluded from the timing table. See [hse-profile.json](hse-profile.json) and the archived CSV reports.

The final GEMM regression passes against both libraries on one and two GPU ranks: 64 reported cases per rank, all four scalar types, transpose/conjugate modes, segmented temporary reuse, independent stage bounds spanning multiple tiles, and invalid/null dimension arrays rejected without matrix writes. Maximum single/double precision errors are `1.23e-7` / `2.29e-16`.

LibRI's 18 contractions and their sum pass with signed and complex accumulation factors, all four scalar types, and one/four OpenMP threads against both DDLA libraries. The fork's CPU build/CTest and CUDA/HIP compilation checks pass for the final candidate. HIP hardware performance was not measured. [ci.json](ci.json) records the exact checks.

[validation-results.tar.gz](validation-results.tar.gz) contains commands, build/test logs, all completed calculation inputs and outputs, CPU bindings, loaded-library records, memory samples, and profiling summaries. [tested-file-sha256.json](tested-file-sha256.json) records binary and source identities. The initial sequential job was stopped after its six completed Si64 single-point runs to split the remaining cases into independent jobs; its unfinished force run is excluded. Startup cancellations and the initial CUDA library-path configuration attempt are preserved in the manifest and logs.

GPU validation ran on [SAI](https://www.open-sai.com/).
