add_library(ddla_lib SHARED)


set_target_properties(ddla_lib PROPERTIES
  OUTPUT_NAME ${LIB_NAME}
  VERSION ${CMAKE_PROJECT_VERSION}
  SOVERSION ${CMAKE_PROJECT_SOVERSION_MAJOR}
)
message(STATUS "Building ddla library version ${CMAKE_PROJECT_VERSION}")



if(DDLA_USE_CPU AND NOT DDLA_DUAL_BUILD)
    # CPU-only build: every backend-templated routine; the sources that only
    # carry GPU kernels (pgetf2_kernels.cpp, pgeqrf_kernels.cpp, pgebrd_kernels.cpp)
    # are left out.  transport_block.cpp is here because the CPU pgemm and the
    # triangular solves call the unified transport_block<DdlaBackend::CPU,T>.
    # The files that keep a GPU kernel next to their CPU arm (getrf_nopiv, pdam,
    # pgeadd, potrf_bottom_right_transform) compile as plain C++ here: all kernel
    # code is behind DDLA_USE_CUDA / DDLA_USE_HIP.
    list(APPEND lib_sources
        gemm.cpp
        herk.cpp
        her2k.cpp
        hemv.cpp
        hemm.cpp
        her2.cpp
        gemv.cpp
        trmv.cpp
        nrm2.cpp
        dot.cpp
        trsm.cpp
        syrk.cpp
        trmm.cpp
        gemmBatched.cpp
        scal.cpp
        geam.cpp
        swap.cpp
        getrf.cpp
        potrf.cpp
        laswp.cpp
        axpy.cpp
        iamax.cpp
        geru.cpp
        write_matrix.cpp
        random_generate.cpp
        pgemm.cpp
        pgemm_select.cpp
        pgemm_stationary.cpp
        transport_block.cpp
        ptrtrs.cpp
        pgetf2.cpp
        pgetf2_panel.cpp
        pgetrf.cpp
        pgetrf_bpiv.cpp
        pgetrf_nopiv.cpp
        getrf_nopiv.cpp
        plapiv.cpp
        pswap.cpp
        pgetrs.cpp
        pgetrs_nopiv.cpp
        pgetrs_bpiv.cpp
        pgesv.cpp
        pgesv_nopiv.cpp
        pgesv_bpiv.cpp
        pgeadd.cpp
        pdam.cpp
        ppotrf.cpp
        potrf_bottom_right.cpp
        potrf_bottom_right_transform.cpp
        ppotrf_bottom_right.cpp
        ppotrs.cpp
        pposv.cpp
        ddla_handle_t.cpp
        ddla_desc.cpp
    )
else()
    # GPU-only or dual build: full accelerator source set.
    list(APPEND lib_sources
        gemm.cpp
        herk.cpp
        her2k.cpp
        hemv.cpp
        hemm.cpp
        her2.cpp
        gemv.cpp
        trmv.cpp
        nrm2.cpp
        dot.cpp
        trsm.cpp
        syrk.cpp
        trmm.cpp
        gemmBatched.cpp
        scal.cpp
        geam.cpp
        swap.cpp
        getrf.cpp
        potrf.cpp
        laswp.cpp
        axpy.cpp
        iamax.cpp
        geru.cpp
        write_matrix.cpp
        random_generate.cpp
        ptrtrs.cpp
        pgetf2.cpp
        pgetf2_kernels.cpp
        pgetf2_panel.cpp
        pgetrf.cpp
        pgetrf_bpiv.cpp
        pgetrf_nopiv.cpp
        getrf_nopiv.cpp
        plapiv.cpp
        pswap.cpp
        pgetrs.cpp
        pgetrs_nopiv.cpp
        pgetrs_bpiv.cpp
        pgesv.cpp
        pgesv_nopiv.cpp
        pgesv_bpiv.cpp
        pgemm.cpp
        pgemm_select.cpp
        pgemm_stationary.cpp
        pgeadd.cpp
        pdam.cpp
        ppotrf.cpp
        potrf_bottom_right.cpp
        potrf_bottom_right_transform.cpp
        ppotrf_bottom_right.cpp
        ppotrs.cpp
        pposv.cpp
        transport_block.cpp
        ddla_handle_t.cpp
        ddla_desc.cpp
    )
endif()

if(DDLA_USE_CUDA)
    list(APPEND lib_sources
        gemm_vbatched.cpp
        vbatched/sgemm_vbatched_core.cpp
        vbatched/dgemm_vbatched_core.cpp
        vbatched/cgemm_vbatched_core.cpp
        vbatched/zgemm_vbatched_core.cpp
    )
    set_source_files_properties(
        getrf_nopiv.cpp
        gemm_vbatched.cpp
        pdam.cpp
        pgetf2_kernels.cpp
        potrf_bottom_right_transform.cpp
        pgeadd.cpp
        vbatched/sgemm_vbatched_core.cpp
        vbatched/dgemm_vbatched_core.cpp
        vbatched/cgemm_vbatched_core.cpp
        vbatched/zgemm_vbatched_core.cpp
        PROPERTIES
        LANGUAGE CUDA
    )
endif()

if(DDLA_USE_HIP)
    list(APPEND lib_sources
        gemm_vbatched.cpp
        vbatched/sgemm_vbatched_core.cpp
        vbatched/dgemm_vbatched_core.cpp
        vbatched/cgemm_vbatched_core.cpp
        vbatched/zgemm_vbatched_core.cpp
    )
    set_source_files_properties(
        getrf_nopiv.cpp
        gemm_vbatched.cpp
        pdam.cpp
        pgetf2_kernels.cpp
        potrf_bottom_right_transform.cpp
        pgeadd.cpp
        vbatched/sgemm_vbatched_core.cpp
        vbatched/dgemm_vbatched_core.cpp
        vbatched/cgemm_vbatched_core.cpp
        vbatched/zgemm_vbatched_core.cpp
        PROPERTIES
        LANGUAGE HIP
    )
endif()

target_sources(ddla_lib PRIVATE ${lib_sources})

# ---------------------------------------------------------------------------
# Per-target compile definitions (moved from directory-global)
# ---------------------------------------------------------------------------
if(DDLA_USE_CUDA)
    target_compile_definitions(ddla_lib PRIVATE DDLA_USE_CUDA)
endif()
if(DDLA_USE_HIP)
    target_compile_definitions(ddla_lib PRIVATE DDLA_USE_HIP __HIP_PLATFORM_AMD__)
endif()
if(DDLA_USE_CPU AND NOT DDLA_DUAL_BUILD)
    target_compile_definitions(ddla_lib PRIVATE DDLA_USE_CPU)
endif()
if(DDLA_USE_CCL)
    target_compile_definitions(ddla_lib PRIVATE DDLA_USE_CCL)
endif()
if(DDLA_USE_GPU_CPU_TUNNEL)
    target_compile_definitions(ddla_lib PRIVATE DDLA_USE_GPU_CPU_TUNNEL)
endif()
if(DDLA_USE_DEBUG)
    target_compile_definitions(ddla_lib PRIVATE DDLA_USE_DEBUG)
endif()

if(DDLA_USE_HIP)
    # CXX-language TUs (gemm.cpp, scal.cpp, the tests, ...) include vendor HIP
    # headers through ddla_connector.h. Only the LANGUAGE HIP sources get ROCm
    # include paths from hipcc itself, so point every other compile at
    # ${ROCM_PATH}/include explicitly (the df_dcu module env does this via
    # CPATH; the GitHub rocm container does not). PUBLIC: test targets also
    # include ddla_connector.h from their own sources.
    target_include_directories(ddla_lib PUBLIC
        "${ROCM_PATH}/include"
    )
    target_link_directories(ddla_lib PUBLIC
        "${ROCM_PATH}/lib"
        "${ROCM_PATH}/lib64"
        "${ROCM_PATH}/dcc/lib"
    )
    set_target_properties(ddla_lib PROPERTIES
        BUILD_RPATH "${ROCM_PATH}/lib;${ROCM_PATH}/lib64;${ROCM_PATH}/dcc/lib"
        INSTALL_RPATH "${ROCM_PATH}/lib;${ROCM_PATH}/lib64;${ROCM_PATH}/dcc/lib"
    )
endif()
target_link_libraries(ddla_lib PUBLIC ${math_libs})

# ---------------------------------------------------------------------------
# geam.cpp, gemmBatched.cpp: the CPU arms of ddla::geam and ddla::gemmBatched run on OpenMP threads.  Those translation
# units are compiled by the plain host compiler in every build (never by nvcc/hipcc) with -ffp-contract=off; with OpenMP found they get -fopenmp
# and libgomp is linked PRIVATE (the shared library records the NEEDED entry); without OpenMP it runs serially
# with identical results.
# ---------------------------------------------------------------------------
option(DDLA_USE_OPENMP "OpenMP threads for the host kernels (geam, and the CPU eigensolver kernels)" ON)
set(_ddla_cpu_kernel_opts "-ffp-contract=off")
if(DDLA_USE_OPENMP)
    find_package(OpenMP COMPONENTS CXX)
    if(OpenMP_CXX_FOUND)
        list(APPEND _ddla_cpu_kernel_opts ${OpenMP_CXX_FLAGS})
        target_link_libraries(ddla_lib PRIVATE ${OpenMP_CXX_LIBRARIES})
        message(STATUS "ddla: OpenMP host kernels enabled (${OpenMP_CXX_FLAGS})")
    else()
        message(WARNING "ddla: OpenMP not found; the host kernels run serially")
    endif()
endif()
set_source_files_properties(geam.cpp gemmBatched.cpp PROPERTIES COMPILE_OPTIONS "${_ddla_cpu_kernel_opts}")

install(TARGETS ddla_lib
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    PATTERN "*.h.in" EXCLUDE
)
