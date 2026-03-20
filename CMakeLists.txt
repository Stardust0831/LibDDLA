cmake_minimum_required(VERSION 3.13)

project(LibDDLA NONE)

file(READ ${PROJECT_SOURCE_DIR}/src/version.h version_file)
string(REGEX MATCH "LIBDDLA_MAJOR_VERSION ([0-9]+)" libddla_major_version ${version_file})
set(libddla_major_version ${CMAKE_MATCH_1})
string(REGEX MATCH "LIBDDLA_MINOR_VERSION ([0-9]+)" libddla_minor_version ${version_file})
set(libddla_minor_version ${CMAKE_MATCH_1})
string(REGEX MATCH "LIBDDLA_MICRO_VERSION ([0-9]+)" libddla_micro_version ${version_file})
set(libddla_micro_version ${CMAKE_MATCH_1})
set(version "${libddla_major_version}.${libddla_minor_version}.${libddla_micro_version}")

message(STATUS "LibDDLA version: ${version}")

set(CMAKE_PROJECT_SOVERSION_MAJOR "${libddla_major_version}")
set(CMAKE_PROJECT_VERSION "${version}")
set(LIB_NAME            "ddla")
set(math_libs "")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

option(BUILD_TESTS "Build test programs" OFF) 
option(ENABLE_CUDA "Build with CUDA" OFF)
option(ENABLE_HIP "Build with HIP" OFF)
option(ENABLE_DEBUG "DEBUG for program" OFF)
if(ENABLE_DEBUG)
add_compile_definitions("ENABLE_DEBUG")
endif()
if(ENABLE_CUDA AND ENABLE_HIP)
    message(FATAL_ERROR "Cannot enable both CUDA and HIP")
endif()
if ((NOT ENABLE_CUDA) AND (NOT ENABLE_HIP))
    message(FATAL_ERROR "Must enable either CUDA or HIP")
endif()
if(BUILD_TESTS)
    # enable_testing()
    add_subdirectory(tests)
endif()


enable_language(CXX)

if(ENABLE_HIP)
    enable_language(HIP)
    # if(NOT EXISTS ${ROCM_PATH})
    #     message(FATAL_ERROR "ROCm not found at ${ROCM_PATH}")
    # endif()
    # list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")
    # find_package(hip REQUIRED)
    # find_package(amd_comgr REQUIRED)
    # enable_language(HIP)
    set(CMAKE_HIP_STANDARD 11)
    set(CMAKE_HIP_STANDARD_REQUIRED True)
    add_compile_definitions("ENABLE_HIP")
endif()
# cuda setting
# find_package(CUDA REQUIRED)
# find_package(CUDAToolkit REQUIRED)
if(ENABLE_CUDA)
    add_compile_definitions("ENABLE_CUDA")
    set(CUDA_LIBRARIES "-lcurand -lcublas -lcudart -lcusolver -lnccl -std=c++11")
    list(APPEND math_libs
        ${CUDA_LIBRARIES}
    )
endif(ENABLE_CUDA)
# hip setting
if(ENABLE_HIP)
    set(HIP_LIBRARIES "-lhipblas -lrccl -lhipsolver -lamdhip64 -lhiprand -lgalaxyhip -lmpi")
    # -lamdhip64 -lhsa-runtime64
    find_package(HIP REQUIRED)
    list(APPEND math_libs
        ${HIP_LIBRARIES}
    )
endif(ENABLE_HIP)
include_directories(src)
include_directories(include)
add_subdirectory(src)
