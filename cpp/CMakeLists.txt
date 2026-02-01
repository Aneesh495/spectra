cmake_minimum_required(VERSION 3.16)
project(SpectraSIMDBenchmark LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
    add_compile_options(-O3 -march=native -Wall -Wextra)
endif()

add_executable(benchmark_simd
    benchmark_simd.cpp
    ../src/c_kernel/spectra_kernel.c
)

target_include_directories(benchmark_simd PRIVATE ../src/c_kernel)
