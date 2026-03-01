# Spectra: Unified Build and Automation Pipeline
# Zero external package dependencies. Standard POSIX makefile.

CC ?= clang
CXX ?= clang++
JULIA ?= julia
PYTHON ?= python3

CFLAGS = -O3 -fPIC -Wall -Wextra
CXXFLAGS = -O3 -std=c++20 -Wall -Wextra

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	SHLIB_EXT = dylib
	SHLIB_FLAGS = -dynamiclib
else
	SHLIB_EXT = so
	SHLIB_FLAGS = -shared
endif

all: build

build: build/libspectra_kernel.$(SHLIB_EXT) build/benchmark_simd

build/libspectra_kernel.$(SHLIB_EXT): src/c_kernel/spectra_kernel.c src/c_kernel/spectra_kernel.h
	@mkdir -p build
	$(CC) $(CFLAGS) $(SHLIB_FLAGS) -o $@ src/c_kernel/spectra_kernel.c
	@echo "[+] Compiled native SIMD kernel: $@"

build/benchmark_simd: cpp/benchmark_simd.cpp src/c_kernel/spectra_kernel.c
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -I src/c_kernel -o $@ cpp/benchmark_simd.cpp src/c_kernel/spectra_kernel.c
	@echo "[+] Compiled C++20 SIMD benchmark: $@"

test: build
	@echo "[*] Running Spectra comprehensive verification suite..."
	./bin/spectra test

bench: build
	@echo "[*] Running Spectra high-performance benchmark suite..."
	./bin/spectra bench

simd-bench: build/benchmark_simd
	@echo "[*] Running native C++20 SIMD microbenchmark..."
	./build/benchmark_simd

python-verify: build
	@echo "[*] Running Python FFI bridge verification..."
	$(PYTHON) python/verify_operators.py

demo-1:
	./bin/spectra demo 1

demo-2:
	./bin/spectra demo 2

demo-3:
	./bin/spectra demo 3

demo-4:
	./bin/spectra demo 4

demo-5:
	./bin/spectra demo 5

demo-all: demo-1 demo-2 demo-3 demo-4 demo-5

clean:
	rm -rf build/*.dylib build/*.so build/*.dSYM build/benchmark_simd

.PHONY: all build test bench simd-bench python-verify demo-1 demo-2 demo-3 demo-4 demo-5 demo-all clean
