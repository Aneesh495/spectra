/*
 * Spectra C++20 SIMD Acceleration Benchmark
 *
 * Measures cache-blocked GEMM throughput, multi-mode tensor contraction GFLOPS,
 * and Clenshaw Chebyshev polynomial recurrence latency.
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <numeric>
#include <cmath>
#include <iomanip>

extern "C" {
    #include "../src/c_kernel/spectra_kernel.h"
}

using namespace std::chrono;

void benchmark_gemm(int64_t M, int64_t N, int64_t K, int iterations) {
    std::vector<double> A(M * K);
    std::vector<double> B(K * N);
    std::vector<double> C(M * N, 0.0);

    std::mt19937_64 rng(42);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (auto &x : A) x = dist(rng);
    for (auto &x : B) x = dist(rng);

    // Warm-up
    spectra_simd_gemm(M, N, K, 1.0, A.data(), M, B.data(), K, 0.0, C.data(), M);

    auto start = high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        spectra_simd_gemm(M, N, K, 1.0, A.data(), M, B.data(), K, 0.0, C.data(), M);
    }
    auto end = high_resolution_clock::now();

    double total_sec = duration<double>(end - start).count();
    double avg_sec = total_sec / iterations;
    double gflops = (2.0 * M * N * K * 1e-9) / avg_sec;

    std::cout << "  [GEMM " << M << "x" << K << "x" << N << "] "
              << std::fixed << std::setprecision(3)
              << avg_sec * 1e3 << " ms/iter | "
              << gflops << " GFLOPS" << std::endl;
}

void benchmark_tensor_contraction(int64_t modes, int64_t in_c, int64_t out_c, int iterations) {
    std::vector<double> weights(out_c * in_c * modes);
    std::vector<double> X_real(modes * in_c);
    std::vector<double> X_imag(modes * in_c);
    std::vector<double> Y_real(modes * out_c, 0.0);
    std::vector<double> Y_imag(modes * out_c, 0.0);

    std::mt19937_64 rng(123);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (auto &x : weights) x = dist(rng);
    for (auto &x : X_real) x = dist(rng);
    for (auto &x : X_imag) x = dist(rng);

    // Warm-up
    spectra_simd_tensor_contract(modes, in_c, out_c, weights.data(),
                                 X_real.data(), X_imag.data(),
                                 Y_real.data(), Y_imag.data());

    auto start = high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        spectra_simd_tensor_contract(modes, in_c, out_c, weights.data(),
                                     X_real.data(), X_imag.data(),
                                     Y_real.data(), Y_imag.data());
    }
    auto end = high_resolution_clock::now();

    double total_sec = duration<double>(end - start).count();
    double avg_sec = total_sec / iterations;
    double ops = 4.0 * modes * in_c * out_c; // Multiply-add for real and imag
    double gflops = (ops * 1e-9) / avg_sec;

    std::cout << "  [TensorContract modes=" << modes << " in=" << in_c << " out=" << out_c << "] "
              << std::fixed << std::setprecision(3)
              << avg_sec * 1e6 << " us/iter | "
              << gflops << " GFLOPS" << std::endl;
}

void benchmark_clenshaw(int64_t n_points, int64_t degree, int iterations) {
    std::vector<double> coeffs(degree);
    std::vector<double> x_points(n_points);
    std::vector<double> out(n_points, 0.0);

    for (int64_t i = 0; i < degree; ++i) coeffs[i] = 1.0 / (i + 1);
    for (int64_t i = 0; i < n_points; ++i) x_points[i] = -1.0 + 2.0 * i / (n_points - 1);

    // Warm-up
    spectra_chebyshev_eval(n_points, degree, x_points.data(), coeffs.data(), out.data());

    auto start = high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        spectra_chebyshev_eval(n_points, degree, x_points.data(), coeffs.data(), out.data());
    }
    auto end = high_resolution_clock::now();

    double total_sec = duration<double>(end - start).count();
    double avg_sec = total_sec / iterations;
    double ns_per_point = (avg_sec * 1e9) / n_points;

    std::cout << "  [Clenshaw N=" << n_points << " deg=" << degree << "] "
              << std::fixed << std::setprecision(3)
              << avg_sec * 1e6 << " us/iter | "
              << ns_per_point << " ns/point" << std::endl;
}

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << "Spectra Native C/C++ SIMD Acceleration Performance Suite" << std::endl;
    std::cout << "==========================================================" << std::endl;

    std::cout << "[*] Running Cache-Blocked SIMD GEMM Benchmark..." << std::endl;
    benchmark_gemm(128, 128, 128, 200);
    benchmark_gemm(256, 256, 256, 100);
    benchmark_gemm(512, 512, 512, 20);

    std::cout << "\n[*] Running Neural Operator Tensor Contraction Benchmark..." << std::endl;
    benchmark_tensor_contraction(32, 64, 64, 2000);
    benchmark_tensor_contraction(64, 128, 128, 1000);

    std::cout << "\n[*] Running Clenshaw-Smith Chebyshev Recurrence Benchmark..." << std::endl;
    benchmark_clenshaw(1024, 32, 1000);
    benchmark_clenshaw(4096, 64, 500);

    std::cout << "\n[+] Benchmark suite completed successfully." << std::endl;
    return 0;
}
