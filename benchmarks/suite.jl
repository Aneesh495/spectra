"""
    Spectra Performance Benchmark Suite

Measures runtime latency, memory allocation, and computational throughput across:
1. Tensor Contraction & Native C SIMD GEMM
2. 1D/2D Fast Fourier & Chebyshev Spectral Transforms
3. Symplectic Integrators vs Adaptive Explicit Runge-Kutta
4. Cellular Sheaf Laplacian Construction & Diffusion
5. Fourier Neural Operator (FNO-1D & FNO-2D) Inference Throughput
6. Dynamic Mode Decomposition (Exact DMD & Hankel DMD)
"""

using Spectra
using LinearAlgebra
using Printf

function run_benchmark(name::String, f::Function; iterations::Int=50, warmup::Int=5)
    for _ in 1:warmup
        f()
    end
    GC.gc()
    t_start = time_ns()
    for _ in 1:iterations
        f()
    end
    t_total = (time_ns() - t_start) * 1e-9
    avg_ms = (t_total / iterations) * 1e3
    return avg_ms
end

println("==========================================================================")
println("                   Spectra High-Performance Benchmarks                   ")
println("==========================================================================")
println(" Platform: ", Sys.isapple() ? "macOS" : "Linux", " (", Sys.ARCH, ") | CPU Threads: ", Threads.nthreads())
println(" Native C SIMD Kernel: ", is_c_kernel_available() ? "ENABLED" : "FALLBACK")
println("--------------------------------------------------------------------------")

# 1. Tensor Contraction & GEMM
let
    A = randn(256, 256)
    B = randn(256, 256)
    C = zeros(256, 256)
    ms_gemm = run_benchmark("Blocked SIMD GEMM (256x256)", () -> c_simd_gemm!(C, A, B); iterations=100)
    gflops = (2.0 * 256^3 * 1e-9) / (ms_gemm * 1e-3)
    @printf("  %-42s : %8.3f ms | %7.2f GFLOPS\n", "SIMD GEMM (256x256)", ms_gemm, gflops)
end

# 2. Spectral Transforms
let
    x_1d = randn(ComplexF64, 1024)
    ms_fft1d = run_benchmark("1D FFT (N=1024)", () -> fft_1d(x_1d); iterations=200)
    @printf("  %-42s : %8.3f ms\n", "1D FFT (N=1024)", ms_fft1d)

    x_2d = randn(ComplexF64, 64, 64)
    ms_fft2d = run_benchmark("2D FFT (64x64)", () -> fft_2d(x_2d); iterations=100)
    @printf("  %-42s : %8.3f ms\n", "2D FFT (64x64)", ms_fft2d)

    v_cheb = randn(256)
    ms_dct = run_benchmark("1D Fast DCT-II (N=256)", () -> dct_1d(v_cheb); iterations=200)
    @printf("  %-42s : %8.3f ms\n", "1D Fast DCT-II (N=256)", ms_dct)
end

# 3. Symplectic Integrators vs DP54
let
    H = (q, p) -> 0.5 * (p[1]^2 + q[1]^2)
    grad_V = q -> q
    inv_M = [1.0]

    ms_verlet = run_benchmark("Verlet Symplectic Step", () -> verlet_step(grad_V, inv_M, [1.0], [0.0], 0.01); iterations=2000)
    @printf("  %-42s : %8.3f us\n", "Verlet Symplectic Step", ms_verlet * 1e3)

    ms_forest = run_benchmark("Forest-Ruth 4th-Order Symplectic", () -> forest_ruth_step(grad_V, inv_M, [1.0], [0.0], 0.01); iterations=1000)
    @printf("  %-42s : %8.3f us\n", "Forest-Ruth 4th-Order Step", ms_forest * 1e3)

    ode_f = (x, p, t) -> -x
    ms_dp54 = run_benchmark("Adaptive DP54 Integration (t=0..1)", () -> integrate_ode(ode_f, [1.0], (0.0, 1.0); dt_init=0.05); iterations=100)
    @printf("  %-42s : %8.3f ms\n", "Adaptive DP54 Integration (t=0..1)", ms_dp54)
end

# 4. Sheaf Laplacian Diffusion
let
    complex = CellComplex(12)
    for i in 1:11
        add_edge!(complex, i, i + 1)
    end
    add_edge!(complex, 12, 1)
    sheaf = CellularSheaf(complex, 2)
    I2 = Matrix{Float64}(I, 2, 2)
    for (idx, (u, v)) in enumerate(complex.edges)
        set_restriction!(sheaf, u, idx, I2)
        set_restriction!(sheaf, v, idx, I2)
    end

    ms_lap = run_benchmark("Sheaf Laplacian Construction", () -> sheaf_laplacian(sheaf); iterations=200)
    @printf("  %-42s : %8.3f ms\n", "Sheaf Laplacian Construction", ms_lap)

    x0 = randn(24)
    ms_diff = run_benchmark("Sheaf Heat Diffusion Step", () -> sheaf_diffusion_step(sheaf, x0, 0.01); iterations=500)
    @printf("  %-42s : %8.3f ms\n", "Sheaf Heat Diffusion Step", ms_diff)
end

# 5. Neural Operators (FNO-1D & FNO-2D)
let
    fno1d = FourierNeuralOperator1D(2, 1; hidden_c=16, modes=8, n_layers=4)
    x1d = randn(64, 2)
    ms_fno1d = run_benchmark("FNO-1D Forward Pass (N=64, L=4)", () -> fno1d(x1d); iterations=100)
    @printf("  %-42s : %8.3f ms\n", "FNO-1D Forward Pass (N=64, L=4)", ms_fno1d)

    fno2d = FourierNeuralOperator2D(1, 1; hidden_c=16, modes1=6, modes2=6, n_layers=2)
    x2d = randn(24, 24, 1)
    ms_fno2d = run_benchmark("FNO-2D Forward Pass (24x24, L=2)", () -> fno2d(x2d); iterations=50)
    @printf("  %-42s : %8.3f ms\n", "FNO-2D Forward Pass (24x24, L=2)", ms_fno2d)
end

# 6. Koopman Exact DMD
let
    X = randn(20, 50)
    Y = randn(20, 50)
    ms_dmd = run_benchmark("Exact DMD Fit (20x50 snapshots)", () -> exact_dmd(X, Y, 0.01; rank=10); iterations=100)
    @printf("  %-42s : %8.3f ms\n", "Exact DMD Fit (20x50 snapshots)", ms_dmd)
end

println("==========================================================================")
println("                      Benchmark Run Completed                             ")
println("==========================================================================")
