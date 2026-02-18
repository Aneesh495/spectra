using Test
using Spectra
using LinearAlgebra

@testset "Native C SIMD Acceleration & FFI Bridge" begin
    # Check availability
    avail = is_c_kernel_available()
    @test avail isa Bool

    # GEMM
    A = randn(16, 12)
    B = randn(12, 14)
    C = zeros(16, 14)
    c_simd_gemm!(C, A, B; alpha=1.5, beta=0.0)
    @test C ≈ 1.5 .* (A * B) atol=1e-12

    # Tensor contraction
    W = randn(6, 4, 8)
    X = ComplexF64.(randn(8, 4), randn(8, 4))
    Y = c_simd_tensor_contract(W, X)
    @test size(Y) == (8, 6)

    # Clenshaw Chebyshev evaluation
    coeffs = [1.0, 0.5, 0.25]
    xs = [-0.5, 0.0, 0.5]
    p_vals = c_chebyshev_eval(coeffs, xs)
    # T0=1, T1=x, T2=2x^2-1 -> p(x) = 1 + 0.5x + 0.25(2x^2 - 1)
    p_ref = [1.0 + 0.5*x + 0.25*(2x^2 - 1) for x in xs]
    @test p_vals ≈ p_ref atol=1e-12

    # Symplectic step
    q = [1.0]
    p = [0.0]
    q_next, p_next = c_symplectic_step(q, p, [1.0], [1.0], 0.01)
    @test length(q_next) == 1 && length(p_next) == 1

    # Sheaf diffusion
    L_dense = [2.0 -1.0; -1.0 2.0]
    L_csc = dense_to_csc(L_dense)
    x_in = [1.0, 3.0]
    x_diff = c_sheaf_diffuse(L_csc, x_in, 0.1)
    @test x_diff ≈ x_in .- 0.1 .* (L_dense * x_in) atol=1e-12
end
