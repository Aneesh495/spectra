using Test
using Spectra
using LinearAlgebra

@testset "Linear Algebra & Krylov Solvers" begin
    # Blocked GEMM
    A = randn(20, 15)
    B = randn(15, 25)
    C = gemm_blocked(A, B)
    @test C ≈ A * B

    # Householder QR
    M_rect = randn(16, 10)
    qr_res = householder_qr(M_rect)
    Q = qr_res.Q
    R = qr_res.R
    @test size(Q) == (16, 16)
    @test size(R) == (16, 10)
    @test Q' * Q ≈ I(16)
    @test Q * R ≈ M_rect

    # Bidiagonal SVD
    A_sq = randn(10, 10)
    svd_res = bidiagonal_svd(A_sq; max_iter=200)
    @test svd_res.U * Diagonal(svd_res.S) * svd_res.Vt ≈ A_sq atol=1e-5

    # Lanczos on symmetric matrix
    H_sym = A_sq + A_sq'
    q1 = randn(10); q1 ./= norm(q1)
    lanc = lanczos(H_sym, q1, 6)
    @test size(lanc.T) == (6, 6)
    @test isapprox(lanc.T, lanc.T'; atol=1e-12)
    @test lanc.V' * lanc.V ≈ I(6) atol=1e-12

    # Arnoldi iteration
    arn = arnoldi(A_sq, q1, 6)
    @test size(arn.H) == (7, 6)
    @test arn.V' * arn.V ≈ I(7) atol=1e-12


    # GMRES
    b_vec = randn(10)
    x_gmres, conv_g, _ = gmres(A_sq, b_vec; tol=1e-8, max_iter=100)
    @test conv_g
    @test A_sq * x_gmres ≈ b_vec atol=1e-6

    # Conjugate Gradient on SPD matrix
    SPD_mat = A_sq' * A_sq + 2.0 * I(10)
    x_cg, conv_c, _ = cg(SPD_mat, b_vec; tol=1e-10)
    @test conv_c
    @test SPD_mat * x_cg ≈ b_vec atol=1e-8
end
