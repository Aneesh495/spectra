using Test
using Spectra
using LinearAlgebra

@testset "Neural Operators & Numerical PDE Solvers" begin
    # 1D FNO
    fno1d = FourierNeuralOperator1D(2, 1; hidden_c=16, modes=8, n_layers=2)
    x1d = randn(32, 2)
    y1d = fno1d(x1d)
    @test size(y1d) == (32, 1)

    # 2D FNO
    fno2d = FourierNeuralOperator2D(1, 1; hidden_c=8, modes1=4, modes2=4, n_layers=2)
    x2d = randn(16, 16, 1)
    y2d = fno2d(x2d)
    @test size(y2d) == (16, 16, 1)


    # 1D CNO (Chebyshev)
    cno = ChebyshevNeuralOperator1D(2, 1; hidden_c=16, modes=8, n_layers=2)
    ycno = cno(x1d)
    @test size(ycno) == (32, 1)

    # 1D WNO (Wavelet)
    wno = WaveletNeuralOperator1D(2, 1; hidden_c=16, levels=3, n_layers=2, family=:haar)
    ywno = wno(x1d)
    @test size(ywno) == (32, 1)

    # DeepONet
    deeponet = DeepONet(10, 2; hidden_branch=[16, 16], hidden_trunk=[16, 16], latent_dim=16)
    u_sens = randn(10)
    y_coord = [0.5, 0.5]
    out_deep = deeponet(u_sens, y_coord)
    @test isfinite(out_deep)

    U_batch = randn(4, 10)
    Y_batch = randn(6, 2)
    out_b = deeponet(U_batch, Y_batch)
    @test size(out_b) == (4, 6)

    # PDE Solvers: 1D Burgers
    u0 = sin.(2.0 .* pi .* (0:31) ./ 32)
    sol_burg = solve_burgers_1d(u0, 0.01, (0.0, 0.02); dt=0.005)
    @test size(sol_burg, 2) == 32

    # PDE Solvers: 2D Darcy
    a_mat = ones(8, 8)
    u_darcy = solve_darcy_2d(a_mat, 1.0; tol=1e-5)
    @test size(u_darcy) == (8, 8)

    # PDE Solvers: 2D Navier-Stokes
    sx = sin.(collect(range(0, 2pi, length=8)))
    cy = cos.(collect(range(0, 2pi, length=8)))
    w0 = sx * cy'
    w_ns = solve_navier_stokes_2d(w0, 0.001, 0.01; dt=0.005)
    @test size(w_ns) == (8, 8)
end
