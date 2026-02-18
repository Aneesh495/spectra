using Test
using Spectra

@testset "Dual & HyperDual Forward AD" begin
    # Scalar differentiation: f(x) = x^3 - 4x + 2 -> f'(x) = 3x^2 - 4
    f_scalar = x -> x^3 - 4.0*x + 2.0
    x0 = 2.5
    df = forward_diff(f_scalar, x0)
    @test df ≈ 3.0 * x0^2 - 4.0

    # HyperDual second derivative: f''(x) = 6x
    hd = HyperDual(x0, 1.0, 1.0, 0.0)
    res_hd = f_scalar(hd)
    @test res_hd.val ≈ f_scalar(x0)
    @test res_hd.eps1 ≈ 3.0 * x0^2 - 4.0
    @test res_hd.eps12 ≈ 6.0 * x0

    # Transcendental functions
    f_trans = x -> sin(x) * exp(cos(x))
    df_trans = forward_diff(f_trans, 1.2)
    expected_df = cos(1.2)*exp(cos(1.2)) - (sin(1.2)^2)*exp(cos(1.2))
    @test df_trans ≈ expected_df atol=1e-12


    # Vector gradient: f(x, y) = x^2 * y + y^3 -> grad = [2xy, x^2 + 3y^2]
    f_vec = v -> v[1]^2 * v[2] + v[2]^3
    v0 = [3.0, 2.0]
    grad = forward_gradient(f_vec, v0)
    @test grad[1] ≈ 2.0 * 3.0 * 2.0
    @test grad[2] ≈ 3.0^2 + 3.0 * 2.0^2

    # Hessian: H = [2y, 2x; 2x, 6y]
    H = forward_hessian(f_vec, v0)
    @test H[1, 1] ≈ 2.0 * v0[2]
    @test H[1, 2] ≈ 2.0 * v0[1]
    @test H[2, 1] ≈ 2.0 * v0[1]
    @test H[2, 2] ≈ 6.0 * v0[2]

    # Jacobian: F(x, y) = [x^2 + y, x*y^2] -> J = [2x, 1; y^2, 2xy]
    F_map = v -> [v[1]^2 + v[2], v[1] * v[2]^2]
    J = forward_jacobian(F_map, v0)
    @test J[1, 1] ≈ 2.0 * v0[1]
    @test J[1, 2] ≈ 1.0
    @test J[2, 1] ≈ v0[2]^2
    @test J[2, 2] ≈ 2.0 * v0[1] * v0[2]
end
