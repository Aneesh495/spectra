using Test
using Spectra
using LinearAlgebra

@testset "Riemannian Geometry & Manifold Optimization" begin
    # 2-Sphere S^2
    sphere = Sphere{2}()
    @test manifold_dim(sphere) == 2
    @test ambient_dim(sphere) == 3

    p = [1.0, 0.0, 0.0]
    @test check_point(sphere, p)

    v = [0.0, 1.0, 2.0] # tangent to p since p' * v = 0
    @test check_tangent(sphere, p, v)

    # Exp and Log map consistency
    q = exp_map(sphere, p, v)
    @test check_point(sphere, q)
    v_rec = log_map(sphere, p, q)
    @test v_rec ≈ v atol=1e-10

    # Parallel transport
    w = [0.0, 0.5, -0.2]
    w_transported = parallel_transport(sphere, p, q, w)
    @test check_tangent(sphere, q, w_transported)
    @test norm_tangent(sphere, q, w_transported) ≈ norm_tangent(sphere, p, w)

    # Hyperbolic Geometry: Poincare Ball & Lorentz Hyperboloid
    x_poinc = [0.2, 0.3]
    x_lor = poincare_to_lorentz(x_poinc)


    @test minkowski_dot(x_lor, x_lor) ≈ -1.0 atol=1e-10
    x_poinc_rec = lorentz_to_poincare(x_lor)
    @test x_poinc_rec ≈ x_poinc atol=1e-10

    # Stiefel Manifold St(2, 3)
    st = Stiefel(2, 3)
    X_st = [1.0 0.0; 0.0 1.0; 0.0 0.0]
    @test check_point(st, X_st)

    # Grassmannian Gr(2, 3)
    gr = Grassmannian(2, 3)
    @test check_point(gr, X_st)


    # SPD(3)
    spd = SPDManifold(3)
    P_spd = [2.0 1.0 0.0; 1.0 3.0 0.5; 0.0 0.5 2.0]
    @test check_point(spd, P_spd)
    @test geodesic_distance(spd, P_spd, P_spd) ≈ 0.0 atol=1e-10

    # Lie Groups: SO(3)
    so3 = SO3Group()
    omega = [0.1, -0.2, 0.3]
    R_rot = exp_so3(omega)
    @test check_point(so3, R_rot)
    omega_rec = log_so3(R_rot)
    @test omega_rec ≈ omega atol=1e-10

    # Riemannian Optimization on Sphere S^2: minimize f(x) = x[1] + 2*x[2] + 3*x[3]
    # Optimal point is -c / ||c||
    cost_and_grad = x -> (x[1] + 2.0*x[2] + 3.0*x[3], [1.0, 2.0, 3.0])
    x_opt, hist = riemannian_gradient_descent(sphere, cost_and_grad, [0.0, 0.0, 1.0];
                                              lr=0.1, max_iter=100, tol=1e-8)
    expected_opt = -[1.0, 2.0, 3.0] ./ norm([1.0, 2.0, 3.0])
    @test x_opt ≈ expected_opt atol=1e-4
end

