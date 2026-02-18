using Test
using Spectra
using LinearAlgebra

@testset "Continuous Dynamics & Geometric Neural ODEs" begin
    # Adaptive Runge-Kutta DP54: dx/dt = -x -> x(t) = exp(-t)
    ode_f = (x, p, t) -> -x
    sol_dp = integrate_ode(ode_f, [1.0], (0.0, 1.0); tableau=DP54Tableau(), dt_init=0.1, atol=1e-8, rtol=1e-6)
    @test sol_dp.u[end][1] ≈ exp(-1.0) atol=1e-5

    # Symplectic Verlet: Harmonic oscillator H = 0.5*p^2 + 0.5*q^2
    H_osc = (q, p) -> 0.5 * (p[1]^2 + q[1]^2)
    grad_V = q -> q
    inv_M = [1.0]
    sol_symp = integrate_symplectic(H_osc, grad_V, inv_M, [1.0], [0.0], (0.0, 2pi), 0.01; order=2)
    # At t=2pi, q ≈ 1.0, p ≈ 0.0, energy conserved
    q_end = sol_symp.q[end][1]
    p_end = sol_symp.p[end][1]
    energy_init = 0.5 * (1.0^2 + 0.0^2)
    energy_end = 0.5 * (q_end^2 + p_end^2)
    @test energy_end ≈ energy_init atol=1e-3

    # SDE Euler-Maruyama: Ornstein-Uhlenbeck dx = -theta*x*dt + sigma*dW
    drift = (x, t) -> -1.0 .* x
    diffusion = (x, t) -> fill(0.2, 1, 1)
    sol_sde = integrate_sde_euler_maruyama(drift, diffusion, [1.0], (0.0, 0.5), 0.01)


    @test length(sol_sde.t) == 51

    # Continuous Adjoint Sensitivity Analysis
    loss_fn = x -> 0.5 * sum(x .^ 2)
    adj = adjoint_sensitivity(ode_f, loss_fn, [1.0], Float64[], (0.0, 0.5))
    # For dx/dt = -x, x(T) = x0 * exp(-T), L = 0.5 * x0^2 * exp(-2T)
    # dL/dx0 = x0 * exp(-2T) = 1.0 * exp(-1.0)
    @test adj.grad_z0[1] ≈ exp(-1.0) atol=1e-3

    # Neural ODE
    node = NeuralODE(2, 16; tspan=(0.0, 0.2))
    x0 = [1.0, 0.5]
    x_T = forward(node, x0)
    @test length(x_T) == 2

    # Hamiltonian Neural Network
    hnn = HamiltonianNeuralNetwork(1, 16)
    vf = vector_field(hnn, [1.0, 0.0])
    @test length(vf) == 2 # [dq/dt, dp/dt]
    sol_hnn = integrate_trajectory(hnn, [1.0], [0.0], (0.0, 0.2), 0.05)

    @test length(sol_hnn.t) > 1

    # Port-Hamiltonian System
    phnn = PortHamiltonianNN(2, 16)
    dissip = energy_dissipation_rate(phnn, [1.0, 0.5])
    @test dissip <= 1e-12 # dH/dt <= 0 always dissipative

end
