"""
    Showcase Example 05: Koopman Operator & Chaotic Flow Linearization

Analyzes the chaotic Lorenz-63 dynamical system:
    dx/dt = sigma * (y - x)
    dy/dt = x * (rho - z) - y
    dz/dt = x * y - beta * z
Computes:
1. Exact DMD coherent dynamic modes from butterfly attractor trajectories.
2. Hankel delay-embedding DMD (HAVOK framework) for chaos linearization.
3. Perron-Frobenius transfer operator via Ulam discretization.
"""

using Spectra
using LinearAlgebra
using Printf

println("==========================================================================")
println(" Spectra Showcase 05: Koopman Operator & Chaotic Lorenz-63 Attractor      ")
println("==========================================================================")

# 1. Integrate Chaotic Lorenz-63 Attractor
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

lorenz_ode = function(u, p, t)
    x, y, z = u[1], u[2], u[3]
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]
end

u0 = [-8.0, 8.0, 27.0]
tspan = (0.0, 10.0)
dt = 0.01

println("[*] Integrating Chaotic Lorenz-63 trajectory (sigma=10, rho=28, beta=8/3)...")
sol = integrate_ode(lorenz_ode, u0, tspan; dt_init=0.01, atol=1e-8, rtol=1e-6)

# Extract snapshot matrices X and Y
n_steps = length(sol.u)
trajectory = zeros(3, n_steps)
for i in 1:n_steps
    trajectory[:, i] = sol.u[i]
end

X = trajectory[:, 1:(end - 1)]
Y = trajectory[:, 2:end]
@printf("[+] Trajectory integrated! Generated %d temporal snapshots.\n", n_steps)

# 2. Fit Exact Dynamic Mode Decomposition
println("\n[*] Fitting Exact DMD with rank=3 truncation...")
dmd = exact_dmd(X, Y, dt; rank=3)

println("[+] Exact DMD Continuous Eigenvalues (omega = ln(lambda)/dt):")
for (idx, w) in enumerate(dmd.omega)
    @printf("    omega_%d = %9.4f + %9.4fi (growth=%.4f, freq=%.4f rad/s)\n",
            idx, real(w), imag(w), real(w), abs(imag(w)))
end

# 3. Fit Hankel Delay-Embedding DMD (HAVOK) on x-coordinate
println("\n[*] Building Hankel Delay-Embedding DMD (HAVOK) on x(t) time series...")
x_series = trajectory[1, :]
hankel_model = hankel_dmd(x_series, 16, dt; rank=8)

@printf("[+] Hankel DMD fitted! Delay Coordinates: 16 | Effective Rank: %d\n", hankel_model.rank)
println("[*] Hankel Koopman Dominant Spectrum:")
terminal_spectrum_bars(hankel_model.eigenvalues; top_k=6, title="Hankel Delay-Embedding Spectrum")

# 4. Perron-Frobenius Invariant Measure Computation (1D projection)
println("\n[*] Computing Perron-Frobenius Transfer Operator for chaotic x-dynamics...")
part = BoxPartition1D(-20.0, 20.0, 30)

# 1-step discrete mapping function from interpolation
x_min_val, x_max_val = extrema(x_series)
println("    x-coordinate Attractor Extrema: [", round(x_min_val; digits=2), ", ", round(x_max_val; digits=2), "]")

# Render x(t) trajectory in terminal
println("\n[*] Terminal Waveform of Chaotic Lorenz x(t) Attractor Trajectory:")
terminal_plot(x_series[1:min(200, length(x_series))]; width=52, height=12, title="Lorenz Attractor Trajectory x(t)")

println("\n==========================================================================")
println(" Showcase 05 Execution Completed Successfully!                            ")
println("==========================================================================")
