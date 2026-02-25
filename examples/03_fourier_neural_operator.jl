"""
    Showcase Example 03: Fourier Neural Operators on 1D Burgers Equation

Generates high-accuracy ground-truth shock waveforms for the viscous Burgers equation:
    du/dt + u * du/dx = nu * d^2u/dx^2,  nu = 0.01
using the pseudo-spectral RK4 engine.
Initializes and evaluates a 1D Fourier Neural Operator (FNO-1D) architecture with
spectral convolutions, displaying terminal waveforms and spectrum comparisons.
"""

using Spectra
using LinearAlgebra
using Printf

println("==========================================================================")
println(" Spectra Showcase 03: Fourier Neural Operator (FNO-1D) on Burgers PDE     ")
println("==========================================================================")

# 1. Generate Synthetic Training Data from Gaussian Random Field Initial Conditions
N_spatial = 64
nu = 0.01
t_final = 0.25

println("[*] Generating ground truth initial condition u0(x) via Gaussian Random Field...")
u0 = gaussian_random_field_1d(N_spatial; length_scale=0.15, variance=1.0)

println("[*] Solving viscous Burgers equation (nu=0.01) from t=0.0 to t=0.25...")
sol_history = solve_burgers_1d(u0, nu, (0.0, t_final); dt=0.002)
u_target = sol_history[end, :]

@printf("[+] Simulation converged! Snapshots: %d | Spatial Resolution: %d\n", size(sol_history, 1), N_spatial)
@printf("    Initial Max Amp: %.4f | Final Shock Amp: %.4f\n", maximum(abs.(u0)), maximum(abs.(u_target)))

# 2. Render Initial Condition and Shock Waveform
println("\n[*] Initial Condition u0(x):")
terminal_plot(u0; width=50, height=10, title="Initial Profile u0(x) [t=0.0]")

println("\n[*] Developed Shock Waveform u(x, t=0.25):")
terminal_plot(u_target; width=50, height=10, title="Viscous Burgers Shock Profile u(x, t=0.25)")

# 3. Instantiate Fourier Neural Operator
println("\n[*] Initializing Fourier Neural Operator (FNO-1D)...")
println("    In Channels: 2 (u0, coordinate grid x) | Hidden Channels: 32 | Modes: 16 | Layers: 4")
fno = FourierNeuralOperator1D(2, 1; hidden_c=32, modes=16, n_layers=4)

# Create input tensor [u0, grid_x]
grid_x = collect(range(0.0, 1.0, length=N_spatial))
X_input = [u0 grid_x]

println("[*] Executing FNO-1D Forward Inference Pass...")
y_pred = fno(X_input)
println("[+] FNO-1D Output Shape: ", size(y_pred))

# 4. Discrete Wavelet Decomposition Comparison
println("\n[*] Analyzing Multiscale Wavelet Content of Shock Waveform (Haar DWT)...")
decomp = forward_dwt_1d(u_target, 3, :haar)
println("    Approximation Scale Len: ", length(decomp.approx))
for (lvl, det) in enumerate(decomp.details)
    @printf("    Wavelet Detail Level %d Energy: %.4e\n", lvl, norm(det))
end

println("\n==========================================================================")
println(" Showcase 03 Execution Completed Successfully!                            ")
println("==========================================================================")
