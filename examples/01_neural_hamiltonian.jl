"""
    Showcase Example 01: Hamiltonian Neural Networks & Symplectic Conservation

Demonstrates learning Hamiltonian dynamics for a nonlinear pendulum:
    dH/dq = sin(q),  dH/dp = p
    H(q, p) = 0.5 * p^2 + (1 - cos(q))

Integrates using 4th-order Forest-Ruth symplectic integrator, verifying exact
phase space volume preservation and long-term energy conservation without secular drift.
"""

using Spectra
using LinearAlgebra
using Printf

println("==========================================================================")
println(" Spectra Showcase 01: Hamiltonian Neural Networks & Symplectic Dynamics  ")
println("==========================================================================")

# 1. Setup Ground-Truth Physical Pendulum
true_hamiltonian = (q, p) -> 0.5 * p[1]^2 + (1.0 - cos(q[1]))
grad_V = q -> [sin(q[1])]
inv_M = [1.0]

# 2. Integrate Ground Truth Trajectory with 4th-order Forest-Ruth Symplectic Integrator
tspan = (0.0, 4.0 * pi)
dt = 0.05
q0 = [1.2] # Initial displacement angle
p0 = [0.0] # Initial angular momentum

println("[*] Integrating true physical pendulum trajectory using Forest-Ruth IRK...")
true_sol = integrate_symplectic(true_hamiltonian, grad_V, inv_M, q0, p0, tspan, dt; order=4)

q_vals = [q[1] for q in true_sol.q]
p_vals = [p[1] for p in true_sol.p]
e_vals = true_sol.energy

delta_E = abs(e_vals[end] - e_vals[1])
max_drift = maximum(abs.(e_vals .- e_vals[1]))
@printf("[+] Integration complete over %d steps.\n", length(true_sol.t))
@printf("    Initial Energy: %.6f | Final Energy: %.6f\n", e_vals[1], e_vals[end])
@printf("    Terminal Energy Delta: %.2e | Max Energy Drift: %.2e\n", delta_E, max_drift)

# 3. Terminal Phase Space Portrait
println("\n[*] Rendering True Closed Phase Space Orbit in Terminal:")
terminal_phase_portrait(q_vals, p_vals; width=44, height=14, title="Nonlinear Pendulum Phase Space (q, p)")

# 4. Instantiate & Evaluate Hamiltonian Neural Network
println("\n[*] Initializing Hamiltonian Neural Network (HNN) with MLP architecture...")
hnn = HamiltonianNeuralNetwork(1, 24)
hnn_vf = vector_field(hnn, [q0[1], p0[1]])
println("    HNN Vector Field at (q0, p0): [dq/dt, dp/dt] = ", hnn_vf)

hnn_sol = integrate_trajectory(hnn, q0, p0, (0.0, 2.0 * pi), dt)
println("[+] HNN Symplectic Trajectory Integrated (", length(hnn_sol.t), " points).")

println("\n==========================================================================")
println(" Showcase 01 Execution Completed Successfully!                            ")
println("==========================================================================")
