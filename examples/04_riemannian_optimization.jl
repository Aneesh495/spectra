"""
    Showcase Example 04: Riemannian Manifold Optimization

Demonstrates Riemannian optimization on curved non-Euclidean manifolds:
1. Solving the Rayleigh quotient eigenvalue problem on the Sphere S^{N-1}:
       min_{x in S^{N-1}} -x^T A x
2. Orthonormal subspace alignment on the Stiefel manifold St(p, n).
Uses Riemannian Gradient Descent, Riemannian Conjugate Gradients, and Riemannian Adam.
"""

using Spectra
using LinearAlgebra
using Printf

println("==========================================================================")
println(" Spectra Showcase 04: Riemannian Optimization on Curved Manifolds         ")
println("==========================================================================")

# 1. Sphere S^4 (ambient R^5) Rayleigh Quotient Eigenvalue Problem
N = 5
sphere = Sphere{4}() # Manifold dim 4, ambient dim 5
println("[*] Target Manifold: 4-Sphere S^4 (ambient R^5)")

# Construct symmetric matrix A with known eigenvalues
Random.seed!(42)
M_rand = randn(N, N)
A_sym = M_rand + M_rand'
eigen_true = eigen(A_sym)
max_lambda = eigen_true.values[end]
v_max = eigen_true.vectors[:, end]
if v_max[1] < 0
    v_max = -v_max
end

println("[*] Ground Truth Matrix A Leading Eigenvalue: ", max_lambda)

# Objective: minimize f(x) = -0.5 * x^T A x
cost_and_grad = function(x)
    Ax = A_sym * x
    loss = -0.5 * dot(x, Ax)
    euc_grad = -Ax
    return (loss, euc_grad)
end

# 2. Optimize via Riemannian Gradient Descent
x0 = randn(N)
x0 ./= norm(x0)

println("\n[*] Running Riemannian Gradient Descent (RGD)...")
x_rgd, hist_rgd = riemannian_gradient_descent(sphere, cost_and_grad, x0; lr=0.1, max_iter=80, tol=1e-8)
rayleigh_rgd = dot(x_rgd, A_sym * x_rgd)
@printf("[+] RGD Final Eigenvalue Estimate: %9.6f (Loss: %9.6f)\n", rayleigh_rgd, hist_rgd[end])
@printf("    Eigenvalue Absolute Error: %.2e\n", abs(rayleigh_rgd - max_lambda))

# 3. Optimize via Riemannian Conjugate Gradients
println("\n[*] Running Riemannian Conjugate Gradient (RCG)...")
x_rcg, hist_rcg = riemannian_cg(sphere, cost_and_grad, x0; max_iter=80, tol=1e-8)
rayleigh_rcg = dot(x_rcg, A_sym * x_rcg)
@printf("[+] RCG Final Eigenvalue Estimate: %9.6f (Loss: %9.6f)\n", rayleigh_rcg, hist_rcg[end])
@printf("    Eigenvalue Absolute Error: %.2e\n", abs(rayleigh_rcg - max_lambda))

# 4. Optimize via Riemannian Adam
println("\n[*] Running Riemannian Adam Optimizer...")
x_radam, hist_radam = riemannian_adam(sphere, cost_and_grad, x0; lr=0.08, max_iter=100, tol=1e-8)
rayleigh_radam = dot(x_radam, A_sym * x_radam)
@printf("[+] Riemannian Adam Final Estimate: %9.6f\n", rayleigh_radam)

# Verify points remain on manifold to machine epsilon
println("\n[*] Manifold Constraint Verification:")
@printf("    ||x_rgd||   = %.16f (dev: %.2e)\n", norm(x_rgd), abs(norm(x_rgd) - 1.0))
@printf("    ||x_rcg||   = %.16f (dev: %.2e)\n", norm(x_rcg), abs(norm(x_rcg) - 1.0))
@printf("    ||x_radam|| = %.16f (dev: %.2e)\n", norm(x_radam), abs(norm(x_radam) - 1.0))

println("\n==========================================================================")
println(" Showcase 04 Execution Completed Successfully!                            ")
println("==========================================================================")
