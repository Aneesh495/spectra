"""
    Showcase Example 02: Cellular Sheaves, Cohomology & Neural Sheaf Diffusion

Constructs a cellular sheaf over an oriented pentagon complex (5 vertices, 5 edges).
Assigns non-trivial orthogonal restriction maps (SO(2) rotations) along edges,
computes Sheaf Laplacians, extracts zeroth cohomology global sections, solves
the harmonic Dirichlet boundary value problem, and executes neural sheaf diffusion.
"""

using Spectra
using LinearAlgebra
using Printf

println("==========================================================================")
println(" Spectra Showcase 02: Cellular Sheaf Cohomology & Neural Diffusion        ")
println("==========================================================================")

# 1. Construct 1-skeleton Complex (5-cycle pentagon)
nV = 5
complex = CellComplex(nV)
for i in 1:4
    add_edge!(complex, i, i + 1)
end
add_edge!(complex, 5, 1)

println("[*] Cell Complex initialized with ", num_vertices(complex), " vertices and ", num_edges(complex), " edges.")

# 2. Build Cellular Sheaf with stalk dimension d = 2
d = 2
sheaf = CellularSheaf(complex, d)
println("[*] Cellular Sheaf initialized with stalk dimension d = ", d)
println("    Total 0-Cochain Dimension: ", total_cochain_dim_0(sheaf))
println("    Total 1-Cochain Dimension: ", total_cochain_dim_1(sheaf))

# Assign orthogonal restriction maps: Identity on source, SO(2) rotation on target
I2 = Matrix{Float64}(I, 2, 2)
angles = [0.0, pi/4, pi/2, 3pi/4, pi]
for (e_idx, (u, v)) in enumerate(complex.edges)
    theta = angles[e_idx]
    R_rot = [cos(theta) -sin(theta); sin(theta) cos(theta)]
    set_restriction!(sheaf, u, e_idx, I2)
    set_restriction!(sheaf, v, e_idx, R_rot)
end

# 3. Form Sheaf Laplacian
println("\n[*] Computing Sheaf Laplacian L_F = delta^T * delta...")
slap = SheafLaplacian(sheaf)
L_dense = slap.dense_matrix

println("    Sheaf Laplacian Dimension: ", size(L_dense))
@printf("    Symmetry Error ||L - L^T||: %.2e\n", norm(L_dense - L_dense'))

# 4. Sheaf Spectrum & Cohomology
evals = sheaf_spectrum(sheaf)
h0_dim = sheaf_cohomology_dim(sheaf; tol=1e-5)
println("[+] Sheaf Cohomology H^0(X; F) Dimension (Global Sections): ", h0_dim)
println("[*] Sheaf Laplacian Eigenvalues (lowest 6):")
for (idx, ev) in enumerate(evals[1:min(6, length(evals))])
    @printf("    lambda_%d = %8.4f\n", idx, ev)
end

# 5. Solve Harmonic Dirichlet Boundary Problem
# Fix vertex 1 to value [1.0, 0.0] and vertex 3 to [0.0, 1.0], solve interior
println("\n[*] Solving Harmonic Dirichlet Boundary Value Problem (L_F * x = 0 on interior)...")
fixed_vertices = [1, 3]
fixed_values = [1.0, 0.0, 0.0, 1.0]
x_harmonic = solve_sheaf_dirichlet(sheaf, fixed_vertices, fixed_values)


d_energy = dirichlet_energy(sheaf, x_harmonic)
@printf("[+] Dirichlet Harmonic Field Solved! Total Sheaf Dirichlet Energy: %.6f\n", d_energy)
for v in 1:nV
    st = x_harmonic[((v - 1)*d + 1):(v*d)]
    @printf("    Stalk x(v_%d) = [%7.4f, %7.4f]\n", v, st[1], st[2])
end

# 6. Neural Sheaf Layer Forward Step
println("\n[*] Executing Neural Sheaf Diffusion Layer step...")
nsl = NeuralSheafLayer(4, 4, d)
node_features = randn(nV, 4)
y_features = forward_sheaf_layer(nsl, complex, node_features)
println("[+] Neural Sheaf output feature dimensions: ", size(y_features))

println("\n==========================================================================")
println(" Showcase 02 Execution Completed Successfully!                            ")
println("==========================================================================")
