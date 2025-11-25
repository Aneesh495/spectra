"""
    Spectra.Sheaves.SheafDiffusion

Continuous and discrete sheaf heat diffusion, harmonic extensions, and Dirichlet boundary value solvers.
x(t) = exp(-t L_F) x(0) converges to the orthogonal projection onto the zeroth cohomology H^0(X; F).
"""

"""
    sheaf_diffusion_step(sheaf::CellularSheaf, x::Vector{Float64}, dt::Float64) -> Vector{Float64}

Explicit Euler time step of continuous sheaf heat diffusion:
x_{k+1} = x_k - dt * L_F * x_k.
"""
function sheaf_diffusion_step(sheaf::CellularSheaf, x::Vector{Float64}, dt::Float64)::Vector{Float64}
    L = sheaf_laplacian(sheaf)
    Lx = L.matrix * x
    return x .- dt .* Lx
end

"""
    sheaf_diffusion_solve(sheaf::CellularSheaf, x0::Vector{Float64}, t_final::Float64; n_steps::Int=50) -> (x_final, energy_history)

Integrate sheaf heat diffusion from t = 0 to t = t_final via matrix exponential
x(t) = exp(-t L_F) x0.
Guarantees monotonic decrease of Dirichlet energy dE/dt = -2 ||L_F x||^2 <= 0.
"""
function sheaf_diffusion_solve(sheaf::CellularSheaf, x0::Vector{Float64}, t_final::Float64; n_steps::Int=50)
    L = sheaf_laplacian(sheaf)
    F = eigen(Symmetric(L.dense_matrix))

    dt = t_final / n_steps
    energy_history = Float64[dirichlet_energy(sheaf, x0)]
    x_curr = copy(x0)

    # Spectral evaluation: exp(-t L) = V exp(-t Lambda) V^T
    for step in 1:n_steps
        t = step * dt
        exp_diag = exp.(-t .* F.values)
        x_curr = F.vectors * (exp_diag .* (F.vectors' * x0))
        push!(energy_history, dirichlet_energy(sheaf, x_curr))
    end

    return (x_curr, energy_history)
end

"""
    solve_sheaf_dirichlet(sheaf::CellularSheaf, boundary_vertices::Vector{Int}, boundary_values::Vector{Float64}) -> Vector{Float64}

Solve the sheaf Dirichlet boundary value problem:
Find harmonic cochain x satisfying L_F x = 0 on interior vertices with x|_B = boundary_values.
Solves L_{II} x_I = -L_{IB} x_B.
"""
function solve_sheaf_dirichlet(sheaf::CellularSheaf, boundary_vertices::Vector{Int}, boundary_values::Vector{Float64})::Vector{Float64}
    nV = sheaf.complex.num_vertices
    interior_vertices = filter(v -> !(v in boundary_vertices), 1:nV)

    # Collect boundary indices in 0-cochain vector
    b_indices = Int[]
    for v in boundary_vertices
        append!(b_indices, vertex_range(sheaf, v))
    end
    @assert length(b_indices) == length(boundary_values) "Boundary values count must match boundary cochain dimension"

    # Collect interior indices
    i_indices = Int[]
    for v in interior_vertices
        append!(i_indices, vertex_range(sheaf, v))
    end

    L = sheaf_laplacian(sheaf)
    L_dense = L.dense_matrix

    L_II = L_dense[i_indices, i_indices]
    L_IB = L_dense[i_indices, b_indices]

    rhs = -L_IB * boundary_values
    x_I = L_II \ rhs

    # Assemble complete solution
    total_dim = total_cochain_dim_0(sheaf)
    x_sol = zeros(Float64, total_dim)
    x_sol[b_indices] = boundary_values
    x_sol[i_indices] = x_I

    return x_sol
end
