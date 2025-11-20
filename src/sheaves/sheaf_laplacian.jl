"""
    Spectra.Sheaves.SheafLaplacian

Sheaf coboundary operators delta : C^0(X; F) -> C^1(X; F),
Sheaf Laplacian L_F = delta^T * delta, Dirichlet energy, and zeroth sheaf cohomology H^0(X; F).
"""

struct SheafLaplacian <: AbstractSheafLaplacian
    sheaf::CellularSheaf
    matrix::CSCMatrix{Float64}
    dense_matrix::Matrix{Float64}
end

Base.size(L::SheafLaplacian) = size(L.matrix)
Base.size(L::SheafLaplacian, d::Int) = size(L.matrix, d)

"""
    sheaf_coboundary(sheaf::CellularSheaf) -> CSCMatrix{Float64}

Construct the sparse sheaf coboundary matrix delta : C^0(X; F) -> C^1(X; F).
Dimension: (sum_e dim(F(e))) x (sum_v dim(F(v))).
For each directed edge e = (u, v):
  (delta x)_e = F_{v <= e} x_v - F_{u <= e} x_u.
"""
function sheaf_coboundary(sheaf::CellularSheaf)::CSCMatrix{Float64}
    dim0 = total_cochain_dim_0(sheaf)
    dim1 = total_cochain_dim_1(sheaf)
    coo = COOMatrix(dim1, dim0)

    for (e_idx, (u, v)) in enumerate(sheaf.complex.edges)
        e_rng = edge_range(sheaf, e_idx)
        u_rng = vertex_range(sheaf, u)
        v_rng = vertex_range(sheaf, v)

        # R_u: F_{u <= e}
        R_u = get_restriction(sheaf, u, e_idx)
        # R_v: F_{v <= e}
        R_v = get_restriction(sheaf, v, e_idx)

        # -R_u enters with negative sign (source/tail)
        de, du = size(R_u)
        for j in 1:du
            c_idx = u_rng[j]
            for i in 1:de
                r_idx = e_rng[i]
                val = -R_u[i, j]
                if abs(val) > 1e-15
                    add_entry!(coo, r_idx, c_idx, val)
                end
            end
        end

        # +R_v enters with positive sign (target/head)
        de_v, dv = size(R_v)
        for j in 1:dv
            c_idx = v_rng[j]
            for i in 1:de_v
                r_idx = e_rng[i]
                val = R_v[i, j]
                if abs(val) > 1e-15
                    add_entry!(coo, r_idx, c_idx, val)
                end
            end
        end
    end

    return from_coo(coo)
end

"""
    sheaf_laplacian(sheaf::CellularSheaf) -> SheafLaplacian

Construct the Sheaf Laplacian L_F = delta^T * delta.
Symmetric and positive semi-definite operator acting on 0-cochains C^0(X; F).
"""
function sheaf_laplacian(sheaf::CellularSheaf)::SheafLaplacian
    delta = sheaf_coboundary(sheaf)
    delta_dense = to_dense(delta)
    L_dense = delta_dense' * delta_dense
    # Symmetrize to eliminate numerical asymmetry
    L_dense = 0.5 .* (L_dense .+ L_dense')
    L_sparse = dense_to_csc(L_dense)
    return SheafLaplacian(sheaf, L_sparse, L_dense)
end

"""
    dirichlet_energy(sheaf::CellularSheaf, x::Vector{Float64}) -> Float64

Evaluate the sheaf Dirichlet energy:
E_F(x) = x^T L_F x = ||delta x||^2 = sum_{e=(u,v)} ||F_{v <= e} x_v - F_{u <= e} x_u||^2.
Quantifies discrepancy of cochain x with respect to the sheaf restriction maps.
"""
function dirichlet_energy(sheaf::CellularSheaf, x::Vector{Float64})::Float64
    delta = sheaf_coboundary(sheaf)
    dx = delta * x
    return dot(dx, dx)
end

"""
    global_sections(sheaf::CellularSheaf; tol::Float64=1e-8) -> Matrix{Float64}

Compute an orthonormal basis for the space of global sections H^0(X; F) = ker(L_F) = ker(delta).
Harmonic cochains where the Dirichlet energy is exactly zero.
"""
function global_sections(sheaf::CellularSheaf; tol::Float64=1e-8)::Matrix{Float64}
    L = sheaf_laplacian(sheaf)
    F = eigen(Symmetric(L.dense_matrix))
    null_indices = findall(lambda -> lambda < tol, F.values)
    if isempty(null_indices)
        return zeros(Float64, total_cochain_dim_0(sheaf), 0)
    end
    return F.vectors[:, null_indices]
end

"""
    sheaf_cohomology_dim(sheaf::CellularSheaf; tol::Float64=1e-8) -> Int

Compute the zeroth sheaf cohomology dimension: dim H^0(X; F).
"""
function sheaf_cohomology_dim(sheaf::CellularSheaf; tol::Float64=1e-8)::Int
    sections = global_sections(sheaf; tol=tol)
    return size(sections, 2)
end

"""
    sheaf_spectrum(L::SheafLaplacian) -> Vector{Float64}

Compute all sorted eigenvalues of the Sheaf Laplacian 0 <= lambda_1 <= ... <= lambda_N.
"""
function sheaf_spectrum(L::SheafLaplacian)::Vector{Float64}
    vals = eigvals(Symmetric(L.dense_matrix))
    return sort(max.(vals, 0.0))
end
