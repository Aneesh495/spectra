"""
    Spectra.Geometry.SPD

Manifold of N x N Symmetric Positive Definite matrices equipped with the
Affine-Invariant Riemannian Metric (AIRM) and Log-Euclidean Metric.
"""

struct SPDManifold{N} <: AbstractManifold{0}
    function SPDManifold{N}() where {N}
        @assert N >= 1 "Dimension N must be at least 1"
        new{N}()
    end
end

SPDManifold(n::Int) = SPDManifold{n}()

manifold_dim(::SPDManifold{N}) where {N} = div(N * (N + 1), 2)
ambient_dim(::SPDManifold{N}) where {N} = N * N

function matrix_sqrt_sym(A::Matrix{Float64})::Matrix{Float64}
    F = eigen(Symmetric(A))
    vals = sqrt.(max.(F.values, 1e-15))
    return F.vectors * Diagonal(vals) * F.vectors'
end

function matrix_inv_sqrt_sym(A::Matrix{Float64})::Matrix{Float64}
    F = eigen(Symmetric(A))
    inv_vals = inv.(sqrt.(max.(F.values, 1e-15)))
    return F.vectors * Diagonal(inv_vals) * F.vectors'
end

function matrix_exp_sym(A::Matrix{Float64})::Matrix{Float64}
    F = eigen(Symmetric(A))
    return F.vectors * Diagonal(exp.(F.values)) * F.vectors'
end

function matrix_log_sym(A::Matrix{Float64})::Matrix{Float64}
    F = eigen(Symmetric(A))
    return F.vectors * Diagonal(log.(max.(F.values, 1e-15))) * F.vectors'
end

function check_point(::SPDManifold{N}, P::Matrix{Float64}; tol::Float64=1e-8)::Bool where {N}
    size(P) == (N, N) || return false
    is_symmetric_matrix(P, tol) || return false
    # Check strict positive definiteness
    F = eigvals(Symmetric(P))
    return minimum(F) > tol
end

function check_tangent(::SPDManifold{N}, P::Matrix{Float64}, V::Matrix{Float64}; tol::Float64=1e-8)::Bool where {N}
    size(V) == (N, N) || return false
    return is_symmetric_matrix(V, tol)
end

function project_tangent(::SPDManifold{N}, P::Matrix{Float64}, Z::Matrix{Float64})::Matrix{Float64} where {N}
    # Tangent space is the space of symmetric matrices
    return 0.5 .* (Z .+ Z')
end

"""
    inner_product(::SPDManifold, P, U, V) -> Float64

Affine-Invariant Riemannian Metric (AIRM):
<U, V>_P = tr(P^{-1} U P^{-1} V).
"""
function inner_product(::SPDManifold{N}, P::Matrix{Float64}, U::Matrix{Float64}, V::Matrix{Float64})::Float64 where {N}
    inv_P = inv(P)
    return tr(inv_P * U * inv_P * V)
end

"""
    exp_map(::SPDManifold, P, V) -> Matrix{Float64}

AIRM exponential map: Exp_P(V) = P^{1/2} exp(P^{-1/2} V P^{-1/2}) P^{1/2}.
"""
function exp_map(::SPDManifold{N}, P::Matrix{Float64}, V::Matrix{Float64})::Matrix{Float64} where {N}
    P_sqrt = matrix_sqrt_sym(P)
    P_inv_sqrt = matrix_inv_sqrt_sym(P)

    W = P_inv_sqrt * V * P_inv_sqrt
    exp_W = matrix_exp_sym(W)

    res = P_sqrt * exp_W * P_sqrt
    return 0.5 .* (res .+ res')
end

"""
    log_map(::SPDManifold, P, Q) -> Matrix{Float64}

AIRM logarithmic map: Log_P(Q) = P^{1/2} log(P^{-1/2} Q P^{-1/2}) P^{1/2}.
"""
function log_map(::SPDManifold{N}, P::Matrix{Float64}, Q::Matrix{Float64})::Matrix{Float64} where {N}
    P_sqrt = matrix_sqrt_sym(P)
    P_inv_sqrt = matrix_inv_sqrt_sym(P)

    W = P_inv_sqrt * Q * P_inv_sqrt
    log_W = matrix_log_sym(W)

    res = P_sqrt * log_W * P_sqrt
    return 0.5 .* (res .+ res')
end

"""
    geodesic_distance(::SPDManifold, P, Q) -> Float64

AIRM geodesic distance: d(P, Q) = ||log(P^{-1/2} Q P^{-1/2})||_F.
"""
function geodesic_distance(::SPDManifold{N}, P::Matrix{Float64}, Q::Matrix{Float64})::Float64 where {N}
    P_inv_sqrt = matrix_inv_sqrt_sym(P)
    W = P_inv_sqrt * Q * P_inv_sqrt
    log_W = matrix_log_sym(W)
    return norm(log_W)
end

function parallel_transport(::SPDManifold{N}, P::Matrix{Float64}, Q::Matrix{Float64}, V::Matrix{Float64})::Matrix{Float64} where {N}
    # E = (Q P^{-1})^{1/2}
    QP_inv = Q * inv(P)
    # Using matrix square root
    E = matrix_sqrt_sym(0.5 .* (QP_inv .+ QP_inv'))
    res = E * V * E'
    return 0.5 .* (res .+ res')
end

function random_point(::SPDManifold{N})::Matrix{Float64} where {N}
    A = randn(N, N)
    return A * A' + 0.1 * I
end
