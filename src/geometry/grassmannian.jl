"""
    Spectra.Geometry.Grassmannian

Grassmann manifold Gr(p, n) of p-dimensional linear subspaces in R^n.
Implements exact geodesic exponential maps via SVD, principal angles, and subspace metrics.
"""

struct Grassmannian{P, N} <: AbstractManifold{0}
    function Grassmannian{P, N}() where {P, N}
        @assert 1 <= P <= N "Dimensions must satisfy 1 <= P <= N"
        new{P, N}()
    end
end

Grassmannian(p::Int, n::Int) = Grassmannian{p, n}()

manifold_dim(::Grassmannian{P, N}) where {P, N} = P * (N - P)
ambient_dim(::Grassmannian{P, N}) where {P, N} = N * P

function check_point(::Grassmannian{P, N}, X::Matrix{Float64}; tol::Float64=1e-8)::Bool where {P, N}
    size(X) == (N, P) || return false
    return norm(X' * X - I) < tol
end

function check_tangent(::Grassmannian{P, N}, X::Matrix{Float64}, V::Matrix{Float64}; tol::Float64=1e-8)::Bool where {P, N}
    size(V) == (N, P) || return false
    return norm(X' * V) < tol
end

function project_tangent(::Grassmannian{P, N}, X::Matrix{Float64}, Z::Matrix{Float64})::Matrix{Float64} where {P, N}
    # Horizontal tangent space condition: X^T V = 0
    # Proj(Z) = (I - X X^T) Z
    return Z .- X * (X' * Z)
end

function exp_map(::Grassmannian{P, N}, X::Matrix{Float64}, V::Matrix{Float64})::Matrix{Float64} where {P, N}
    norm_V = norm(V)
    if norm_V < 1e-15
        return copy(X)
    end

    # Thin SVD of tangent vector V = U Sigma V_R^T
    F = svd(V)
    U = F.U[:, 1:P]
    sigma = F.S[1:P]
    Vt = F.Vt

    # Geodesic: X(t) = X V_R cos(Sigma) + U sin(Sigma)
    cos_sigma = Diagonal(cos.(sigma))
    sin_sigma = Diagonal(sin.(sigma))

    X_next = X * (Vt' * cos_sigma) + U * sin_sigma
    # Orthonormalize via QR to eliminate numerical drift
    return Matrix(qr(X_next).Q)[:, 1:P]
end

"""
    principal_angles(::Grassmannian{P, N}, X::Matrix{Float64}, Y::Matrix{Float64}) -> Vector{Float64}

Compute canonical principal angles 0 <= theta_1 <= ... <= theta_P <= pi/2 between subspaces.
"""
function principal_angles(::Grassmannian{P, N}, X::Matrix{Float64}, Y::Matrix{Float64})::Vector{Float64} where {P, N}
    XtY = X' * Y
    s = svd(XtY).S
    clamped_s = clamp.(s, -1.0, 1.0)
    return sort(acos.(clamped_s))
end

function geodesic_distance(M::Grassmannian{P, N}, X::Matrix{Float64}, Y::Matrix{Float64})::Float64 where {P, N}
    theta = principal_angles(M, X, Y)
    return norm(theta)
end

function random_point(::Grassmannian{P, N})::Matrix{Float64} where {P, N}
    A = randn(N, P)
    return Matrix(qr(A).Q)[:, 1:P]
end
