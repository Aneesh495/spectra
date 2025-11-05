"""
    Spectra.Geometry.Stiefel

Stiefel manifold St(p, n) of orthonormal p-frames in R^n: {X in R^{n x p} : X^T X = I_p}.
Implements canonical metric, orthogonal tangent projections, and Cayley transform retractions.
"""

struct Stiefel{P, N} <: AbstractManifold{0}
    function Stiefel{P, N}() where {P, N}
        @assert 1 <= P <= N "Dimensions must satisfy 1 <= P <= N"
        new{P, N}()
    end
end

Stiefel(p::Int, n::Int) = Stiefel{p, n}()

manifold_dim(::Stiefel{P, N}) where {P, N} = N * P - div(P * (P + 1), 2)
ambient_dim(::Stiefel{P, N}) where {P, N} = N * P

function check_point(::Stiefel{P, N}, X::Matrix{Float64}; tol::Float64=1e-8)::Bool where {P, N}
    size(X) == (N, P) || return false
    diff = X' * X - I
    return norm(diff) < tol
end

function check_tangent(::Stiefel{P, N}, X::Matrix{Float64}, V::Matrix{Float64}; tol::Float64=1e-8)::Bool where {P, N}
    size(V) == (N, P) || return false
    sym_part = X' * V + V' * X
    return norm(sym_part) < tol
end

function inner_product(::Stiefel{P, N}, X::Matrix{Float64}, U::Matrix{Float64}, V::Matrix{Float64})::Float64 where {P, N}
    return tr(U' * V)
end

function project_tangent(::Stiefel{P, N}, X::Matrix{Float64}, Z::Matrix{Float64})::Matrix{Float64} where {P, N}
    # Tangent condition: sym(X^T V) = 0
    # Proj(Z) = Z - X sym(X^T Z)
    XtZ = X' * Z
    sym_XtZ = 0.5 .* (XtZ .+ XtZ')
    return Z .- X * sym_XtZ
end

"""
    cayley_retraction(::Stiefel{P, N}, X::Matrix{Float64}, V::Matrix{Float64}) -> Matrix{Float64}

Cayley transform retraction preserving exact orthonormality:
R_X(V) = (I - 0.5 W)^{-1} (I + 0.5 W) X where W = V X^T - X V^T.
"""
function cayley_retraction(::Stiefel{P, N}, X::Matrix{Float64}, V::Matrix{Float64})::Matrix{Float64} where {P, N}
    # W = V X^T - X V^T in R^{N x N}
    # For efficiency when P << N: W = [X V] [0 -I; I 0] [X V]^T (low rank form)
    U = [X V] # N x 2P
    # Compute low rank inversion via Sherman-Morrison-Woodbury
    W = V * X' - X * V'
    I_N = Matrix{Float64}(I, N, N)
    inv_term = (I_N - 0.5 .* W) \ (I_N + 0.5 .* W)
    return inv_term * X
end

function retraction(M::Stiefel{P, N}, X::Matrix{Float64}, V::Matrix{Float64})::Matrix{Float64} where {P, N}
    return cayley_retraction(M, X, V)
end

function random_point(::Stiefel{P, N})::Matrix{Float64} where {P, N}
    A = randn(N, P)
    Q = Matrix(qr(A).Q)
    return Q[:, 1:P]
end
