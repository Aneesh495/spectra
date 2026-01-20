"""
    Spectra.Koopman.EDMD

Extended Dynamic Mode Decomposition (EDMD) for approximating the infinite-dimensional
Koopman operator over finite-dimensional feature dictionary spaces (Williams et al. 2015).
Computes finite-rank Koopman matrix representation, Koopman eigenfunctions, and Koopman modes.
"""

"""
    AbstractDictionary

Abstract type for dictionary observables in EDMD.
"""
abstract type AbstractDictionary end

"""
    PolynomialDictionary <: AbstractDictionary

Monomial polynomial dictionary up to a maximum degree.
"""
struct PolynomialDictionary <: AbstractDictionary
    dim::Int
    max_degree::Int
    exponents::Matrix{Int} # (n_basis, dim)

    function PolynomialDictionary(dim::Int, max_degree::Int)
        exponents = Vector{Vector{Int}}()
        # Always include constant 1
        push!(exponents, zeros(Int, dim))

        # Helper recursive generator
        function generate_degrees!(curr_deg::Int, current::Vector{Int}, pos::Int)
            if pos > dim
                if sum(current) == curr_deg
                    push!(exponents, copy(current))
                end
                return
            end
            for d in 0:(curr_deg - sum(current[1:(pos - 1)]))
                current[pos] = d
                generate_degrees!(curr_deg, current, pos + 1)
            end
        end

        for deg in 1:max_degree
            temp = zeros(Int, dim)
            generate_degrees!(deg, temp, 1)
        end

        mat = zeros(Int, length(exponents), dim)
        for i in 1:length(exponents)
            mat[i, :] = exponents[i]
        end
        new(dim, max_degree, mat)
    end
end

"""
    evaluate(dict::PolynomialDictionary, x::Vector{Float64}) -> Vector{Float64}
"""
function evaluate(dict::PolynomialDictionary, x::Vector{Float64})::Vector{Float64}
    n_basis = size(dict.exponents, 1)
    vals = zeros(Float64, n_basis)

    for i in 1:n_basis
        prod_val = 1.0
        for d in 1:dict.dim
            p = dict.exponents[i, d]
            if p > 0
                prod_val *= x[d]^p
            end
        end
        vals[i] = prod_val
    end
    return vals
end

"""
    RBFDictionary <: AbstractDictionary

Radial Basis Function (Gaussian RBF) dictionary over centers c_k.
"""
struct RBFDictionary <: AbstractDictionary
    dim::Int
    centers::Matrix{Float64} # (dim, n_centers)
    gamma::Float64

    function RBFDictionary(centers::Matrix{Float64}; gamma::Float64=1.0)
        dim, n_c = size(centers)
        new(dim, centers, gamma)
    end
end

"""
    evaluate(dict::RBFDictionary, x::Vector{Float64}) -> Vector{Float64}
"""
function evaluate(dict::RBFDictionary, x::Vector{Float64})::Vector{Float64}
    n_centers = size(dict.centers, 2)
    vals = zeros(Float64, n_centers + 1)
    vals[1] = 1.0 # constant term

    for k in 1:n_centers
        diff = x - dict.centers[:, k]
        dist_sq = dot(diff, diff)
        vals[k + 1] = exp(-dict.gamma * dist_sq)
    end
    return vals
end

"""
    EDMDResult

Container for fitted Extended Dynamic Mode Decomposition model.
"""
struct EDMDResult <: AbstractKoopmanModel
    K::Matrix{Float64}                 # Koopman matrix approximation: psi(y) ~ K * psi(x)
    eigenvalues::Vector{ComplexF64}    # Koopman eigenvalues
    eigenvectors::Matrix{ComplexF64}   # Left eigenvectors defining eigenfunctions
    modes::Matrix{ComplexF64}          # Koopman modes for state reconstruction
    C::Matrix{Float64}                 # State reconstruction projection C = X * pinv(Psi_X)
    dictionary::AbstractDictionary
end

"""
    extended_dmd(X::Matrix{Float64}, Y::Matrix{Float64}, dict::AbstractDictionary;
                 reg::Float64=1e-8) -> EDMDResult

Fits EDMD given snapshot pairs X and Y using observable dictionary dict.
Computes K = (Y_psi * X_psi') * inv(X_psi * X_psi' + reg * I).
"""
function extended_dmd(X::Matrix{Float64}, Y::Matrix{Float64}, dict::AbstractDictionary;
                      reg::Float64=1e-8)::EDMDResult
    n_states, M = size(X)
    @assert size(Y) == (n_states, M) "Snapshot matrices X and Y must have same size"

    # Evaluate dictionary for all snapshots
    psi_x0 = evaluate(dict, X[:, 1])
    K_basis = length(psi_x0)

    Psi_X = zeros(Float64, K_basis, M)
    Psi_Y = zeros(Float64, K_basis, M)

    for m in 1:M
        Psi_X[:, m] = evaluate(dict, X[:, m])
        Psi_Y[:, m] = evaluate(dict, Y[:, m])
    end

    # Gram matrices: G = (1/M) * Psi_X * Psi_X', A = (1/M) * Psi_Y * Psi_X'
    G = (Psi_X * Psi_X') ./ M
    A = (Psi_Y * Psi_X') ./ M

    # Regularized least-squares solve: K = A * (G + reg * I)^{-1}
    G_reg = G + reg * I
    K = A * inv(G_reg)

    # Eigendecomposition of Koopman operator
    eigen_decomp = eigen(K)
    lambda = eigen_decomp.values
    V = eigen_decomp.vectors

    # State reconstruction matrix C = X * pinv(Psi_X)
    pinv_psi = pinv(Psi_X)
    C = X * pinv_psi
    modes = ComplexF64.(C) * V

    return EDMDResult(K, lambda, V, modes, C, dict)
end

"""
    koopman_eigenfunction(edmd::EDMDResult, j::Int, x::Vector{Float64}) -> ComplexF64

Evaluates the j-th Koopman eigenfunction phi_j(x) = v_j^T * psi(x).
"""
function koopman_eigenfunction(edmd::EDMDResult, j::Int, x::Vector{Float64})::ComplexF64
    psi_val = evaluate(edmd.dictionary, x)
    v_j = edmd.eigenvectors[:, j]
    return dot(conj.(v_j), ComplexF64.(psi_val))
end

"""
    predict_next(edmd::EDMDResult, x::Vector{Float64}) -> Vector{Float64}

Predicts next continuous state x_{k+1} using Koopman matrix and state projection matrix C:
    x_{k+1} = C * K * psi(x_k)
"""
function predict_next(edmd::EDMDResult, x::Vector{Float64})::Vector{Float64}
    psi_x = evaluate(edmd.dictionary, x)
    psi_next = edmd.K * psi_x
    return edmd.C * psi_next
end

