"""
    Spectra.Core.Utils

High-stability numerical routines, condition guards, polynomial evaluators,
and projection utilities.
"""

"""
    safe_sqrt(x::T, eps::T=T(1e-15)) where {T<:Real}

Square root with non-negative clamping to prevent domain errors near machine zero.
"""
@inline function safe_sqrt(x::T, eps::T=T(1e-15)) where {T<:Real}
    return sqrt(max(x, eps))
end

"""
    clamped_div(num::T, den::T, eps::T=T(1e-15)) where {T<:Real}

Division with sign-preserving regularized denominator to avoid division by zero.
"""
@inline function clamped_div(num::T, den::T, eps::T=T(1e-15)) where {T<:Real}
    if abs(den) < eps
        reg_den = den >= 0 ? eps : -eps
        return num / reg_den
    end
    return num / den
end

"""
    log_sum_exp(v::AbstractVector{T}) where {T<:Real}

Numerically stable computation of log(sum(exp(v_i))).
"""
function log_sum_exp(v::AbstractVector{T}) where {T<:Real}
    isempty(v) && return T(-Inf)
    m = maximum(v)
    m == -Inf && return T(-Inf)
    s = zero(T)
    for x in v
        s += exp(x - m)
    end
    return m + log(s)
end

"""
    softmax!(out::AbstractVector{T}, v::AbstractVector{T}) where {T<:Real}

In-place stable softmax transformation.
"""
function softmax!(out::AbstractVector{T}, v::AbstractVector{T}) where {T<:Real}
    @assert length(out) == length(v)
    m = maximum(v)
    s = zero(T)
    @inbounds for i in eachindex(v)
        out[i] = exp(v[i] - m)
        s += out[i]
    end
    inv_s = inv(s)
    @inbounds for i in eachindex(out)
        out[i] *= inv_s
    end
    return out
end

function softmax(v::AbstractVector{T}) where {T<:Real}
    out = similar(v)
    return softmax!(out, v)
end

"""
    horner(coeffs::AbstractVector{T}, x::T) where {T<:Number}

Evaluate polynomial P(x) = c_1 + c_2 x + ... + c_n x^(n-1) using Horner's rule.
"""
function horner(coeffs::AbstractVector{T}, x::T) where {T<:Number}
    n = length(coeffs)
    n == 0 && return zero(T)
    val = coeffs[n]
    @inbounds for i in (n-1):-1:1
        val = muladd(val, x, coeffs[i])
    end
    return val
end

"""
    is_symmetric_matrix(A::AbstractMatrix{T}, tol::T=T(1e-10)) where {T<:Real}

Verify whether matrix A is numerically symmetric within tolerance.
"""
function is_symmetric_matrix(A::AbstractMatrix{T}, tol::T=T(1e-10)) where {T<:Real}
    r, c = size(A)
    r == c || return false
    @inbounds for j in 1:c
        for i in (j+1):r
            if abs(A[i, j] - A[j, i]) > tol
                return false
            end
        end
    end
    return true
end

"""
    is_skew_symmetric(A::AbstractMatrix{T}, tol::T=T(1e-10)) where {T<:Real}

Check if A is skew-symmetric: A^T = -A.
"""
function is_skew_symmetric(A::AbstractMatrix{T}, tol::T=T(1e-10)) where {T<:Real}
    r, c = size(A)
    r == c || return false
    @inbounds for j in 1:c
        if abs(A[j, j]) > tol
            return false
        end
        for i in (j+1):r
            if abs(A[i, j] + A[j, i]) > tol
                return false
            end
        end
    end
    return true
end

"""
    project_psd(A::AbstractMatrix{T}, min_eigenval::T=T(1e-12)) where {T<:Real}

Project a symmetric matrix onto the cone of Positive Semi-Definite (PSD) matrices
by clamping negative eigenvalues to min_eigenval.
"""
function project_psd(A::AbstractMatrix{T}, min_eigenval::T=T(1e-12)) where {T<:Real}
    @assert is_symmetric_matrix(A, T(1e-6)) "Input must be symmetric"
    F = eigen(Symmetric(A))
    clamped_vals = max.(F.values, min_eigenval)
    return F.vectors * Diagonal(clamped_vals) * F.vectors'
end

"""
    gram_schmidt(V::AbstractMatrix{T}) where {T<:Real}

Modified Gram-Schmidt orthogonalization producing orthonormal columns.
"""
function gram_schmidt(V::AbstractMatrix{T}) where {T<:Real}
    m, n = size(V)
    Q = copy(V)
    for j in 1:n
        v = view(Q, :, j)
        for i in 1:(j-1)
            u = view(Q, :, i)
            proj = dot(u, v)
            v .-= proj .* u
        end
        nrm = norm(v)
        if nrm > T(1e-14)
            v ./= nrm
        else
            fill!(v, zero(T))
        end
    end
    return Q
end

"""
    numerical_jacobian(f, x::AbstractVector{T}, h::T=T(1e-7)) where {T<:Real}

Finite difference Jacobian estimation via central differences.
Used as an invariant baseline to test automatic differentiation.
"""
function numerical_jacobian(f, x::AbstractVector{T}, h::T=T(1e-7)) where {T<:Real}
    fx = f(x)
    m = length(fx)
    n = length(x)
    J = zeros(T, m, n)
    x_perturb = copy(x)
    two_h = 2 * h

    for j in 1:n
        orig = x[j]
        x_perturb[j] = orig + h
        f_plus = f(x_perturb)

        x_perturb[j] = orig - h
        f_minus = f(x_perturb)

        x_perturb[j] = orig
        for i in 1:m
            J[i, j] = (f_plus[i] - f_minus[i]) / two_h
        end
    end
    return J
end

"""
    numerical_gradient(f, x::AbstractVector{T}, h::T=T(1e-7)) where {T<:Real}

Finite difference gradient estimation for scalar valued functions.
"""
function numerical_gradient(f, x::AbstractVector{T}, h::T=T(1e-7)) where {T<:Real}
    n = length(x)
    grad = zeros(T, n)
    x_pert = copy(x)
    two_h = 2 * h

    for i in 1:n
        orig = x[i]
        x_pert[i] = orig + h
        fp = f(x_pert)

        x_pert[i] = orig - h
        fm = f(x_pert)

        x_pert[i] = orig
        grad[i] = (fp - fm) / two_h
    end
    return grad
end

"""
    numerical_hessian(f, x::AbstractVector{T}, h::T=T(1e-4)) where {T<:Real}

Finite difference Hessian estimation for scalar functions using second-order stencils.
"""
function numerical_hessian(f, x::AbstractVector{T}, h::T=T(1e-4)) where {T<:Real}
    n = length(x)
    H = zeros(T, n, n)
    f0 = f(x)
    xp = copy(x)

    # Diagonal elements: (f(x+h*e_i) - 2f(x) + f(x-h*e_i)) / h^2
    for i in 1:n
        orig_i = x[i]
        xp[i] = orig_i + h
        fp = f(xp)
        xp[i] = orig_i - h
        fm = f(xp)
        xp[i] = orig_i
        H[i, i] = (fp - 2*f0 + fm) / (h * h)
    end

    # Off-diagonal elements: (f(x+h_i+h_j) - f(x+h_i-h_j) - f(x-h_i+h_j) + f(x-h_i-h_j)) / (4 h^2)
    four_h2 = 4 * h * h
    for i in 1:n
        for j in (i+1):n
            orig_i, orig_j = x[i], x[j]

            xp[i] = orig_i + h; xp[j] = orig_j + h
            fpp = f(xp)
            xp[i] = orig_i + h; xp[j] = orig_j - h
            fpm = f(xp)
            xp[i] = orig_i - h; xp[j] = orig_j + h
            fmp = f(xp)
            xp[i] = orig_i - h; xp[j] = orig_j - h
            fmm = f(xp)

            xp[i] = orig_i; xp[j] = orig_j
            val = (fpp - fpm - fmp + fmm) / four_h2
            H[i, j] = val
            H[j, i] = val
        end
    end
    return H
end
