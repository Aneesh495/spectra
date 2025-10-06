"""
    Spectra.Math.Dual

Forward-mode automatic differentiation engine utilizing multi-directional dual numbers
and hyper-dual numbers for exact gradient, Jacobian, and Hessian calculations to machine precision.
"""

# First-Order Dual Number with N directional derivatives
struct Dual{T<:Real, N} <: Real
    val::T
    partials::NTuple{N, T}

    function Dual{T, N}(val::T, partials::NTuple{N, T}) where {T<:Real, N}
        new{T, N}(val, partials)
    end
end

Dual(val::T, partials::NTuple{N, T}) where {T<:Real, N} = Dual{T, N}(val, partials)
Dual(val::Real, partials::NTuple{N, Real}) where {N} = Dual(promote(val, partials...)...)
Dual(val::T, partial::T) where {T<:Real} = Dual{T, 1}(val, (partial,))
Dual(val::T) where {T<:Real} = Dual{T, 0}(val, ())

# Promotion and conversion rules
Base.promote_rule(::Type{Dual{T, N}}, ::Type{S}) where {T<:Real, N, S<:Real} = Dual{promote_type(T, S), N}
Base.promote_rule(::Type{Dual{T, N}}, ::Type{Dual{S, N}}) where {T<:Real, S<:Real, N} = Dual{promote_type(T, S), N}

Base.convert(::Type{Dual{T, N}}, x::Real) where {T<:Real, N} = Dual{T, N}(convert(T, x), ntuple(_ -> zero(T), N))
Base.convert(::Type{Dual{T, N}}, d::Dual{S, N}) where {T<:Real, S<:Real, N} = Dual{T, N}(convert(T, d.val), ntuple(i -> convert(T, d.partials[i]), N))

# Accessors
value(d::Dual) = d.val
value(x::Real) = x
partials(d::Dual) = d.partials

# Basic arithmetic operations
function Base.:+(a::Dual{T, N}, b::Dual{T, N}) where {T, N}
    return Dual{T, N}(a.val + b.val, ntuple(i -> a.partials[i] + b.partials[i], N))
end
Base.:+(a::Dual{T, N}, b::Real) where {T, N} = Dual{T, N}(a.val + convert(T, b), a.partials)
Base.:+(a::Real, b::Dual{T, N}) where {T, N} = b + a

Base.:-(a::Dual{T, N}) where {T, N} = Dual{T, N}(-a.val, ntuple(i -> -a.partials[i], N))
function Base.:-(a::Dual{T, N}, b::Dual{T, N}) where {T, N}
    return Dual{T, N}(a.val - b.val, ntuple(i -> a.partials[i] - b.partials[i], N))
end
Base.:-(a::Dual{T, N}, b::Real) where {T, N} = Dual{T, N}(a.val - convert(T, b), a.partials)
Base.:-(a::Real, b::Dual{T, N}) where {T, N} = Dual{T, N}(convert(T, a) - b.val, ntuple(i -> -b.partials[i], N))

# Product rule: (u v)' = u' v + u v'
function Base.:*(a::Dual{T, N}, b::Dual{T, N}) where {T, N}
    new_partials = ntuple(i -> muladd(a.partials[i], b.val, a.val * b.partials[i]), N)
    return Dual{T, N}(a.val * b.val, new_partials)
end
Base.:*(a::Dual{T, N}, b::Real) where {T, N} = Dual{T, N}(a.val * convert(T, b), ntuple(i -> a.partials[i] * convert(T, b), N))
Base.:*(a::Real, b::Dual{T, N}) where {T, N} = b * a

# Quotient rule: (u / v)' = (u' v - u v') / v^2
function Base.:/(a::Dual{T, N}, b::Dual{T, N}) where {T, N}
    inv_b = inv(b.val)
    inv_b2 = inv_b * inv_b
    new_val = a.val * inv_b
    new_partials = ntuple(i -> (a.partials[i] * b.val - a.val * b.partials[i]) * inv_b2, N)
    return Dual{T, N}(new_val, new_partials)
end
Base.:/(a::Dual{T, N}, b::Real) where {T, N} = Dual{T, N}(a.val / convert(T, b), ntuple(i -> a.partials[i] / convert(T, b), N))
function Base.:/(a::Real, b::Dual{T, N}) where {T, N}
    c_a = convert(T, a)
    inv_b = inv(b.val)
    inv_b2 = inv_b * inv_b
    return Dual{T, N}(c_a * inv_b, ntuple(i -> -c_a * b.partials[i] * inv_b2, N))
end

# Elementary functions via chain rule
function Base.exp(d::Dual{T, N}) where {T, N}
    e = exp(d.val)
    return Dual{T, N}(e, ntuple(i -> e * d.partials[i], N))
end

function Base.log(d::Dual{T, N}) where {T, N}
    l = log(d.val)
    inv_val = inv(d.val)
    return Dual{T, N}(l, ntuple(i -> d.partials[i] * inv_val, N))
end

function Base.sin(d::Dual{T, N}) where {T, N}
    s, c = sincos(d.val)
    return Dual{T, N}(s, ntuple(i -> c * d.partials[i], N))
end

function Base.cos(d::Dual{T, N}) where {T, N}
    s, c = sincos(d.val)
    return Dual{T, N}(c, ntuple(i -> -s * d.partials[i], N))
end

function Base.tan(d::Dual{T, N}) where {T, N}
    t = tan(d.val)
    sec2 = 1 + t * t
    return Dual{T, N}(t, ntuple(i -> sec2 * d.partials[i], N))
end

function Base.tanh(d::Dual{T, N}) where {T, N}
    th = tanh(d.val)
    sech2 = 1 - th * th
    return Dual{T, N}(th, ntuple(i -> sech2 * d.partials[i], N))
end

function Base.sqrt(d::Dual{T, N}) where {T, N}
    s = sqrt(d.val)
    inv_2s = inv(2 * s)
    return Dual{T, N}(s, ntuple(i -> d.partials[i] * inv_2s, N))
end

function Base.:^(d::Dual{T, N}, p::Integer) where {T, N}
    val_p = d.val^p
    deriv = p * (d.val^(p - 1))
    return Dual{T, N}(val_p, ntuple(i -> deriv * d.partials[i], N))
end

function Base.:^(d::Dual{T, N}, p::Real) where {T, N}
    c_p = convert(T, p)
    val_p = d.val^c_p
    deriv = c_p * (d.val^(c_p - 1))
    return Dual{T, N}(val_p, ntuple(i -> deriv * d.partials[i], N))
end

Base.isless(a::Dual, b::Dual) = a.val < b.val
Base.isless(a::Dual, b::Real) = a.val < b
Base.isless(a::Real, b::Dual) = a < b.val

Base.:<(a::Dual, b::Dual) = a.val < b.val
Base.:<(a::Dual, b::Real) = a.val < b
Base.:<(a::Real, b::Dual) = a < b.val

function relu(d::Dual{T, N}) where {T, N}
    if d.val > zero(T)
        return d
    else
        return Dual{T, N}(zero(T), ntuple(_ -> zero(T), N))
    end
end

# ====================================================================
# Hyper-Dual Numbers for Exact Second-Order Differentiation & Hessians
# ====================================================================
"""
    HyperDual{T<:Real}

Hyper-dual number representation x + eps1 * x1 + eps2 * x2 + eps1*eps2 * x12.
Calculates exact first and second directional derivatives simultaneously.
"""
struct HyperDual{T<:Real} <: Real
    val::T
    eps1::T
    eps2::T
    eps12::T
end

HyperDual(v::T) where {T<:Real} = HyperDual{T}(v, zero(T), zero(T), zero(T))
HyperDual(v::Real, e1::Real, e2::Real, e12::Real) = HyperDual(promote(v, e1, e2, e12)...)

Base.promote_rule(::Type{HyperDual{T}}, ::Type{S}) where {T<:Real, S<:Real} = HyperDual{promote_type(T, S)}
Base.convert(::Type{HyperDual{T}}, x::Real) where {T<:Real} = HyperDual{T}(convert(T, x), zero(T), zero(T), zero(T))

# Hyper-Dual Arithmetic
function Base.:+(a::HyperDual{T}, b::HyperDual{T}) where {T}
    return HyperDual{T}(a.val + b.val, a.eps1 + b.eps1, a.eps2 + b.eps2, a.eps12 + b.eps12)
end
Base.:+(a::HyperDual{T}, b::Real) where {T} = HyperDual{T}(a.val + convert(T, b), a.eps1, a.eps2, a.eps12)
Base.:+(a::Real, b::HyperDual{T}) where {T} = b + a

Base.:-(a::HyperDual{T}) where {T} = HyperDual{T}(-a.val, -a.eps1, -a.eps2, -a.eps12)
function Base.:-(a::HyperDual{T}, b::HyperDual{T}) where {T}
    return HyperDual{T}(a.val - b.val, a.eps1 - b.eps1, a.eps2 - b.eps2, a.eps12 - b.eps12)
end
Base.:-(a::HyperDual{T}, b::Real) where {T} = HyperDual{T}(a.val - convert(T, b), a.eps1, a.eps2, a.eps12)
Base.:-(a::Real, b::HyperDual{T}) where {T} = HyperDual{T}(convert(T, a) - b.val, -b.eps1, -b.eps2, -b.eps12)

function Base.:*(a::HyperDual{T}, b::HyperDual{T}) where {T}
    val = a.val * b.val
    e1 = muladd(a.val, b.eps1, a.eps1 * b.val)
    e2 = muladd(a.val, b.eps2, a.eps2 * b.val)
    e12 = a.val * b.eps12 + a.eps1 * b.eps2 + a.eps2 * b.eps1 + a.eps12 * b.val
    return HyperDual{T}(val, e1, e2, e12)
end
Base.:*(a::HyperDual{T}, b::Real) where {T} = HyperDual{T}(a.val * convert(T, b), a.eps1 * convert(T, b), a.eps2 * convert(T, b), a.eps12 * convert(T, b))
Base.:*(a::Real, b::HyperDual{T}) where {T} = b * a

function Base.inv(a::HyperDual{T}) where {T}
    inv_v = inv(a.val)
    inv_v2 = inv_v * inv_v
    inv_v3 = inv_v2 * inv_v
    val = inv_v
    e1 = -a.eps1 * inv_v2
    e2 = -a.eps2 * inv_v2
    e12 = -a.eps12 * inv_v2 + 2 * a.eps1 * a.eps2 * inv_v3
    return HyperDual{T}(val, e1, e2, e12)
end
Base.:/(a::HyperDual{T}, b::HyperDual{T}) where {T} = a * inv(b)
Base.:/(a::HyperDual{T}, b::Real) where {T} = a * inv(convert(T, b))
Base.:/(a::Real, b::HyperDual{T}) where {T} = convert(T, a) * inv(b)

function Base.exp(a::HyperDual{T}) where {T}
    ev = exp(a.val)
    e1 = ev * a.eps1
    e2 = ev * a.eps2
    e12 = ev * (a.eps12 + a.eps1 * a.eps2)
    return HyperDual{T}(ev, e1, e2, e12)
end

function Base.log(a::HyperDual{T}) where {T}
    lv = log(a.val)
    inv_v = inv(a.val)
    inv_v2 = inv_v * inv_v
    e1 = a.eps1 * inv_v
    e2 = a.eps2 * inv_v
    e12 = a.eps12 * inv_v - (a.eps1 * a.eps2) * inv_v2
    return HyperDual{T}(lv, e1, e2, e12)
end

function Base.sin(a::HyperDual{T}) where {T}
    s, c = sincos(a.val)
    e1 = c * a.eps1
    e2 = c * a.eps2
    e12 = c * a.eps12 - s * (a.eps1 * a.eps2)
    return HyperDual{T}(s, e1, e2, e12)
end

function Base.cos(a::HyperDual{T}) where {T}
    s, c = sincos(a.val)
    e1 = -s * a.eps1
    e2 = -s * a.eps2
    e12 = -s * a.eps12 - c * (a.eps1 * a.eps2)
    return HyperDual{T}(c, e1, e2, e12)
end

function Base.sqrt(a::HyperDual{T}) where {T}
    s = sqrt(a.val)
    inv_2s = inv(2 * s)
    inv_4s3 = inv(4 * s * a.val)
    e1 = a.eps1 * inv_2s
    e2 = a.eps2 * inv_2s
    e12 = a.eps12 * inv_2s - (a.eps1 * a.eps2) * inv_4s3
    return HyperDual{T}(s, e1, e2, e12)
end

function Base.:^(a::HyperDual{T}, p::Integer) where {T}
    c_p = convert(T, p)
    vp = a.val^p
    d1 = c_p * (a.val^(p - 1))
    d2 = c_p * (c_p - 1) * (a.val^(p - 2))
    e1 = d1 * a.eps1
    e2 = d1 * a.eps2
    e12 = d1 * a.eps12 + d2 * (a.eps1 * a.eps2)
    return HyperDual{T}(vp, e1, e2, e12)
end

function Base.:^(a::HyperDual{T}, p::Real) where {T}
    c_p = convert(T, p)
    vp = a.val^c_p
    d1 = c_p * (a.val^(c_p - 1))
    d2 = c_p * (c_p - 1) * (a.val^(c_p - 2))
    e1 = d1 * a.eps1
    e2 = d1 * a.eps2
    e12 = d1 * a.eps12 + d2 * (a.eps1 * a.eps2)
    return HyperDual{T}(vp, e1, e2, e12)
end

# ====================================================================
# User-Facing High-Level Forward AD API
# ====================================================================

"""
    forward_diff(f, x::T) where {T<:Real}

Compute exact derivative f'(x) via single-direction dual numbers.
"""
function forward_diff(f, x::T) where {T<:Real}
    d = Dual{T, 1}(x, (one(T),))
    res = f(d)
    return res.partials[1]
end

"""
    forward_gradient(f, x::AbstractVector{T}) where {T<:Real}

Compute exact gradient ∇f(x) using N-dimensional dual vector seeding.
"""
function forward_gradient(f, x::AbstractVector{T}) where {T<:Real}
    n = length(x)
    grad = zeros(T, n)
    for i in 1:n
        # Seed i-th direction
        dx = ntuple(k -> (k == 1 ? one(T) : zero(T)), 1)
        d_vec = [k == i ? Dual{T, 1}(x[k], (one(T),)) : Dual{T, 1}(x[k], (zero(T),)) for k in 1:n]
        res = f(d_vec)
        grad[i] = res.partials[1]
    end
    return grad
end

"""
    forward_jacobian(f, x::AbstractVector{T}) where {T<:Real}

Compute exact Jacobian matrix J_ij = ∂f_i/∂x_j via directional forward seeding.
"""
function forward_jacobian(f, x::AbstractVector{T}) where {T<:Real}
    n = length(x)
    # Evaluate at base point to find output dimensionality m
    f0 = f(x)
    m = length(f0)
    J = zeros(T, m, n)

    for j in 1:n
        d_vec = [k == j ? Dual{T, 1}(x[k], (one(T),)) : Dual{T, 1}(x[k], (zero(T),)) for k in 1:n]
        fj = f(d_vec)
        for i in 1:m
            J[i, j] = fj[i].partials[1]
        end
    end
    return J
end

"""
    forward_hessian(f, x::AbstractVector{T}) where {T<:Real}

Compute exact Hessian matrix H_ij = ∂^2 f / (∂x_i ∂x_j) via hyper-dual number pairs.
Guaranteed exact without numerical step-size cancellation errors.
"""
function forward_hessian(f, x::AbstractVector{T}) where {T<:Real}
    n = length(x)
    H = zeros(T, n, n)

    for i in 1:n
        for j in i:n
            hd_vec = [
                k == i && k == j ? HyperDual{T}(x[k], one(T), one(T), zero(T)) :
                k == i           ? HyperDual{T}(x[k], one(T), zero(T), zero(T)) :
                k == j           ? HyperDual{T}(x[k], zero(T), one(T), zero(T)) :
                                   HyperDual{T}(x[k], zero(T), zero(T), zero(T))
                for k in 1:n
            ]
            res = f(hd_vec)
            H[i, j] = res.eps12
            H[j, i] = res.eps12
        end
    end
    return H
end
