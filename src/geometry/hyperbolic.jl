"""
    Spectra.Geometry.Hyperbolic

Hyperbolic spaces of constant negative sectional curvature K = -c.
Implements the Poincare Ball model via Mobius gyrovector operations and the
Lorentz / Hyperboloid model with Minkowski indefinite metric.
"""

# ====================================================================
# Poincare Ball Model
# ====================================================================

struct PoincareBall{N} <: AbstractManifold{N}
    c::Float64 # Curvature constant, default c = 1.0 (K = -1)

    function PoincareBall{N}(c::Float64=1.0) where {N}
        @assert c > 0.0 "Curvature parameter c must be strictly positive"
        new{N}(c)
    end
end

PoincareBall(n::Int, c::Float64=1.0) = PoincareBall{n}(c)

manifold_dim(::PoincareBall{N}) where {N} = N
ambient_dim(::PoincareBall{N}) where {N} = N

function conformal_factor(M::PoincareBall{N}, x::Vector{Float64})::Float64 where {N}
    sq_norm = dot(x, x)
    return 2.0 / (1.0 - M.c * sq_norm)
end

function check_point(M::PoincareBall{N}, p::Vector{Float64}; tol::Float64=1e-8)::Bool where {N}
    length(p) == N || return false
    return dot(p, p) < (1.0 / M.c) - tol
end

function check_tangent(::PoincareBall{N}, p::Vector{Float64}, v::Vector{Float64}; tol::Float64=1e-8)::Bool where {N}
    return length(v) == N
end

function inner_product(M::PoincareBall{N}, p::Vector{Float64}, u::Vector{Float64}, v::Vector{Float64})::Float64 where {N}
    lambda = conformal_factor(M, p)
    return (lambda^2) * dot(u, v)
end

function project_tangent(::PoincareBall{N}, p::Vector{Float64}, v::Vector{Float64})::Vector{Float64} where {N}
    return copy(v) # Entire R^N is tangent to open ball
end

"""
    mobius_add(M::PoincareBall, x::Vector{Float64}, y::Vector{Float64}) -> Vector{Float64}

Mobius gyrovector addition in the Poincare ball.
"""
function mobius_add(M::PoincareBall{N}, x::Vector{Float64}, y::Vector{Float64})::Vector{Float64} where {N}
    c = M.c
    x2 = dot(x, x)
    y2 = dot(y, y)
    xy = dot(x, y)

    num = (1.0 + 2.0 * c * xy + c * y2) .* x .+ (1.0 - c * x2) .* y
    den = 1.0 + 2.0 * c * xy + (c^2) * x2 * y2

    return num ./ max(den, 1e-15)
end

function exp_map(M::PoincareBall{N}, p::Vector{Float64}, v::Vector{Float64})::Vector{Float64} where {N}
    v_norm = norm(v)
    if v_norm < 1e-15
        return copy(p)
    end
    c = M.c
    sqrt_c = sqrt(c)
    lambda_p = conformal_factor(M, p)

    coeff = tanh(sqrt_c * lambda_p * v_norm * 0.5) / (sqrt_c * v_norm)
    second_term = coeff .* v

    return mobius_add(M, p, second_term)
end

function log_map(M::PoincareBall{N}, p::Vector{Float64}, q::Vector{Float64})::Vector{Float64} where {N}
    diff = mobius_add(M, -p, q)
    diff_norm = norm(diff)
    if diff_norm < 1e-15
        return zeros(Float64, N)
    end
    c = M.c
    sqrt_c = sqrt(c)
    lambda_p = conformal_factor(M, p)

    # atanh argument
    arg = clamp(sqrt_c * diff_norm, -1.0 + 1e-12, 1.0 - 1e-12)
    coeff = (2.0 / (sqrt_c * lambda_p)) * atanh(arg) / diff_norm

    return coeff .* diff
end

function geodesic_distance(M::PoincareBall{N}, p::Vector{Float64}, q::Vector{Float64})::Float64 where {N}
    diff = mobius_add(M, -p, q)
    diff_norm = norm(diff)
    c = M.c
    sqrt_c = sqrt(c)
    arg = clamp(sqrt_c * diff_norm, -1.0 + 1e-12, 1.0 - 1e-12)
    return (2.0 / sqrt_c) * atanh(arg)
end

# ====================================================================
# Lorentz / Hyperboloid Model
# ====================================================================

struct LorentzHyperboloid{N} <: AbstractManifold{N}
    c::Float64

    function LorentzHyperboloid{N}(c::Float64=1.0) where {N}
        @assert c > 0.0 "Curvature parameter c must be positive"
        new{N}(c)
    end
end

LorentzHyperboloid(n::Int, c::Float64=1.0) = LorentzHyperboloid{n}(c)

manifold_dim(::LorentzHyperboloid{N}) where {N} = N
ambient_dim(::LorentzHyperboloid{N}) where {N} = N + 1

"""
    minkowski_dot(x::Vector{Float64}, y::Vector{Float64}) -> Float64

Minkowski indefinite inner product <x, y>_{-1} = -x_1 y_1 + sum_{i=2}^{N+1} x_i y_i.
"""
function minkowski_dot(x::Vector{Float64}, y::Vector{Float64})::Float64
    s = -x[1] * y[1]
    @inbounds for i in 2:length(x)
        s += x[i] * y[i]
    end
    return s
end

function check_point(M::LorentzHyperboloid{N}, p::Vector{Float64}; tol::Float64=1e-8)::Bool where {N}
    length(p) == N + 1 || return false
    p[1] > 0.0 || return false # Forward time sheet
    return abs(minkowski_dot(p, p) - (-1.0 / M.c)) < tol
end

function project_tangent(M::LorentzHyperboloid{N}, p::Vector{Float64}, v::Vector{Float64})::Vector{Float64} where {N}
    # Tangent space condition: <p, v>_{-1} = 0
    # v_proj = v + c <p, v>_{-1} p
    return v .+ M.c * minkowski_dot(p, v) .* p
end

function exp_map(M::LorentzHyperboloid{N}, p::Vector{Float64}, v::Vector{Float64})::Vector{Float64} where {N}
    v_norm_sq = minkowski_dot(v, v)
    if v_norm_sq < 1e-15
        return copy(p)
    end
    sqrt_c = sqrt(M.c)
    theta = sqrt(v_norm_sq) * sqrt_c
    return cosh(theta) .* p .+ (sinh(theta) / (theta * sqrt_c)) .* v
end

function log_map(M::LorentzHyperboloid{N}, p::Vector{Float64}, q::Vector{Float64})::Vector{Float64} where {N}
    inner = minkowski_dot(p, q)
    c = M.c
    # -c <p, q>_{-1} >= 1
    arg = max(1.0, -c * inner)
    dist = acosh(arg) / sqrt(c)
    if dist < 1e-15
        return zeros(Float64, N + 1)
    end
    coeff = dist / sinh(dist * sqrt(c))
    return coeff .* (q .+ c * inner .* p)
end

function geodesic_distance(M::LorentzHyperboloid{N}, p::Vector{Float64}, q::Vector{Float64})::Float64 where {N}
    arg = max(1.0, -M.c * minkowski_dot(p, q))
    return acosh(arg) / sqrt(M.c)
end

# Conversion between Poincare Ball and Lorentz Hyperboloid
function poincare_to_lorentz(x::Vector{Float64}, c::Float64=1.0)::Vector{Float64}
    x2 = dot(x, x)
    den = 1.0 - c * x2
    p0 = (1.0 + c * x2) / (sqrt(c) * den)
    p_rest = (2.0 / den) .* x
    return [p0; p_rest]
end

function lorentz_to_poincare(p::Vector{Float64}, c::Float64=1.0)::Vector{Float64}
    p0 = p[1]
    p_rest = p[2:end]
    den = 1.0 + sqrt(c) * p0
    return p_rest ./ den
end
