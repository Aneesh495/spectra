"""
    Spectra.Geometry.Sphere

Hypersphere S^n embedded in R^{n+1} with constant positive sectional curvature K = +1.
Implements exact exponential maps, logarithmic maps, and parallel transport along great circles.
"""

struct Sphere{N} <: AbstractManifold{N}
    # N is the intrinsic dimension; ambient dimension is N + 1
end

Sphere(n::Int) = Sphere{n}()

manifold_dim(::Sphere{N}) where {N} = N
ambient_dim(::Sphere{N}) where {N} = N + 1

function check_point(::Sphere{N}, p::Vector{Float64}; tol::Float64=1e-8)::Bool where {N}
    length(p) == N + 1 || return false
    return abs(norm(p) - 1.0) < tol
end

function check_tangent(::Sphere{N}, p::Vector{Float64}, v::Vector{Float64}; tol::Float64=1e-8)::Bool where {N}
    length(v) == N + 1 || return false
    return abs(dot(p, v)) < tol
end

function project_tangent(::Sphere{N}, p::Vector{Float64}, v::Vector{Float64})::Vector{Float64} where {N}
    return v .- dot(p, v) .* p
end

function exp_map(::Sphere{N}, p::Vector{Float64}, v::Vector{Float64})::Vector{Float64} where {N}
    theta = norm(v)
    if theta < 1e-15
        return p ./ norm(p)
    end
    u = v ./ theta
    res = cos(theta) .* p .+ sin(theta) .* u
    return res ./ norm(res)
end

function retraction(::Sphere{N}, p::Vector{Float64}, v::Vector{Float64})::Vector{Float64} where {N}
    # First-order metric projection retraction: (p + v) / ||p + v||
    res = p .+ v
    return res ./ norm(res)
end

function log_map(::Sphere{N}, p::Vector{Float64}, q::Vector{Float64})::Vector{Float64} where {N}
    cos_theta = clamp(dot(p, q), -1.0, 1.0)
    theta = acos(cos_theta)
    if theta < 1e-15
        return zeros(Float64, N + 1)
    end
    sin_theta = sin(theta)
    u = (q .- cos_theta .* p) ./ sin_theta
    return theta .* u
end

function geodesic_distance(::Sphere{N}, p::Vector{Float64}, q::Vector{Float64})::Float64 where {N}
    cos_theta = clamp(dot(p, q), -1.0, 1.0)
    return acos(cos_theta)
end

function parallel_transport(::Sphere{N}, p::Vector{Float64}, q::Vector{Float64}, v::Vector{Float64})::Vector{Float64} where {N}
    cos_theta = clamp(dot(p, q), -1.0, 1.0)
    theta = acos(cos_theta)
    if theta < 1e-15
        return copy(v)
    end
    log_pq = log_map(Sphere{N}(), p, q)
    log_qp = log_map(Sphere{N}(), q, p)
    dist2 = theta * theta
    proj = dot(log_pq, v) / dist2
    return v .- proj .* (log_pq .+ log_qp)
end

function random_point(::Sphere{N})::Vector{Float64} where {N}
    p = randn(N + 1)
    return p ./ norm(p)
end
