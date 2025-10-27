"""
    Spectra.Geometry.Manifold

Abstract Riemannian manifold foundation, intrinsic metrics, tangent space projections,
affine connections, exponential and logarithmic maps, and parallel transport operators.
"""

# Abstract Manifold Interface Definition
"""
    manifold_dim(M::AbstractManifold) -> Int

Intrinsic topological dimension of the manifold.
"""
function manifold_dim end

"""
    ambient_dim(M::AbstractManifold) -> Int

Dimension of the ambient Euclidean embedding space.
"""
function ambient_dim end

"""
    inner_product(M::AbstractManifold, p, u, v) -> Float64

Riemannian inner product g_p(u, v) on the tangent space T_p M.
"""
function inner_product(M::AbstractManifold, p, u, v)
    # Default Euclidean fallback for embedded submanifolds
    return dot(u, v)
end

"""
    norm_tangent(M::AbstractManifold, p, v) -> Float64

Riemannian norm of tangent vector v in T_p M: ||v||_p = sqrt(g_p(v, v)).
"""
function norm_tangent(M::AbstractManifold, p, v)::Float64
    val = inner_product(M, p, v, v)
    return safe_sqrt(val)
end

"""
    project_tangent(M::AbstractManifold, p, v)

Orthogonal projection of an ambient vector v onto the tangent space T_p M.
"""
function project_tangent end

"""
    check_point(M::AbstractManifold, p; tol::Float64=1e-8) -> Bool

Verify whether point p lies on the manifold within numerical tolerance.
"""
function check_point end

"""
    check_tangent(M::AbstractManifold, p, v; tol::Float64=1e-8) -> Bool

Verify whether vector v lies in the tangent space T_p M.
"""
function check_tangent end

"""
    exp_map(M::AbstractManifold, p, v)

Riemannian exponential map Exp_p(v) mapping tangent vector v in T_p M
to the manifold point along the geodesic gamma(1) with gamma(0) = p and gamma'(0) = v.
"""
function exp_map end

"""
    log_map(M::AbstractManifold, p, q)

Riemannian logarithmic map Log_p(q) yielding tangent vector v in T_p M
such that Exp_p(v) = q.
"""
function log_map end

"""
    geodesic_distance(M::AbstractManifold, p, q) -> Float64

Intrinsic geodesic distance d_M(p, q) between points p and q.
"""
function geodesic_distance(M::AbstractManifold, p, q)::Float64
    v = log_map(M, p, q)
    return norm_tangent(M, p, v)
end

"""
    parallel_transport(M::AbstractManifold, p, q, v)

Parallel transport of tangent vector v in T_p M to T_q M along the minimizing geodesic.
Preserves the Riemannian metric: <u, v>_p = <PT(u), PT(v)>_q.
"""
function parallel_transport end

"""
    retraction(M::AbstractManifold, p, v)

First-order retraction mapping T_p M to M, satisfying R_p(0) = p and d/dt R_p(t v)|_{t=0} = v.
Provides high-efficiency approximation to the exponential map.
"""
function retraction(M::AbstractManifold, p, v)
    # Default to exact exponential map when available
    return exp_map(M, p, v)
end

"""
    geodesic(M::AbstractManifold, p, q, t::Float64)

Evaluate geodesic curve gamma(t) connecting p at t=0 and q at t=1.
"""
function geodesic(M::AbstractManifold, p, q, t::Float64)
    v = log_map(M, p, q)
    return exp_map(M, p, t .* v)
end
