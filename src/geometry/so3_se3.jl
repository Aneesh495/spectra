"""
    Spectra.Geometry.SO3SE3

Lie groups SO(3) (3D rotations) and SE(3) (rigid body transformations).
Implements hat/vee maps, Rodrigues rotation formula, left Jacobians, and Lie algebra generators.
"""

# ====================================================================
# SO(3) Lie Group & so(3) Lie Algebra
# ====================================================================

struct SO3Group <: AbstractLieGroup end

manifold_dim(::SO3Group) = 3
ambient_dim(::SO3Group) = 9

"""
    hat_so3(w::Vector{Float64}) -> Matrix{Float64}

Isomorphism mapping w in R^3 to skew-symmetric matrix [w]_x in so(3).
"""
function hat_so3(w::Vector{Float64})::Matrix{Float64}
    @assert length(w) == 3 "Vector must have length 3"
    return [
         0.0   -w[3]   w[2];
        w[3]    0.0   -w[1];
       -w[2]   w[1]    0.0
    ]
end

"""
    vee_so3(W::Matrix{Float64}) -> Vector{Float64}

Inverse isomorphism mapping skew-symmetric matrix W in so(3) to w in R^3.
"""
function vee_so3(W::Matrix{Float64})::Vector{Float64}
    @assert size(W) == (3, 3) "Matrix must be 3 x 3"
    return [W[3, 2], W[1, 3], W[2, 1]]
end

"""
    exp_so3(w::Vector{Float64}) -> Matrix{Float64}

Matrix exponential mapping axis-angle vector w in so(3) to rotation matrix R in SO(3)
via Rodrigues formula: R = I + (sin theta / theta) [w]_x + ((1 - cos theta) / theta^2) [w]_x^2.
"""
function exp_so3(w::Vector{Float64})::Matrix{Float64}
    theta = norm(w)
    I3 = Matrix{Float64}(I, 3, 3)
    if theta < 1e-12
        # Taylor expansion near identity: I + [w]_x + 0.5 [w]_x^2
        W = hat_so3(w)
        return I3 .+ W .+ 0.5 .* (W * W)
    end
    W = hat_so3(w)
    s = sin(theta) / theta
    c = (1.0 - cos(theta)) / (theta * theta)
    return I3 .+ s .* W .+ c .* (W * W)
end

"""
    log_so3(R::Matrix{Float64}) -> Vector{Float64}

Matrix logarithm mapping rotation matrix R in SO(3) to axis-angle vector w in so(3).
"""
function log_so3(R::Matrix{Float64})::Vector{Float64}
    @assert size(R) == (3, 3) "Matrix must be 3 x 3"
    cos_theta = clamp(0.5 * (tr(R) - 1.0), -1.0, 1.0)
    theta = acos(cos_theta)
    if theta < 1e-12
        return 0.5 .* [R[3, 2] - R[2, 3], R[1, 3] - R[3, 1], R[2, 1] - R[1, 2]]
    end
    scale = theta / (2.0 * sin(theta))
    return scale .* [R[3, 2] - R[2, 3], R[1, 3] - R[3, 1], R[2, 1] - R[1, 2]]
end

function check_point(::SO3Group, R::Matrix{Float64}; tol::Float64=1e-8)::Bool
    size(R) == (3, 3) || return false
    is_ortho = norm(R' * R - I) < tol
    is_det1 = abs(det(R) - 1.0) < tol
    return is_ortho && is_det1
end

# ====================================================================
# SE(3) Lie Group & se(3) Lie Algebra
# ====================================================================

struct SE3Group <: AbstractLieGroup end

manifold_dim(::SE3Group) = 6
ambient_dim(::SE3Group) = 16

"""
    hat_se3(xi::Vector{Float64}) -> Matrix{Float64}

Map 6D twist xi = [v; w] to 4 x 4 matrix in se(3).
"""
function hat_se3(xi::Vector{Float64})::Matrix{Float64}
    @assert length(xi) == 6 "Twist vector must have length 6"
    v = xi[1:3]
    w = xi[4:6]
    W = hat_so3(w)
    Xi = zeros(Float64, 4, 4)
    Xi[1:3, 1:3] = W
    Xi[1:3, 4] = v
    return Xi
end

"""
    vee_se3(Xi::Matrix{Float64}) -> Vector{Float64}

Map 4 x 4 matrix in se(3) to 6D twist xi = [v; w].
"""
function vee_se3(Xi::Matrix{Float64})::Vector{Float64}
    @assert size(Xi) == (4, 4) "Matrix must be 4 x 4"
    v = Xi[1:3, 4]
    w = vee_so3(Xi[1:3, 1:3])
    return [v; w]
end

"""
    left_jacobian_so3(w::Vector{Float64}) -> Matrix{Float64}

Compute left Jacobian J_l(w) of SO(3).
"""
function left_jacobian_so3(w::Vector{Float64})::Matrix{Float64}
    theta = norm(w)
    I3 = Matrix{Float64}(I, 3, 3)
    W = hat_so3(w)
    if theta < 1e-12
        return I3 .+ 0.5 .* W
    end
    c1 = (1.0 - cos(theta)) / (theta * theta)
    c2 = (theta - sin(theta)) / (theta * theta * theta)
    return I3 .+ c1 .* W .+ c2 .* (W * W)
end

"""
    exp_se3(xi::Vector{Float64}) -> Matrix{Float64}

Matrix exponential mapping 6D twist xi = [v; w] in se(3) to 4 x 4 transformation matrix T in SE(3).
"""
function exp_se3(xi::Vector{Float64})::Matrix{Float64}
    @assert length(xi) == 6 "Twist vector must have length 6"
    v = xi[1:3]
    w = xi[4:6]

    R = exp_so3(w)
    Jl = left_jacobian_so3(w)
    t = Jl * v

    T = Matrix{Float64}(I, 4, 4)
    T[1:3, 1:3] = R
    T[1:3, 4] = t
    return T
end

"""
    log_se3(T::Matrix{Float64}) -> Vector{Float64}

Matrix logarithm mapping 4 x 4 transformation matrix T in SE(3) to 6D twist xi in se(3).
"""
function log_se3(T::Matrix{Float64})::Vector{Float64}
    @assert size(T) == (4, 4) "Matrix must be 4 x 4"
    R = T[1:3, 1:3]
    t = T[1:3, 4]

    w = log_so3(R)
    Jl = left_jacobian_so3(w)
    v = Jl \ t

    return [v; w]
end

function check_point(::SE3Group, T::Matrix{Float64}; tol::Float64=1e-8)::Bool
    size(T) == (4, 4) || return false
    R = T[1:3, 1:3]
    bottom = T[4, :]
    check_point(SO3Group(), R; tol=tol) || return false
    return norm(bottom - [0.0, 0.0, 0.0, 1.0]) < tol
end
