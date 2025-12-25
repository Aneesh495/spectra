"""
    Spectra.Dynamics.NeuralODE

Neural Ordinary Differential Equations (Neural ODEs) and Augmented Neural ODEs (ANODEs).
Models continuous-depth transformations dz/dt = f_theta(z, t) trained via continuous adjoint sensitivity.
"""

struct NeuralODEVectorField
    in_dim::Int
    hidden_dim::Int
    out_dim::Int
    W1::Matrix{Float64}
    b1::Vector{Float64}
    W2::Matrix{Float64}
    b2::Vector{Float64}

    function NeuralODEVectorField(in_d::Int, hidden_d::Int, out_d::Int)
        scale1 = sqrt(2.0 / (in_d + hidden_d))
        W1 = randn(hidden_d, in_d) .* scale1
        b1 = zeros(Float64, hidden_d)

        scale2 = sqrt(2.0 / (hidden_d + out_d))
        W2 = randn(out_d, hidden_d) .* scale2
        b2 = zeros(Float64, out_d)

        new(in_d, hidden_d, out_d, W1, b1, W2, b2)
    end
end

function get_params(vf::NeuralODEVectorField)::Vector{Float64}
    return [vec(vf.W1); vf.b1; vec(vf.W2); vf.b2]
end

function set_params!(vf::NeuralODEVectorField, theta::Vector{Float64})
    h, d = vf.hidden_dim, vf.in_dim
    o = vf.out_dim

    idx1 = h * d
    vf.W1 .= reshape(theta[1:idx1], (h, d))

    idx2 = idx1 + h
    vf.b1 .= theta[(idx1 + 1):idx2]

    idx3 = idx2 + o * h
    vf.W2 .= reshape(theta[(idx2 + 1):idx3], (o, h))

    idx4 = idx3 + o
    vf.b2 .= theta[(idx3 + 1):idx4]
    return vf
end

function (vf::NeuralODEVectorField)(z::AbstractVector{T}, t::Real=0.0)::Vector{T} where {T}
    # Two-layer MLP: W2 * tanh(W1 * z + b1) + b2
    h = tanh.(vf.W1 * z .+ vf.b1)
    return vf.W2 * h .+ vf.b2
end

# Vector field evaluation from explicit parameter vector theta
function eval_vector_field_params(z::AbstractVector{T}, theta::AbstractVector{S},
                                  in_d::Int, hidden_d::Int, out_d::Int) where {T, S}
    idx1 = hidden_d * in_d
    W1 = reshape(theta[1:idx1], (hidden_d, in_d))

    idx2 = idx1 + hidden_d
    b1 = theta[(idx1 + 1):idx2]

    idx3 = idx2 + out_d * hidden_d
    W2 = reshape(theta[(idx2 + 1):idx3], (out_d, hidden_d))

    idx4 = idx3 + out_d
    b2 = theta[(idx3 + 1):idx4]

    h = tanh.(W1 * z .+ b1)
    return W2 * h .+ b2
end

struct NeuralODE
    vf::NeuralODEVectorField
    tspan::Tuple{Float64, Float64}
    atol::Float64
    rtol::Float64

    function NeuralODE(dim::Int, hidden_dim::Int; tspan::Tuple{Float64, Float64}=(0.0, 1.0),
                       atol::Float64=1e-6, rtol::Float64=1e-5)
        vf = NeuralODEVectorField(dim, hidden_dim, dim)
        new(vf, tspan, atol, rtol)
    end
end

"""
    forward(node::NeuralODE, z0::Vector{Float64}) -> Vector{Float64}

Forward execution of Neural ODE: integrate dz/dt = f(z, theta, t) from t0 to tf.
"""
function forward(node::NeuralODE, z0::Vector{Float64})::Vector{Float64}
    f = (z, p, t) -> node.vf(z, t)
    sol = integrate_ode(f, z0, node.tspan; atol=node.atol, rtol=node.rtol)
    return sol.u[end]
end

"""
    train_step!(node::NeuralODE, z0::Vector{Float64}, z_target::Vector{Float64};
                lr::Float64=0.01) -> Float64

Single training step using adjoint sensitivity analysis and gradient descent.
"""
function train_step!(node::NeuralODE, z0::Vector{Float64}, z_target::Vector{Float64};
                     lr::Float64=0.01)::Float64
    theta = get_params(node.vf)
    in_d, hid_d, out_d = node.vf.in_dim, node.vf.hidden_dim, node.vf.out_dim

    f = (z, th, t) -> eval_vector_field_params(z, th, in_d, hid_d, out_d)
    loss_fn = z_final -> 0.5 * sum((z_final .- z_target) .^ 2)

    adj = adjoint_sensitivity(f, loss_fn, z0, theta, node.tspan; atol=node.atol, rtol=node.rtol)

    # Gradient update
    theta .-= lr .* adj.grad_theta
    set_params!(node.vf, theta)

    return adj.loss
end

# ====================================================================
# Augmented Neural ODE (ANODE)
# ====================================================================

struct AugmentedNeuralODE
    base_dim::Int
    augment_dim::Int
    node::NeuralODE

    function AugmentedNeuralODE(base_dim::Int, augment_dim::Int, hidden_dim::Int;
                                tspan::Tuple{Float64, Float64}=(0.0, 1.0))
        total_dim = base_dim + augment_dim
        node = NeuralODE(total_dim, hidden_dim; tspan=tspan)
        new(base_dim, augment_dim, node)
    end
end

"""
    forward(anode::AugmentedNeuralODE, x::Vector{Float64}) -> Vector{Float64}

Forward evaluation of Augmented Neural ODE by padding input with zero augmentations.
"""
function forward(anode::AugmentedNeuralODE, x::Vector{Float64})::Vector{Float64}
    @assert length(x) == anode.base_dim "Input length must match base_dim"
    # Zero padding
    z0 = [x; zeros(Float64, anode.augment_dim)]
    z_final = forward(anode.node, z0)
    # Return base state projection
    return z_final[1:anode.base_dim]
end
