"""
    Spectra.Koopman.DeepKoopman

Deep Koopman Autoencoder neural network architecture (Lusch, Kutz & Brunton 2018).
Discovers continuous intrinsic coordinate representations in which nonlinear dynamical
systems behave linearly.
Comprises:
1. Encoder network phi(x): maps state to latent Koopman coordinates z.
2. Parameterized linear operator K: governs evolution z_{k+1} = K * z_k.
3. Decoder network psi(z): reconstructs physical state x_hat = psi(z).
"""

"""
    DeepKoopmanNet <: AbstractKoopmanModel

Deep Koopman autoencoder model with linear latent dynamics.
"""
mutable struct DeepKoopmanNet <: AbstractKoopmanModel
    state_dim::Int
    latent_dim::Int

    # Encoder MLP layers
    encoder_w1::Matrix{Float64}
    encoder_b1::Vector{Float64}
    encoder_w2::Matrix{Float64}
    encoder_b2::Vector{Float64}

    # Koopman transition matrix K
    K::Matrix{Float64}

    # Decoder MLP layers
    decoder_w1::Matrix{Float64}
    decoder_b1::Vector{Float64}
    decoder_w2::Matrix{Float64}
    decoder_b2::Vector{Float64}

    function DeepKoopmanNet(state_dim::Int, latent_dim::Int; hidden_dim::Int=64)
        # Xavier/He initialization
        ew1 = randn(hidden_dim, state_dim) .* sqrt(2.0 / (state_dim + hidden_dim))
        eb1 = zeros(Float64, hidden_dim)
        ew2 = randn(latent_dim, hidden_dim) .* sqrt(2.0 / (hidden_dim + latent_dim))
        eb2 = zeros(Float64, latent_dim)

        # Initialize K near identity with small skew-symmetric perturbations
        K = Matrix{Float64}(I, latent_dim, latent_dim) .+ 0.01 .* (randn(latent_dim, latent_dim) .- randn(latent_dim, latent_dim)')

        dw1 = randn(hidden_dim, latent_dim) .* sqrt(2.0 / (latent_dim + hidden_dim))
        db1 = zeros(Float64, hidden_dim)
        dw2 = randn(state_dim, hidden_dim) .* sqrt(2.0 / (hidden_dim + state_dim))
        db2 = zeros(Float64, state_dim)

        new(state_dim, latent_dim, ew1, eb1, ew2, eb2, K, dw1, db1, dw2, db2)
    end
end

"""
    encode(net::DeepKoopmanNet, x::Vector{Float64}) -> Vector{Float64}
"""
function encode(net::DeepKoopmanNet, x::Vector{Float64})::Vector{Float64}
    h = tanh.(net.encoder_w1 * x + net.encoder_b1)
    return net.encoder_w2 * h + net.encoder_b2
end

"""
    decode(net::DeepKoopmanNet, z::Vector{Float64}) -> Vector{Float64}
"""
function decode(net::DeepKoopmanNet, z::Vector{Float64})::Vector{Float64}
    h = tanh.(net.decoder_w1 * z + net.decoder_b1)
    return net.decoder_w2 * h + net.decoder_b2
end

"""
    predict_next(net::DeepKoopmanNet, x::Vector{Float64}) -> Vector{Float64}

Computes one-step forward forecast: x_{k+1} = decode(K * encode(x_k)).
"""
function predict_next(net::DeepKoopmanNet, x::Vector{Float64})::Vector{Float64}
    z = encode(net, x)
    z_next = net.K * z
    return decode(net, z_next)
end

"""
    rollout(net::DeepKoopmanNet, x0::Vector{Float64}, steps::Int) -> Matrix{Float64}

Rolls out multi-step trajectory starting from initial state x0. Returns (steps + 1, state_dim).
"""
function rollout(net::DeepKoopmanNet, x0::Vector{Float64}, steps::Int)::Matrix{Float64}
    traj = zeros(Float64, steps + 1, net.state_dim)
    traj[1, :] = x0
    z = encode(net, x0)

    for s in 1:steps
        z = net.K * z
        traj[s + 1, :] = decode(net, z)
    end
    return traj
end

"""
    compute_loss(net::DeepKoopmanNet, X::Matrix{Float64}, Y::Matrix{Float64};
                 alpha_rec::Float64=1.0, alpha_lin::Float64=1.0, alpha_pred::Float64=1.0) -> Float64

Computes composite Deep Koopman training loss over snapshot pairs (X, Y):
L = alpha_rec * L_recon + alpha_lin * L_linear + alpha_pred * L_pred
"""
function compute_loss(net::DeepKoopmanNet, X::Matrix{Float64}, Y::Matrix{Float64};
                      alpha_rec::Float64=1.0, alpha_lin::Float64=1.0, alpha_pred::Float64=1.0)::Float64
    n_states, M = size(X)
    loss_rec = 0.0
    loss_lin = 0.0
    loss_pred = 0.0

    for m in 1:M
        x = X[:, m]
        y = Y[:, m]

        zx = encode(net, x)
        zy = encode(net, y)

        # 1. Reconstruction loss: ||decode(encode(x)) - x||^2
        x_rec = decode(net, zx)
        loss_rec += dot(x_rec - x, x_rec - x)

        # 2. Latent linearity loss: ||K * zx - zy||^2
        z_pred = net.K * zx
        loss_lin += dot(z_pred - zy, z_pred - zy)

        # 3. State prediction loss: ||decode(K * zx) - y||^2
        y_pred = decode(net, z_pred)
        loss_pred += dot(y_pred - y, y_pred - y)
    end

    total_loss = (alpha_rec * loss_rec + alpha_lin * loss_lin + alpha_pred * loss_pred) / M
    return total_loss
end

"""
    fit_linear_operator!(net::DeepKoopmanNet, X::Matrix{Float64}, Y::Matrix{Float64}; reg::Float64=1e-6)

Analytically updates the Koopman matrix K given current encoder representations:
K = (Z_Y * Z_X') * inv(Z_X * Z_X' + reg * I).
"""
function fit_linear_operator!(net::DeepKoopmanNet, X::Matrix{Float64}, Y::Matrix{Float64}; reg::Float64=1e-6)
    M = size(X, 2)
    Z_X = zeros(Float64, net.latent_dim, M)
    Z_Y = zeros(Float64, net.latent_dim, M)

    for m in 1:M
        Z_X[:, m] = encode(net, X[:, m])
        Z_Y[:, m] = encode(net, Y[:, m])
    end

    G = Z_X * Z_X'
    A = Z_Y * Z_X'
    net.K = A * inv(G + reg * I)
end
