"""
    Spectra.Operators.DeepONet

Deep Operator Network (DeepONet) architecture based on the universal approximation
theorem for operators (Chen & Chen 1995, Lu et al. 2021).
Decomposes operator mapping G: u -> G(u) into a Branch network processing function
sensor measurements u(x_1), ..., u(x_m) and a Trunk network evaluating continuous
coordinate queries y in R^d.
"""

"""
    BranchNet

Branch neural network that encodes discrete function sensor measurements.
Maps u in R^m to latent representation b in R^p.
"""
struct BranchNet
    input_dim::Int
    hidden_dims::Vector{Int}
    latent_dim::Int
    weights::Vector{Matrix{Float64}}
    biases::Vector{Vector{Float64}}

    function BranchNet(input_dim::Int, hidden_dims::Vector{Int}, latent_dim::Int)
        dims = vcat([input_dim], hidden_dims, [latent_dim])
        weights = Vector{Matrix{Float64}}()
        biases = Vector{Vector{Float64}}()

        for i in 1:(length(dims) - 1)
            in_d = dims[i]
            out_d = dims[i + 1]
            scale = sqrt(2.0 / (in_d + out_d))
            push!(weights, randn(in_d, out_d) .* scale)
            push!(biases, zeros(Float64, out_d))
        end

        new(input_dim, hidden_dims, latent_dim, weights, biases)
    end
end

function (branch::BranchNet)(u::Vector{Float64})::Vector{Float64}
    x = copy(u)
    L = length(branch.weights)
    for l in 1:L
        x = branch.weights[l]' * x + branch.biases[l]
        if l < L
            x = tanh.(x)
        end
    end
    return x
end

function (branch::BranchNet)(U::Matrix{Float64})::Matrix{Float64}
    # U: (batch_size, input_dim)
    X = copy(U)
    L = length(branch.weights)
    for l in 1:L
        X = X * branch.weights[l] .+ branch.biases[l]'
        if l < L
            X = tanh.(X)
        end
    end
    return X
end

"""
    TrunkNet

Trunk neural network that evaluates continuous coordinate basis functions.
Maps coordinate y in R^d to latent basis representation t in R^p.
"""
struct TrunkNet
    coord_dim::Int
    hidden_dims::Vector{Int}
    latent_dim::Int
    weights::Vector{Matrix{Float64}}
    biases::Vector{Vector{Float64}}

    function TrunkNet(coord_dim::Int, hidden_dims::Vector{Int}, latent_dim::Int)
        dims = vcat([coord_dim], hidden_dims, [latent_dim])
        weights = Vector{Matrix{Float64}}()
        biases = Vector{Vector{Float64}}()

        for i in 1:(length(dims) - 1)
            in_d = dims[i]
            out_d = dims[i + 1]
            scale = sqrt(2.0 / (in_d + out_d))
            push!(weights, randn(in_d, out_d) .* scale)
            push!(biases, zeros(Float64, out_d))
        end

        new(coord_dim, hidden_dims, latent_dim, weights, biases)
    end
end

function (trunk::TrunkNet)(y::Vector{Float64})::Vector{Float64}
    x = copy(y)
    L = length(trunk.weights)
    for l in 1:L
        x = trunk.weights[l]' * x + trunk.biases[l]
        if l < L
            x = tanh.(x)
        end
    end
    return x
end

function (trunk::TrunkNet)(Y::Matrix{Float64})::Matrix{Float64}
    # Y: (num_points, coord_dim)
    X = copy(Y)
    L = length(trunk.weights)
    for l in 1:L
        X = X * trunk.weights[l] .+ trunk.biases[l]'
        if l < L
            X = tanh.(X)
        end
    end
    return X
end

"""
    DeepONet <: AbstractNeuralOperator

Unstacked Deep Operator Network combining Branch and Trunk networks.
Computes inner product G(u)(y) = sum_{k=1}^p b_k(u) * t_k(y) + bias_0.
"""
struct DeepONet <: AbstractNeuralOperator
    branch::BranchNet
    trunk::TrunkNet
    bias0::Float64

    function DeepONet(sensor_dim::Int, coord_dim::Int;
                      hidden_branch::Vector{Int}=[64, 64],
                      hidden_trunk::Vector{Int}=[64, 64],
                      latent_dim::Int=64)
        branch = BranchNet(sensor_dim, hidden_branch, latent_dim)
        trunk = TrunkNet(coord_dim, hidden_trunk, latent_dim)
        new(branch, trunk, 0.0)
    end
end

"""
    (net::DeepONet)(u::Vector{Float64}, y::Vector{Float64}) -> Float64

Evaluates operator G(u)(y) at a single function input u and query point y.
"""
function (net::DeepONet)(u::Vector{Float64}, y::Vector{Float64})::Float64
    b = net.branch(u)
    t = net.trunk(y)
    return dot(b, t) + net.bias0
end

"""
    (net::DeepONet)(U::Matrix{Float64}, Y::Matrix{Float64}) -> Matrix{Float64}

Batch evaluation for N_samples function inputs and N_queries coordinate points.
Returns matrix of dimensions (N_samples, N_queries).
"""
function (net::DeepONet)(U::Matrix{Float64}, Y::Matrix{Float64})::Matrix{Float64}
    B = net.branch(U) # (N_samples, latent_dim)
    T = net.trunk(Y)  # (N_queries, latent_dim)
    return (B * T') .+ net.bias0
end
