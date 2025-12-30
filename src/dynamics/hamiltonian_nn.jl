"""
    Spectra.Dynamics.HamiltonianNN

Symplectic Hamiltonian Neural Networks (HNN).
Parameterizes an exact scalar energy function H_theta(q, p) and generates
symplectic vector fields dq/dt = dH/dp, dp/dt = -dH/dq via automatic differentiation.
Guarantees energy conservation and phase space volume preservation (Liouville theorem).
"""

struct HamiltonianEnergyMLP
    in_dim::Int       # 2 * n_dof
    hidden_dim::Int
    W1::Matrix{Float64}
    b1::Vector{Float64}
    W2::Matrix{Float64}
    b2::Vector{Float64}
    W3::Vector{Float64}
    b3::Float64

    function HamiltonianEnergyMLP(n_dof::Int, hidden_d::Int)
        in_d = 2 * n_dof
        s1 = sqrt(2.0 / (in_d + hidden_d))
        W1 = randn(hidden_d, in_d) .* s1
        b1 = zeros(Float64, hidden_d)

        s2 = sqrt(2.0 / (hidden_d + hidden_d))
        W2 = randn(hidden_d, hidden_d) .* s2
        b2 = zeros(Float64, hidden_d)

        W3 = randn(hidden_d) .* sqrt(2.0 / hidden_d)
        b3 = 0.0

        new(in_d, hidden_d, W1, b1, W2, b2, W3, b3)
    end
end

function (mlp::HamiltonianEnergyMLP)(q::AbstractVector{T}, p::AbstractVector{T})::T where {T}
    z = [q; p]
    h1 = tanh.(mlp.W1 * z .+ mlp.W1[:, 1] .* zero(T) .+ mlp.b1)
    h2 = tanh.(mlp.W2 * h1 .+ mlp.b2)
    return dot(mlp.W3, h2) + mlp.b3
end

function get_params(mlp::HamiltonianEnergyMLP)::Vector{Float64}
    return [vec(mlp.W1); mlp.b1; vec(mlp.W2); mlp.b2; mlp.W3; [mlp.b3]]
end

function set_params!(mlp::HamiltonianEnergyMLP, theta::Vector{Float64})
    h = mlp.hidden_dim
    ind = mlp.in_dim

    idx1 = h * ind
    mlp.W1 .= reshape(theta[1:idx1], (h, ind))

    idx2 = idx1 + h
    mlp.b1 .= theta[(idx1 + 1):idx2]

    idx3 = idx2 + h * h
    mlp.W2 .= reshape(theta[(idx2 + 1):idx3], (h, h))

    idx4 = idx3 + h
    mlp.b2 .= theta[(idx3 + 1):idx4]

    idx5 = idx4 + h
    mlp.W3 .= theta[(idx4 + 1):idx5]

    mlp.b3 = theta[end]
    return mlp
end

struct HamiltonianNeuralNetwork
    n_dof::Int
    energy_mlp::HamiltonianEnergyMLP

    function HamiltonianNeuralNetwork(n_dof::Int, hidden_dim::Int=16)
        mlp = HamiltonianEnergyMLP(n_dof, hidden_dim)
        new(n_dof, mlp)
    end
end

"""
    hamiltonian(hnn::HamiltonianNeuralNetwork, q::Vector{Float64}, p::Vector{Float64}) -> Float64

Evaluate total scalar Hamiltonian energy H_theta(q, p).
"""
function hamiltonian(hnn::HamiltonianNeuralNetwork, q::Vector{Float64}, p::Vector{Float64})::Float64
    return hnn.energy_mlp(q, p)
end

"""
    symplectic_gradient(hnn::HamiltonianNeuralNetwork, q::Vector{Float64}, p::Vector{Float64}) -> (dq_dt, dp_dt)

Compute symplectic vector field:
dq/dt = dH/dp, dp/dt = -dH/dq.
Computed via exact forward-mode automatic differentiation.
"""
function symplectic_gradient(hnn::HamiltonianNeuralNetwork, q::Vector{Float64}, p::Vector{Float64})
    n = hnn.n_dof
    z = [q; p]
    # Energy scalar function
    energy_fn = z_vec -> hnn.energy_mlp(z_vec[1:n], z_vec[(n + 1):end])
    grad_H = forward_gradient(energy_fn, z)

    grad_q = grad_H[1:n]
    grad_p = grad_H[(n + 1):end]

    dq_dt = grad_p
    dp_dt = -grad_q
    return (dq_dt, dp_dt)
end

"""
    vector_field(hnn::HamiltonianNeuralNetwork, z::Vector{Float64}) -> Vector{Float64}

Canonical Hamiltonian vector field J * grad H(z).
"""
function vector_field(hnn::HamiltonianNeuralNetwork, z::Vector{Float64})::Vector{Float64}
    n = hnn.n_dof
    q = z[1:n]
    p = z[(n + 1):end]
    dq, dp = symplectic_gradient(hnn, q, p)
    return [dq; dp]
end

"""
    integrate_trajectory(hnn::HamiltonianNeuralNetwork, q0::Vector{Float64}, p0::Vector{Float64},
                         tspan::Tuple{Float64, Float64}, dt::Float64) -> SymplecticSolution

Integrate learned Hamiltonian dynamics via Gauss-Legendre symplectic IRK.
"""
function integrate_trajectory(hnn::HamiltonianNeuralNetwork, q0::Vector{Float64}, p0::Vector{Float64},
                              tspan::Tuple{Float64, Float64}, dt::Float64)::SymplecticSolution
    t0, tf = tspan
    t = t0
    z = [q0; p0]
    n = hnn.n_dof

    t_hist = Float64[t0]
    q_hist = Vector{Float64}[copy(q0)]
    p_hist = Vector{Float64}[copy(p0)]
    e_hist = Float64[hamiltonian(hnn, q0, p0)]

    vf = z_state -> vector_field(hnn, z_state)

    while t < tf - 1e-12
        step_dt = min(dt, tf - t)
        z = gauss_legendre_4_step(vf, z, step_dt)
        t += step_dt

        q_curr = z[1:n]
        p_curr = z[(n + 1):end]
        push!(t_hist, t)
        push!(q_hist, copy(q_curr))
        push!(p_hist, copy(p_curr))
        push!(e_hist, hamiltonian(hnn, q_curr, p_curr))
    end

    return SymplecticSolution(t_hist, q_hist, p_hist, e_hist)
end
