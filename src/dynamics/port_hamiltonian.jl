"""
    Spectra.Dynamics.PortHamiltonian

Port-Hamiltonian Neural Networks with structured dissipation and interconnection geometry:
dx/dt = (J(x) - R(x)) * grad H(x) + G(x) u.
Enforces structural skew-symmetry J = -J^T and positive semi-definiteness R >= 0,
guaranteeing continuous Lyapunov energy dissipation: dH/dt = -grad H^T * R * grad H <= 0.
"""

struct PortHamiltonianNN
    dim::Int
    energy_mlp::HamiltonianEnergyMLP
    J_skew::Matrix{Float64}         # Skew-symmetric interconnection matrix
    R_cholesky::Matrix{Float64}     # Lower triangular factor for R = L * L^T >= 0

    function PortHamiltonianNN(dim::Int, hidden_dim::Int=16)
        mlp = HamiltonianEnergyMLP(div(dim, 2), hidden_dim)

        # Skew-symmetric J: A - A^T
        A = randn(dim, dim) .* 0.1
        J = 0.5 .* (A .- A')

        # Positive semi-definite dissipation R = L * L^T
        L = randn(dim, dim) .* 0.05
        # Zero out upper triangle to make strictly lower triangular
        for j in 1:dim
            for i in 1:(j - 1)
                L[i, j] = 0.0
            end
        end

        new(dim, mlp, J, L)
    end
end

function get_dissipation_matrix(ph::PortHamiltonianNN)::Matrix{Float64}
    return ph.R_cholesky * ph.R_cholesky'
end

function energy(ph::PortHamiltonianNN, x::Vector{Float64})::Float64
    half = div(ph.dim, 2)
    return ph.energy_mlp(x[1:half], x[(half + 1):end])
end

"""
    vector_field(ph::PortHamiltonianNN, x::Vector{Float64}) -> Vector{Float64}

Compute Port-Hamiltonian vector field: (J - R) * grad H(x).
"""
function vector_field(ph::PortHamiltonianNN, x::Vector{Float64})::Vector{Float64}
    half = div(ph.dim, 2)
    energy_fn = x_vec -> ph.energy_mlp(x_vec[1:half], x_vec[(half + 1):end])
    grad_H = forward_gradient(energy_fn, x)

    R = get_dissipation_matrix(ph)
    M = ph.J_skew .- R

    return M * grad_H
end

"""
    energy_dissipation_rate(ph::PortHamiltonianNN, x::Vector{Float64}) -> Float64

Evaluate instantaneous energy rate dH/dt = -grad H^T * R * grad H <= 0.
"""
function energy_dissipation_rate(ph::PortHamiltonianNN, x::Vector{Float64})::Float64
    half = div(ph.dim, 2)
    energy_fn = x_vec -> ph.energy_mlp(x_vec[1:half], x_vec[(half + 1):end])
    grad_H = forward_gradient(energy_fn, x)

    R = get_dissipation_matrix(ph)
    return -dot(grad_H, R * grad_H)
end

"""
    integrate_port_hamiltonian(ph::PortHamiltonianNN, x0::Vector{Float64},
                               tspan::Tuple{Float64, Float64};
                               atol::Float64=1e-8, rtol::Float64=1e-6) -> (ODESolution, Vector{Float64})

Integrate dissipative Port-Hamiltonian system and record energy decay trajectory.
"""
function integrate_port_hamiltonian(ph::PortHamiltonianNN, x0::Vector{Float64},
                                    tspan::Tuple{Float64, Float64};
                                    atol::Float64=1e-8, rtol::Float64=1e-6)
    f = (x, p, t) -> vector_field(ph, x)
    sol = integrate_ode(f, x0, tspan; atol=atol, rtol=rtol)
    energies = [energy(ph, u) for u in sol.u]
    return (sol, energies)
end
