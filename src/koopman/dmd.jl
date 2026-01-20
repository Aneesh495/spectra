"""
    Spectra.Koopman.DMD

Dynamic Mode Decomposition (DMD) algorithms for extracting linear dynamical operators
and coherent spatiotemporal structures from nonlinear snapshot sequences.
Includes:
1. Exact DMD (Tu et al. 2014) with SVD rank truncation and continuous eigenvalues.
2. Hankel Delay-Embedding DMD (HAVOK framework) for chaotic and attractor dynamics.
3. Streaming Online DMD for real-time streaming state estimation.
"""

"""
    ExactDMDResult

Container for fitted Exact Dynamic Mode Decomposition model.
"""
struct ExactDMDResult <: AbstractKoopmanModel
    modes::Matrix{ComplexF64}        # Exact DMD modes Phi
    eigenvalues::Vector{ComplexF64}  # Discrete eigenvalues lambda
    omega::Vector{ComplexF64}        # Continuous eigenvalues ln(lambda)/dt
    amplitudes::Vector{ComplexF64}   # Initial mode amplitudes b
    A_reduced::Matrix{ComplexF64}    # Reduced dynamics matrix A_tilde
    rank::Int
    dt::Float64
end

"""
    exact_dmd(X::Matrix{Float64}, Y::Matrix{Float64}, dt::Float64;
              rank::Int=0, tol::Float64=1e-8) -> ExactDMDResult

Fits Exact Dynamic Mode Decomposition between consecutive snapshot matrices X and Y.
If rank <= 0, automatically determines rank truncation based on singular value threshold.
"""
function exact_dmd(X::Matrix{Float64}, Y::Matrix{Float64}, dt::Float64;
                   rank::Int=0, tol::Float64=1e-8)::ExactDMDResult
    n_states, n_snapshots = size(X)
    @assert size(Y) == (n_states, n_snapshots) "X and Y snapshot matrices must have identical dimensions"

    # Step 1: SVD of X: X = U * S * V'
    svd_decomp = svd(X)
    s = svd_decomp.S
    total_r = length(s)

    # Rank selection
    r = rank
    if r <= 0
        r = count(val -> (val / (s[1] + 1e-15)) > tol, s)
        r = max(1, min(r, total_r))
    else
        r = min(r, total_r)
    end

    U_r = svd_decomp.U[:, 1:r]
    S_r = Diagonal(s[1:r])
    S_r_inv = Diagonal(1.0 ./ s[1:r])
    V_r = svd_decomp.V[:, 1:r]

    # Step 2: Compute reduced dynamics matrix A_tilde = U_r' * Y * V_r * S_r^{-1}
    A_tilde = U_r' * Y * V_r * S_r_inv

    # Step 3: Spectral decomposition of A_tilde: A_tilde * W = W * Lambda
    eigen_decomp = eigen(A_tilde)
    lambda = eigen_decomp.values
    W = eigen_decomp.vectors

    # Step 4: Compute exact dynamic modes: Phi = Y * V_r * S_r^{-1} * W
    Phi = (Y * V_r * S_r_inv) * W

    # Step 5: Continuous eigenvalues omega = ln(lambda) / dt
    omega = log.(ComplexF64.(lambda)) ./ dt

    # Step 6: Initial amplitudes b = pinv(Phi) * x0
    x0 = X[:, 1]
    b = Phi \ ComplexF64.(x0)

    return ExactDMDResult(Phi, lambda, omega, b, A_tilde, r, dt)
end

"""
    predict(dmd::ExactDMDResult, t::Float64) -> Vector{Float64}

Predicts the continuous state x(t) using modal expansion:
    x(t) = sum_k Phi_k * exp(omega_k * t) * b_k
"""
function predict(dmd::ExactDMDResult, t::Float64)::Vector{Float64}
    exp_term = exp.(dmd.omega .* t) .* dmd.amplitudes
    return real.(dmd.modes * exp_term)
end

"""
    predict_trajectory(dmd::ExactDMDResult, times::Vector{Float64}) -> Matrix{Float64}

Predicts state trajectory across an array of time instances. Returns (n_times, n_states).
"""
function predict_trajectory(dmd::ExactDMDResult, times::Vector{Float64})::Matrix{Float64}
    n_times = length(times)
    n_states = size(dmd.modes, 1)
    traj = zeros(Float64, n_times, n_states)

    for i in 1:n_times
        traj[i, :] = predict(dmd, times[i])
    end
    return traj
end

"""
    build_hankel_matrix(data::Vector{Float64}, n_delays::Int) -> Matrix{Float64}

Constructs a block Hankel delay-embedding matrix from a 1D scalar time series.
Dimensions: (n_delays, N - n_delays + 1).
"""
function build_hankel_matrix(data::Vector{Float64}, n_delays::Int)::Matrix{Float64}
    N = length(data)
    @assert n_delays < N "Number of delays must be less than time series length"
    n_cols = N - n_delays + 1
    H = zeros(Float64, n_delays, n_cols)

    for j in 1:n_cols
        for i in 1:n_delays
            H[i, j] = data[j + i - 1]
        end
    end
    return H
end

"""
    hankel_dmd(data::Vector{Float64}, n_delays::Int, dt::Float64; rank::Int=0) -> ExactDMDResult

Fits Hankel Delay-Embedding DMD (HAVOK framework) to a scalar nonlinear time series.
"""
function hankel_dmd(data::Vector{Float64}, n_delays::Int, dt::Float64; rank::Int=0)::ExactDMDResult
    H = build_hankel_matrix(data, n_delays)
    X = H[:, 1:(end - 1)]
    Y = H[:, 2:end]
    return exact_dmd(X, Y, dt; rank=rank)
end

"""
    StreamingDMD

Online streaming DMD maintaining running statistics without storing historical data.
"""
mutable struct StreamingDMD <: AbstractKoopmanModel
    dim::Int
    weight::Float64
    Gx::Matrix{Float64}
    Gyx::Matrix{Float64}
    count::Int

    function StreamingDMD(dim::Int; forgetting_factor::Float64=1.0)
        Gx = zeros(Float64, dim, dim)
        Gyx = zeros(Float64, dim, dim)
        new(dim, forgetting_factor, Gx, Gyx, 0)
    end
end

"""
    update!(sdmd::StreamingDMD, x::Vector{Float64}, y::Vector{Float64})

Updates streaming DMD with a new observation pair (x_k, y_k).
"""
function update!(sdmd::StreamingDMD, x::Vector{Float64}, y::Vector{Float64})
    gamma = sdmd.weight
    sdmd.Gx  .= gamma .* sdmd.Gx .+ (x * x')
    sdmd.Gyx .= gamma .* sdmd.Gyx .+ (y * x')
    sdmd.count += 1
end

"""
    get_operator(sdmd::StreamingDMD; reg::Float64=1e-6) -> Matrix{Float64}

Computes current estimated linear operator A = Gyx * (Gx + reg * I)^(-1).
"""
function get_operator(sdmd::StreamingDMD; reg::Float64=1e-6)::Matrix{Float64}
    Gx_reg = sdmd.Gx + reg * I
    return sdmd.Gyx * inv(Gx_reg)
end
