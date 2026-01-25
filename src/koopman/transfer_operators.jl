"""
    Spectra.Koopman.TransferOperators

Computation and spectral analysis of the Perron-Frobenius transfer operator (pushforward of densities),
the dual of the Koopman operator on L^1(Omega):
    <P rho, g> = <rho, K g>

Implements:
1. Ulam discretization of continuous dynamical systems via box partitioning and Monte Carlo shooting.
2. Stationary invariant measure computation via Perron-Frobenius leading eigenvector.
3. Mixing rates, correlation decay timescales, and spectral gap analysis.
4. Coherent and almost-invariant set extraction via sub-dominant eigenmodes.
"""

"""
    BoxPartition1D

Uniform 1D grid partition of interval [x_min, x_max] into n_boxes disjoint intervals.
"""
struct BoxPartition1D
    x_min::Float64
    x_max::Float64
    n_boxes::Int
    dx::Float64

    function BoxPartition1D(x_min::Float64, x_max::Float64, n_boxes::Int)
        @assert x_max > x_min "x_max must exceed x_min"
        dx = (x_max - x_min) / n_boxes
        new(x_min, x_max, n_boxes, dx)
    end
end

"""
    find_box(partition::BoxPartition1D, x::Float64) -> Int

Returns 1-based box index containing point x, or 0 if outside domain.
"""
function find_box(partition::BoxPartition1D, x::Float64)::Int
    if x < partition.x_min || x > partition.x_max
        return 0
    end
    idx = floor(Int, (x - partition.x_min) / partition.dx) + 1
    return min(idx, partition.n_boxes)
end

"""
    BoxPartition2D

Uniform 2D grid partition of [x_min, x_max] x [y_min, y_max] into nx x ny boxes.
"""
struct BoxPartition2D
    x_min::Float64
    x_max::Float64
    y_min::Float64
    y_max::Float64
    nx::Int
    ny::Int
    dx::Float64
    dy::Float64

    function BoxPartition2D(x_min::Float64, x_max::Float64, y_min::Float64, y_max::Float64, nx::Int, ny::Int)
        dx = (x_max - x_min) / nx
        dy = (y_max - y_min) / ny
        new(x_min, x_max, y_min, y_max, nx, ny, dx, dy)
    end
end

"""
    find_box(partition::BoxPartition2D, x::Float64, y::Float64) -> Int

Returns 1-based linear box index for coordinate (x, y), or 0 if outside domain.
"""
function find_box(partition::BoxPartition2D, x::Float64, y::Float64)::Int
    if x < partition.x_min || x > partition.x_max || y < partition.y_min || y > partition.y_max
        return 0
    end
    ix = min(floor(Int, (x - partition.x_min) / partition.dx) + 1, partition.nx)
    iy = min(floor(Int, (y - partition.y_min) / partition.dy) + 1, partition.ny)
    return (iy - 1) * partition.nx + ix
end

"""
    PerronFrobeniusResult

Container for discretized Perron-Frobenius transfer operator matrix and its spectral decomposition.
"""
struct PerronFrobeniusResult <: AbstractTransferOperator
    P::Matrix{Float64}               # Markov transition matrix P_ij = Prob(box_i -> box_j)
    invariant_measure::Vector{Float64} # Stationary distribution: P' * mu = mu
    eigenvalues::Vector{ComplexF64}  # Spectrum of P
    eigenvectors::Matrix{ComplexF64} # Eigenvectors of P
    spectral_gap::Float64            # 1 - |lambda_2|
    mixing_time::Float64             # 1 / (-ln(|lambda_2|))
end

"""
    ulam_perron_frobenius(map_fn::Function, partition::BoxPartition1D;
                          n_samples_per_box::Int=200) -> PerronFrobeniusResult

Discretizes the 1D Perron-Frobenius transfer operator using Ulam's method.
For each box B_i, shoots n_samples_per_box test points through map_fn and counts
transitions into boxes B_j to construct row-stochastic transition matrix P.
"""
function ulam_perron_frobenius(map_fn::Function, partition::BoxPartition1D;
                               n_samples_per_box::Int=200)::PerronFrobeniusResult
    n = partition.n_boxes
    P = zeros(Float64, n, n)

    for i in 1:n
        b_left = partition.x_min + (i - 1) * partition.dx
        for _ in 1:n_samples_per_box
            x = b_left + rand() * partition.dx
            y = map_fn(x)
            j = find_box(partition, y)
            if j >= 1 && j <= n
                P[i, j] += 1.0
            end
        end

        row_sum = sum(P[i, :])
        if row_sum > 0.0
            P[i, :] ./= row_sum
        else
            # Leakage / absorbing boundary
            P[i, i] = 1.0
        end
    end

    # Eigendecomposition of P' (left eigenvectors of P)
    decomp = eigen(P')
    vals = decomp.values
    vecs = decomp.vectors

    # Sort eigenvalues by magnitude descending
    perm = sortperm(abs.(vals); rev=true)
    sorted_vals = vals[perm]
    sorted_vecs = vecs[:, perm]

    # Stationary invariant distribution mu: leading eigenvector with lambda = 1
    mu_raw = real.(sorted_vecs[:, 1])
    # Ensure positive sign convention
    if sum(mu_raw) < 0.0
        mu_raw = -mu_raw
    end
    mu_raw = max.(mu_raw, 0.0)
    mu = mu_raw ./ (sum(mu_raw) + 1e-15)

    # Spectral gap: 1 - |lambda_2|
    lambda2_abs = length(sorted_vals) > 1 ? abs(sorted_vals[2]) : 0.0
    gap = 1.0 - lambda2_abs
    mix_time = (lambda2_abs < 1.0 && lambda2_abs > 1e-10) ? -1.0 / log(lambda2_abs) : Inf

    return PerronFrobeniusResult(P, mu, sorted_vals, sorted_vecs, gap, mix_time)
end

"""
    ulam_perron_frobenius(map_fn::Function, partition::BoxPartition2D;
                          n_samples_per_box::Int=100) -> PerronFrobeniusResult

Discretizes the 2D Perron-Frobenius transfer operator using Ulam's method on a 2D box partition.
"""
function ulam_perron_frobenius(map_fn::Function, partition::BoxPartition2D;
                               n_samples_per_box::Int=100)::PerronFrobeniusResult
    n = partition.nx * partition.ny
    P = zeros(Float64, n, n)

    for iy in 1:partition.ny
        y_low = partition.y_min + (iy - 1) * partition.dy
        for ix in 1:partition.nx
            x_low = partition.x_min + (ix - 1) * partition.dx
            i = (iy - 1) * partition.nx + ix

            for _ in 1:n_samples_per_box
                x = x_low + rand() * partition.dx
                y = y_low + rand() * partition.dy
                x_next, y_next = map_fn(x, y)
                j = find_box(partition, x_next, y_next)
                if j >= 1 && j <= n
                    P[i, j] += 1.0
                end
            end

            row_sum = sum(P[i, :])
            if row_sum > 0.0
                P[i, :] ./= row_sum
            else
                P[i, i] = 1.0
            end
        end
    end

    decomp = eigen(P')
    vals = decomp.values
    vecs = decomp.vectors

    perm = sortperm(abs.(vals); rev=true)
    sorted_vals = vals[perm]
    sorted_vecs = vecs[:, perm]

    mu_raw = real.(sorted_vecs[:, 1])
    if sum(mu_raw) < 0.0
        mu_raw = -mu_raw
    end
    mu_raw = max.(mu_raw, 0.0)
    mu = mu_raw ./ (sum(mu_raw) + 1e-15)

    lambda2_abs = length(sorted_vals) > 1 ? abs(sorted_vals[2]) : 0.0
    gap = 1.0 - lambda2_abs
    mix_time = (lambda2_abs < 1.0 && lambda2_abs > 1e-10) ? -1.0 / log(lambda2_abs) : Inf

    return PerronFrobeniusResult(P, mu, sorted_vals, sorted_vecs, gap, mix_time)
end

"""
    almost_invariant_sets(res::PerronFrobeniusResult) -> (Vector{Int}, Vector{Int})

Partitions domain into two almost-invariant / metastable sets using the sign of
the second eigenvector of the transfer operator (spectral bisection).
"""
function almost_invariant_sets(res::PerronFrobeniusResult)::Tuple{Vector{Int}, Vector{Int}}
    @assert length(res.eigenvalues) >= 2 "At least two eigenmodes required for almost-invariant set partition"
    v2 = real.(res.eigenvectors[:, 2])

    set_A = findall(v -> v >= 0.0, v2)
    set_B = findall(v -> v < 0.0, v2)

    return set_A, set_B
end

"""
    pushforward_density(res::PerronFrobeniusResult, rho0::Vector{Float64}, steps::Int=1) -> Vector{Float64}

Evolves initial probability distribution rho0 forward under the Perron-Frobenius operator:
rho_t = rho_0 * P^t.
"""
function pushforward_density(res::PerronFrobeniusResult, rho0::Vector{Float64}, steps::Int=1)::Vector{Float64}
    rho = copy(rho0)
    # rho is probability row vector: rho_{k+1} = rho_k * P
    for _ in 1:steps
        rho = P_transpose_mult = res.P' * rho
        rho ./= (sum(rho) + 1e-15)
    end
    return rho
end
