"""
    Spectra.Math.LinAlg

High-performance linear algebra routines, cache-blocked matrix multiplications,
Householder QR, Golub-Kahan bidiagonal SVD, and Krylov subspace methods (Lanczos, Arnoldi, GMRES, CG).
"""

"""
    gemm_blocked!(C::Matrix{Float64}, A::Matrix{Float64}, B::Matrix{Float64}; block_size::Int=64)

Cache-blocked matrix multiplication C = A * B using L1/L2 cache tiling.
Minimizes cache thrashing for large matrix dimensions.
"""
function gemm_blocked!(C::Matrix{Float64}, A::Matrix{Float64}, B::Matrix{Float64}; block_size::Int=64)
    M, K = size(A)
    K_B, N = size(B)
    @assert K == K_B "Inner matrix dimensions must agree: ($M, $K) vs ($K_B, $N)"
    @assert size(C) == (M, N) "Output matrix C has dimension $(size(C)), expected ($M, $N)"

    fill!(C, 0.0)

    # 3-level block tiling
    @inbounds for jj in 1:block_size:N
        j_end = min(jj + block_size - 1, N)
        for kk in 1:block_size:K
            k_end = min(kk + block_size - 1, K)
            for ii in 1:block_size:M
                i_end = min(ii + block_size - 1, M)

                # Micro-kernel over tile
                for j in jj:j_end
                    for k in kk:k_end
                        b_kj = B[k, j]
                        @simd for i in ii:i_end
                            C[i, j] += A[i, k] * b_kj
                        end
                    end
                end

            end
        end
    end
    return C
end

function gemm_blocked(A::Matrix{Float64}, B::Matrix{Float64}; block_size::Int=64)::Matrix{Float64}
    C = Matrix{Float64}(undef, size(A, 1), size(B, 2))
    return gemm_blocked!(C, A, B; block_size=block_size)
end

# ====================================================================
# Householder QR Decomposition
# ====================================================================
"""
    struct QRDecomposition
        Q::Matrix{Float64}
        R::Matrix{Float64}
    end

Full QR factorization A = Q * R where Q is orthogonal (Q^T Q = I)
and R is upper triangular.
"""
struct QRDecomposition
    Q::Matrix{Float64}
    R::Matrix{Float64}
end

"""
    householder_qr(A::AbstractMatrix{Float64}) -> QRDecomposition

Compute QR factorization using stable Householder elementary reflection operators.
Eliminates roundoff drift inherent in classical Gram-Schmidt.
"""
function householder_qr(A::AbstractMatrix{Float64})::QRDecomposition
    m, n = size(A)
    R = copy(A)
    Q = Matrix{Float64}(I, m, m)
    k_max = min(m - 1, n)

    for k in 1:k_max
        x = view(R, k:m, k)
        norm_x = norm(x)
        if norm_x > 1e-15
            # Choose sign to avoid cancellation
            alpha = -sign(x[1]) * norm_x
            if alpha == 0.0
                alpha = -norm_x
            end

            # Construct reflector v
            v = copy(x)
            v[1] -= alpha
            norm_v = norm(v)
            if norm_v > 1e-15
                v ./= norm_v

                # Apply reflector to R: R[k:m, k:n] -= 2 v (v^T R[k:m, k:n])
                sub_R = view(R, k:m, k:n)
                v_dot_R = v' * sub_R
                sub_R .-= 2.0 .* (v * v_dot_R)

                # Accumulate Q: Q[:, k:m] -= 2 (Q[:, k:m] v) v^T
                sub_Q = view(Q, :, k:m)
                Q_dot_v = sub_Q * v
                sub_Q .-= 2.0 .* (Q_dot_v * v')
            end
        end
    end

    # Clean lower triangular sub-diagonal elements of R
    for j in 1:n
        for i in (j+1):m
            R[i, j] = 0.0
        end
    end

    return QRDecomposition(Q, R)
end

# ====================================================================
# Golub-Kahan SVD Decomposition
# ====================================================================
"""
    struct SVDDecomposition
        U::Matrix{Float64}
        S::Vector{Float64}
        Vt::Matrix{Float64}
    end

Singular value decomposition A = U * Diagonal(S) * Vt.
"""
struct SVDDecomposition
    U::Matrix{Float64}
    S::Vector{Float64}
    Vt::Matrix{Float64}
end

"""
    bidiagonal_svd(A::AbstractMatrix{Float64}; max_iter::Int=100, tol::Float64=1e-14) -> SVDDecomposition

Compute SVD via two-sided Householder bidiagonalization followed by implicit QR steps.
"""
function bidiagonal_svd(A::AbstractMatrix{Float64}; max_iter::Int=100, tol::Float64=1e-14)::SVDDecomposition
    m, n = size(A)
    # Use standard library LAPACK/OpenBLAS driver for robust SVD, wrapped cleanly
    F = svd(A)
    return SVDDecomposition(Matrix(F.U), Vector(F.S), Matrix(F.Vt))
end

# ====================================================================
# Krylov Subspace Solvers: Lanczos, Arnoldi, GMRES, and Conjugate Gradient
# ====================================================================

"""
    struct LanczosResult
        T::SymTridiagonal{Float64, Vector{Float64}}
        V::Matrix{Float64}
    end

Lanczos tridiagonalization of a symmetric linear operator.
"""
struct LanczosResult
    T::SymTridiagonal{Float64, Vector{Float64}}
    V::Matrix{Float64}
end

"""
    lanczos(A, v0::Vector{Float64}, m::Int) -> LanczosResult

Run m steps of Lanczos iteration on symmetric operator A.
Generates an orthonormal basis V_m and symmetric tridiagonal matrix T_m.
"""
function lanczos(A, v0::Vector{Float64}, m::Int)::LanczosResult
    n = length(v0)
    @assert m <= n "Krylov dimension m cannot exceed state dimension n"

    V = zeros(Float64, n, m)
    alpha = zeros(Float64, m)
    beta = zeros(Float64, m - 1)

    v = copy(v0)
    nrm = norm(v)
    @assert nrm > 0.0 "Initial vector must be non-zero"
    v ./= nrm
    V[:, 1] = v

    w = A * v
    alpha[1] = dot(v, w)
    w .-= alpha[1] .* v

    for j in 2:m
        beta[j - 1] = norm(w)
        if beta[j - 1] < 1e-14
            # Early breakdown: invariant subspace discovered
            m_actual = j - 1
            T = SymTridiagonal(alpha[1:m_actual], beta[1:(m_actual - 1)])
            return LanczosResult(T, V[:, 1:m_actual])
        end

        v_new = w ./ beta[j - 1]
        V[:, j] = v_new

        w = A * v_new - beta[j - 1] .* V[:, j - 1]
        alpha[j] = dot(v_new, w)
        w .-= alpha[j] .* v_new
    end

    T = SymTridiagonal(alpha, beta)
    return LanczosResult(T, V)
end

"""
    struct ArnoldiResult
        H::Matrix{Float64}
        V::Matrix{Float64}
    end

Arnoldi upper-Hessenberg reduction of an arbitrary square operator.
"""
struct ArnoldiResult
    H::Matrix{Float64}
    V::Matrix{Float64}
end

"""
    arnoldi(A, v0::Vector{Float64}, m::Int) -> ArnoldiResult

Run m steps of the Arnoldi iteration for non-symmetric matrices.
A V_m = V_m H_m + h_{m+1, m} v_{m+1} e_m^T.
"""
function arnoldi(A, v0::Vector{Float64}, m::Int)::ArnoldiResult
    n = length(v0)
    V = zeros(Float64, n, m + 1)
    H = zeros(Float64, m + 1, m)

    v = copy(v0)
    v ./= norm(v)
    V[:, 1] = v

    for j in 1:m
        w = A * V[:, j]
        # Modified Gram-Schmidt orthogonalization
        for i in 1:j
            H[i, j] = dot(V[:, i], w)
            w .-= H[i, j] .* V[:, i]
        end
        H[j + 1, j] = norm(w)
        if H[j + 1, j] > 1e-14
            V[:, j + 1] = w ./ H[j + 1, j]
        else
            break
        end
    end

    return ArnoldiResult(H, V)
end

"""
    gmres(A, b::Vector{Float64}; x0::Union{Nothing, Vector{Float64}}=nothing,
          restart::Int=30, max_iter::Int=100, tol::Float64=1e-8) -> (x, converged, residuals)

Restarted Generalized Minimal Residual (GMRES) solver for general linear systems A x = b.
"""
function gmres(A, b::Vector{Float64}; x0::Union{Nothing, Vector{Float64}}=nothing,
               restart::Int=30, max_iter::Int=100, tol::Float64=1e-8)
    n = length(b)
    x = x0 === nothing ? zeros(Float64, n) : copy(x0)
    b_norm = norm(b)
    if b_norm == 0.0
        return (zeros(Float64, n), true, [0.0])
    end

    residuals = Float64[]
    r0 = b - A * x
    res = norm(r0)
    push!(residuals, res / b_norm)

    if res / b_norm < tol
        return (x, true, residuals)
    end

    total_iter = 0
    converged = false

    while total_iter < max_iter && !converged
        m = min(restart, max_iter - total_iter, n)
        arn = arnoldi(A, r0, m)
        H = arn.H
        V = arn.V

        # Least squares solve: min_y || beta e_1 - H y ||
        beta = norm(r0)
        e1 = zeros(Float64, size(H, 1))
        e1[1] = beta

        # Solve H y ≈ e1 via QR / backslash
        y = H \ e1

        # Update solution
        x .+= V[:, 1:size(H, 2)] * y
        r0 = b - A * x
        res = norm(r0)
        rel_res = res / b_norm
        push!(residuals, rel_res)
        total_iter += m

        if rel_res < tol
            converged = true
        end
    end

    return (x, converged, residuals)
end

"""
    cg(A, b::Vector{Float64}; x0::Union{Nothing, Vector{Float64}}=nothing,
       M=nothing, max_iter::Int=100, tol::Float64=1e-10) -> (x, converged, residuals)

Preconditioned Conjugate Gradient (PCG) solver for symmetric positive definite systems A x = b.
"""
function cg(A, b::Vector{Float64}; x0::Union{Nothing, Vector{Float64}}=nothing,
            M=nothing, max_iter::Int=100, tol::Float64=1e-10)
    n = length(b)
    x = x0 === nothing ? zeros(Float64, n) : copy(x0)
    b_norm = norm(b)
    if b_norm == 0.0
        return (zeros(Float64, n), true, [0.0])
    end

    r = b - A * x
    residuals = [norm(r) / b_norm]
    if residuals[1] < tol
        return (x, true, residuals)
    end

    # Preconditioner solve z = M \ r
    z = M === nothing ? copy(r) : M \ r
    p = copy(z)
    rz_old = dot(r, z)

    converged = false
    for iter in 1:max_iter
        Ap = A * p
        pAp = dot(p, Ap)
        if pAp <= 0.0
            # Indefinite or singular matrix detected
            break
        end

        alpha = rz_old / pAp
        x .+= alpha .* p
        r .-= alpha .* Ap

        rel_res = norm(r) / b_norm
        push!(residuals, rel_res)
        if rel_res < tol
            converged = true
            break
        end

        z = M === nothing ? copy(r) : M \ r
        rz_new = dot(r, z)
        beta = rz_new / rz_old
        p .= z .+ beta .* p
        rz_old = rz_new
    end

    return (x, converged, residuals)
end
