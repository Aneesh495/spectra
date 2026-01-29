"""
    Spectra.FFI.CBridge

Foreign Function Interface (FFI) bridge between pure Julia and the native C SIMD acceleration kernel.
Provides zero-overhead ccall bindings for cache-blocked GEMM, tensor contraction,
Clenshaw recurrence, symplectic leapfrog integration, and sparse sheaf diffusion.
Gracefully falls back to pure Julia implementations if native binary is unavailable.
"""

const KERNEL_LIB_NAME = Sys.isapple() ? "libspectra_kernel.dylib" : (Sys.iswindows() ? "spectra_kernel.dll" : "libspectra_kernel.so")
const KERNEL_PATH = normpath(joinpath(@__DIR__, "..", "..", "build", KERNEL_LIB_NAME))

"""
    is_c_kernel_available() -> Bool

Returns true if native compiled C acceleration library exists on disk.
"""
function is_c_kernel_available()::Bool
    return isfile(KERNEL_PATH)
end

"""
    c_simd_gemm!(C::Matrix{Float64}, A::Matrix{Float64}, B::Matrix{Float64};
                 alpha::Float64=1.0, beta::Float64=0.0)

Cache-blocked GEMM: C = alpha * A * B + beta * C using native C SIMD kernel.
"""
function c_simd_gemm!(C::Matrix{Float64}, A::Matrix{Float64}, B::Matrix{Float64};
                      alpha::Float64=1.0, beta::Float64=0.0)
    M, K = size(A)
    K_b, N = size(B)
    @assert K == K_b "Inner dimension mismatch in c_simd_gemm"
    @assert size(C) == (M, N) "Output dimension mismatch in c_simd_gemm"

    if is_c_kernel_available() && GLOBAL_CONFIG.enable_c_kernel
        ccall((:spectra_simd_gemm, KERNEL_PATH), Cvoid,
              (Int64, Int64, Int64, Float64,
               Ptr{Float64}, Int64,
               Ptr{Float64}, Int64,
               Float64,
               Ptr{Float64}, Int64),
              M, N, K, alpha,
              A, M,
              B, K,
              beta,
              C, M)
    else
        # Pure Julia fallback
        if beta == 0.0
            C .= alpha .* (A * B)
        else
            C .= alpha .* (A * B) .+ beta .* C
        end
    end
    return C
end

"""
    c_simd_tensor_contract(weights::Array{Float64, 3}, X::Matrix{ComplexF64}) -> Matrix{ComplexF64}

Contracts spectral weights (out_c, in_c, modes) with mode matrix (modes, in_c).
"""
function c_simd_tensor_contract(weights::Array{Float64, 3}, X::Matrix{ComplexF64})::Matrix{ComplexF64}
    out_c, in_c, modes = size(weights)
    modes_x, in_cx = size(X)
    @assert modes == modes_x && in_c == in_cx "Dimension mismatch in tensor contraction"

    X_real = real.(X)
    X_imag = imag.(X)
    Y_real = zeros(Float64, modes, out_c)
    Y_imag = zeros(Float64, modes, out_c)

    if is_c_kernel_available() && GLOBAL_CONFIG.enable_c_kernel
        ccall((:spectra_simd_tensor_contract, KERNEL_PATH), Cvoid,
              (Int64, Int64, Int64,
               Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
               Ptr{Float64}, Ptr{Float64}),
              modes, in_c, out_c,
              weights, X_real, X_imag,
              Y_real, Y_imag)
    else
        for m in 1:modes
            for oc in 1:out_c
                sr = 0.0
                si = 0.0
                for ic in 1:in_c
                    w = weights[oc, ic, m]
                    sr += w * X_real[m, ic]
                    si += w * X_imag[m, ic]
                end
                Y_real[m, oc] = sr
                Y_imag[m, oc] = si
            end
        end
    end

    return ComplexF64.(Y_real, Y_imag)
end

"""
    c_chebyshev_eval(coeffs::Vector{Float64}, x_points::Vector{Float64}) -> Vector{Float64}

Vectorized Clenshaw-Smith recurrence for Chebyshev series evaluation: sum_k c_k * T_k(x).
"""
function c_chebyshev_eval(coeffs::Vector{Float64}, x_points::Vector{Float64})::Vector{Float64}
    n_points = length(x_points)
    degree = length(coeffs)
    out = zeros(Float64, n_points)

    if is_c_kernel_available() && GLOBAL_CONFIG.enable_c_kernel
        ccall((:spectra_chebyshev_eval, KERNEL_PATH), Cvoid,
              (Int64, Int64, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}),
              n_points, degree, x_points, coeffs, out)
    else
        for p in 1:n_points
            x = x_points[p]
            two_x = 2.0 * x
            b_k1 = 0.0
            b_k2 = 0.0
            for k in degree:-1:2
                b_k = coeffs[k] + two_x * b_k1 - b_k2
                b_k2 = b_k1
                b_k1 = b_k
            end
            out[p] = coeffs[1] + x * b_k1 - b_k2
        end
    end

    return out
end

"""
    c_symplectic_step(q::Vector{Float64}, p::Vector{Float64},
                      grad_V_n::Vector{Float64}, grad_V_next::Vector{Float64},
                      dt::Float64; inv_mass::Union{Vector{Float64}, Nothing}=nothing) -> (q_next, p_next)
"""
function c_symplectic_step(q::Vector{Float64}, p::Vector{Float64},
                           grad_V_n::Vector{Float64}, grad_V_next::Vector{Float64},
                           dt::Float64; inv_mass::Union{Vector{Float64}, Nothing}=nothing)::Tuple{Vector{Float64}, Vector{Float64}}
    dim = length(q)
    q_out = zeros(Float64, dim)
    p_out = zeros(Float64, dim)
    m_ptr = inv_mass === nothing ? C_NULL : pointer(inv_mass)

    if is_c_kernel_available() && GLOBAL_CONFIG.enable_c_kernel
        ccall((:spectra_symplectic_step, KERNEL_PATH), Cvoid,
              (Int64, Float64, Ptr{Float64}, Ptr{Float64},
               Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
               Ptr{Float64}, Ptr{Float64}),
              dim, dt, q, p, grad_V_n, grad_V_next, m_ptr, q_out, p_out)
    else
        half_dt = 0.5 * dt
        for i in 1:dim
            im_val = inv_mass === nothing ? 1.0 : inv_mass[i]
            p_half = p[i] - half_dt * grad_V_n[i]
            q_out[i] = q[i] + dt * im_val * p_half
            p_out[i] = p_half - half_dt * grad_V_next[i]
        end
    end

    return q_out, p_out
end

"""
    c_sheaf_diffuse(L::CSCMatrix{Float64}, x_in::Vector{Float64}, tau::Float64) -> Vector{Float64}

Sparse sheaf diffusion step: x_{t+1} = x_t - tau * L * x_t.
"""
function c_sheaf_diffuse(L::CSCMatrix{Float64}, x_in::Vector{Float64}, tau::Float64)::Vector{Float64}
    n = length(x_in)
    x_out = zeros(Float64, n)

    if is_c_kernel_available() && GLOBAL_CONFIG.enable_c_kernel
        ccall((:spectra_sheaf_diffuse, KERNEL_PATH), Cvoid,
              (Int64, Float64, Ptr{Int64}, Ptr{Int64}, Ptr{Float64},
               Ptr{Float64}, Ptr{Float64}),
              n, tau, L.colptr, L.rowval, L.nzval, x_in, x_out)
    else
        # Pure Julia fallback: x_out = x_in - tau * (L * x_in)
        Lx = zeros(Float64, n)
        spmv!(Lx, L, x_in)
        x_out .= x_in .- tau .* Lx
    end

    return x_out
end
