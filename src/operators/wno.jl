"""
    Spectra.Operators.WNO

Multi-Scale Wavelet Neural Operator (WNO-1D) for operator learning with localized multi-scale features.
Utilizes Discrete Wavelet Transforms (DWT) including Haar and Daubechies-4 (D4) orthonormal wavelets.
Captures both frequency localization and compact spatial support, effectively handling shocks and singularities.
"""

# Daubechies-4 filter coefficients
const D4_H0 = (1.0 + sqrt(3.0)) / (4.0 * sqrt(2.0))
const D4_H1 = (3.0 + sqrt(3.0)) / (4.0 * sqrt(2.0))
const D4_H2 = (3.0 - sqrt(3.0)) / (4.0 * sqrt(2.0))
const D4_H3 = (1.0 - sqrt(3.0)) / (4.0 * sqrt(2.0))

const D4_G0 =  D4_H3
const D4_G1 = -D4_H2
const D4_G2 =  D4_H1
const D4_G3 = -D4_H0

"""
    dwt_step_haar(x::Vector{Float64}) -> (Vector{Float64}, Vector{Float64})

Single-level 1D Haar forward discrete wavelet transform step.
Returns (low_pass_approximation, high_pass_detail). Length must be even.
"""
function dwt_step_haar(x::Vector{Float64})::Tuple{Vector{Float64}, Vector{Float64}}
    n = length(x)
    half = div(n, 2)
    approx = zeros(Float64, half)
    detail = zeros(Float64, half)
    inv_sqrt2 = 1.0 / sqrt(2.0)

    for i in 1:half
        approx[i] = (x[2*i - 1] + x[2*i]) * inv_sqrt2
        detail[i] = (x[2*i - 1] - x[2*i]) * inv_sqrt2
    end
    return approx, detail
end

"""
    idwt_step_haar(approx::Vector{Float64}, detail::Vector{Float64}) -> Vector{Float64}

Single-level 1D Haar inverse discrete wavelet transform reconstruction.
"""
function idwt_step_haar(approx::Vector{Float64}, detail::Vector{Float64})::Vector{Float64}
    half = length(approx)
    n = 2 * half
    x = zeros(Float64, n)
    inv_sqrt2 = 1.0 / sqrt(2.0)

    for i in 1:half
        a = approx[i] * inv_sqrt2
        d = detail[i] * inv_sqrt2
        x[2*i - 1] = a + d
        x[2*i]     = a - d
    end
    return x
end

"""
    dwt_step_d4(x::Vector{Float64}) -> (Vector{Float64}, Vector{Float64})

Single-level 1D Daubechies-4 forward DWT with periodic boundary extension.
"""
function dwt_step_d4(x::Vector{Float64})::Tuple{Vector{Float64}, Vector{Float64}}
    n = length(x)
    half = div(n, 2)
    approx = zeros(Float64, half)
    detail = zeros(Float64, half)

    for i in 1:half
        idx0 = 2*i - 1
        idx1 = 2*i
        idx2 = mod1(2*i + 1, n)
        idx3 = mod1(2*i + 2, n)

        approx[i] = D4_H0 * x[idx0] + D4_H1 * x[idx1] + D4_H2 * x[idx2] + D4_H3 * x[idx3]
        detail[i] = D4_G0 * x[idx0] + D4_G1 * x[idx1] + D4_G2 * x[idx2] + D4_G3 * x[idx3]
    end
    return approx, detail
end

"""
    idwt_step_d4(approx::Vector{Float64}, detail::Vector{Float64}) -> Vector{Float64}

Single-level 1D Daubechies-4 inverse DWT reconstruction.
"""
function idwt_step_d4(approx::Vector{Float64}, detail::Vector{Float64})::Vector{Float64}
    half = length(approx)
    n = 2 * half
    x = zeros(Float64, n)

    for i in 1:half
        # Dual synthesis reconstruction
        prev_i = mod1(i - 1, half)
        prev2_i = mod1(i - 2, half)

        x[2*i - 1] = D4_H2 * approx[prev_i] + D4_H0 * approx[i] +
                     D4_G2 * detail[prev_i] + D4_G0 * detail[i]
        x[2*i]     = D4_H3 * approx[prev_i] + D4_H1 * approx[i] +
                     D4_G3 * detail[prev_i] + D4_G1 * detail[i]
    end
    return x
end

"""
    MultiLevelWaveletDecomposition

Holds multi-scale approximation and detail wavelet coefficients.
"""
struct MultiLevelWaveletDecomposition
    approx::Vector{Float64}
    details::Vector{Vector{Float64}}
    levels::Int
    family::Symbol # :haar or :d4
end

"""
    forward_dwt_1d(x::Vector{Float64}, levels::Int=3, family::Symbol=:haar) -> MultiLevelWaveletDecomposition
"""
function forward_dwt_1d(x::Vector{Float64}, levels::Int=3, family::Symbol=:haar)::MultiLevelWaveletDecomposition
    cur = copy(x)
    details = Vector{Vector{Float64}}()

    for lvl in 1:levels
        if length(cur) < 4 || isodd(length(cur))
            levels = lvl - 1
            break
        end
        if family == :d4
            a, d = dwt_step_d4(cur)
        else
            a, d = dwt_step_haar(cur)
        end
        push!(details, d)
        cur = a
    end

    return MultiLevelWaveletDecomposition(cur, details, length(details), family)
end

"""
    inverse_dwt_1d(decomp::MultiLevelWaveletDecomposition) -> Vector{Float64}
"""
function inverse_dwt_1d(decomp::MultiLevelWaveletDecomposition)::Vector{Float64}
    cur = copy(decomp.approx)
    for lvl in decomp.levels:-1:1
        d = decomp.details[lvl]
        if decomp.family == :d4
            cur = idwt_step_d4(cur, d)
        else
            cur = idwt_step_haar(cur, d)
        end
    end
    return cur
end

"""
    WaveletConv1D

Multi-scale wavelet domain convolution layer.
Transforms spatial features to wavelet domain, scales coefficients across scales, and inverts.
"""
struct WaveletConv1D <: AbstractPolynomialOperator
    in_channels::Int
    out_channels::Int
    levels::Int
    family::Symbol

    # Parameter weights for approximation scale and detail scales
    weight_approx::Matrix{Float64} # (out_channels, in_channels)
    weight_details::Vector{Matrix{Float64}} # levels elements of (out_channels, in_channels)

    function WaveletConv1D(in_c::Int, out_c::Int; levels::Int=3, family::Symbol=:haar)
        scale = sqrt(2.0 / (in_c + out_c))
        w_approx = randn(out_c, in_c) .* scale
        w_details = [randn(out_c, in_c) .* scale for _ in 1:levels]
        new(in_c, out_c, levels, family, w_approx, w_details)
    end
end

"""
    forward_wavelet_conv(conv::WaveletConv1D, x::Matrix{Float64}) -> Matrix{Float64}
"""
function forward_wavelet_conv(conv::WaveletConv1D, x::Matrix{Float64})::Matrix{Float64}
    N_spatial, in_c = size(x)
    @assert in_c == conv.in_channels "Input channel mismatch in WaveletConv1D"
    out_c = conv.out_channels

    # Decompose all input channels
    decomps = [forward_dwt_1d(x[:, ic], conv.levels, conv.family) for ic in 1:in_c]
    actual_levels = decomps[1].levels

    # Mix approximation coefficients
    approx_len = length(decomps[1].approx)
    out_approx = zeros(Float64, approx_len, out_c)
    for oc in 1:out_c
        for ic in 1:in_c
            w = conv.weight_approx[oc, ic]
            @inbounds for k in 1:approx_len
                out_approx[k, oc] += w * decomps[ic].approx[k]
            end
        end
    end

    # Mix detail coefficients for each level
    out_details = Vector{Matrix{Float64}}()
    for lvl in 1:actual_levels
        det_len = length(decomps[1].details[lvl])
        lvl_out = zeros(Float64, det_len, out_c)
        w_mat = conv.weight_details[min(lvl, length(conv.weight_details))]
        for oc in 1:out_c
            for ic in 1:in_c
                w = w_mat[oc, ic]
                @inbounds for k in 1:det_len
                    lvl_out[k, oc] += w * decomps[ic].details[lvl][k]
                end
            end
        end
        push!(out_details, lvl_out)
    end

    # Reconstruct each output channel via inverse DWT
    y = zeros(Float64, N_spatial, out_c)
    for oc in 1:out_c
        ch_details = [out_details[lvl][:, oc] for lvl in 1:actual_levels]
        ch_decomp = MultiLevelWaveletDecomposition(out_approx[:, oc], ch_details, actual_levels, conv.family)
        y[:, oc] = inverse_dwt_1d(ch_decomp)
    end

    return y
end

"""
    WaveletNeuralOperator1D <: AbstractNeuralOperator

Wavelet Neural Operator architecture for infinite-dimensional operator learning with multi-scale wavelets.
Combines pointwise lifting, cascaded multi-scale wavelet convolution with bypass residuals, and two-stage projection.
"""
struct WaveletNeuralOperator1D <: AbstractNeuralOperator
    in_features::Int
    hidden_channels::Int
    out_features::Int
    levels::Int
    n_layers::Int
    family::Symbol

    P_weights::Matrix{Float64}
    P_bias::Vector{Float64}

    wavelet_layers::Vector{WaveletConv1D}
    linear_bypass::Vector{Matrix{Float64}}

    Q1_weights::Matrix{Float64}
    Q1_bias::Vector{Float64}
    Q2_weights::Matrix{Float64}
    Q2_bias::Vector{Float64}

    function WaveletNeuralOperator1D(in_f::Int, out_f::Int;
                                     hidden_c::Int=32, levels::Int=3,
                                     n_layers::Int=4, family::Symbol=:haar)
        P_w = randn(in_f, hidden_c) .* sqrt(2.0 / (in_f + hidden_c))
        P_b = zeros(Float64, hidden_c)

        w_layers = [WaveletConv1D(hidden_c, hidden_c; levels=levels, family=family) for _ in 1:n_layers]
        bypass = [randn(hidden_c, hidden_c) .* sqrt(2.0 / (2 * hidden_c)) for _ in 1:n_layers]

        Q1_w = randn(hidden_c, 48) .* sqrt(2.0 / (hidden_c + 48))
        Q1_b = zeros(Float64, 48)

        Q2_w = randn(48, out_f) .* sqrt(2.0 / (48 + out_f))
        Q2_b = zeros(Float64, out_f)

        new(in_f, hidden_c, out_f, levels, n_layers, family,
            P_w, P_b, w_layers, bypass, Q1_w, Q1_b, Q2_w, Q2_b)
    end
end

function (wno::WaveletNeuralOperator1D)(x::Matrix{Float64})::Matrix{Float64}
    N_spatial, in_f = size(x)
    @assert in_f == wno.in_features "Input feature dimension mismatch"

    v = x * wno.P_weights .+ wno.P_bias'

    for l in 1:wno.n_layers
        v_wav = forward_wavelet_conv(wno.wavelet_layers[l], v)
        v_lin = v * wno.linear_bypass[l]
        v = gelu.(v_wav .+ v_lin)
    end

    h = gelu.(v * wno.Q1_weights .+ wno.Q1_bias')
    return h * wno.Q2_weights .+ wno.Q2_bias'
end
