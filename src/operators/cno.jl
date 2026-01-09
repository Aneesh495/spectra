"""
    Spectra.Operators.CNO

Continuous Chebyshev Neural Operator (CNO-1D) for non-periodic boundary value problems.
Uses Chebyshev polynomial basis expansions T_n(x) and Fast Discrete Cosine Transforms (DCT-II/III).
Eliminates Gibbs oscillations at non-periodic boundaries where Fourier bases degrade.
"""

struct ChebyshevConv1D <: AbstractPolynomialOperator
    in_channels::Int
    out_channels::Int
    modes::Int # Number of Chebyshev polynomial coefficients retained
    weights::Array{Float64, 3} # (out_channels, in_channels, modes)

    function ChebyshevConv1D(in_c::Int, out_c::Int, modes::Int)
        scale = sqrt(2.0 / (in_c * modes))
        weights = randn(out_c, in_c, modes) .* scale
        new(in_c, out_c, modes, weights)
    end
end

"""
    forward_chebyshev_conv(conv::ChebyshevConv1D, x::Matrix{Float64}) -> Matrix{Float64}

Forward pass of Chebyshev spectral convolution:
1. Compute Chebyshev coefficients via DCT-II.
2. Filter/multiply truncated Chebyshev modes.
3. Reconstruct nodal field via Inverse DCT (DCT-III).
"""
function forward_chebyshev_conv(conv::ChebyshevConv1D, x::Matrix{Float64})::Matrix{Float64}
    N_spatial, in_c = size(x)
    @assert in_c == conv.in_channels "Input channel mismatch"

    out_c = conv.out_channels
    modes = min(conv.modes, N_spatial)

    # DCT-II along spatial dimension
    c_coeffs = zeros(Float64, N_spatial, in_c)
    for c in 1:in_c
        c_coeffs[:, c] = dct_1d(x[:, c])
    end

    y_coeffs = zeros(Float64, N_spatial, out_c)
    for k in 1:modes
        for oc in 1:out_c
            s = 0.0
            for ic in 1:in_c
                s += conv.weights[oc, ic, k] * c_coeffs[k, ic]
            end
            y_coeffs[k, oc] = s
        end
    end

    # IDCT (DCT-III) along spatial dimension
    y = zeros(Float64, N_spatial, out_c)
    for oc in 1:out_c
        y[:, oc] = idct_1d(y_coeffs[:, oc])
    end

    return y
end

struct ChebyshevNeuralOperator1D <: AbstractNeuralOperator
    in_features::Int
    hidden_channels::Int
    out_features::Int
    modes::Int
    n_layers::Int

    P_weights::Matrix{Float64}
    P_bias::Vector{Float64}

    chebyshev_layers::Vector{ChebyshevConv1D}
    linear_bypass::Vector{Matrix{Float64}}

    Q1_weights::Matrix{Float64}
    Q1_bias::Vector{Float64}
    Q2_weights::Matrix{Float64}
    Q2_bias::Vector{Float64}

    function ChebyshevNeuralOperator1D(in_f::Int, out_f::Int;
                                       hidden_c::Int=32, modes::Int=16, n_layers::Int=4)
        P_w = randn(in_f, hidden_c) .* sqrt(2.0 / (in_f + hidden_c))
        P_b = zeros(Float64, hidden_c)

        c_layers = [ChebyshevConv1D(hidden_c, hidden_c, modes) for _ in 1:n_layers]
        bypass = [randn(hidden_c, hidden_c) .* sqrt(2.0 / (2 * hidden_c)) for _ in 1:n_layers]

        Q1_w = randn(hidden_c, 48) .* sqrt(2.0 / (hidden_c + 48))
        Q1_b = zeros(Float64, 48)

        Q2_w = randn(48, out_f) .* sqrt(2.0 / (48 + out_f))
        Q2_b = zeros(Float64, out_f)

        new(in_f, hidden_c, out_f, modes, n_layers, P_w, P_b, c_layers, bypass, Q1_w, Q1_b, Q2_w, Q2_b)
    end
end

function (cno::ChebyshevNeuralOperator1D)(x::Matrix{Float64})::Matrix{Float64}
    N_spatial, in_f = size(x)
    @assert in_f == cno.in_features "Input feature mismatch"

    v = x * cno.P_weights .+ cno.P_bias'

    for l in 1:cno.n_layers
        v_spec = forward_chebyshev_conv(cno.chebyshev_layers[l], v)
        v_lin = v * cno.linear_bypass[l]
        v = gelu.(v_spec .+ v_lin)
    end

    h = gelu.(v * cno.Q1_weights .+ cno.Q1_bias')
    return h * cno.Q2_weights .+ cno.Q2_bias'
end
