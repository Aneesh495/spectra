"""
    Spectra.Operators.FNO1D

1D Fourier Neural Operator (FNO-1D) for learning infinite-dimensional mappings between function spaces.
Operates directly in the frequency domain with spectral convolutions and mode truncation.
Mesh-independent and resolution-invariant: trained on coarse grids, evaluates on fine grids.
"""

struct SpectralConv1D <: AbstractSpectralOperator
    in_channels::Int
    out_channels::Int
    modes::Int # Number of Fourier modes to retain
    weights::Array{ComplexF64, 3} # (out_channels, in_channels, modes)

    function SpectralConv1D(in_c::Int, out_c::Int, modes::Int)
        scale = 1.0 / (in_c * out_c)
        # Random complex weights
        re = randn(out_c, in_c, modes) .* scale
        im = randn(out_c, in_c, modes) .* scale
        weights = ComplexF64.(re, im)
        new(in_c, out_c, modes, weights)
    end
end

"""
    forward_spectral_conv(conv::SpectralConv1D, x::Matrix{Float64}) -> Matrix{Float64}

Forward pass of 1D spectral convolution:
Input x is (N_spatial, in_channels).
1. FFT along spatial dimension: x_hat = FFT(x).
2. Multiply lowest `modes` frequencies by complex weights: y_hat[k] = W[k] * x_hat[k].
3. IFFT back to spatial domain: y = IFFT(y_hat).
"""
function forward_spectral_conv(conv::SpectralConv1D, x::Matrix{Float64})::Matrix{Float64}
    N_spatial, in_c = size(x)
    @assert in_c == conv.in_channels "Input channels mismatch"

    out_c = conv.out_channels
    modes = min(conv.modes, div(N_spatial, 2))

    # Compute 1D FFT for each channel
    x_hat = zeros(ComplexF64, N_spatial, in_c)
    for c in 1:in_c
        x_hat[:, c] = fft_1d(x[:, c])
    end

    y_hat = zeros(ComplexF64, N_spatial, out_c)

    # Complex matrix multiplication for truncated modes
    # Positive frequencies 1:modes
    for k in 1:modes
        for oc in 1:out_c
            s = ComplexF64(0.0, 0.0)
            for ic in 1:in_c
                s += conv.weights[oc, ic, k] * x_hat[k, ic]
            end
            y_hat[k, oc] = s
        end
    end

    # Inverse FFT along spatial dimension
    y = zeros(Float64, N_spatial, out_c)
    for oc in 1:out_c
        ifft_res = ifft_1d(y_hat[:, oc])
        y[:, oc] = real.(ifft_res)
    end

    return y
end

struct FourierNeuralOperator1D <: AbstractNeuralOperator
    in_features::Int
    hidden_channels::Int
    out_features::Int
    modes::Int
    n_layers::Int

    # Lifting layer
    P_weights::Matrix{Float64}
    P_bias::Vector{Float64}

    # Fourier layers
    spectral_layers::Vector{SpectralConv1D}
    linear_bypass::Vector{Matrix{Float64}}

    # Projection layers
    Q1_weights::Matrix{Float64}
    Q1_bias::Vector{Float64}
    Q2_weights::Matrix{Float64}
    Q2_bias::Vector{Float64}

    function FourierNeuralOperator1D(in_f::Int, out_f::Int;
                                     hidden_c::Int=32, modes::Int=16, n_layers::Int=4)
        P_w = randn(in_f, hidden_c) .* sqrt(2.0 / (in_f + hidden_c))
        P_b = zeros(Float64, hidden_c)

        s_layers = [SpectralConv1D(hidden_c, hidden_c, modes) for _ in 1:n_layers]
        bypass = [randn(hidden_c, hidden_c) .* sqrt(2.0 / (2 * hidden_c)) for _ in 1:n_layers]

        Q1_w = randn(hidden_c, 64) .* sqrt(2.0 / (hidden_c + 64))
        Q1_b = zeros(Float64, 64)

        Q2_w = randn(64, out_f) .* sqrt(2.0 / (64 + out_f))
        Q2_b = zeros(Float64, out_f)

        new(in_f, hidden_c, out_f, modes, n_layers, P_w, P_b, s_layers, bypass, Q1_w, Q1_b, Q2_w, Q2_b)
    end
end

"""
    (fno::FourierNeuralOperator1D)(x::Matrix{Float64}) -> Matrix{Float64}

Predict solution field from input field.
Input x is (N_spatial, in_features).
Returns output (N_spatial, out_features).
"""
function (fno::FourierNeuralOperator1D)(x::Matrix{Float64})::Matrix{Float64}
    N_spatial, in_f = size(x)
    @assert in_f == fno.in_features "Input features mismatch"

    # 1. Lifting: P(x)
    v = x * fno.P_weights .+ fno.P_bias'

    # 2. Fourier layers
    for l in 1:fno.n_layers
        v_spec = forward_spectral_conv(fno.spectral_layers[l], v)
        v_lin = v * fno.linear_bypass[l]
        v = gelu.(v_spec .+ v_lin)
    end

    # 3. Projection: Q(v)
    h = gelu.(v * fno.Q1_weights .+ fno.Q1_bias')
    out = h * fno.Q2_weights .+ fno.Q2_bias'

    return out
end

"""
    gelu(x::Float64) -> Float64

Gaussian Error Linear Unit activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3))).
"""
@inline function gelu(x::Float64)::Float64
    return 0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x^3)))
end
