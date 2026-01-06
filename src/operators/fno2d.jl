"""
    Spectra.Operators.FNO2D

2D Fourier Neural Operator (FNO-2D) for continuous spatial PDEs (Darcy Flow, 2D Navier-Stokes).
Executes 2D spectral convolutions in the frequency domain with multi-frequency truncation.
"""

struct SpectralConv2D <: AbstractSpectralOperator
    in_channels::Int
    out_channels::Int
    modes1::Int
    modes2::Int
    weights1::Array{ComplexF64, 4} # (out_c, in_c, modes1, modes2)
    weights2::Array{ComplexF64, 4} # (out_c, in_c, modes1, modes2) for upper corner

    function SpectralConv2D(in_c::Int, out_c::Int, m1::Int, m2::Int)
        scale = 1.0 / (in_c * out_c)
        w1_re = randn(out_c, in_c, m1, m2) .* scale
        w1_im = randn(out_c, in_c, m1, m2) .* scale

        w2_re = randn(out_c, in_c, m1, m2) .* scale
        w2_im = randn(out_c, in_c, m1, m2) .* scale

        new(in_c, out_c, m1, m2, ComplexF64.(w1_re, w1_im), ComplexF64.(w2_re, w2_im))
    end
end

"""
    forward_spectral_conv_2d(conv::SpectralConv2D, x::Array{Float64, 3}) -> Array{Float64, 3}

2D spectral convolution on input field x of size (Nx, Ny, in_channels).
"""
function forward_spectral_conv_2d(conv::SpectralConv2D, x::Array{Float64, 3})::Array{Float64, 3}
    Nx, Ny, in_c = size(x)
    @assert in_c == conv.in_channels "Input channel mismatch"

    out_c = conv.out_channels
    m1 = min(conv.modes1, div(Nx, 2))
    m2 = min(conv.modes2, div(Ny, 2))

    # Compute 2D FFT channel-wise
    x_hat = zeros(ComplexF64, Nx, Ny, in_c)
    for c in 1:in_c
        x_hat[:, :, c] = fft_2d(x[:, :, c])
    end

    y_hat = zeros(ComplexF64, Nx, Ny, out_c)

    # 1. Lower corner (0:m1, 0:m2)
    for k1 in 1:m1
        for k2 in 1:m2
            for oc in 1:out_c
                s = ComplexF64(0.0, 0.0)
                for ic in 1:in_c
                    s += conv.weights1[oc, ic, k1, k2] * x_hat[k1, k2, ic]
                end
                y_hat[k1, k2, oc] = s
            end
        end
    end

    # 2. Upper corner (Nx - m1 + 1:Nx, 0:m2)
    for k1 in 1:m1
        for k2 in 1:m2
            idx1 = Nx - m1 + k1
            for oc in 1:out_c
                s = ComplexF64(0.0, 0.0)
                for ic in 1:in_c
                    s += conv.weights2[oc, ic, k1, k2] * x_hat[idx1, k2, ic]
                end
                y_hat[idx1, k2, oc] = s
            end
        end
    end

    # Compute 2D IFFT channel-wise
    y = zeros(Float64, Nx, Ny, out_c)
    for oc in 1:out_c
        y[:, :, oc] = real.(ifft_2d(y_hat[:, :, oc]))
    end

    return y
end

struct FourierNeuralOperator2D <: AbstractNeuralOperator
    in_features::Int
    hidden_channels::Int
    out_features::Int
    modes1::Int
    modes2::Int
    n_layers::Int

    # Lifting
    P_weights::Matrix{Float64}
    P_bias::Vector{Float64}

    # Fourier layers
    spectral_layers::Vector{SpectralConv2D}
    linear_bypass::Vector{Matrix{Float64}}

    # Projection
    Q1_weights::Matrix{Float64}
    Q1_bias::Vector{Float64}
    Q2_weights::Matrix{Float64}
    Q2_bias::Vector{Float64}

    function FourierNeuralOperator2D(in_f::Int, out_f::Int;
                                     hidden_c::Int=24, modes1::Int=12, modes2::Int=12, n_layers::Int=4)
        P_w = randn(in_f, hidden_c) .* sqrt(2.0 / (in_f + hidden_c))
        P_b = zeros(Float64, hidden_c)

        s_layers = [SpectralConv2D(hidden_c, hidden_c, modes1, modes2) for _ in 1:n_layers]
        bypass = [randn(hidden_c, hidden_c) .* sqrt(2.0 / (2 * hidden_c)) for _ in 1:n_layers]

        Q1_w = randn(hidden_c, 32) .* sqrt(2.0 / (hidden_c + 32))
        Q1_b = zeros(Float64, 32)

        Q2_w = randn(32, out_f) .* sqrt(2.0 / (32 + out_f))
        Q2_b = zeros(Float64, out_f)

        new(in_f, hidden_c, out_f, modes1, modes2, n_layers, P_w, P_b, s_layers, bypass, Q1_w, Q1_b, Q2_w, Q2_b)
    end
end

"""
    (fno::FourierNeuralOperator2D)(x::Array{Float64, 3}) -> Array{Float64, 3}

Forward pass of 2D Fourier Neural Operator.
Input is (Nx, Ny, in_features). Returns (Nx, Ny, out_features).
"""
function (fno::FourierNeuralOperator2D)(x::Array{Float64, 3})::Array{Float64, 3}
    Nx, Ny, in_f = size(x)
    @assert in_f == fno.in_features "Input feature mismatch"

    # Flatten spatial grid for point-wise operations: (Nx * Ny, in_f)
    flat_x = reshape(x, (Nx * Ny, in_f))
    flat_v = flat_x * fno.P_weights .+ fno.P_bias'
    v = reshape(flat_v, (Nx, Ny, fno.hidden_channels))

    # Fourier layers
    for l in 1:fno.n_layers
        v_spec = forward_spectral_conv_2d(fno.spectral_layers[l], v)
        # Linear pointwise bypass
        flat_curr = reshape(v, (Nx * Ny, fno.hidden_channels))
        flat_lin = flat_curr * fno.linear_bypass[l]
        v_lin = reshape(flat_lin, (Nx, Ny, fno.hidden_channels))

        v = gelu.(v_spec .+ v_lin)
    end

    # Projection
    flat_final = reshape(v, (Nx * Ny, fno.hidden_channels))
    flat_h = gelu.(flat_final * fno.Q1_weights .+ fno.Q1_bias')
    flat_out = flat_h * fno.Q2_weights .+ fno.Q2_bias'

    return reshape(flat_out, (Nx, Ny, fno.out_features))
end
