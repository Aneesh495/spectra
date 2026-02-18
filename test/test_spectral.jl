using Test
using Spectra
using LinearAlgebra

@testset "Spectral Transforms & Wavelets" begin
    # 1D FFT & IFFT (Power of 2)
    x = randn(ComplexF64, 32)
    X_fft = fft_1d(x)
    x_rec = ifft_1d(X_fft)
    @test x_rec ≈ x atol=1e-12

    # 1D FFT Bluestein chirp-z (Non-power-of-2)
    x_odd = randn(ComplexF64, 37)
    X_odd_fft = fft_1d(x_odd)
    x_odd_rec = ifft_1d(X_odd_fft)
    @test x_odd_rec ≈ x_odd atol=1e-11

    # 2D FFT & IFFT
    x2d = randn(ComplexF64, 16, 16)
    X2d_fft = fft_2d(x2d)
    x2d_rec = ifft_2d(X2d_fft)
    @test x2d_rec ≈ x2d atol=1e-12

    # 1D DCT-II & IDCT (DCT-III)
    v_real = randn(32)
    c_dct = dct_1d(v_real)
    v_rec = idct_1d(c_dct)
    @test v_rec ≈ v_real atol=1e-12

    # Chebyshev differentiation matrix
    N_cheb = 10
    nodes = chebyshev_nodes(N_cheb)
    @test length(nodes) == N_cheb + 1
    D = chebyshev_diff_matrix(N_cheb)
    # Test derivative of f(x) = x^2 is 2x
    f_vals = nodes .^ 2
    df_vals = D * f_vals
    @test df_vals ≈ 2.0 .* nodes atol=1e-8

    # Haar Wavelet DWT & IDWT
    sig = randn(64)
    decomp_haar = forward_dwt_1d(sig, 3, :haar)
    sig_rec_haar = inverse_dwt_1d(decomp_haar)
    @test sig_rec_haar ≈ sig atol=1e-12

    # Daubechies-4 Wavelet DWT & IDWT
    decomp_d4 = forward_dwt_1d(sig, 2, :d4)
    sig_rec_d4 = inverse_dwt_1d(decomp_d4)
    @test sig_rec_d4 ≈ sig atol=1e-10
end
