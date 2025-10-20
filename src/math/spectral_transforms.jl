"""
    Spectra.Math.SpectralTransforms

Fast Fourier Transforms (1D/2D Cooley-Tukey & Bluestein Chirp-Z), Discrete Cosine Transforms (DCT-II/III),
and Chebyshev spectral collocation operators on Gauss-Lobatto grids.
"""

# ====================================================================
# Cooley-Tukey Radix-2 & Bluestein Arbitrary-Length Fast Fourier Transform
# ====================================================================

function bit_reverse_permutation(n::Int)::Vector{Int}
    perm = zeros(Int, n)
    bits = trailing_zeros(n)
    for i in 0:(n - 1)
        rev = 0
        temp = i
        for _ in 1:bits
            rev = (rev << 1) | (temp & 1)
            temp >>= 1
        end
        perm[i + 1] = rev + 1
    end
    return perm
end

"""
    fft_radix2!(out::Vector{ComplexF64}, x::Vector{ComplexF64}, inverse::Bool=false)

In-place Radix-2 Cooley-Tukey FFT for power-of-2 lengths.
"""
function fft_radix2!(out::Vector{ComplexF64}, x::Vector{ComplexF64}, inverse::Bool=false)
    n = length(x)
    @assert ispow2(n) "Length must be a power of 2 for Radix-2 FFT"

    perm = bit_reverse_permutation(n)
    @inbounds for i in 1:n
        out[i] = x[perm[i]]
    end

    sign_dir = inverse ? 1.0 : -1.0
    len = 2
    while len <= n
        half = div(len, 2)
        angle = sign_dir * 2.0 * π / len
        w_step = cis(angle)

        for i in 1:len:n
            w = ComplexF64(1.0, 0.0)
            for j in 0:(half - 1)
                u = out[i + j]
                v = out[i + j + half] * w
                out[i + j] = u + v
                out[i + j + half] = u - v
                w *= w_step
            end
        end
        len <<= 1
    end

    if inverse
        inv_n = 1.0 / n
        @inbounds for i in 1:n
            out[i] *= inv_n
        end
    end
    return out
end

"""
    fft_1d(x::Vector{ComplexF64}) -> Vector{ComplexF64}

1D Fast Fourier Transform. Uses Radix-2 for power-of-2 sizes and Bluestein Chirp-Z algorithm
for general arbitrary lengths.
"""
function fft_1d(x::Vector{ComplexF64})::Vector{ComplexF64}
    n = length(x)
    n <= 1 && return copy(x)

    if ispow2(n)
        out = similar(x)
        return fft_radix2!(out, x, false)
    end

    # Bluestein's Chirp-Z algorithm for arbitrary non-power-of-2 n
    # X_k = e^(-i pi k^2 / n) sum_{j=0}^{n-1} [x_j e^(-i pi j^2 / n)] [e^(i pi (k-j)^2 / n)]
    m = nextpow(2, 2 * n - 1)

    a = zeros(ComplexF64, m)
    b = zeros(ComplexF64, m)

    for j in 0:(n - 1)
        angle = -π * (j^2) / n
        chirp = cis(angle)
        a[j + 1] = x[j + 1] * chirp
    end

    for k in 0:(n - 1)
        angle = π * (k^2) / n
        chirp = cis(angle)
        b[k + 1] = chirp
        if k > 0
            b[m - k + 1] = chirp
        end
    end

    # Cyclic convolution via Radix-2 FFT
    fa = zeros(ComplexF64, m)
    fb = zeros(ComplexF64, m)
    fft_radix2!(fa, a, false)
    fft_radix2!(fb, b, false)

    fc = fa .* fb

    c = zeros(ComplexF64, m)
    fft_radix2!(c, fc, true)

    out = zeros(ComplexF64, n)
    for k in 0:(n - 1)
        angle = -π * (k^2) / n
        chirp = cis(angle)
        out[k + 1] = c[k + 1] * chirp
    end

    return out
end

function fft_1d(x::Vector{Float64})::Vector{ComplexF64}
    return fft_1d(ComplexF64.(x))
end

"""
    ifft_1d(X::Vector{ComplexF64}) -> Vector{ComplexF64}

Inverse 1D Fast Fourier Transform via conjugation duality: ifft(X) = conj(fft(conj(X))) / n.
"""
function ifft_1d(X::Vector{ComplexF64})::Vector{ComplexF64}
    n = length(X)
    n <= 1 && return copy(X)

    if ispow2(n)
        out = similar(X)
        return fft_radix2!(out, X, true)
    end

    # Duality rule
    x_conj = conj.(X)
    res = fft_1d(x_conj)
    return (1.0 / n) .* conj.(res)
end

"""
    fft_2d(M::Matrix{ComplexF64}) -> Matrix{ComplexF64}

2D Fast Fourier Transform computed via row-column decomposition.
"""
function fft_2d(M::Matrix{ComplexF64})::Matrix{ComplexF64}
    r, c = size(M)
    temp = zeros(ComplexF64, r, c)

    # 1D FFT along rows
    for i in 1:r
        row_vec = [M[i, j] for j in 1:c]
        row_fft = fft_1d(row_vec)
        for j in 1:c
            temp[i, j] = row_fft[j]
        end
    end

    out = zeros(ComplexF64, r, c)
    # 1D FFT along columns
    for j in 1:c
        col_vec = [temp[i, j] for i in 1:r]
        col_fft = fft_1d(col_vec)
        for i in 1:r
            out[i, j] = col_fft[i]
        end
    end

    return out
end
fft_2d(M::Matrix{Float64}) = fft_2d(ComplexF64.(M))

"""
    ifft_2d(M::Matrix{ComplexF64}) -> Matrix{ComplexF64}

Inverse 2D Fast Fourier Transform.
"""
function ifft_2d(M::Matrix{ComplexF64})::Matrix{ComplexF64}
    r, c = size(M)
    temp = zeros(ComplexF64, r, c)

    for i in 1:r
        row_vec = [M[i, j] for j in 1:c]
        row_ifft = ifft_1d(row_vec)
        for j in 1:c
            temp[i, j] = row_ifft[j]
        end
    end

    out = zeros(ComplexF64, r, c)
    for j in 1:c
        col_vec = [temp[i, j] for i in 1:r]
        col_ifft = ifft_1d(col_vec)
        for i in 1:r
            out[i, j] = col_ifft[i]
        end
    end

    return out
end

# ====================================================================
# Discrete Cosine Transform (DCT-II and DCT-III / IDCT)
# ====================================================================

"""
    dct_1d(x::Vector{Float64}) -> Vector{Float64}

Compute standard Discrete Cosine Transform (DCT-II):
X_k = 2 sum_{n=0}^{N-1} x_n cos(pi (2n+1) k / (2 N)).
Computed via FFT of mirrored sequence in O(N log N) time.
"""
function dct_1d(x::Vector{Float64})::Vector{Float64}
    n = length(x)
    n == 0 && return Float64[]
    n == 1 && return [2.0 * x[1]]

    # Form 2N symmetric extension
    ext = zeros(ComplexF64, 2 * n)
    for i in 1:n
        ext[i] = ComplexF64(x[i], 0.0)
        ext[2 * n - i + 1] = ComplexF64(x[i], 0.0)
    end

    f_ext = fft_1d(ext)
    out = zeros(Float64, n)

    for k in 0:(n - 1)
        phase = cis(-π * k / (2 * n))
        val = f_ext[k + 1] * phase
        out[k + 1] = real(val)
    end

    return out
end

"""
    idct_1d(X::Vector{Float64}) -> Vector{Float64}

Inverse Discrete Cosine Transform (DCT-III).
"""
function idct_1d(X::Vector{Float64})::Vector{Float64}
    n = length(X)
    n == 0 && return Float64[]
    n == 1 && return [0.5 * X[1]]

    ext = zeros(ComplexF64, 2 * n)
    ext[1] = ComplexF64(0.5 * X[1], 0.0)
    for k in 1:(n - 1)
        phase = cis(π * k / (2 * n))
        ext[k + 1] = ComplexF64(X[k + 1] * 0.5 * phase)
        ext[2 * n - k + 1] = ComplexF64(X[k + 1] * 0.5 * conj(phase))
    end

    f_ext = ifft_1d(ext)
    out = zeros(Float64, n)
    for i in 1:n
        out[i] = real(f_ext[i]) * 2.0
    end
    return out
end

# ====================================================================
# Chebyshev Polynomials & Gauss-Lobatto Spectral Collocation
# ====================================================================

"""
    chebyshev_nodes(N::Int) -> Vector{Float64}

Compute N+1 Chebyshev-Gauss-Lobatto collocation nodes in [-1, 1]:
x_j = cos(pi j / N), j = 0, ..., N.
"""
function chebyshev_nodes(N::Int)::Vector{Float64}
    @assert N >= 1 "Polynomial degree N must be >= 1"
    nodes = zeros(Float64, N + 1)
    for j in 0:N
        nodes[j + 1] = cos(π * j / N)
    end
    return nodes
end

"""
    chebyshev_polynomial(n::Int, x::Float64) -> Float64

Evaluate Chebyshev polynomial of the first kind T_n(x) at x in [-1, 1]
via stable three-term recurrence.
"""
function chebyshev_polynomial(n::Int, x::Float64)::Float64
    n < 0 && throw(DomainError(n, "Degree must be non-negative"))
    n == 0 && return 1.0
    n == 1 && return x

    t_prev = 1.0
    t_curr = x
    two_x = 2.0 * x

    for _ in 2:n
        t_next = two_x * t_curr - t_prev
        t_prev = t_curr
        t_curr = t_next
    end
    return t_curr
end

"""
    chebyshev_diff_matrix(N::Int) -> Matrix{Float64}

Construct the (N+1) x (N+1) Chebyshev spectral differentiation matrix D.
Computes exact polynomial derivative values at Lobatto nodes: u' = D u.
"""
function chebyshev_diff_matrix(N::Int)::Matrix{Float64}
    @assert N >= 1 "N must be >= 1"
    x = chebyshev_nodes(N)
    c = zeros(Float64, N + 1)
    c[1] = 2.0
    c[N + 1] = 2.0
    for j in 2:N
        c[j] = 1.0
    end

    D = zeros(Float64, N + 1, N + 1)
    for i in 0:N
        for j in 0:N
            if i != j
                num = c[i + 1] * ((-1.0)^(i + j))
                den = c[j + 1] * (x[i + 1] - x[j + 1])
                D[i + 1, j + 1] = num / den
            end
        end
    end

    # Negative sum trick for diagonal stability
    for i in 1:(N + 1)
        s = 0.0
        for j in 1:(N + 1)
            if i != j
                s += D[i, j]
            end
        end
        D[i, i] = -s
    end

    return D
end
