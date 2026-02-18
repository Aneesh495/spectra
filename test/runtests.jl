"""
    Spectra Test Suite Runner

Unified test driver executing modular unit and integration tests across all Spectra components:
- Tensor engine
- Forward-mode Dual & HyperDual automatic differentiation
- Reverse-mode AD and dynamic computational graph
- Linear algebra, Householder QR, SVD, and Krylov subspace solvers
- Sparse matrices (COO/CSC) and sparse CG
- Spectral transforms (1D/2D FFT, DCT, Chebyshev, Haar/D4 wavelets)
- Riemannian manifolds and manifold optimization
- Cellular sheaves, Hodge Laplacians, and persistent homology
- Continuous dynamics, symplectic integrators, adjoints, and Neural ODEs
- Neural operators (FNO, CNO, WNO, DeepONet) and numerical PDE solvers
- Koopman operator theory (DMD, EDMD, Deep Koopman) and Perron-Frobenius transfer operators
- Native C SIMD acceleration kernel FFI bridge
"""

using Test
using Spectra

println("=================================================================")
println("                 Spectra Complete Test Suite                     ")
println("=================================================================")

@testset "Spectra Core & Mathematics" begin
    include("test_tensor.jl")
    include("test_dual.jl")
    include("test_reverse_ad.jl")
    include("test_linalg.jl")
    include("test_sparse.jl")
    include("test_spectral.jl")
end

@testset "Spectra Geometry & Sheaves" begin
    include("test_manifold.jl")
    include("test_sheaves.jl")
end

@testset "Spectra Continuous Dynamics" begin
    include("test_dynamics.jl")
end

@testset "Spectra Neural Operators & PDEs" begin
    include("test_operators.jl")
end

@testset "Spectra Koopman & Transfer Operators" begin
    include("test_koopman.jl")
end

@testset "Spectra Native C FFI" begin
    include("test_ffi.jl")
end

println("=================================================================")
println("         ALL SPECTRA TEST SUITES PASSED FLAWLESSLY!              ")
println("=================================================================")
