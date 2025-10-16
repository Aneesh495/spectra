module Spectra

"""
    Spectra.jl

Spectral operator learning, cellular sheaf cohomology, and continuous geometric
neural dynamics in pure Julia.
"""

# Standard libraries
using LinearAlgebra
using Statistics
using Random
using Printf
using Dates

# Core submodule
include("core/types.jl")
include("core/config.jl")
include("core/utils.jl")

# Math submodule
include("math/tensor.jl")
include("math/dual.jl")
include("math/reverse_ad.jl")
include("math/linalg.jl")
include("math/sparse.jl")

# Exports from Core
export AbstractSpectraType, AbstractTensor, AbstractDenseTensor, AbstractSparseTensor
export AbstractManifold, AbstractEmbeddedManifold, AbstractMatrixManifold, AbstractLieGroup
export AbstractCellComplex, AbstractSimplicialComplex, AbstractCellularSheaf, AbstractSheafLaplacian
export AbstractIntegrator, AbstractExplicitIntegrator, AbstractSymplecticIntegrator
export AbstractImplicitIntegrator, AbstractStochasticIntegrator
export AbstractNeuralOperator, AbstractSpectralOperator, AbstractWaveletOperator, AbstractPolynomialOperator
export AbstractKoopmanModel, AbstractTransferOperator

export GLOBAL_CONFIG, set_tolerance!, get_tolerance!, enable_c_acceleration!
export safe_sqrt, clamped_div, log_sum_exp, softmax, softmax!, horner
export is_symmetric_matrix, is_skew_symmetric, project_psd, gram_schmidt
export numerical_jacobian, numerical_gradient, numerical_hessian

# Exports from Math - Tensor
export Tensor, zeros_tensor, ones_tensor, randn_tensor, to_array
export is_contiguous, hadamard, contract, frobenius_norm

# Re-export key linear algebra functions
export norm, dot, I, Diagonal, Symmetric, tr, det, eigen, svd, qr

# Exports from Math - Dual & HyperDual
export Dual, HyperDual, value, partials
export forward_diff, forward_gradient, forward_jacobian, forward_hessian

# Exports from Math - Reverse AD
export TrackedReal, GradientTape, track, backward!, value_and_gradient, reverse_gradient
export relu, sigmoid

# Exports from Math - LinAlg
export gemm_blocked, gemm_blocked!, QRDecomposition, householder_qr
export SVDDecomposition, bidiagonal_svd, LanczosResult, lanczos, ArnoldiResult, arnoldi
export gmres, cg

# Exports from Math - Sparse
export COOMatrix, CSCMatrix, add_entry!, from_coo, sparse_identity
export spmv!, to_dense, sparse_cg, nnz

end # module Spectra
