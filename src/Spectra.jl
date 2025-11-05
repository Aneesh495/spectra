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
include("math/spectral_transforms.jl")
include("math/optimization.jl")

# Geometry submodule
include("geometry/manifold.jl")
include("geometry/sphere.jl")
include("geometry/hyperbolic.jl")
include("geometry/stiefel.jl")
include("geometry/grassmannian.jl")
include("geometry/spd.jl")
include("geometry/so3_se3.jl")

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

# Exports from Math - Spectral Transforms
export fft_1d, ifft_1d, fft_2d, ifft_2d, dct_1d, idct_1d
export chebyshev_nodes, chebyshev_polynomial, chebyshev_diff_matrix

# Exports from Math - Optimization
export AbstractOptimizer, SGDMomentum, AdamOptimizer, LBFGSOptimizer
export update!, optimize_lbfgs, optimize

# Exports from Geometry - Manifolds
export manifold_dim, ambient_dim, inner_product, norm_tangent, project_tangent
export check_point, check_tangent, exp_map, log_map, geodesic_distance
export parallel_transport, retraction, geodesic
export Sphere, random_point, PoincareBall, mobius_add, LorentzHyperboloid
export minkowski_dot, poincare_to_lorentz, lorentz_to_poincare
export Stiefel, cayley_retraction, Grassmannian, principal_angles
export SPDManifold, matrix_sqrt_sym, matrix_log_sym, matrix_exp_sym
export SO3Group, hat_so3, vee_so3, exp_so3, log_so3
export SE3Group, hat_se3, vee_se3, exp_se3, log_se3, left_jacobian_so3

end # module Spectra
