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

end # module Spectra
