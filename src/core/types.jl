"""
    Spectra.Core.Types

Foundational type hierarchy and mathematical abstractions for Spectra.
All high level mathematical entities inherit from these base types.
"""

abstract type AbstractSpectraType end

# Mathematical Tensor Abstractions
abstract type AbstractTensor{T, N} <: AbstractSpectraType end
abstract type AbstractDenseTensor{T, N} <: AbstractTensor{T, N} end
abstract type AbstractSparseTensor{T, N} <: AbstractTensor{T, N} end

# Differential Geometry Abstractions
abstract type AbstractManifold{N} <: AbstractSpectraType end
abstract type AbstractEmbeddedManifold{N, E} <: AbstractManifold{N} end
abstract type AbstractMatrixManifold{Rows, Cols} <: AbstractManifold{Rows} end
abstract type AbstractLieGroup <: AbstractManifold{0} end

# Algebraic Topology & Sheaf Abstractions
abstract type AbstractCellComplex <: AbstractSpectraType end
abstract type AbstractSimplicialComplex <: AbstractCellComplex end
abstract type AbstractCellularSheaf <: AbstractSpectraType end
abstract type AbstractSheafLaplacian <: AbstractSpectraType end

# Dynamical Systems & Integrators
abstract type AbstractIntegrator <: AbstractSpectraType end
abstract type AbstractExplicitIntegrator <: AbstractIntegrator end
abstract type AbstractSymplecticIntegrator <: AbstractIntegrator end
abstract type AbstractImplicitIntegrator <: AbstractIntegrator end
abstract type AbstractStochasticIntegrator <: AbstractIntegrator end

# Infinite Dimensional Operator Learning
abstract type AbstractNeuralOperator <: AbstractSpectraType end
abstract type AbstractSpectralOperator <: AbstractNeuralOperator end
abstract type AbstractWaveletOperator <: AbstractNeuralOperator end
abstract type AbstractPolynomialOperator <: AbstractNeuralOperator end

# Spectral & Koopman Operator Theory
abstract type AbstractKoopmanModel <: AbstractSpectraType end
abstract type AbstractTransferOperator <: AbstractSpectraType end

# Precision and Execution Traits
abstract type AbstractPrecisionMode end
struct PrecisionFP64 <: AbstractPrecisionMode end
struct PrecisionFP32 <: AbstractPrecisionMode end
struct PrecisionComplex64 <: AbstractPrecisionMode end

abstract type AbstractDeviceBackend end
struct CPUBackend <: AbstractDeviceBackend end
struct SIMDBackend <: AbstractDeviceBackend end

# Coordinate and Layout Enums
@enum MemoryOrder ColMajor RowMajor
@enum NormType L1Norm L2Norm LinfNorm FrobeniusNorm NuclearNorm
