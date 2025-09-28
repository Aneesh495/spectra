"""
    Spectra.Math.Tensor

High-performance N-dimensional strided tensor abstraction with zero-copy views,
generalized multidimensional broadcasting, and tensor contractions.
"""

struct Tensor{T, N} <: AbstractDenseTensor{T, N}
    data::Vector{T}
    shape::NTuple{N, Int}
    strides::NTuple{N, Int}
    offset::Int

    function Tensor{T, N}(data::Vector{T}, shape::NTuple{N, Int}, strides::NTuple{N, Int}, offset::Int=0) where {T, N}
        @assert all(s >= 0 for s in shape) "Dimensions must be non-negative"
        @assert offset >= 0 "Offset must be non-negative"
        new{T, N}(data, shape, strides, offset)
    end
end

# Primary constructors
function Tensor(data::Vector{T}, shape::NTuple{N, Int}) where {T, N}
    strides = compute_column_major_strides(shape)
    expected_len = prod(shape)
    @assert length(data) >= expected_len "Data vector length $(length(data)) smaller than shape product $expected_len"
    return Tensor{T, N}(data, shape, strides, 0)
end

function Tensor(arr::Array{T, N}) where {T, N}
    shape = size(arr)
    data = vec(copy(arr))
    strides = compute_column_major_strides(shape)
    return Tensor{T, N}(data, shape, strides, 0)
end

function Tensor{T}(::UndefInitializer, shape::NTuple{N, Int}) where {T, N}
    total_elems = prod(shape)
    data = Vector{T}(undef, total_elems)
    strides = compute_column_major_strides(shape)
    return Tensor{T, N}(data, shape, strides, 0)
end

function Tensor(shape::Vararg{Int, N}) where {N}
    return Tensor{Float64}(undef, shape)
end

function zeros_tensor(::Type{T}, shape::NTuple{N, Int}) where {T, N}
    total_elems = prod(shape)
    data = zeros(T, total_elems)
    strides = compute_column_major_strides(shape)
    return Tensor{T, N}(data, shape, strides, 0)
end
zeros_tensor(shape::Vararg{Int, N}) where {N} = zeros_tensor(Float64, shape)

function ones_tensor(::Type{T}, shape::NTuple{N, Int}) where {T, N}
    total_elems = prod(shape)
    data = fill(one(T), total_elems)
    strides = compute_column_major_strides(shape)
    return Tensor{T, N}(data, shape, strides, 0)
end
ones_tensor(shape::Vararg{Int, N}) where {N} = ones_tensor(Float64, shape)

function randn_tensor(::Type{T}, shape::NTuple{N, Int}) where {T, N}
    total_elems = prod(shape)
    data = randn(T, total_elems)
    strides = compute_column_major_strides(shape)
    return Tensor{T, N}(data, shape, strides, 0)
end
randn_tensor(shape::Vararg{Int, N}) where {N} = randn_tensor(Float64, shape)

# Stride computations
function compute_column_major_strides(shape::NTuple{N, Int})::NTuple{N, Int} where {N}
    if N == 0
        return ()
    end
    strides = Vector{Int}(undef, N)
    strides[1] = 1
    for i in 2:N
        strides[i] = strides[i - 1] * shape[i - 1]
    end
    return NTuple{N, Int}(strides)
end

# Basic properties
Base.size(t::Tensor) = t.shape
Base.size(t::Tensor, d::Int) = (d <= length(t.shape)) ? t.shape[d] : 1
Base.length(t::Tensor) = prod(t.shape)
Base.ndims(::Tensor{T, N}) where {T, N} = N
Base.eltype(::Tensor{T}) where {T} = T

@inline function is_contiguous(t::Tensor{T, N}) where {T, N}
    expected = compute_column_major_strides(t.shape)
    return t.strides == expected && t.offset == 0
end

# Multi-index to linear index mapping
@inline function linear_index(t::Tensor{T, N}, indices::NTuple{N, Int})::Int where {T, N}
    idx = t.offset + 1
    @inbounds for i in 1:N
        @boundscheck (1 <= indices[i] <= t.shape[i]) || throw(BoundsError(t, indices))
        idx += (indices[i] - 1) * t.strides[i]
    end
    return idx
end

Base.getindex(t::Tensor{T, N}, indices::Vararg{Int, N}) where {T, N} = t.data[linear_index(t, indices)]
Base.setindex!(t::Tensor{T, N}, val, indices::Vararg{Int, N}) where {T, N} = (t.data[linear_index(t, indices)] = convert(T, val))

# 1D linear indexing fallback for contiguous tensors
function Base.getindex(t::Tensor{T, N}, i::Int) where {T, N}
    @boundscheck (1 <= i <= length(t)) || throw(BoundsError(t, i))
    if is_contiguous(t)
        return t.data[i]
    end
    # Cartesian unraveling
    rem = i - 1
    idx = zeros(Int, N)
    for d in 1:N
        idx[d] = (rem % t.shape[d]) + 1
        rem = div(rem, t.shape[d])
    end
    return t[idx...]
end

function Base.setindex!(t::Tensor{T, N}, val, i::Int) where {T, N}
    @boundscheck (1 <= i <= length(t)) || throw(BoundsError(t, i))
    if is_contiguous(t)
        t.data[i] = convert(T, val)
        return val
    end
    rem = i - 1
    idx = zeros(Int, N)
    for d in 1:N
        idx[d] = (rem % t.shape[d]) + 1
        rem = div(rem, t.shape[d])
    end
    t[idx...] = val
    return val
end

# Conversion to standard Julia Array
function to_array(t::Tensor{T, N})::Array{T, N} where {T, N}
    arr = Array{T, N}(undef, t.shape)
    if is_contiguous(t)
        copyto!(arr, 1, t.data, 1, length(t))
    else
        for I in CartesianIndices(t.shape)
            arr[I] = t[Tuple(I)...]
        end
    end
    return arr
end

# Reshape & View Operations (Zero-Copy when possible)
function Base.reshape(t::Tensor{T, N}, new_shape::NTuple{M, Int}) where {T, N, M}
    @assert prod(new_shape) == length(t) "Total elements must be invariant under reshape"
    if is_contiguous(t)
        new_strides = compute_column_major_strides(new_shape)
        return Tensor{T, M}(t.data, new_shape, new_strides, t.offset)
    else
        # Materialize if striding is non-contiguous
        return Tensor(to_array(t)) |> t_cont -> reshape(t_cont, new_shape)
    end
end
Base.reshape(t::Tensor, new_shape::Vararg{Int, M}) where {M} = reshape(t, new_shape)

"""
    permutedims(t::Tensor{T, N}, perm::NTuple{N, Int}) where {T, N}

Permute tensor dimensions via strided view manipulation without data copy.
"""
function Base.permutedims(t::Tensor{T, N}, perm::NTuple{N, Int}) where {T, N}
    @assert sort(collect(perm)) == collect(1:N) "Permutation must be a bijection of 1:N"
    new_shape = ntuple(i -> t.shape[perm[i]], N)
    new_strides = ntuple(i -> t.strides[perm[i]], N)
    return Tensor{T, N}(t.data, new_shape, new_strides, t.offset)
end
Base.permutedims(t::Tensor{T, N}, perm::Vararg{Int, N}) where {T, N} = permutedims(t, perm)

"""
    transpose(t::Tensor{T, 2})

Matrix transpose via stride reversal.
"""
function Base.transpose(t::Tensor{T, 2}) where {T}
    return permutedims(t, (2, 1))
end

# Tensor Arithmetic
function Base.:+(A::Tensor{T, N}, B::Tensor{T, N}) where {T, N}
    @assert A.shape == B.shape "Tensor shapes must match for addition: $(A.shape) vs $(B.shape)"
    out = zeros_tensor(T, A.shape)
    if is_contiguous(A) && is_contiguous(B)
        @inbounds @simd for i in 1:length(A)
            out.data[i] = A.data[i] + B.data[i]
        end
    else
        for I in CartesianIndices(A.shape)
            out[Tuple(I)...] = A[Tuple(I)...] + B[Tuple(I)...]
        end
    end
    return out
end

function Base.:-(A::Tensor{T, N}, B::Tensor{T, N}) where {T, N}
    @assert A.shape == B.shape "Tensor shapes must match for subtraction"
    out = zeros_tensor(T, A.shape)
    if is_contiguous(A) && is_contiguous(B)
        @inbounds @simd for i in 1:length(A)
            out.data[i] = A.data[i] - B.data[i]
        end
    else
        for I in CartesianIndices(A.shape)
            out[Tuple(I)...] = A[Tuple(I)...] - B[Tuple(I)...]
        end
    end
    return out
end

function Base.:*(alpha::Number, A::Tensor{T, N}) where {T, N}
    c_alpha = convert(T, alpha)
    out = zeros_tensor(T, A.shape)
    if is_contiguous(A)
        @inbounds @simd for i in 1:length(A)
            out.data[i] = c_alpha * A.data[i]
        end
    else
        for I in CartesianIndices(A.shape)
            out[Tuple(I)...] = c_alpha * A[Tuple(I)...]
        end
    end
    return out
end
Base.:*(A::Tensor, alpha::Number) = alpha * A
Base.:/(A::Tensor{T, N}, alpha::Number) where {T, N} = (one(T) / convert(T, alpha)) * A

"""
    hadamard(A::Tensor{T, N}, B::Tensor{T, N})

Elementwise Hadamard product.
"""
function hadamard(A::Tensor{T, N}, B::Tensor{T, N}) where {T, N}
    @assert A.shape == B.shape "Tensor shapes must match for Hadamard product"
    out = zeros_tensor(T, A.shape)
    if is_contiguous(A) && is_contiguous(B)
        @inbounds @simd for i in 1:length(A)
            out.data[i] = A.data[i] * B.data[i]
        end
    else
        for I in CartesianIndices(A.shape)
            out[Tuple(I)...] = A[Tuple(I)...] * B[Tuple(I)...]
        end
    end
    return out
end

"""
    contract(A::Tensor{T, NA}, B::Tensor{T, NB}, dimsA::NTuple{K, Int}, dimsB::NTuple{K, Int})

General tensor contraction along specified dimensions dimsA of A and dimsB of B.
Reduces the specified indices via sum over product.
"""
function contract(A::Tensor{T, NA}, B::Tensor{T, NB}, dimsA::NTuple{K, Int}, dimsB::NTuple{K, Int}) where {T, NA, NB, K}
    @assert K <= NA && K <= NB "Cannot contract more dimensions than tensor order"
    for k in 1:K
        @assert A.shape[dimsA[k]] == B.shape[dimsB[k]] "Contracted dimension lengths must match: $(A.shape[dimsA[k]]) vs $(B.shape[dimsB[k]])"
    end

    freeA = filter(d -> !(d in dimsA), 1:NA)
    freeB = filter(d -> !(d in dimsB), 1:NB)

    permA = (freeA..., dimsA...)
    permB = (dimsB..., freeB...)

    At = permutedims(A, permA)
    Bt = permutedims(B, permB)

    # Flatten into matrices for BLAS-accelerated GEMM
    dim_freeA = prod(ntuple(i -> At.shape[i], length(freeA)); init=1)
    dim_contract = prod(ntuple(i -> At.shape[length(freeA) + i], K); init=1)
    dim_freeB = prod(ntuple(i -> Bt.shape[K + i], length(freeB)); init=1)

    matA = reshape(to_array(At), (dim_freeA, dim_contract))
    matB = reshape(to_array(Bt), (dim_contract, dim_freeB))

    matC = matA * matB

    out_shape = (ntuple(i -> A.shape[freeA[i]], length(freeA))..., ntuple(i -> B.shape[freeB[i]], length(freeB))...)
    if isempty(out_shape)
        # Scalar output wrapped in rank-0 tensor
        return Tensor(vec(matC), ())
    end
    return Tensor(reshape(matC, out_shape))
end

"""
    frobenius_norm(t::Tensor{T}) where {T}

Compute Frobenius / Euclidean L2 norm of the tensor.
"""
function frobenius_norm(t::Tensor{T})::Float64 where {T}
    s = zero(Float64)
    if is_contiguous(t)
        @inbounds for x in t.data
            s += Float64(abs2(x))
        end
    else
        for I in CartesianIndices(t.shape)
            s += Float64(abs2(t[Tuple(I)...]))
        end
    end
    return sqrt(s)
end
