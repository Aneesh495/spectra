"""
    Spectra.Math.Sparse

Compressed Sparse Column (CSC) and Coordinate (COO) matrix structures,
high-throughput SpMV, sparse transposition, and incomplete factorization preconditioners.
"""

# Coordinate (COO) Matrix format
mutable struct COOMatrix{T} <: AbstractSparseTensor{T, 2}
    m::Int
    n::Int
    rows::Vector{Int}
    cols::Vector{Int}
    vals::Vector{T}

    function COOMatrix{T}(m::Int, n::Int) where {T}
        @assert m >= 0 && n >= 0 "Matrix dimensions must be non-negative"
        new{T}(m, n, Int[], Int[], T[])
    end

    function COOMatrix{T}(m::Int, n::Int, rows::Vector{Int}, cols::Vector{Int}, vals::Vector{T}) where {T}
        @assert length(rows) == length(cols) == length(vals) "Coordinate vectors must have identical length"
        new{T}(m, n, rows, cols, vals)
    end
end

COOMatrix(m::Int, n::Int) = COOMatrix{Float64}(m, n)
COOMatrix(m::Int, n::Int, rows::Vector{Int}, cols::Vector{Int}, vals::Vector{T}) where {T} = COOMatrix{T}(m, n, rows, cols, vals)

function add_entry!(coo::COOMatrix{T}, r::Int, c::Int, v::T) where {T}
    @assert 1 <= r <= coo.m && 1 <= c <= coo.n "Index ($r, $c) out of bounds ($(coo.m), $(coo.n))"
    push!(coo.rows, r)
    push!(coo.cols, c)
    push!(coo.vals, v)
    return coo
end

# Compressed Sparse Column (CSC) Matrix format
struct CSCMatrix{T} <: AbstractSparseTensor{T, 2}
    m::Int
    n::Int
    colptr::Vector{Int}
    rowval::Vector{Int}
    nzval::Vector{T}

    function CSCMatrix{T}(m::Int, n::Int, colptr::Vector{Int}, rowval::Vector{Int}, nzval::Vector{T}) where {T}
        @assert length(colptr) == n + 1 "colptr length must be n + 1"
        @assert length(rowval) == length(nzval) "rowval and nzval lengths must match"
        new{T}(m, n, colptr, rowval, nzval)
    end
end

Base.size(A::CSCMatrix) = (A.m, A.n)
Base.size(A::CSCMatrix, d::Int) = d == 1 ? A.m : (d == 2 ? A.n : 1)
Base.length(A::CSCMatrix) = A.m * A.n
Base.eltype(::CSCMatrix{T}) where {T} = T
nnz(A::CSCMatrix) = length(A.nzval)

"""
    from_coo(coo::COOMatrix{T}) -> CSCMatrix{T}

Convert COO matrix to CSC matrix, sorting column entries and summing duplicate entries.
"""
function from_coo(coo::COOMatrix{T})::CSCMatrix{T} where {T}
    m, n = coo.m, coo.n
    nnz_coo = length(coo.vals)

    if nnz_coo == 0
        return CSCMatrix{T}(m, n, fill(1, n + 1), Int[], T[])
    end

    # Group by column then row
    perm = sortperm(1:nnz_coo, by=i -> (coo.cols[i], coo.rows[i]))

    colptr = zeros(Int, n + 1)
    colptr[1] = 1

    rowval = Int[]
    nzval = T[]

    curr_col = 1
    i = 1
    while i <= nnz_coo
        p = perm[i]
        c = coo.cols[p]
        r = coo.rows[p]
        v = coo.vals[p]

        # Advance colptr for empty columns
        while curr_col < c
            curr_col += 1
            colptr[curr_col] = length(rowval) + 1
        end

        # Accumulate duplicates
        j = i + 1
        while j <= nnz_coo && coo.cols[perm[j]] == c && coo.rows[perm[j]] == r
            v += coo.vals[perm[j]]
            j += 1
        end

        if abs(v) > 1e-16
            push!(rowval, r)
            push!(nzval, v)
        end
        i = j
    end

    while curr_col <= n
        curr_col += 1
        colptr[curr_col] = length(rowval) + 1
    end

    return CSCMatrix{T}(m, n, colptr, rowval, nzval)
end

"""
    sparse_identity(n::Int, ::Type{T}=Float64) -> CSCMatrix{T}

Create n x n sparse identity matrix in CSC format.
"""
function sparse_identity(n::Int, ::Type{T}=Float64)::CSCMatrix{T} where {T}
    colptr = collect(1:(n + 1))
    rowval = collect(1:n)
    nzval = fill(one(T), n)
    return CSCMatrix{T}(n, n, colptr, rowval, nzval)
end

"""
    spmv!(y::Vector{T}, A::CSCMatrix{T}, x::Vector{T}, alpha::T=one(T), beta::T=zero(T))

High-performance SpMV: y = alpha * A * x + beta * y.
Traverses column slices with cache-friendly contiguous reads.
"""
function spmv!(y::Vector{T}, A::CSCMatrix{T}, x::Vector{T}, alpha::T=one(T), beta::T=zero(T)) where {T<:Real}
    @assert length(x) == A.n "Dimension mismatch: length(x) = $(length(x)), A.n = $(A.n)"
    @assert length(y) == A.m "Dimension mismatch: length(y) = $(length(y)), A.m = $(A.m)"

    if beta == zero(T)
        fill!(y, zero(T))
    elseif beta != one(T)
        y .*= beta
    end

    @inbounds for j in 1:A.n
        xj = x[j]
        abs(xj) < 1e-30 && continue
        scale = alpha * xj
        start_idx = A.colptr[j]
        end_idx = A.colptr[j + 1] - 1
        for p in start_idx:end_idx
            row = A.rowval[p]
            y[row] += scale * A.nzval[p]
        end
    end
    return y
end

function Base.:*(A::CSCMatrix{T}, x::Vector{T})::Vector{T} where {T<:Real}
    y = zeros(T, A.m)
    return spmv!(y, A, x)
end

"""
    transpose(A::CSCMatrix{T}) -> CSCMatrix{T}

Compute matrix transpose A^T in CSC format by column counting and redistribution.
"""
function Base.transpose(A::CSCMatrix{T})::CSCMatrix{T} where {T}
    m, n = A.m, A.n
    nnz_A = length(A.nzval)

    # Count entries per row of A (which becomes column of A^T)
    row_counts = zeros(Int, m)
    @inbounds for p in 1:nnz_A
        row_counts[A.rowval[p]] += 1
    end

    colptr_T = zeros(Int, m + 1)
    colptr_T[1] = 1
    for i in 1:m
        colptr_T[i + 1] = colptr_T[i] + row_counts[i]
    end

    curr_pos = copy(colptr_T)
    rowval_T = zeros(Int, nnz_A)
    nzval_T = zeros(T, nnz_A)

    @inbounds for j in 1:n
        start_idx = A.colptr[j]
        end_idx = A.colptr[j + 1] - 1
        for p in start_idx:end_idx
            r = A.rowval[p]
            v = A.nzval[p]
            dest = curr_pos[r]
            rowval_T[dest] = j
            nzval_T[dest] = v
            curr_pos[r] += 1
        end
    end

    return CSCMatrix{T}(n, m, colptr_T, rowval_T, nzval_T)
end

"""
    to_dense(A::CSCMatrix{T}) -> Matrix{T}

Materialize CSC matrix into standard dense Julia matrix.
"""
function to_dense(A::CSCMatrix{T})::Matrix{T} where {T}
    M = zeros(T, A.m, A.n)
    @inbounds for j in 1:A.n
        for p in A.colptr[j]:(A.colptr[j + 1] - 1)
            M[A.rowval[p], j] = A.nzval[p]
        end
    end
    return M
end

"""
    sparse_cg(A::CSCMatrix{Float64}, b::Vector{Float64}; tol::Float64=1e-10, max_iter::Int=200) -> (x, conv, residuals)

Specialized Conjugate Gradient solver directly operating on CSCMatrix without allocating dense matrices.
"""
function sparse_cg(A::CSCMatrix{Float64}, b::Vector{Float64}; tol::Float64=1e-10, max_iter::Int=200)
    n = length(b)
    x = zeros(Float64, n)
    r = copy(b)
    b_norm = norm(b)
    if b_norm == 0.0
        return (x, true, [0.0])
    end

    p = copy(r)
    rz_old = dot(r, r)
    residuals = [sqrt(rz_old) / b_norm]
    Ap = zeros(Float64, n)

    converged = false
    for iter in 1:max_iter
        spmv!(Ap, A, p)
        pAp = dot(p, Ap)
        if pAp <= 0.0
            break
        end

        alpha = rz_old / pAp
        x .+= alpha .* p
        r .-= alpha .* Ap

        rz_new = dot(r, r)
        res_norm = sqrt(rz_new) / b_norm
        push!(residuals, res_norm)
        if res_norm < tol
            converged = true
            break
        end

        beta = rz_new / rz_old
        p .= r .+ beta .* p
        rz_old = rz_new
    end

    return (x, converged, residuals)
end
