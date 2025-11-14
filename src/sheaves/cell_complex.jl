"""
    Spectra.Sheaves.CellComplex

Oriented cell complexes, simplicial complexes, boundary operators, and Hodge Laplacians.
Enforces the fundamental topological identity: partial_{k} * partial_{k+1} = 0.
"""

struct Simplex
    dim::Int
    vertices::Vector{Int}

    function Simplex(vertices::Vector{Int})
        sorted_v = sort(vertices)
        new(length(vertices) - 1, sorted_v)
    end
end

mutable struct CellComplex <: AbstractCellComplex
    num_vertices::Int
    edges::Vector{Tuple{Int, Int}}
    faces::Vector{Tuple{Int, Int, Int}}

    function CellComplex(n_vertices::Int)
        @assert n_vertices >= 0 "Vertex count must be non-negative"
        new(n_vertices, Tuple{Int, Int}[], Tuple{Int, Int, Int}[])
    end
end

num_vertices(X::CellComplex) = X.num_vertices
num_edges(X::CellComplex) = length(X.edges)
num_faces(X::CellComplex) = length(X.faces)

function add_edge!(X::CellComplex, u::Int, v::Int)
    @assert 1 <= u <= X.num_vertices && 1 <= v <= X.num_vertices "Vertex index out of bounds"
    @assert u != v "Self loops not allowed in simplicial 1-skeleton"
    edge = u < v ? (u, v) : (v, u)
    if !(edge in X.edges)
        push!(X.edges, edge)
    end
    return X
end

function add_face!(X::CellComplex, u::Int, v::Int, w::Int)
    @assert 1 <= u <= X.num_vertices && 1 <= v <= X.num_vertices && 1 <= w <= X.num_vertices
    @assert u != v && v != w && u != w "Degenerate face"
    # Ensure boundary edges exist
    add_edge!(X, u, v)
    add_edge!(X, v, w)
    add_edge!(X, w, u)

    # Sorted canonical face
    sorted_tri = Tuple(sort([u, v, w]))
    if !(sorted_tri in X.faces)
        push!(X.faces, sorted_tri)
    end
    return X
end

"""
    boundary_matrix_1(X::CellComplex) -> CSCMatrix{Float64}

Boundary operator partial_1 : C_1 -> C_0.
Dimension: |V| x |E|.
Entry (v, e) is +1 if v is the target of e, -1 if source, 0 otherwise.
"""
function boundary_matrix_1(X::CellComplex)::CSCMatrix{Float64}
    nV = X.num_vertices
    nE = length(X.edges)
    coo = COOMatrix(nV, nE)

    for (e_idx, (u, v)) in enumerate(X.edges)
        # Orientation: u -> v (u is tail -1, v is head +1)
        add_entry!(coo, v, e_idx, 1.0)
        add_entry!(coo, u, e_idx, -1.0)
    end

    return from_coo(coo)
end

"""
    boundary_matrix_2(X::CellComplex) -> CSCMatrix{Float64}

Boundary operator partial_2 : C_2 -> C_1.
Dimension: |E| x |F|.
Entry (e, f) is +1 or -1 based on relative orientation of edge e on face f.
"""
function boundary_matrix_2(X::CellComplex)::CSCMatrix{Float64}
    nE = length(X.edges)
    nF = length(X.faces)
    coo = COOMatrix(nE, nF)

    edge_map = Dict{Tuple{Int, Int}, Int}()
    for (idx, e) in enumerate(X.edges)
        edge_map[e] = idx
    end

    for (f_idx, (u, v, w)) in enumerate(X.faces)
        # Oriented boundary of [u, v, w] is [v, w] - [u, w] + [u, v]
        # 1. Edge (v, w) with sign +1
        e1 = v < w ? (v, w) : (w, v)
        sign1 = v < w ? 1.0 : -1.0
        add_entry!(coo, edge_map[e1], f_idx, sign1)

        # 2. Edge (u, w) with sign -1
        e2 = u < w ? (u, w) : (w, u)
        sign2 = u < w ? -1.0 : 1.0
        add_entry!(coo, edge_map[e2], f_idx, sign2)

        # 3. Edge (u, v) with sign +1
        e3 = u < v ? (u, v) : (v, u)
        sign3 = u < v ? 1.0 : -1.0
        add_entry!(coo, edge_map[e3], f_idx, sign3)
    end

    return from_coo(coo)
end

"""
    verify_boundary_property(X::CellComplex) -> Bool

Verify the topological chain complex property: partial_1 * partial_2 = 0.
"""
function verify_boundary_property(X::CellComplex)::Bool
    length(X.faces) == 0 && return true
    B1 = to_dense(boundary_matrix_1(X))
    B2 = to_dense(boundary_matrix_2(X))
    prod = B1 * B2
    return norm(prod) < 1e-12
end

"""
    graph_laplacian(X::CellComplex) -> CSCMatrix{Float64}

Standard 0-Laplacian L_0 = partial_1 * partial_1^T.
"""
function graph_laplacian(X::CellComplex)::CSCMatrix{Float64}
    B1 = boundary_matrix_1(X)
    B1_dense = to_dense(B1)
    L0_dense = B1_dense * B1_dense'
    return dense_to_csc(L0_dense)
end

"""
    hodge_laplacian_1(X::CellComplex) -> Matrix{Float64}

Hodge 1-Laplacian on 1-chains (edges):
L_1 = partial_1^T * partial_1 + partial_2 * partial_2^T.
Kernel corresponds to 1st homology group H_1(X; R) (independent 1D cycles/holes).
"""
function hodge_laplacian_1(X::CellComplex)::Matrix{Float64}
    B1 = to_dense(boundary_matrix_1(X))
    L1_down = B1' * B1

    if length(X.faces) > 0
        B2 = to_dense(boundary_matrix_2(X))
        L1_up = B2 * B2'
        return L1_down .+ L1_up
    else
        return L1_down
    end
end

"""
    betti_numbers(X::CellComplex) -> (beta_0::Int, beta_1::Int)

Compute the 0-th and 1-st Betti numbers of the cell complex.
beta_0 is the number of connected components.
beta_1 is the number of 1-dimensional topological holes (independent cycles).
"""
function betti_numbers(X::CellComplex)::Tuple{Int, Int}
    B1 = to_dense(boundary_matrix_1(X))
    r1 = rank(B1)
    beta_0 = X.num_vertices - r1

    if length(X.faces) == 0
        beta_1 = length(X.edges) - r1
    else
        B2 = to_dense(boundary_matrix_2(X))
        r2 = rank(B2)
        # beta_1 = dim(ker partial_1) - dim(im partial_2)
        dim_ker_B1 = length(X.edges) - r1
        beta_1 = dim_ker_B1 - r2
    end

    return (beta_0, max(0, beta_1))
end
