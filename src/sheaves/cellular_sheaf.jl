"""
    Spectra.Sheaves.CellularSheaf

Cellular sheaf data structure over cell complexes.
Assigns stalk vector spaces F(sigma) to cells and linear restriction maps F_{v <= e} to incident relations.
Generalizes graph signals to topological sheaf cochains with local coordinate frames and gauge connections.
"""

mutable struct CellularSheaf <: AbstractCellularSheaf
    complex::CellComplex
    vertex_stalk_dims::Vector{Int}
    edge_stalk_dims::Vector{Int}
    restrictions::Dict{Tuple{Int, Int}, Matrix{Float64}} # (vertex_idx, edge_idx) => Matrix (dim_e x dim_v)
    v_offsets::Vector{Int} # Precomputed 0-cochain vector offsets
    e_offsets::Vector{Int} # Precomputed 1-cochain vector offsets

    function CellularSheaf(complex::CellComplex, vertex_dims::Vector{Int}, edge_dims::Vector{Int})
        nV = complex.num_vertices
        nE = length(complex.edges)
        @assert length(vertex_dims) == nV "Vertex stalk dimensions must match vertex count"
        @assert length(edge_dims) == nE "Edge stalk dimensions must match edge count"

        v_offsets = zeros(Int, nV + 1)
        v_offsets[1] = 1
        for i in 1:nV
            v_offsets[i + 1] = v_offsets[i] + vertex_dims[i]
        end

        e_offsets = zeros(Int, nE + 1)
        e_offsets[1] = 1
        for i in 1:nE
            e_offsets[i + 1] = e_offsets[i] + edge_dims[i]
        end

        new(complex, vertex_dims, edge_dims, Dict{Tuple{Int, Int}, Matrix{Float64}}(), v_offsets, e_offsets)
    end
end

"""
    CellularSheaf(complex::CellComplex, stalk_dim::Int=1)

Construct cellular sheaf with uniform stalk dimension across all vertices and edges.
"""
function CellularSheaf(complex::CellComplex, stalk_dim::Int=1)
    nV = complex.num_vertices
    nE = length(complex.edges)
    v_dims = fill(stalk_dim, nV)
    e_dims = fill(stalk_dim, nE)
    return CellularSheaf(complex, v_dims, e_dims)
end

total_cochain_dim_0(sheaf::CellularSheaf) = sheaf.v_offsets[end] - 1
total_cochain_dim_1(sheaf::CellularSheaf) = sheaf.e_offsets[end] - 1

vertex_range(sheaf::CellularSheaf, v::Int) = sheaf.v_offsets[v]:(sheaf.v_offsets[v + 1] - 1)
edge_range(sheaf::CellularSheaf, e::Int) = sheaf.e_offsets[e]:(sheaf.e_offsets[e + 1] - 1)

"""
    set_restriction!(sheaf::CellularSheaf, v::Int, e_idx::Int, R::Matrix{Float64})

Assign linear restriction map F_{v <= e} : F(v) -> F(e).
"""
function set_restriction!(sheaf::CellularSheaf, v::Int, e_idx::Int, R::Matrix{Float64})
    u1, u2 = sheaf.complex.edges[e_idx]
    @assert v == u1 || v == u2 "Vertex $v is not incident to edge $e_idx = ($u1, $u2)"
    dv = sheaf.vertex_stalk_dims[v]
    de = sheaf.edge_stalk_dims[e_idx]
    @assert size(R) == (de, dv) "Restriction map dimension $(size(R)) must match ($de, $dv)"
    sheaf.restrictions[(v, e_idx)] = copy(R)
    return sheaf
end

"""
    get_restriction(sheaf::CellularSheaf, v::Int, e_idx::Int) -> Matrix{Float64}

Retrieve restriction map F_{v <= e}. Defaults to identity or zero if unassigned.
"""
function get_restriction(sheaf::CellularSheaf, v::Int, e_idx::Int)::Matrix{Float64}
    if haskey(sheaf.restrictions, (v, e_idx))
        return sheaf.restrictions[(v, e_idx)]
    end
    dv = sheaf.vertex_stalk_dims[v]
    de = sheaf.edge_stalk_dims[e_idx]
    if dv == de
        return Matrix{Float64}(I, de, dv)
    else
        return zeros(Float64, de, dv)
    end
end

"""
    trivial_sheaf(X::CellComplex) -> CellularSheaf

Create 1D trivial sheaf where all stalk dimensions are 1 and all restriction maps are identity [1.0].
Recovers classical graph calculus.
"""
function trivial_sheaf(X::CellComplex)::CellularSheaf
    sheaf = CellularSheaf(X, 1)
    for (e_idx, (u, v)) in enumerate(X.edges)
        set_restriction!(sheaf, u, e_idx, ones(1, 1))
        set_restriction!(sheaf, v, e_idx, ones(1, 1))
    end
    return sheaf
end

"""
    connection_sheaf(X::CellComplex, stalk_dim::Int, edge_rotations::Vector{Matrix{Float64}}) -> CellularSheaf

Create orthogonal gauge connection sheaf where restriction from source is I
and restriction from target is orthogonal rotation matrix R_e in SO(d).
"""
function connection_sheaf(X::CellComplex, stalk_dim::Int, edge_rotations::Vector{Matrix{Float64}})::CellularSheaf
    @assert length(edge_rotations) == length(X.edges) "Number of rotations must match edge count"
    sheaf = CellularSheaf(X, stalk_dim)
    I_d = Matrix{Float64}(I, stalk_dim, stalk_dim)

    for (e_idx, (u, v)) in enumerate(X.edges)
        R = edge_rotations[e_idx]
        @assert size(R) == (stalk_dim, stalk_dim) "Rotation matrix must be ($stalk_dim, $stalk_dim)"
        set_restriction!(sheaf, u, e_idx, I_d)
        set_restriction!(sheaf, v, e_idx, R)
    end

    return sheaf
end
