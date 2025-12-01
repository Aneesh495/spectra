"""
    Spectra.Sheaves.PersistentHomology

Topological Data Analysis (TDA), Vietoris-Rips filtration, boundary matrix reduction,
and persistent homology barcodes for multiscale topological feature extraction.
"""

struct PersistenceInterval
    dim::Int
    birth::Float64
    death::Float64
end

Base.show(io::IO, p::PersistenceInterval) = print(io, "H_$(p.dim) [$(round(p.birth, digits=4)), $(p.death == Inf ? "Inf" : round(p.death, digits=4)))")

struct FilteredSimplex
    dim::Int
    vertices::Vector{Int}
    birth::Float64
end

"""
    vietoris_rips_filtration(points::Matrix{Float64}; max_radius::Float64=Inf) -> Vector{FilteredSimplex}

Construct the filtered Vietoris-Rips simplicial complex from a point cloud.
Points is N x D where N is the number of points and D is the spatial dimension.
"""
function vietoris_rips_filtration(points::Matrix{Float64}; max_radius::Float64=Inf)::Vector{FilteredSimplex}
    N, D = size(points)
    simplices = FilteredSimplex[]

    # 0-simplices (vertices, birth = 0.0)
    for i in 1:N
        push!(simplices, FilteredSimplex(0, [i], 0.0))
    end

    # Pairwise distance matrix
    dist_mat = zeros(Float64, N, N)
    for i in 1:N
        for j in (i + 1):N
            d = norm(points[i, :] - points[j, :])
            dist_mat[i, j] = d
            dist_mat[j, i] = d
            if d <= max_radius
                push!(simplices, FilteredSimplex(1, [i, j], d))
            end
        end
    end

    # 2-simplices (triangles)
    for i in 1:N
        for j in (i + 1):N
            for k in (j + 1):N
                d_ij = dist_mat[i, j]
                d_jk = dist_mat[j, k]
                d_ik = dist_mat[i, k]
                birth = max(d_ij, d_jk, d_ik)
                if birth <= max_radius
                    push!(simplices, FilteredSimplex(2, [i, j, k], birth))
                end
            end
        end
    end

    # Sort simplices: primary by birth scale, secondary by dimension
    sort!(simplices, by=s -> (s.birth, s.dim))
    return simplices
end

"""
    compute_persistence_diagram(points::Matrix{Float64}; max_radius::Float64=Inf) -> Vector{PersistenceInterval}

Compute persistent homology intervals H_0 and H_1 using standard column reduction over Z_2.
"""
function compute_persistence_diagram(points::Matrix{Float64}; max_radius::Float64=Inf)::Vector{PersistenceInterval}
    simplices = vietoris_rips_filtration(points; max_radius=max_radius)
    m = length(simplices)

    # Simplex to index mapping
    simplex_map = Dict{Vector{Int}, Int}()
    for (idx, s) in enumerate(simplices)
        simplex_map[s.vertices] = idx
    end

    # Build boundary matrix columns as sorted index sets in Z_2
    boundary_cols = [Set{Int}() for _ in 1:m]
    for (j, s) in enumerate(simplices)
        if s.dim == 1
            # Boundary of [u, v] is [u] and [v]
            u, v = s.vertices
            push!(boundary_cols[j], simplex_map[[u]])
            push!(boundary_cols[j], simplex_map[[v]])
        elseif s.dim == 2
            # Boundary of [u, v, w] is [v, w], [u, w], [u, v]
            u, v, w = s.vertices
            e1 = [v, w]; e2 = [u, w]; e3 = [u, v]
            if haskey(simplex_map, e1) push!(boundary_cols[j], simplex_map[e1]) end
            if haskey(simplex_map, e2) push!(boundary_cols[j], simplex_map[e2]) end
            if haskey(simplex_map, e3) push!(boundary_cols[j], simplex_map[e3]) end
        end
    end

    # Standard Z_2 column reduction algorithm
    low = zeros(Int, m)
    low_to_col = Dict{Int, Int}()

    for j in 1:m
        while !isempty(boundary_cols[j])
            max_row = maximum(boundary_cols[j])
            if haskey(low_to_col, max_row)
                k = low_to_col[max_row]
                # Symmetric difference (Z_2 addition)
                boundary_cols[j] = symdiff(boundary_cols[j], boundary_cols[k])
            else
                low[j] = max_row
                low_to_col[max_row] = j
                break
            end
        end
    end

    # Extract birth-death persistence intervals
    intervals = PersistenceInterval[]
    paired_rows = Set{Int}()

    for j in 1:m
        if low[j] > 0
            i = low[j]
            push!(paired_rows, i)
            birth = simplices[i].birth
            death = simplices[j].birth
            if death > birth + 1e-12
                push!(intervals, PersistenceInterval(simplices[i].dim, birth, death))
            end
        end
    end

    # Essential classes (infinite persistence)
    for i in 1:m
        if !(i in paired_rows) && isempty(boundary_cols[i])
            if simplices[i].dim <= 1
                push!(intervals, PersistenceInterval(simplices[i].dim, simplices[i].birth, Inf))
            end
        end
    end

    return intervals
end
