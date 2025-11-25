"""
    Spectra.Sheaves.NeuralSheaf

Neural Sheaf Diffusion (NSD) layers for geometric deep learning on graphs and cell complexes.
Learns dynamic cellular sheaf restriction maps to solve the oversmoothing problem
and adaptively filter heterophilic topological signals.
"""

struct NeuralSheafLayer
    in_features::Int
    out_features::Int
    stalk_dim::Int
    weight::Matrix{Float64}
    restriction_weights::Matrix{Float64} # (stalk_dim * stalk_dim) x (2 * in_features)
    epsilon::Float64

    function NeuralSheafLayer(in_f::Int, out_f::Int, stalk_d::Int=2; eps::Float64=0.1)
        scale_w = sqrt(2.0 / (in_f + out_f))
        W = randn(in_f, out_f) .* scale_w

        # Restriction predictor mapping concatenated node features to stalk_dim x stalk_dim matrix
        scale_r = 0.01
        W_r = randn(stalk_d * stalk_d, 2 * in_f) .* scale_r

        new(in_f, out_f, stalk_d, W, W_r, eps)
    end
end

"""
    predict_restriction(layer::NeuralSheafLayer, xu::Vector{Float64}, xv::Vector{Float64}) -> Matrix{Float64}

Predict orthogonal restriction map in O(stalk_dim) from concatenated node features via QR orthogonalization.
"""
function predict_restriction(layer::NeuralSheafLayer, xu::Vector{Float64}, xv::Vector{Float64})::Matrix{Float64}
    d = layer.stalk_dim
    feat = [xu; xv]
    raw_vec = layer.restriction_weights * feat
    raw_mat = reshape(raw_vec, (d, d))
    # Project to orthogonal group O(d) via QR decomposition
    Q = Matrix(qr(raw_mat + I).Q)
    return Q
end

"""
    build_learned_sheaf(layer::NeuralSheafLayer, X::CellComplex, node_features::Matrix{Float64}) -> CellularSheaf

Construct cellular sheaf with dynamically learned restriction maps based on node feature interactions.
"""
function build_learned_sheaf(layer::NeuralSheafLayer, X::CellComplex, node_features::Matrix{Float64})::CellularSheaf
    d = layer.stalk_dim
    sheaf = CellularSheaf(X, d)
    I_d = Matrix{Float64}(I, d, d)

    for (e_idx, (u, v)) in enumerate(X.edges)
        xu = node_features[u, :]
        xv = node_features[v, :]
        R = predict_restriction(layer, xu, xv)
        set_restriction!(sheaf, u, e_idx, I_d)
        set_restriction!(sheaf, v, e_idx, R)
    end

    return sheaf
end

"""
    forward_sheaf_layer(layer::NeuralSheafLayer, X::CellComplex, node_features::Matrix{Float64}) -> Matrix{Float64}

Execute forward pass of Neural Sheaf Diffusion layer:
1. Predict dynamic cellular sheaf F from current features.
2. Form Sheaf Laplacian L_F.
3. Apply diffusion step (I - eps * L_F) and linear projection W.
4. Apply non-linear activation (ReLU).
"""
function forward_sheaf_layer(layer::NeuralSheafLayer, X::CellComplex, node_features::Matrix{Float64})::Matrix{Float64}
    nV, nF = size(node_features)
    @assert nV == X.num_vertices "Node features rows must match vertex count"
    @assert nF == layer.in_features "Feature dimension must match layer in_features"

    sheaf = build_learned_sheaf(layer, X, node_features)
    L = sheaf_laplacian(sheaf)

    # Flatten node features into stalk representation
    d = layer.stalk_dim
    # Repeat or project features into stalk dimension
    # When in_features >= stalk_dim, chunk into channels
    # Generalized Kronecker diffusion: (I - eps L_F) applied channel-wise
    diffused_features = copy(node_features)

    # Simple standard Sheaf Diffusion: apply (I - eps L_F) on each feature channel
    L_dense = L.dense_matrix
    # If stalk_dim == 1, L_dense is nV x nV
    if d == 1
        I_nV = Matrix{Float64}(I, nV, nV)
        diff_op = I_nV .- layer.epsilon .* L_dense
        diffused_features = diff_op * node_features
    else
        # For d > 1, expand node features into cochain C^0(X; F)
        # Process in blocks of d channels
        n_blocks = div(nF, d)
        if n_blocks > 0
            for b in 1:n_blocks
                cols = ((b - 1) * d + 1):(b * d)
                # Reshape nV x d into nV*d cochain vector
                cochain = vec(node_features[:, cols]')
                diffused_cochain = cochain .- layer.epsilon .* (L.matrix * cochain)
                diffused_features[:, cols] = reshape(diffused_cochain, (d, nV))'
            end
        end
    end

    # Linear transformation & ReLU activation
    out = diffused_features * layer.weight
    return max.(0.0, out)
end
