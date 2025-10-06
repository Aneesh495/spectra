"""
    Spectra.Math.ReverseAD

Dynamic tape-based reverse-mode automatic differentiation engine with custom
vector-Jacobian products (VJPs) and reverse-mode topological backpropagation.
"""

# Computational Graph Node
mutable struct TapeNode
    id::Int
    val::Float64
    parents::Vector{Int}
    vjps::Vector{Function} # Each function takes adjoint dz (Float64) and returns parent adjoint contribution
end

mutable struct GradientTape
    nodes::Vector{TapeNode}
    is_active::Bool

    function GradientTape()
        new(Vector{TapeNode}(), true)
    end
end

const ACTIVE_TAPE = Ref{Union{Nothing, GradientTape}}(nothing)

function reset_tape!()
    ACTIVE_TAPE[] = GradientTape()
    return ACTIVE_TAPE[]
end

function get_tape()::GradientTape
    if ACTIVE_TAPE[] === nothing
        ACTIVE_TAPE[] = GradientTape()
    end
    return ACTIVE_TAPE[]
end

# Tracked Real Scalar
struct TrackedReal <: Real
    val::Float64
    node_id::Int
end

value(t::TrackedReal) = t.val
node_id(t::TrackedReal) = t.node_id

Base.show(io::IO, t::TrackedReal) = print(io, "TrackedReal($(t.val), id=$(t.node_id))")

function record_node!(tape::GradientTape, val::Float64, parents::Vector{Int}, vjps::Vector{Function})::Int
    id = length(tape.nodes) + 1
    node = TapeNode(id, val, parents, vjps)
    push!(tape.nodes, node)
    return id
end

function track(x::Real)::TrackedReal
    tape = get_tape()
    val = Float64(x)
    id = record_node!(tape, val, Int[], Function[])
    return TrackedReal(val, id)
end

function track(arr::AbstractArray{<:Real})::Array{TrackedReal}
    return [track(x) for x in arr]
end

# Overloaded Arithmetic Operations on TrackedReal
function Base.:+(a::TrackedReal, b::TrackedReal)
    tape = get_tape()
    val = a.val + b.val
    vjp_a = dz -> dz
    vjp_b = dz -> dz
    id = record_node!(tape, val, [a.node_id, b.node_id], Function[vjp_a, vjp_b])
    return TrackedReal(val, id)
end

function Base.:+(a::TrackedReal, b::Real)
    tape = get_tape()
    val = a.val + Float64(b)
    vjp_a = dz -> dz
    id = record_node!(tape, val, [a.node_id], Function[vjp_a])
    return TrackedReal(val, id)
end
Base.:+(a::Real, b::TrackedReal) = b + a

function Base.:-(a::TrackedReal)
    tape = get_tape()
    val = -a.val
    vjp_a = dz -> -dz
    id = record_node!(tape, val, [a.node_id], Function[vjp_a])
    return TrackedReal(val, id)
end

function Base.:-(a::TrackedReal, b::TrackedReal)
    tape = get_tape()
    val = a.val - b.val
    vjp_a = dz -> dz
    vjp_b = dz -> -dz
    id = record_node!(tape, val, [a.node_id, b.node_id], Function[vjp_a, vjp_b])
    return TrackedReal(val, id)
end

function Base.:-(a::TrackedReal, b::Real)
    tape = get_tape()
    val = a.val - Float64(b)
    vjp_a = dz -> dz
    id = record_node!(tape, val, [a.node_id], Function[vjp_a])
    return TrackedReal(val, id)
end

function Base.:-(a::Real, b::TrackedReal)
    tape = get_tape()
    val = Float64(a) - b.val
    vjp_b = dz -> -dz
    id = record_node!(tape, val, [b.node_id], Function[vjp_b])
    return TrackedReal(val, id)
end

function Base.:*(a::TrackedReal, b::TrackedReal)
    tape = get_tape()
    val = a.val * b.val
    vjp_a = dz -> dz * b.val
    vjp_b = dz -> dz * a.val
    id = record_node!(tape, val, [a.node_id, b.node_id], Function[vjp_a, vjp_b])
    return TrackedReal(val, id)
end

function Base.:*(a::TrackedReal, b::Real)
    tape = get_tape()
    cb = Float64(b)
    val = a.val * cb
    vjp_a = dz -> dz * cb
    id = record_node!(tape, val, [a.node_id], Function[vjp_a])
    return TrackedReal(val, id)
end
Base.:*(a::Real, b::TrackedReal) = b * a

function Base.:/(a::TrackedReal, b::TrackedReal)
    tape = get_tape()
    val = a.val / b.val
    inv_b = 1.0 / b.val
    vjp_a = dz -> dz * inv_b
    vjp_b = dz -> -dz * (a.val * inv_b * inv_b)
    id = record_node!(tape, val, [a.node_id, b.node_id], Function[vjp_a, vjp_b])
    return TrackedReal(val, id)
end

function Base.:/(a::TrackedReal, b::Real)
    return a * (1.0 / Float64(b))
end

function Base.:/(a::Real, b::TrackedReal)
    tape = get_tape()
    ca = Float64(a)
    val = ca / b.val
    inv_b = 1.0 / b.val
    vjp_b = dz -> -dz * (ca * inv_b * inv_b)
    id = record_node!(tape, val, [b.node_id], Function[vjp_b])
    return TrackedReal(val, id)
end

# Elementary nonlinear functions
function Base.exp(a::TrackedReal)
    tape = get_tape()
    val = exp(a.val)
    vjp_a = dz -> dz * val
    id = record_node!(tape, val, [a.node_id], Function[vjp_a])
    return TrackedReal(val, id)
end

function Base.log(a::TrackedReal)
    tape = get_tape()
    val = log(a.val)
    inv_a = 1.0 / a.val
    vjp_a = dz -> dz * inv_a
    id = record_node!(tape, val, [a.node_id], Function[vjp_a])
    return TrackedReal(val, id)
end

function Base.sin(a::TrackedReal)
    tape = get_tape()
    s, c = sincos(a.val)
    vjp_a = dz -> dz * c
    id = record_node!(tape, s, [a.node_id], Function[vjp_a])
    return TrackedReal(s, id)
end

function Base.cos(a::TrackedReal)
    tape = get_tape()
    s, c = sincos(a.val)
    vjp_a = dz -> -dz * s
    id = record_node!(tape, c, [a.node_id], Function[vjp_a])
    return TrackedReal(c, id)
end

function Base.tanh(a::TrackedReal)
    tape = get_tape()
    th = tanh(a.val)
    sech2 = 1.0 - th * th
    vjp_a = dz -> dz * sech2
    id = record_node!(tape, th, [a.node_id], Function[vjp_a])
    return TrackedReal(th, id)
end

function Base.sqrt(a::TrackedReal)
    tape = get_tape()
    val = sqrt(a.val)
    inv_2s = 0.5 / val
    vjp_a = dz -> dz * inv_2s
    id = record_node!(tape, val, [a.node_id], Function[vjp_a])
    return TrackedReal(val, id)
end

function Base.:^(a::TrackedReal, p::Integer)
    tape = get_tape()
    val = a.val^p
    deriv = Float64(p) * (a.val^(p - 1))
    vjp_a = dz -> dz * deriv
    id = record_node!(tape, val, [a.node_id], Function[vjp_a])
    return TrackedReal(val, id)
end

function Base.:^(a::TrackedReal, p::Real)
    tape = get_tape()
    cp = Float64(p)
    val = a.val^cp
    deriv = cp * (a.val^(cp - 1))
    vjp_a = dz -> dz * deriv
    id = record_node!(tape, val, [a.node_id], Function[vjp_a])
    return TrackedReal(val, id)
end

# Relu and Sigmoid Activations
function relu(a::TrackedReal)
    tape = get_tape()
    val = max(0.0, a.val)
    vjp_a = dz -> a.val > 0.0 ? dz : 0.0
    id = record_node!(tape, val, [a.node_id], Function[vjp_a])
    return TrackedReal(val, id)
end
relu(x::Real) = max(zero(x), x)

function sigmoid(a::TrackedReal)
    tape = get_tape()
    s = 1.0 / (1.0 + exp(-a.val))
    vjp_a = dz -> dz * s * (1.0 - s)
    id = record_node!(tape, s, [a.node_id], Function[vjp_a])
    return TrackedReal(s, id)
end
sigmoid(x::Real) = 1.0 / (1.0 + exp(-Float64(x)))

Base.isless(a::TrackedReal, b::TrackedReal) = a.val < b.val
Base.isless(a::TrackedReal, b::Real) = a.val < Float64(b)
Base.isless(a::Real, b::TrackedReal) = Float64(a) < b.val

# Backward Sweep (Topological Order Propagation)
function backward!(tape::GradientTape, output_id::Int)::Vector{Float64}
    n_nodes = length(tape.nodes)
    @assert 1 <= output_id <= n_nodes "Invalid output node ID"
    adjoints = zeros(Float64, n_nodes)
    adjoints[output_id] = 1.0

    # Reverse order traversal
    for i in n_nodes:-1:1
        dz = adjoints[i]
        abs(dz) < 1e-30 && continue
        node = tape.nodes[i]
        n_parents = length(node.parents)
        for k in 1:n_parents
            parent_id = node.parents[k]
            vjp_fn = node.vjps[k]
            grad_contrib = vjp_fn(dz)
            adjoints[parent_id] += grad_contrib
        end
    end
    return adjoints
end

"""
    value_and_gradient(f, x::AbstractVector{Float64}) -> (loss::Float64, grad::Vector{Float64})

Compute function value and exact gradient via dynamic tape reverse-mode AD in a single pass.
"""
function value_and_gradient(f, x::AbstractVector{Float64})
    tape = reset_tape!()
    tracked_x = track(x)
    input_ids = [t.node_id for t in tracked_x]

    out = f(tracked_x)
    @assert out isa TrackedReal "Target function must return a scalar TrackedReal"

    loss = out.val
    adjoints = backward!(tape, out.node_id)
    grad = [adjoints[id] for id in input_ids]

    return (loss, grad)
end

"""
    reverse_gradient(f, x::AbstractVector{Float64}) -> Vector{Float64}

Compute exact gradient via reverse-mode AD.
"""
function reverse_gradient(f, x::AbstractVector{Float64})::Vector{Float64}
    _, grad = value_and_gradient(f, x)
    return grad
end
