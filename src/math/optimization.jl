"""
    Spectra.Math.Optimization

First-order and quasi-Newton optimization algorithms for continuous parameter spaces.
Includes SGD with momentum, Adam, and Limited-Memory BFGS (L-BFGS) with Armijo-Wolfe line search.
"""

abstract type AbstractOptimizer end

# ====================================================================
# Stochastic Gradient Descent with Momentum & Weight Decay
# ====================================================================

mutable struct SGDMomentum <: AbstractOptimizer
    lr::Float64
    momentum::Float64
    weight_decay::Float64
    nesterov::Bool
    velocity::Vector{Float64}

    function SGDMomentum(; lr::Float64=0.01, momentum::Float64=0.9, weight_decay::Float64=0.0, nesterov::Bool=false)
        new(lr, momentum, weight_decay, nesterov, Float64[])
    end
end

function update!(opt::SGDMomentum, x::Vector{Float64}, grad::Vector{Float64})
    n = length(x)
    if length(opt.velocity) != n
        opt.velocity = zeros(Float64, n)
    end

    g = opt.weight_decay > 0.0 ? grad .+ opt.weight_decay .* x : grad
    opt.velocity .= opt.momentum .* opt.velocity .+ g

    if opt.nesterov
        step = g .+ opt.momentum .* opt.velocity
        x .-= opt.lr .* step
    else
        x .-= opt.lr .* opt.velocity
    end
    return x
end

# ====================================================================
# Adam Optimizer with Adaptive First and Second Moments
# ====================================================================

mutable struct AdamOptimizer <: AbstractOptimizer
    lr::Float64
    beta1::Float64
    beta2::Float64
    eps::Float64
    weight_decay::Float64
    m::Vector{Float64}
    v::Vector{Float64}
    step_count::Int

    function AdamOptimizer(; lr::Float64=0.001, beta1::Float64=0.9, beta2::Float64=0.999, eps::Float64=1e-8, weight_decay::Float64=0.0)
        new(lr, beta1, beta2, eps, weight_decay, Float64[], Float64[], 0)
    end
end

function update!(opt::AdamOptimizer, x::Vector{Float64}, grad::Vector{Float64})
    n = length(x)
    if length(opt.m) != n
        opt.m = zeros(Float64, n)
        opt.v = zeros(Float64, n)
        opt.step_count = 0
    end

    opt.step_count += 1
    t = opt.step_count

    g = opt.weight_decay > 0.0 ? grad .+ opt.weight_decay .* x : grad

    # Moment updates
    opt.m .= opt.beta1 .* opt.m .+ (1.0 - opt.beta1) .* g
    opt.v .= opt.beta2 .* opt.v .+ (1.0 - opt.beta2) .* (g .^ 2)

    # Bias correction
    bc1 = 1.0 - opt.beta1^t
    bc2 = 1.0 - opt.beta2^t

    inv_bc1 = 1.0 / bc1
    inv_bc2 = 1.0 / bc2

    @inbounds for i in 1:n
        m_hat = opt.m[i] * inv_bc1
        v_hat = opt.v[i] * inv_bc2
        x[i] -= opt.lr * m_hat / (sqrt(v_hat) + opt.eps)
    end
    return x
end

# ====================================================================
# Limited-Memory BFGS (L-BFGS) with Backtracking Armijo Line Search
# ====================================================================

mutable struct LBFGSOptimizer <: AbstractOptimizer
    m_history::Int
    max_iter::Int
    tol::Float64
    c1::Float64 # Armijo sufficient decrease condition
    c2::Float64 # Curvature condition

    function LBFGSOptimizer(; m_history::Int=10, max_iter::Int=100, tol::Float64=1e-6, c1::Float64=1e-4, c2::Float64=0.9)
        new(m_history, max_iter, tol, c1, c2)
    end
end

"""
    optimize_lbfgs(loss_and_grad_fn, x0::Vector{Float64}; opt::LBFGSOptimizer=LBFGSOptimizer()) -> (x_opt, loss_hist)

Quasi-Newton L-BFGS optimization loop with two-loop recursion and Armijo backtracking line search.
"""
function optimize_lbfgs(loss_and_grad_fn, x0::Vector{Float64}; opt::LBFGSOptimizer=LBFGSOptimizer())
    n = length(x0)
    x = copy(x0)
    loss, g = loss_and_grad_fn(x)

    loss_history = Float64[loss]
    if norm(g) < opt.tol
        return (x, loss_history)
    end

    s_history = Vector{Vector{Float64}}()
    y_history = Vector{Vector{Float64}}()
    rho_history = Float64[]

    for iter in 1:opt.max_iter
        # L-BFGS Two-Loop Recursion to compute descent direction p = -H_k * g
        q = copy(g)
        k_hist = length(s_history)
        alphas = zeros(Float64, k_hist)

        # First loop (backward)
        for i in k_hist:-1:1
            alphas[i] = rho_history[i] * dot(s_history[i], q)
            q .-= alphas[i] .* y_history[i]
        end

        # Initial Hessian scaling: gamma_k = (s_{k-1}^T y_{k-1}) / (y_{k-1}^T y_{k-1})
        gamma = k_hist > 0 ? dot(s_history[end], y_history[end]) / dot(y_history[end], y_history[end]) : 1.0
        r = gamma .* q

        # Second loop (forward)
        for i in 1:k_hist
            beta = rho_history[i] * dot(y_history[i], r)
            r .+= s_history[i] .* (alphas[i] - beta)
        end

        p = -r # Search direction

        # Backtracking Armijo Line Search
        alpha_step = k_hist == 0 ? min(1.0, 1.0 / max(norm(g), 1e-8)) : 1.0
        slope = dot(g, p)
        if slope >= 0.0
            # Reset direction to steepest descent if not a descent direction
            p = -g
            slope = dot(g, p)
            alpha_step = min(1.0, 1.0 / max(norm(g), 1e-8))
        end

        x_next = copy(x)
        loss_next = loss
        g_next = copy(g)

        for _ in 1:30
            x_next = x .+ alpha_step .* p
            loss_next, g_next = loss_and_grad_fn(x_next)
            if loss_next <= loss + opt.c1 * alpha_step * slope
                break
            end
            alpha_step *= 0.5
        end

        s = x_next - x
        y = g_next - g
        sy = dot(s, y)

        if sy > 1e-12
            if length(s_history) >= opt.m_history
                popfirst!(s_history)
                popfirst!(y_history)
                popfirst!(rho_history)
            end
            push!(s_history, s)
            push!(y_history, y)
            push!(rho_history, 1.0 / sy)
        end

        x = x_next
        g = g_next
        loss = loss_next
        push!(loss_history, loss)

        if norm(g) < opt.tol
            break
        end
    end

    return (x, loss_history)
end

"""
    optimize(loss_and_grad_fn, x0::Vector{Float64}, opt::AbstractOptimizer; max_iters::Int=200, tol::Float64=1e-6)

Unified optimization runner for SGD and Adam optimizers.
"""
function optimize(loss_and_grad_fn, x0::Vector{Float64}, opt::AbstractOptimizer; max_iters::Int=200, tol::Float64=1e-6)
    if opt isa LBFGSOptimizer
        return optimize_lbfgs(loss_and_grad_fn, x0; opt=opt)
    end

    x = copy(x0)
    loss_history = Float64[]

    for iter in 1:max_iters
        loss, g = loss_and_grad_fn(x)
        push!(loss_history, loss)

        if norm(g) < tol
            break
        end
        update!(opt, x, g)
    end

    return (x, loss_history)
end
