"""
    Spectra.Dynamics.SDEIntegrator

Stochastic Differential Equation (SDE) integrators for Ito and Stratonovich diffusions:
du = f(u, t) dt + g(u, t) dW.
Implements Euler-Maruyama (strong order 0.5) and Milstein (strong order 1.0) schemes
with Brownian path generators and Ito correction terms via automatic differentiation.
"""

struct SDESolution
    t::Vector{Float64}
    u::Vector{Vector{Float64}}
    W::Vector{Vector{Float64}}
end

Base.length(sol::SDESolution) = length(sol.t)

"""
    integrate_sde_euler_maruyama(f, g, u0::Vector{Float64}, tspan::Tuple{Float64, Float64}, dt::Float64;
                                 seed::Union{Nothing, Int}=nothing) -> SDESolution

Integrate Ito SDE du = f(u, t) dt + g(u, t) dW via Euler-Maruyama scheme.
"""
function integrate_sde_euler_maruyama(f, g, u0::Vector{Float64}, tspan::Tuple{Float64, Float64}, dt::Float64;
                                      seed::Union{Nothing, Int}=nothing)::SDESolution
    if seed !== nothing
        Random.seed!(seed)
    end

    t0, tf = tspan
    @assert tf > t0 "tf must exceed t0"
    dim = length(u0)

    t = t0
    u = copy(u0)
    w_curr = zeros(Float64, dim)

    t_history = Float64[t0]
    u_history = Vector{Float64}[copy(u0)]
    w_history = Vector{Float64}[copy(w_curr)]

    sqrt_dt = sqrt(dt)

    while t < tf - 1e-12
        step_dt = min(dt, tf - t)
        s_dt = sqrt(step_dt)

        # Brownian increment dW ~ N(0, step_dt)
        dW = randn(dim) .* s_dt
        w_curr .+= dW

        drift = f(u, t)
        diffusion = g(u, t)

        # Vector or diagonal diffusion update
        diff_term = diffusion isa AbstractMatrix ? diffusion * dW : diffusion .* dW
        u .+= step_dt .* drift .+ diff_term
        t += step_dt

        push!(t_history, t)
        push!(u_history, copy(u))
        push!(w_history, copy(w_curr))
    end

    return SDESolution(t_history, u_history, w_history)
end

"""
    integrate_sde_milstein(f, g, u0::Vector{Float64}, tspan::Tuple{Float64, Float64}, dt::Float64;
                           seed::Union{Nothing, Int}=nothing) -> SDESolution

Integrate 1D/diagonal Ito SDE via Milstein scheme with strong convergence order 1.0:
u_{n+1} = u_n + f dt + g dW + 0.5 g' g (dW^2 - dt).
Derivative g' is evaluated via exact dual number forward automatic differentiation.
"""
function integrate_sde_milstein(f, g, u0::Vector{Float64}, tspan::Tuple{Float64, Float64}, dt::Float64;
                                seed::Union{Nothing, Int}=nothing)::SDESolution
    if seed !== nothing
        Random.seed!(seed)
    end

    t0, tf = tspan
    @assert tf > t0 "tf must exceed t0"
    dim = length(u0)

    t = t0
    u = copy(u0)
    w_curr = zeros(Float64, dim)

    t_history = Float64[t0]
    u_history = Vector{Float64}[copy(u0)]
    w_history = Vector{Float64}[copy(w_curr)]

    while t < tf - 1e-12
        step_dt = min(dt, tf - t)
        s_dt = sqrt(step_dt)
        dW = randn(dim) .* s_dt
        w_curr .+= dW

        drift = f(u, t)
        diff = g(u, t)

        # Milstein correction term: 0.5 * g'(u) * g(u) * (dW^2 - dt)
        milstein_corr = zeros(Float64, dim)
        for i in 1:dim
            # Derivative of g_i with respect to u_i via dual numbers
            g_i_scalar = x -> g([k == i ? x : u[k] for k in 1:dim], t)[i]
            dg_du = forward_diff(g_i_scalar, u[i])
            milstein_corr[i] = 0.5 * dg_du * diff[i] * (dW[i]^2 - step_dt)
        end

        u .+= step_dt .* drift .+ diff .* dW .+ milstein_corr
        t += step_dt

        push!(t_history, t)
        push!(u_history, copy(u))
        push!(w_history, copy(w_curr))
    end

    return SDESolution(t_history, u_history, w_history)
end
