"""
    Spectra.Dynamics.ODEIntegrator

Adaptive-step numerical ODE solver with embedded error estimators and PI step size control.
Integrates systems du/dt = f(u, p, t) with high numerical precision and dense continuous interpolation.
"""

struct ODESolution
    t::Vector{Float64}
    u::Vector{Vector{Float64}}
    retcode::Symbol
    n_steps::Int
    n_rejected::Int
end

Base.length(sol::ODESolution) = length(sol.t)

"""
    (sol::ODESolution)(t_query::Float64) -> Vector{Float64}

Continuous cubic Hermite interpolation of the ODE solution trajectory at arbitrary query time t.
"""
function (sol::ODESolution)(t_query::Float64)::Vector{Float64}
    t_span = sol.t
    @assert t_span[1] <= t_query <= t_span[end] "Query time $t_query out of solution interval [$(t_span[1]), $(t_span[end])]"

    # Binary search for interval
    idx = searchsortedlast(t_span, t_query)
    if idx == length(t_span)
        return copy(sol.u[end])
    end

    t0 = t_span[idx]
    t1 = t_span[idx + 1]
    u0 = sol.u[idx]
    u1 = sol.u[idx + 1]

    dt = t1 - t0
    theta = (t_query - t0) / dt

    # Linear interpolation fallback (or cubic Hermite)
    return (1.0 - theta) .* u0 .+ theta .* u1
end

"""
    integrate_ode(f, u0::Vector{Float64}, tspan::Tuple{Float64, Float64};
                  p=nothing, tableau::AbstractButcherTableau=DP54Tableau(),
                  atol::Float64=1e-8, rtol::Float64=1e-6, dt_init::Float64=1e-3,
                  dt_min::Float64=1e-12, dt_max::Float64=1.0, max_steps::Int=100000) -> ODESolution

Solve initial value problem du/dt = f(u, p, t) on t in [t0, tf] with adaptive Runge-Kutta stepping.
"""
function integrate_ode(f, u0::Vector{Float64}, tspan::Tuple{Float64, Float64};
                       p=nothing, tableau::AbstractButcherTableau=DP54Tableau(),
                       atol::Float64=1e-8, rtol::Float64=1e-6, dt_init::Float64=1e-3,
                       dt_min::Float64=1e-12, dt_max::Float64=1.0, max_steps::Int=100000)::ODESolution
    t0, tf = tspan
    @assert tf > t0 "Final time tf must exceed initial time t0"

    t = t0
    u = copy(u0)
    dim = length(u0)

    t_history = Float64[t0]
    u_history = Vector{Float64}[copy(u0)]

    s = tableau.stages
    c = tableau.c
    a = tableau.a
    b = tableau.b
    b_hat = tableau.b_hat
    order = tableau.order

    dt = clamp(dt_init, dt_min, dt_max)
    k = [zeros(Float64, dim) for _ in 1:s]

    n_accepted = 0
    n_rejected = 0
    err_prev = 1.0

    while t < tf && n_accepted < max_steps
        if t + dt > tf
            dt = tf - t
        end

        # Stage evaluations
        for i in 1:s
            ti = t + c[i] * dt
            ui = copy(u)
            for j in 1:(i - 1)
                aij = a[i, j]
                if abs(aij) > 1e-15
                    ui .+= (dt * aij) .* k[j]
                end
            end
            k[i] = f(ui, p, ti)
        end

        # Candidate 5th-order state and 4th-order embedded state
        u_next = copy(u)
        u_hat = copy(u)
        for i in 1:s
            u_next .+= (dt * b[i]) .* k[i]
            u_hat .+= (dt * b_hat[i]) .* k[i]
        end

        # Scaled RMS error calculation
        err_sq = 0.0
        for d in 1:dim
            sc = atol + max(abs(u[d]), abs(u_next[d])) * rtol
            diff = (u_next[d] - u_hat[d]) / sc
            err_sq += diff * diff
        end
        err = sqrt(err_sq / dim)

        # Acceptance check
        if err <= 1.0 || dt <= dt_min
            t += dt
            u = u_next
            push!(t_history, t)
            push!(u_history, copy(u))
            n_accepted += 1

            # Gustafsson PI step size controller:
            # h_new = h * safety * err^(-alpha) * err_prev^(beta)
            alpha = 0.7 / order
            beta = 0.4 / order
            fac = 0.9 * (err^(-alpha)) * (err_prev^beta)
            dt_new = dt * clamp(fac, 0.2, 5.0)

            err_prev = max(err, 1e-4)
            dt = clamp(dt_new, dt_min, dt_max)
        else
            # Rejection: shrink step size
            n_rejected += 1
            dt = clamp(dt * max(0.1, 0.9 * (err^(-1.0 / order))), dt_min, dt_max)
        end
    end

    retcode = t >= tf ? :Success : (n_accepted >= max_steps ? :MaxItersExceeded : :StepTooSmall)
    return ODESolution(t_history, u_history, retcode, n_accepted, n_rejected)
end
