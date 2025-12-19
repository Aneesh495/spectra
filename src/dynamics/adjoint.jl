"""
    Spectra.Dynamics.Adjoint

Continuous adjoint sensitivity analysis for Neural Ordinary Differential Equations.
Computes exact gradients of scalar losses with respect to initial conditions and parameters
in O(1) constant memory by integrating the augmented adjoint system backwards in time:
  d a(t) / dt = - a(t)^T * df / dz (z(t), theta, t)
  d (dL / d theta) / dt = - a(t)^T * df / d theta (z(t), theta, t).
"""

struct AdjointResult
    loss::Float64
    z_final::Vector{Float64}
    grad_z0::Vector{Float64}
    grad_theta::Vector{Float64}
end

"""
    adjoint_sensitivity(f, loss_fn, z0::Vector{Float64}, theta::Vector{Float64},
                        tspan::Tuple{Float64, Float64};
                        atol::Float64=1e-8, rtol::Float64=1e-6) -> AdjointResult

Compute exact loss and gradients with respect to z0 and parameter vector theta
via continuous Pontryagin adjoint backward time integration.
"""
function adjoint_sensitivity(f, loss_fn, z0::Vector{Float64}, theta::Vector{Float64},
                            tspan::Tuple{Float64, Float64};
                            atol::Float64=1e-8, rtol::Float64=1e-6)::AdjointResult
    t0, t1 = tspan
    dim_z = length(z0)
    dim_theta = length(theta)

    # 1. Forward Pass to compute z(t1)
    f_forward = (z, p, t) -> f(z, theta, t)
    sol_forward = integrate_ode(f_forward, z0, (t0, t1); atol=atol, rtol=rtol)
    z_final = sol_forward.u[end]

    # 2. Evaluate scalar loss and terminal adjoint condition a(t1) = dL / dz(t1)
    loss, a_terminal = value_and_gradient(loss_fn, z_final)

    # 3. Augmented backward system state:
    # s(t) = [z(t) (dim_z); a(t) (dim_z); grad_theta (dim_theta)]
    s_init = [z_final; a_terminal; zeros(Float64, dim_theta)]

    # Augmented backward vector field (integrating backwards in tau = t1 - t, d/d tau = - d/dt)
    function augmented_backward(s, p, tau)
        t = t1 - tau
        z = s[1:dim_z]
        a = s[(dim_z + 1):(2 * dim_z)]

        # Forward vector field at current state
        fz = f(z, theta, t)

        # Vector-Jacobian products:
        # a^T (df / dz) and a^T (df / d theta)
        # Using forward AD directional derivatives:
        # (J_z)^T a:
        Jz_T_a = zeros(Float64, dim_z)
        for i in 1:dim_z
            for m in 1:dim_z
                f_scalar = z_val -> f([k == i ? z_val : z[k] for k in 1:dim_z], theta, t)[m]
                d_val = forward_diff(f_scalar, z[i])
                Jz_T_a[i] += a[m] * d_val
            end
        end

        # (J_theta)^T a:
        Jtheta_T_a = zeros(Float64, dim_theta)
        for j in 1:dim_theta
            for m in 1:dim_z
                f_th_scalar = th_val -> f(z, [k == j ? th_val : theta[k] for k in 1:dim_theta], t)[m]
                d_th = forward_diff(f_th_scalar, theta[j])
                Jtheta_T_a[j] += a[m] * d_th
            end
        end

        # Backward integration: dz/dtau = -f, da/dtau = Jz_T_a, d(grad_th)/dtau = Jtheta_T_a
        ds_dz = -fz
        ds_da = Jz_T_a
        ds_dtheta = Jtheta_T_a

        return [ds_dz; ds_da; ds_dtheta]
    end

    sol_backward = integrate_ode(augmented_backward, s_init, (0.0, t1 - t0); atol=atol, rtol=rtol)
    s_final = sol_backward.u[end]

    grad_z0 = s_final[(dim_z + 1):(2 * dim_z)]
    grad_theta = s_final[(2 * dim_z + 1):end]

    return AdjointResult(loss, z_final, grad_z0, grad_theta)
end
