"""
    Spectra.Geometry.RiemannianOpt

Optimization algorithms on Riemannian manifolds.
Implements Riemannian Gradient Descent (RGD), Riemannian Conjugate Gradients (RCG),
and Riemannian Adam (R-Adam) with vector transport and geodesic retractions.
"""

"""
    riemannian_gradient(M::AbstractManifold, p, euclidean_grad)

Convert Euclidean gradient into intrinsic Riemannian gradient via tangent space projection.
"""
function riemannian_gradient(M::AbstractManifold, p, euclidean_grad)
    return project_tangent(M, p, euclidean_grad)
end

"""
    riemannian_gradient_descent(M::AbstractManifold, cost_and_grad, p0;
                                lr::Float64=0.05, max_iter::Int=100, tol::Float64=1e-6)

Riemannian Gradient Descent (RGD) with retraction step p_{k+1} = R_{p_k}(-lr * grad_M f(p_k)).
"""
function riemannian_gradient_descent(M::AbstractManifold, cost_and_grad, p0;
                                     lr::Float64=0.05, max_iter::Int=100, tol::Float64=1e-6)
    p = copy(p0)
    loss_history = Float64[]

    for iter in 1:max_iter
        loss, euc_grad = cost_and_grad(p)
        push!(loss_history, loss)

        r_grad = riemannian_gradient(M, p, euc_grad)
        grad_norm = norm_tangent(M, p, r_grad)

        if grad_norm < tol
            break
        end

        step = -lr .* r_grad
        p = retraction(M, p, step)
    end

    return (p, loss_history)
end

"""
    riemannian_cg(M::AbstractManifold, cost_and_grad, p0;
                  max_iter::Int=100, tol::Float64=1e-6) -> (p_opt, loss_history)

Riemannian Nonlinear Conjugate Gradient method with Polak-Ribiere parameter and vector transport.
"""
function riemannian_cg(M::AbstractManifold, cost_and_grad, p0;
                       max_iter::Int=100, tol::Float64=1e-6)
    p = copy(p0)
    loss, euc_grad = cost_and_grad(p)
    loss_history = Float64[loss]

    r_grad = riemannian_gradient(M, p, euc_grad)
    if norm_tangent(M, p, r_grad) < tol
        return (p, loss_history)
    end

    eta = -copy(r_grad) # Search direction

    for iter in 1:max_iter
        # Armijo line search along retraction
        alpha = 1.0
        slope = inner_product(M, p, r_grad, eta)
        if slope >= 0.0
            eta = -copy(r_grad)
            slope = inner_product(M, p, r_grad, eta)
        end

        p_next = p
        loss_next = loss
        euc_grad_next = euc_grad

        for _ in 1:20
            p_next = retraction(M, p, alpha .* eta)
            loss_next, euc_grad_next = cost_and_grad(p_next)
            if loss_next <= loss + 1e-4 * alpha * slope
                break
            end
            alpha *= 0.5
        end

        r_grad_next = riemannian_gradient(M, p_next, euc_grad_next)
        norm_next = norm_tangent(M, p_next, r_grad_next)
        push!(loss_history, loss_next)

        if norm_next < tol
            p = p_next
            break
        end

        # Vector transport of eta and r_grad to T_{p_next} M
        eta_transported = project_tangent(M, p_next, eta)
        r_grad_transported = project_tangent(M, p_next, r_grad)

        # Polak-Ribiere beta
        diff_grad = r_grad_next .- r_grad_transported
        num = inner_product(M, p_next, r_grad_next, diff_grad)
        den = inner_product(M, p, r_grad, r_grad)
        beta_PR = max(0.0, num / max(den, 1e-15))

        eta = -r_grad_next .+ beta_PR .* eta_transported

        p = p_next
        loss = loss_next
        r_grad = r_grad_next
        euc_grad = euc_grad_next
    end

    return (p, loss_history)
end

"""
    riemannian_adam(M::AbstractManifold, cost_and_grad, p0;
                    lr::Float64=0.01, beta1::Float64=0.9, beta2::Float64=0.999,
                    eps::Float64=1e-8, max_iter::Int=100, tol::Float64=1e-6)

Riemannian Adam optimizer with parallel/vector transport of first and second moment tangent vectors.
"""
function riemannian_adam(M::AbstractManifold, cost_and_grad, p0;
                         lr::Float64=0.01, beta1::Float64=0.9, beta2::Float64=0.999,
                         eps::Float64=1e-8, max_iter::Int=100, tol::Float64=1e-6)
    p = copy(p0)
    loss_history = Float64[]

    m_tangent = zeros(Float64, size(p))
    v_tangent = zeros(Float64, size(p))

    for t in 1:max_iter
        loss, euc_grad = cost_and_grad(p)
        push!(loss_history, loss)

        r_grad = riemannian_gradient(M, p, euc_grad)
        if norm_tangent(M, p, r_grad) < tol
            break
        end

        # Update biased moments on T_p M
        m_tangent .= beta1 .* m_tangent .+ (1.0 - beta1) .* r_grad
        v_tangent .= beta2 .* v_tangent .+ (1.0 - beta2) .* (r_grad .^ 2)

        # Bias correction
        bc1 = 1.0 - beta1^t
        bc2 = 1.0 - beta2^t

        m_hat = m_tangent ./ bc1
        v_hat = v_tangent ./ bc2

        # Step direction
        step_dir = -lr .* (m_hat ./ (sqrt.(v_hat) .+ eps))
        step_dir_proj = project_tangent(M, p, step_dir)

        p_next = retraction(M, p, step_dir_proj)

        # Transport moment vectors to new tangent space
        m_tangent = project_tangent(M, p_next, m_tangent)
        v_tangent = project_tangent(M, p_next, v_tangent)

        p = p_next
    end

    return (p, loss_history)
end
