"""
    Spectra.Operators.PDESolvers

High-accuracy synthetic data generators and numerical PDE benchmarks in pure Julia.
Provides ground-truth simulation engines for operator learning benchmark problems:
1. 1D Viscous Burgers equation (pseudo-spectral RK4).
2. 2D Darcy flow through heterogeneous porous media (finite-difference CG).
3. 2D Incompressible Navier-Stokes vorticity transport (Fourier-Galerkin pseudo-spectral).
"""

"""
    gaussian_random_field_1d(n_points::Int; length_scale::Float64=0.2, variance::Float64=1.0) -> Vector{Float64}

Generates a smooth 1D periodic Gaussian Random Field via spectral representation.
"""
function gaussian_random_field_1d(n_points::Int; length_scale::Float64=0.2, variance::Float64=1.0)::Vector{Float64}
    k_max = div(n_points, 2)
    spec = zeros(ComplexF64, n_points)

    for k in 0:(n_points - 1)
        k_val = k <= k_max ? k : k - n_points
        # Power spectrum E(k) ~ exp(-length_scale^2 * k^2 / 2)
        power = variance * exp(-0.5 * (k_val * 2.0 * pi * length_scale)^2)
        amp = sqrt(max(power, 1e-14))
        phase = rand() * 2.0 * pi
        spec[k + 1] = amp * exp(im * phase)
    end
    # Ensure real signal property: spec[n - k] = conj(spec[k])
    spec[1] = real(spec[1])
    if iseven(n_points)
        spec[k_max + 1] = real(spec[k_max + 1])
    end

    field = real.(ifft_1d(spec)) .* sqrt(n_points)
    return field
end

"""
    solve_burgers_1d(u0::Vector{Float64}, nu::Float64, t_span::Tuple{Float64, Float64};
                     dt::Float64=1e-3) -> Matrix{Float64}

Solves 1D viscous Burgers equation on [0, 1) with periodic boundary conditions:
    du/dt + u * du/dx = nu * d^2u/dx^2
Using pseudo-spectral spatial differentiation and classical Runge-Kutta 4th order (RK4).
Returns matrix of snapshots (n_steps + 1, n_spatial).
"""
function solve_burgers_1d(u0::Vector{Float64}, nu::Float64, t_span::Tuple{Float64, Float64};
                          dt::Float64=1e-3)::Matrix{Float64}
    n = length(u0)
    t0, t_final = t_span
    n_steps = max(1, round(Int, (t_final - t0) / dt))
    actual_dt = (t_final - t0) / n_steps

    # Wavenumbers for domain [0, 1) -> L = 1.0 -> k_scale = 2*pi
    k_vec = zeros(Float64, n)
    half = div(n, 2)
    for i in 1:n
        k = i <= half ? (i - 1) : (i - 1 - n)
        k_vec[i] = 2.0 * pi * k
    end

    # Spatial derivative operator in Fourier space
    rhs = function(u::Vector{Float64})
        u_hat = fft_1d(ComplexF64.(u))
        # du/dx = ifft(im * k * u_hat)
        du_hat = (im .* k_vec) .* u_hat
        du_dx = real.(ifft_1d(du_hat))

        # d^2u/dx^2 = ifft(-k^2 * u_hat)
        d2u_hat = (-k_vec .^ 2) .* u_hat
        d2u_dx2 = real.(ifft_1d(d2u_hat))

        return -u .* du_dx .+ nu .* d2u_dx2
    end

    history = zeros(Float64, n_steps + 1, n)
    history[1, :] = u0
    u_curr = copy(u0)

    for step in 1:n_steps
        k1 = rhs(u_curr)
        k2 = rhs(u_curr .+ 0.5 .* actual_dt .* k1)
        k3 = rhs(u_curr .+ 0.5 .* actual_dt .* k2)
        k4 = rhs(u_curr .+ actual_dt .* k3)

        u_curr .+= (actual_dt / 6.0) .* (k1 .+ 2.0 .* k2 .+ 2.0 .* k3 .+ k4)
        history[step + 1, :] = u_curr
    end

    return history
end

"""
    solve_darcy_2d(a_field::Matrix{Float64}, f_val::Float64=1.0;
                   tol::Float64=1e-7, max_iters::Int=2000) -> Matrix{Float64}

Solves 2D steady-state Darcy flow on unit square [0, 1]^2 with zero Dirichlet BC:
    -div(a(x, y) * grad(u)) = f(x, y)
Discretized using conservative 5-point finite differences with harmonic average permeability.
System is solved via Preconditioned Conjugate Gradient (PCG).
"""
function solve_darcy_2d(a_field::Matrix{Float64}, f_val::Float64=1.0;
                        tol::Float64=1e-7, max_iters::Int=2000)::Matrix{Float64}
    Nx, Ny = size(a_field)
    dx = 1.0 / (Nx + 1)
    dy = 1.0 / (Ny + 1)
    dx2 = dx^2
    dy2 = dy^2

    N_interior = Nx * Ny
    # Grid indexing: (i, j) -> idx = (j - 1) * Nx + i
    grid_idx = (i, j) -> (j - 1) * Nx + i

    coo = COOMatrix(N_interior, N_interior)
    b = fill(f_val, N_interior)

    for j in 1:Ny
        for i in 1:Nx
            curr = grid_idx(i, j)
            a_center = a_field[i, j]

            diag_val = 0.0

            # West neighbor (i-1, j)
            if i > 1
                a_w = 2.0 * a_center * a_field[i-1, j] / (a_center + a_field[i-1, j] + 1e-14)
                add_entry!(coo, curr, grid_idx(i-1, j), -a_w / dx2)
                diag_val += a_w / dx2
            else
                # Dirichlet boundary u=0 at x=0
                diag_val += 2.0 * a_center / dx2
            end

            # East neighbor (i+1, j)
            if i < Nx
                a_e = 2.0 * a_center * a_field[i+1, j] / (a_center + a_field[i+1, j] + 1e-14)
                add_entry!(coo, curr, grid_idx(i+1, j), -a_e / dx2)
                diag_val += a_e / dx2
            else
                # Dirichlet boundary u=0 at x=1
                diag_val += 2.0 * a_center / dx2
            end

            # South neighbor (i, j-1)
            if j > 1
                a_s = 2.0 * a_center * a_field[i, j-1] / (a_center + a_field[i, j-1] + 1e-14)
                add_entry!(coo, curr, grid_idx(i, j-1), -a_s / dy2)
                diag_val += a_s / dy2
            else
                # Dirichlet boundary u=0 at y=0
                diag_val += 2.0 * a_center / dy2
            end

            # North neighbor (i, j+1)
            if j < Ny
                a_n = 2.0 * a_center * a_field[i, j+1] / (a_center + a_field[i, j+1] + 1e-14)
                add_entry!(coo, curr, grid_idx(i, j+1), -a_n / dy2)
                diag_val += a_n / dy2
            else
                # Dirichlet boundary u=0 at y=1
                diag_val += 2.0 * a_center / dy2
            end

            add_entry!(coo, curr, curr, diag_val)
        end
    end

    A_csc = from_coo(coo)
    u_vec, conv, residuals = sparse_cg(A_csc, b; tol=tol, max_iter=max_iters)

    u_field = reshape(u_vec, Nx, Ny)
    return u_field
end

"""
    solve_navier_stokes_2d(w0::Matrix{Float64}, nu::Float64, t_final::Float64;
                           dt::Float64=1e-3) -> Matrix{Float64}

Solves 2D incompressible Navier-Stokes vorticity transport equation on [0, 2pi]^2:
    d(omega)/dt + u * grad(omega) = nu * laplacian(omega)
    laplacian(psi) = -omega
    u = (d(psi)/dy, -d(psi)/dx)
Using Fourier pseudo-spectral method with RK4 time stepping.
Returns final vorticity field omega(t_final).
"""
function solve_navier_stokes_2d(w0::Matrix{Float64}, nu::Float64, t_final::Float64;
                                dt::Float64=1e-3)::Matrix{Float64}
    Nx, Ny = size(w0)
    n_steps = max(1, round(Int, t_final / dt))
    actual_dt = t_final / n_steps

    # 2D Wavenumber grid
    kx = zeros(Float64, Nx)
    half_x = div(Nx, 2)
    for i in 1:Nx
        kx[i] = i <= half_x ? (i - 1) : (i - 1 - Nx)
    end

    ky = zeros(Float64, Ny)
    half_y = div(Ny, 2)
    for j in 1:Ny
        ky[j] = j <= half_y ? (j - 1) : (j - 1 - Ny)
    end

    # K^2 grid
    k_sq = zeros(Float64, Nx, Ny)
    for j in 1:Ny
        for i in 1:Nx
            k_sq[i, j] = kx[i]^2 + ky[j]^2
        end
    end

    rhs = function(w::Matrix{Float64})
        w_hat = fft_2d(ComplexF64.(w))

        # Solve Poisson equation: psi_hat = w_hat / k^2 (zero at k=0)
        psi_hat = zeros(ComplexF64, Nx, Ny)
        for j in 1:Ny
            for i in 1:Nx
                if k_sq[i, j] > 1e-12
                    psi_hat[i, j] = w_hat[i, j] / k_sq[i, j]
                end
            end
        end

        # Velocities: u = d(psi)/dy, v = -d(psi)/dx
        u_hat = zeros(ComplexF64, Nx, Ny)
        v_hat = zeros(ComplexF64, Nx, Ny)
        dw_dx_hat = zeros(ComplexF64, Nx, Ny)
        dw_dy_hat = zeros(ComplexF64, Nx, Ny)

        for j in 1:Ny
            for i in 1:Nx
                u_hat[i, j] = im * ky[j] * psi_hat[i, j]
                v_hat[i, j] = -im * kx[i] * psi_hat[i, j]
                dw_dx_hat[i, j] = im * kx[i] * w_hat[i, j]
                dw_dy_hat[i, j] = im * ky[j] * w_hat[i, j]
            end
        end

        u_phys = real.(ifft_2d(u_hat))
        v_phys = real.(ifft_2d(v_hat))
        dw_dx  = real.(ifft_2d(dw_dx_hat))
        dw_dy  = real.(ifft_2d(dw_dy_hat))

        # Advection term: u * dw/dx + v * dw/dy
        advection = u_phys .* dw_dx .+ v_phys .* dw_dy

        # Diffusion term: nu * laplacian(w) = ifft(-nu * k^2 * w_hat)
        diff_hat = -nu .* k_sq .* w_hat
        diffusion = real.(ifft_2d(diff_hat))

        return -advection .+ diffusion
    end

    w_curr = copy(w0)
    for step in 1:n_steps
        k1 = rhs(w_curr)
        k2 = rhs(w_curr .+ 0.5 .* actual_dt .* k1)
        k3 = rhs(w_curr .+ 0.5 .* actual_dt .* k2)
        k4 = rhs(w_curr .+ actual_dt .* k3)

        w_curr .+= (actual_dt / 6.0) .* (k1 .+ 2.0 .* k2 .+ 2.0 .* k3 .+ k4)
    end

    return w_curr
end
