"""
    Spectra.Core.Config

Global configuration, tolerance presets, threading configurations, and precision controls.
"""

mutable struct SpectraRuntimeConfig
    float_tolerance::Float64
    stiff_solver_atol::Float64
    stiff_solver_rtol::Float64
    max_picard_iterations::Int
    krylov_max_iter::Int
    krylov_restart::Int
    simd_vector_width::Int
    enable_c_kernel::Bool
    verbose_logging::Bool
    deterministic_seed::Int

    function SpectraRuntimeConfig()
        new(
            1e-12, # float_tolerance
            1e-8,  # stiff_solver_atol
            1e-6,  # stiff_solver_rtol
            50,    # max_picard_iterations
            100,   # krylov_max_iter
            30,    # krylov_restart
            4,     # simd_vector_width (AVX2 / NEON)
            true,  # enable_c_kernel
            false, # verbose_logging
            42     # deterministic_seed
        )
    end
end

const GLOBAL_CONFIG = SpectraRuntimeConfig()

"""
    set_tolerance!(tol::Float64)

Set the default numerical zero tolerance for rank determinations and convergence checks.
"""
function set_tolerance!(tol::Float64)
    @assert tol > 0.0 "Tolerance must be strictly positive"
    GLOBAL_CONFIG.float_tolerance = tol
    return tol
end

"""
    get_tolerance() -> Float64

Retrieve current numerical tolerance.
"""
function get_tolerance()::Float64
    return GLOBAL_CONFIG.float_tolerance
end

"""
    enable_c_acceleration!(enable::Bool=true)

Toggle whether native C computational kernels are preferred when available.
"""
function enable_c_acceleration!(enable::Bool=true)
    GLOBAL_CONFIG.enable_c_kernel = enable
    return enable
end
