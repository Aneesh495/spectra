"""
    Spectra.CLI.Dispatcher

Unified command-line interface entry point for Spectra.
Dispatches commands:
  spectra test        Run complete verification test suite
  spectra bench       Run performance benchmark suite
  spectra solve       Solve benchmark PDE (burgers, darcy, navier-stokes)
  spectra visualize   Generate interactive WebGL studio dashboard
  spectra demo [1-5]  Run showcase example demonstrations
  spectra info        Display runtime configuration and SIMD status
"""

const SPECTRA_BANNER = """
   _____ ____  ________________  ___ 
  / ___// __ \\/ ____/ ____/_  __/ __ \\
  \\__ \\/ /_/ / __/ / /     / / / /_/ /
 ___/ / ____/ /___/ /___  / / / _, _/ 
/____/_/   /_____/\\____/ /_/ /_/ |_|  
Spectral Operator Learning & Sheaf Cohomology Engine
"""

"""
    cli_main(args::Vector{String}=ARGS) -> Int

Main entry point for command-line execution.
"""
function cli_main(args::Vector{String}=ARGS)::Int
    if isempty(args) || args[1] in ["-h", "--help", "help"]
        print_cli_help()
        return 0
    end

    cmd = lowercase(args[1])
    sub_args = length(args) > 1 ? args[2:end] : String[]

    if cmd == "test"
        return handle_test(sub_args)
    elseif cmd == "bench"
        return handle_bench(sub_args)
    elseif cmd == "solve"
        return handle_solve(sub_args)
    elseif cmd == "visualize" || cmd == "viz"
        return handle_visualize(sub_args)
    elseif cmd == "demo"
        return handle_demo(sub_args)
    elseif cmd == "info"
        return handle_info(sub_args)
    else
        println(stderr, "Unknown command: ", cmd)
        println(stderr, "Run 'spectra help' for usage instructions.")
        return 1
    end
end

function print_cli_help()
    println(SPECTRA_BANNER)
    println("USAGE:")
    println("  spectra <command> [arguments...]\n")
    println("COMMANDS:")
    println("  test                  Execute comprehensive unit and integration test suite")
    println("  bench                 Run microbenchmarks (GEMM, tensor contraction, integrators)")
    println("  solve --pde <type>    Solve benchmark PDE: burgers, darcy, or navier-stokes")
    println("  visualize             Generate standalone WebGL interactive studio (HTML5)")
    println("  demo <1-5>            Run showcase example script")
    println("  info                  Inspect runtime configuration and SIMD acceleration status")
    println("  help, --help          Print this help message\n")
end

function handle_info(args::Vector{String})::Int
    println(SPECTRA_BANNER)
    println("--- Runtime Environment ---")
    println("  Julia Version       : ", VERSION)
    println("  OS Platform         : ", Sys.isapple() ? "macOS" : (Sys.islinux() ? "Linux" : "Windows"), " (", Sys.ARCH, ")")
    println("  CPU Threads         : ", Threads.nthreads())
    println("  C Kernel Compiled   : ", is_c_kernel_available() ? "YES" : "NO (Pure Julia fallback active)")
    println("  C Acceleration Mode : ", GLOBAL_CONFIG.enable_c_kernel ? "ENABLED" : "DISABLED")
    println("  Default Tolerance   : ", GLOBAL_CONFIG.float_tolerance)
    println("  SIMD Vector Width   : ", GLOBAL_CONFIG.simd_vector_width, " (AVX2 / ARM NEON)")
    return 0
end

function handle_visualize(args::Vector{String})::Int
    output = "spectra_dashboard.html"
    for i in 1:length(args)
        if args[i] in ["-o", "--output"] && i < length(args)
            output = args[i + 1]
        end
    end
    println("[*] Generating Spectra Interactive Studio Dashboard...")
    path = generate_html_dashboard(output)
    println("[+] Successfully written dashboard to: ", path)
    println("[*] Open file in browser: open ", path)
    return 0
end

function handle_solve(args::Vector{String})::Int
    pde = "burgers"
    for i in 1:length(args)
        if args[i] in ["--pde", "-p"] && i < length(args)
            pde = lowercase(args[i + 1])
        end
    end

    if pde == "burgers"
        println("[*] Solving 1D Viscous Burgers equation (nu=0.01)...")
        u0 = sin.(2.0 .* pi .* (0:63) ./ 64)
        sol = solve_burgers_1d(u0, 0.01, (0.0, 0.2); dt=0.002)
        println("[+] Solution computed (", size(sol, 1), " timesteps x ", size(sol, 2), " spatial points).")
        println("[*] Terminal Phase Wavefield at t=0.2:")
        terminal_plot(sol[end, :]; title="Burgers Shock Waveform u(x, t=0.2)")
    elseif pde == "darcy"
        println("[*] Solving 2D Steady Darcy Flow through heterogeneous medium...")
        a_field = [1.0 + 0.5 * sin(2pi*x)*cos(2pi*y) for x in range(0, 1, length=24), y in range(0, 1, length=24)]
        u_field = solve_darcy_2d(a_field, 1.0; tol=1e-6)
        println("[+] Darcy flow pressure field converged.")
        println("[*] Pressure Distribution Heatmap:")
        terminal_heatmap(u_field; title="Darcy Pressure Field u(x, y)", width=24, height=8)
    elseif pde == "navier-stokes" || pde == "ns"
        println("[*] Solving 2D Incompressible Navier-Stokes vorticity equation...")
        sx = sin.(collect(range(0, 2pi, length=20)))
        cy = cos.(collect(range(0, 2pi, length=20)))
        w0 = sx * cy'
        w_final = solve_navier_stokes_2d(w0, 0.001, 0.02; dt=0.002)
        println("[+] Navier-Stokes vorticity evolved to t=0.02.")
        terminal_heatmap(w_final; title="Vorticity Field omega(x, y, t=0.02)", width=20, height=8)
    else
        println(stderr, "Unknown PDE type: ", pde, ". Options: burgers, darcy, navier-stokes.")
        return 1
    end
    return 0
end

function handle_test(args::Vector{String})::Int
    test_file = normpath(joinpath(@__DIR__, "..", "..", "test", "runtests.jl"))
    if !isfile(test_file)
        println(stderr, "Error: test/runtests.jl not found at ", test_file)
        return 1
    end
    include(test_file)
    return 0
end

function handle_bench(args::Vector{String})::Int
    bench_file = normpath(joinpath(@__DIR__, "..", "..", "benchmarks", "suite.jl"))
    if !isfile(bench_file)
        println(stderr, "Error: benchmarks/suite.jl not found at ", bench_file)
        return 1
    end
    include(bench_file)
    return 0
end

function handle_demo(args::Vector{String})::Int
    demo_id = isempty(args) ? "1" : args[1]
    demos = Dict(
        "1" => "01_neural_hamiltonian.jl",
        "2" => "02_cellular_sheaf_learning.jl",
        "3" => "03_fourier_neural_operator.jl",
        "4" => "04_riemannian_optimization.jl",
        "5" => "05_koopman_chaotic_flow.jl"
    )

    if !haskey(demos, demo_id)
        println(stderr, "Unknown demo ID: ", demo_id, ". Available demos: 1, 2, 3, 4, 5.")
        return 1
    end

    file = normpath(joinpath(@__DIR__, "..", "..", "examples", demos[demo_id]))
    if !isfile(file)
        println(stderr, "Demo file not found at: ", file)
        return 1
    end

    println("[*] Launching Spectra Showcase Demo #", demo_id, " (", demos[demo_id], ")...")
    include(file)
    return 0
end
