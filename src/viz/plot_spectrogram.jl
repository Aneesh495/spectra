"""
    Spectra.Viz.PlotSpectrogram

High-level terminal visualization routines for dynamical systems, neural operators,
and spectral decompositions.
Includes:
1. terminal_plot: Line plots of 1D wavefields and trajectories.
2. terminal_phase_portrait: (q, p) phase space orbits and symplectic trajectories.
3. terminal_heatmap: 2D scalar fields (Darcy pressure, vorticity) via ANSI truecolor blocks.
4. terminal_spectrum_bars: Koopman/sheaf eigenvalue spectra and spectral gaps.
"""

"""
    terminal_plot(y::Vector{Float64}; x::Union{Vector{Float64}, Nothing}=nothing,
                  width::Int=60, height::Int=18, title::String="")

Renders a 1D function or time series directly in stdout using sub-pixel Braille curves.
"""
function terminal_plot(y::Vector{Float64}; x::Union{Vector{Float64}, Nothing}=nothing,
                       width::Int=60, height::Int=18, title::String="")
    n = length(y)
    xs = x === nothing ? collect(range(0.0, 1.0, length=n)) : copy(x)

    x_min, x_max = extrema(xs)
    y_min, y_max = extrema(y)

    # Add small padding if flat
    if y_min == y_max
        y_min -= 0.5
        y_max += 0.5
    end
    if x_min == x_max
        x_min -= 0.5
        x_max += 0.5
    end

    canvas = BrailleCanvas(width, height; x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

    for i in 1:(n - 1)
        draw_line!(canvas, xs[i], y[i], xs[i + 1], y[i + 1])
    end

    rendered = render_to_string(canvas; title=title)
    print(rendered)
    return rendered
end

"""
    terminal_phase_portrait(q::Vector{Float64}, p::Vector{Float64};
                            width::Int=50, height::Int=20, title::String="Phase Portrait")

Renders a (q, p) phase space trajectory.
"""
function terminal_phase_portrait(q::Vector{Float64}, p::Vector{Float64};
                                 width::Int=50, height::Int=20, title::String="Phase Space Orbit")
    @assert length(q) == length(p) "q and p vectors must have identical length"
    n = length(q)

    q_min, q_max = extrema(q)
    p_min, p_max = extrema(p)

    canvas = BrailleCanvas(width, height; x_min=q_min, x_max=q_max, y_min=p_min, y_max=p_max)

    for i in 1:(n - 1)
        draw_line!(canvas, q[i], p[i], q[i + 1], p[i + 1])
    end

    rendered = render_to_string(canvas; title=title)
    print(rendered)
    return rendered
end

"""
    rgb_colormap(val_normalized::Float64) -> (Int, Int, Int)

Viridis-inspired smooth colormap mapping [0, 1] to (R, G, B) [0, 255].
"""
function rgb_colormap(val_norm::Float64)::Tuple{Int, Int, Int}
    t = clamp(val_norm, 0.0, 1.0)
    # Piecewise approximation to viridis/plasma
    r = round(Int, 255.0 * clamp(0.5 + 0.5 * sin(2.0 * pi * (t - 0.25)), 0.0, 1.0))
    g = round(Int, 255.0 * clamp(sin(pi * t), 0.0, 1.0))
    b = round(Int, 255.0 * clamp(0.5 + 0.5 * cos(2.0 * pi * t), 0.0, 1.0))
    return r, g, b
end

"""
    terminal_heatmap(M::Matrix{Float64}; title::String="", width::Int=40, height::Int=20)

Renders a 2D scalar field directly to terminal using ANSI truecolor block characters.
Resamples field to terminal character dimensions with half-block upper/lower resolution.
"""
function terminal_heatmap(M::Matrix{Float64}; title::String="", width::Int=40, height::Int=20)
    orig_rows, orig_cols = size(M)
    min_val, max_val = extrema(M)
    val_range = max_val - min_val
    if val_range == 0.0
        val_range = 1.0
    end

    buf = IOBuffer()
    if !isempty(title)
        println(buf, "  " * title)
    end

    # Resample grid to width x height
    # Each terminal row renders two vertical pixels using unicode upper half block '▀'
    # with foreground color = top pixel, background color = bottom pixel.
    total_vert_pixels = height * 2

    for r in 1:height
        row_top = div((2 * r - 2) * orig_rows, total_vert_pixels) + 1
        row_bot = div((2 * r - 1) * orig_rows, total_vert_pixels) + 1
        row_top = min(row_top, orig_rows)
        row_bot = min(row_bot, orig_rows)

        print(buf, "  ")
        for c in 1:width
            col = div((c - 1) * orig_cols, width) + 1
            col = min(col, orig_cols)

            val_t = (M[row_top, col] - min_val) / val_range
            val_b = (M[row_bot, col] - min_val) / val_range

            rt, gt, bt = rgb_colormap(val_t)
            rb, gb, bb = rgb_colormap(val_b)

            # ANSI truecolor: foreground (38;2;r;g;b), background (48;2;r;g;b)
            @printf(buf, "\e[38;2;%d;%d;%dm\e[48;2;%d;%d;%dm▀", rt, gt, bt, rb, gb, bb)
        end
        # Reset ANSI formatting at end of row
        println(buf, "\e[0m")
    end

    # Render min and max colorbar legend
    pad_len = max(0, width - 24)
    @printf(buf, "  min: %6.2e", min_val)
    print(buf, repeat(" ", pad_len))
    @printf(buf, " max: %6.2e\n", max_val)

    rendered = String(take!(buf))
    print(rendered)
    return rendered
end

"""
    terminal_spectrum_bars(eigenvalues::Vector{ComplexF64}; top_k::Int=10, title::String="Operator Spectrum")

Renders horizontal ASCII bars showing the magnitudes of the top dominant eigenvalues.
"""
function terminal_spectrum_bars(eigenvalues::Vector{ComplexF64}; top_k::Int=10, title::String="Operator Spectrum")
    mags = sort(abs.(eigenvalues); rev=true)
    k = min(top_k, length(mags))

    buf = IOBuffer()
    println(buf, "  " * title)
    println(buf, "  " * repeat("─", 50))

    max_mag = mags[1]
    if max_mag == 0.0
        max_mag = 1.0
    end

    for i in 1:k
        mag = mags[i]
        bar_len = round(Int, (mag / max_mag) * 30)
        bar = repeat("█", bar_len)
        @printf(buf, "  λ%-2d │ %-30s │ |λ| = %7.4f\n", i, bar, mag)
    end
    println(buf, "  " * repeat("─", 50))

    if k >= 2
        gap = mags[1] - mags[2]
        @printf(buf, "  Spectral Gap (λ₁ - λ₂): %7.4f\n", gap)
    end

    rendered = String(take!(buf))
    print(rendered)
    return rendered
end
