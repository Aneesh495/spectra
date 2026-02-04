"""
    Spectra.Viz.TerminalCanvas

Zero-dependency Unicode Braille and ANSI 256-color terminal graphics rasterizer.
Renders sub-pixel 2x4 resolution line curves, phase portraits, and 2D field heatmaps
directly in stdout without requiring any plotting packages or graphical display.
"""

"""
    BrailleCanvas

High-resolution sub-pixel terminal canvas using Unicode Braille Patterns (U+2800..U+28FF).
Each character cell contains 2x4 sub-pixels, providing 4x vertical and 2x horizontal resolution.
"""
mutable struct BrailleCanvas
    width::Int          # Character width
    height::Int         # Character height
    pixel_width::Int    # width * 2
    pixel_height::Int   # height * 4
    grid::Matrix{UInt8} # Braille bitmasks for each char cell
    x_min::Float64
    x_max::Float64
    y_min::Float64
    y_max::Float64

    function BrailleCanvas(width::Int=60, height::Int=20;
                           x_min::Float64=0.0, x_max::Float64=1.0,
                           y_min::Float64=0.0, y_max::Float64=1.0)
        grid = zeros(UInt8, height, width)
        new(width, height, width * 2, height * 4, grid, x_min, x_max, y_min, y_max)
    end
end

# Braille dot bit offsets:
# [ 0x01  0x08 ]
# [ 0x02  0x10 ]
# [ 0x04  0x20 ]
# [ 0x40  0x80 ]
const BRAILLE_DOT_MASKS = [
    0x01 0x08;
    0x02 0x10;
    0x04 0x20;
    0x40 0x80
]

"""
    set_pixel!(canvas::BrailleCanvas, px::Int, py::Int)

Sets sub-pixel at (px, py) where 1 <= px <= pixel_width and 1 <= py <= pixel_height.
Note: py=1 is bottom, py=pixel_height is top.
"""
function set_pixel!(canvas::BrailleCanvas, px::Int, py::Int)
    if px < 1 || px > canvas.pixel_width || py < 1 || py > canvas.pixel_height
        return
    end

    # Invert vertical coordinate so py=pixel_height is top of screen
    inv_py = canvas.pixel_height - py + 1

    char_col = div(px - 1, 2) + 1
    char_row = div(inv_py - 1, 4) + 1

    sub_x = mod(px - 1, 2) + 1
    sub_y = mod(inv_py - 1, 4) + 1

    dot_mask = BRAILLE_DOT_MASKS[sub_y, sub_x]
    canvas.grid[char_row, char_col] |= dot_mask
end

"""
    set_point!(canvas::BrailleCanvas, x::Float64, y::Float64)

Maps continuous coordinates (x, y) into canvas sub-pixel grid.
"""
function set_point!(canvas::BrailleCanvas, x::Float64, y::Float64)
    if canvas.x_max == canvas.x_min || canvas.y_max == canvas.y_min
        return
    end

    norm_x = (x - canvas.x_min) / (canvas.x_max - canvas.x_min)
    norm_y = (y - canvas.y_min) / (canvas.y_max - canvas.y_min)

    px = round(Int, norm_x * (canvas.pixel_width - 1)) + 1
    py = round(Int, norm_y * (canvas.pixel_height - 1)) + 1

    set_pixel!(canvas, px, py)
end

"""
    draw_line!(canvas::BrailleCanvas, x0::Float64, y0::Float64, x1::Float64, y1::Float64)

Bresenham line rasterization in sub-pixel coordinate space.
"""
function draw_line!(canvas::BrailleCanvas, x0::Float64, y0::Float64, x1::Float64, y1::Float64)
    norm_x0 = (x0 - canvas.x_min) / (canvas.x_max - canvas.x_min)
    norm_y0 = (y0 - canvas.y_min) / (canvas.y_max - canvas.y_min)
    norm_x1 = (x1 - canvas.x_min) / (canvas.x_max - canvas.x_min)
    norm_y1 = (y1 - canvas.y_min) / (canvas.y_max - canvas.y_min)

    px0 = round(Int, norm_x0 * (canvas.pixel_width - 1)) + 1
    py0 = round(Int, norm_y0 * (canvas.pixel_height - 1)) + 1
    px1 = round(Int, norm_x1 * (canvas.pixel_width - 1)) + 1
    py1 = round(Int, norm_y1 * (canvas.pixel_height - 1)) + 1

    dx = abs(px1 - px0)
    dy = -abs(py1 - py0)
    sx = px0 < px1 ? 1 : -1
    sy = py0 < py1 ? 1 : -1
    err = dx + dy

    cx = px0
    cy = py0

    while true
        set_pixel!(canvas, cx, cy)
        if cx == px1 && cy == py1
            break
        end
        e2 = 2 * err
        if e2 >= dy
            err += dy
            cx += sx
        end
        if e2 <= dx
            err += dx
            cy += sy
        end
    end
end

"""
    render_to_string(canvas::BrailleCanvas; title::String="", border::Bool=true) -> String

Renders the BrailleCanvas as a multi-line formatted string with optional border and axis labels.
"""
function render_to_string(canvas::BrailleCanvas; title::String="", border::Bool=true)::String
    buf = IOBuffer()

    if !isempty(title)
        println(buf, "  " * title)
    end

    top_border = "  ┌" * repeat("─", canvas.width) * "┐"
    bot_border = "  └" * repeat("─", canvas.width) * "┘"

    if border
        println(buf, top_border)
    end

    for r in 1:canvas.height
        if r == 1
            @printf(buf, "%6.2f │", canvas.y_max)
        elseif r == canvas.height
            @printf(buf, "%6.2f │", canvas.y_min)
        else
            print(buf, "       │")
        end

        for c in 1:canvas.width
            mask = canvas.grid[r, c]
            braille_char = Char(0x2800 + mask)
            print(buf, braille_char)
        end
        println(buf, "│")
    end

    if border
        println(buf, bot_border)
    end

    # Bottom axis label
    pad_len = max(0, canvas.width - 20)
    @printf(buf, "        %-10.2f", canvas.x_min)
    print(buf, repeat(" ", pad_len))
    @printf(buf, "%10.2f\n", canvas.x_max)

    return String(take!(buf))
end
