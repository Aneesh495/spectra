using Test
using Spectra

@testset "Reverse-Mode AD & Dynamic Tape" begin
    # 1. Quadratic functional gradient
    f_quad = v -> v[1]^2 + 3.0 * v[1] * v[2] + v[2]^2
    v_in = [1.5, 2.5]
    val, grad_rev = value_and_gradient(f_quad, v_in)
    @test val ≈ f_quad(v_in)
    # df/dv1 = 2*v1 + 3*v2 = 3.0 + 7.5 = 10.5
    # df/dv2 = 3*v1 + 2*v2 = 4.5 + 5.0 = 9.5
    @test grad_rev[1] ≈ 10.5
    @test grad_rev[2] ≈ 9.5

    # 2. Composite polynomial loss: L = (x * y + z)^2
    f_comp = v -> (v[1] * v[2] + v[3])^2
    v_comp = [2.0, 3.0, 4.0]
    val_comp, grad_comp = value_and_gradient(f_comp, v_comp)
    @test val_comp == 100.0
    # dL/dx = 2*(x*y + z)*y = 2*10*3 = 60.0
    # dL/dy = 2*(x*y + z)*x = 2*10*2 = 40.0
    # dL/dz = 2*(x*y + z)*1 = 2*10 = 20.0
    @test grad_comp[1] ≈ 60.0
    @test grad_comp[2] ≈ 40.0
    @test grad_comp[3] ≈ 20.0

    # 3. Activations
    f_relu = v -> sum(relu.(v))
    v_relu = [-1.0, 0.0, 2.0]
    _, grad_relu = value_and_gradient(f_relu, v_relu)
    @test grad_relu[1] == 0.0
    @test grad_relu[3] == 1.0

    # 4. Sigmoid
    f_sig = v -> sigmoid(v[1])
    v_sig = [0.0]
    val_s, grad_s = value_and_gradient(f_sig, v_sig)
    @test val_s ≈ 0.5
    @test grad_s[1] ≈ 0.25 # sig * (1 - sig)
end
