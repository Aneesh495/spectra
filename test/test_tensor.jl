using Test
using Spectra

@testset "Tensor Engine" begin
    # Allocation & Basic properties
    T = zeros_tensor(Float64, (3, 4, 5))
    @test size(T) == (3, 4, 5)
    @test length(T) == 60
    @test eltype(T) == Float64
    @test is_contiguous(T)

    # Ones and Random
    O = ones_tensor(Float64, (2, 3))
    @test all(to_array(O) .== 1.0)

    R = randn_tensor(Float64, (4, 4))
    @test size(R) == (4, 4)


    # Element access
    T[1, 2, 3] = 42.0
    @test T[1, 2, 3] == 42.0

    # Reshape zero-copy
    T_reshaped = reshape(T, (12, 5))
    @test size(T_reshaped) == (12, 5)
    @test T_reshaped[4, 3] == 42.0

    # Hadamard product
    A = Tensor([1.0 2.0; 3.0 4.0])
    B = Tensor([2.0 0.5; 1.0 2.0])
    C = hadamard(A, B)
    @test to_array(C) == [2.0 1.0; 3.0 8.0]

    # Contraction
    # A: (2, 3), B: (3, 4) -> C: (2, 4)
    A_mat = Tensor([1.0 2.0 3.0; 4.0 5.0 6.0])
    B_mat = Tensor([1.0 0.0 1.0 0.0; 0.0 1.0 0.0 1.0; 1.0 1.0 0.0 0.0])
    C_mat = contract(A_mat, B_mat, (2,), (1,))
    @test size(C_mat) == (2, 4)
    @test to_array(C_mat) ≈ to_array(A_mat) * to_array(B_mat)


    # Frobenius norm
    @test frobenius_norm(A) ≈ sqrt(1.0 + 4.0 + 9.0 + 16.0)
end
