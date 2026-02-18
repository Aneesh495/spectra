using Test
using Spectra
using LinearAlgebra

@testset "Sparse Matrix Formats & SpMV" begin
    # COO building
    coo = COOMatrix(5, 5)
    add_entry!(coo, 1, 1, 4.0)
    add_entry!(coo, 1, 2, -1.0)
    add_entry!(coo, 2, 1, -1.0)
    add_entry!(coo, 2, 2, 4.0)
    add_entry!(coo, 3, 3, 3.0)
    add_entry!(coo, 4, 4, 5.0)
    add_entry!(coo, 5, 5, 2.0)

    # Convert to CSC
    csc = from_coo(coo)
    @test csc.m == 5 && csc.n == 5
    @test nnz(csc) == 7

    # Dense equivalence
    dense = to_dense(csc)
    @test dense[1, 1] == 4.0
    @test dense[1, 2] == -1.0
    @test dense[2, 1] == -1.0

    # SpMV
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = zeros(5)
    spmv!(y, csc, x)
    @test y ≈ dense * x

    # Sparse CG
    b = [2.0, 7.0, 9.0, 20.0, 10.0]
    sol_cg, conv, _ = sparse_cg(csc, b; tol=1e-10)
    @test conv
    @test dense * sol_cg ≈ b atol=1e-8
end
