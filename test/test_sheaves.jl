using Test
using Spectra
using LinearAlgebra

@testset "Cellular Sheaves & Cohomology" begin
    # Triangle complex: 3 vertices, 3 edges, 1 face
    complex = CellComplex(3)
    add_edge!(complex, 1, 2)
    add_edge!(complex, 2, 3)
    add_edge!(complex, 1, 3)
    add_face!(complex, 1, 2, 3)


    @test num_vertices(complex) == 3
    @test num_edges(complex) == 3
    @test num_faces(complex) == 1

    # Boundary chain property: d1 * d2 = 0
    @test verify_boundary_property(complex)

    # Betti numbers for filled triangle: b0 = 1, b1 = 0, b2 = 0
    bettis = betti_numbers(complex)
    @test bettis[1] == 1
    @test bettis[2] == 0

    # Cellular Sheaf with stalk dimension 2
    sheaf = CellularSheaf(complex, 2)
    @test total_cochain_dim_0(sheaf) == 6
    @test total_cochain_dim_1(sheaf) == 6

    # Trivial sheaf
    t_sheaf = trivial_sheaf(complex)
    dim_H0 = sheaf_cohomology_dim(t_sheaf)
    @test dim_H0 == 1

    # Sheaf Laplacian
    slap = SheafLaplacian(t_sheaf)
    L0 = slap.dense_matrix
    @test size(L0) == (3, 3)
    @test isapprox(L0, L0'; atol=1e-12)


    # Sheaf diffusion
    x0 = randn(3)
    x_diff = sheaf_diffusion_step(t_sheaf, x0, 0.05)
    @test length(x_diff) == 3


    # Neural Sheaf Layer
    nsl = NeuralSheafLayer(4, 2, 2)
    node_feat = randn(3, 4)
    y_feat = forward_sheaf_layer(nsl, complex, node_feat)
    @test size(y_feat, 1) == 3


    # Vietoris-Rips Persistent Homology
    pts = [
        0.0 0.0;
        1.0 0.0;
        0.5 (sqrt(3.0)/2.0)
    ]
    diag = compute_persistence_diagram(pts; max_radius=2.0)
    @test length(diag) >= 3
end

