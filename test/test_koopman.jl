using Test
using Spectra
using LinearAlgebra

@testset "Koopman Operator Theory & Transfer Operators" begin
    # Linear harmonic system: dx/dt = [0 1; -1 0] x
    dt = 0.05
    times = 0.0:dt:1.0
    N = length(times)
    X_data = zeros(2, N)
    for i in 1:N
        t = times[i]
        X_data[1, i] = cos(t)
        X_data[2, i] = -sin(t)
    end
    X = X_data[:, 1:(end - 1)]
    Y = X_data[:, 2:end]

    # Exact DMD
    dmd = exact_dmd(X, Y, dt; rank=2)
    @test dmd.rank == 2
    pred_0 = predict(dmd, 0.0)
    @test pred_0 ≈ [1.0, 0.0] atol=1e-4

    # Hankel DMD
    sig = [cos(0.05 * i) for i in 1:60]
    hdmd = hankel_dmd(sig, 8, 0.05; rank=4)
    @test hdmd.rank == 4

    # Streaming DMD
    sdmd = StreamingDMD(2)
    for i in 1:size(X, 2)
        update!(sdmd, X[:, i], Y[:, i])
    end
    A_op = get_operator(sdmd)
    @test size(A_op) == (2, 2)

    # EDMD with Polynomial Dictionary
    pdict = PolynomialDictionary(2, 2)
    edmd = extended_dmd(X, Y, pdict)
    x_next = predict_next(edmd, X[:, 1])
    @test x_next ≈ Y[:, 1] atol=1e-3

    # EDMD with RBF Dictionary
    centers = randn(2, 5)
    rbf_dict = RBFDictionary(centers; gamma=0.5)
    edmd_rbf = extended_dmd(X, Y, rbf_dict)
    @test size(edmd_rbf.K) == (6, 6)

    # Deep Koopman Autoencoder
    dk = DeepKoopmanNet(2, 4; hidden_dim=16)
    loss = compute_loss(dk, X, Y)
    @test isfinite(loss)
    fit_linear_operator!(dk, X, Y)
    traj = rollout(dk, [1.0, 0.0], 4)
    @test size(traj) == (5, 2)

    # Transfer Operators: 1D Ulam on Logistic Map
    part1d = BoxPartition1D(0.0, 1.0, 30)
    pf1d = ulam_perron_frobenius(x -> 4.0 * x * (1.0 - x), part1d; n_samples_per_box=200)
    @test isapprox(sum(pf1d.invariant_measure), 1.0; atol=1e-5)
    @test pf1d.spectral_gap > 0.0

    # Almost-invariant sets
    sA, sB = almost_invariant_sets(pf1d)
    @test length(sA) + length(sB) == 30
end
