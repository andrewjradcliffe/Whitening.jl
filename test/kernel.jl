@testset "round-tripping; kernel: $k, $T" for k in (
        PCA,
        PCAcor,
        ZCA,
        ZCAcor,
        Chol,
        GeneralizedPCA,
        GeneralizedPCAcor,
    ),
    T in (Float16, Float32, Float64)

    rng = Xoshiro(0xc0ffee_cafe_beef12)
    ρ₁₂ = ρ₂₁ = T(0.7)
    ρ₁₃ = ρ₃₁ = T(0.5)
    ρ₂₃ = ρ₃₂ = T(0.3)
    σ₁ = T(1.0)
    σ₂ = T(0.5)
    σ₃ = T(2.0)
    μ = T[3, 2, 1]
    Σ = T[
        T(1.0)*abs2(σ₁) ρ₁₂*σ₁*σ₂ ρ₁₃*σ₁*σ₃
        ρ₂₁*σ₂*σ₁ T(1.0)*abs2(σ₂) ρ₂₃*σ₂*σ₃
        ρ₃₁*σ₃*σ₁ ρ₃₂*σ₃*σ₂ T(1.0)*abs2(σ₃)
    ]
    kern = @inferred k(μ, Σ)
    z = randn(rng, T, 3)
    x = @inferred unwhiten(kern, z)
    z_rt = @inferred whiten(kern, x)
    @test all(z .≈ z_rt)

    Z = randn(rng, T, 100, 3)
    X = @inferred unwhiten(kern, Z)
    Z_rt = @inferred whiten(kern, X)
    @test all(eachrow(Z) .≈ eachrow(Z_rt))

    @test_throws DimensionMismatch unwhiten(kern, Z')
    @test_throws DimensionMismatch whiten(kern, X')
end

@testset "mahalanobis: $T" for T in (Float16, Float32, Float64)
    ρ₁₂ = ρ₂₁ = T(0.7)
    ρ₁₃ = ρ₃₁ = T(0.5)
    ρ₂₃ = ρ₃₂ = T(0.3)
    σ₁ = T(1.0)
    σ₂ = T(0.5)
    σ₃ = T(2.0)
    μ = T[3, 2, 1]
    Σ = T[
        T(1.0)*abs2(σ₁) ρ₁₂*σ₁*σ₂ ρ₁₃*σ₁*σ₃
        ρ₂₁*σ₂*σ₁ T(1.0)*abs2(σ₂) ρ₂₃*σ₂*σ₃
        ρ₃₁*σ₃*σ₁ ρ₃₂*σ₃*σ₂ T(1.0)*abs2(σ₃)
    ]
    pca = PCA(μ, Σ)
    pcacor = PCAcor(μ, Σ)
    zca = ZCA(μ, Σ)
    zcacor = ZCAcor(μ, Σ)
    chol = Chol(μ, Σ)
    gpca = GeneralizedPCA(μ, Σ)
    gpcacor = GeneralizedPCAcor(μ, Σ)

    x = T[1, 2, 3]
    @test begin
        mahalanobis(pca, x) ≈
        mahalanobis(pcacor, x) ≈
        mahalanobis(zca, x) ≈
        mahalanobis(zcacor, x) ≈
        mahalanobis(chol, x) ≈
        mahalanobis(gpca, x) ≈
        mahalanobis(gpcacor, x)
    end

    for f in (*, +, /, -, ^, %)
        X = [T(f(i, j)) for i = 1:5, j = 1:3]
        @test begin
            mahalanobis(pca, X) ≈
            mahalanobis(pcacor, X) ≈
            mahalanobis(zca, X) ≈
            mahalanobis(zcacor, X) ≈
            mahalanobis(chol, X) ≈
            mahalanobis(gpca, X) ≈
            mahalanobis(gpcacor, X)
        end

    end
    X = [T(i * j) for i = 1:5, j = 1:3]
    for t in (pca, pcacor, zca, zcacor, gpca, gpcacor, chol)
        @test_throws DimensionMismatch mahalanobis(t, X')
        @inferred mahalanobis(t, X)
        @inferred mahalanobis(t, x)
    end
end

@testset "constructor: from matrix, $k, $T" for k in (
        PCA,
        PCAcor,
        ZCA,
        ZCAcor,
        Chol,
        GeneralizedPCA,
        GeneralizedPCAcor,
    ),
    T in (Float16, Float32, Float64)

    X = randn(T, 100, 10) .* randexp(T, 100, 10)
    t = k(X)
    μ, Σ = _estimate2(X)
    @test t.μ ≈ μ
    @test t.Σ ≈ Σ
end

@testset "dimensionality reduction, 1" begin
    rng = Xoshiro(0xc0ffee_cafe_beef12)
    n = 4
    m = 100
    X = randn(rng, m, n) .* randexp(rng, m, n)
    X = [X 0.1X[:, 2] + 0.2X[:, 4]]

    gpca = GeneralizedPCA(X, num_components = 4)
    @test size(gpca.W) == (4, 5)
    @test size(gpca.W⁻¹) == (5, 4)

    Z = whiten(gpca, X)
    X_rt = unwhiten(gpca, Z)
    @test all(eachrow(X_rt) .≈ eachrow(X))
end

rvs(rng, m, n) = randn(rng, m, n) .* randexp(rng, m, n)
function randmat_1(rng::AbstractRNG, m::Int, n::Int)
    [rvs(rng, m, n ÷ 2) ones(m) rvs(rng, m, n ÷ 2) zeros(m) rvs(m, 1)]
end
function randmat_2(rng::AbstractRNG, m::Int, n::Int)
    X = randn(rng, m, n) .* randexp(rng, m, n)
    X = [X 0.1X[:, 2] + 0.2X[:, 4]]
end

@testset "dimensionality reduction, 2" begin
    rng = Xoshiro(0xc0ffee_cafe_beef12)
    rvs(m, n) = randn(rng, m, n) .* randexp(rng, m, n)
    for n in (4, 8, 16)
        p = n + 1
        m = 100
        X = [rvs(m, n ÷ 2) ones(m) rvs(m, n ÷ 2) zeros(m) rvs(m, 1)]

        gpca = GeneralizedPCA(X)
        @test size(gpca.W) == (p, p + 2)
        @test size(gpca.W⁻¹) == (p + 2, p)

        Z = whiten(gpca, X)
        X_rt = unwhiten(gpca, Z)
        @test all(eachrow(X_rt) .≈ eachrow(X))

        gpca = GeneralizedPCA(X, num_components = 2)
        @test size(gpca.W) == (2, p + 2)
        @test size(gpca.W⁻¹) == (p + 2, 2)
    end
end
