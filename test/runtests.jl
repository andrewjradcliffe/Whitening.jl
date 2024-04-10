using Whitening, LinearAlgebra, Test, Random, Statistics

@testset "basic properties" begin
    rng = Xoshiro(0xbad_c0ffee_bad_cafe)
    μ = Float64[5, 6, 7]
    Σ = Float64[1 0 0; 0 2 0; 0 0 3]
    F = eigen(Σ)
    W⁻¹ = F.vectors * Diagonal(sqrt.(F.values))
    F = svd(Σ)
    W⁻¹ = F.U * Diagonal(sqrt.(F.S))

    Z = rand(rng, 3, 10)

    X = μ .+ W⁻¹ * Z
    # μ = mean(X, dims=2)
    # Σ = cov(X, dims=2, corrected=false)

    K = Whitening.Kernel(vec(μ), Σ)

    @testset "inner product equivalence under `eigen`" begin
        let X = μ .+ W⁻¹ * Z, old = eachcol(Z), new = whiten.(eachcol(X), Ref(K))
            @test all(old .⋅ old .≈ new .⋅ new)
        end
    end


    F = svd(Σ)
    W⁻¹ = F.U * Diagonal(sqrt.(F.S))

    @testset "element-wise equivalence under `svd`" begin
        let X = μ .+ W⁻¹ * Z, old = eachcol(Z), new = whiten.(eachcol(X), Ref(K))
            @test all(old .⋅ old .≈ new .⋅ new)
            @test all(Z .≈ stack(new, dims = 2))
        end
    end
end
