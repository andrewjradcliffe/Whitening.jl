using Whitening, LinearAlgebra, Test, Random, Statistics

@testset "round-tripping; kernel: $k" for k in (Whitening.PCA, Whitening.PCAcor, Whitening.ZCA, Whitening.ZCAcor, Whitening.Chol)
    ρ₁₂ = ρ₂₁ = 0.7
    ρ₁₃ = ρ₃₁ = 0.5
    ρ₂₃ = ρ₃₂ = 0.3
    σ₁ = 1.0
    σ₂ = 0.5
    σ₃ = 2.0
    μ = Float64[3, 2, 1]
    Σ = [
        1.0*abs2(σ₁) ρ₁₂*σ₁*σ₂ ρ₁₃*σ₁*σ₃
        ρ₂₁*σ₂*σ₁ 1.0*abs2(σ₂) ρ₂₃*σ₂*σ₃
        ρ₃₁*σ₃*σ₁ ρ₃₂*σ₃*σ₂ 1.0*abs2(σ₃)
    ]
    kern = k(μ, Σ)
    z = randn(3)
    x = Whitening.unwhiten(kern, z)
    z_rt = Whitening.whiten(kern, x)
    @test all(z .≈ z_rt)

    Z = randn(3, 100)
    X = Whitening.unwhiten(kern, Z)
    Z_rt = Whitening.whiten(kern, X)
    @test all(Z .≈ Z_rt)
end
