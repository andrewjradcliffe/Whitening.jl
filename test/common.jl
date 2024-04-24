@testset "positive semi-definite predicate" begin
    @test ispossemidef([1 0 0; 1 -2eps() 1; 0 0 1])
    @test !ispossemidef([1 0 0; 1 -4eps() 1; 0 0 1])

    A = randn(4, 4)
    @test ispossemidef(A'A)
end

@testset "checkargs" begin
    A = Float64[1 2 3; 4 5 6; 7 8 9]
    μ = Float64[1, 2, 3]

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

    @testset "regular" begin
        @test_throws "must be ℝⁿ" checkargs(μ[1:2], A[:, 1:end-1])
        @test_throws "must be symmetric" checkargs(μ, A)
        @test_throws "must be positive definite" checkargs(μ, A'A)
        @test checkargs(μ, Σ) === nothing
    end

    @testset "generalized" begin
        for arg1 in (nothing, 3), arg2 in (nothing, 0.5)
            @test_throws "must be ℝⁿ" checkargs(μ[1:2], A[:, 1:end-1], arg1, arg2)
            @test_throws "must be symmetric" checkargs(μ, A, arg1, arg2)
            @test_throws "must be positive semi-definite" checkargs(μ, A'A, arg1, arg2)
        end

        @test_throws "vmin must be" checkargs(μ, Σ, nothing, 5.0)
        @test_throws "num_components must be" checkargs(μ, Σ, 10, nothing)
        @test_throws "vmin must be" checkargs(μ, Σ, 10, 5.0)
    end
end

@testset "determine_nstar" begin
    Λ = [
        4.046432330842713,
        3.242985819082353,
        2.8089891675526952,
        2.403091160690958,
        2.135184130847648,
        1.7825546604005504,
        1.566948870122324,
        1.265926795459427,
        0.7913620463499906,
        -4.008388165498776e-17,
        -4.4036492387932593e-16,
    ]

    rtol = 11 * 2 * eps()
    vmin = 0.7
    num_components = 5
    @test determine_nstar(Λ, num_components, vmin, rtol) == 5
    @test determine_nstar(Λ, num_components, vmin, nothing) == 5
    @test determine_nstar(Λ, num_components, nothing, rtol) == 5
    @test determine_nstar(Λ, nothing, vmin, rtol) == 6
    @test determine_nstar(Λ, num_components, nothing, nothing) == num_components
    @test determine_nstar(Λ, nothing, vmin, nothing) == 6
    @test determine_nstar(Λ, nothing, nothing, rtol) == 9
    @test determine_nstar(Λ, nothing, nothing, nothing) == 9

    @test determine_nstar(Float64[], nothing, nothing, nothing) == 0

    # pathological case, but function not public
    @test determine_nstar(Float64[], num_components, nothing, nothing) == num_components
end

@testset "findlastcomponent" begin
    Λ = [
        4.046432330842713,
        3.242985819082353,
        2.8089891675526952,
        2.403091160690958,
        2.135184130847648,
        1.7825546604005504,
        1.566948870122324,
        1.265926795459427,
        0.7913620463499906,
        -4.008388165498776e-17,
        -4.4036492387932593e-16,
    ]

    @test findlastcomponent(0.0, Λ) == 0
    @test findlastcomponent(1.0, Λ) == 11
    for (rtol, rhs) in zip(
        (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99),
        (2, 2, 3, 4, 4, 5, 6, 7, 9, 9, 10),
    )
        @test findlastcomponent(rtol, Λ) == rhs
    end

    @test findlastcomponent(0.5, Float64[]) == 0
end

@testset "findlastrank" begin
    Λ = [
        4.046432330842713,
        3.242985819082353,
        2.8089891675526952,
        2.403091160690958,
        2.135184130847648,
        1.7825546604005504,
        1.566948870122324,
        1.265926795459427,
        0.7913620463499906,
        -4.008388165498776e-17,
        -4.4036492387932593e-16,
    ]

    @test findlastrank(0.0, Λ) == 9
    @test findlastrank(-eps(), Λ) == 11
    for (rtol, rhs) in zip(
        (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        (9, 8, 8, 6, 5, 3, 2, 2, 1, 0),
    )
        @test findlastrank(rtol, Λ) == rhs
    end

    @test findlastrank(eps(), Float64[]) == 0
end

function _estimate2(X::AbstractMatrix{T}) where {T<:Base.IEEEFloat}
    μ = dropdims(mean(X, dims = 1), dims = 1)
    Σ = cov(X, dims = 1, corrected = false)
    μ, Σ
end

@testset "_estimate" for T in (Float16, Float32, Float64)
    for p = 2:10
        n = 1 << p
        for m in (n >> 1, n, n << 1)
            rng = Xoshiro(0xc0ffee_cafe_beef12)
            X = randn(rng, T, m, n) .* randexp(rng, T, m, n)
            μ, Σ = _estimate(X)
            μ′, Σ′ = _estimate2(X)
            @test μ ≈ μ′
            @test Σ ≈ Σ′
        end
    end
end
