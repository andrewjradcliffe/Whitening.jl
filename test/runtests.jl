using Whitening, LinearAlgebra, Test, Random, Statistics
using Whitening:
    PCA,
    PCAcor,
    ZCA,
    ZCAcor,
    Chol,
    GeneralizedPCA,
    GeneralizedPCAcor,
    whiten,
    unwhiten,
    mahalanobis

const TESTS = ["kernel", "common"]

# Utility functions
function _estimate2(X::AbstractMatrix{T}) where {T<:Base.IEEEFloat}
    μ = dropdims(mean(X, dims = 1), dims = 1)
    Σ = cov(X, dims = 1, corrected = false)
    μ, Σ
end

for t in TESTS
    @testset "$t" begin
        include("$t.jl")
    end
end
