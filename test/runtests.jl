using Whitening, LinearAlgebra, Test, Random, Statistics
# public
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
# private
using Whitening:
    ispossemidef,
    checkargs,
    determine_nstar,
    findlastcomponent,
    findlastrank,
    _cov_by_gemm,
    _loc_by_gemv,
    _estimate

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
