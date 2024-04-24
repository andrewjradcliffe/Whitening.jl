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

for t in TESTS
    @testset "$t" begin
        include("$t.jl")
    end
end
