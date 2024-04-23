"""
    Chol{T<:Base.IEEEFloat} <: AbstractWhiteningTransform{T}

Cholesky whitening transform.
"""
struct Chol{T<:Base.IEEEFloat} <: AbstractWhiteningTransform{T}
    μ::Vector{T}
    Σ::Matrix{T}
    F::Cholesky{T,Matrix{T}}
    W::Matrix{T}
    W⁻¹::Matrix{T}
    negWμ::Vector{T}
    function Chol{T}(μ::Vector{T}, Σ::Matrix{T}) where {T<:Base.IEEEFloat}
        checkargs(μ, Σ)
        F = cholesky(Σ)
        W = F.U
        W⁻¹ = inv(F.U)
        new{T}(μ, Σ, F, W, W⁻¹, BLAS.gemv('N', -one(T), W, μ))
    end
end

Chol(μ::AbstractVector{T}, Σ::AbstractMatrix{T}) where {T<:Base.IEEEFloat} =
    Chol{T}(collect(μ), collect(Σ))

"""
    Chol(μ::AbstractVector{T}, Σ::AbstractMatrix{T}) where {T<:Base.IEEEFloat}

Construct a Cholesky transformer from the from the mean vector, `μ` ∈ ℝⁿ,
and a covariance matrix, `Σ` ∈ ℝⁿˣⁿ; `Σ` must be symmetric and
positive definite.
"""
Chol(μ::AbstractVector{T}, Σ::AbstractMatrix{T}) where {T<:Base.IEEEFloat} =
    Chol{T}(collect(μ), collect(Σ))

"""
    Chol(X::AbstractMatrix{T}) where {T<:Base.IEEEFloat}

Construct a Cholesky transformer from the from the `q × n` matrix,
each row of which is a sample of an `n`-dimensional random variable.

In order for the resultant covariance matrix to be positive definite,
`q` must be ≥ `n` and none of the variances may be zero.
"""
function Chol(X::AbstractMatrix{T}) where {T<:Base.IEEEFloat}
    μ, Σ = _estimate(X)
    Chol{T}(μ, Σ)
end
