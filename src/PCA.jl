"""
    PCA{T<:Base.IEEEFloat} <: AbstractWhiteningTransform{T}

Principal component analysis (PCA) whitening transform.
"""
struct PCA{T<:Base.IEEEFloat} <: AbstractWhiteningTransform{T}
    μ::Vector{T}
    Σ::Matrix{T}
    F::Eigen{T,T,Matrix{T},Vector{T}}
    W::Matrix{T}
    W⁻¹::Matrix{T}
    negWμ::Vector{T}
    function PCA{T}(μ::Vector{T}, Σ::Matrix{T}) where {T<:Base.IEEEFloat}
        checkargs(μ, Σ)
        F = eigen(Σ, sortby = -)
        Λ¹² = sqrt.(F.values)
        Λ⁻¹² = inv.(Λ⁻¹²)
        B¹² = Diagonal(Λ¹²)
        B⁻¹² = Diagonal(Λ⁻¹²)
        W = B⁻¹² * F.vectors'
        W⁻¹ = F.vectors * B¹²
        new{T}(μ, Σ, F, W, W⁻¹, BLAS.gemv('N', -one(T), W, μ))
    end
end

"""
    PCA(μ::AbstractVector{T}, Σ::AbstractMatrix{T}) where {T<:Base.IEEEFloat}

Construct a PCA transformer from the from the mean vector, `μ` ∈ ℝⁿ,
and a covariance matrix, `Σ` ∈ ℝⁿˣⁿ; `Σ` must be symmetric and
positive definite.
"""
PCA(μ::AbstractVector{T}, Σ::AbstractMatrix{T}) where {T<:Base.IEEEFloat} =
    PCA{T}(collect(μ), collect(Σ))

"""
    PCA(X::AbstractMatrix{T}) where {T<:Base.IEEEFloat}

Construct a PCA transformer from the from the `q × n` matrix,
each row of which is a sample of an `n`-dimensional random variable.

In order for the resultant covariance matrix to be positive definite,
`q` must be ≥ `n` and none of the variances may be zero.
"""
function PCA(X::AbstractMatrix{T}) where {T<:Base.IEEEFloat}
    μ, Σ = _estimate(X)
    PCA{T}(μ, Σ)
end
