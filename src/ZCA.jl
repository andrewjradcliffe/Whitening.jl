"""
    ZCA{T<:Base.IEEEFloat} <: AbstractWhiteningTransform{T}

Zero-phase component analysis (ZCA) whitening transform.
"""
struct ZCA{T<:Base.IEEEFloat} <: AbstractWhiteningTransform{T}
    μ::Vector{T}
    Σ::Matrix{T}
    F::Eigen{T,T,Matrix{T},Vector{T}}
    W::Matrix{T}
    W⁻¹::Matrix{T}
    negWμ::Vector{T}
    function ZCA{T}(μ::Vector{T}, Σ::Matrix{T}) where {T<:Base.IEEEFloat}
        checkargs(μ, Σ)
        F = eigen(Σ, sortby = -)
        Λ¹² = sqrt.(F.values)
        Λ⁻¹² = inv.(Λ⁻¹²)
        B¹² = Diagonal(Λ¹²)
        B⁻¹² = Diagonal(Λ⁻¹²)
        W = F.vectors * B⁻¹² * F.vectors'
        W⁻¹ = F.vectors * B¹² * F.vectors'
        new{T}(μ, Σ, F, W, W⁻¹, BLAS.gemv('N', -one(T), W, μ))
    end
end

"""
    ZCA(μ::AbstractVector{T}, Σ::AbstractMatrix{T}) where {T<:Base.IEEEFloat}

Construct a ZCA transformer from the from the mean vector, `μ` ∈ ℝⁿ,
and a covariance matrix, `Σ` ∈ ℝⁿˣⁿ; `Σ` must be symmetric and
positive definite.
"""
ZCA(μ::AbstractVector{T}, Σ::AbstractMatrix{T}) where {T<:Base.IEEEFloat} =
    ZCA{T}(collect(μ), collect(Σ))

"""
    ZCA(X::AbstractMatrix{T}) where {T<:Base.IEEEFloat}

Construct a ZCA transformer from the from the `q × n` matrix,
each row of which is a sample of an `n`-dimensional random variable.

In order for the resultant covariance matrix to be positive definite,
`q` must be ≥ `n` and none of the variances may be zero.
"""
function ZCA(X::AbstractMatrix{T}) where {T<:Base.IEEEFloat}
    μ, Σ = _estimate(X)
    ZCA{T}(μ, Σ)
end
