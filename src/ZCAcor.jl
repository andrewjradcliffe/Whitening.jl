"""
    ZCAcor{T<:Base.IEEEFloat} <: AbstractWhiteningTransform{T}

Scale-invariant zero-phase component analysis (ZCA-cor) whitening transform.
"""
struct ZCAcor{T<:Base.IEEEFloat} <: AbstractWhiteningTransform{T}
    μ::Vector{T}
    Σ::Matrix{T}
    F::Eigen{T,T,Matrix{T},Vector{T}}
    W::Matrix{T}
    W⁻¹::Matrix{T}
    negWμ::Vector{T}
    function ZCAcor{T}(μ::Vector{T}, Σ::Matrix{T}) where {T<:Base.IEEEFloat}
        checkargs(μ, Σ)
        v¹² = sqrt.(diag(Σ))
        V⁻¹² = Diagonal(inv.(v¹²))
        V¹² = Diagonal(v¹²)
        P = V⁻¹² * Σ * V⁻¹²
        F = eigen(P, sortby = -)
        Λ¹² = sqrt.(F.values)
        Λ⁻¹² = inv.(Λ¹²)
        B¹² = Diagonal(Λ¹²)
        B⁻¹² = Diagonal(Λ⁻¹²)
        W = F.vectors * B⁻¹² * F.vectors' * V⁻¹²
        W⁻¹ = V¹² * F.vectors * B¹² * F.vectors'
        new{T}(μ, Σ, F, W, W⁻¹, _loc_by_gemv(W, μ))
    end
end

"""
    ZCAcor(μ::AbstractVector{T}, Σ::AbstractMatrix{T}) where {T<:Base.IEEEFloat}

Construct a ZCAcor transformer from the from the mean vector, `μ` ∈ ℝⁿ,
and a covariance matrix, `Σ` ∈ ℝⁿˣⁿ; `Σ` must be symmetric and
positive definite.
"""
ZCAcor(μ::AbstractVector{T}, Σ::AbstractMatrix{T}) where {T<:Base.IEEEFloat} =
    ZCAcor{T}(collect(μ), collect(Σ))

"""
    ZCAcor(X::AbstractMatrix{T}) where {T<:Base.IEEEFloat}

Construct a ZCAcor transformer from the from the `q × n` matrix,
each row of which is a sample of an `n`-dimensional random variable.

In order for the resultant covariance matrix to be positive definite,
`q` must be ≥ `n` and none of the variances may be zero.
"""
function ZCAcor(X::AbstractMatrix{T}) where {T<:Base.IEEEFloat}
    μ, Σ = _estimate(X)
    ZCAcor{T}(μ, Σ)
end
