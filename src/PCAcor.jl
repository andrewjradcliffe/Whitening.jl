@doc raw"""
    PCAcor{T<:Base.IEEEFloat} <: AbstractWhiteningTransform{T}

Scale-invariant principal component analysis (PCA-cor) whitening transform.

Given the eigendecomposition of the correlation matrix,
``P = GΘGᵀ``, and the diagonal variance matrix, ``V``, we have
the whitening matrix, ``W = Θ^{-\frac{1}{2}}GᵀV^{-\frac{1}{2}}``.
"""
struct PCAcor{T<:Base.IEEEFloat} <: AbstractWhiteningTransform{T}
    μ::Vector{T}
    Σ::Matrix{T}
    F::Eigen{T,T,Matrix{T},Vector{T}}
    W::Matrix{T}
    W⁻¹::Matrix{T}
    negWμ::Vector{T}
    function PCAcor{T}(μ::Vector{T}, Σ::Matrix{T}) where {T<:Base.IEEEFloat}
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
        W = B⁻¹² * F.vectors' * V⁻¹²
        W⁻¹ = V¹² * F.vectors * B¹²
        new{T}(μ, Σ, F, W, W⁻¹, _loc_by_gemv(W, μ))
    end
end

"""
    PCAcor(μ::AbstractVector{T}, Σ::AbstractMatrix{T}) where {T<:Base.IEEEFloat}

Construct a PCAcor transformer from the from the mean vector, `μ` ∈ ℝⁿ,
and a covariance matrix, `Σ` ∈ ℝⁿˣⁿ; `Σ` must be symmetric and
positive definite.
"""
PCAcor(μ::AbstractVector{T}, Σ::AbstractMatrix{T}) where {T<:Base.IEEEFloat} =
    PCAcor{T}(collect(μ), collect(Σ))

"""
    PCAcor(X::AbstractMatrix{T}) where {T<:Base.IEEEFloat}

Construct a PCAcor transformer from the from the `q × n` matrix,
each row of which is a sample of an `n`-dimensional random variable.

In order for the resultant covariance matrix to be positive definite,
`q` must be ≥ `n` and none of the variances may be zero.
"""
function PCAcor(X::AbstractMatrix{T}) where {T<:Base.IEEEFloat}
    μ, Σ = _estimate(X)
    PCAcor{T}(μ, Σ)
end
