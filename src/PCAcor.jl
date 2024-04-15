"""
    PCAcor{T<:Base.IEEEFloat} <: AbstractWhiteningTransform{T}

Scale-invariant principal component analysis (PCA-cor) whitening transform.
"""
struct PCAcor{T<:Base.IEEEFloat} <: AbstractWhiteningTransform{T}
    μ::Vector{T}
    Σ::Matrix{T}
    F::Eigen{T,T,Matrix{T},Vector{T}}
    W::Matrix{T}
    W⁻¹::Matrix{T}
    negWμ::Vector{T}
    function PCAcor{T}(μ::Vector{T}, Σ::Matrix{T}) where {T<:Base.IEEEFloat}
        if size(μ, 1) == size(Σ, 1) == size(Σ, 2)
            v = diag(Σ)
            V⁻¹² = Diagonal(inv.(sqrt.(v)))
            P = V⁻¹² * Σ * V⁻¹²
            F = eigen(P, sortby = -)
            B¹² = Diagonal(sqrt.(F.values))
            B⁻¹² = Diagonal(inv.(sqrt.(F.values)))
            W = B⁻¹² * F.vectors' * V⁻¹²
            W⁻¹ = Diagonal(sqrt.(v)) * F.vectors * B¹²
            new{T}(μ, Σ, F, W, W⁻¹, -(W * μ))
        else
            error("μ must be ℝⁿ and Σ must be ℝⁿˣⁿ")
        end
    end
end

# PCAcor(μ::Vector{T}, Σ::Matrix{T}) where {T<:Base.IEEEFloat} = PCAcor{T}(μ, Σ)
PCAcor(μ::AbstractVector{T}, Σ::AbstractMatrix{T}) where {T<:Base.IEEEFloat} =
    PCAcor{T}(collect(μ), collect(Σ))
