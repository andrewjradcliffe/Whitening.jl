struct ZCAcor{T<:Base.IEEEFloat} <: AbstractWhiteningTransform{T}
    μ::Vector{T}
    Σ::Matrix{T}
    F::Eigen{T,T,Matrix{T},Vector{T}}
    W::Matrix{T}
    W⁻¹::Matrix{T}
    negWμ::Vector{T}
    function ZCAcor{T}(μ::Vector{T}, Σ::Matrix{T}) where {T<:Base.IEEEFloat}
        if size(μ, 1) == size(Σ, 1) == size(Σ, 2)
            v = diag(Σ)
            V⁻¹² = Diagonal(inv.(sqrt.(v)))
            P = V⁻¹² * Σ * V⁻¹²
            F = eigen(P, sortby = -)
            B¹² = Diagonal(sqrt.(F.values))
            B⁻¹² = Diagonal(inv.(sqrt.(F.values)))
            W = F.vectors * B⁻¹² * F.vectors' * V⁻¹²
            W⁻¹ = Diagonal(sqrt.(v)) * F.vectors * B¹² * F.vectors'
            new{T}(μ, Σ, F, W, W⁻¹, -(W * μ))
        else
            error("μ must be ℝⁿ and Σ must be ℝⁿˣⁿ")
        end
    end
end

# ZCAcor(μ::Vector{T}, Σ::Matrix{T}) where {T<:Base.IEEEFloat} = ZCAcor{T}(μ, Σ)
ZCAcor(μ::AbstractVector{T}, Σ::AbstractMatrix{T}) where {T<:Base.IEEEFloat} =
    ZCAcor{T}(collect(μ), collect(Σ))
