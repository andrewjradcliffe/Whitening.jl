struct ZCA{T<:Base.IEEEFloat} <: AbstractWhiteningTransform{T}
    μ::Vector{T}
    Σ::Matrix{T}
    F::Eigen{T,T,Matrix{T},Vector{T}}
    W::Matrix{T}
    W⁻¹::Matrix{T}
    negWμ::Vector{T}
    function ZCA{T}(μ::Vector{T}, Σ::Matrix{T}) where {T<:Base.IEEEFloat}
        if size(μ, 1) == size(Σ, 1) == size(Σ, 2)
            F = eigen(Σ, sortby = -)
            B¹² = Diagonal(sqrt.(F.values))
            B⁻¹² = Diagonal(inv.(sqrt.(F.values)))
            W = F.vectors * B⁻¹² * F.vectors'
            W⁻¹ = F.vectors * B¹² * F.vectors'
            new{T}(μ, Σ, F, W, W⁻¹, -(W * μ))
        else
            error("μ must be ℝⁿ and Σ must be ℝⁿˣⁿ")
        end
    end
end

# ZCA(μ::Vector{T}, Σ::Matrix{T}) where {T<:Base.IEEEFloat} = ZCA{T}(μ, Σ)
ZCA(μ::AbstractVector{T}, Σ::AbstractMatrix{T}) where {T<:Base.IEEEFloat} =
    ZCA{T}(collect(μ), collect(Σ))
