struct Chol{T<:Base.IEEEFloat} <: AbstractWhiteningTransform{T}
    μ::Vector{T}
    Σ::Matrix{T}
    F::Cholesky{T,Matrix{T}}
    W::Matrix{T}
    W⁻¹::Matrix{T}
    negWμ::Vector{T}
    function Chol{T}(μ::Vector{T}, Σ::Matrix{T}) where {T<:Base.IEEEFloat}
        if size(μ, 1) == size(Σ, 1) == size(Σ, 2)
            F = cholesky(Σ)
            W = F.U
            W⁻¹ = inv(F.U)
            new{T}(μ, Σ, F, W, W⁻¹, -(W * μ))
        else
            error("μ must be ℝⁿ and Σ must be ℝⁿˣⁿ")
        end
    end
end

# Chol(μ::Vector{T}, Σ::Matrix{T}) where {T<:Base.IEEEFloat} = Chol{T}(μ, Σ)
Chol(μ::AbstractVector{T}, Σ::AbstractMatrix{T}) where {T<:Base.IEEEFloat} =
    Chol{T}(collect(μ), collect(Σ))
