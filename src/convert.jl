import Base: convert


function convert(::Type{ZCA{T}}, x::PCA{T}) where {T<:Base.IEEEFloat}
    μ = copy(x.μ)
    Σ = copy(x.Σ)
    W = x.F.vectors * x.W
    W⁻¹ = x.W⁻¹ * x.F.vectors'
    negWμ = _loc_by_gemv(W, x.μ)
    F = copy(x.F)
    ZCA{T}(μ, Σ, F, W, W⁻¹, negWμ)
end

function convert(::Type{PCA{T}}, x::ZCA{T}) where {T<:Base.IEEEFloat}
    μ = copy(x.μ)
    Σ = copy(x.Σ)
    F = copy(x.F)
    Λ¹² = sqrt.(F.values)
    Λ⁻¹² = inv.(Λ¹²)
    B¹² = Diagonal(Λ¹²)
    B⁻¹² = Diagonal(Λ⁻¹²)
    W = B⁻¹² * F.vectors'
    W⁻¹ = F.vectors * B¹²
    negWμ = _loc_by_gemv(W, μ)
    PCA{T}(μ, Σ, F, W, W⁻¹, negWμ)
end
