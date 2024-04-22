"""
    GeneralizedPCA{T<:Base.IEEEFloat} <: AbstractWhiteningTransform{T}

Principal component analysis (PCA) whitening transform, generalized to
support compression based on either
(1) a pre-determined number of components,
(2) a fraction of the total squared cross-covariance, or
(3) a relative tolerance on the number of eigenvalues greater than `rtol*λ₁`
where `λ₁` is the largest eigenvalue.
"""
struct GeneralizedPCA{T<:Base.IEEEFloat} <: AbstractWhiteningTransform{T}
    μ::Vector{T}
    Σ::Matrix{T}
    F::Eigen{T,T,Matrix{T},Vector{T}}
    W::Matrix{T}
    W⁻¹::Matrix{T}
    negWμ::Vector{T}
    function GeneralizedPCA{T}(
        μ::Vector{T},
        Σ::Matrix{T};
        num_components::Int = size(Σ, 1),
        ratio::T = one(T),
        rtol::T = size(Σ, 1) * eps(T),
    ) where {T<:Base.IEEEFloat}
        checkargs(μ, Σ, num_components, ratio)
        F = eigen(Σ, sortby = -)
        n⃰ = determine_nstar(F.values, num_components, ratio, rtol)
        Λ¹² = sqrt.(@view(F.values[1:n⃰]))
        Λ⁻¹² = inv.(Λ¹²)
        B¹² = Diagonal(Λ¹²)
        B⁻¹² = Diagonal(Λ⁻¹²)
        U = @view(F.vectors[:, 1:n⃰])
        W = B⁻¹² * U'
        W⁻¹ = U * B¹²
        new{T}(μ, Σ, F, W, W⁻¹, -(W * μ))
    end
end

function checkargs(μ::Vector{T}, Σ::Matrix{T}, num_components::Int, ratio::T) where {T}
    n = size(Σ, 1)
    if n == size(μ, 1) == size(Σ, 2)
        if zero(T) ≤ ratio ≤ one(T)
            if 0 ≤ num_components ≤ n
                nothing
            else
                error("num_components must be 0 ≤ x ≤ n")
            end
        else
            error("ratio must be 0 ≤ x ≤ 1")
        end
    else
        error("μ must be ℝⁿ and Σ must be ℝⁿˣⁿ")
    end
end

function determine_nstar(
    Λ::Vector{T},
    num_components::Int,
    ratio::T,
    rtol::T,
) where {T<:Base.IEEEFloat}
    n = length(Λ)
    if num_components < n
        num_components
    else
        min(findlastrank(rtol, Λ), findlastcomponent(ratio, Λ))
    end
end

function findlastcomponent(r::T, Λ::Vector{T}) where {T<:Base.IEEEFloat}
    n = length(Λ)
    if n == 0
        0
    elseif isone(r)
        n
    else
        c = sum(Λ)
        i = 1
        s = zero(T)
        t = c * r
        while s < t && i ≤ n
            s += Λ[i]
            i += 1
        end
        i
    end
end

function findlastrank(rtol::T, Λ::Vector{T}) where {T<:Base.IEEEFloat}
    n = length(Λ)
    if n == 0
        0
    else
        tol = rtol * Λ[1]
        i = findlast(>(tol), Λ)
        isnothing(i) ? 0 : i
    end
end


function GeneralizedPCA(
    μ::AbstractVector{T},
    Σ::AbstractMatrix{T};
    num_components::Int = size(Σ, 1),
    ratio::T = one(T),
    rtol::T = size(Σ, 1) * eps(T),
) where {T<:Base.IEEEFloat}
    GeneralizedPCA{T}(
        collect(μ),
        collect(Σ),
        num_components = num_components,
        ratio = ratio,
        rtol = rtol,
    )
end
