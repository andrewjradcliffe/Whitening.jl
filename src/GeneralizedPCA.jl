"""
    GeneralizedPCA{T<:Base.IEEEFloat} <: AbstractWhiteningTransform{T}

Principal component analysis (PCA) whitening transform, generalized to
support compression based on either
1. a pre-determined number of components,
2. a fraction of the total squared cross-covariance, or
3. a relative tolerance on the number of eigenvalues greater
   than `rtol*λ₁` where `λ₁` is the largest eigenvalue of the covariance matrix.
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
        num_components::Union{Int,Nothing},
        vmin::Union{T,Nothing},
        rtol::Union{T,Nothing},
    ) where {T<:Base.IEEEFloat}
        checkargs(μ, Σ, num_components, vmin)
        F = eigen(Σ, sortby = -)
        n⃰ = determine_nstar(F.values, num_components, vmin, rtol)
        Λ¹² = sqrt.(@view(F.values[1:n⃰]))
        Λ⁻¹² = inv.(Λ¹²)
        B¹² = Diagonal(Λ¹²)
        B⁻¹² = Diagonal(Λ⁻¹²)
        U = @view(F.vectors[:, 1:n⃰])
        W = B⁻¹² * U'
        W⁻¹ = U * B¹²
        new{T}(μ, Σ, F, W, W⁻¹, _loc_by_gemv(W, μ))
    end
end

"""
    GeneralizedPCA(μ::AbstractVector{T}, Σ::AbstractMatrix{T};
                   num_components::Union{Int, Nothing}=nothing,
                   vmin::Union{T, Nothing}=nothing,
                   rtol::Union{T, Nothing}=nothing) where {T<:Base.IEEEFloat}

Construct a generalized PCA transformer from the mean vector, `μ` ∈ ℝⁿ,
and a covariance matrix, `Σ` ∈ ℝⁿˣⁿ; `Σ` must be symmetric and
positive semi-definite.

The output dimension, `m`, of the transformer is determined from
the optional arguments, where
1.  0 ≤ `num_components` ≤ n is a pre-determined size
2.  0 ≤ `vmin` ≤ 1 is the fraction of the total squared cross-covariance,
    hence, `m` is the smallest value such that `sum(λ[1:m]) ≥ vmin*sum(λ)`,
    where `λᵢ, i=1,…,n` are the eigenvalues of `Σ` in descending order.
3. `rtol` is the relative tolerance on the number of eigenvalues greater
   than `rtol*λ₁` where `λ₁` is the largest eigenvalue of `Σ`.

If none of the 3 options are provided, the default is `rtol = n*eps(T)`.
If 2 or more options are provided, the minimum of the resultant sizes will
be chosen.
"""
function GeneralizedPCA(
    μ::AbstractVector{T},
    Σ::AbstractMatrix{T};
    num_components::Union{Int,Nothing} = nothing,
    vmin::Union{T,Nothing} = nothing,
    rtol::Union{T,Nothing} = nothing,
) where {T<:Base.IEEEFloat}
    GeneralizedPCA{T}(
        collect(μ),
        collect(Σ),
        num_components = num_components,
        vmin = vmin,
        rtol = rtol,
    )
end

"""
    GeneralizedPCA(X::AbstractMatrix{T};
                   num_components::Union{Int, Nothing}=nothing,
                   vmin::Union{T, Nothing}=nothing,
                   rtol::Union{T, Nothing}=nothing) where {T<:Base.IEEEFloat}

Construct a generalized PCA transformer from the `q × n` matrix,
each row of which is a sample of an `n`-dimensional random variable.
"""
function GeneralizedPCA(
    X::AbstractMatrix{T};
    num_components::Union{Int,Nothing} = nothing,
    vmin::Union{T,Nothing} = nothing,
    rtol::Union{T,Nothing} = nothing,
) where {T<:Base.IEEEFloat}
    μ, Σ = _estimate(X)
    GeneralizedPCA{T}(μ, Σ, num_components = num_components, vmin = vmin, rtol = rtol)
end
