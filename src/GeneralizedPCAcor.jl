@doc raw"""
    GeneralizedPCAcor{T<:Base.IEEEFloat} <: AbstractWhiteningTransform{T}

Scale-invariant principal component analysis (PCAcor) whitening transform, generalized to
support compression based on either
1. a pre-determined number of components,
2. a fraction of the total squared cross-correlation, or
3. a relative tolerance on the number of eigenvalues greater
   than `rtol*λ₁` where `λ₁` is the largest eigenvalue of the correlation matrix.

Given the eigendecomposition of the ``n × n`` correlation matrix,
``P = GΘGᵀ``, with eigenvalues sorted in descending order, i.e.
``θ₁ ≥ θ₂ ⋯ ≥ θₙ``, the first ``m`` components are selected according
to one or more of the criteria listed above.

If ``m = n``, then we have the canonical PCA-cor whitening matrix,
 ``W = Θ^{-\frac{1}{2}}GᵀV^{-\frac{1}{2}}``. Otherwise, for ``m < n``,
a map from ``ℝⁿ ↦ ℝᵐ`` is formed by removing the ``n - m`` rows from
``W``, i.e. the components with the ``n - m`` smallest eigenvalues are
removed. This is equivalent to selecting the ``m × m`` matrix
from the upper left of ``Θ`` and the ``m × n`` matrix from the
top of ``Gᵀ``. The inverse transform is then formed by
selecting the ``n × m`` matrix from the left of ``G`` and the same matrix
from ``Θ``.
"""
struct GeneralizedPCAcor{T<:Base.IEEEFloat} <: AbstractWhiteningTransform{T}
    μ::Vector{T}
    Σ::Matrix{T}
    F::Eigen{T,T,Matrix{T},Vector{T}}
    W::Matrix{T}
    W⁻¹::Matrix{T}
    negWμ::Vector{T}
    function GeneralizedPCAcor{T}(
        μ::Vector{T},
        Σ::Matrix{T};
        num_components::Union{Int,Nothing},
        vmin::Union{T,Nothing},
        rtol::Union{T,Nothing},
    ) where {T<:Base.IEEEFloat}
        checkargs(μ, Σ, num_components, vmin)
        v¹² = sqrt.(diag(Σ))
        V⁻¹² = Diagonal(inv.(v¹²))
        V¹² = Diagonal(v¹²)
        P = V⁻¹² * Σ * V⁻¹²
        F = eigen(P, sortby = -)
        n⃰ = determine_nstar(F.values, num_components, vmin, rtol)
        Λ¹² = sqrt.(@view(F.values[1:n⃰]))
        Λ⁻¹² = inv.(Λ¹²)
        B¹² = Diagonal(Λ¹²)
        B⁻¹² = Diagonal(Λ⁻¹²)
        U = @view(F.vectors[:, 1:n⃰])
        W = B⁻¹² * U' * V⁻¹²
        W⁻¹ = V¹² * U * B¹²
        new{T}(μ, Σ, F, W, W⁻¹, _loc_by_gemv(W, μ))
    end
end

"""
    GeneralizedPCAcor(μ::AbstractVector{T}, Σ::AbstractMatrix{T};
                      num_components::Union{Int, Nothing}=nothing,
                      vmin::Union{T, Nothing}=nothing,
                      rtol::Union{T, Nothing}=nothing) where {T<:Base.IEEEFloat}

Construct a generalized PCAcor transformer from the mean vector, `μ` ∈ ℝⁿ,
and a covariance matrix, `Σ` ∈ ℝⁿˣⁿ; `Σ` must be symmetric and
positive semi-definite.

The decomposition, ``Σ = V^{\frac{1}{2}} * P * V^{\frac{1}{2}}``,
where ``V`` is the diagonal matrix of variances and ``P`` is a
correlation matrix, must be well-formed
in order to obtain a meaningful result. That is, if the diagonal of `Σ`
contains 1 or more zero elements, then it is not possible to compute
``P = V^{-\frac{1}{2}} * Σ * V^{-\frac{1}{2}}``.


The output dimension, `m`, of the transformer is determined from
the optional arguments, where
1.  0 ≤ `num_components` ≤ n is a pre-determined size
2.  0 ≤ `vmin` ≤ 1 is the fraction of the total squared cross-covariance,
    hence, `m` is the smallest value such that `sum(λ[1:m]) ≥ vmin*sum(λ)`,
    where ``θᵢ, i=1,…,n`` are the eigenvalues of ``P`` in descending order.
3. `rtol` is the relative tolerance on the number of eigenvalues greater
   than `rtol*θ₁` where `θ₁` is the largest eigenvalue of ``P``.

If none of the 3 options are provided, the default is `rtol = n*eps(T)`.
If 2 or more options are provided, the minimum of the resultant sizes will
be chosen.
"""
function GeneralizedPCAcor(
    μ::AbstractVector{T},
    Σ::AbstractMatrix{T};
    num_components::Union{Int,Nothing} = nothing,
    vmin::Union{T,Nothing} = nothing,
    rtol::Union{T,Nothing} = nothing,
) where {T<:Base.IEEEFloat}
    GeneralizedPCAcor{T}(
        collect(μ),
        collect(Σ),
        num_components = num_components,
        vmin = vmin,
        rtol = rtol,
    )
end

"""
    GeneralizedPCAcor(X::AbstractMatrix{T};
                      num_components::Union{Int, Nothing}=nothing,
                      vmin::Union{T, Nothing}=nothing,
                      rtol::Union{T, Nothing}=nothing) where {T<:Base.IEEEFloat}

Construct a generalized PCAcor transformer from the `q × n` matrix,
each row of which is a sample of an `n`-dimensional random variable.
"""
function GeneralizedPCAcor(
    X::AbstractMatrix{T};
    num_components::Union{Int,Nothing} = nothing,
    vmin::Union{T,Nothing} = nothing,
    rtol::Union{T,Nothing} = nothing,
) where {T<:Base.IEEEFloat}
    μ, Σ = _estimate(X)
    GeneralizedPCAcor{T}(μ, Σ, num_components = num_components, vmin = vmin, rtol = rtol)
end
