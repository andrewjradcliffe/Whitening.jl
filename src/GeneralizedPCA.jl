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
        num_components::Union{Int, Nothing},
        vmin::Union{T, Nothing},
        rtol::Union{T, Nothing},
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
        new{T}(μ, Σ, F, W, W⁻¹, -(W * μ))
    end
end

function ispossemidef(A::AbstractMatrix{T}) where {T<:Base.IEEEFloat}
    Λ = eigvals(A)
    isreal(Λ) && all(≥(-eps(T)), Λ)
end

function checkargs(μ::Vector{T}, Σ::Matrix{T}, num_components::Union{Int, Nothing},
                   vmin::Union{T, Nothing}) where {T}
    n = size(Σ, 1)
    if n == size(μ, 1) == size(Σ, 2)
        if issymmetric(Σ)
            if ispossemidef(Σ)
                if isnothing(vmin) || zero(T) ≤ vmin ≤ one(T)
                    if isnothing(num_components) || 0 ≤ num_components ≤ n
                        nothing
                    else
                        error("num_components must be 0 ≤ x ≤ n")
                    end
                else
                    error("vmin must be 0 ≤ x ≤ 1")
                end
            else
                error("Σ must be positive semi-definite")
            end
        else
            error("Σ must be symmetric")
        end
    else
        error("μ must be ℝⁿ and Σ must be ℝⁿˣⁿ")
    end
end

function determine_nstar(
    Λ::Vector{T},
    num_components::Union{Int, Nothing},
    vmin::Union{T, Nothing},
    rtol::Union{T, Nothing},
) where {T<:Base.IEEEFloat}
    a = num_components isa Int
    b = vmin isa T
    c = rtol isa T
    if a & b & c
        min(num_components, findlastcomponent(vmin, Λ), findlastrank(rtol, Λ))
    elseif a & b
        min(num_components, findlastcomponent(vmin, Λ))
    elseif a & c
        min(num_components, findlastrank(rtol, Λ))
    elseif b & c
        min(findlastcomponent(vmin, Λ), findlastrank(rtol, Λ))
    elseif a
        num_components
    elseif b
        findlastcomponent(vmin, Λ)
    elseif c
        findlastrank(rtol, Λ)
    else
        findlastrank(length(Λ) * eps(T), Λ)
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
    num_components::Union{Int, Nothing}=nothing,
    vmin::Union{T, Nothing}=nothing,
    rtol::Union{T, Nothing}=nothing,
) where {T<:Base.IEEEFloat}
    GeneralizedPCA{T}(
        collect(μ),
        collect(Σ),
        num_components = num_components,
        vmin = vmin,
        rtol = rtol,
    )
end
