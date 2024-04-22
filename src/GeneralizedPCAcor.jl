"""
    GeneralizedPCAcor{T<:Base.IEEEFloat} <: AbstractWhiteningTransform{T}

Scale-invariant principal component analysis (PCAcor) whitening transform, generalized to
support compression based on either
(1) a pre-determined number of components,
(2) a fraction of the total squared cross-correlation, or
(3) a relative tolerance on the number of eigenvalues greater than `rtol*λ₁`
where `λ₁` is the largest eigenvalue of the correlation matrix.
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
        num_components::Int = size(Σ, 1),
        ratio::T = one(T),
        rtol::T = size(Σ, 1) * eps(T),
    ) where {T<:Base.IEEEFloat}
        checkargs(μ, Σ, num_components, ratio)
        # J, J₀ = nz_z(v)
        # v¹² = sqrt.(getindex.(Ref(v), J))
        # v⁻¹² = inv.(v¹²)
        # V⁻¹² = Diagonal(inv.(v¹²))
        # V¹² = Diagonal(v¹²)
        # P = V⁻¹² * Σ[J, J] * V⁻¹²
        v¹² = sqrt.(diag(Σ))
        V⁻¹² = Diagonal(inv.(v¹²))
        V¹² = Diagonal(v¹²)
        F = eigen(P, sortby = -)
        n⃰ = determine_nstar(F.values, num_components, ratio, rtol)
        Λ¹² = sqrt.(@view(F.values[1:n⃰]))
        Λ⁻¹² = inv.(Λ¹²)
        B¹² = Diagonal(Λ¹²)
        B⁻¹² = Diagonal(Λ⁻¹²)
        U = @view(F.vectors[:, 1:n⃰])
        W = B⁻¹² * U' * V⁻¹²
        W⁻¹ = V¹² * U * B¹²
        new{T}(μ, Σ, F, W, W⁻¹, -(W * μ))
    end
end

function nz_z(v::Vector{T}) where {T}
    J₀ = findall(iszero, v)
    n = length(v)
    k = length(J₀)
    if k == 0
        collect(1:n), J₀
    else
        m = n - k
        J = Vector{Int}(undef, m)
        i = 1
        l = 1
        j = J₀[l]
        i′ = 1
        while i ≤ m && i′ ≤ n
            if i′ == j
                l += 1
                # j = J₀[l]
                if l ≤ k
                    j = J₀[l]
                else
                    i′ += 1
                    while i ≤ m && i′ ≤ n
                        J[i] = i′
                        i += 1
                        i′ += 1
                    end
                end
            else
                J[i] = i′
                i += 1
            end
            i′ += 1
        end
        J, J₀
    end
end

function GeneralizedPCAcor(
    μ::AbstractVector{T},
    Σ::AbstractMatrix{T};
    num_components::Int = size(Σ, 1),
    ratio::T = one(T),
    rtol::T = size(Σ, 1) * eps(T),
) where {T<:Base.IEEEFloat}
    GeneralizedPCAcor{T}(
        collect(μ),
        collect(Σ),
        num_components = num_components,
        ratio = ratio,
        rtol = rtol,
    )
end
