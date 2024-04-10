module Whitening

using LinearAlgebra

export WhiteningAlgorithm, whiten, distances

# public Kernel

#=
On whitening and Mahalanobis distance with rank-deficient covariance matrices

The obvious option, which is reasonably robust, is to use the Moore-Penrose
pseudo-inverse of Σ. The downside is that when Σ is rank-deficient, `pinv(Σ)`
is not necessarily symmetric. The advantage of this approach is that it hides
the choice of decomposition inside the `pinv` implementation.

The conceptual alternative is to whiten the vectors, then compute the inner
product of each vector with itself.

For whitening methods, we have two alternatives which are feasible with
rank-deficient covariance matrices: Cholesky and eigen-system.

Cholesky whitening can be implemented in several ways (see alternatives
inside `distances2`), but each one uses `pinv`, which, in 2 out of 3
cases, necessitates the construction of a symmetric pseudo-inverse via (A + A') / 2.
All of these do a poor job of preserving the rank of the covariance matrix,
due largely to stability problems in calling `pinv` on a rank-deficient square
matrix. As far as efficiency is concerned, these are all less efficient
than simply calling `pinv` directly, as one must also perform a Cholesky
decomposition.

Eigen-system whitening has several conceptual advantages which arise from
the fact that we do not try to invert a rank-deficient matrix. This confers
numerical stability (insofar as can be achieved), as the inversion
step is simply the inversion of the square root of values ≥(eps(Float64)).
Moreover, enables this enables one to preserve the rank of the input
covariance matrix (in contrast, the Cholesky methods result in loss
due to instability). Lastly, the time and space complexity of the approach
is limited to that of the singular value decomposition, hence, has
yet another advantage over the Cholesky approach.
=#

@enum WhiteningAlgorithm::Int8 ZCA Chol PCA

ispossemidef(A::Matrix{T}) where {T<:AbstractFloat} = all(≥(-eps(T)), eigvals(A))

struct Kernel{T<:Base.IEEEFloat}
    μ::Vector{T}
    Σ::Matrix{T}
    # Unless Σ is full-rank, this is just pinv
    # Σ⁻¹::Matrix{T}
    # The SVD of Σ
    F::SVD{T,T,Matrix{T},Vector{T}}
    I::Vector{Int}
    function Kernel{T}(μ::Vector{T}, Σ::Matrix{T}) where {T<:Base.IEEEFloat}
        if length(μ) == size(Σ, 1) == size(Σ, 2)
            if ispossemidef(Σ)
                F = svd(Σ)
                new{T}(μ, Σ, F, findall(≥(eps(T)), F.S))
            else
                error("Σ should must be positive semi-definite")
            end
        else
            error("μ must be ∈ ℝⁿ and Σ ∈ ℝⁿˣⁿ")
        end
    end
end
Kernel(μ::Vector{T}, Σ::Matrix{T}) where {T<:Base.IEEEFloat} = Kernel{T}(μ, Σ)
Kernel(μ::AbstractVector{T}, Σ::AbstractMatrix{T}) where {T<:Base.IEEEFloat} =
    Kernel{T}(collect(μ), collect(Σ))

function distances(
    xs::AbstractVector{T},
    μ::AbstractVector{U},
    Σ::AbstractMatrix{U};
    alg::WhiteningAlgorithm = PCA,
) where {T<:AbstractVector{U}} where {U<:Base.IEEEFloat}
    if alg == PCA
        _pca_distances(xs, μ, Σ)
    elseif alg == ZCA
        _zca_distances(xs, μ, Σ)
    else
        _chol_distances(xs, μ, Σ)
    end
end
function distances(
    xs::AbstractVector{T},
    K::Kernel;
    alg::WhiteningAlgorithm = PCA,
) where {U<:Base.IEEEFloat,T<:AbstractVector{U}}
    if alg == PCA
        _pca_distances(K)
    else
        distances(xs, K.μ, K.Σ, alg = alg)
    end
end

function _zca_mahalanobis(x, μ, Σ⁻¹)
    δ = x - μ
    √(δ' * Σ⁻¹ * δ)
end
function _zca_distances(
    xs::AbstractVector{T},
    μ::AbstractVector{U},
    Σ::AbstractMatrix{U},
) where {T<:AbstractVector{U}} where {U<:Base.IEEEFloat}
    Σ⁻¹ = pinv(Σ)
    map(x -> _zca_mahalanobis(x, μ, Σ⁻¹), xs)
end

function _whiten_mahalanobis(
    x::AbstractVector{T},
    μ::AbstractVector{T},
    W::AbstractMatrix{T},
) where {T<:Real}
    z = W * (x - μ)
    √(z ⋅ z)
end

function _chol_distances(
    xs::AbstractVector{T},
    μ::AbstractVector{U},
    Σ::AbstractMatrix{U},
) where {T<:AbstractVector{U}} where {U<:Base.IEEEFloat}
    F = cholesky(hermitianpart(pinv(Σ)), RowMaximum(), check = false)
    W = F.U[1:F.rank, invperm(F.p)]

    # Cholesky alternative 1
    # F = cholesky(hermitianpart(Σ), RowMaximum(), check = false)
    # Lp = [F.L[invperm(F.p), 1:F.rank] zeros(size(Σ, 1), size(Σ, 2) - F.rank)]
    # W = pinv(Lp)

    # Cholesky alternative 2
    # F = cholesky(hermitianpart(pinv(Σ)), RowMaximum(), check = false)
    # W = [F.U[1:F.rank, invperm(F.p)]; zeros(size(Σ, 1) - F.rank, size(Σ, 2))]
    map(x -> _whiten_mahalanobis(x, μ, W), xs)
end

function _pca_distances(
    xs::AbstractVector{T},
    μ::AbstractVector{U},
    Σ::AbstractMatrix{U},
) where {T<:AbstractVector{U}} where {U<:Base.IEEEFloat}
    F = svd(Σ)
    I = findall(≥(eps(U)), K.F.S)
    W = Diagonal(inv.(sqrt.(@view(F.S[I])))) * @view(F.U'[I, :])
    map(x -> _whiten_mahalanobis(x, μ, W), xs)
end

function _pca_distances(
    xs::AbstractVector{T},
    K::Kernel{U},
) where {T<:AbstractVector{U}} where {U<:Base.IEEEFloat}
    I = findall(≥(eps(U)), K.F.S)
    W = Diagonal(inv.(sqrt.(@view(K.F.S[K.I])))) * @view(K.F.U'[K.I, :])
    map(x -> _whiten_mahalanobis(x, K.μ, W), xs)
end

function whiten(
    x::AbstractVector{T},
    μ::AbstractVector{T},
    Σ::AbstractMatrix{T};
    alg::WhiteningAlgorithm = PCA,
) where {T<:Base.IEEEFloat}
    if alg == PCA
        _pca_whiten(x, μ, Σ)
    elseif alg == Chol
        _chol_whiten(x, μ, Σ)
    else
        _zca_whiten(x, μ, Σ)
    end
end
function whiten(
    x::AbstractVector{T},
    K::Kernel;
    alg::WhiteningAlgorithm = PCA,
) where {U<:Base.IEEEFloat,T<:AbstractVector{U}}
    if alg == PCA
        _pca_whiten(x, K)
    else
        whiten(x, K.μ, K.Σ, alg = alg)
    end
end


function _zca_whiten(
    x::AbstractVector{T},
    μ::AbstractVector{T},
    Σ::AbstractMatrix{T},
) where {T<:Base.IEEEFloat}
    W = sqrt(inv(Σ))
    W * (x - μ)
end
function _chol_whiten(
    x::AbstractVector{T},
    μ::AbstractVector{T},
    Σ::AbstractMatrix{T},
) where {T<:Base.IEEEFloat}
    F = cholesky(hermitianpart(pinv(Σ)), RowMaximum(), check = false)
    Lᵀ = F.U[1:F.rank, invperm(F.p)]
    # Or, if a full-size vector is desired (pointless)
    # Lᵀ = [F.U[1:F.rank, invperm(F.p)]; zeros(size(Σ, 1) - F.rank, size(Σ, 2))]
    Lᵀ * (x - μ)
end
function _pca_whiten(
    x::AbstractVector{T},
    μ::AbstractVector{T},
    Σ::AbstractMatrix{T},
) where {T<:Base.IEEEFloat}
    F = svd(Σ)
    I = findall(≥(eps(T)), F.S)
    Diagonal(inv.(sqrt.(F.S[I]))) * F.U'[I, :] * (x - μ)
end

function _pca_whiten(x, K::Kernel{T}) where {T<:Base.IEEEFloat}
    Diagonal(inv.(sqrt.(@view(K.F.S[K.I])))) * @view(K.F.U'[K.I, :]) * (x - K.μ)
end



end # module Whitening
