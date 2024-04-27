var documenterSearchIndex = {"docs":
[{"location":"#Whitening.jl-Documentation","page":"Home","title":"Whitening.jl Documentation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"#Types","page":"Home","title":"Types","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Modules = [Whitening]\nOrder   = [:type]","category":"page"},{"location":"#Whitening.AbstractWhiteningTransform","page":"Home","title":"Whitening.AbstractWhiteningTransform","text":"Abstract type which represents a whitening transformation.\n\n\n\n\n\n","category":"type"},{"location":"#Whitening.Chol","page":"Home","title":"Whitening.Chol","text":"Chol{T<:Base.IEEEFloat} <: AbstractWhiteningTransform{T}\n\nCholesky whitening transform.\n\nGiven the Cholesky decomposition of the inverse covariance matrix, Σ¹ = LLᵀ, we have the whitening matrix, W = Lᵀ.\n\n\n\n\n\n","category":"type"},{"location":"#Whitening.Chol-Union{Tuple{AbstractMatrix{T}}, Tuple{T}} where T<:Union{Float16, Float32, Float64}","page":"Home","title":"Whitening.Chol","text":"Chol(X::AbstractMatrix{T}) where {T<:Base.IEEEFloat}\n\nConstruct a Cholesky transformer from the from the q × n matrix, each row of which is a sample of an n-dimensional random variable.\n\nIn order for the resultant covariance matrix to be positive definite, q must be ≥ n and none of the variances may be zero.\n\n\n\n\n\n","category":"method"},{"location":"#Whitening.Chol-Union{Tuple{T}, Tuple{AbstractVector{T}, AbstractMatrix{T}}} where T<:Union{Float16, Float32, Float64}","page":"Home","title":"Whitening.Chol","text":"Chol(μ::AbstractVector{T}, Σ::AbstractMatrix{T}) where {T<:Base.IEEEFloat}\n\nConstruct a Cholesky transformer from the from the mean vector, μ ∈ ℝⁿ, and a covariance matrix, Σ ∈ ℝⁿˣⁿ; Σ must be symmetric and positive definite.\n\n\n\n\n\n","category":"method"},{"location":"#Whitening.GeneralizedPCA","page":"Home","title":"Whitening.GeneralizedPCA","text":"GeneralizedPCA{T<:Base.IEEEFloat} <: AbstractWhiteningTransform{T}\n\nPrincipal component analysis (PCA) whitening transform, generalized to support compression based on either\n\na pre-determined number of components,\na fraction of the total squared cross-covariance, or\na relative tolerance on the number of eigenvalues greater than rtol*λ₁ where λ₁ is the largest eigenvalue of the covariance matrix.\n\nGiven the eigendecomposition of the n  n covariance matrix, Σ = UΛUᵀ, with eigenvalues sorted in descending order, i.e. λ₁  λ₂   λₙ, the first m components are selected according to one or more of the criteria listed above.\n\nIf m = n, then we have the canonical PCA whitening matrix,  W = Λ^-frac12Uᵀ. Otherwise, for m  n, a map from ℝⁿ  ℝᵐ is formed by removing the n - m rows from W, i.e. the components with the n - m smallest eigenvalues are removed. This is equivalent to selecting the m  m matrix from the upper left of Λ and the m  n matrix from the top of Uᵀ. The inverse transform is then formed by selecting the n  m matrix from the left of U and the same matrix from Λ.\n\n\n\n\n\n","category":"type"},{"location":"#Whitening.GeneralizedPCA-Union{Tuple{AbstractMatrix{T}}, Tuple{T}} where T<:Union{Float16, Float32, Float64}","page":"Home","title":"Whitening.GeneralizedPCA","text":"GeneralizedPCA(X::AbstractMatrix{T};\n               num_components::Union{Int, Nothing}=nothing,\n               vmin::Union{T, Nothing}=nothing,\n               rtol::Union{T, Nothing}=nothing) where {T<:Base.IEEEFloat}\n\nConstruct a generalized PCA transformer from the q × n matrix, each row of which is a sample of an n-dimensional random variable.\n\n\n\n\n\n","category":"method"},{"location":"#Whitening.GeneralizedPCA-Union{Tuple{T}, Tuple{AbstractVector{T}, AbstractMatrix{T}}} where T<:Union{Float16, Float32, Float64}","page":"Home","title":"Whitening.GeneralizedPCA","text":"GeneralizedPCA(μ::AbstractVector{T}, Σ::AbstractMatrix{T};\n               num_components::Union{Int, Nothing}=nothing,\n               vmin::Union{T, Nothing}=nothing,\n               rtol::Union{T, Nothing}=nothing) where {T<:Base.IEEEFloat}\n\nConstruct a generalized PCA transformer from the mean vector, μ ∈ ℝⁿ, and a covariance matrix, Σ ∈ ℝⁿˣⁿ; Σ must be symmetric and positive semi-definite.\n\nThe output dimension, m, of the transformer is determined from the optional arguments, where\n\n0 ≤ num_components ≤ n is a pre-determined size\n0 ≤ vmin ≤ 1 is the fraction of the total squared cross-covariance,  hence, m is the smallest value such that sum(λ[1:m]) ≥ vmin*sum(λ),  where λᵢ i=1n are the eigenvalues of Σ in descending order.\nrtol is the relative tolerance on the number of eigenvalues greater than rtol*λ₁ where λ₁ is the largest eigenvalue of Σ.\n\nIf none of the 3 options are provided, the default is rtol = n*eps(T). If 2 or more options are provided, the minimum of the resultant sizes will be chosen.\n\n\n\n\n\n","category":"method"},{"location":"#Whitening.GeneralizedPCAcor","page":"Home","title":"Whitening.GeneralizedPCAcor","text":"GeneralizedPCAcor{T<:Base.IEEEFloat} <: AbstractWhiteningTransform{T}\n\nScale-invariant principal component analysis (PCAcor) whitening transform, generalized to support compression based on either\n\na pre-determined number of components,\na fraction of the total squared cross-correlation, or\na relative tolerance on the number of eigenvalues greater than rtol*λ₁ where λ₁ is the largest eigenvalue of the correlation matrix.\n\nGiven the eigendecomposition of the n  n correlation matrix, P = GΘGᵀ, with eigenvalues sorted in descending order, i.e. θ₁  θ₂   θₙ, the first m components are selected according to one or more of the criteria listed above.\n\nIf m = n, then we have the canonical PCA-cor whitening matrix,  W = Θ^-frac12GᵀV^-frac12. Otherwise, for m  n, a map from ℝⁿ  ℝᵐ is formed by removing the n - m rows from W, i.e. the components with the n - m smallest eigenvalues are removed. This is equivalent to selecting the m  m matrix from the upper left of Θ and the m  n matrix from the top of Gᵀ. The inverse transform is then formed by selecting the n  m matrix from the left of G and the same matrix from Θ.\n\n\n\n\n\n","category":"type"},{"location":"#Whitening.GeneralizedPCAcor-Union{Tuple{AbstractMatrix{T}}, Tuple{T}} where T<:Union{Float16, Float32, Float64}","page":"Home","title":"Whitening.GeneralizedPCAcor","text":"GeneralizedPCAcor(X::AbstractMatrix{T};\n                  num_components::Union{Int, Nothing}=nothing,\n                  vmin::Union{T, Nothing}=nothing,\n                  rtol::Union{T, Nothing}=nothing) where {T<:Base.IEEEFloat}\n\nConstruct a generalized PCAcor transformer from the q × n matrix, each row of which is a sample of an n-dimensional random variable.\n\n\n\n\n\n","category":"method"},{"location":"#Whitening.GeneralizedPCAcor-Union{Tuple{T}, Tuple{AbstractVector{T}, AbstractMatrix{T}}} where T<:Union{Float16, Float32, Float64}","page":"Home","title":"Whitening.GeneralizedPCAcor","text":"GeneralizedPCAcor(μ::AbstractVector{T}, Σ::AbstractMatrix{T};\n                  num_components::Union{Int, Nothing}=nothing,\n                  vmin::Union{T, Nothing}=nothing,\n                  rtol::Union{T, Nothing}=nothing) where {T<:Base.IEEEFloat}\n\nConstruct a generalized PCAcor transformer from the mean vector, μ ∈ ℝⁿ, and a covariance matrix, Σ ∈ ℝⁿˣⁿ; Σ must be symmetric and positive semi-definite.\n\nThe decomposition, Σ = V^\frac12 * P * V^\frac12, where V is the diagonal matrix of variances and P is a correlation matrix, must be well-formed in order to obtain a meaningful result. That is, if the diagonal of Σ contains 1 or more zero elements, then it is not possible to compute P = V^-\frac12 * Σ * V^-\frac12.\n\nThe output dimension, m, of the transformer is determined from the optional arguments, where\n\n0 ≤ num_components ≤ n is a pre-determined size\n0 ≤ vmin ≤ 1 is the fraction of the total squared cross-covariance,  hence, m is the smallest value such that sum(λ[1:m]) ≥ vmin*sum(λ),  where θᵢ i=1n are the eigenvalues of P in descending order.\nrtol is the relative tolerance on the number of eigenvalues greater than rtol*θ₁ where θ₁ is the largest eigenvalue of P.\n\nIf none of the 3 options are provided, the default is rtol = n*eps(T). If 2 or more options are provided, the minimum of the resultant sizes will be chosen.\n\n\n\n\n\n","category":"method"},{"location":"#Whitening.PCA","page":"Home","title":"Whitening.PCA","text":"PCA{T<:Base.IEEEFloat} <: AbstractWhiteningTransform{T}\n\nPrincipal component analysis (PCA) whitening transform.\n\nGiven the eigendecomposition of the covariance matrix, Σ = UΛUᵀ, we have the whitening matrix, W = Λ^-frac12Uᵀ.\n\n\n\n\n\n","category":"type"},{"location":"#Whitening.PCA-Union{Tuple{AbstractMatrix{T}}, Tuple{T}} where T<:Union{Float16, Float32, Float64}","page":"Home","title":"Whitening.PCA","text":"PCA(X::AbstractMatrix{T}) where {T<:Base.IEEEFloat}\n\nConstruct a PCA transformer from the from the q × n matrix, each row of which is a sample of an n-dimensional random variable.\n\nIn order for the resultant covariance matrix to be positive definite, q must be ≥ n and none of the variances may be zero.\n\n\n\n\n\n","category":"method"},{"location":"#Whitening.PCA-Union{Tuple{T}, Tuple{AbstractVector{T}, AbstractMatrix{T}}} where T<:Union{Float16, Float32, Float64}","page":"Home","title":"Whitening.PCA","text":"PCA(μ::AbstractVector{T}, Σ::AbstractMatrix{T}) where {T<:Base.IEEEFloat}\n\nConstruct a PCA transformer from the from the mean vector, μ ∈ ℝⁿ, and a covariance matrix, Σ ∈ ℝⁿˣⁿ; Σ must be symmetric and positive definite.\n\n\n\n\n\n","category":"method"},{"location":"#Whitening.PCAcor","page":"Home","title":"Whitening.PCAcor","text":"PCAcor{T<:Base.IEEEFloat} <: AbstractWhiteningTransform{T}\n\nScale-invariant principal component analysis (PCA-cor) whitening transform.\n\nGiven the eigendecomposition of the correlation matrix, P = GΘGᵀ, and the diagonal variance matrix, V, we have the whitening matrix, W = Θ^-frac12GᵀV^-frac12.\n\n\n\n\n\n","category":"type"},{"location":"#Whitening.PCAcor-Union{Tuple{AbstractMatrix{T}}, Tuple{T}} where T<:Union{Float16, Float32, Float64}","page":"Home","title":"Whitening.PCAcor","text":"PCAcor(X::AbstractMatrix{T}) where {T<:Base.IEEEFloat}\n\nConstruct a PCAcor transformer from the from the q × n matrix, each row of which is a sample of an n-dimensional random variable.\n\nIn order for the resultant covariance matrix to be positive definite, q must be ≥ n and none of the variances may be zero.\n\n\n\n\n\n","category":"method"},{"location":"#Whitening.PCAcor-Union{Tuple{T}, Tuple{AbstractVector{T}, AbstractMatrix{T}}} where T<:Union{Float16, Float32, Float64}","page":"Home","title":"Whitening.PCAcor","text":"PCAcor(μ::AbstractVector{T}, Σ::AbstractMatrix{T}) where {T<:Base.IEEEFloat}\n\nConstruct a PCAcor transformer from the from the mean vector, μ ∈ ℝⁿ, and a covariance matrix, Σ ∈ ℝⁿˣⁿ; Σ must be symmetric and positive definite.\n\n\n\n\n\n","category":"method"},{"location":"#Whitening.ZCA","page":"Home","title":"Whitening.ZCA","text":"ZCA{T<:Base.IEEEFloat} <: AbstractWhiteningTransform{T}\n\nZero-phase component analysis (ZCA) whitening transform.\n\nGiven the covariance matrix, Σ, we have the whitening matrix, W = Σ^-frac12.\n\n\n\n\n\n","category":"type"},{"location":"#Whitening.ZCA-Union{Tuple{AbstractMatrix{T}}, Tuple{T}} where T<:Union{Float16, Float32, Float64}","page":"Home","title":"Whitening.ZCA","text":"ZCA(X::AbstractMatrix{T}) where {T<:Base.IEEEFloat}\n\nConstruct a ZCA transformer from the from the q × n matrix, each row of which is a sample of an n-dimensional random variable.\n\nIn order for the resultant covariance matrix to be positive definite, q must be ≥ n and none of the variances may be zero.\n\n\n\n\n\n","category":"method"},{"location":"#Whitening.ZCA-Union{Tuple{T}, Tuple{AbstractVector{T}, AbstractMatrix{T}}} where T<:Union{Float16, Float32, Float64}","page":"Home","title":"Whitening.ZCA","text":"ZCA(μ::AbstractVector{T}, Σ::AbstractMatrix{T}) where {T<:Base.IEEEFloat}\n\nConstruct a ZCA transformer from the from the mean vector, μ ∈ ℝⁿ, and a covariance matrix, Σ ∈ ℝⁿˣⁿ; Σ must be symmetric and positive definite.\n\n\n\n\n\n","category":"method"},{"location":"#Whitening.ZCAcor","page":"Home","title":"Whitening.ZCAcor","text":"ZCAcor{T<:Base.IEEEFloat} <: AbstractWhiteningTransform{T}\n\nScale-invariant zero-phase component analysis (ZCA-cor) whitening transform.\n\nGiven the correlation matrix, P, and the diagonal variance matrix, V, we have the whitening matrix, W = P^-frac12V^-frac12.\n\n\n\n\n\n","category":"type"},{"location":"#Whitening.ZCAcor-Union{Tuple{AbstractMatrix{T}}, Tuple{T}} where T<:Union{Float16, Float32, Float64}","page":"Home","title":"Whitening.ZCAcor","text":"ZCAcor(X::AbstractMatrix{T}) where {T<:Base.IEEEFloat}\n\nConstruct a ZCAcor transformer from the from the q × n matrix, each row of which is a sample of an n-dimensional random variable.\n\nIn order for the resultant covariance matrix to be positive definite, q must be ≥ n and none of the variances may be zero.\n\n\n\n\n\n","category":"method"},{"location":"#Whitening.ZCAcor-Union{Tuple{T}, Tuple{AbstractVector{T}, AbstractMatrix{T}}} where T<:Union{Float16, Float32, Float64}","page":"Home","title":"Whitening.ZCAcor","text":"ZCAcor(μ::AbstractVector{T}, Σ::AbstractMatrix{T}) where {T<:Base.IEEEFloat}\n\nConstruct a ZCAcor transformer from the from the mean vector, μ ∈ ℝⁿ, and a covariance matrix, Σ ∈ ℝⁿˣⁿ; Σ must be symmetric and positive definite.\n\n\n\n\n\n","category":"method"},{"location":"#Functions","page":"Home","title":"Functions","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Modules = [Whitening]\nOrder   = [:function]","category":"page"},{"location":"#Whitening.mahalanobis-Union{Tuple{T}, Tuple{Whitening.AbstractWhiteningTransform{T}, AbstractMatrix{T}}} where T<:Union{Float16, Float32, Float64}","page":"Home","title":"Whitening.mahalanobis","text":"mahalanobis(K::AbstractWhiteningTransform{T}, X::AbstractMatrix{T}) where {T<:Base.IEEEFloat}\n\nReturn the Mahalanobis distance, √((x - μ)' * Σ⁻¹ * (x - μ)), computed for each row in X, using the transformation kernel, K.\n\n\n\n\n\n","category":"method"},{"location":"#Whitening.mahalanobis-Union{Tuple{T}, Tuple{Whitening.AbstractWhiteningTransform{T}, AbstractVector{T}}} where T<:Union{Float16, Float32, Float64}","page":"Home","title":"Whitening.mahalanobis","text":"mahalanobis(K::AbstractWhiteningTransform{T}, x::AbstractVector{T}) where {T<:Base.IEEEFloat}\n\nReturn the Mahalanobis distance, √((x - μ)' * Σ⁻¹ * (x - μ)), computed using the transformation kernel, K.\n\n\n\n\n\n","category":"method"},{"location":"#Whitening.unwhiten-Union{Tuple{T}, Tuple{Whitening.AbstractWhiteningTransform{T}, AbstractMatrix{T}}} where T<:Union{Float16, Float32, Float64}","page":"Home","title":"Whitening.unwhiten","text":"unwhiten(K::AbstractWhiteningTransform{T}, Z::AbstractMatrix{T}) where {T<:Base.IEEEFloat}\n\nTransform the rows of Z to unwhitened vectors, i.e. X = Z * (W⁻¹)ᵀ .+ μᵀ, using the provided kernel. That is, Z is an m × p matrix and K is a transformation kernel whose output dimension is p.\n\nIf K compresses n ↦ p, i.e. z = Wx : ℝⁿ ↦ ℝᵖ, then X is an m × n matrix.\n\n\n\n\n\n","category":"method"},{"location":"#Whitening.unwhiten-Union{Tuple{T}, Tuple{Whitening.AbstractWhiteningTransform{T}, AbstractVector{T}}} where T<:Union{Float16, Float32, Float64}","page":"Home","title":"Whitening.unwhiten","text":"unwhiten(K::AbstractWhiteningTransform{T}, z::AbstractVector{T}) where {T<:Base.IEEEFloat}\n\nTransform z to the original coordinate system of a non-whitened vector belonging to the kernel, K, i.e. x = μ + W⁻¹ * z. This is the inverse of whiten(K, x).\n\nIf K compresses n ↦ p, then x ∈ ℝⁿ.\n\n\n\n\n\n","category":"method"},{"location":"#Whitening.whiten-Union{Tuple{T}, Tuple{Whitening.AbstractWhiteningTransform{T}, AbstractMatrix{T}}} where T<:Union{Float16, Float32, Float64}","page":"Home","title":"Whitening.whiten","text":"whiten(K::AbstractWhiteningTransform{T}, X::AbstractMatrix{T}) where {T<:Base.IEEEFloat}\n\nTransform the rows of X to whitened vectors, i.e. Z = (X .- μᵀ) * Wᵀ, using the provided kernel. That is, X is an m × n matrix and K is a transformation kernel whose input dimension is n.\n\nIf K compresses n ↦ p, i.e. z = Wx : ℝⁿ ↦ ℝᵖ, then Z is an m × p matrix.\n\n\n\n\n\n","category":"method"},{"location":"#Whitening.whiten-Union{Tuple{T}, Tuple{Whitening.AbstractWhiteningTransform{T}, AbstractVector{T}}} where T<:Union{Float16, Float32, Float64}","page":"Home","title":"Whitening.whiten","text":"whiten(K::AbstractWhiteningTransform{T}, x::AbstractVector{T}) where {T<:Base.IEEEFloat}\n\nTransform x to a whitened vector, i.e. z = W * (x - μ), using the transformation kernel, K.\n\nIf K compresses n ↦ p, then z ∈ ℝᵖ.\n\n\n\n\n\n","category":"method"},{"location":"#Index","page":"Home","title":"Index","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Modules = [Whitening]\nOrder   = [:function, :type]","category":"page"}]
}
