using LinearAlgebra
function cov_varcov(X::AbstractMatrix{T}) where {T<:Number}
    n, p = size(X)
    x̄ = ldiv!(n, dropdims(sum(X, dims=1), dims=1))
    Z = X .- x̄'
    W = Array{T}(undef, n, p, p)
    for j in 1:p
        for i in 1:p
            for k in 1:n
                W[k, i, j] = Z[k, i] * Z[k, j]
            end
        end
    end
    W̄ = ldiv!(n, dropdims(sum(W, dims=1), dims=1))
    S = (n / (n - 1)) * W̄
    V̂ = Array{T}(undef, p, p)
    for j in 1:p
        for i in 1:p
            s = zero(T)
            for k in 1:n
                δ = W[k, i, j] - W̄[i, j]
                s += δ * δ
            end
            V̂[i, j] = (n / ((n - 1)^3)) * s
        end
    end
    S, V̂
end
function cor_varcor(X::AbstractMatrix{T}) where {T<:Number}
    n, p = size(X)
    x̄ = ldiv!(n, dropdims(sum(X, dims=1), dims=1))
    v = Vector{T}(undef, p)
    for i in 1:p
        s = zero(T)
        for k in 1:n
            δ = X[k, i] - x̄[i]
            s += δ * δ
        end
        v[i] = (1 / (n - 1)) * s
    end
    V⁻¹² = Diagonal(inv.(sqrt.(v)))
    Z = X * V⁻¹²
    cov_varcov(Z)
end

function shrinkage(X::AbstractMatrix{T}) where {T<:Number}
    R, var_R = cor_varcor(X)
    S, var_S = cov_varcov(X)
    n, p = size(X)
    num = sum(var_R) - tr(var_R)
    den = sum(abs2, R) - sum(abs2, diag(R))
    λ̂ = num / den
    R⃰ = similar(R)
    for j in 1:p
        for i in 1:p
            R⃰[i, j] = i == j ? one(eltype(R)) : R[i, j] * min(1, max(0, 1 - λ̂))
        end
    end
    S⃰ = similar(R⃰)
    for j in 1:p
        for i in 1:p
            S⃰[i, j] = i == j ? S[i, i] : R⃰[i, j] * √(S[i, i] * S[j, j])
        end
    end
    S⃰, R⃰
end
