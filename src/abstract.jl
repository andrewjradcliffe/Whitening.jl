"""
Abstract type to which represents a whitening transformation.
"""
abstract type AbstractWhiteningTransform{T<:Base.IEEEFloat} end

function whiten!(
    z::AbstractVector{T},
    kern::AbstractWhiteningTransform{T},
    x::AbstractVector{T},
) where {T<:Base.IEEEFloat}
    z .= kern.negWμ
    mul!(z, kern.W, x, true, true)
    # mul!(z, kern.W, x - kern.μ)
end
function unwhiten!(
    x::AbstractVector{T},
    kern::AbstractWhiteningTransform{T},
    z::AbstractVector{T},
) where {T<:Base.IEEEFloat}
    x .= kern.μ
    mul!(x, kern.W⁻¹, z, true, true)
end

function whiten(
    kern::AbstractWhiteningTransform{T},
    x::AbstractVector{T},
) where {T<:Base.IEEEFloat}
    # kern.W * (x - kern.μ)
    whiten!(similar(x), kern, x)
end
function unwhiten(
    kern::AbstractWhiteningTransform{T},
    z::AbstractVector{T},
) where {T<:Base.IEEEFloat}
    # kern.μ .+ kern.W⁻¹ * z
    # muladd(kern.W⁻¹, z, kern.μ)
    unwhiten!(similar(z), kern, z)
end
function mahalanobis(
    kern::AbstractWhiteningTransform{T},
    x::AbstractVector{T},
) where {T<:Base.IEEEFloat}
    z = whiten(kern, x)
    √(z ⋅ z)
end

# This results in fewer allocations if the operands fit in the cache.
# In most circumstances, it will be faster to do gemv! and axpy! (i.e. 5-arg mul!)
# and pay the cost of the allocation.
function mahalanobis_noalloc(
    kern::AbstractWhiteningTransform{T},
    x::AbstractVector{T},
) where {T<:Base.IEEEFloat}
    s = zero(T)
    for i in eachindex(kern.negWμ)
        t = @view(kern.W[i, :]) ⋅ x + kern.negWμ[i]
        s += t * t
    end
    √s
end


function whiten(
    kern::AbstractWhiteningTransform{T},
    X::AbstractMatrix{T},
) where {T<:Base.IEEEFloat}
    if size(X, 2) == size(kern.Σ, 1)
        # (X .- kern.μ') * kern.W'
        # muladd(X, kern.W', -(kern.W * kern.μ)')
        muladd(X, kern.W', kern.negWμ')
    else
        # kern.W * (X .- kern.μ)
        # muladd(kern.W, X, -(kern.W * kern.μ))
        muladd(kern.W, X, kern.negWμ)
    end
end
function unwhiten(
    kern::AbstractWhiteningTransform{T},
    Z::AbstractMatrix{T},
) where {T<:Base.IEEEFloat}
    if size(Z, 2) == size(kern.Σ, 1)
        # kern.μ' .+ Z * kern.W⁻¹'
        muladd(Z, kern.W⁻¹', kern.μ')
    else
        # kern.μ .+ kern.W⁻¹ * Z
        muladd(kern.W⁻¹, Z, kern.μ)
    end
end


function mahalanobis(
    kern::AbstractWhiteningTransform{T},
    X::AbstractMatrix{T},
) where {T<:Base.IEEEFloat}
    Z = whiten(kern, X)
    m, n = size(Z)
    if n == size(kern.Σ, 1)
        out = similar(Z, m, 1)
        for j in axes(Z, 2)
            for i in eachindex(axes(Z, 1), axes(out, 1))
                out[i, 1] += abs2(Z[i, j])
            end
        end
        for i in eachindex(out)
            out[i] = √out[i]
        end
        out
    else
        out = similar(Z, 1, n)
        for j in eachindex(axes(Z, 2), axes(out, 2))
            s = zero(T)
            for i in axes(Z, 1)
                s += abs2(Z[i, j])
            end
            out[1, j] = √s
        end
        out
    end
end
