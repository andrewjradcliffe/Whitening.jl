"""
Abstract type to which represents a whitening transformation.
"""
abstract type AbstractWhiteningTransform{T<:Base.IEEEFloat} end

@inline function input_size(kern::AbstractWhiteningTransform{T}) where {T<:Base.IEEEFloat}
    size(kern.W⁻¹, 1)
end
@inline function output_size(kern::AbstractWhiteningTransform{T}) where {T<:Base.IEEEFloat}
    size(kern.W, 1)
end

function whiten!(
    z::AbstractVector{T},
    kern::AbstractWhiteningTransform{T},
    x::AbstractVector{T},
) where {T<:Base.IEEEFloat}
    z .= kern.negWμ
    mul!(z, kern.W, x, true, true)
end
function unwhiten!(
    x::AbstractVector{T},
    kern::AbstractWhiteningTransform{T},
    z::AbstractVector{T},
) where {T<:Base.IEEEFloat}
    x .= kern.μ
    mul!(x, kern.W⁻¹, z, true, true)
end

"""
    whiten(kernel::AbstractWhiteningTransform{T}, x::AbstractVector{T}) where {T<:Base.IEEEFloat}

Transform `x` to a whitened vector, i.e. `z = W * (x - μ)`, using the provided kernel.
"""
function whiten(
    kern::AbstractWhiteningTransform{T},
    x::AbstractVector{T},
) where {T<:Base.IEEEFloat}
    # kern.W * (x - kern.μ)
    whiten!(similar(x, output_size(kern)), kern, x)
end

"""
    unwhiten(kernel::AbstractWhiteningTransform{T}, z::AbstractVector{T}) where {T<:Base.IEEEFloat}

Transform `z` to a non-centered, correlated and scaled vector, i.e.
`x = μ + W⁻¹ * z`, using the provided kernel. This is the inverse of `whiten(kernel, x)`.
"""
function unwhiten(
    kern::AbstractWhiteningTransform{T},
    z::AbstractVector{T},
) where {T<:Base.IEEEFloat}
    # kern.μ .+ kern.W⁻¹ * z
    # muladd(kern.W⁻¹, z, kern.μ)
    unwhiten!(similar(z, input_size(kern)), kern, z)
end

"""
    mahalanobis(kernel::AbstractWhiteningTransform{T}, x::AbstractVector{T}) where {T<:Base.IEEEFloat}

Return the Mahalanobis distance, `√((x - μ)' * Σ⁻¹ * (x - μ))`.
"""
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
    for i in eachindex(axes(kern.W, 1), kern.negWμ)
        t = @view(kern.W[i, :]) ⋅ x + kern.negWμ[i]
        s += t * t
    end
    √s
end

function whiten!(
    Z::AbstractMatrix{T},
    kern::AbstractWhiteningTransform{T},
    X::AbstractMatrix{T},
) where {T<:Base.IEEEFloat}
    if size(X, 2) == input_size(kern)
        Z .= kern.negWμ'
        mul!(Z, X, kern.W', true, true)
    else
        Z .= kern.negWμ
        mul!(Z, kern.W, X, true, true)
    end
end
function unwhiten!(
    X::AbstractMatrix{T},
    kern::AbstractWhiteningTransform{T},
    Z::AbstractMatrix{T},
) where {T<:Base.IEEEFloat}
    if size(X, 2) == input_size(kern)
        X .= kern.μ'
        mul!(X, Z, kern.W⁻¹', true, true)
    else
        X .= kern.μ
        mul!(X, kern.W⁻¹, Z, true, true)
    end
end

function whiten(
    kern::AbstractWhiteningTransform{T},
    X::AbstractMatrix{T},
) where {T<:Base.IEEEFloat}
    m, n = size(X)
    if n == input_size(kern)
        # muladd(X, kern.W', kern.negWμ')
        whiten!(similar(X, m, output_size(kern)), kern, X)
    else
        # muladd(kern.W, X, kern.negWμ)
        whiten!(similar(X, output_size(kern), n), kern, X)
    end
end
function unwhiten(
    kern::AbstractWhiteningTransform{T},
    Z::AbstractMatrix{T},
) where {T<:Base.IEEEFloat}
    m, n = size(Z)
    if n == output_size(kern)
        # muladd(Z, kern.W⁻¹', kern.μ')
        unwhiten!(similar(Z, m, input_size(kern)), kern, Z)
    else
        # muladd(kern.W⁻¹, Z, kern.μ)
        unwhiten!(similar(Z, input_size(kern), n), kern, Z)
    end
end


function mahalanobis(
    kern::AbstractWhiteningTransform{T},
    X::AbstractMatrix{T},
) where {T<:Base.IEEEFloat}
    m, n = size(X)
    Z = whiten(kern, X)
    if n == input_size(kern)
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
