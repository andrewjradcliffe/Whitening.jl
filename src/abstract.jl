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
    whiten(K::AbstractWhiteningTransform{T}, x::AbstractVector{T}) where {T<:Base.IEEEFloat}

Transform `x` to a whitened vector, i.e. `z = W * (x - μ)`, using the transformation
kernel, `K`.

If `K` compresses `n ↦ p`, then `z ∈ ℝᵖ`.
"""
function whiten(
    kern::AbstractWhiteningTransform{T},
    x::AbstractVector{T},
) where {T<:Base.IEEEFloat}
    # kern.W * (x - kern.μ)
    whiten!(similar(x, output_size(kern)), kern, x)
end

"""
    unwhiten(K::AbstractWhiteningTransform{T}, z::AbstractVector{T}) where {T<:Base.IEEEFloat}

Transform `z` to the original coordinate system of a non-whitened vector
belonging to the kernel, `K`, i.e. `x = μ + W⁻¹ * z`.
This is the inverse of `whiten(K, x)`.

If `K` compresses `n ↦ p`, then `x ∈ ℝⁿ`.
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
    mahalanobis(K::AbstractWhiteningTransform{T}, x::AbstractVector{T}) where {T<:Base.IEEEFloat}

Return the Mahalanobis distance, `√((x - μ)' * Σ⁻¹ * (x - μ))`, computed using the
transformation kernel, `K`.
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
# and pay the cost of the allocation. In practice, this is almost always slower.
#=
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
=#

function whiten!(
    Z::AbstractMatrix{T},
    kern::AbstractWhiteningTransform{T},
    X::AbstractMatrix{T},
) where {T<:Base.IEEEFloat}
    Z .= kern.negWμ'
    mul!(Z, X, kern.W', true, true)
    #=
    # Body if supporting both orientations
    if size(X, 2) == input_size(kern)
        Z .= kern.negWμ'
        mul!(Z, X, kern.W', true, true)
    else
        Z .= kern.negWμ
        mul!(Z, kern.W, X, true, true)
    end
    =#
end
function unwhiten!(
    X::AbstractMatrix{T},
    kern::AbstractWhiteningTransform{T},
    Z::AbstractMatrix{T},
) where {T<:Base.IEEEFloat}
    X .= kern.μ'
    mul!(X, Z, kern.W⁻¹', true, true)
    #=
    # Body if supporting both orientations
    if size(X, 2) == input_size(kern)
        X .= kern.μ'
        mul!(X, Z, kern.W⁻¹', true, true)
    else
        X .= kern.μ
        mul!(X, kern.W⁻¹, Z, true, true)
    end
    =#
end

"""
    whiten(K::AbstractWhiteningTransform{T}, X::AbstractMatrix{T}) where {T<:Base.IEEEFloat}

Transform the rows of `X` to whitened vectors, i.e. `Z = (X .- μᵀ) * Wᵀ`,
using the provided kernel. That is, `X` is an `m × n` matrix and `K` is a transformation
kernel whose input dimension is `n`.

If `K` compresses `n ↦ p`, i.e. `z = Wx : ℝⁿ ↦ ℝᵖ`, then `Z` is an `m × p` matrix.
"""
function whiten(
    kern::AbstractWhiteningTransform{T},
    X::AbstractMatrix{T},
) where {T<:Base.IEEEFloat}
    whiten!(similar(X, size(X, 1), output_size(kern)), kern, X)
    #=
    # Body if supporting both orientations
    m, n = size(X)
    if n == input_size(kern)
        muladd(X, kern.W', kern.negWμ')
        whiten!(similar(X, size(X, 1), output_size(kern)), kern, X)
    else
        muladd(kern.W, X, kern.negWμ)
        whiten!(similar(X, output_size(kern), n), kern, X)
    end
    =#
end

"""
    unwhiten(K::AbstractWhiteningTransform{T}, Z::AbstractMatrix{T}) where {T<:Base.IEEEFloat}

Transform the rows of `Z` to unwhitened vectors, i.e. `X = Z * (W⁻¹)ᵀ .+ μᵀ`,
using the provided kernel. That is, `Z` is an `m × p` matrix and `K` is a transformation
kernel whose output dimension is `p`.

If `K` compresses `n ↦ p`, i.e. `z = Wx : ℝⁿ ↦ ℝᵖ`, then `X` is an `m × n` matrix.
"""
function unwhiten(
    kern::AbstractWhiteningTransform{T},
    Z::AbstractMatrix{T},
) where {T<:Base.IEEEFloat}
    unwhiten!(similar(Z, size(Z, 1), input_size(kern)), kern, Z)
    #=
    # Body if supporting both orientations
    m, n = size(Z)
    if n == output_size(kern)
        muladd(Z, kern.W⁻¹', kern.μ')
        unwhiten!(similar(Z, size(Z, 1), input_size(kern)), kern, Z)
    else
        muladd(kern.W⁻¹, Z, kern.μ)
        unwhiten!(similar(Z, input_size(kern), n), kern, Z)
    end
   =#
end


"""
    mahalanobis(K::AbstractWhiteningTransform{T}, X::AbstractMatrix{T}) where {T<:Base.IEEEFloat}

Return the Mahalanobis distance, `√((x - μ)' * Σ⁻¹ * (x - μ))`, computed for each
row in `X`, using the transformation kernel, `K`.
"""
function mahalanobis(
    kern::AbstractWhiteningTransform{T},
    X::AbstractMatrix{T},
) where {T<:Base.IEEEFloat}
    Z = whiten(kern, X)
    out = zeros(T, size(Z, 1))
    for j in axes(Z, 2)
        for i in eachindex(axes(Z, 1), out)
            out[i] += abs2(Z[i, j])
        end
    end
    for i in eachindex(out)
        out[i] = √out[i]
    end
    out
    #=
    # Body if supporting both orientations
    m, n = size(X)
    Z = whiten(kern, X)
    if n == input_size(kern)
        out = zeros(T, m, 1)
        k = firstindex(Z, 2)
        for j in axes(Z, 2)
            for i in eachindex(axes(Z, 1), axes(out, 1))
                out[i, k] += abs2(Z[i, j])
            end
        end
        for i in eachindex(out)
            out[i] = √out[i]
        end
        out
    else
        out = zeros(T, 1, n)
        k = firstindex(Z, 1)
        for j in eachindex(axes(Z, 2), axes(out, 2))
            s = zero(T)
            for i in axes(Z, 1)
                s += abs2(Z[i, j])
            end
            out[k, j] = √s
        end
        out
    end
    =#
end
