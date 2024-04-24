function ispossemidef(A::AbstractMatrix{T}) where {T<:Base.IEEEFloat}
    Λ = eigvals(A)
    tol = -(length(Λ) * eps(T))
    isreal(Λ) && all(≥(tol), Λ)
end

function checkargs(μ::Vector{T}, Σ::Matrix{T}) where {T}
    n = size(Σ, 1)
    if n == size(μ, 1) == size(Σ, 2)
        if issymmetric(Σ)
            if isposdef(Σ)
                nothing
            else
                error("Σ must be positive definite")
            end
        else
            error("Σ must be symmetric")
        end
    else
        error("μ must be ℝⁿ and Σ must be ℝⁿˣⁿ")
    end
end

function checkargs(
    μ::Vector{T},
    Σ::Matrix{T},
    num_components::Union{Int,Nothing},
    vmin::Union{T,Nothing},
) where {T}
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
    num_components::Union{Int,Nothing},
    vmin::Union{T,Nothing},
    rtol::Union{T,Nothing},
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

function _estimate(X::AbstractMatrix{T}) where {T<:Base.IEEEFloat}
    m⁻¹ = convert(T, inv(size(X, 1)))
    μ = rmul!(dropdims(sum(X, dims = 1), dims = 1), m⁻¹)
    A = X .- μ'
    Σ = BLAS.gemm('T', 'N', m⁻¹, A, A)
    μ, Σ
end
