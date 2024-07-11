using LinearAlgebra, Statistics, Random, Test, Base.Threads
using Whitening: PCA, unwhiten
function covariance(p::Int, q::Int, λ::Vector{Float64})
    m = length(λ)
    if m != min(p, q)
        error("mismatched sizes")
    else
        if m == p
            U = [Diagonal(λ) zeros(p, q - p)]
            [I(p) U; U' I(q)]
        else # m == q
            U = [Diagonal(λ); zeros(p - q, q)]
            # d = p - m
            # L = [Diagonal(λ) zeros(q, d)]
            # [I(p) L'; L I(q)]
            [I(p) U; U' I(q)]
        end
    end
end
covariance(p::Int, q::Int, λ::Vector{T}) where {T<:Real} = covariance(p, q, Float64.(λ))
function canonical_correlations(x::T, n::Int) where {T<:Real}
    x′ = abs(Float64(x))
    λ = Vector{Float64}(undef, n)
    @inbounds for i = 1:n
        if i & 1 == 1
            λ[i] = x′
        else
            λ[i] = -x′
        end
    end
    # u = reinterpret(UInt64, x′)
    # @inbounds for i in 1:n
    #     # Wrong
    #     # λ[i] = reinterpret(Float64, ((UInt64(i) & UInt64(1)) << 63) | u)
    #     Right
    #     λ[i] = reinterpret(Float64, (UInt64(trailing_zeros(i) != 0) << 63) | u)
    # end
    λ
end

function positive_diag!(Q::Matrix{T}, Λ::Vector{T}) where {T<:Base.IEEEFloat}
    for i in eachindex(Λ)
        if Q[i, i] < zero(T)
            for j in axes(Q, 2)
                Q[i, j] = -Q[i, j]
            end
            Λ[i] = -Λ[i]
        end
    end
end

function adjust_diagonals(F::SVD{T,T,Matrix{T},Vector{T}}) where {T<:Base.IEEEFloat}
    Q_x = copy(F.U')
    Λ = copy(F.S)
    Q_y = copy(F.Vt)
    positive_diag!(Q_x, Λ)
    positive_diag!(Q_y, Λ)
    Q_x, Λ, Q_y
end

function cca(Σ_x::Matrix{T}, Σ_y::Matrix{T}, Σ_xy::Matrix{T}) where {T<:Base.IEEEFloat}
    v¹²_x = sqrt.(diag(Σ_x))
    V⁻¹²_x = Diagonal(inv.(v¹²_x))
    P_x = V⁻¹²_x * Σ_x * V⁻¹²_x
    v¹²_y = sqrt.(diag(Σ_y))
    V⁻¹²_y = Diagonal(inv.(v¹²_y))
    P_y = V⁻¹²_y * Σ_y * V⁻¹²_y
    P_xy = V⁻¹²_x * Σ_xy * V⁻¹²_y
    F_x = eigen(P_x)
    P⁻¹²_x = F_x.vectors * Diagonal(inv.(sqrt.(F_x.values))) * F_x.vectors'
    F_y = eigen(P_y)
    P⁻¹²_y = F_y.vectors * Diagonal(inv.(sqrt.(F_y.values))) * F_y.vectors'
    K = P⁻¹²_x * P_xy * P⁻¹²_y
    F = svd(K)
end

struct CCA{T<:Base.IEEEFloat}
    Σ_x::Matrix{T}
    Σ_y::Matrix{T}
    Σ_xy::Matrix{T}
    W_x::Matrix{T}
    W_y::Matrix{T}
    W⁻¹_x::Matrix{T}
    W⁻¹_y::Matrix{T}
    F_x::Eigen{T,T,Matrix{T},Vector{T}}
    F_y::Eigen{T,T,Matrix{T},Vector{T}}
    F::SVD{T,T,Matrix{T},Vector{T}}
    K::Matrix{T}
    function CCA{T}(
        Σ_x::Matrix{T},
        Σ_y::Matrix{T},
        Σ_xy::Matrix{T},
    ) where {T<:Base.IEEEFloat}
        v¹²_x = sqrt.(diag(Σ_x))
        V⁻¹²_x = Diagonal(inv.(v¹²_x))
        P_x = V⁻¹²_x * Σ_x * V⁻¹²_x
        v¹²_y = sqrt.(diag(Σ_y))
        V⁻¹²_y = Diagonal(inv.(v¹²_y))
        P_y = V⁻¹²_y * Σ_y * V⁻¹²_y
        P_xy = V⁻¹²_x * Σ_xy * V⁻¹²_y
        F_x = eigen(P_x)
        P⁻¹²_x = F_x.vectors * Diagonal(inv.(sqrt.(F_x.values))) * F_x.vectors'
        F_y = eigen(P_y)
        P⁻¹²_y = F_y.vectors * Diagonal(inv.(sqrt.(F_y.values))) * F_y.vectors'
        K = P⁻¹²_x * P_xy * P⁻¹²_y
        F = svd(K)
        Q_x = F.U'
        Q_y = F.Vt
        W_x = Q_x * P⁻¹²_x * V⁻¹²_x
        W_y = Q_y * P⁻¹²_y * V⁻¹²_y
        W⁻¹_x =
            Diagonal(v¹²_x) *
            F_x.vectors *
            Diagonal(sqrt.(F_x.values)) *
            F_x.vectors' *
            Q_x'
        W⁻¹_y =
            Diagonal(v¹²_y) *
            F_y.vectors *
            Diagonal(sqrt.(F_y.values)) *
            F_y.vectors' *
            Q_y'
        new{T}(Σ_x, Σ_y, Σ_xy, W_x, W_y, W⁻¹_x, W⁻¹_y, F_x, F_y, F, K)
    end
end
function CCA(
    Σ_x::AbstractMatrix{T},
    Σ_y::AbstractMatrix{T},
    Σ_xy::AbstractMatrix{T},
) where {T<:Base.IEEEFloat}
    CCA{T}(collect(Σ_x), collect(Σ_y), collect(Σ_xy))
end
function CCA(X::AbstractMatrix{T}, Y::AbstractMatrix{T}) where {T<:Base.IEEEFloat}
    m = size(X, 1)
    if m == size(Y, 1)
        m⁻¹ = inv(m)
        μ_x = mean(X, dims = 1)
        μ_y = mean(Y, dims = 1)
        𝐗 = X .- μ_x
        𝐘 = Y .- μ_y
        Σ_x = m⁻¹ * 𝐗' * 𝐗
        Σ_y = m⁻¹ * 𝐘' * 𝐘
        Σ_xy = m⁻¹ * 𝐗' * 𝐘
        CCA{T}(Σ_x, Σ_y, Σ_xy)
    else
        error("mismatched dimensions")
    end
end
function whiten_x(tf::CCA{T}, X::AbstractMatrix{T}) where {T}
    X * tf.W_x'
end
function whiten_y(tf::CCA{T}, Y::AbstractMatrix{T}) where {T}
    Y * tf.W_y'
end

function unwhiten_x(tf::CCA{T}, X̃::AbstractMatrix{T}) where {T}
    X̃ * tf.W⁻¹_x'
end
function unwhiten_y(tf::CCA{T}, Ỹ::AbstractMatrix{T}) where {T}
    Ỹ * tf.W⁻¹_y'
end

function correctly_identified(t::CCA{T}, proposed::CCA{T}) where {T}
    lhs = t.K
    rhs = proposed.K
    inds = diagind(lhs)
    n = length(inds)
    s = 0
    for i in inds
        # s += Int(sign(lhs[i]) == sign(rhs[i]))
        s += Int(signbit(lhs[i]) ⊻ signbit(rhs[i]))
    end
    (n - s) / n
end
function correctly_identified_adjusted(t::CCA{T}, proposed::CCA{T}) where {T}
    _, lhs, _ = adjust_diagonals(t.F)
    _, rhs, _ = adjust_diagonals(proposed.F)
    s = 0
    for i in eachindex(lhs, rhs)
        s += Int(signbit(lhs[i]) ⊻ signbit(rhs[i]))
    end
    n = length(lhs)
    (n - s) / n
end


function canoncor_signs(rng::AbstractRNG, p::Int, q::Int, λ::Float64, n::Int)
    Σ = covariance(p, q, canonical_correlations(λ, q))
    Σ_x = Σ[1:p, 1:p]
    Σ_y = Σ[p+1:end, p+1:end]
    Σ_xy = Σ[1:p, p+1:end]

    tform = PCA(zeros(p + q), Σ)
    XY = unwhiten(tform, randn(rng, n, p + q))
    X = XY[:, 1:p]
    Y = XY[:, p+1:end]

    tf = CCA(Σ_x, Σ_y, Σ_xy)
    tf2 = CCA(X, Y)

    # tf, tf2, correctly_identified(tf, tf2)
    correctly_identified_adjusted(tf, tf2)
end

function canoncor_signs_expectation(
    rng::AbstractRNG,
    p::Int,
    q::Int,
    λ::Float64,
    n::Int,
    n_repeat::Int,
)
    s = 0.0
    for _ = 1:n_repeat
        s += canoncor_signs(rng, p, q, λ, n)
    end
    s / n_repeat
end

function canoncor_signs_matrix(
    rng::Xoshiro,
    p::Int,
    q::Int,
    λs::T,
    ns::U,
    n_repeat::Int,
) where {T<:AbstractArray{Float64},U<:AbstractArray{Int}}
    m = length(ns)
    n = length(λs)
    out = Matrix{Float64}(undef, m, n)
    x = Atomic{Int}(0)
    @threads :greedy for CI in CartesianIndices((1:m, 1:n))
        n_jump = atomic_add!(x, 1)
        rng′ = Random.jump_128(rng, n_jump)
        i = CI[1]
        j = CI[2]
        out[i, j] = canoncor_signs_expectation(rng′, p, q, λs[j], ns[i], n_repeat)
    end
    out
end



####
p = 10
q = 5

Σ = covariance(p, q, canonical_correlations(0.5, q))
Σ_x = Σ[1:p, 1:p]
Σ_y = Σ[p+1:end, p+1:end]
Σ_xy = Σ[1:p, p+1:end]

F = cca(Σ_x, Σ_y, Σ_xy)
F2 = cca(Σ_y, Σ_x, collect(Σ_xy'))


tform = PCA(zeros(p + q), Σ)

XY = unwhiten(tform, randn(4, p + q))

X = XY[:, 1:p]
Y = XY[:, p+1:end]



tf = CCA(Σ_x, Σ_y, Σ_xy)
tf2 = CCA(X, Y)

X̃ = whiten_x(tf, X)
Ỹ = whiten_y(tf, Y)
P_x̃ỹ_1 = cov(X̃, Ỹ, dims = 1, corrected = false)
P_x̃ỹ_2 = cor(X̃, Ỹ, dims = 1)
#=

y = a + bᵀx

a = 0    ⟹    y = bᵀx

yᵀ = xᵀb    ⟹    b⃰ = xᵀ \ yᵀ

=#
P_x̃ỹ_2 = X̃ \ Ỹ
P_x̃ỹ_3 = tf.W_x * Σ_xy * tf.W_y'
P_x̃ỹ_4 = tf2.W_x * Σ_xy * tf2.W_y'

X_rt = unwhiten_x(tf, X̃)
Y_rt = unwhiten_y(tf, Ỹ)

ϕ_x = tf.W_x * tf.Σ_x

tf.W_x * tf.Σ_xy * tf.W_y'


correctly_identified(tf, tf2)

Q_x, Λ, Q_y = adjust_diagonals(tf.F)
Q_x_2, Λ_2, Q_y_2 = adjust_diagonals(tf2.F)

@test Q_x' * Diagonal(Λ) * Q_y ≈ tf.K
@test Q_x_2' * Diagonal(Λ_2) * Q_y_2 ≈ tf2.K

let λ = 0.2:0.1:0.9, n = 100:100:500 # 0x0123456789abcdef
    pct = [canoncor_signs(Xoshiro(rand(UInt64)), 60, 10, λ, n) for n in n, λ in λ]
    plot(pct, labels = permutedims(map(λ -> "λ = $λ", λ)), xticks = (eachindex(n), n))
end

let λs = 0.2:0.1:0.9, ns = 100:100:500, n_repeat = 500
    pct = canoncor_signs_matrix(Xoshiro(0x0123456789abcdef), p, q, λs, ns, n_repeat)
    plot(pct, labels = permutedims(map(λ -> "λ = $λ", λ)), xticks = (eachindex(ns), ns))
end

# squared correlation loadings
p1 = heatmap(abs.(cor(X, X̃, dims=1)));
p2 = heatmap(abs2.(cor(Y, Ỹ, dims=1)));
plot(p1, p2, size=(1200,400))


# with mean included
struct CCA4{T<:Base.IEEEFloat}
    μ_x::Vector{T}
    μ_y::Vector{T}
    Σ_x::Matrix{T}
    Σ_y::Matrix{T}
    Σ_xy::Matrix{T}
    W_x::Matrix{T}
    W_y::Matrix{T}
    W⁻¹_x::Matrix{T}
    W⁻¹_y::Matrix{T}
    F_x::Eigen{T,T,Matrix{T},Vector{T}}
    F_y::Eigen{T,T,Matrix{T},Vector{T}}
    F::SVD{T,T,Matrix{T},Vector{T}}
    K::Matrix{T}
    function CCA4{T}(
        μ_x::Vector{T},
        μ_y::Vector{T},
        Σ_x::Matrix{T},
        Σ_y::Matrix{T},
        Σ_xy::Matrix{T},
    ) where {T<:Base.IEEEFloat}
        v¹²_x = sqrt.(diag(Σ_x))
        V⁻¹²_x = Diagonal(inv.(v¹²_x))
        P_x = V⁻¹²_x * Σ_x * V⁻¹²_x
        v¹²_y = sqrt.(diag(Σ_y))
        V⁻¹²_y = Diagonal(inv.(v¹²_y))
        P_y = V⁻¹²_y * Σ_y * V⁻¹²_y
        P_xy = V⁻¹²_x * Σ_xy * V⁻¹²_y
        F_x = eigen(P_x, sortby=-)
        P⁻¹²_x = F_x.vectors * Diagonal(inv.(sqrt.(F_x.values))) * F_x.vectors'
        F_y = eigen(P_y, sortby=-)
        P⁻¹²_y = F_y.vectors * Diagonal(inv.(sqrt.(F_y.values))) * F_y.vectors'
        K = P⁻¹²_x * P_xy * P⁻¹²_y
        F = svd(K)
        Q_x = F.U'
        Q_y = F.Vt
        W_x = Q_x * P⁻¹²_x * V⁻¹²_x
        W_y = Q_y * P⁻¹²_y * V⁻¹²_y
        W⁻¹_x =
            Diagonal(v¹²_x) *
            F_x.vectors *
            Diagonal(sqrt.(F_x.values)) *
            F_x.vectors' *
            Q_x'
        W⁻¹_y =
            Diagonal(v¹²_y) *
            F_y.vectors *
            Diagonal(sqrt.(F_y.values)) *
            F_y.vectors' *
            Q_y'
        new{T}(μ_x, μ_y, Σ_x, Σ_y, Σ_xy, W_x, W_y, W⁻¹_x, W⁻¹_y, F_x, F_y, F, K)
    end
end
function CCA4(
    μ_x::AbstractVector{T},
    μ_y::AbstractVector{T},
    Σ_x::AbstractMatrix{T},
    Σ_y::AbstractMatrix{T},
    Σ_xy::AbstractMatrix{T},
) where {T<:Base.IEEEFloat}
    CCA4{T}(collect(μ_x), collect(μ_y), collect(Σ_x), collect(Σ_y), collect(Σ_xy))
end
function CCA4(X::AbstractMatrix{T}, Y::AbstractMatrix{T}) where {T<:Base.IEEEFloat}
    m = size(X, 1)
    if m == size(Y, 1)
        m⁻¹ = inv(m)
        μ_x = mean(X, dims = 1)
        μ_y = mean(Y, dims = 1)
        𝐗 = X .- μ_x
        𝐘 = Y .- μ_y
        Σ_x,_ = shrinkage(𝐗)
        Σ_y,_ = shrinkage(𝐘)
        Σ_xy = m⁻¹ * 𝐗' * 𝐘
        CCA4{T}(vec(μ_x), vec(μ_y), Σ_x, Σ_y, Σ_xy)
    else
        error("mismatched dimensions")
    end
end
function whiten_x(tf::CCA4{T}, X::AbstractMatrix{T}) where {T}
    (X .- tf.μ_x') * tf.W_x'
end
function whiten_y(tf::CCA4{T}, Y::AbstractMatrix{T}) where {T}
    (Y .- tf.μ_y') * tf.W_y'
end

function unwhiten_x(tf::CCA4{T}, X̃::AbstractMatrix{T}) where {T}
    (X̃ * tf.W⁻¹_x') .+ tf.μ_x'
end
function unwhiten_y(tf::CCA4{T}, Ỹ::AbstractMatrix{T}) where {T}
    (Ỹ * tf.W⁻¹_y') .+ tf.μ_y'
end
