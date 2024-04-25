# Internals
for T in (Float16, Float32, Float64)
    @eval precompile(_estimate, (Matrix{$T},))
    @eval precompile(_estimate, (Adjoint{$T,Matrix{$T}},))
    @eval precompile(_estimate, (Transpose{$T,Matrix{$T}},))
    @eval precompile(findlastcomponent, ($T, Vector{$T}))
    @eval precompile(findlastrank, ($T, Vector{$T}))
    @eval precompile(ispossemidef, (Matrix{$T},))
    @eval precompile(ispossemidef, (Adjoint{$T,Matrix{$T}},))
    @eval precompile(ispossemidef, (Transpose{$T,Matrix{$T}},))
    @eval precompile(checkargs, (Vector{$T}, Matrix{$T}))
    @eval precompile(checkargs, (Vector{$T}, Matrix{$T}, Int, $T))
    @eval precompile(checkargs, (Vector{$T}, Matrix{$T}, Nothing, $T))
    @eval precompile(checkargs, (Vector{$T}, Matrix{$T}, Nothing, Nothing))
end

# General operations
for T in (Float16, Float32, Float64)
    for U in (PCA, PCAcor, ZCA, ZCAcor, Chol, GeneralizedPCA, GeneralizedPCAcor)
        for f in (whiten, unwhiten, mahalanobis)
            @eval precompile($f, ($U{$T}, Vector{$T}))
            @eval precompile($f, ($U{$T}, Matrix{$T}))
        end
        @eval precompile($U, (Vector{$T}, Matrix{$T}))
        @eval precompile($U, (Vector{$T}, Adjoint{$T,Matrix{$T}}))
        @eval precompile($U, (Vector{$T}, Transpose{$T,Matrix{$T}}))
        @eval precompile($U, (Matrix{$T},))
        @eval precompile($U, (Adjoint{$T,Matrix{$T}},))
        @eval precompile($U, (Transpose{$T,Matrix{$T}},))
    end
end
