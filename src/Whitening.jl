module Whitening

public AbstractWhiteningTransform, PCA, PCAcor, ZCA, ZCAcor, Chol
public GeneralizedPCA, GeneralizedPCAcor
public whiten, unwhiten, mahalanobis

using LinearAlgebra

include("abstract.jl")
include("common.jl")
include("ZCA.jl")
include("ZCAcor.jl")
include("PCA.jl")
include("PCAcor.jl")
include("Chol.jl")
include("GeneralizedPCA.jl")
include("GeneralizedPCAcor.jl")
include("precompile.jl")

end # module Whitening
