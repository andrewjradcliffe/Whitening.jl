module Whitening

public AbstractWhiteningTransform, PCA, PCAcor, ZCA, ZCAcor, Chol
public whiten, unwhiten, mahalanobis

using LinearAlgebra

include("abstract.jl")
include("ZCA.jl")
include("ZCAcor.jl")
include("PCA.jl")
include("PCAcor.jl")
include("Chol.jl")

end # module Whitening
