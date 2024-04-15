module Whitening

public whiten, unwhiten, mahalanobis

using LinearAlgebra

include("abstract.jl")
include("ZCA.jl")
include("ZCAcor.jl")
include("PCA.jl")
include("PCAcor.jl")
include("Chol.jl")

end # module Whitening
