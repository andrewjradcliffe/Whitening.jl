module Whitening

# Until Julia 1.11 is stable, we hack in `public` symbol support
@static if VERSION >= v"1.11"
    @eval $(Meta.parse(
        "public AbstractWhiteningTransform, PCA, PCAcor, ZCA, ZCAcor, Chol, GeneralizedPCA, GeneralizedPCAcor, whiten, unwhiten, mahalanobis",
    ))
end

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
