using Documenter, Whitening

makedocs(sitename="Whitening.jl")

deploydocs(
    repo = "github.com/andrewjradcliffe/Whitening.jl.git",
    devbranch="main",
    versions = nothing,
)
