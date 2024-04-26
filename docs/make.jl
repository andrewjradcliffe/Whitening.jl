using Documenter, Whitening

makedocs(
    sitename="Whitening.jl",
    pages = [
        "Home" => "index.md",
    ]
)

deploydocs(
    repo = "github.com/andrewjradcliffe/Whitening.jl.git",
    branch="gh-pages",
    devbranch="main",
    versions = ["stable" => "v^", "v#.#", "dev" => "main"],
    devurl = "dev",
)
