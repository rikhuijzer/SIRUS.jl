using Documenter:
    DocMeta,
    HTML,
    asset,
    deploydocs,
    makedocs
using StableTrees

DocMeta.setdocmeta!(
    StableTrees,
    :DocTestSetup,
    :(using StableTrees);
    recursive=true
)

sitename = "StableTrees.jl"
pages = [
    "StableTrees" => "index.md"
]

prettyurls = get(ENV, "CI", nothing) == "true"
format = HTML(; prettyurls)
modules = [StableTrees]
strict = true
checkdocs = :none
makedocs(; sitename, pages, format, modules, strict, checkdocs)

deploydocs(;
    branch="docs",
    devbranch="main",
    repo="github.com/rikhuijzer/StableTrees.jl.git",
    push_preview=false
)
