using Documenter:
    DocMeta,
    HTML,
    asset,
    deploydocs,
    makedocs
using PlutoStaticHTML
using StableTrees

DocMeta.setdocmeta!(
    StableTrees,
    :DocTestSetup,
    :(using StableTrees);
    recursive=true
)

sitename = "StableTrees.jl"
tutorials_dir = joinpath(pkgdir(StableTrees), "docs", "src")

function build()
    println("Building notebooks")
    use_distributed = false
    output_format = documenter_output
    bopts = BuildOptions(tutorials_dir; use_distributed, output_format)
    build_notebooks(bopts)
    return nothing
end

# Build the notebooks; defaults to "true".
if get(ENV, "BUILD_DOCS_NOTEBOOKS", "true") == "true"
    build()
end

pages = [
    "StableTrees" => "stabletrees.md"
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
