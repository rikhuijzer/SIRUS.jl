using Documenter:
    DocMeta,
    HTML,
    MathJax3,
    asset,
    deploydocs,
    makedocs
using Pkg: Pkg
using PlutoStaticHTML
using StableTrees

DocMeta.setdocmeta!(
    StableTrees,
    :DocTestSetup,
    :(using StableTrees);
    recursive=true
)

sitename = "StableTrees.jl"
tutorials_dir = joinpath(dirname(@__DIR__), "docs", "src")

function build()
    println("Building notebooks in $tutorials_dir")
    use_distributed = false
    output_format = documenter_output
    bopts = BuildOptions(tutorials_dir; use_distributed, output_format)
    build_notebooks(bopts)
    Pkg.activate(@__DIR__)
    return nothing
end

# Build the notebooks; defaults to "true".
if get(ENV, "BUILD_DOCS_NOTEBOOKS", "true") == "true"
    build()
    cd(tutorials_dir) do
        mv("stabletrees.md", "index.md"; force=true)
    end
end

pages = [
    "StableTrees" => "index.md"
]

prettyurls = get(ENV, "CI", nothing) == "true"
format = HTML(; mathengine=MathJax3(), prettyurls)
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
