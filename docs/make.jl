using Documenter:
    DocMeta,
    HTML,
    MathJax3,
    asset,
    deploydocs,
    makedocs
using Pkg: Pkg
using PlutoStaticHTML
using SIRUS

DocMeta.setdocmeta!(
    SIRUS,
    :DocTestSetup,
    :(using SIRUS);
    recursive=true
)

sitename = "SIRUS.jl"
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
        mv("sirus.md", "index.md"; force=true)
    end
end

pages = [
    "SIRUS" => "index.md",
    "API" => "api.md"
]

prettyurls = get(ENV, "CI", nothing) == "true"
format = HTML(; mathengine=MathJax3(), prettyurls)
modules = [SIRUS]
strict = true
checkdocs = :none
makedocs(; sitename, pages, format, modules, strict, checkdocs)

deploydocs(;
    branch="docs",
    devbranch="main",
    repo="github.com/rikhuijzer/SIRUS.jl.git",
    push_preview=false
)
