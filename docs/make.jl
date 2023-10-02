is_ci = haskey(ENV, "CI")

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

pages = Pair{String, Any}[
    "SIRUS" => "index.md",
    "Implementation Overview" => "implementation-overview.md",
    "API" => "api.md"
]

do_build_notebooks = is_ci

if do_build_notebooks
    build()
    getting_started::Pair = "Getting Started" => [
        "Basic Example" => "basic-example.md",
        "Advanced Example" => "binary-classification.md"
    ]
    insert!(pages, 2, getting_started)
end

prettyurls = is_ci
format = HTML(; mathengine=MathJax3(), prettyurls)
modules = [SIRUS]
warnonly = !do_build_notebooks
checkdocs = :none
makedocs(; sitename, pages, format, modules, warnonly, checkdocs)

deploydocs(;
    branch="docs-output",
    devbranch="main",
    repo="github.com/rikhuijzer/SIRUS.jl.git",
    push_preview=false
)
