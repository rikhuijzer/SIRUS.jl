include("preliminaries.jl")

@testset "doctests" begin
    # warn suppresses warnings when keys already exist.
    DocMeta.setdocmeta!(SIRUS, :DocTestSetup, :(using SIRUS); recursive=true, warn=false)
    doctest(SIRUS)
end

@testset "empiricalquantiles" begin
    include("empiricalquantiles.jl")
end

@testset "forest" begin
    include("forest.jl")
end

@testset "classification" begin
    include("classification.jl")
end

@testset "regression" begin
    include("regression.jl")
end

@testset "rules" begin
    include("rules.jl")
end

@testset "ruleshow" begin
    include("ruleshow.jl")
end

@testset "dependent" begin
    include("dependent.jl")
end

@testset "weights" begin
    include("weights.jl")
end

@testset "mlj" begin
    include("mlj.jl")
end

nothing
