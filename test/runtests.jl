include("preliminaries.jl")

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

@testset "tmp" begin
    include("tmp.jl")
end

@testset "weights" begin
    include("weights.jl")
end

@testset "mlj" begin
    include("mlj.jl")
end

nothing
