include("preliminaries.jl")

@testset "forest" begin
    include("forest.jl")
end

@testset "rules" begin
    include("rules.jl")
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
