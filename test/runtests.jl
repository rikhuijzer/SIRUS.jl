include("preliminaries.jl")

@testset "empiricalquantiles" begin
    include("empiricalquantiles.jl")
end

@testset "forest" begin
    include("forest.jl")
end

# @testset "regression" begin
#     include("regression.jl")
# end

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
