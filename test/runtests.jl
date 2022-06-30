import Base

using CategoricalArrays: CategoricalValue, categorical
using MLJBase
using StableRNGs: StableRNG
using StableTrees
using Test

ST = StableTrees
Float = ST.Float

@testset "forest" begin
    include("forest.jl")
end

@testset "rules" begin
    include("rules.jl")
end

@testset "interface" begin
    include("mlj.jl")
end
