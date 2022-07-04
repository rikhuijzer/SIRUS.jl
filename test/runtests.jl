import Base

using CategoricalArrays: CategoricalValue, categorical, unwrap
using DecisionTree: DecisionTree
using MLJBase
using LightGBM.MLJInterface: LGBMClassifier
using StableRNGs: StableRNG
using StableTrees
using Tables: Tables
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
