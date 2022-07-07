import Base

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

using CategoricalArrays: CategoricalValue, categorical, unwrap
using MLDatasets: Titanic
using DataFrames: DataFrames, select, dropmissing!
using DecisionTree: DecisionTree
using MLJBase
using LightGBM.MLJInterface: LGBMClassifier
using StableRNGs: StableRNG
using StableTrees
using Tables: Tables
using Test

ST = StableTrees
Float = ST.Float
_rng() = StableRNG(1)

@testset "forest" begin
    include("forest.jl")
end

@testset "rules" begin
    include("rules.jl")
end

@testset "dependent" begin
    include("dependent.jl")
end

@testset "interface" begin
    include("mlj.jl")
end
