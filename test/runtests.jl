import Base

using MLJBase
using StableRNGs: StableRNG
using StableTrees
using Test

ST = StableTrees
Float = ST.Float

@testset "forest" begin
    include("forest.jl")
end

@testset "interface" begin
    include("mlj.jl")
end
