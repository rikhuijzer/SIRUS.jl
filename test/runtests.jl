include("preliminaries.jl")

@testset "docs" begin
    include("docs.jl")
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

@testset "importance" begin
    include("importance.jl")
end

if get(ENV, "CAN_RUN_R_SIRUS", "false")::String == "true"
    @testset "rcall" begin
        include("rcall.jl")
    end
end

@testset "mlj" begin
    include("mlj.jl")
end

nothing
