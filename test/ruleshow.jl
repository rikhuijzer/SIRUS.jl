@testset "binary show" begin
    r = S.Rule(S.Clause(" X[i, 1] < 5 "), [0.1, 0.9], [0.2, 0.8])
    classes = [0, 1]
    weights = Float16[1.0]
    algo = SIRUS.Classification()
    model = S.StableRules([r], algo, classes, weights)
    pretty = repr(model)
    @test contains(pretty, "0.9")
    @test contains(pretty, "0.8")
    @test contains(pretty, "showing only")
    @test !contains(pretty, "unexpected")
end

@testset "regression show" begin
    r1 = S.Rule(S.Clause(" X[i, 1] < 5 "), [0.1], [0.8])
    r2 = S.Rule(S.Clause(" X[i, 2] < 3 "), [0.4], [0.6])
    weights = Float16[0.6, 0.7]
    algo = SIRUS.Regression()
    classes = []
    model = S.StableRules([r1, r2], algo, classes, weights)
    pretty = split(repr(model), '\n')
    @test pretty[1] == "StableRules model with 2 rules:"
    then = round(weights[1] * 0.1; digits=3)
    otherwise = round(weights[1] * 0.8; digits=3)
    @test pretty[2] == " if X[i, 1] < 5.0 then $then else $otherwise +"

    then = round(weights[2] * 0.4; digits=3)
    otherwise = round(weights[2] * 0.6; digits=3)
    @test pretty[3] == " if X[i, 2] < 3.0 then $then else $otherwise"
end
